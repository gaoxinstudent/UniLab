"""Experimental async RSL-RL PPO runner."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from copy import deepcopy
from typing import Any

import torch
from omegaconf import OmegaConf
from rsl_rl.algorithms import PPO

from unilab.algos.torch.rsl_rl_async_ppo.buffer import RslRlPpoRolloutBuffer
from unilab.algos.torch.rsl_rl_async_ppo.staging import stage_ppo_rollout
from unilab.algos.torch.rsl_rl_async_ppo.storage_adapter import fill_rollout_storage
from unilab.algos.torch.rsl_rl_async_ppo.worker import rsl_rl_ppo_collector_fn
from unilab.ipc import AsyncRunner, SharedWeightSync
from unilab.logging.onpolicy import OnPolicyLogger
from unilab.training.seed import derive_worker_seed
from unilab.utils.device import get_default_device


def validate_async_ppo_v1_config(cfg: Any, train_cfg: dict[str, Any]) -> None:
    """Fail fast for async PPO v1 unsupported features."""
    if not bool(OmegaConf.select(cfg, "training.async", default=False)):
        return
    checks = {
        "training.async_queue_size": 1,
        "training.async_policy_lag_max": 1,
        "training.async_stale_rollout_policy": "block",
        "training.async_strict_onpolicy": True,
    }
    for path, expected in checks.items():
        value = OmegaConf.select(cfg, path, default=expected)
        if value != expected:
            raise ValueError(
                f"Async RSL-RL PPO v1 requires {path}={expected!r}, got {value!r}. "
                "Use the default synchronous PPO path for unsupported async settings."
            )

    algorithm_cfg = train_cfg.get("algorithm", {})
    if isinstance(algorithm_cfg, dict):
        unsupported = {
            "rnd_cfg": algorithm_cfg.get("rnd_cfg"),
            "symmetry_cfg": algorithm_cfg.get("symmetry_cfg"),
        }
        for key, value in unsupported.items():
            if value:
                raise ValueError(
                    f"Async RSL-RL PPO v1 does not support algorithm.{key}. "
                    "Use synchronous PPO for this configuration."
                )

    multi_gpu = train_cfg.get("multi_gpu")
    if multi_gpu:
        raise ValueError("Async RSL-RL PPO v1 does not support multi-GPU PPO.")


def ppo_construct_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a deepcopy with RSL-RL runner-provided defaults applied."""
    cfg = deepcopy(train_cfg)
    cfg.setdefault("multi_gpu", None)
    return cfg


class AsyncRslRlPpoRunner(AsyncRunner):
    """Single-collector, single in-flight async RSL-RL PPO runtime."""

    def __init__(
        self,
        *,
        cfg: Any,
        train_cfg: dict[str, Any],
        env_cfg_override: dict[str, Any],
        wrapper_cls: type,
        log_dir: str,
        device: str,
        collector_device: str | None,
        logger_type: str,
        resume_path: str | None,
        nan_guard_cfg: Any = None,
        wandb_settings: dict[str, Any] | None = None,
    ) -> None:
        validate_async_ppo_v1_config(cfg, train_cfg)
        super().__init__(
            env_name=str(cfg.training.task_name),
            env_cfg_overrides=env_cfg_override,
            rl_cfg=train_cfg,
            device=device,
            collector_device=collector_device or device,
            sim_backend=str(cfg.training.sim_backend),
            num_envs=int(cfg.algo.num_envs),
        )
        self.cfg = cfg
        self.train_cfg = deepcopy(train_cfg)
        self.wrapper_cls = wrapper_cls
        self.log_dir = log_dir
        self.logger_type = logger_type
        self.resume_path = resume_path
        self.nan_guard_cfg = nan_guard_cfg
        self.wandb_settings = wandb_settings or {}
        self.steps_per_env = int(cfg.algo.num_steps_per_env)
        self.current_learning_iteration = 0
        self.last_run_summary: dict[str, Any] | None = None
        self._probe_env = None
        self._probe_wrapped_env = None

    def _get_default_device(self) -> str:
        return get_default_device()

    def _build_learner(self) -> PPO:
        from unilab.training import create_env

        self._probe_env = create_env(
            self.cfg,
            num_envs=self.num_envs,
            env_cfg_override=self.env_cfg_overrides,
        )
        self._probe_wrapped_env = self.wrapper_cls(self._probe_env, device=self.device)
        obs = self._probe_wrapped_env.get_observations().to(self.device)
        alg = PPO.construct_algorithm(obs, self._probe_wrapped_env, ppo_construct_cfg(self.train_cfg), self.device)
        if alg.actor.is_recurrent or alg.critic.is_recurrent:
            raise ValueError("Async RSL-RL PPO v1 does not support recurrent actor or critic.")
        if alg.rnd:
            raise ValueError("Async RSL-RL PPO v1 does not support RND.")
        if alg.symmetry:
            raise ValueError("Async RSL-RL PPO v1 does not support symmetry augmentation.")
        if alg.is_multi_gpu:
            raise ValueError("Async RSL-RL PPO v1 does not support multi-GPU PPO.")
        return alg

    def _collector_fn(self, stop_event: Any, **kwargs: Any) -> None:
        rsl_rl_ppo_collector_fn(stop_event=stop_event, **kwargs)

    def _infer_rollout_schema(
        self, alg: PPO
    ) -> tuple[dict[str, tuple[int, ...]], tuple[tuple[int, ...], ...]]:
        assert self._probe_wrapped_env is not None
        obs = self._probe_wrapped_env.get_observations().to(self.device)
        with torch.no_grad():
            _ = alg.act(obs)
        obs_shapes = {key: tuple(value.shape[1:]) for key, value in obs.items()}
        distribution_param_shapes = tuple(
            tuple(param.shape[1:]) for param in alg.transition.distribution_params
        )
        alg.transition.clear()
        return obs_shapes, distribution_param_shapes

    def _make_logger(self, max_iterations: int) -> OnPolicyLogger:
        settings = self.wandb_settings
        logger = OnPolicyLogger(
            algo_name="Async RSL-RL PPO",
            max_iterations=max_iterations,
            num_envs=self.num_envs,
            num_steps=self.steps_per_env,
            env_name=self.env_name,
            log_dir=self.log_dir,
            log_backend=self.logger_type,
            wandb_project=settings.get(
                "project", getattr(self.cfg.training, "wandb_project", "unilab")
            ),
            wandb_entity=settings.get("entity", getattr(self.cfg.training, "wandb_entity", None)),
            wandb_name=settings.get("name", getattr(self.cfg.training, "wandb_name", "")),
            wandb_group=settings.get("group", getattr(self.cfg.training, "wandb_group", None)),
            wandb_job_type=settings.get("job_type", getattr(self.cfg.training, "wandb_job_type", None)),
            wandb_tags=settings.get("tags", getattr(self.cfg.training, "wandb_tags", [])),
            wandb_notes=settings.get("notes", getattr(self.cfg.training, "wandb_notes", None)),
        )
        return logger

    def _save(
        self, alg: PPO, path: str, logger: OnPolicyLogger, infos: dict[str, Any] | None = None
    ) -> None:
        saved_dict = alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        logger.log_save(path)

    def _drain_metrics(
        self,
        queue: Any,
        reward_history: deque[float],
        length_history: deque[float],
        reward_components: dict[str, float],
        logger: OnPolicyLogger,
    ) -> None:
        while not queue.empty():
            msg = queue.get_nowait()
            if "mean_ep_reward" in msg:
                reward_history.append(float(msg["mean_ep_reward"]))
            if "mean_ep_length" in msg:
                length_history.append(float(msg["mean_ep_length"]))
                logger.update_ep_length(float(msg["mean_ep_length"]))
            if "reward_components" in msg:
                reward_components.clear()
                reward_components.update(
                    {str(key): float(value) for key, value in msg["reward_components"].items()}
                )

    def learn(
        self,
        max_iterations: int,
        save_interval: int = 50,
        log_dir: str | None = None,
    ) -> None:
        del log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        train_start_wall = time.time()
        alg = self._build_learner()
        if self.resume_path:
            loaded = torch.load(self.resume_path, map_location=self.device, weights_only=False)
            if alg.load(loaded, load_cfg=None, strict=True):
                self.current_learning_iteration = int(loaded["iter"])

        obs_shapes, distribution_param_shapes = self._infer_rollout_schema(alg)
        rollout_buffer = RslRlPpoRolloutBuffer(
            num_envs=self.num_envs,
            num_steps=self.steps_per_env,
            obs_shapes=obs_shapes,
            action_dim=int(self._probe_wrapped_env.num_actions),  # type: ignore[union-attr]
            distribution_param_shapes=distribution_param_shapes,
            num_slots=1,
            create=True,
        )
        self._shared_resources.append(rollout_buffer)

        actor_weight_sync = SharedWeightSync.from_state_dict(alg.actor.state_dict(), create=True)
        critic_weight_sync = SharedWeightSync.from_state_dict(alg.critic.state_dict(), create=True)
        collector_actor_state_sync = SharedWeightSync.from_state_dict(
            alg.actor.state_dict(), create=True
        )
        collector_critic_state_sync = SharedWeightSync.from_state_dict(
            alg.critic.state_dict(), create=True
        )
        self._shared_resources.extend(
            [
                actor_weight_sync,
                critic_weight_sync,
                collector_actor_state_sync,
                collector_critic_state_sync,
            ]
        )

        import multiprocessing as mp

        metrics_queue: mp.Queue = mp.get_context("spawn").Queue(maxsize=100)
        self._start_collector(
            target_fn=rsl_rl_ppo_collector_fn,
            kwargs={
                "stop_event": self._stop_event,
                "cfg": self.cfg,
                "env_cfg_override": self.env_cfg_overrides,
                "wrapper_cls": self.wrapper_cls,
                "train_cfg": self.train_cfg,
                "num_envs": self.num_envs,
                "num_steps": self.steps_per_env,
                "obs_shapes": obs_shapes,
                "action_dim": int(self._probe_wrapped_env.num_actions),  # type: ignore[union-attr]
                "distribution_param_shapes": distribution_param_shapes,
                "shm_rollout_buffer_names": rollout_buffer.name,
                "sync_primitives": (rollout_buffer._write_ptr, rollout_buffer._read_ptr),
                "actor_weight_sync_name": actor_weight_sync.name,
                "actor_weight_param_shapes": {
                    name: p.shape for name, p in alg.actor.state_dict().items()
                },
                "critic_weight_sync_name": critic_weight_sync.name,
                "critic_weight_param_shapes": {
                    name: p.shape for name, p in alg.critic.state_dict().items()
                },
                "collector_actor_state_sync_name": collector_actor_state_sync.name,
                "collector_critic_state_sync_name": collector_critic_state_sync.name,
                "metrics_queue": metrics_queue,
                "collector_device": self.collector_device,
                "seed": derive_worker_seed(int(self.cfg.algo.seed), worker_index=0),
                "nan_guard_cfg": self.nan_guard_cfg,
                "_error_label": "async-rsl-ppo-collector",
            },
        )

        if self._probe_wrapped_env is not None:
            self._probe_wrapped_env.close()
            self._probe_wrapped_env = None
            self._probe_env = None

        logger = self._make_logger(max_iterations)
        logger.start(status="Training")
        reward_history: deque[float] = deque(maxlen=200)
        length_history: deque[float] = deque(maxlen=200)
        reward_components: dict[str, float] = {}
        last_ckpt_path: str | None = None

        try:
            start_it = self.current_learning_iteration
            total_it = start_it + max_iterations
            for it in range(start_it, total_it):
                iteration_start = time.perf_counter()
                self._drain_metrics(metrics_queue, reward_history, length_history, reward_components, logger)
                wait_start = time.time()
                deadline = time.monotonic() + 60.0
                data_ready = False
                while time.monotonic() < deadline:
                    timeout = min(0.5, max(0.0, deadline - time.monotonic()))
                    if rollout_buffer.wait_for_data(timeout=timeout):
                        data_ready = True
                        break
                    if not self._check_collector_alive():
                        raise RuntimeError(
                            "Async RSL-RL PPO collector died before producing data. "
                            "Check stderr for the collector traceback."
                        )
                if not data_ready:
                    raise TimeoutError("Timed out waiting for async RSL-RL PPO rollout data.")

                staging_start = time.perf_counter()
                rollout = stage_ppo_rollout(
                    rollout_buffer.read_numpy_views(),
                    obs_shapes=obs_shapes,
                    distribution_param_count=len(distribution_param_shapes),
                    device=self.device,
                )
                rollout_buffer.advance_read()
                staging_time = time.perf_counter() - staging_start

                collect_time = time.time() - wait_start
                learn_start = time.time()
                actor_sd = dict(alg.actor.state_dict())
                collector_actor_state_sync.read_weights_into(actor_sd)
                alg.actor.load_state_dict(actor_sd)
                critic_sd = dict(alg.critic.state_dict())
                collector_critic_state_sync.read_weights_into(critic_sd)
                alg.critic.load_state_dict(critic_sd)
                fill_rollout_storage(alg.storage, rollout)
                alg.compute_returns(rollout["last_obs"])
                loss_dict = alg.update()
                learn_time = time.time() - learn_start
                weight_sync_start = time.perf_counter()
                actor_weight_sync.write_weights(alg.actor.state_dict())
                critic_weight_sync.write_weights(alg.critic.state_dict())
                weight_sync_time = time.perf_counter() - weight_sync_start

                self.current_learning_iteration = it
                policy_lag = int(rollout["policy_version_at_collect_end"]) - int(
                    rollout["policy_version_at_collect_start"]
                )
                rollout_age_ms = (
                    time.monotonic() - float(rollout["rollout_created_time_ns"])
                ) * 1000.0
                metrics = {
                    **{str(key): float(value) for key, value in loss_dict.items()},
                    "async/staging_time": staging_time,
                    "async/learner_wait_time": collect_time,
                    "async/learn_time": learn_time,
                    "async/policy_lag_versions": float(policy_lag),
                    "async/rollout_age_ms": float(rollout_age_ms),
                    "async/weight_sync_time": weight_sync_time,
                    "learning_rate": float(alg.learning_rate),
                }
                self._drain_metrics(metrics_queue, reward_history, length_history, reward_components, logger)
                reward = statistics.mean(reward_history) if reward_history else None
                iteration_time = time.perf_counter() - iteration_start
                logger.log_step(
                    iteration=it,
                    metrics=metrics,
                    reward=reward,
                    reward_components=reward_components,
                    collect_time=collect_time,
                    train_time=learn_time,
                    iteration_time=iteration_time,
                    collect_label="Wait Rollout",
                )
                if save_interval > 0 and it % save_interval == 0:
                    last_ckpt_path = os.path.join(self.log_dir, f"model_{it}.pt")
                    self._save(alg, last_ckpt_path, logger)

            last_ckpt_path = os.path.join(
                self.log_dir, f"model_{self.current_learning_iteration}.pt"
            )
            self._save(alg, last_ckpt_path, logger)
            logger.finish()
            self.last_run_summary = {
                "status": "completed",
                "completed_iterations": int(self.current_learning_iteration),
                "total_env_steps": int(max_iterations * self.num_envs * self.steps_per_env),
                "final_mean_reward": (
                    float(statistics.mean(reward_history)) if reward_history else None
                ),
                "best_mean_reward": float(max(reward_history)) if reward_history else None,
                "mean_episode_length": float(statistics.mean(length_history)) if length_history else None,
                "last_checkpoint": last_ckpt_path,
                "training_wall_time_sec": time.time() - train_start_wall,
            }
        finally:
            logger.close()
            self.close()
