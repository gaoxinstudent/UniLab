"""Collector subprocess for experimental async RSL-RL PPO."""

from __future__ import annotations

import time
from collections import defaultdict
from queue import Empty, Full
from typing import Any

import numpy as np
import torch
from rsl_rl.utils import check_nan

from unilab.algos.torch.rsl_rl_async_ppo.buffer import RslRlPpoRolloutBuffer
from unilab.base.registry import ensure_registries
from unilab.training.seed import apply_training_seed


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _put_latest_metrics(metrics_queue: Any, msg: dict[str, Any]) -> None:
    """Best-effort metrics enqueue that drops stale messages under learner stalls."""
    try:
        metrics_queue.put_nowait(msg)
        return
    except Full:
        pass

    try:
        metrics_queue.get_nowait()
    except Empty:
        pass

    try:
        metrics_queue.put_nowait(msg)
    except Full:
        pass


def _record_reward_components(
    sink: defaultdict[str, list[float]],
    extras: dict[str, Any],
) -> None:
    log_info = extras.get("log")
    if not isinstance(log_info, dict):
        return
    for key, value in log_info.items():
        if key.startswith("Episode/rew_"):
            component_name = key.removeprefix("Episode/rew_")
        elif key.startswith("rew_"):
            component_name = key.removeprefix("rew_")
        elif key.startswith("reward/"):
            component_name = key.removeprefix("reward/")
        else:
            continue
        if isinstance(value, torch.Tensor):
            sink[component_name].append(float(value.detach().mean().cpu()))
        elif isinstance(value, (int, float)):
            sink[component_name].append(float(value))


def rsl_rl_ppo_collector_fn(
    *,
    stop_event: Any,
    cfg: Any,
    env_cfg_override: dict[str, Any] | None,
    wrapper_cls: type,
    train_cfg: dict[str, Any],
    num_envs: int,
    num_steps: int,
    obs_shapes: dict[str, tuple[int, ...]],
    action_dim: int,
    distribution_param_shapes: tuple[tuple[int, ...], ...],
    shm_rollout_buffer_names: dict[str, str],
    sync_primitives: tuple[Any, Any],
    actor_weight_sync_name: str,
    actor_weight_param_shapes: dict[str, torch.Size],
    critic_weight_sync_name: str,
    critic_weight_param_shapes: dict[str, torch.Size],
    collector_actor_state_sync_name: str,
    collector_critic_state_sync_name: str,
    metrics_queue: Any,
    collector_device: str,
    seed: int | None,
    nan_guard_cfg: Any = None,
) -> None:
    """Collect PPO rollouts with the synced policy and publish shared slots."""
    from rsl_rl.algorithms import PPO

    from unilab.algos.torch.rsl_rl_async_ppo.runner import ppo_construct_cfg
    from unilab.ipc import SharedWeightSync
    from unilab.training import create_env

    ensure_registries()
    apply_training_seed(seed, torch_runtime=True, cuda=True)

    rollout_buffer = RslRlPpoRolloutBuffer(
        num_envs=num_envs,
        num_steps=num_steps,
        obs_shapes=obs_shapes,
        action_dim=action_dim,
        distribution_param_shapes=distribution_param_shapes,
        num_slots=1,
        create=False,
        shm_names=shm_rollout_buffer_names,
    )
    rollout_buffer.attach_sync_primitives(*sync_primitives)

    actor_weight_sync = SharedWeightSync(
        actor_weight_param_shapes, create=False, shm_name=actor_weight_sync_name
    )
    critic_weight_sync = SharedWeightSync(
        critic_weight_param_shapes, create=False, shm_name=critic_weight_sync_name
    )
    collector_actor_state_sync = SharedWeightSync(
        actor_weight_param_shapes, create=False, shm_name=collector_actor_state_sync_name
    )
    collector_critic_state_sync = SharedWeightSync(
        critic_weight_param_shapes, create=False, shm_name=collector_critic_state_sync_name
    )

    env = create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override)
    if nan_guard_cfg is not None and getattr(nan_guard_cfg, "enabled", False):
        from unilab.utils.nan_guard import NanGuard

        env.set_nan_guard(
            NanGuard(
                nan_guard_cfg,
                num_envs=env.num_envs,
                supports_state_playback=env.play_capabilities.supports_physics_state_playback,
            )
        )
    wrapped_env = wrapper_cls(env, device=collector_device)

    obs = wrapped_env.get_observations().to(collector_device)
    alg = PPO.construct_algorithm(obs, wrapped_env, ppo_construct_cfg(train_cfg), collector_device)
    alg.eval_mode()

    actor_sd = dict(alg.actor.state_dict())
    actor_version = actor_weight_sync.read_weights_into(actor_sd)
    alg.actor.load_state_dict(actor_sd)
    critic_sd = dict(alg.critic.state_dict())
    critic_version = critic_weight_sync.read_weights_into(critic_sd)
    alg.critic.load_state_dict(critic_sd)

    current_ep_rewards = torch.zeros(num_envs, device=collector_device)
    current_ep_lengths = torch.zeros(num_envs, device=collector_device)
    completed_rewards: list[float] = []
    completed_lengths: list[float] = []
    completed_reward_components: defaultdict[str, list[float]] = defaultdict(list)
    total_steps = 0

    try:
        while not stop_event.is_set():
            while rollout_buffer.available() >= 1 and not stop_event.is_set():
                time.sleep(0.001)
            if stop_event.is_set():
                break

            actor_sd = dict(alg.actor.state_dict())
            actor_version = actor_weight_sync.read_weights_into(actor_sd)
            alg.actor.load_state_dict(actor_sd)
            critic_sd = dict(alg.critic.state_dict())
            critic_version = critic_weight_sync.read_weights_into(critic_sd)
            alg.critic.load_state_dict(critic_sd)
            policy_version_start = min(actor_version, critic_version)

            write_buf = rollout_buffer.write_buffer
            rollout_collect_start = time.perf_counter()
            for step in range(num_steps):
                with torch.no_grad():
                    actions = alg.act(obs)
                for group, tensor in obs.items():
                    write_buf[f"obs/{group}"][:, step, ...] = _to_numpy(tensor).astype(
                        np.float32, copy=False
                    )
                write_buf["actions"][:, step, :] = _to_numpy(actions).astype(
                    np.float32, copy=False
                )
                write_buf["values"][:, step, :] = _to_numpy(alg.transition.values).astype(
                    np.float32, copy=False
                )
                write_buf["actions_log_prob"][:, step, :] = _to_numpy(
                    alg.transition.actions_log_prob.view(-1, 1)
                ).astype(np.float32, copy=False)
                for index, param in enumerate(alg.transition.distribution_params):
                    write_buf[f"distribution_params/{index}"][:, step, ...] = _to_numpy(
                        param
                    ).astype(np.float32, copy=False)

                next_obs, rewards, dones, extras = wrapped_env.step(actions.to(wrapped_env.device))
                if train_cfg.get("check_for_nan", True):
                    check_nan(next_obs, rewards, dones)
                next_obs = next_obs.to(collector_device)
                rewards = rewards.to(collector_device)
                dones = dones.to(collector_device)
                alg.actor.update_normalization(next_obs)
                alg.critic.update_normalization(next_obs)

                raw_rewards = rewards.clone()
                timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
                timeout_correction = torch.zeros_like(rewards)
                if isinstance(timeouts, torch.Tensor):
                    timeout_mask = timeouts.to(collector_device).bool()
                    bootstrap_obs = extras.get("time_out_bootstrap_obs")
                    if bootstrap_obs is not None and torch.count_nonzero(timeout_mask) > 0:
                        with torch.no_grad():
                            bootstrap_values = alg.critic(
                                bootstrap_obs.to(collector_device)
                            ).detach()
                        timeout_correction = alg.gamma * torch.squeeze(
                            bootstrap_values * timeout_mask.unsqueeze(1), 1
                        )
                    elif torch.count_nonzero(timeout_mask) > 0:
                        transition_values = alg.transition.values
                        timeout_correction = alg.gamma * torch.squeeze(
                            transition_values * timeout_mask.unsqueeze(1), 1
                        )
                    rewards = rewards + timeout_correction
                    write_buf["truncated"][:, step] = _to_numpy(timeout_mask.float()).astype(
                        np.float32, copy=False
                    )
                else:
                    write_buf["truncated"][:, step] = 0.0

                write_buf["rewards"][:, step] = _to_numpy(rewards).astype(np.float32, copy=False)
                write_buf["raw_rewards"][:, step] = _to_numpy(raw_rewards).astype(
                    np.float32, copy=False
                )
                write_buf["timeout_bootstrap_reward"][:, step] = _to_numpy(
                    timeout_correction
                ).astype(np.float32, copy=False)
                write_buf["dones"][:, step] = _to_numpy(dones.float()).astype(
                    np.float32, copy=False
                )

                current_ep_rewards += raw_rewards
                current_ep_lengths += 1
                if isinstance(extras, dict):
                    _record_reward_components(completed_reward_components, extras)
                done_idx = torch.nonzero(dones).flatten()
                if len(done_idx) > 0:
                    completed_rewards.extend(current_ep_rewards[done_idx].detach().cpu().tolist())
                    completed_lengths.extend(current_ep_lengths[done_idx].detach().cpu().tolist())
                    current_ep_rewards[done_idx] = 0
                    current_ep_lengths[done_idx] = 0

                obs = next_obs
                alg.transition.clear()
                alg.actor.reset(dones)
                alg.critic.reset(dones)
                total_steps += num_envs

            for group, tensor in obs.items():
                write_buf[f"last_obs/{group}"][:] = _to_numpy(tensor).astype(
                    np.float32, copy=False
                )
            policy_version_end = min(actor_weight_sync.version, critic_weight_sync.version)
            collector_actor_state_sync.write_weights(alg.actor.state_dict())
            collector_critic_state_sync.write_weights(alg.critic.state_dict())
            write_buf["policy_version_start"][0] = float(policy_version_start)
            write_buf["policy_version_end"][0] = float(policy_version_end)
            write_buf["rollout_created_time_ns"][0] = float(time.monotonic())
            write_buf["rollout_collect_time"][0] = float(time.perf_counter() - rollout_collect_start)
            rollout_buffer.signal_write_done()

            metrics: dict[str, Any] = {"total_steps": total_steps}
            if completed_rewards:
                recent_rewards = completed_rewards[-100:]
                recent_lengths = completed_lengths[-100:]
                metrics["mean_ep_reward"] = float(sum(recent_rewards) / len(recent_rewards))
                metrics["mean_ep_length"] = float(sum(recent_lengths) / len(recent_lengths))
            if completed_reward_components:
                metrics["reward_components"] = {
                    key: float(sum(values) / len(values))
                    for key, values in completed_reward_components.items()
                    if values
                }
                completed_reward_components.clear()
            _put_latest_metrics(metrics_queue, metrics)
    finally:
        wrapped_env.close()
        rollout_buffer.close()
        actor_weight_sync.close()
        critic_weight_sync.close()
        collector_actor_state_sync.close()
        collector_critic_state_sync.close()
