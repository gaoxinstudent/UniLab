"""RSL-RL-specific training helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import open_dict
from tensordict import TensorDict

from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.np_env import NpEnvState
from unilab.utils.tensor import to_numpy, to_torch


class _NullRslRlWriter:
    """Writer sink used when a PPO run explicitly disables metric logging.

    Recent RSL-RL releases accept only ``tensorboard``, ``wandb`` and
    ``neptune`` logger names, but UniLab's training contract also permits
    ``training.logger=none`` for headless runs.  The upstream runner uses a
    non-null writer as the condition for saving checkpoints, so this sink is
    deliberately retained instead of passing ``log_dir=None`` (which would
    silently disable checkpoint persistence as well).
    """

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def add_video(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def close(self) -> None:
        return None


def configure_rsl_rl_null_logger(runner: Any) -> None:
    """Make an RSL-RL runner honor UniLab's ``logger=none`` setting.

    RSL-RL initializes its writer lazily inside ``learn`` and raises for the
    otherwise valid UniLab value ``none``.  Replace only that initialization
    hook; tensorboard/W&B/Neptune modes continue to use the upstream logger,
    while checkpoint saves remain enabled through the non-null sink.
    """

    logger = getattr(runner, "logger", None)
    if logger is None:
        raise ValueError("RSL-RL runner must expose a logger to disable metric output")

    def _init_null_writer() -> None:
        logger.logger_type = "none"
        # Preserve RSL-RL's distributed rank gate.  Non-zero workers must
        # keep ``writer=None`` so they do not race rank zero while saving a
        # shared checkpoint path.
        logger.writer = None if bool(getattr(logger, "disable_logs", False)) else _NullRslRlWriter()

    logger.init_logging_writer = _init_null_writer


def apply_rsl_rl_rank_seed(cfg: Any, rank: int) -> int:
    """Apply RSL-RL's ``base seed + global rank`` data-parallel contract."""
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    base_seed = int(cfg.algo.seed)
    with open_dict(cfg):
        cfg.algo.seed = base_seed + int(rank)
    return int(cfg.algo.seed)


def resolve_rsl_rl_device(
    *,
    configured_device: str | None,
    devices: tuple[int, ...] | None,
    world_size: int,
    local_rank: int,
    default_device: str,
) -> str:
    """Resolve the exact device string expected by RSL-RL's runner.

    Integrated multi-GPU workers see the selected physical devices through a
    remapped ``CUDA_VISIBLE_DEVICES`` list, so RSL-RL must receive
    ``cuda:LOCAL_RANK`` rather than the original host-global device index.
    """
    if configured_device is not None and devices is not None:
        raise ValueError("Set either training.device or training.devices, not both")
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"local_rank={local_rank} is out of range for world_size={world_size}")
    if world_size > 1:
        if configured_device is not None:
            raise ValueError(
                "training.device cannot select one device in a distributed run; "
                "use training.devices"
            )
        if devices is not None and len(devices) != world_size:
            raise ValueError(
                f"training.devices has {len(devices)} entries but WORLD_SIZE={world_size}"
            )
        return f"cuda:{local_rank}"
    if devices is not None:
        return f"cuda:{devices[0]}"
    return configured_device or default_device


def ppo_samples_per_iteration(*, num_envs: int, num_steps_per_env: int, world_size: int) -> int:
    """Return the global fresh rollout sample count for one PPO iteration."""
    return int(num_envs) * int(num_steps_per_env) * int(world_size)


def finish_rsl_rl_distributed(*, training_succeeded: bool) -> None:
    """Synchronize successful ranks and release RSL-RL's process group."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    try:
        if training_succeeded:
            torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@contextmanager
def rsl_rl_single_process_topology() -> Iterator[None]:
    """Temporarily hide torchrun's worker topology from rank-0-only work.

    Destroying a process group does not clear ``WORLD_SIZE`` / ``RANK`` /
    ``LOCAL_RANK``. RSL-RL would therefore initialize a second distributed
    group when rank 0 constructs a fresh runner for post-training playback,
    even though every other rank has already exited. Present the playback
    scope as a single-process runtime, then restore launcher-owned variables.
    """
    single_process_topology = {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    previous = {name: os.environ.get(name) for name in single_process_topology}
    os.environ.update(single_process_topology)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def get_policy_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:
    """Return ``(actor_obs_dim, flat_policy_obs_dim)`` for RSL-RL policies."""
    actor_obs_dim = int(obs_groups_spec.get("obs", 0))
    flat_policy_obs_dim = int(
        sum(dim for group_name, dim in obs_groups_spec.items() if group_name != "critic")
    )
    return actor_obs_dim, flat_policy_obs_dim or actor_obs_dim


def normalize_ppo_train_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map UniLab PPO owner config to the current RSL-RL schema."""
    normalized = deepcopy(train_cfg)
    algorithm_cfg = normalized.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        # ``enable_compile`` belongs to UniLab's FinalObservationAwarePPO
        # wrapper.  Upstream rsl_rl.algorithms.ppo.PPO has no such argument;
        # keep the source-vanilla algorithm selectable for migration A/B runs
        # without leaking owner-only options into its constructor.
        algorithm_class = str(algorithm_cfg.get("class_name", ""))
        if not algorithm_class.startswith("unilab."):
            algorithm_cfg.pop("enable_compile", None)
        for key in (
            "target_kl_stop",
            "adaptive_kl_beta",
            "adaptive_lr_growth",
            "adaptive_lr_decay",
            "adaptive_lr_update_interval",
            "metrics_interval",
            "finite_check_interval",
            "warmup_strict_iters",
            "warmup_metrics_interval",
            "warmup_finite_check_interval",
            "disable_finite_checks",
        ):
            algorithm_cfg.pop(key, None)

    if "actor" in normalized and "critic" in normalized:
        return normalized

    policy_cfg = normalized.pop("policy", None)
    if not isinstance(policy_cfg, dict):
        return normalized

    actor_hidden_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
    critic_hidden_dims = policy_cfg.get("critic_hidden_dims", actor_hidden_dims)
    activation = policy_cfg.get("activation", "elu")
    init_noise_std = float(policy_cfg.get("init_noise_std", 1.0))
    obs_normalization = bool(normalized.get("empirical_normalization", False))

    normalized["actor"] = {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": actor_hidden_dims,
        "activation": activation,
        "obs_normalization": obs_normalization,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": init_noise_std,
            "std_type": "scalar",
        },
    }
    normalized["critic"] = {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": critic_hidden_dims,
        "activation": activation,
        "obs_normalization": obs_normalization,
    }

    obs_groups = normalized.get("obs_groups")
    if isinstance(obs_groups, dict) and "actor" not in obs_groups and "default" in obs_groups:
        default_groups = obs_groups.pop("default")
        if isinstance(default_groups, list) and default_groups:
            obs_groups["actor"] = list(default_groups)

    return normalized


# The upstream ``wheeled-legged_RL`` runner serializes its vanilla PPO models
# under one ``model_state_dict`` with keys such as ``actor.0.weight`` and a
# top-level ``std`` parameter.  Current RSL-RL keeps the two models separate
# and prefixes the same layers with ``mlp``.  Keep this translation at the
# training/checkpoint boundary so scripts do not grow a second model format.
_WHEELBIPE_SOURCE_MLP_KEY = re.compile(r"^(actor|critic)\.(\d+)\.(weight|bias)$")


def _extract_wheelbipe_source_ppo_state(
    payload: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Validate and extract an upstream Wheelbipe vanilla-PPO state mapping.

    The source project has several custom algorithms with a ``model_state_dict``
    key as well.  Requiring the exact vanilla ``actor.*``/``critic.*``/``std``
    shape keeps this adapter from silently treating those compact graphs as a
    standard actor.  Structural validation happens before any target module is
    touched, so malformed artifacts fail closed without partial loading.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Upstream Wheelbipe PPO checkpoint must be a mapping, got {type(payload).__name__}"
        )
    raw_state = payload.get("model_state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError(
            "Upstream Wheelbipe PPO checkpoint must contain a mapping under model_state_dict"
        )
    if not raw_state:
        raise ValueError("Upstream Wheelbipe PPO model_state_dict cannot be empty")

    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str):
            raise ValueError("Upstream Wheelbipe PPO model_state_dict keys must be strings")
        if key != "std" and _WHEELBIPE_SOURCE_MLP_KEY.fullmatch(key) is None:
            raise ValueError(
                "Upstream Wheelbipe PPO model_state_dict contains unsupported key "
                f"{key!r}; expected std, actor.<layer>.<weight|bias>, or "
                "critic.<layer>.<weight|bias>"
            )
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "Upstream Wheelbipe PPO model_state_dict values must be tensors; "
                f"key {key!r} has {type(value).__name__}"
            )
        if not value.is_floating_point():
            raise ValueError(
                "Upstream Wheelbipe PPO model_state_dict tensors must use a floating dtype; "
                f"key {key!r} has dtype {value.dtype}"
            )
        state[key] = value

    std = state.get("std")
    if std is None:
        raise ValueError("Upstream Wheelbipe PPO model_state_dict is missing std")
    if std.ndim != 1 or std.numel() == 0:
        raise ValueError(
            "Upstream Wheelbipe PPO std must be a non-empty rank-1 tensor, "
            f"got shape {tuple(std.shape)}"
        )
    if not bool(torch.isfinite(std).all()):
        raise ValueError("Upstream Wheelbipe PPO std contains non-finite values")
    if not bool(torch.gt(std, 0).all()):
        raise ValueError("Upstream Wheelbipe PPO std must contain strictly positive values")

    for branch in ("actor", "critic"):
        branch_keys = [key for key in state if key.startswith(f"{branch}.")]
        if not branch_keys:
            raise ValueError(f"Upstream Wheelbipe PPO model_state_dict is missing the {branch} MLP")
        layer_parameters: dict[str, set[str]] = {}
        for key in branch_keys:
            match = _WHEELBIPE_SOURCE_MLP_KEY.fullmatch(key)
            # The key was checked above; this guard keeps the invariant obvious
            # to both readers and static analyzers.
            if match is None:
                raise ValueError(f"Invalid upstream Wheelbipe PPO {branch} key {key!r}")
            layer_parameters.setdefault(match.group(2), set()).add(match.group(3))
        incomplete = sorted(
            layer
            for layer, parameters in layer_parameters.items()
            if parameters != {"weight", "bias"}
        )
        if incomplete:
            raise ValueError(
                f"Upstream Wheelbipe PPO {branch} MLP has incomplete weight/bias pairs "
                f"for layer(s) {incomplete}"
            )

    return state


def is_wheelbipe_source_ppo_checkpoint(payload: object) -> bool:
    """Return whether ``payload`` has the strict upstream vanilla-PPO format."""

    if not isinstance(payload, Mapping):
        return False
    try:
        _extract_wheelbipe_source_ppo_state(payload)
    except (TypeError, ValueError, RuntimeError):
        return False
    return True


def _validate_wheelbipe_target_state(
    target_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    """Check keys and tensor shapes before invoking ``load_state_dict``."""

    target_keys = set(target_state)
    candidate_keys = set(candidate_state)
    missing = sorted(target_keys - candidate_keys)
    unexpected = sorted(candidate_keys - target_keys)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(
            "Upstream Wheelbipe PPO "
            f"{label} state does not match the target RSL-RL model (" + "; ".join(details) + "). "
            "Use the matching Wheelbipe policy hidden dimensions and disable observation "
            "normalization when loading a source checkpoint."
        )

    for key in sorted(target_keys):
        target_shape = tuple(target_state[key].shape)
        candidate_shape = tuple(candidate_state[key].shape)
        if target_shape != candidate_shape:
            # Keep the canonical phrase used by ``policy_load_dim_guard`` so a
            # cross-backend shape mismatch is surfaced with its richer context.
            raise ValueError(
                f"size mismatch for {label}.{key}: checkpoint shape {candidate_shape}, "
                f"target shape {target_shape}"
            )


def load_wheelbipe_source_ppo_checkpoint(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    *,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """Load an upstream Wheelbipe vanilla-PPO checkpoint into RSL-RL models.

    Upstream checkpoints contain one ``model_state_dict`` and an optimizer
    state whose parameter IDs belong to the source runner.  Playback only needs
    the actor/critic weights, so this adapter deliberately does not restore the
    source optimizer or iteration counters.  Training resume continues to use
    the native RSL-RL checkpoint contract.

    Both target modules are preflight-validated before either is mutated.  A
    hidden-dimension, observation-dimension, or normalization mismatch thus
    produces an actionable error rather than a partially loaded policy.
    """

    if isinstance(checkpoint, Mapping):
        payload: Any = checkpoint
    else:
        payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Upstream Wheelbipe PPO checkpoint must be a mapping, got {type(payload).__name__}"
        )
    source_state = _extract_wheelbipe_source_ppo_state(payload)

    actor_target = actor.state_dict()
    critic_target = critic.state_dict()
    actor_candidate: dict[str, torch.Tensor] = {}
    critic_candidate: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key == "std":
            continue
        match = _WHEELBIPE_SOURCE_MLP_KEY.fullmatch(key)
        if match is None:  # pragma: no cover - extraction validates this branch
            raise ValueError(f"Invalid upstream Wheelbipe PPO model key {key!r}")
        target_key = f"mlp.{match.group(2)}.{match.group(3)}"
        if match.group(1) == "actor":
            actor_candidate[target_key] = value
        else:
            critic_candidate[target_key] = value

    if "distribution.std_param" in actor_target:
        std_key = "distribution.std_param"
        actor_candidate[std_key] = source_state["std"]
    elif "distribution.log_std_param" in actor_target:
        std_key = "distribution.log_std_param"
        actor_candidate[std_key] = torch.log(source_state["std"])
    else:
        raise ValueError(
            "Target RSL-RL actor has no supported Gaussian standard-deviation parameter; "
            "the upstream Wheelbipe PPO checkpoint requires distribution.std_param "
            "or distribution.log_std_param"
        )

    _validate_wheelbipe_target_state(actor_target, actor_candidate, label="actor")
    _validate_wheelbipe_target_state(critic_target, critic_candidate, label="critic")

    # The preflight above ensures this pair of strict loads cannot expose a
    # key/shape mismatch after mutating only one of the two models.
    try:
        actor.load_state_dict(actor_candidate, strict=True)
        critic.load_state_dict(critic_candidate, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "Upstream Wheelbipe PPO checkpoint could not be loaded into the target "
            f"RSL-RL models: {exc}"
        ) from exc
    return dict(payload)


def maybe_load_wheelbipe_source_ppo_checkpoint(
    runner: Any,
    checkpoint: str | os.PathLike[str],
    *,
    map_location: str | torch.device | None = "cpu",
) -> bool:
    """Load a source Wheelbipe checkpoint for a fresh RSL-RL training run.

    The upstream artifact has no compatible RSL-RL optimizer state, rollout
    storage, or runner iteration counter.  When the checkpoint is in the
    strict source format, copy only actor/critic weights into the already
    materialized runner and leave its optimizer and iteration at their fresh
    values.  Native RSL-RL checkpoints return ``False`` so the caller can use
    the normal resume path unchanged.
    """

    payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not is_wheelbipe_source_ppo_checkpoint(payload):
        return False
    try:
        algorithm = runner.alg
        actor = algorithm.actor
        critic = algorithm.critic
    except AttributeError as exc:
        raise RuntimeError(
            "Source Wheelbipe PPO warm-start requires an RSL-RL runner exposing "
            "alg.actor and alg.critic"
        ) from exc
    load_wheelbipe_source_ppo_checkpoint(
        actor,
        critic,
        payload,
        map_location=map_location,
    )
    return True


class RslRlVecEnvWrapper:
    """Adapter from UniLab's env contract to the RSL-RL VecEnv contract."""

    def __init__(
        self,
        env: Any,
        device: str = "cpu",
        policy_obs_mode: str = "flat",
    ) -> None:
        if policy_obs_mode == "auto":
            policy_obs_mode = "flat"
        if policy_obs_mode not in {"actor", "flat"}:
            raise ValueError(
                f"Unsupported policy_obs_mode={policy_obs_mode!r}; expected 'actor' or 'flat'."
            )

        self.env = env
        self.cfg = env.cfg
        self.device = device
        self.policy_obs_mode = policy_obs_mode
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._actor_obs_dim, self._flat_obs_dim = get_policy_obs_dims(env.obs_groups_spec)
        self.num_obs = (
            self._actor_obs_dim if self.policy_obs_mode == "actor" else self._flat_obs_dim
        )
        self.num_privileged_obs = int(env.obs_groups_spec.get("critic", self.num_obs))
        action_shape = env.action_space.shape
        if action_shape is None:
            raise ValueError("env.action_space.shape must be defined")
        self.num_actions = int(action_shape[0])

        self.episode_returns = torch.zeros(self.num_envs, device=device)
        # RSL-RL uses ``episode_length_buf`` as a public initialization hook:
        # its runner assigns a randomly sampled length before the first
        # rollout.  Keep that hook backed by the same tensor used for logging
        # and, importantly, mirror assignments into UniLab's authoritative
        # ``state.info['steps']`` counter.  Without this bridge the runner's
        # random-start contract only changes a detached wrapper tensor while
        # NpEnv still starts every episode at step zero.
        self._episode_length_buf = torch.zeros(self.num_envs, device=device)
        self.episode_lengths = self._episode_length_buf
        self.episode_length_buf = self._episode_length_buf
        self.max_episode_length = np.ceil(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)
        self.reset()

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Current episode lengths exposed through the RSL-RL VecEnv contract."""

        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Set episode lengths and synchronize the underlying NpEnv counter.

        Upstream RSL-RL assigns a new tensor here for
        ``init_at_random_ep_len``.  Treat that assignment as an owner-layer
        lifecycle event rather than allowing the wrapper and NpEnv to drift.
        Lightweight test doubles may not expose ``state.info['steps']``; in
        that case the public wrapper buffer still behaves normally.
        """

        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device)
        value = value.to(device=self.device)
        if value.ndim != 1 or value.shape[0] != self.num_envs:
            raise ValueError(
                "RSL-RL episode_length_buf must have shape "
                f"({self.num_envs},), got {tuple(value.shape)}"
            )
        self._episode_length_buf = value
        # Keep the wrapper's logging alias in lockstep even when upstream
        # replaces the tensor object rather than mutating it in-place.
        self.episode_lengths = self._episode_length_buf
        self._sync_episode_length_to_env()

    def _sync_episode_length_to_env(self) -> None:
        """Mirror the public RSL-RL episode counter into NpEnv state."""

        # Some source-compatible owners intentionally keep RSL-RL's random
        # episode-length buffer as a learner/logging-only counter.  In that
        # mode the wrapper must not inject a synthetic age into the owner's
        # authoritative reset state.  The opt-out is an explicit owner config
        # field; environments without it retain the generic synchronization
        # contract (and its tests).
        cfg = getattr(self.env, "cfg", None)
        if getattr(cfg, "source_episode_age_sync", None) is False:
            return

        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        if not isinstance(info, dict):
            return
        steps = info.get("steps")
        if not isinstance(steps, np.ndarray) or steps.shape != (self.num_envs,):
            return
        np.copyto(steps, to_numpy(self._episode_length_buf), casting="unsafe")
        # Source Wheelbipe reset envelopes are expressed in absolute episode
        # time in IsaacLab.  Vanilla RSL-RL assigns a randomized age after the
        # initial reset, so give that owner one explicit lifecycle hook to
        # consume the age without resetting the environment or leaking the
        # rule into the generic NpEnv contract.
        sync_source_age = getattr(self.env, "sync_source_episode_length", None)
        if callable(sync_source_age):
            sync_source_age(np.asarray(steps))

    def sync_training_iteration(self, iteration: int) -> None:
        """Forward a resumed runner iteration to an owner, when supported.

        This is a lifecycle hook rather than a Wheelbipe-specific rule: most
        environments simply do not implement it.  Source-compatible owners
        use it to keep curriculum schedules aligned with the checkpoint's
        stored RSL-RL iteration after a fresh environment materialization.
        """

        sync_iteration = getattr(self.env, "sync_training_iteration", None)
        if callable(sync_iteration):
            sync_iteration(int(iteration))

    def _policy_obs(self, obs: dict[str, Any]) -> torch.Tensor:
        if self.policy_obs_mode == "actor":
            return to_torch(obs["obs"], self.device)

        policy_groups = [
            to_numpy(value) for group_name, value in obs.items() if group_name != "critic"
        ]
        if not policy_groups:
            raise KeyError("Observation dict must contain at least one non-critic group")
        if len(policy_groups) == 1:
            return to_torch(policy_groups[0], self.device)
        return to_torch(np.concatenate(policy_groups, axis=1), self.device)

    def _obs_to_tensordict(
        self,
        obs: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> TensorDict:
        del info
        actor_obs = to_torch(obs["obs"], self.device)
        td_dict: dict[str, torch.Tensor] = {
            "actor": actor_obs,
            "policy": self._policy_obs(obs),
        }
        if "critic" in obs:
            td_dict["critic"] = to_torch(obs["critic"], self.device)
        # Preserve explicitly named source observation leaves for custom
        # runners (notably NP3O's ``on_constraint`` / ``policy_hist`` /
        # ``priv_latent`` streams).  Normal RSL-RL callers continue to consume
        # only actor/policy/critic; retaining the leaves here avoids forcing a
        # source adapter to reconstruct them from backend-private state.
        for key, raw_value in obs.items():
            if key in {"obs", "critic", "actor", "policy"} or key in td_dict:
                continue
            if isinstance(raw_value, (torch.Tensor, np.ndarray)):
                td_dict[str(key)] = to_torch(raw_value, self.device)
        return TensorDict(td_dict, batch_size=self.num_envs, device=self.device)

    def _resolve_final_observation(self, state: NpEnvState) -> dict[str, Any] | None:
        if isinstance(state.final_observation, dict):
            return state.final_observation
        if isinstance(state.info, dict):
            final_observation = state.info.get("final_observation")
            if isinstance(final_observation, dict):
                return final_observation
        return None

    def _resolve_done(self, state: NpEnvState) -> torch.Tensor:
        return to_torch(state.terminated | state.truncated, self.device).bool()

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = to_numpy(actions)
        state = self.env.step(actions_np)
        rewards = to_torch(state.reward, self.device)
        dones = self._resolve_done(state)

        self.episode_returns += rewards
        self.episode_lengths += 1

        infos: dict[str, torch.Tensor | TensorDict | dict[str, Any]] = {}
        done_idx = torch.nonzero(dones).flatten()
        if len(done_idx) > 0:
            infos["time_outs"] = to_torch(state.truncated, self.device).bool()

            final_observation = self._resolve_final_observation(state)
            terminal_contract = resolve_terminal_observation_contract(
                next_obs_batch_size=self.num_envs,
                final_observation=final_observation,
                done=to_numpy(dones),
                info=state.info,
                truncated=to_numpy(infos["time_outs"]),
            )
            if np.any(terminal_contract.timeout_terminal_mask) and final_observation is not None:
                infos["time_out_bootstrap_obs"] = self._obs_to_tensordict(final_observation)

            self.episode_returns[done_idx] = 0
            self.episode_lengths[done_idx] = 0

        if "log" in state.info:
            infos["log"] = state.info["log"]
        if "costs" in state.info:
            infos["costs"] = to_torch(state.info["costs"], self.device)

        return (
            self._obs_to_tensordict(state.obs, getattr(state, "info", None)),
            rewards,
            dones,
            infos,
        )

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        if self.env.state is None:
            self.env.init_state()

        env_indices = np.arange(self.num_envs, dtype=np.int32)
        obs_out, info = self.env.reset(env_indices)
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return self._obs_to_tensordict(obs_out, info), info

    def get_observations(self) -> TensorDict:
        assert self.env.state is not None
        return self._obs_to_tensordict(self.env.state.obs, self.env.state.info)

    def get_privileged_observations(self) -> torch.Tensor:
        assert self.env.state is not None
        obs = self.env.state.obs
        return to_torch(obs.get("critic", obs["obs"]), self.device)

    def close(self) -> None:
        self.env.close()
