"""Gymnasium compatibility entry points for the upstream WheelBipe IDs.

The UniLab registry remains the owner of configuration and backend selection.
This module only supplies the small single-environment adapter expected by the
upstream ``gym.make`` examples.  It deliberately does not duplicate task
logic or bypass :mod:`unilab.base.registry`: exact source IDs therefore select
the same named source-semantic owners and fail at the same validation boundary
as the UniLab CLI.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Callable, TypeVar

import gymnasium as gym
import numpy as np

from unilab.base import registry

_OwnerResult = TypeVar("_OwnerResult")


def _construction_seed(task_id: str, sim_backend: str) -> int:
    """Return a stable seed for cold-path assets built by the Gym facade.

    Rough WheelBipe owners generate their procedural heightfield in
    ``registry.make`` (before Gymnasium can deliver ``reset(seed=...)``).  The
    terrain generator falls back to the process-global NumPy stream when no
    explicit terrain seed is configured.  If two facades are constructed in
    sequence, they would otherwise receive different static maps even when
    both are reset with the same seed.  A stable task/backend-derived seed
    keeps that cold-path asset deterministic and leaves the caller's stream
    untouched; dynamic reset/step draws are still controlled by the facade's
    per-instance RNG snapshot below.
    """

    digest = hashlib.blake2s(
        f"unilab-wheelbipe-gym:{task_id}:{sim_backend}".encode("utf-8"),
        digest_size=4,
    ).digest()
    # NumPy's legacy RandomState accepts the full uint32 range except for
    # values that are not representable as a signed Python int on some older
    # versions; converting explicitly keeps behavior stable across versions.
    return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF


def _unbatch_info(value: Any) -> Any:
    """Return a detached one-environment view while preserving nested metadata."""

    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
        return value[0].copy()
    if isinstance(value, Mapping):
        return {str(key): _unbatch_info(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unbatch_info(item) for item in value)
    return value


class WheelbipeGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-environment Gymnasium facade over a UniLab WheelBipe owner.

    ``NpEnv`` is intentionally vectorized and returns ``NpEnvState``.  The
    upstream examples use the ordinary Gymnasium five-value API, so this
    facade constrains construction to one environment and unwraps row zero.
    The policy observation group (normal 35D or compact 28D) is returned as a
    flat array; privileged observations remain available in ``info`` under
    ``"critic_observation"``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        task_id: str,
        sim_backend: str = "mujoco",
        env_cfg_override: dict[str, Any] | None = None,
        render_mode: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(
                "Wheelbipe Gym adapter supports render_mode=None, 'human', or 'rgb_array'; "
                f"got {render_mode!r}"
            )
        self.task_id = str(task_id)
        self.sim_backend = str(sim_backend).strip().lower()
        self.render_mode = render_mode
        # The vectorized NumPy owner predates Gymnasium's per-environment RNG
        # contract and currently draws from ``np.random``'s process-global
        # RandomState in reset/step/domain-randomization paths.  Keep a
        # private snapshot for this single-env facade so ``reset(seed=...)``
        # controls the complete subsequent trajectory without leaking draws
        # into another Gym env (or into the caller's global RNG).  The owner
        # itself remains unchanged and vectorized training keeps its existing
        # run-level seed behavior.
        self._owner_rng_state: tuple[Any, ...] | None = None
        # ``registry.make`` performs the exact source capability validation.
        # In particular, this call must not translate an exact source id to a
        # nearby flat owner with different state-machine/gimbal semantics.
        # Rough owners also materialize a procedural heightfield on this cold
        # path.  Build it under a stable task/backend stream and restore the
        # caller's stream immediately; otherwise two facades constructed in
        # sequence would have different static maps before ``reset(seed=...)``
        # has a chance to establish their per-env stream.
        caller_state = np.random.get_state()
        try:
            np.random.seed(_construction_seed(self.task_id, self.sim_backend))
            self._env = registry.make(
                self.task_id,
                sim_backend=self.sim_backend,
                env_cfg_override=env_cfg_override,
                num_envs=1,
            )
        finally:
            np.random.set_state(caller_state)
        self._closed = False
        self._renderer_initialized = False
        policy_dim = int(self._env.obs_groups_spec.get("obs", 0))
        if policy_dim < 1:
            self._env.close()
            raise ValueError(
                f"Wheelbipe owner {self.task_id!r} has no positive policy observation dimension"
            )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(policy_dim,),
            dtype=np.float32,
        )
        action_space = self._env.action_space
        # Keep a detached copy: callers should not be able to mutate the
        # vectorized owner's space through the Gym facade.
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_space.low, dtype=np.float32).reshape(-1),
            high=np.asarray(action_space.high, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )

    @property
    def unilab_env(self) -> Any:
        """Expose the owner for advanced diagnostics without changing Gym I/O."""

        return self._env

    def _info_from_state(self, state: Any) -> dict[str, Any]:
        info = _unbatch_info(getattr(state, "info", {}))
        if not isinstance(info, dict):
            info = {"owner_info": info}
        critic = getattr(state, "obs", {}).get("critic")
        if isinstance(critic, np.ndarray) and critic.ndim == 2 and critic.shape[0] == 1:
            info["critic_observation"] = critic[0].copy()
        info["wheelbipe_task_id"] = self.task_id
        info["unilab_sim_backend"] = self.sim_backend
        return info

    def _run_owner_with_rng(
        self,
        callback: Callable[[], _OwnerResult],
        *,
        seed: int | None = None,
    ) -> _OwnerResult:
        """Run an owner operation under this facade's isolated NumPy stream.

        WheelBipe's NumPy owner deliberately uses the legacy process-global
        ``np.random`` API in several cold and hot paths.  Gymnasium, in
        contrast, promises that an explicit reset seed belongs to one env and
        that two identically seeded envs produce the same trajectory even when
        stepped interleaved.  Snapshotting/restoring the caller state around
        each owner call gives us that boundary without broad, risky rewrites
        of every owner random draw.  Once initialized, ``reset(None)`` follows
        the stream left by the previous call, matching Gymnasium semantics.
        """

        caller_state = np.random.get_state()
        if seed is not None:
            np.random.seed(int(seed))
        elif self._owner_rng_state is not None:
            np.random.set_state(self._owner_rng_state)
        try:
            result = callback()
            # Capture the state *after* all owner draws, including autoreset
            # randomization performed inside ``step``.
            self._owner_rng_state = np.random.get_state()
            return result
        finally:
            np.random.set_state(caller_state)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if self._closed:
            raise RuntimeError("Wheelbipe Gym environment is closed")
        super().reset(seed=seed)
        # NpEnv owns vectorized reset/randomization.  Execute it through the
        # isolated stream so Gym's explicit seed controls all owner draws.
        state = self._run_owner_with_rng(self._env.init_state, seed=seed)
        obs = np.asarray(state.obs["obs"][0], dtype=np.float32).copy()
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                "Wheelbipe owner returned an observation with the wrong Gym shape: "
                f"expected {self.observation_space.shape}, got {obs.shape}"
            )
        return obs, self._info_from_state(state)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Wheelbipe Gym environment is closed")
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape != self.action_space.shape:
            raise ValueError(
                f"Wheelbipe action must have shape {self.action_space.shape}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Wheelbipe action must contain only finite values")
        state = self._run_owner_with_rng(lambda: self._env.step(arr[None, :]))
        obs = np.asarray(state.obs["obs"][0], dtype=np.float32).copy()
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                "Wheelbipe owner returned an observation with the wrong Gym shape: "
                f"expected {self.observation_space.shape}, got {obs.shape}"
            )
        reward = float(np.asarray(state.reward).reshape(-1)[0])
        terminated = bool(np.asarray(state.terminated).reshape(-1)[0])
        truncated = bool(np.asarray(state.truncated).reshape(-1)[0])
        return obs, reward, terminated, truncated, self._info_from_state(state)

    def render(self) -> np.ndarray | None:
        if self._closed or self.render_mode is None:
            return None
        if not self._renderer_initialized:
            self._env.init_play_renderer(
                headless=self.render_mode == "rgb_array",
                capture=self.render_mode == "rgb_array",
            )
            self._renderer_initialized = True
        if self.render_mode == "rgb_array":
            return np.asarray(self._env.capture_play_video_frame(), dtype=np.uint8).copy()
        self._env.render_play_frame()
        return None

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True


def make_wheelbipe_gym_env(
    *,
    task_id: str,
    sim_backend: str = "mujoco",
    **kwargs: Any,
) -> WheelbipeGymEnv:
    """Gym entry point used by the registered upstream task IDs."""

    return WheelbipeGymEnv(task_id=task_id, sim_backend=sim_backend, **kwargs)


def register_wheelbipe_gym_envs() -> tuple[str, ...]:
    """Register the exact upstream IDs once and return the registered names."""

    # Import lazily so this module can be imported while registry bootstrap is
    # still loading the variant classes.
    from .variants import UPSTREAM_WHEELBIPE_TASK_IDS

    registered: list[str] = []
    for task_id in UPSTREAM_WHEELBIPE_TASK_IDS:
        existing = gym.registry.get(task_id)
        if existing is not None:
            expected_entry = (
                "unilab.envs.locomotion.wheelbipe_v14.gym_adapter:make_wheelbipe_gym_env"
            )
            if str(existing.entry_point) != expected_entry:
                raise RuntimeError(
                    f"Gymnasium id {task_id!r} is already registered by "
                    f"{existing.entry_point!r}, refusing to replace it"
                )
            registered.append(task_id)
            continue
        gym.register(
            id=task_id,
            entry_point=("unilab.envs.locomotion.wheelbipe_v14.gym_adapter:make_wheelbipe_gym_env"),
            kwargs={"task_id": task_id},
            disable_env_checker=True,
        )
        registered.append(task_id)
    return tuple(registered)


__all__ = ["WheelbipeGymEnv", "make_wheelbipe_gym_env", "register_wheelbipe_gym_envs"]
