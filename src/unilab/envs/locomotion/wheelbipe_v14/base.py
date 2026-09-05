"""Wheelbipe V14 owner-layer control and observation contracts.

The deployment model has eight MuJoCo actuators (four active leg motors, two
wheels and two gas-spring motors), while the learned policy exposes six
actions.  This module keeps that distinction explicit: policy actions and
observations are always in the public ROS order, and the backend actuator
mapping is resolved once during environment construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)
from unilab.utils.rotation import np_quat_apply_inverse

from .state_machine import WheelbipeStateMachineConfig

# Public order defined by the ROS2 controller contract.  Do not replace this
# with MuJoCo's XML actuator order (the latter interleaves wheel/spring slots).
POLICY_JOINT_NAMES: tuple[str, ...] = (
    "left_front1_joint",
    "left_rear1_joint",
    "right_front1_joint",
    "right_rear1_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)
LEG_JOINT_NAMES: tuple[str, ...] = POLICY_JOINT_NAMES[:4]
WHEEL_JOINT_NAMES: tuple[str, ...] = POLICY_JOINT_NAMES[4:]
SPRING_JOINT_NAMES: tuple[str, ...] = ("left_spring2_joint", "right_spring2_joint")
NATIVE_ACTUATOR_NAMES: tuple[str, ...] = (
    "left_front1_joint_ctrl",
    "left_rear1_joint_ctrl",
    "left_wheel_joint_ctrl",
    "left_spring2_joint_ctrl",
    "right_front1_joint_ctrl",
    "right_rear1_joint_ctrl",
    "right_wheel_joint_ctrl",
    "right_spring2_joint_ctrl",
)
# The source Isaac V14 asset adds these two physical channels after the
# historical eight-actuator ROS/MJCF order.  They are deliberately separate
# constants so a normal owner cannot silently acquire extra policy outputs.
GIMBAL_JOINT_NAMES: tuple[str, ...] = ("gimbal_yaw_joint", "gimbal_pitch_joint")
GIMBAL_ACTUATOR_NAMES: tuple[str, ...] = (
    "gimbal_yaw_joint_ctrl",
    "gimbal_pitch_joint_ctrl",
)
NATIVE_ACTUATOR_NAMES_WITH_GIMBAL: tuple[str, ...] = NATIVE_ACTUATOR_NAMES + GIMBAL_ACTUATOR_NAMES

NUM_POLICY_ACTIONS = 6
NUM_LEG_ACTIONS = 4
NUM_WHEEL_ACTIONS = 2
NUM_NATIVE_ACTUATORS = 8
NUM_GIMBAL_ACTUATORS = 2
NUM_NATIVE_ACTUATORS_WITH_GIMBAL = NUM_NATIVE_ACTUATORS + NUM_GIMBAL_ACTUATORS
POLICY_OBS_DIM = 35
PRIVILEGED_OBS_DIM = 78
# The upstream history-based WheelBipe variants (DreamWaQ/HIM/NP3O) disable
# the seven-element control-mode tail.  Their one-step policy frame is 28D
# and the privileged frame appends body-frame linear velocity (3) plus the
# observed height (1), for a 32D critic input.  Keep these dimensions in the
# owner contract rather than deriving them by slicing a normal-mode tensor in
# the runner.
COMPACT_POLICY_OBS_DIM = 28
COMPACT_PRIVILEGED_OBS_DIM = 32
# Source V14 compact privileged observations apply the same named input
# contracts as the Isaac owner: body-frame linear velocity is clipped to
# ``[-100, 100]`` (with no additional scale), while measured height is clipped
# in raw metres to ``[-10, 10]`` and then multiplied by five.  Keep these
# values explicit at the owner boundary so a compact critic cannot silently
# drift from the source checkpoint's feature scaling.
COMPACT_PRIVILEGED_ROOT_LINVEL_CLIP = 100.0
COMPACT_PRIVILEGED_OBS_HEIGHT_CLIP: tuple[float, float] = (-10.0, 10.0)
COMPACT_PRIVILEGED_OBS_HEIGHT_SCALE = 5.0
# The ROS deployment controller applies a finite absolute clamp to every
# basic policy-observation field before invoking ONNX.  Keep the same bound at
# the owner boundary so malformed sensor/command values cannot silently turn
# into an out-of-contract policy input.  (The fixed normal-mode one-hot block
# is already inside this range.)
POLICY_OBS_CLIP = 100.0
NORMAL_CONTROL_MODE = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# The upstream Isaac task expresses sensor/action latency as an integer lag
# in *physics* steps.  Keep the representation explicit at the owner layer so
# a deployment profile cannot accidentally turn a physical-step contract into
# the older one-control-step ``simulate_action_latency`` shortcut.
DelayRange = tuple[int, int]
# Delay profiles are intentionally owner-level names rather than claims about
# complete upstream/source parity.  ``local_physics`` is the stable UniLab
# contract (and preserves the historical inclusive sampler); the source
# profile is explicit and fail-closed when its timing assumptions are absent.
WHEELBIPE_DELAY_PROFILE_LOCAL = "local_physics"
WHEELBIPE_DELAY_PROFILE_SOURCE_V14 = "source_v14_physics"
# Short constant alias for callers that do not need the versioned spelling.
WHEELBIPE_DELAY_PROFILE_SOURCE = WHEELBIPE_DELAY_PROFILE_SOURCE_V14
WHEELBIPE_DELAY_PROFILES: tuple[str, ...] = (
    WHEELBIPE_DELAY_PROFILE_LOCAL,
    WHEELBIPE_DELAY_PROFILE_SOURCE_V14,
)
WHEELBIPE_DELAY_RANGE_SEMANTICS: tuple[str, ...] = ("inclusive", "exclusive")
WHEELBIPE_OBS_DELAY_ALIASES: dict[str, str] = {
    "root_ang_vel_b": "gyro",
    "gyro": "gyro",
    "projected_gravity_b": "gravity",
    "projected_gravity": "gravity",
    "gravity": "gravity",
    "joint_pos": "joint_pos",
    "joint_vel": "joint_vel",
}


def canonical_wheelbipe_delay_profile(value: str) -> str:
    """Return a canonical delay-profile name or raise a contract error.

    A small set of spelling aliases is accepted for CLI ergonomics.  The
    canonical names are persisted in owner metadata so a run cannot be
    mistaken for a source/dynamics parity claim merely because a shorthand
    was supplied on the command line.
    """

    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "local": WHEELBIPE_DELAY_PROFILE_LOCAL,
        "default": WHEELBIPE_DELAY_PROFILE_LOCAL,
        "source": WHEELBIPE_DELAY_PROFILE_SOURCE_V14,
        "source_v14": WHEELBIPE_DELAY_PROFILE_SOURCE_V14,
    }
    profile = aliases.get(raw, raw)
    if profile not in WHEELBIPE_DELAY_PROFILES:
        choices = ", ".join(WHEELBIPE_DELAY_PROFILES)
        raise ValueError(f"unknown Wheelbipe delay_profile {value!r}; expected one of: {choices}")
    return profile


def canonical_wheelbipe_delay_range_semantics(value: str) -> str:
    """Normalize delay range endpoint semantics used by the sampler."""

    semantics = str(value).strip().lower().replace("-", "_")
    semantics = {
        "closed": "inclusive",
        "high_exclusive": "exclusive",
        "half_open": "exclusive",
        "left_closed": "exclusive",
    }.get(semantics, semantics)
    if semantics not in WHEELBIPE_DELAY_RANGE_SEMANTICS:
        choices = ", ".join(WHEELBIPE_DELAY_RANGE_SEMANTICS)
        raise ValueError(f"delay_range_semantics must be one of: {choices}, got {value!r}")
    return semantics


def _wheelbipe_delay_max_lag(bounds: DelayRange, *, inclusive: bool) -> int:
    """Return the largest sampled lag for an endpoint pair."""

    low, high = bounds
    # For an empty-width exclusive interval we keep the deterministic fixed
    # value behaviour of the public sampler.  This is useful for tests and
    # fixed-lag deployments while preserving high-exclusive behaviour for
    # ordinary ranges.
    return high if inclusive or low == high else high - 1


def build_wheelbipe_timing_contract(
    *,
    sim_dt: float,
    ctrl_dt: float,
    obs_delay_step_unit: str,
    use_obs_delay: bool,
    use_act_delay: bool,
    delay_range_semantics: str,
    delay_profile: str,
) -> dict[str, Any]:
    """Build and validate the explicit Wheelbipe timing/delay contract.

    The source V14 task uses a 5 ms physics step, four substeps per 20 ms
    control step, and high-exclusive integer delay ranges sampled in physics
    steps.  This helper validates exactly those *timing* assumptions when the
    source profile is requested.  It deliberately does not claim source
    dynamics, asset, controller-loop, or reward parity.
    """

    profile = canonical_wheelbipe_delay_profile(delay_profile)
    semantics = canonical_wheelbipe_delay_range_semantics(delay_range_semantics)
    unit = str(obs_delay_step_unit).strip().lower()
    if unit not in {"control", "physics"}:
        raise ValueError(f"obs_delay_step_unit must be 'control' or 'physics', got {unit!r}")
    try:
        sim = float(sim_dt)
        ctrl = float(ctrl_dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sim_dt and ctrl_dt must be finite positive numbers, got {sim_dt!r}, {ctrl_dt!r}"
        ) from exc
    if not math.isfinite(sim) or not math.isfinite(ctrl) or sim <= 0.0 or ctrl <= 0.0:
        raise ValueError(
            f"sim_dt and ctrl_dt must be finite positive numbers, got {sim_dt!r}, {ctrl_dt!r}"
        )
    ratio = ctrl / sim
    substeps = int(round(ratio))
    ratio_is_integer = abs(ratio - substeps) <= max(1.0e-9, abs(ratio) * 1.0e-7)
    if not ratio_is_integer or substeps < 1:
        raise ValueError(
            "ctrl_dt/sim_dt must be a positive integer for the Wheelbipe timing contract; "
            f"got sim_dt={sim_dt!r}, ctrl_dt={ctrl_dt!r}, ratio={ratio!r}"
        )

    source_timing_match = bool(
        abs(sim - 0.005) <= 1.0e-9 and abs(ctrl - 0.02) <= 1.0e-9 and substeps == 4
    )
    source_delay_sampling_match = bool(
        unit == "physics"
        and semantics == "exclusive"
        and bool(use_obs_delay)
        and bool(use_act_delay)
    )
    if profile == WHEELBIPE_DELAY_PROFILE_SOURCE_V14:
        missing: list[str] = []
        if not source_timing_match:
            missing.append("sim_dt=0.005, ctrl_dt=0.02, and four physics substeps")
        if not source_delay_sampling_match:
            missing.append(
                "physics-step observation/action delays enabled with high-exclusive ranges"
            )
        if missing:
            raise ValueError(
                "source_v14_physics delay profile contract mismatch: " + "; ".join(missing)
            )

    return {
        "profile": profile,
        "sim_dt": sim,
        "ctrl_dt": ctrl,
        "physics_hz": 1.0 / sim,
        "control_hz": 1.0 / ctrl,
        "substeps": substeps,
        "sim_substeps": substeps,
        "obs_delay_step_unit": unit,
        "delay_range_semantics": semantics,
        "use_obs_delay": bool(use_obs_delay),
        "use_act_delay": bool(use_act_delay),
        "source_timing_match": source_timing_match,
        "source_delay_sampling_match": source_delay_sampling_match,
        # This field is intentionally false even for the source timing
        # profile: dynamics/assets/ROS scheduling remain separate contracts.
        "full_source_parity": False,
        "source_parity_claim": False,
        "parity_note": "Timing and delay sampling only; no source dynamics or asset parity claim.",
    }


def wheelbipe_delay_profile_overrides(profile: str) -> dict[str, Any]:
    """Return a complete owner override for a CLI-selected delay profile.

    A profile name is a contract selector, rather than a label attached to
    whichever timing values happen to be present on the task's default
    dataclass.  In particular, applying ``local_physics`` to a compact custom
    owner must not leave that owner at the source-style 5 ms/physics-delay
    settings while reporting a local profile.  Keep every timing, unit,
    enablement, history and range field explicit here so the normal and
    custom sim2sim entrypoints have the same deterministic meaning.
    """

    canonical = canonical_wheelbipe_delay_profile(profile)
    if canonical == WHEELBIPE_DELAY_PROFILE_SOURCE_V14:
        return {
            "delay_profile": canonical,
            "delay_range_semantics": "exclusive",
            "sim_dt": 0.005,
            "ctrl_dt": 0.02,
            "obs_delay_step_unit": "physics",
            "use_obs_delay": True,
            "obs_history_len": 10,
            "obs_default_time_lag": 1,
            "obs_delay_cfg": {
                "root_ang_vel_b": [1, 4],
                "projected_gravity_b": [1, 4],
                "joint_pos": [1, 4],
                "joint_vel": [1, 4],
            },
            "use_act_delay": True,
            "act_history_len": 5,
            "act_delay_cfg": {
                "leg_actions": [1, 3],
                "wheel_actions": [1, 3],
            },
        }
    return {
        "delay_profile": canonical,
        "delay_range_semantics": "inclusive",
        "sim_dt": 0.001,
        "ctrl_dt": 0.02,
        "obs_delay_step_unit": "control",
        "use_obs_delay": False,
        "obs_history_len": 10,
        "obs_default_time_lag": 1,
        "obs_delay_cfg": {
            "root_ang_vel_b": [1, 4],
            "projected_gravity_b": [1, 4],
            "joint_pos": [1, 4],
            "joint_vel": [1, 4],
        },
        "use_act_delay": False,
        "act_history_len": 5,
        "act_delay_cfg": {"leg_actions": [1, 3], "wheel_actions": [1, 3]},
    }


class WheelbipeDelayBuffer:
    """Small preallocated NumPy ring buffer for per-environment delays.

    ``compute(value)`` writes the newest frame and returns the frame ``lag``
    steps old for each vectorized environment.  The returned array is a
    detached copy: the ring is mutated on the next call.  Resetting selected
    environments clears their complete history, matching Isaac Lab's
    ``DelayBuffer.reset(env_ids)`` semantics.

    The class intentionally has no simulator/backend dependency.  It is used
    by the owner env for both the callback's physics-step action path and the
    control-step observation path.
    """

    def __init__(self, history_len: int, num_envs: int, width: int, dtype: np.dtype):
        if isinstance(history_len, bool) or int(history_len) < 1:
            raise ValueError(f"history_len must be a positive integer, got {history_len!r}")
        if isinstance(num_envs, bool) or int(num_envs) < 1:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if isinstance(width, bool) or int(width) < 1:
            raise ValueError(f"width must be a positive integer, got {width!r}")
        self.history_len = int(history_len)
        self.num_envs = int(num_envs)
        self.width = int(width)
        self._capacity = self.history_len + 1
        self._history = np.zeros((self._capacity, self.num_envs, self.width), dtype=dtype)
        self._output = np.zeros((self.num_envs, self.width), dtype=dtype)
        self._env_indices = np.arange(self.num_envs, dtype=np.intp)
        self._lags = np.zeros((self.num_envs,), dtype=np.intp)
        # Isaac Lab's CircularBuffer treats the first appended sample as the
        # available history for a newly reset environment.  Track fill depth
        # per environment so a positive lag does not expose uninitialized
        # zeros during the startup window.
        self._valid_counts = np.zeros((self.num_envs,), dtype=np.intp)
        self._cursor = 0

    @property
    def lags(self) -> np.ndarray:
        """Current per-environment integer lags (read-only view)."""

        return self._lags

    def set_time_lag(self, lags: int | np.ndarray) -> None:
        values = np.asarray(lags, dtype=np.intp)
        if values.ndim == 0:
            values = np.full((self.num_envs,), int(values), dtype=np.intp)
        if values.shape != (self.num_envs,):
            raise ValueError(f"time lags must have shape ({self.num_envs},), got {values.shape}")
        if np.any(values < 0) or np.any(values > self.history_len):
            raise ValueError(
                f"time lags must be in [0, {self.history_len}], got "
                f"[{int(values.min(initial=0))}, {int(values.max(initial=0))}]"
            )
        self._lags[...] = values

    def reset(self, env_ids: np.ndarray | Sequence[int] | None = None) -> None:
        if env_ids is None:
            ids = self._env_indices
        else:
            ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
            if np.any(ids < 0) or np.any(ids >= self.num_envs):
                raise IndexError(f"env_ids out of range for {self.num_envs} environments: {ids}")
        if ids.size:
            self._history[:, ids, :] = 0.0
            self._output[ids] = 0.0
            self._valid_counts[ids] = 0

    def compute(self, values: np.ndarray) -> np.ndarray:
        sample = np.asarray(values, dtype=self._history.dtype)
        if sample.shape != (self.num_envs, self.width):
            raise ValueError(
                f"delay sample must have shape ({self.num_envs}, {self.width}), got {sample.shape}"
            )
        self._history[self._cursor, :, :] = sample
        available = np.minimum(self._valid_counts + 1, self._capacity)
        effective_lags = np.minimum(self._lags, available - 1)
        read_indices = (self._cursor - effective_lags) % self._capacity
        self._output[...] = self._history[read_indices, self._env_indices]
        self._valid_counts = available
        self._cursor = (self._cursor + 1) % self._capacity
        return self._output.copy()


def normalize_wheelbipe_delay_range(value: Sequence[int], *, name: str) -> DelayRange:
    """Validate and normalize one integer delay range.

    Endpoint semantics are supplied separately by the owner profile.  Keeping
    this validator agnostic lets the same serialized bounds represent either
    the local inclusive interval ``[low, high]`` or source-style
    high-exclusive ``[low, high)`` sampling.
    """

    values = np.asarray(value).reshape(-1)
    if values.shape != (2,):
        raise ValueError(f"{name} must contain exactly two integer bounds, got {values.shape}")
    raw_values = values.tolist()
    if any(isinstance(item, (bool, np.bool_)) for item in raw_values):
        raise ValueError(f"{name} bounds must be non-negative integers, got {value!r}")
    for item in raw_values:
        if isinstance(item, (float, np.floating)) and not float(item).is_integer():
            raise ValueError(f"{name} bounds must be integers, got {value!r}")
    try:
        low, high = (int(item) for item in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} bounds must be non-negative integers, got {value!r}") from exc
    if low < 0 or high < 0 or low > high:
        raise ValueError(f"{name} must satisfy 0 <= low <= high, got {(low, high)!r}")
    return low, high


def sample_wheelbipe_delay_lags(
    value: Sequence[int],
    num_envs: int,
    *,
    name: str,
    inclusive: bool = True,
) -> np.ndarray:
    """Sample integer lags for a vectorized environment.

    ``inclusive=True`` preserves the historical UniLab behaviour.  The source
    V14 profile passes ``inclusive=False`` because its ``torch.randint`` call
    treats the upper endpoint as exclusive.  A fixed ``(low, low)`` pair is
    deterministic in either mode, which is useful for explicit fixed-lag
    deployment tests.
    """

    if isinstance(num_envs, bool) or int(num_envs) < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
    low, high = normalize_wheelbipe_delay_range(value, name=name)
    if high == low:
        return np.full((int(num_envs),), low, dtype=np.intp)
    upper = high + 1 if bool(inclusive) else high
    return np.random.randint(low, upper, size=(int(num_envs),), dtype=np.intp)


@dataclass
class WheelbipeNoiseConfig(BaseNoiseConfig):
    """Noise magnitudes matching the ROS deployment parameters."""

    scale_joint_angle: float = 0.025
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.25
    scale_gravity: float = 0.05
    scale_wheel_vel: float = 1.0


@dataclass
class WheelbipeControlConfig(PdControlConfig):
    """Six-action policy control and eight-actuator low-level parameters."""

    action_scale: float = 0.5
    wheel_action_scale: float = 10.0
    Kp: float = 60.0  # noqa: N815 - kept as a stable Hydra key.
    Kd: float = 2.0  # noqa: N815 - kept as a stable Hydra key.
    wheel_Kd: float = 0.2  # noqa: N815 - kept as a stable Hydra key.
    # The V14 source owner uses a position-dependent spring (400--600 N over
    # 70 mm of compression) rather than the older constant-force fallback.
    spring_mode: str = "linear"
    spring_force: float = 240.0
    spring_random_force: tuple[float, float] = (-50.0, 50.0)
    spring_linear_up: float = 600.0
    spring_linear_down: float = 400.0
    spring_linear_length: float = 0.07
    # The Isaac V14 articulation keeps a fixed 50 N s/m spring-actuator
    # damping. ``spring_settings.damping=False`` in the source only disables
    # its additional randomized stretch/contract damping term.
    spring_damping: float = 50.0
    spring_offset: float = 0.06076
    leg_torque_limit: float = 40.0
    wheel_torque_limit: float = 5.0
    spring_torque_limit: float = 1000.0
    clip_actions: float = 1.0
    # Source Isaac applies no runner-side action clamp.  It constrains the
    # decoded physical targets instead: leg position to ±3.14 rad and wheel
    # velocity to ±100 rad/s.  Exact variant owners materialize these fields;
    # legacy/ROS owners may retain ``None`` and use their external contract.
    leg_position_target_limit: tuple[float, float] | None = None
    wheel_velocity_target_limit: float | None = None


@dataclass
class WheelbipeGimbalConfig:
    """Physical gimbal owner contract.

    The policy still emits six actions.  These settings describe the two
    additional low-level channels present in the source V14 asset: yaw is
    either velocity-controlled or heading-PD controlled, while pitch tracks a
    fixed position.  All values are validated before backend materialization.
    """

    # All published V14 training configs inherit the gimbal-capable Isaac
    # articulation while retaining the public six-action policy contract.
    enabled: bool = True
    control_mode: str = "velocity"
    pitch_target: float = -0.5
    yaw_velocity_range: tuple[float, float] = (-math.pi, math.pi)
    yaw_heading_range: tuple[float, float] = (-math.pi, math.pi)
    # Source v2 samples a world-frame heading target at reset. Its Play
    # subclass changes only this target owner to a fixed zero heading.
    heading_target_mode: str = "sampled"
    fixed_heading: float = 0.0
    yaw_kp: float = 20.0
    yaw_kd: float = 0.1
    velocity_kd: float = 0.5
    pitch_kp: float = 20.0
    pitch_kd: float = 0.5
    yaw_effort_limit: float = 2.0
    pitch_effort_limit: float = 10.0
    randomize_heading: bool = False

    def validate(self) -> None:
        mode = str(self.control_mode).strip().lower()
        if mode not in {"velocity", "heading_pd"}:
            raise ValueError(
                f"gimbal.control_mode must be 'velocity' or 'heading_pd', got {self.control_mode!r}"
            )
        target_mode = str(self.heading_target_mode).strip().lower()
        if target_mode not in {"sampled", "fixed"}:
            raise ValueError(
                "gimbal.heading_target_mode must be 'sampled' or 'fixed', got "
                f"{self.heading_target_mode!r}"
            )
        if not math.isfinite(float(self.fixed_heading)):
            raise ValueError(f"gimbal.fixed_heading must be finite, got {self.fixed_heading!r}")
        pitch_target = float(self.pitch_target)
        if not math.isfinite(pitch_target):
            raise ValueError(f"gimbal.pitch_target must be finite, got {pitch_target!r}")
        for name in (
            "yaw_kp",
            "yaw_kd",
            "velocity_kd",
            "pitch_kp",
            "pitch_kd",
            "yaw_effort_limit",
            "pitch_effort_limit",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"gimbal.{name} must be finite and non-negative, got {value!r}")
        for name in ("yaw_velocity_range", "yaw_heading_range"):
            raw = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if raw.shape != (2,) or np.any(~np.isfinite(raw)) or raw[0] > raw[1]:
                raise ValueError(f"gimbal.{name} must be an ordered finite pair, got {raw!r}")


def resolve_wheelbipe_torque_limits(
    control_config: WheelbipeControlConfig,
    actuator_ctrl_range: np.ndarray,
    *,
    native_leg_indices: Sequence[int] | np.ndarray,
    native_wheel_indices: Sequence[int] | np.ndarray,
    native_spring_indices: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Intersect owner torque limits with materialized actuator limits.

    Wheelbipe's public owner values (for example, 9.99 N m for a wheel) and
    the backend/MJCF actuator range are separate contracts.  Silently using
    only the former lets a backend clip torque after the owner controller has
    reported it, while using only the latter hides an owner safety limit.  The
    returned bounds are therefore the explicit intersection of both.  The
    helper is pure and intended for the environment's cold initialization
    path; no model metadata is consulted from the hot step loop.
    """

    ranges = np.asarray(actuator_ctrl_range, dtype=np.float64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(
            f"actuator_ctrl_range must have shape (num_actuators, 2), got {ranges.shape}"
        )
    if np.any(np.isnan(ranges)) or np.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError("actuator_ctrl_range must contain ordered non-NaN [low, high] bounds")
    num_actuators = int(ranges.shape[0])

    def _indices(value: Sequence[int] | np.ndarray, expected: int, name: str) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.size != expected:
            raise ValueError(f"{name} must contain {expected} actuator slots, got {raw.shape}")
        raw_items = raw.tolist()
        if raw.dtype.kind == "b" or any(isinstance(item, (bool, np.bool_)) for item in raw_items):
            raise ValueError(f"{name} must contain integer actuator slots, got {value!r}")
        if raw.dtype.kind not in "iu":
            # ``np.asarray([1.0])`` is not an unambiguous actuator slot and
            # should not be silently truncated.
            try:
                numeric = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must contain integer actuator slots, got {value!r}"
                ) from exc
            if raw.dtype.kind not in "f" and any(
                not isinstance(item, (int, float, np.integer, np.floating)) for item in raw_items
            ):
                raise ValueError(f"{name} must contain integer actuator slots, got {value!r}")
            if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
                raise ValueError(f"{name} must contain integer actuator slots, got {value!r}")
        slots = raw.astype(np.intp)
        if np.any(slots < 0) or np.any(slots >= num_actuators):
            raise ValueError(
                f"{name} contains slots outside [0, {num_actuators}), got {slots.tolist()}"
            )
        if np.unique(slots).size != slots.size:
            raise ValueError(f"{name} must not contain duplicate actuator slots")
        return slots

    groups = {
        "leg": _indices(native_leg_indices, NUM_LEG_ACTIONS, "native_leg_indices"),
        "wheel": _indices(native_wheel_indices, NUM_WHEEL_ACTIONS, "native_wheel_indices"),
        "spring": _indices(native_spring_indices, 2, "native_spring_indices"),
    }
    all_slots = np.concatenate(tuple(groups.values()))
    if np.unique(all_slots).size != all_slots.size:
        raise ValueError("native actuator groups must not overlap")

    configured = {
        "leg": float(getattr(control_config, "leg_torque_limit")),
        "wheel": float(getattr(control_config, "wheel_torque_limit")),
        "spring": float(getattr(control_config, "spring_torque_limit")),
    }
    for group, limit in configured.items():
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError(f"{group}_torque_limit must be finite and non-negative, got {limit!r}")

    lower = np.full((num_actuators,), -np.inf, dtype=np.float64)
    upper = np.full((num_actuators,), np.inf, dtype=np.float64)
    metadata_groups: dict[str, Any] = {}
    for group, slots in groups.items():
        requested_low = -configured[group]
        requested_high = configured[group]
        backend_group = ranges[slots]
        effective_group = np.column_stack(
            (
                np.maximum(backend_group[:, 0], requested_low),
                np.minimum(backend_group[:, 1], requested_high),
            )
        )
        if np.any(effective_group[:, 0] > effective_group[:, 1]):
            raise ValueError(
                f"{group} torque contract has no intersection between owner limit "
                f"±{configured[group]} and backend range {backend_group.tolist()}"
            )
        lower[slots] = effective_group[:, 0]
        upper[slots] = effective_group[:, 1]
        clipped = bool(
            np.any(effective_group[:, 0] > requested_low)
            or np.any(effective_group[:, 1] < requested_high)
        )
        metadata_groups[group] = {
            "requested_limit": configured[group],
            "backend_range": [[float(v) for v in row] for row in backend_group],
            "effective_range": [[float(v) for v in row] for row in effective_group],
            "backend_clips_owner": clipped,
            "slots": [int(slot) for slot in slots],
        }

    contract = {
        "groups": metadata_groups,
        "num_actuators": num_actuators,
        "note": "Effective controller bounds are owner/backend intersections; no source parity claim.",
    }
    return lower, upper, contract


@dataclass
class WheelbipeAsset:
    base_name: str = "base_link"
    ground: str = "floor"


@dataclass
class WheelbipeV14BaseCfg(LocomotionBaseCfg):
    noise_config: WheelbipeNoiseConfig = field(default_factory=WheelbipeNoiseConfig)  # type: ignore[assignment]
    control_config: WheelbipeControlConfig = field(default_factory=WheelbipeControlConfig)  # type: ignore[assignment]
    gimbal: WheelbipeGimbalConfig = field(default_factory=WheelbipeGimbalConfig)
    # Flat-v1/Rough-v1 enable the owner state machine.  Keeping the nested
    # config explicit lets Hydra compose the transition envelope without
    # leaking Isaac-specific manager objects into the backend.
    state_machine: WheelbipeStateMachineConfig = field(default_factory=WheelbipeStateMachineConfig)
    asset: WheelbipeAsset = field(default_factory=WheelbipeAsset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 20.0
    # Explicitly recorded in the owner YAML so sim2sim audits can verify the
    # reset/sampling contract even though the normal task uses fixed starts.
    sampling_mode: str = "start"
    # MotrixSim's current constraint solver can fail on the six closed-loop
    # MuJoCo ``connect`` constraints in this mechanism.  Keep the compatibility
    # choice explicit in the owner config; MuJoCo always retains the canonical
    # constraints and the policy I/O contract is unchanged.
    motrix_disable_equality: bool = True
    # Source V14 sensor/action latency.  Exact source owners enable these
    # physics-step buffers; the explicit ``local_physics`` profile is the
    # delay-free UniLab alternative.  Neither choice changes the 35->6 policy
    # shape, and timing alignment alone does not imply full dynamics parity.
    obs_delay_cfg: dict[str, DelayRange] = field(
        default_factory=lambda: {
            "root_ang_vel_b": (1, 4),
            "projected_gravity_b": (1, 4),
            "joint_pos": (1, 4),
            "joint_vel": (1, 4),
        }
    )
    obs_history_len: int = 10
    obs_default_time_lag: int = 1
    use_obs_delay: bool = True
    # ``control`` is available for generic callers; ``physics`` matches the
    # upstream V14 DelayBuffer placement and samples on every backend substep
    # through the owner callback.
    obs_delay_step_unit: str = "physics"
    # ``local_physics`` records the delay-free UniLab semantics and uses
    # inclusive ranges.  Exact source owners default to the fail-closed
    # ``source_v14_physics`` timing profile; it does not imply source dynamics
    # or asset parity.
    delay_profile: str = WHEELBIPE_DELAY_PROFILE_SOURCE_V14
    delay_range_semantics: str = "exclusive"
    act_delay_cfg: dict[str, DelayRange] = field(
        default_factory=lambda: {
            "leg_actions": (1, 3),
            "wheel_actions": (1, 3),
        }
    )
    act_history_len: int = 5
    use_act_delay: bool = True
    # The normal policy has a fixed seven-element mode tail.  Owners with a
    # state machine may replace the per-env vector in ``state.info`` while
    # retaining the same dimensional contract.
    ctrl_mode_obs_enabled: bool = True
    ctrl_mode_obs_dim: int = 7
    # ``normal`` is the ROS-compatible 35D frame; ``compact`` is the source
    # history-algorithm frame used by the named DreamWaQ/HIM/NP3O owners.
    # This field lives on the base config so Hydra overrides are validated at
    # the owner boundary before a backend is materialized.
    policy_observation_mode: str = "normal"

    def validate(self) -> None:
        """Validate latency and policy-tail contracts before backend creation."""

        super().validate()
        try:
            clip_actions = float(self.control_config.clip_actions)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "control_config.clip_actions must be positive or +inf, got "
                f"{self.control_config.clip_actions!r}"
            ) from exc
        if clip_actions <= 0.0 or (not np.isfinite(clip_actions) and not np.isposinf(clip_actions)):
            raise ValueError(
                "control_config.clip_actions must be positive or +inf, got "
                f"{self.control_config.clip_actions!r}"
            )
        if not isinstance(self.gimbal, WheelbipeGimbalConfig):
            raise ValueError("gimbal must be a WheelbipeGimbalConfig instance")
        self.gimbal.validate()
        if not isinstance(self.state_machine, WheelbipeStateMachineConfig):
            raise ValueError("state_machine must be a WheelbipeStateMachineConfig instance")
        self.state_machine.validate()
        observation_mode = str(getattr(self, "policy_observation_mode", "normal")).lower()
        if observation_mode == "normal":
            if not bool(self.ctrl_mode_obs_enabled) or int(self.ctrl_mode_obs_dim) != 7:
                raise ValueError(
                    "normal Wheelbipe V14 owners require "
                    "ctrl_mode_obs_enabled=true and ctrl_mode_obs_dim=7 "
                    "for the 35D policy contract"
                )
        elif observation_mode == "compact":
            if bool(self.ctrl_mode_obs_enabled) or int(self.ctrl_mode_obs_dim) != 0:
                raise ValueError(
                    "compact Wheelbipe V14 owners require "
                    "ctrl_mode_obs_enabled=false and ctrl_mode_obs_dim=0 "
                    "for the 28D history-policy contract"
                )
        else:
            raise ValueError(
                f"policy_observation_mode must be 'normal' or 'compact', got {observation_mode!r}"
            )
        unit = str(self.obs_delay_step_unit).lower()
        if unit not in {"control", "physics"}:
            raise ValueError(f"obs_delay_step_unit must be 'control' or 'physics', got {unit!r}")
        semantics = canonical_wheelbipe_delay_range_semantics(self.delay_range_semantics)
        # Build once during config validation so malformed timing cannot make
        # it as far as backend materialization.  The returned snapshot is
        # recomputed/cached by the env on construction.
        build_wheelbipe_timing_contract(
            sim_dt=self.sim_dt,
            ctrl_dt=self.ctrl_dt,
            obs_delay_step_unit=unit,
            use_obs_delay=bool(self.use_obs_delay),
            use_act_delay=bool(self.use_act_delay),
            delay_range_semantics=semantics,
            delay_profile=self.delay_profile,
        )
        if isinstance(self.obs_history_len, bool) or int(self.obs_history_len) < 1:
            raise ValueError(f"obs_history_len must be positive, got {self.obs_history_len!r}")
        if isinstance(self.act_history_len, bool) or int(self.act_history_len) < 1:
            raise ValueError(f"act_history_len must be positive, got {self.act_history_len!r}")
        if int(self.obs_default_time_lag) < 0 or int(self.obs_default_time_lag) > int(
            self.obs_history_len
        ):
            raise ValueError(
                f"obs_default_time_lag must be in [0, {self.obs_history_len}], "
                f"got {self.obs_default_time_lag!r}"
            )
        obs_allowed = set(WHEELBIPE_OBS_DELAY_ALIASES)
        if bool(self.use_obs_delay):
            if not isinstance(self.obs_delay_cfg, dict) or not self.obs_delay_cfg:
                raise ValueError("use_obs_delay=true requires a non-empty obs_delay_cfg")
            seen: set[str] = set()
            for group, raw_range in self.obs_delay_cfg.items():
                group_name = str(group)
                if group_name not in obs_allowed:
                    raise ValueError(f"unsupported observation delay group {group_name!r}")
                canonical = WHEELBIPE_OBS_DELAY_ALIASES[group_name]
                if canonical in seen:
                    raise ValueError(f"duplicate observation delay group for {canonical!r}")
                seen.add(canonical)
                bounds = normalize_wheelbipe_delay_range(
                    raw_range, name=f"obs_delay_cfg.{group_name}"
                )
                max_lag = _wheelbipe_delay_max_lag(bounds, inclusive=semantics == "inclusive")
                if max_lag > int(self.obs_history_len):
                    raise ValueError(
                        f"obs_delay_cfg.{group_name} max lag {max_lag} exceeds "
                        f"obs_history_len {self.obs_history_len}"
                    )
        if bool(self.use_act_delay):
            if bool(self.control_config.simulate_action_latency):
                raise ValueError(
                    "use_act_delay and control_config.simulate_action_latency are mutually exclusive"
                )
            if not isinstance(self.act_delay_cfg, dict) or not self.act_delay_cfg:
                raise ValueError("use_act_delay=true requires a non-empty act_delay_cfg")
            for group, raw_range in self.act_delay_cfg.items():
                group_name = str(group)
                if group_name not in {"leg_actions", "wheel_actions"}:
                    raise ValueError(f"unsupported action delay group {group_name!r}")
                bounds = normalize_wheelbipe_delay_range(
                    raw_range, name=f"act_delay_cfg.{group_name}"
                )
                max_lag = _wheelbipe_delay_max_lag(bounds, inclusive=semantics == "inclusive")
                if max_lag > int(self.act_history_len):
                    raise ValueError(
                        f"act_delay_cfg.{group_name} max lag {max_lag} exceeds "
                        f"act_history_len {self.act_history_len}"
                    )

    # NP3O constraint channels.  They are opt-in so normal PPO/HIM/DreamWaQ
    # runs preserve the original observation/reward contract.
    num_costs: int = 0
    np3o_tilt_limit_deg: float = 15.0
    np3o_body_height_min: float = 0.18
    np3o_body_height_max: float = 0.42
    np3o_ang_vel_xy_limit: float = 4.0
    np3o_torque_limit: float = 30.0
    np3o_joint_velocity_limit: float = 80.0
    np3o_cost_clip: float = 100.0


def _as_batch(values: np.ndarray, width: int, *, name: str, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}), got {arr.shape}")
    return arr


def build_wheelbipe_policy_observation(
    commands: np.ndarray,
    height_command: np.ndarray,
    gyro: np.ndarray,
    projected_gravity: np.ndarray,
    leg_position: np.ndarray,
    leg_velocity: np.ndarray,
    wheel_velocity: np.ndarray,
    previous_actions: np.ndarray,
    *,
    noise_fn: Any | None = None,
    noise_config: WheelbipeNoiseConfig | None = None,
    control_mode: np.ndarray | None = None,
    control_mode_scale: np.ndarray | None = None,
    source_training_clips: bool = False,
) -> np.ndarray:
    """Build the 35D normal-mode observation used by WheelBipe policies.

    Layout (and scales) are intentionally kept in one pure function so the
    training env and the ONNX sim2sim runner cannot silently drift apart::

        command[3], height*5, gyro*0.5, gravity[3], leg_pos[4], wheel_pos[2],
        leg_vel*0.1[4], wheel_vel*0.1[2], previous_action[6], mode[7].

    By default the ROS ``scaleClamp`` deployment contract applies each scale
    first and then clamps the resulting field to ``[-100, 100]``.  Exact
    source task owners pass ``source_training_clips=True`` to reproduce the
    pinned Isaac training order instead: configured raw components are clipped
    first (including height to ``[0, 1]`` and joint velocity to ``[-200,
    200]``), then scaled.  The extra ``ctrl_mode_obs`` block has a scale entry
    but no clip entry in the pinned config and therefore remains unclipped.
    Keeping the switch explicit prevents the training
    checkpoint contract and the independently published ROS safety wrapper
    from silently overwriting one another.
    """

    dtype = get_global_dtype()
    commands = _as_batch(commands, 3, name="commands", dtype=dtype)
    height = np.asarray(height_command, dtype=dtype).reshape(-1, 1)
    gyro = _as_batch(gyro, 3, name="gyro", dtype=dtype)
    gravity = _as_batch(projected_gravity, 3, name="projected_gravity", dtype=dtype)
    leg_position = _as_batch(leg_position, 4, name="leg_position", dtype=dtype)
    leg_velocity = _as_batch(leg_velocity, 4, name="leg_velocity", dtype=dtype)
    wheel_velocity = _as_batch(wheel_velocity, 2, name="wheel_velocity", dtype=dtype)
    previous_actions = _as_batch(previous_actions, 6, name="previous_actions", dtype=dtype)
    n = commands.shape[0]
    arrays = (height, gyro, gravity, leg_position, leg_velocity, wheel_velocity, previous_actions)
    if any(value.shape[0] != n for value in arrays):
        raise ValueError("all Wheelbipe observation components must have the same batch size")

    if control_mode is None:
        mode = np.broadcast_to(NORMAL_CONTROL_MODE.astype(dtype), (n, 7))
    else:
        mode = _as_batch(control_mode, 7, name="control_mode", dtype=dtype)
        if mode.shape[0] != n:
            raise ValueError("all Wheelbipe observation components must have the same batch size")
    if control_mode_scale is not None:
        mode_scale = np.asarray(control_mode_scale, dtype=dtype).reshape(-1)
        if mode_scale.shape != (7,) or not np.all(np.isfinite(mode_scale)):
            raise ValueError("control_mode_scale must contain seven finite values")
    else:
        mode_scale = np.ones((7,), dtype=dtype)

    cfg = noise_config or WheelbipeNoiseConfig()
    if noise_fn is not None:
        gyro = noise_fn(gyro, cfg.scale_gyro)
        gravity = noise_fn(gravity, cfg.scale_gravity)
        leg_position = noise_fn(leg_position, cfg.scale_joint_angle)
        leg_velocity = noise_fn(leg_velocity, cfg.scale_joint_vel)
        wheel_velocity = noise_fn(wheel_velocity, cfg.scale_wheel_vel)

    if source_training_clips:
        # Pinned ``V14_BASIC_OBS_CLIP`` values.  The source clips each raw
        # tensor in ``_get_observations`` before applying
        # ``V14_BASIC_OBS_SCALE``; these limits are deliberately component
        # specific rather than a post-concatenation global bound.
        commands = np.clip(commands, -100.0, 100.0)
        height = np.clip(height, 0.0, 1.0)
        gyro = np.clip(gyro, -100.0, 100.0)
        gravity = np.clip(gravity, -100.0, 100.0)
        leg_position = np.clip(leg_position, -100.0, 100.0)
        leg_velocity = np.clip(leg_velocity, -200.0, 200.0)
        wheel_velocity = np.clip(wheel_velocity, -200.0, 200.0)
        previous_actions = np.clip(previous_actions, -100.0, 100.0)
    mode = mode * mode_scale[None, :]

    wheel_position = np.zeros((n, NUM_WHEEL_ACTIONS), dtype=dtype)
    obs = np.concatenate(
        (
            commands,
            height * 5.0,
            gyro * 0.5,
            gravity,
            leg_position,
            wheel_position,
            leg_velocity * 0.1,
            wheel_velocity * 0.1,
            previous_actions,
            mode,
        ),
        axis=1,
    )
    if obs.shape != (n, POLICY_OBS_DIM):
        raise RuntimeError(f"Wheelbipe policy observation contract produced {obs.shape}")
    # Only the independent ROS deployment wrapper adds a final global clamp.
    # The source training owner has already applied its component bounds and
    # must preserve post-clip scales (for example ctrl-mode slot five ×5).
    if not source_training_clips:
        np.clip(obs, -POLICY_OBS_CLIP, POLICY_OBS_CLIP, out=obs)
    else:
        # Pinned ``wheelbipe25_v3._get_observations`` sanitizes the fully
        # assembled actor tensor after every component-specific clip/scale.
        # Keep this source-only: the generic/ROS helper owns its independent
        # scaleClamp contract, while raw policy actions remain untouched until
        # the environment's numerical-safety boundary evaluates them.
        np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return np.asarray(obs, dtype=dtype)


def map_policy_action_to_native_targets(
    actions: np.ndarray,
    *,
    action_scale: float,
    wheel_action_scale: float,
    default_leg_position: Sequence[float] | np.ndarray,
    native_leg_indices: np.ndarray,
    native_wheel_indices: np.ndarray,
    native_spring_indices: np.ndarray,
    spring_target: float = 0.0,
    num_native_actuators: int = NUM_NATIVE_ACTUATORS,
    leg_position_limit: tuple[float, float] | None = None,
    wheel_velocity_limit: float | None = None,
) -> np.ndarray:
    """Map six public actions to native actuator target slots.

    ``num_native_actuators`` is normally eight; gimbal owners pass ten and
    fill the two extra targets in their low-level controller.  Keeping the
    policy mapper agnostic preserves the six-action deployment contract.
    """

    dtype = get_global_dtype()
    actions = _as_batch(actions, NUM_POLICY_ACTIONS, name="actions", dtype=dtype)
    leg_indices = np.asarray(native_leg_indices, dtype=np.intp).reshape(NUM_LEG_ACTIONS)
    wheel_indices = np.asarray(native_wheel_indices, dtype=np.intp).reshape(NUM_WHEEL_ACTIONS)
    spring_indices = np.asarray(native_spring_indices, dtype=np.intp).reshape(2)
    default = np.asarray(default_leg_position, dtype=dtype).reshape(NUM_LEG_ACTIONS)
    count = int(num_native_actuators)
    if count < NUM_NATIVE_ACTUATORS:
        raise ValueError(
            f"num_native_actuators must be at least {NUM_NATIVE_ACTUATORS}, got {count}"
        )
    targets = np.zeros((actions.shape[0], count), dtype=dtype)
    targets[:, leg_indices] = actions[:, :NUM_LEG_ACTIONS] * float(action_scale) + default
    targets[:, wheel_indices] = actions[:, NUM_LEG_ACTIONS:] * float(wheel_action_scale)
    if leg_position_limit is not None:
        leg_low, leg_high = (float(value) for value in leg_position_limit)
        if not math.isfinite(leg_low) or not math.isfinite(leg_high) or leg_high < leg_low:
            raise ValueError(
                f"leg_position_limit must be an ordered finite pair, got {leg_position_limit!r}"
            )
        targets[:, leg_indices] = np.clip(targets[:, leg_indices], leg_low, leg_high)
    if wheel_velocity_limit is not None:
        wheel_limit = float(wheel_velocity_limit)
        if not math.isfinite(wheel_limit) or wheel_limit < 0.0:
            raise ValueError(
                "wheel_velocity_limit must be finite and non-negative, "
                f"got {wheel_velocity_limit!r}"
            )
        targets[:, wheel_indices] = np.clip(targets[:, wheel_indices], -wheel_limit, wheel_limit)
    targets[:, spring_indices] = float(spring_target)
    return targets


def compute_wheelbipe_motor_ctrl(
    native_targets: np.ndarray,
    full_dof_pos: np.ndarray,
    full_dof_vel: np.ndarray,
    *,
    leg_pos_indices: np.ndarray,
    leg_vel_indices: np.ndarray,
    wheel_vel_indices: np.ndarray,
    spring_pos_indices: np.ndarray,
    spring_vel_indices: np.ndarray,
    native_leg_indices: np.ndarray,
    native_wheel_indices: np.ndarray,
    native_spring_indices: np.ndarray,
    leg_kp: np.ndarray,
    leg_kd: np.ndarray,
    wheel_kd: np.ndarray,
    spring_force: float,
    spring_damping: float,
    lower: np.ndarray,
    upper: np.ndarray,
    out: np.ndarray,
    spring_mode: str = "constant",
    spring_offset: float = 0.06076,
    spring_linear_up: float = 600.0,
    spring_linear_down: float = 400.0,
    spring_linear_length: float = 0.07,
    spring_force_random: np.ndarray | None = None,
) -> np.ndarray:
    """Convert native position/velocity targets to bounded motor torques.

    All indices are resolved on the cold init path.  The function itself is
    allocation-light and contains no backend feature probing, which keeps the
    hot step path within the SimBackend contract.
    """

    dtype = out.dtype
    native_targets = np.asarray(native_targets, dtype=dtype)
    leg_pos = np.asarray(full_dof_pos[:, np.asarray(leg_pos_indices, dtype=np.intp)], dtype=dtype)
    leg_vel = np.asarray(full_dof_vel[:, np.asarray(leg_vel_indices, dtype=np.intp)], dtype=dtype)
    wheel_vel = np.asarray(
        full_dof_vel[:, np.asarray(wheel_vel_indices, dtype=np.intp)], dtype=dtype
    )
    spring_pos = np.asarray(
        full_dof_pos[:, np.asarray(spring_pos_indices, dtype=np.intp)], dtype=dtype
    )
    spring_vel = np.asarray(
        full_dof_vel[:, np.asarray(spring_vel_indices, dtype=np.intp)], dtype=dtype
    )

    out.fill(0.0)
    leg_slots = np.asarray(native_leg_indices, dtype=np.intp)
    wheel_slots = np.asarray(native_wheel_indices, dtype=np.intp)
    spring_slots = np.asarray(native_spring_indices, dtype=np.intp)
    out[:, leg_slots] = (
        np.asarray(native_targets[:, leg_slots], dtype=dtype) - leg_pos
    ) * np.asarray(leg_kp, dtype=dtype) - np.asarray(leg_kd, dtype=dtype) * leg_vel
    out[:, wheel_slots] = (
        np.asarray(native_targets[:, wheel_slots], dtype=dtype) - wheel_vel
    ) * np.asarray(wheel_kd, dtype=dtype)
    if spring_mode == "linear":
        length = np.maximum(float(spring_offset) - spring_pos, 0.0)
        if float(spring_linear_length) <= 0.0:
            raise ValueError("spring_linear_length must be positive in linear mode")
        spring_base = float(spring_linear_down) + (
            (float(spring_linear_up) - float(spring_linear_down))
            / float(spring_linear_length)
            * length
        )
    elif spring_mode == "constant":
        spring_base = np.full_like(spring_pos, float(spring_force), dtype=dtype)
    else:
        raise ValueError(f"unsupported spring_mode {spring_mode!r}")
    if spring_force_random is not None:
        spring_base = spring_base + np.asarray(spring_force_random, dtype=dtype)
    out[:, spring_slots] = spring_base - float(spring_damping) * spring_vel
    np.clip(out, np.asarray(lower, dtype=dtype), np.asarray(upper, dtype=dtype), out=out)
    return out


class WheelbipeV14BaseEnv(LocomotionBaseEnv):
    """Shared six-action owner env for flat and generated-rough tasks."""

    _cfg: WheelbipeV14BaseCfg

    def _init_action_space(self) -> None:
        # The pinned RSL-RL owner declares ``clip_actions: null`` and accepts
        # finite raw Gaussian actions; only decoded joint targets are bounded.
        # Reflect that contract in Gym's public space so wrappers do not
        # silently squash an action that this environment intentionally keeps
        # in its history/reward path.  Legacy owners with a finite clip retain
        # their bounded Box.
        action_bound = float(self._cfg.control_config.clip_actions)
        self._action_space = gym.spaces.Box(
            low=-action_bound,
            high=action_bound,
            shape=(NUM_POLICY_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        dtype = get_global_dtype()
        self.default_angles = np.zeros((NUM_POLICY_ACTIONS,), dtype=dtype)
        gimbal_cfg = getattr(self._cfg, "gimbal", None)
        self._gimbal_enabled = bool(getattr(gimbal_cfg, "enabled", False))
        expected_actuator_names = (
            NATIVE_ACTUATOR_NAMES_WITH_GIMBAL if self._gimbal_enabled else NATIVE_ACTUATOR_NAMES
        )
        self._num_native_actuators = len(expected_actuator_names)
        # These are stable backend contract lookups, intentionally performed
        # once while the model is materialized.
        self._policy_pos_indices = np.asarray(
            self._backend.get_joint_dof_pos_indices(POLICY_JOINT_NAMES), dtype=np.intp
        )
        self._policy_vel_indices = np.asarray(
            self._backend.get_joint_dof_vel_indices(POLICY_JOINT_NAMES), dtype=np.intp
        )
        self._leg_pos_indices = self._policy_pos_indices[:NUM_LEG_ACTIONS]
        self._leg_vel_indices = self._policy_vel_indices[:NUM_LEG_ACTIONS]
        self._wheel_pos_indices = self._policy_pos_indices[NUM_LEG_ACTIONS:]
        self._wheel_vel_indices = self._policy_vel_indices[NUM_LEG_ACTIONS:]
        self._spring_pos_indices = np.asarray(
            self._backend.get_joint_dof_pos_indices(SPRING_JOINT_NAMES), dtype=np.intp
        )
        self._spring_vel_indices = np.asarray(
            self._backend.get_joint_dof_vel_indices(SPRING_JOINT_NAMES), dtype=np.intp
        )

        try:
            actuator_names = tuple(self._backend.get_actuator_names())
        except NotImplementedError:
            # Motrix 0.8 exposes the actuator vector but not names.  The
            # vendored MJCF's actuator order is a frozen asset contract, so a
            # cold-path fallback is safe and keeps the hot path backend-neutral.
            actuator_names = expected_actuator_names
        missing = [name for name in expected_actuator_names if name not in actuator_names]
        if missing:
            raise ValueError(
                "Wheelbipe V14 actuator contract is missing names: " + ", ".join(missing)
            )
        slots = np.asarray(
            [actuator_names.index(name) for name in expected_actuator_names], dtype=np.intp
        )
        self._native_leg_indices = slots[[0, 1, 4, 5]]
        self._native_wheel_indices = slots[[2, 6]]
        self._native_spring_indices = slots[[3, 7]]
        if self._gimbal_enabled:
            self._native_gimbal_indices = slots[8:10]
            self._gimbal_pos_indices = np.asarray(
                self._backend.get_joint_dof_pos_indices(GIMBAL_JOINT_NAMES), dtype=np.intp
            )
            self._gimbal_vel_indices = np.asarray(
                self._backend.get_joint_dof_vel_indices(GIMBAL_JOINT_NAMES), dtype=np.intp
            )
        else:
            self._native_gimbal_indices = np.zeros((0,), dtype=np.intp)
            self._gimbal_pos_indices = np.zeros((0,), dtype=np.intp)
            self._gimbal_vel_indices = np.zeros((0,), dtype=np.intp)
        if self._backend.num_actuators != self._num_native_actuators:
            raise ValueError(
                f"Wheelbipe V14 requires {self._num_native_actuators} native actuators "
                f"for gimbal_enabled={self._gimbal_enabled}, got {self._backend.num_actuators}"
            )

    def get_dof_pos(self) -> np.ndarray:
        full = np.asarray(self._backend.get_dof_pos(), dtype=get_global_dtype())
        return np.asarray(full[:, self._policy_pos_indices], dtype=get_global_dtype())

    def get_dof_vel(self) -> np.ndarray:
        full = np.asarray(self._backend.get_dof_vel(), dtype=get_global_dtype())
        return np.asarray(full[:, self._policy_vel_indices], dtype=get_global_dtype())

    def get_dof_acc(self) -> np.ndarray:
        full = np.asarray(self._backend.get_dof_acc(), dtype=get_global_dtype())
        return np.asarray(full[:, self._policy_vel_indices], dtype=get_global_dtype())

    def get_full_dof_pos(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_pos(), dtype=get_global_dtype())

    def get_full_dof_vel(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_vel(), dtype=get_global_dtype())

    def get_full_dof_acc(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_acc(), dtype=get_global_dtype())

    def get_local_linvel(self) -> np.ndarray:
        quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        world_vel = np.asarray(self._backend.get_base_lin_vel(), dtype=get_global_dtype())
        return np.asarray(np_quat_apply_inverse(quat, world_vel), dtype=get_global_dtype())

    def get_projected_gravity(self) -> np.ndarray:
        quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
        world_gravity = np.broadcast_to(
            np.asarray([0.0, 0.0, -1.0], dtype=get_global_dtype()),
            (self._num_envs, 3),
        )
        return np.asarray(np_quat_apply_inverse(quat, world_gravity), dtype=get_global_dtype())
