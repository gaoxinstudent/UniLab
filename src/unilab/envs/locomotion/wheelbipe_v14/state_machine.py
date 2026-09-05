"""Backend-neutral WheelBipe V14 task state machines.

The pinned SCUT owner composes Airborne, JumpTakeoff, StepUp and Stair as
orthogonal machines.  This module preserves that composition.  The public
``state`` is only a priority-resolved diagnostic marker; all component flags
remain available in the transition result.

Every hot-path input crosses :class:`WheelbipeStateMachineSensors`.  Contact
force magnitudes come from the public ``SimBackend`` contract and terrain
probes come from the cached terrain sampler.  Geometry contact is available
only when a caller selects it explicitly in the sensor frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from unilab.base.backend.base import SimBackend
from unilab.envs.locomotion.common.terrain_spawn import (
    BaseSpawnManager,
    TerrainSpawnManager,
)
from unilab.utils.geometry import np_roll_pitch_from_quat
from unilab.utils.rotation import np_yaw_from_quat

from ._random import GLOBAL_NUMPY_RANDOM
from .semantics import (
    SOURCE_V14_RESET_CONTACT_BODY_NAMES,
    SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES,
    SOURCE_V14_WHEEL_BODY_NAMES,
)

if TYPE_CHECKING:
    from .joystick import WheelbipeRewardConfig, WheelbipeV14FlatCfg

_AIRBORNE_BASE_SCALE_OVERRIDE_TERMS = frozenset(
    {
        "action_rate",
        "action_smoothness_leg",
        "action_smoothness_wheel",
        "joint_torque",
        "leg_joint_acc",
        "leg_joint_vel",
        "track_lin_vel_xy",
        "wheel_acc",
        "wheel_power",
        "wheel_vel",
    }
)
_AIRBORNE_SCALE_MULTIPLIER_TERMS = frozenset(
    {
        "flat_orientation_y_v",
        "foot_bound_square",
        "termination",
        "track_height_square",
        "undesired_contact",
    }
)


class WheelbipeMotionState(IntEnum):
    """Priority-resolved marker/diagnostic state."""

    NORMAL = 0
    AIRBORNE = 1
    LANDING = 2
    RECOVER = 3
    JUMP_PUSH = 4
    JUMP_TUCK = 5
    STEP_UP = 6
    STAIR = 7
    WALL_BLOCKED = 8


class WheelbipeJumpPhase(IntEnum):
    """Pinned JumpTakeoff phases (there is no implemented preload phase)."""

    IDLE = 0
    PUSH = 1
    TUCK = 2


class WheelbipeContactSource(IntEnum):
    """Explicit contact decision source for a sensor frame."""

    GEOMETRY = 0
    FORCE = 1


def _non_negative(name: str, value: float) -> None:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


def _probability(name: str, value: float) -> None:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _two_finite(name: str, value: tuple[float, float]) -> None:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain two finite values, got {value!r}")


def _range_spec(
    name: str,
    value: tuple[tuple[float, float], ...] | None,
) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if not value:
        raise ValueError(f"{name} cannot be empty")
    normalized: list[tuple[float, float]] = []
    for segment in value:
        _two_finite(name, segment)
        low, high = (float(entry) for entry in segment)
        normalized.append((min(low, high), max(low, high)))
    return tuple(normalized)


@dataclass
class WheelbipeJumpTakeoffConfig:
    """Ballistic PUSH/TUCK reference owner.

    The pinned machine deliberately preserves ordinary XY/yaw and height
    commands.  External velocity/force assist is outside this sensor/state
    owner because it requires a separate simulator mutation contract.
    """

    enabled: bool = False
    trigger_mode: str = "flag"
    probability_per_step: float = 0.0
    min_episode_time_s: float = 0.0
    cooldown_s: float = 1.0
    peak_height_range: tuple[float, float] = (0.45, 0.55)
    push_start_height: float = 0.20
    release_height: float = 0.38
    gravity: float = 9.81
    exit_time_scale_after_peak: float = 1.3
    min_duration_s: float = 0.20
    tuck_timing_mode: str = "dynamic_push_time"
    fixed_tuck_time_s: float = 0.14
    tuck_start_height: float = 0.38
    tuck_start_height_ratio: float | None = None
    tuck_start_time_s: float | None = None
    tuck_start_time_offset_s: float = 0.05
    tuck_wheel_air_margin: float = 0.03
    enter_airborne_on_exit: bool = True

    def validate(self) -> None:
        if self.trigger_mode not in {"flag", "manual", "random"}:
            raise ValueError(
                "state_machine.jump_takeoff.trigger_mode must be flag, manual, or random"
            )
        _probability("state_machine.jump_takeoff.probability_per_step", self.probability_per_step)
        _two_finite("state_machine.jump_takeoff.peak_height_range", self.peak_height_range)
        for name in (
            "min_episode_time_s",
            "cooldown_s",
            "push_start_height",
            "release_height",
            "gravity",
            "exit_time_scale_after_peak",
            "min_duration_s",
            "fixed_tuck_time_s",
            "tuck_start_height",
            "tuck_start_time_offset_s",
            "tuck_wheel_air_margin",
        ):
            _non_negative(f"state_machine.jump_takeoff.{name}", float(getattr(self, name)))
        if float(self.gravity) <= 0.0:
            raise ValueError("state_machine.jump_takeoff.gravity must be positive")
        if float(self.exit_time_scale_after_peak) < 1.0:
            raise ValueError("state_machine.jump_takeoff.exit_time_scale_after_peak must be >= 1")
        if self.tuck_timing_mode not in {"dynamic_push_time", "fixed_tuck_time"}:
            raise ValueError(
                "state_machine.jump_takeoff.tuck_timing_mode must be "
                "dynamic_push_time or fixed_tuck_time"
            )
        if self.tuck_start_height_ratio is not None:
            _non_negative(
                "state_machine.jump_takeoff.tuck_start_height_ratio",
                self.tuck_start_height_ratio,
            )
        if self.tuck_start_time_s is not None:
            _non_negative("state_machine.jump_takeoff.tuck_start_time_s", self.tuck_start_time_s)


@dataclass
class WheelbipeStepUpConfig:
    """Temporal wheel-forward step/wall detector and height hold."""

    enabled: bool = False
    forward_offset: float = 0.50
    step_height_min: float = 0.12
    step_height_max: float = 0.14
    wall_height: float = 0.14
    height_command_bias: float = 0.16
    hold_s: float = 2.0
    height_command_max: float | None = 0.40

    def validate(self) -> None:
        for name in (
            "forward_offset",
            "step_height_min",
            "step_height_max",
            "wall_height",
            "height_command_bias",
            "hold_s",
        ):
            _non_negative(f"state_machine.step_up.{name}", getattr(self, name))
        if float(self.step_height_min) >= float(self.step_height_max):
            raise ValueError("state_machine.step_up step_height_min must be < step_height_max")
        if float(self.wall_height) < float(self.step_height_max):
            raise ValueError("state_machine.step_up wall_height must be >= step_height_max")
        if self.height_command_max is not None:
            _non_negative("state_machine.step_up.height_command_max", self.height_command_max)


@dataclass
class WheelbipeStairConfig:
    """Short-range stair attempt lifecycle with force-window success."""

    enabled: bool = False
    scan_offset: float = 0.20
    step_height_min: float = 0.12
    step_height_max: float = 0.14
    height_command_range: tuple[float, float] = (0.37, 0.40)
    contact_force_threshold: float = 5.0
    success_height_error: float = 0.05
    success_duration_s: float = 0.30
    drop_threshold: float = 0.10
    timeout_s: float = 5.0
    failure_terminate: bool = True

    def validate(self) -> None:
        _two_finite("state_machine.stair.height_command_range", self.height_command_range)
        for name in (
            "scan_offset",
            "step_height_min",
            "step_height_max",
            "contact_force_threshold",
            "success_height_error",
            "success_duration_s",
            "drop_threshold",
            "timeout_s",
        ):
            _non_negative(f"state_machine.stair.{name}", getattr(self, name))
        if float(self.step_height_min) >= float(self.step_height_max):
            raise ValueError("state_machine.stair step_height_min must be < step_height_max")


@dataclass
class WheelbipeAirborneCommandResampleConfig:
    """Terrain-gated command override sampled on an Airborne entry.

    The pinned Rough-v1 owner enables this only for its stair terrain
    profiles.  ``terrain_names`` names real generated-terrain columns; the
    owner resolves them once during construction and supplies a numeric
    profile id to the hot-path sensor frame.
    """

    enabled: bool = False
    probability: float = 0.15
    terrain_names: tuple[str, ...] = ()
    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0)
    ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_x_sign_from_current: bool = True

    def validate(self) -> None:
        _probability(
            "state_machine.airborne_command_resample.probability",
            self.probability,
        )
        for name in ("lin_vel_x_range", "lin_vel_y_range", "ang_vel_z_range"):
            _two_finite(
                f"state_machine.airborne_command_resample.{name}",
                getattr(self, name),
            )
        if self.enabled and not self.terrain_names:
            raise ValueError(
                "enabled state_machine.airborne_command_resample requires at least "
                "one explicit generated terrain name"
            )
        if any(not str(name).strip() for name in self.terrain_names):
            raise ValueError(
                "state_machine.airborne_command_resample.terrain_names cannot contain empty names"
            )


@dataclass
class WheelbipeAirborneRewardConfig:
    """Non-zero reward behavior of the pinned Flat/Rough-v1 Airborne owner."""

    enabled: bool = False
    base_scale_overrides: dict[str, float] = field(default_factory=dict)
    airborne_scale_multipliers: dict[str, float] = field(default_factory=dict)
    joint_pos_limits_weight: float = 0.0
    wheel_heading_x_centering_weight: float = 0.0
    wheel_zero_torque_exp_weight: float = 0.0
    wheel_directional_speed_weight: float = 0.0
    wheel_directional_speed_shortfall_weight: float = 0.0
    rear2_joint_lower: float = np.deg2rad(-1.0)
    rear2_joint_upper: float = np.deg2rad(68.5)
    rear2_lower_boundary_ratio: float = 0.10
    rear2_upper_boundary_ratio: float = 0.05
    wheel_heading_contact_duration_s: float = 0.02
    wheel_heading_base_contact_duration_s: float = 0.02
    wheel_heading_z_max: float = -0.10
    wheel_heading_sigma: float = 0.02
    wheel_zero_torque_sigma: float = 1.5
    wheel_zero_torque_before_contact_s: float = 0.02
    directional_speed_start: float = 0.0
    directional_speed_full: float = 10.0
    directional_command_x_threshold: float = 1.0
    directional_root_x_threshold: float = 1.0
    directional_before_contact_s: float = 0.02
    directional_shortfall_before_contact_s: float = 0.05

    def validate(self) -> None:
        for mapping_name in ("base_scale_overrides", "airborne_scale_multipliers"):
            mapping = getattr(self, mapping_name)
            if not isinstance(mapping, dict):
                raise ValueError(f"state_machine.airborne_reward.{mapping_name} must be a dict")
            for name, value in mapping.items():
                if not str(name).strip() or not np.isfinite(float(value)):
                    raise ValueError(
                        f"state_machine.airborne_reward.{mapping_name} contains an "
                        f"invalid entry {name!r}: {value!r}"
                    )
        unknown_base = set(self.base_scale_overrides) - _AIRBORNE_BASE_SCALE_OVERRIDE_TERMS
        if unknown_base:
            raise ValueError(
                "state_machine.airborne_reward.base_scale_overrides contains unsupported "
                f"source terms: {sorted(unknown_base)}"
            )
        unknown_multipliers = (
            set(self.airborne_scale_multipliers) - _AIRBORNE_SCALE_MULTIPLIER_TERMS
        )
        if unknown_multipliers:
            raise ValueError(
                "state_machine.airborne_reward.airborne_scale_multipliers contains "
                f"unsupported source terms: {sorted(unknown_multipliers)}"
            )
        for name in (
            "joint_pos_limits_weight",
            "wheel_heading_x_centering_weight",
            "wheel_zero_torque_exp_weight",
            "wheel_directional_speed_weight",
            "wheel_directional_speed_shortfall_weight",
            "rear2_joint_lower",
            "rear2_joint_upper",
            "wheel_heading_z_max",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"state_machine.airborne_reward.{name} must be finite")
        if float(self.rear2_joint_upper) <= float(self.rear2_joint_lower):
            raise ValueError(
                "state_machine.airborne_reward rear2_joint_upper must be greater than lower"
            )
        for name in ("rear2_lower_boundary_ratio", "rear2_upper_boundary_ratio"):
            ratio = float(getattr(self, name))
            if not 0.0 <= ratio < 0.5:
                raise ValueError(f"state_machine.airborne_reward.{name} must be in [0, 0.5)")
        for name in (
            "wheel_heading_contact_duration_s",
            "wheel_heading_base_contact_duration_s",
            "wheel_heading_sigma",
            "wheel_zero_torque_sigma",
            "wheel_zero_torque_before_contact_s",
            "directional_speed_start",
            "directional_speed_full",
            "directional_command_x_threshold",
            "directional_root_x_threshold",
            "directional_before_contact_s",
            "directional_shortfall_before_contact_s",
        ):
            _non_negative(f"state_machine.airborne_reward.{name}", getattr(self, name))
        if float(self.wheel_heading_sigma) <= 0.0 or float(self.wheel_zero_torque_sigma) <= 0.0:
            raise ValueError("state_machine Airborne reward sigmas must be positive")
        if float(self.directional_speed_full) <= float(self.directional_speed_start):
            raise ValueError(
                "state_machine.airborne_reward directional_speed_full must exceed start"
            )


@dataclass
class WheelbipeTerrainCommandProfileConfig:
    """One pinned generated-terrain command/reset override profile."""

    height_ranges: tuple[tuple[float, float], ...] | None = None
    lin_vel_x_ranges: tuple[tuple[float, float], ...] | None = None
    lin_vel_y_ranges: tuple[tuple[float, float], ...] | None = None
    ang_vel_z_heading_ranges: tuple[tuple[float, float], ...] | None = None
    ang_vel_z_non_heading_ranges: tuple[tuple[float, float], ...] | None = None
    reset_heading_axis_aligned_only: bool = False
    disable_predefined_reset_air: bool = False
    disable_predefined_reset_ground: bool = False
    disable_special_mode: bool = False

    def validate(self, *, name: str) -> None:
        for field_name in (
            "height_ranges",
            "lin_vel_x_ranges",
            "lin_vel_y_ranges",
            "ang_vel_z_heading_ranges",
            "ang_vel_z_non_heading_ranges",
        ):
            normalized = _range_spec(
                f"terrain_commands.profiles.{name}.{field_name}",
                getattr(self, field_name),
            )
            setattr(self, field_name, normalized)


@dataclass
class WheelbipeTerrainCommandConfig:
    """Config-first terrain-profile command owner for exact rough aliases."""

    enabled: bool = False
    profiles: dict[str, WheelbipeTerrainCommandProfileConfig] = field(default_factory=dict)
    force_forward: bool = False
    """For play-only showcases, make profile speeds travel along +x.

    Training keeps the source's signed forward/backward ranges.  A playback
    profile may enable this switch so every vectorized terrain cell is crossed
    in the same direction and the camera reliably reaches its obstacle.
    """

    def __post_init__(self) -> None:
        self.profiles = {
            str(name): (
                WheelbipeTerrainCommandProfileConfig(**profile)
                if isinstance(profile, dict)
                else profile
            )
            for name, profile in self.profiles.items()
        }

    def validate(self) -> None:
        if self.enabled and not self.profiles:
            raise ValueError("enabled terrain_commands requires explicit source profiles")
        for name, profile in self.profiles.items():
            if not name.strip():
                raise ValueError("terrain_commands profile names cannot be empty")
            if not isinstance(profile, WheelbipeTerrainCommandProfileConfig):
                raise ValueError(
                    f"terrain_commands.profiles.{name} must be WheelbipeTerrainCommandProfileConfig"
                )
            profile.validate(name=name)


@dataclass
class WheelbipeStateMachineConfig:
    """State-machine stack configuration with legacy field compatibility."""

    enabled: bool = False
    control_dt: float = 0.02
    reset_airborne_probability: float = 0.0
    airborne_enabled: bool = True
    airborne_enter_steps: int = 2
    landing_contact_steps: int = 2
    landing_hold_steps: int = 5
    max_airborne_steps: int = 150
    wheel_radius: float = 0.06
    body_airborne_height: float = 0.30
    wheel_airborne_clearance: float = 0.08
    wheel_contact_height: float = 0.15
    wheel_contact_force_threshold: float = 20.0
    base_contact_force_threshold: float = 5.0
    base_contact_steps: int = 13
    contact_history_steps: int = 3
    allow_geometric_contact_fallback: bool = False
    airborne_height_target: float = 0.30
    landing_height_target: float = 0.24
    airborne_height_override_enabled: bool = True
    landing_height_override_enabled: bool = True
    command_scale_airborne: float = 0.35
    command_scale_landing: float = 0.0
    landing_trajectory_enabled: bool = False
    landing_trajectory_start_steps: int = 1
    landing_trajectory_end_vel_z: float = 0.0
    landing_trajectory_min_height_margin: float = 0.02
    landing_trajectory_min_down_vel: float = 0.20
    landing_trajectory_duration_s: float = 0.30
    landing_trajectory_max_abs_acc: float | None = 30.0
    slope_height_difference_min: float = 0.02
    airborne_command_resample: WheelbipeAirborneCommandResampleConfig = field(
        default_factory=WheelbipeAirborneCommandResampleConfig
    )
    airborne_reward: WheelbipeAirborneRewardConfig = field(
        default_factory=WheelbipeAirborneRewardConfig
    )
    jump_takeoff: WheelbipeJumpTakeoffConfig = field(default_factory=WheelbipeJumpTakeoffConfig)
    step_up: WheelbipeStepUpConfig = field(default_factory=WheelbipeStepUpConfig)
    stair: WheelbipeStairConfig = field(default_factory=WheelbipeStairConfig)

    def __post_init__(self) -> None:
        if isinstance(self.airborne_command_resample, dict):
            self.airborne_command_resample = WheelbipeAirborneCommandResampleConfig(
                **self.airborne_command_resample
            )
        if isinstance(self.airborne_reward, dict):
            self.airborne_reward = WheelbipeAirborneRewardConfig(**self.airborne_reward)
        if isinstance(self.jump_takeoff, dict):
            self.jump_takeoff = WheelbipeJumpTakeoffConfig(**self.jump_takeoff)
        if isinstance(self.step_up, dict):
            self.step_up = WheelbipeStepUpConfig(**self.step_up)
        if isinstance(self.stair, dict):
            self.stair = WheelbipeStairConfig(**self.stair)

    def validate(self) -> None:
        _probability("state_machine.reset_airborne_probability", self.reset_airborne_probability)
        _non_negative("state_machine.control_dt", self.control_dt)
        if float(self.control_dt) <= 0.0:
            raise ValueError("state_machine.control_dt must be positive")
        for name in (
            "airborne_enter_steps",
            "landing_contact_steps",
            "landing_hold_steps",
            "max_airborne_steps",
            "base_contact_steps",
            "contact_history_steps",
            "landing_trajectory_start_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"state_machine.{name} must be a positive integer, got {value!r}")
        for name in (
            "wheel_radius",
            "body_airborne_height",
            "wheel_airborne_clearance",
            "wheel_contact_height",
            "wheel_contact_force_threshold",
            "base_contact_force_threshold",
            "airborne_height_target",
            "landing_height_target",
            "command_scale_airborne",
            "command_scale_landing",
            "landing_trajectory_min_height_margin",
            "landing_trajectory_min_down_vel",
            "landing_trajectory_duration_s",
            "slope_height_difference_min",
        ):
            _non_negative(f"state_machine.{name}", getattr(self, name))
        if float(self.landing_trajectory_duration_s) <= 0.0:
            raise ValueError("state_machine.landing_trajectory_duration_s must be positive")
        if float(self.landing_trajectory_end_vel_z) > 0.0:
            raise ValueError("state_machine.landing_trajectory_end_vel_z cannot be positive")
        if self.landing_trajectory_max_abs_acc is not None:
            _non_negative(
                "state_machine.landing_trajectory_max_abs_acc",
                self.landing_trajectory_max_abs_acc,
            )
        if not isinstance(self.jump_takeoff, WheelbipeJumpTakeoffConfig):
            raise ValueError("state_machine.jump_takeoff has the wrong config type")
        if not isinstance(self.step_up, WheelbipeStepUpConfig):
            raise ValueError("state_machine.step_up has the wrong config type")
        if not isinstance(self.stair, WheelbipeStairConfig):
            raise ValueError("state_machine.stair has the wrong config type")
        if not isinstance(
            self.airborne_command_resample,
            WheelbipeAirborneCommandResampleConfig,
        ):
            raise ValueError("state_machine.airborne_command_resample has the wrong config type")
        if not isinstance(self.airborne_reward, WheelbipeAirborneRewardConfig):
            raise ValueError("state_machine.airborne_reward has the wrong config type")
        self.airborne_command_resample.validate()
        self.airborne_reward.validate()
        self.jump_takeoff.validate()
        self.step_up.validate()
        self.stair.validate()


@dataclass(frozen=True)
class WheelbipeStateMachineSensors:
    """One explicit vectorized state-machine input frame.

    Required shapes are ``wheel_pos_w (N,2,3)``, per-wheel terrain heights
    ``(N,2)``, base position/velocity ``(N,3)``, base quaternion ``(N,4)``,
    commands ``(N,3)`` and scalar batches ``(N,)``.  Force magnitudes are
    ``wheel (N,2)``, ``base (N,K)`` and owner history ``(N,H,2)`` (the pinned
    history length is three).  ``terrain_profile_id`` is a cold-path-resolved
    numeric identity; ``-1`` means that no Airborne command profile applies.
    """

    wheel_pos_w: np.ndarray
    wheel_ground_height: np.ndarray
    base_pos_w: np.ndarray
    base_quat_w: np.ndarray
    base_lin_vel_w: np.ndarray
    base_ground_height: np.ndarray
    wheel_forward_ground_height: np.ndarray
    wheel_stair_ground_height: np.ndarray
    commands: np.ndarray
    height_commands: np.ndarray
    jump_request: np.ndarray
    slope: np.ndarray
    contact_source: WheelbipeContactSource
    wheel_contact_force_norm: np.ndarray | None = None
    base_contact_force_norm: np.ndarray | None = None
    wheel_contact_force_history_norm: np.ndarray | None = None
    terrain_profile_id: np.ndarray | None = None

    @classmethod
    def from_geometry(
        cls, wheel_pos_w: np.ndarray, terrain_height: np.ndarray
    ) -> "WheelbipeStateMachineSensors":
        """Translate the legacy two-array API into an explicit geometry frame."""

        wheel = np.asarray(wheel_pos_w, dtype=np.float64)
        if wheel.ndim != 3 or wheel.shape[1:] != (2, 3):
            raise ValueError(f"wheel_pos_w must have shape (N, 2, 3), got {wheel.shape}")
        n = wheel.shape[0]
        terrain = np.asarray(terrain_height, dtype=np.float64).reshape(-1)
        if terrain.shape != (n,):
            raise ValueError(f"terrain_height must have shape {(n,)}, got {terrain.shape}")
        per_wheel = np.broadcast_to(terrain[:, None], (n, 2)).copy()
        base_pos = np.mean(wheel, axis=1)
        base_pos[:, 2] = np.max(wheel[:, :, 2], axis=1)
        zeros3 = np.zeros((n, 3), dtype=np.float64)
        return cls(
            wheel_pos_w=wheel,
            wheel_ground_height=per_wheel,
            base_pos_w=base_pos,
            base_quat_w=np.broadcast_to(
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                (n, 4),
            ).copy(),
            base_lin_vel_w=zeros3,
            base_ground_height=terrain,
            wheel_forward_ground_height=per_wheel.copy(),
            wheel_stair_ground_height=per_wheel.copy(),
            commands=zeros3.copy(),
            height_commands=np.zeros((n,), dtype=np.float64),
            jump_request=np.zeros((n,), dtype=bool),
            slope=np.zeros((n,), dtype=bool),
            contact_source=WheelbipeContactSource.GEOMETRY,
            terrain_profile_id=np.full((n,), -1, dtype=np.int16),
        )

    def validate(self, num_envs: int) -> None:
        required = {
            "wheel_pos_w": (num_envs, 2, 3),
            "wheel_ground_height": (num_envs, 2),
            "base_pos_w": (num_envs, 3),
            "base_quat_w": (num_envs, 4),
            "base_lin_vel_w": (num_envs, 3),
            "base_ground_height": (num_envs,),
            "wheel_forward_ground_height": (num_envs, 2),
            "wheel_stair_ground_height": (num_envs, 2),
            "commands": (num_envs, 3),
            "height_commands": (num_envs,),
            "jump_request": (num_envs,),
            "slope": (num_envs,),
        }
        for name, expected in required.items():
            actual = np.asarray(getattr(self, name)).shape
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")
        try:
            source = WheelbipeContactSource(self.contact_source)
        except ValueError as exc:
            raise ValueError(
                f"invalid state-machine contact_source {self.contact_source!r}"
            ) from exc
        if source is WheelbipeContactSource.FORCE:
            if self.wheel_contact_force_norm is None or self.base_contact_force_norm is None:
                raise ValueError(
                    "force contact_source requires current wheel and base force magnitudes"
                )
        if self.wheel_contact_force_norm is not None:
            actual = np.asarray(self.wheel_contact_force_norm).shape
            if actual != (num_envs, 2):
                raise ValueError(
                    f"wheel_contact_force_norm must have shape {(num_envs, 2)}, got {actual}"
                )
        if self.base_contact_force_norm is not None:
            actual = np.asarray(self.base_contact_force_norm).shape
            if len(actual) != 2 or actual[0] != num_envs:
                raise ValueError(f"base_contact_force_norm must have shape (N, K), got {actual}")
        if self.wheel_contact_force_history_norm is not None:
            actual = np.asarray(self.wheel_contact_force_history_norm).shape
            if len(actual) != 3 or actual[0] != num_envs or actual[1] < 1 or actual[2] != 2:
                raise ValueError(
                    f"wheel_contact_force_history_norm must have shape (N, H, 2), got {actual}"
                )
        if self.terrain_profile_id is not None:
            profile = np.asarray(self.terrain_profile_id)
            if profile.shape != (num_envs,) or profile.dtype.kind not in "iu":
                raise ValueError(
                    "terrain_profile_id must be an integer array with shape "
                    f"{(num_envs,)}, got {profile.shape} and {profile.dtype}"
                )


class WheelbipeStateMachine:
    """Vectorized Airborne/JumpTakeoff/StepUp/Stair manager."""

    def __init__(self, cfg: WheelbipeStateMachineConfig, num_envs: int):
        cfg.validate()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs!r}")
        n = self.num_envs
        self.state = np.full((n,), WheelbipeMotionState.NORMAL, dtype=np.int8)
        self.elapsed = np.zeros((n,), dtype=np.int32)
        self.episode_elapsed = np.zeros((n,), dtype=np.int32)
        self.failure = np.zeros((n,), dtype=bool)
        self.timeout = np.zeros((n,), dtype=bool)
        self.reset_generation = np.zeros((n,), dtype=np.uint64)
        self.airborne_state = np.zeros((n,), dtype=bool)
        self.landing_state = np.zeros((n,), dtype=bool)
        self.airborne_elapsed = np.zeros((n,), dtype=np.int32)
        self.airborne_enter_streak = np.zeros((n,), dtype=np.int32)
        self.wheel_contact_steps = np.zeros((n, 2), dtype=np.int32)
        self.base_contact_steps = np.zeros((n,), dtype=np.int32)
        # Backward-compatible public streak arrays.
        self.contact_streak = np.zeros((n,), dtype=np.int32)
        self.no_contact_streak = np.zeros((n,), dtype=np.int32)
        self.airborne_entry_command = np.zeros((n, 3), dtype=np.float64)
        self.airborne_command_override_active = np.zeros((n,), dtype=bool)
        self.airborne_command_override_values = np.zeros((n, 3), dtype=np.float64)
        self.landing_traj_active = np.zeros((n,), dtype=bool)
        self.landing_traj_time = np.zeros((n,), dtype=np.float64)
        self.landing_traj_start_height = np.zeros((n,), dtype=np.float64)
        self.landing_traj_start_vel = np.zeros((n,), dtype=np.float64)
        self.landing_traj_acc = np.zeros((n,), dtype=np.float64)
        self.landing_traj_height = np.zeros((n,), dtype=np.float64)
        self.landing_traj_vel_z = np.zeros((n,), dtype=np.float64)
        self.jump_phase = np.full((n,), WheelbipeJumpPhase.IDLE, dtype=np.int8)
        self.jump_phase_time = np.zeros((n,), dtype=np.float64)
        self.jump_cooldown_time = np.zeros((n,), dtype=np.float64)
        self.jump_target_height = np.zeros((n,), dtype=np.float64)
        self.jump_duration = np.zeros((n,), dtype=np.float64)
        self.jump_push_time = np.zeros((n,), dtype=np.float64)
        self.jump_release_vel_z = np.zeros((n,), dtype=np.float64)
        self.jump_ref_vel_z = np.zeros((n,), dtype=np.float64)
        self.jump_ref_phase = np.zeros((n,), dtype=np.float64)
        self.jump_trigger_event = np.zeros((n,), dtype=bool)
        self.jump_tuck_event = np.zeros((n,), dtype=bool)
        self.jump_exit_event = np.zeros((n,), dtype=bool)
        self.jump_force_airborne_request = np.zeros((n,), dtype=bool)
        self.step_prev_ground = np.zeros((n, 2), dtype=np.float64)
        self.step_prev_valid = np.zeros((n,), dtype=bool)
        self.step_prev_direction = np.zeros((n,), dtype=np.int8)
        self.step_hold_remaining = np.zeros((n,), dtype=np.int32)
        self.step_height_command = np.zeros((n,), dtype=np.float64)
        self.step_detect_event = np.zeros((n,), dtype=bool)
        self.wall_event = np.zeros((n,), dtype=bool)
        self.stair_prev_ground = np.zeros((n, 2), dtype=np.float64)
        self.stair_prev_valid = np.zeros((n,), dtype=bool)
        self.stair_prev_direction = np.zeros((n,), dtype=np.int8)
        self.stair_state = np.zeros((n,), dtype=bool)
        self.stair_reference_height = np.zeros((n,), dtype=np.float64)
        self.stair_height_command = np.zeros((n,), dtype=np.float64)
        self.stair_elapsed = np.zeros((n,), dtype=np.int32)
        self.stair_success_steps = np.zeros((n,), dtype=np.int32)
        self.stair_detect_event = np.zeros((n,), dtype=bool)
        self.stair_success_event = np.zeros((n,), dtype=bool)
        self.stair_failure_event = np.zeros((n,), dtype=bool)
        self.stair_timeout_event = np.zeros((n,), dtype=bool)
        self.slope_state = np.zeros((n,), dtype=bool)
        self._rng = GLOBAL_NUMPY_RANDOM

    @property
    def _dt(self) -> float:
        return float(self.cfg.control_dt)

    def reset(self, env_ids: np.ndarray, *, airborne: np.ndarray | None = None) -> None:
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        if ids.size == 0:
            return
        if np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise IndexError(f"state-machine env ids out of range: {ids.tolist()}")
        self.reset_generation[ids] += np.uint64(1)
        for array in (
            self.elapsed,
            self.episode_elapsed,
            self.failure,
            self.timeout,
            self.airborne_state,
            self.landing_state,
            self.airborne_elapsed,
            self.airborne_enter_streak,
            self.wheel_contact_steps,
            self.base_contact_steps,
            self.contact_streak,
            self.no_contact_streak,
            self.airborne_entry_command,
            self.airborne_command_override_active,
            self.airborne_command_override_values,
            self.landing_traj_active,
            self.landing_traj_time,
            self.landing_traj_start_height,
            self.landing_traj_start_vel,
            self.landing_traj_acc,
            self.landing_traj_height,
            self.landing_traj_vel_z,
            self.jump_phase,
            self.jump_phase_time,
            self.jump_cooldown_time,
            self.jump_target_height,
            self.jump_duration,
            self.jump_push_time,
            self.jump_release_vel_z,
            self.jump_ref_vel_z,
            self.jump_ref_phase,
            self.jump_trigger_event,
            self.jump_tuck_event,
            self.jump_exit_event,
            self.jump_force_airborne_request,
            self.step_prev_ground,
            self.step_prev_valid,
            self.step_prev_direction,
            self.step_hold_remaining,
            self.step_height_command,
            self.step_detect_event,
            self.wall_event,
            self.stair_prev_ground,
            self.stair_prev_valid,
            self.stair_prev_direction,
            self.stair_state,
            self.stair_reference_height,
            self.stair_height_command,
            self.stair_elapsed,
            self.stair_success_steps,
            self.stair_detect_event,
            self.stair_success_event,
            self.stair_failure_event,
            self.stair_timeout_event,
            self.slope_state,
        ):
            array[ids] = 0
        self.state[ids] = WheelbipeMotionState.NORMAL
        if airborne is not None:
            mask = np.asarray(airborne, dtype=bool).reshape(-1)
            if mask.shape != (ids.size,):
                raise ValueError(
                    f"airborne reset mask must have shape {(ids.size,)}, got {mask.shape}"
                )
            air_ids = ids[mask]
            self.airborne_state[air_ids] = True
            self.state[air_ids] = WheelbipeMotionState.AIRBORNE

    def _sample_airborne_command_override(
        self,
        enter: np.ndarray,
        sensors: WheelbipeStateMachineSensors,
    ) -> None:
        """Sample the pinned Rough-v1 terrain command profile once per entry."""

        cfg = self.cfg.airborne_command_resample
        entering = np.asarray(enter, dtype=bool)
        self.airborne_command_override_active[entering] = False
        self.airborne_command_override_values[entering] = 0.0
        if not cfg.enabled or not np.any(entering):
            return
        if sensors.terrain_profile_id is None:
            raise ValueError(
                "enabled Airborne terrain command resampling requires terrain_profile_id"
            )
        profile = np.asarray(sensors.terrain_profile_id, dtype=np.int64)
        eligible = entering & (profile >= 0)
        if float(cfg.probability) < 1.0:
            eligible &= self._rng.random(self.num_envs) < float(cfg.probability)
        ids = np.flatnonzero(eligible)
        if ids.size == 0:
            return

        x_low, x_high = sorted(float(value) for value in cfg.lin_vel_x_range)
        current_x = np.asarray(sensors.commands, dtype=np.float64)[ids, 0]
        sampled_x = np.zeros((ids.size,), dtype=np.float64)
        positive = current_x > 0.0
        negative = current_x < 0.0
        zero = ~(positive | negative)
        if cfg.lin_vel_x_sign_from_current:
            pos_low, pos_high = max(x_low, 0.0), max(x_high, 0.0)
            neg_low, neg_high = min(x_low, 0.0), min(x_high, 0.0)
            sampled_x[positive] = self._rng.uniform(
                min(pos_low, pos_high), max(pos_low, pos_high), size=int(np.count_nonzero(positive))
            )
            sampled_x[negative] = self._rng.uniform(
                min(neg_low, neg_high), max(neg_low, neg_high), size=int(np.count_nonzero(negative))
            )
            sampled_x[zero] = 0.0
        else:
            sampled_x = self._rng.uniform(x_low, x_high, size=ids.size)
        y_low, y_high = sorted(float(value) for value in cfg.lin_vel_y_range)
        yaw_low, yaw_high = sorted(float(value) for value in cfg.ang_vel_z_range)
        self.airborne_command_override_values[ids, 0] = sampled_x
        self.airborne_command_override_values[ids, 1] = self._rng.uniform(
            y_low, y_high, size=ids.size
        )
        self.airborne_command_override_values[ids, 2] = self._rng.uniform(
            yaw_low, yaw_high, size=ids.size
        )
        self.airborne_command_override_active[ids] = True

    @staticmethod
    def _direction(commands: np.ndarray) -> np.ndarray:
        return np.where(commands[:, 0] < 0.0, -1, 1).astype(np.int8)

    @staticmethod
    def _temporal_difference(
        current: np.ndarray,
        direction: np.ndarray,
        previous: np.ndarray,
        previous_valid: np.ndarray,
        previous_direction: np.ndarray,
    ) -> np.ndarray:
        valid = previous_valid & (direction == previous_direction)
        difference = np.where(valid[:, None], current - previous, 0.0)
        previous[...] = current
        previous_valid[...] = np.all(np.isfinite(current), axis=1)
        previous_direction[...] = direction
        return difference

    def _update_step_up(self, sensors: WheelbipeStateMachineSensors, direction: np.ndarray) -> None:
        cfg = self.cfg.step_up
        self.step_detect_event.fill(False)
        self.wall_event.fill(False)
        if not cfg.enabled:
            self.step_hold_remaining.fill(0)
            self.step_height_command.fill(0.0)
            self.step_prev_valid.fill(False)
            return
        np.maximum(self.step_hold_remaining - 1, 0, out=self.step_hold_remaining)
        current = np.asarray(sensors.wheel_forward_ground_height, dtype=np.float64)
        difference = self._temporal_difference(
            current,
            direction,
            self.step_prev_ground,
            self.step_prev_valid,
            self.step_prev_direction,
        )
        wall = np.any(difference > float(cfg.wall_height), axis=1)
        step = (
            np.any(
                (difference > float(cfg.step_height_min))
                & (difference < float(cfg.step_height_max)),
                axis=1,
            )
            & ~wall
        )
        self.wall_event[...] = wall
        self.step_detect_event[...] = step
        self.timeout |= wall
        self.step_hold_remaining[wall] = 0
        self.step_height_command[wall] = 0.0
        if np.any(step) and float(cfg.hold_s) > 0.0:
            target = np.asarray(sensors.height_commands, dtype=np.float64) + float(
                cfg.height_command_bias
            )
            if cfg.height_command_max is not None:
                target = np.minimum(target, float(cfg.height_command_max))
            self.step_height_command[step] = target[step]
            self.step_hold_remaining[step] = max(int(np.ceil(float(cfg.hold_s) / self._dt)), 1)

    def _update_stair(
        self,
        sensors: WheelbipeStateMachineSensors,
        direction: np.ndarray,
        force_history: np.ndarray | None,
        geometry_contact: np.ndarray,
    ) -> None:
        cfg = self.cfg.stair
        self.stair_detect_event.fill(False)
        self.stair_success_event.fill(False)
        self.stair_failure_event.fill(False)
        self.stair_timeout_event.fill(False)
        if not cfg.enabled:
            self.stair_state.fill(False)
            self.stair_prev_valid.fill(False)
            return
        current = np.asarray(sensors.wheel_stair_ground_height, dtype=np.float64)
        difference = self._temporal_difference(
            current,
            direction,
            self.stair_prev_ground,
            self.stair_prev_valid,
            self.stair_prev_direction,
        )
        per_wheel = (difference > float(cfg.step_height_min)) & (
            difference < float(cfg.step_height_max)
        )
        detected = np.any(per_wheel, axis=1)
        self.stair_detect_event[...] = detected
        enter = detected & ~self.stair_state & ~self.wall_event
        if np.any(enter):
            candidates = np.where(per_wheel, current, -np.inf)
            reference = np.max(candidates, axis=1)
            reference = np.where(np.isfinite(reference), reference, current[:, 0])
            low, high = sorted(float(v) for v in cfg.height_command_range)
            self.stair_reference_height[enter] = reference[enter]
            self.stair_height_command[enter] = self._rng.uniform(
                low, high, size=int(np.count_nonzero(enter))
            )
            self.stair_elapsed[enter] = 0
            self.stair_success_steps[enter] = 0
        active = (self.stair_state | enter) & ~self.wall_event
        self.stair_state[...] = active
        self.stair_elapsed[active] += 1
        self.stair_elapsed[~active] = 0
        if force_history is None:
            both_contact = np.all(geometry_contact, axis=1)
        else:
            peaks = np.max(force_history, axis=1)
            both_contact = np.all(peaks > float(cfg.contact_force_threshold), axis=1)
        relative_height = (
            np.asarray(sensors.base_pos_w, dtype=np.float64)[:, 2] - self.stair_reference_height
        )
        success_condition = (
            active
            & both_contact
            & (relative_height >= self.stair_height_command - float(cfg.success_height_error))
        )
        self.stair_success_steps[success_condition] += 1
        self.stair_success_steps[~success_condition] = 0
        success_steps = max(int(np.ceil(float(cfg.success_duration_s) / self._dt)), 1)
        success = active & (self.stair_success_steps >= success_steps)
        dropped = active & np.all(
            current < self.stair_reference_height[:, None] - float(cfg.drop_threshold),
            axis=1,
        )
        timeout_steps = max(int(np.ceil(float(cfg.timeout_s) / self._dt)), 1)
        timed_out = active & (self.stair_elapsed >= timeout_steps)
        failed = dropped | timed_out
        self.stair_success_event[...] = success
        self.stair_failure_event[...] = failed
        self.stair_timeout_event[...] = timed_out
        clear = success | failed | self.wall_event
        self.stair_state[clear] = False
        if np.any(failed):
            if cfg.failure_terminate:
                self.failure |= failed
            else:
                self.timeout |= failed
        for array in (
            self.stair_reference_height,
            self.stair_height_command,
            self.stair_elapsed,
            self.stair_success_steps,
        ):
            array[clear] = 0
        # Pinned Stair has priority over Airborne and clears its timers.
        if np.any(self.stair_state):
            mask = self.stair_state
            self.airborne_state[mask] = False
            self.landing_state[mask] = False
            self.airborne_elapsed[mask] = 0
            self.wheel_contact_steps[mask] = 0
            self.base_contact_steps[mask] = 0

    def _sample_jump(self, enter: np.ndarray) -> None:
        cfg = self.cfg.jump_takeoff
        count = int(np.count_nonzero(enter))
        if count == 0:
            return
        low, high = sorted(float(v) for v in cfg.peak_height_range)
        peak = self._rng.uniform(low, high, size=count)
        gravity = float(cfg.gravity)
        if cfg.tuck_timing_mode == "fixed_tuck_time":
            push_time = np.full((count,), max(float(cfg.fixed_tuck_time_s), self._dt))
            discriminant = np.square(gravity * push_time) + 8.0 * gravity * np.maximum(
                peak - float(cfg.push_start_height), 0.0
            )
            release_velocity = 0.5 * (-gravity * push_time + np.sqrt(discriminant))
            raw_release_height = float(cfg.push_start_height) + 0.5 * release_velocity * push_time
            release_height = np.minimum(raw_release_height, float(cfg.release_height))
            release_velocity = np.sqrt(np.maximum(2.0 * gravity * (peak - release_height), 0.0))
            flight_peak_time = release_velocity / gravity
        else:
            delta_height = np.maximum(peak - float(cfg.release_height), 0.0)
            flight_peak_time = np.sqrt(2.0 * delta_height / gravity)
            release_velocity = gravity * flight_peak_time
            push_distance = max(float(cfg.release_height) - float(cfg.push_start_height), 0.0)
            push_time = np.divide(
                2.0 * push_distance,
                release_velocity,
                out=np.zeros_like(release_velocity),
                where=release_velocity > 1.0e-6,
            )
        duration = np.maximum(
            push_time + flight_peak_time * float(cfg.exit_time_scale_after_peak),
            max(float(cfg.min_duration_s), self._dt),
        )
        self.jump_phase[enter] = WheelbipeJumpPhase.PUSH
        self.jump_phase_time[enter] = 0.0
        self.jump_target_height[enter] = peak
        self.jump_duration[enter] = duration
        self.jump_push_time[enter] = push_time
        self.jump_release_vel_z[enter] = release_velocity
        self.jump_trigger_event[enter] = True

    def _update_jump(
        self, sensors: WheelbipeStateMachineSensors, wheel_clearance: np.ndarray
    ) -> None:
        cfg = self.cfg.jump_takeoff
        self.jump_trigger_event.fill(False)
        self.jump_tuck_event.fill(False)
        self.jump_exit_event.fill(False)
        self.jump_force_airborne_request.fill(False)
        if not cfg.enabled:
            self.jump_phase.fill(WheelbipeJumpPhase.IDLE)
            self.jump_phase_time.fill(0.0)
            return
        active = self.jump_phase != WheelbipeJumpPhase.IDLE
        self.jump_phase_time[active] += self._dt
        self.jump_phase_time[~active] = 0.0
        self.jump_cooldown_time[...] = np.maximum(self.jump_cooldown_time - self._dt, 0.0)
        request = np.asarray(sensors.jump_request, dtype=bool).copy()
        if cfg.trigger_mode == "random" and float(cfg.probability_per_step) > 0.0:
            request |= self._rng.random(self.num_envs) < float(cfg.probability_per_step)
        ready = self.episode_elapsed.astype(np.float64) * self._dt >= float(cfg.min_episode_time_s)
        enter = (
            (self.jump_phase == WheelbipeJumpPhase.IDLE)
            & request
            & ready
            & (self.jump_cooldown_time <= 0.0)
            & ~self.stair_state
            & ~self.wall_event
        )
        self._sample_jump(enter)
        active = self.jump_phase != WheelbipeJumpPhase.IDLE
        duration = np.maximum(self.jump_duration, self._dt)
        self.jump_ref_phase[active] = np.clip(
            self.jump_phase_time[active] / duration[active], 0.0, 1.0
        )
        push_time = np.maximum(self.jump_push_time, 1.0e-6)
        push_acceleration = self.jump_release_vel_z / push_time
        after_release = np.maximum(self.jump_phase_time - self.jump_push_time, 0.0)
        reference = np.where(
            self.jump_phase_time <= self.jump_push_time,
            push_acceleration * self.jump_phase_time,
            self.jump_release_vel_z - float(cfg.gravity) * after_release,
        )
        self.jump_ref_vel_z[active] = reference[active]
        push = self.jump_phase == WheelbipeJumpPhase.PUSH
        body_height = np.asarray(sensors.base_pos_w, dtype=np.float64)[:, 2] - np.asarray(
            sensors.base_ground_height, dtype=np.float64
        )
        if cfg.tuck_start_time_s is not None:
            natural = np.zeros((self.num_envs,), dtype=bool)
            by_time = push & (self.jump_phase_time >= float(cfg.tuck_start_time_s))
        elif cfg.tuck_start_height_ratio is not None:
            natural = push & (
                body_height >= self.jump_target_height * float(cfg.tuck_start_height_ratio)
            )
            by_time = np.zeros((self.num_envs,), dtype=bool)
        else:
            natural = push & (
                (body_height >= float(cfg.tuck_start_height))
                | np.all(wheel_clearance > float(cfg.tuck_wheel_air_margin), axis=1)
            )
            tuck_time = (
                self.jump_push_time
                if cfg.tuck_timing_mode == "fixed_tuck_time"
                else self.jump_push_time + float(cfg.tuck_start_time_offset_s)
            )
            by_time = push & (self.jump_phase_time >= tuck_time)
        tuck = natural | by_time
        self.jump_phase[tuck] = WheelbipeJumpPhase.TUCK
        self.jump_tuck_event[tuck] = True
        active = self.jump_phase != WheelbipeJumpPhase.IDLE
        done = active & (self.jump_phase_time >= self.jump_duration)
        self.jump_exit_event[done] = True
        if cfg.enter_airborne_on_exit:
            self.jump_force_airborne_request[done] = True
        self.jump_phase[done] = WheelbipeJumpPhase.IDLE
        self.jump_phase_time[done] = 0.0
        self.jump_ref_phase[done] = 0.0
        self.jump_ref_vel_z[done] = 0.0
        self.jump_cooldown_time[done] = float(cfg.cooldown_s)

    def _start_landing_trajectory(
        self,
        sensors: WheelbipeStateMachineSensors,
        contact_started: np.ndarray,
    ) -> None:
        if not self.cfg.landing_trajectory_enabled:
            return
        height = np.asarray(sensors.base_pos_w, dtype=np.float64)[:, 2] - np.asarray(
            sensors.base_ground_height, dtype=np.float64
        )
        velocity = np.asarray(sensors.base_lin_vel_w, dtype=np.float64)[:, 2]
        duration = float(self.cfg.landing_trajectory_duration_s)
        end_velocity = float(self.cfg.landing_trajectory_end_vel_z)
        displacement = float(self.cfg.landing_height_target) - height
        start_velocity = 2.0 * displacement / duration - end_velocity
        acceleration = (end_velocity - start_velocity) / duration
        start = (
            contact_started
            & ~self.landing_traj_active
            & (velocity < -float(self.cfg.landing_trajectory_min_down_vel))
            & (
                height
                > float(self.cfg.landing_height_target)
                + float(self.cfg.landing_trajectory_min_height_margin)
            )
        )
        max_acc = self.cfg.landing_trajectory_max_abs_acc
        if max_acc is not None and float(max_acc) > 0.0:
            start &= np.abs(acceleration) <= float(max_acc)
        self.landing_traj_active[start] = True
        self.landing_traj_time[start] = 0.0
        self.landing_traj_start_height[start] = height[start]
        self.landing_traj_start_vel[start] = start_velocity[start]
        self.landing_traj_acc[start] = acceleration[start]
        self.landing_traj_height[start] = height[start]
        self.landing_traj_vel_z[start] = start_velocity[start]

    def _update_airborne(
        self,
        sensors: WheelbipeStateMachineSensors,
        *,
        wheel_contact: np.ndarray,
        geometry_low: np.ndarray,
        base_contact: np.ndarray,
        wheel_clearance: np.ndarray,
        finite: np.ndarray,
    ) -> None:
        if not self.cfg.airborne_enabled:
            self.airborne_state.fill(False)
            self.landing_state.fill(False)
            return
        body_height = np.asarray(sensors.base_pos_w, dtype=np.float64)[:, 2] - np.asarray(
            sensors.base_ground_height, dtype=np.float64
        )
        natural = (
            finite
            & ~self.stair_state
            & (body_height > float(self.cfg.body_airborne_height))
            & np.all(wheel_clearance > float(self.cfg.wheel_airborne_clearance), axis=1)
        )
        inactive = ~self.airborne_state
        self.airborne_enter_streak[natural & inactive] += 1
        self.airborne_enter_streak[~(natural & inactive)] = 0
        enter = inactive & (
            (self.airborne_enter_streak >= int(self.cfg.airborne_enter_steps))
            | self.jump_force_airborne_request
        )
        self.airborne_state[enter] = True
        self.airborne_elapsed[enter] = 0
        self.airborne_entry_command[enter] = np.asarray(sensors.commands, dtype=np.float64)[enter]
        self._sample_airborne_command_override(enter, sensors)
        self.airborne_elapsed[self.airborne_state] += 1
        self.airborne_elapsed[~self.airborne_state] = 0
        active = self.airborne_state & ~enter
        count = active[:, None] & wheel_contact
        # Force dropout while a wheel remains geometrically low freezes the
        # timer; lifting it resets the timer, matching pinned Airborne.
        freeze = active[:, None] & geometry_low & ~wheel_contact
        self.wheel_contact_steps[count] += 1
        self.wheel_contact_steps[~count & ~freeze] = 0
        self.base_contact_steps[active & base_contact] += 1
        # Pinned Airborne retains accumulated base-contact time across a
        # transient force dropout and clears it only after leaving the state.
        self.base_contact_steps[~active] = 0
        any_contact = np.any(wheel_contact, axis=1)
        self.contact_streak[any_contact] += 1
        self.contact_streak[~any_contact] = 0
        self.no_contact_streak[~any_contact] += 1
        self.no_contact_streak[any_contact] = 0
        landing_started = np.any(
            self.wheel_contact_steps >= int(self.cfg.landing_contact_steps), axis=1
        )
        self.landing_state[...] = self.airborne_state & landing_started
        trajectory_started = np.any(
            self.wheel_contact_steps >= int(self.cfg.landing_trajectory_start_steps), axis=1
        )
        self._start_landing_trajectory(sensors, self.airborne_state & trajectory_started)
        trajectory = self.landing_traj_active & self.airborne_state
        self.landing_traj_time[trajectory] += self._dt
        time = self.landing_traj_time
        self.landing_traj_height[trajectory] = (
            self.landing_traj_start_height
            + self.landing_traj_start_vel * time
            + 0.5 * self.landing_traj_acc * np.square(time)
        )[trajectory]
        self.landing_traj_vel_z[trajectory] = (
            self.landing_traj_start_vel + self.landing_traj_acc * time
        )[trajectory]
        trajectory_done = trajectory & (time >= float(self.cfg.landing_trajectory_duration_s))
        self.landing_traj_active[trajectory_done] = False
        wheel_exit_steps = int(self.cfg.landing_contact_steps) + int(self.cfg.landing_hold_steps)
        wheel_exit = np.any(self.wheel_contact_steps >= wheel_exit_steps, axis=1)
        base_exit = self.base_contact_steps >= int(self.cfg.base_contact_steps)
        max_exit = self.airborne_elapsed >= int(self.cfg.max_airborne_steps)
        exit_mask = self.airborne_state & (wheel_exit | base_exit | max_exit | self.stair_state)
        self.airborne_state[exit_mask] = False
        self.landing_state[exit_mask] = False
        self.landing_traj_active[exit_mask] = False
        self.wheel_contact_steps[exit_mask] = 0
        self.base_contact_steps[exit_mask] = 0
        self.airborne_command_override_active[exit_mask] = False
        self.airborne_command_override_values[exit_mask] = 0.0

    def _resolved_state(self) -> np.ndarray:
        state = np.full((self.num_envs,), WheelbipeMotionState.NORMAL, dtype=np.int8)
        state[self.airborne_state] = WheelbipeMotionState.AIRBORNE
        state[self.landing_state] = WheelbipeMotionState.LANDING
        state[self.step_hold_remaining > 0] = WheelbipeMotionState.STEP_UP
        state[self.stair_state] = WheelbipeMotionState.STAIR
        state[self.jump_phase == WheelbipeJumpPhase.PUSH] = WheelbipeMotionState.JUMP_PUSH
        state[self.jump_phase == WheelbipeJumpPhase.TUCK] = WheelbipeMotionState.JUMP_TUCK
        state[self.failure] = WheelbipeMotionState.RECOVER
        state[self.wall_event] = WheelbipeMotionState.WALL_BLOCKED
        return state

    def update(
        self,
        sensors_or_wheel_pos: WheelbipeStateMachineSensors | np.ndarray,
        terrain_height: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Advance one control step.

        ``update(wheel_pos, terrain_height)`` remains as an explicitly
        geometry-selected compatibility form for old direct callers.
        """

        if isinstance(sensors_or_wheel_pos, WheelbipeStateMachineSensors):
            if terrain_height is not None:
                raise ValueError("terrain_height cannot accompany an explicit sensor frame")
            sensors = sensors_or_wheel_pos
        else:
            if terrain_height is None:
                raise ValueError("legacy update requires terrain_height")
            sensors = WheelbipeStateMachineSensors.from_geometry(
                sensors_or_wheel_pos, terrain_height
            )
        sensors.validate(self.num_envs)
        self.timeout.fill(False)
        self.episode_elapsed += 1
        old_state = self.state.copy()
        wheel = np.asarray(sensors.wheel_pos_w, dtype=np.float64)
        wheel_ground = np.asarray(sensors.wheel_ground_height, dtype=np.float64)
        base_pos = np.asarray(sensors.base_pos_w, dtype=np.float64)
        base_vel = np.asarray(sensors.base_lin_vel_w, dtype=np.float64)
        finite = (
            np.all(np.isfinite(wheel), axis=(1, 2))
            & np.all(np.isfinite(wheel_ground), axis=1)
            & np.all(np.isfinite(base_pos), axis=1)
            & np.all(np.isfinite(base_vel), axis=1)
            & np.isfinite(np.asarray(sensors.base_ground_height, dtype=np.float64))
            & np.all(
                np.isfinite(np.asarray(sensors.wheel_forward_ground_height, dtype=np.float64)),
                axis=1,
            )
            & np.all(
                np.isfinite(np.asarray(sensors.wheel_stair_ground_height, dtype=np.float64)),
                axis=1,
            )
        )
        geometry_height = wheel[:, :, 2] - wheel_ground
        geometry_low = geometry_height < float(self.cfg.wheel_contact_height)
        wheel_clearance = geometry_height - float(self.cfg.wheel_radius)
        source = WheelbipeContactSource(sensors.contact_source)
        if source is WheelbipeContactSource.FORCE:
            wheel_force = np.asarray(sensors.wheel_contact_force_norm, dtype=np.float64)
            base_force = np.asarray(sensors.base_contact_force_norm, dtype=np.float64)
            finite &= np.all(np.isfinite(wheel_force), axis=1)
            finite &= np.all(np.isfinite(base_force), axis=1)
            wheel_contact = (
                wheel_force > float(self.cfg.wheel_contact_force_threshold)
            ) & geometry_low
            base_contact = np.any(base_force > float(self.cfg.base_contact_force_threshold), axis=1)
            history = sensors.wheel_contact_force_history_norm
            history_array = None if history is None else np.asarray(history, dtype=np.float64)
            if history_array is not None:
                if history_array.shape[1] != int(self.cfg.contact_history_steps):
                    raise ValueError(
                        "wheel_contact_force_history_norm history length must match "
                        "state_machine.contact_history_steps; got "
                        f"{history_array.shape[1]} and {self.cfg.contact_history_steps}"
                    )
                finite &= np.all(np.isfinite(history_array), axis=(1, 2))
            elif self.cfg.stair.enabled:
                raise ValueError(
                    "force contact_source with an enabled Stair machine requires "
                    "wheel_contact_force_history_norm"
                )
        else:
            wheel_contact = geometry_low & finite[:, None]
            base_contact = np.zeros((self.num_envs,), dtype=bool)
            history_array = None
        self.failure |= ~finite
        direction = self._direction(np.asarray(sensors.commands, dtype=np.float64))
        self._update_step_up(sensors, direction)
        self._update_stair(sensors, direction, history_array, geometry_low & finite[:, None])
        self._update_jump(sensors, wheel_clearance)
        self._update_airborne(
            sensors,
            wheel_contact=wheel_contact,
            geometry_low=geometry_low,
            base_contact=base_contact,
            wheel_clearance=wheel_clearance,
            finite=finite,
        )
        self.slope_state[...] = np.asarray(sensors.slope, dtype=bool) & ~self.stair_state
        self.state[...] = self._resolved_state()
        same = self.state == old_state
        self.elapsed[same & (self.state != WheelbipeMotionState.NORMAL)] += 1
        self.elapsed[~same | (self.state == WheelbipeMotionState.NORMAL)] = 0
        return {
            "state": self.state.copy(),
            "contact": np.any(wheel_contact, axis=1),
            "wheel_contact": wheel_contact.copy(),
            "base_contact": base_contact.copy(),
            "contact_source": np.full((self.num_envs,), source, dtype=np.int8),
            "airborne": self.airborne_state.copy(),
            "landing": self.landing_state.copy(),
            "recover": self.failure.copy(),
            "jump_phase": self.jump_phase.copy(),
            "jump_active": (self.jump_phase != WheelbipeJumpPhase.IDLE),
            "jump_trigger_event": self.jump_trigger_event.copy(),
            "jump_tuck_event": self.jump_tuck_event.copy(),
            "jump_exit_event": self.jump_exit_event.copy(),
            "jump_target_height": self.jump_target_height.copy(),
            "jump_ref_vel_z": self.jump_ref_vel_z.copy(),
            "jump_ref_phase": self.jump_ref_phase.copy(),
            "step_up": (self.step_hold_remaining > 0),
            "step_detect_event": self.step_detect_event.copy(),
            "wall_blocked": self.wall_event.copy(),
            "stair": self.stair_state.copy(),
            "stair_detect_event": self.stair_detect_event.copy(),
            "stair_success_event": self.stair_success_event.copy(),
            "stair_failure_event": self.stair_failure_event.copy(),
            "stair_timeout_event": self.stair_timeout_event.copy(),
            "slope": self.slope_state.copy(),
            "failure": self.failure.copy(),
            "timeout": self.timeout.copy(),
            "state_time": self.elapsed.astype(np.float64, copy=True),
            "state_time_s": self.elapsed.astype(np.float64) * self._dt,
            "landing_trajectory_active": self.landing_traj_active.copy(),
            "landing_trajectory_height": self.landing_traj_height.copy(),
            "landing_trajectory_vel_z": self.landing_traj_vel_z.copy(),
        }

    def apply_command_overrides(
        self, commands: np.ndarray, heights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply machine envelopes in pinned manager priority order."""

        out_commands = np.asarray(commands).copy()
        out_heights = np.asarray(heights).copy()
        airborne = self.airborne_state & ~self.landing_state
        landing = self.landing_state
        out_commands[airborne] *= float(self.cfg.command_scale_airborne)
        out_commands[landing] *= float(self.cfg.command_scale_landing)
        if self.cfg.airborne_height_override_enabled:
            out_heights[airborne] = float(self.cfg.airborne_height_target)
        if self.cfg.landing_height_override_enabled:
            out_heights[landing] = float(self.cfg.landing_height_target)
        command_override = self.airborne_command_override_active & self.airborne_state
        out_commands[command_override] = self.airborne_command_override_values[command_override]
        trajectory = self.landing_traj_active
        out_heights[trajectory] = self.landing_traj_height[trajectory]
        step = self.step_hold_remaining > 0
        out_heights[step] = self.step_height_command[step]
        out_heights[self.stair_state] = self.stair_height_command[self.stair_state]
        # JumpTakeoff intentionally leaves all ordinary commands intact.
        return out_commands, out_heights

    def control_mode_obs(self, *, state_dtype: np.dtype) -> np.ndarray:
        """Return pinned normal/stair/slope/recover/jump/target/phase tail."""

        out = np.zeros((self.num_envs, 7), dtype=state_dtype)
        jump = self.jump_phase != WheelbipeJumpPhase.IDLE
        recover = self.failure
        stair = self.stair_state
        slope = self.slope_state & ~stair
        normal = ~(jump | recover | stair | slope)
        out[:, 0] = normal
        out[:, 1] = stair
        out[:, 2] = slope
        out[:, 3] = recover
        out[:, 4] = jump
        out[jump, 5] = self.jump_target_height[jump]
        out[jump, 6] = self.jump_ref_phase[jump]
        return out


class _WheelbipeStateMachineSuper(Protocol):
    """Next cooperative owner in the concrete WheelBipe environment MRO."""

    def _update_commands(self, info: dict[str, Any]) -> None: ...

    def _compute_reward(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray: ...

    def _compute_truncated(self, state: Any) -> np.ndarray: ...


class WheelbipeStateMachineOwnerMixin:
    """Map stable ``SimBackend`` methods to the explicit sensor frame.

    Body IDs and force columns are resolved once after backend construction.
    Source-V14 owners reuse the physics-step contact history maintained by the
    environment owner; legacy owners retain a control-step ring.  Hot updates
    use only public backend methods and the cached terrain height callable.
    """

    _cfg: WheelbipeV14FlatCfg
    _backend: SimBackend
    _num_envs: int
    _np_dtype: np.dtype[Any]
    _source_semantics: bool
    _spawn: BaseSpawnManager
    _state_machine: WheelbipeStateMachine | None
    _reward_cfg: WheelbipeRewardConfig
    _terrain_surface_sample_height: Callable[[np.ndarray], np.ndarray] | None
    _source_contact_history: np.ndarray
    _source_contact_history_cursor: int
    _source_wheel_contact_columns: np.ndarray
    _native_leg_indices: np.ndarray
    _native_wheel_indices: np.ndarray

    def _state_machine_super(self) -> _WheelbipeStateMachineSuper:
        return cast(_WheelbipeStateMachineSuper, super())

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        terrain_commands = getattr(
            self._cfg,
            "terrain_commands",
            WheelbipeTerrainCommandConfig(),
        )
        if isinstance(terrain_commands, dict):
            terrain_commands = WheelbipeTerrainCommandConfig(**terrain_commands)
            setattr(self._cfg, "terrain_commands", terrain_commands)
        if not isinstance(terrain_commands, WheelbipeTerrainCommandConfig):
            raise ValueError("terrain_commands must be WheelbipeTerrainCommandConfig")
        terrain_commands.validate()
        self._terrain_commands_cfg = terrain_commands
        self._init_terrain_command_owner()
        machine = self._state_machine
        if machine is None:
            return
        reward_cfg = machine.cfg.airborne_reward
        if reward_cfg.enabled and not self._source_semantics:
            raise ValueError(
                "WheelBipe Airborne reward semantics require training_semantics='source_v14'"
            )
        self._state_machine_wheel_body_ids = self._backend.get_body_ids(SOURCE_V14_WHEEL_BODY_NAMES)
        contact_names = tuple(
            dict.fromkeys(
                (
                    *SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES,
                    *SOURCE_V14_WHEEL_BODY_NAMES,
                    *SOURCE_V14_RESET_CONTACT_BODY_NAMES,
                )
            )
        )
        self._state_machine_contact_body_ids = np.asarray(
            self._backend.get_body_ids(contact_names), dtype=np.int32
        )
        contact_columns = {name: index for index, name in enumerate(contact_names)}
        self._state_machine_wheel_contact_columns = np.asarray(
            [contact_columns[name] for name in SOURCE_V14_WHEEL_BODY_NAMES],
            dtype=np.intp,
        )
        self._state_machine_undesired_contact_columns = np.asarray(
            [contact_columns[name] for name in SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES],
            dtype=np.intp,
        )
        history = int(machine.cfg.contact_history_steps)
        if self._source_semantics and history != 3:
            raise ValueError(
                "source_v14 state-machine contact history is fixed at three "
                f"physics samples; got contact_history_steps={history}"
            )
        self._state_machine_wheel_force_history = np.zeros(
            (self._num_envs, history, 2), dtype=self._np_dtype
        )
        self._state_machine_history_generation = machine.reset_generation.copy()
        self._state_machine_rear2_pos_indices = np.zeros((0,), dtype=np.intp)
        if reward_cfg.enabled and float(reward_cfg.joint_pos_limits_weight) != 0.0:
            self._state_machine_rear2_pos_indices = np.asarray(
                self._backend.get_joint_dof_pos_indices(("left_rear2_joint", "right_rear2_joint")),
                dtype=np.intp,
            )
            if self._state_machine_rear2_pos_indices.shape != (2,):
                raise RuntimeError(
                    "Airborne joint-limit reward requires exactly two rear2 DOF indices; "
                    f"got {self._state_machine_rear2_pos_indices.shape}"
                )
        self._state_machine_airborne_terrain_type = np.zeros(
            (len(self._terrain_command_type_names),), dtype=bool
        )
        if machine.cfg.airborne_command_resample.enabled:
            if not self._terrain_commands_cfg.enabled:
                raise ValueError(
                    "Airborne terrain command resampling requires enabled terrain_commands "
                    "metadata owner"
                )
            available = {name: index for index, name in enumerate(self._terrain_command_type_names)}
            matched = []
            for name in machine.cfg.airborne_command_resample.terrain_names:
                type_id = available.get(name)
                if type_id is not None:
                    self._state_machine_airborne_terrain_type[type_id] = True
                    matched.append(name)
            if not matched:
                raise ValueError(
                    "Airborne terrain command profiles do not match generated terrain names: "
                    f"requested={machine.cfg.airborne_command_resample.terrain_names!r}, "
                    f"available={self._terrain_command_type_names!r}"
                )

    def _init_terrain_command_owner(self) -> None:
        """Resolve generated-terrain type metadata once on the cold path."""

        n = self._num_envs
        self._terrain_command_type_names: tuple[str, ...] = ()
        self._terrain_command_type_ids_grid = np.zeros((0, 0), dtype=np.int32)
        self._terrain_command_profiles_by_type: tuple[
            WheelbipeTerrainCommandProfileConfig | None, ...
        ] = ()
        self._terrain_command_last_type_ids = np.full((n,), -1, dtype=np.int32)
        self._terrain_command_last_generation = np.full((n,), -1, dtype=np.int64)
        self._terrain_command_override_values = np.zeros((n, 3), dtype=self._np_dtype)
        self._terrain_command_override_masks = np.zeros((n, 3), dtype=bool)
        self._terrain_command_yaw_heading_values = np.zeros((n,), dtype=self._np_dtype)
        self._terrain_command_yaw_heading_valid = np.zeros((n,), dtype=bool)
        self._terrain_command_yaw_non_heading_values = np.zeros((n,), dtype=self._np_dtype)
        self._terrain_command_yaw_non_heading_valid = np.zeros((n,), dtype=bool)
        self._terrain_command_height_values = np.zeros((n,), dtype=self._np_dtype)
        self._terrain_command_height_valid = np.zeros((n,), dtype=bool)
        self._terrain_command_rng = GLOBAL_NUMPY_RANDOM
        if not self._terrain_commands_cfg.enabled:
            return
        if not self._source_semantics:
            raise ValueError("terrain_commands requires training_semantics='source_v14'")
        if not isinstance(self._spawn, TerrainSpawnManager):
            raise ValueError("enabled terrain_commands requires a generated-terrain spawn owner")
        spawn_data = self._backend.get_terrain_spawn_data()
        if spawn_data is None:
            raise ValueError("enabled terrain_commands requires BackendTerrainSpawnData")
        type_names = tuple(spawn_data.terrain_type_names)
        type_ids = np.asarray(spawn_data.terrain_type_ids, dtype=np.int32)
        if not type_names or type_ids.ndim != 2:
            raise ValueError(
                "enabled terrain_commands requires generated terrain_type_names and a "
                "two-dimensional terrain_type_ids grid"
            )
        if type_ids.shape != self._spawn.terrain_grid_shape:
            raise ValueError(
                "terrain type metadata must match TerrainSpawnManager grid: "
                f"{type_ids.shape} != {self._spawn.terrain_grid_shape}"
            )
        if np.any(type_ids < 0) or np.any(type_ids >= len(type_names)):
            raise ValueError("terrain_type_ids contains an out-of-range generated terrain id")
        unknown = sorted(set(self._terrain_commands_cfg.profiles) - set(type_names))
        if unknown:
            raise ValueError(
                "terrain_commands contains profiles absent from the generated terrain: "
                f"{unknown}; available={type_names!r}"
            )
        self._terrain_command_type_names = type_names
        self._terrain_command_type_ids_grid = type_ids.copy()
        self._terrain_command_profiles_by_type = tuple(
            self._terrain_commands_cfg.profiles.get(name) for name in type_names
        )

    def _terrain_command_type_ids(
        self,
        env_ids: np.ndarray | None = None,
        *,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._terrain_commands_cfg.enabled:
            ids = np.arange(self._num_envs, dtype=np.intp) if env_ids is None else env_ids
            return np.full((len(ids),), -1, dtype=np.int32)
        ids = (
            np.arange(self._num_envs, dtype=np.intp)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.intp).reshape(-1)
        )
        spawn = self._spawn
        if not isinstance(spawn, TerrainSpawnManager):
            raise RuntimeError(
                "enabled terrain command owner lost its TerrainSpawnManager contract"
            )
        if positions is None:
            rows, cols = spawn.terrain_cells_for(ids)
        else:
            current = np.asarray(positions, dtype=self._np_dtype)
            if current.ndim != 2 or current.shape[0] != ids.size:
                raise ValueError(
                    "runtime terrain positions must have one row per env id; got "
                    f"positions={current.shape}, env_ids={ids.shape}"
                )
            rows, cols = spawn.terrain_cells_at_positions(current)
        return self._terrain_command_type_ids_grid[rows, cols]

    def _sample_terrain_ranges(
        self,
        ranges: tuple[tuple[float, float], ...],
        count: int,
    ) -> np.ndarray:
        segments = np.asarray(ranges, dtype=np.float64).reshape(-1, 2)
        widths = np.maximum(segments[:, 1] - segments[:, 0], 0.0)
        probabilities = (
            np.full((len(segments),), 1.0 / len(segments))
            if float(np.sum(widths)) <= 0.0
            else widths / np.sum(widths)
        )
        selected = self._terrain_command_rng.choice(len(segments), size=int(count), p=probabilities)
        return self._terrain_command_rng.uniform(
            segments[selected, 0], segments[selected, 1], size=int(count)
        ).astype(self._np_dtype, copy=False)

    def _source_terrain_reset_profile(self, env_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Return reset flags resolved from the assigned generated-terrain cells."""

        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        zeros = np.zeros((ids.size,), dtype=bool)
        if not self._terrain_commands_cfg.enabled or ids.size == 0:
            return {
                "disable_predefined_reset_air": zeros.copy(),
                "disable_predefined_reset_ground": zeros.copy(),
                "reset_heading_axis_aligned_only": zeros.copy(),
            }
        type_ids = self._terrain_command_type_ids(ids)
        profiles = [self._terrain_command_profiles_by_type[int(type_id)] for type_id in type_ids]
        return {
            "disable_predefined_reset_air": np.asarray(
                [bool(profile and profile.disable_predefined_reset_air) for profile in profiles]
            ),
            "disable_predefined_reset_ground": np.asarray(
                [bool(profile and profile.disable_predefined_reset_ground) for profile in profiles]
            ),
            "reset_heading_axis_aligned_only": np.asarray(
                [bool(profile and profile.reset_heading_axis_aligned_only) for profile in profiles]
            ),
        }

    def _resample_normal_commands_after_special_disable(
        self,
        sampled: dict[str, np.ndarray],
        local_ids: np.ndarray,
        current_yaw: np.ndarray,
    ) -> None:
        if local_ids.size == 0:
            return
        commands = np.asarray(sampled["commands"], dtype=self._np_dtype)
        low = np.asarray(self._cfg.commands.vel_limit[0], dtype=self._np_dtype)
        high = np.asarray(self._cfg.commands.vel_limit[1], dtype=self._np_dtype)
        commands[local_ids] = self._terrain_command_rng.uniform(
            np.minimum(low, high),
            np.maximum(low, high),
            size=(local_ids.size, 3),
        )
        commands[local_ids, 1] = 0.0
        heading = np.asarray(sampled["is_heading_env"], dtype=bool)
        heading[local_ids] = self._terrain_command_rng.random(local_ids.size) <= float(
            self._cfg.commands.rel_heading_envs
        )
        bounds = np.sort(np.asarray(self._cfg.commands.heading_range, dtype=np.float64))
        heading_targets = np.asarray(sampled["heading_commands"], dtype=self._np_dtype)
        heading_targets[local_ids] = self._terrain_command_rng.uniform(
            bounds[0], bounds[1], size=local_ids.size
        )
        yaw_for_rows = np.asarray(current_yaw, dtype=self._np_dtype)[local_ids]
        heading_local = local_ids[heading[local_ids]]
        if heading_local.size:
            heading_yaw = yaw_for_rows[heading[local_ids]]
            error = (heading_targets[heading_local] - heading_yaw + np.pi) % (2.0 * np.pi) - np.pi
            commands[heading_local, 2] = np.clip(
                float(self._cfg.commands.heading_control_stiffness) * error,
                min(float(low[2]), float(high[2])),
                max(float(low[2]), float(high[2])),
            )
        np.asarray(sampled["special_mode_id"])[local_ids] = -1
        np.asarray(sampled["is_standing_env"])[local_ids] = False

    def _source_apply_terrain_command_profile(
        self,
        env_ids: np.ndarray,
        sampled: dict[str, np.ndarray],
        height_commands: np.ndarray,
        *,
        current_yaw: np.ndarray,
        force_resample: bool,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Apply reset-time overrides from each env's assigned terrain cell."""

        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        return self._apply_terrain_command_profile_for_type_ids(
            ids,
            sampled,
            height_commands,
            self._terrain_command_type_ids(ids),
            current_yaw=current_yaw,
            force_resample=force_resample,
        )

    def _apply_terrain_command_profile_for_type_ids(
        self,
        env_ids: np.ndarray,
        sampled: dict[str, np.ndarray],
        height_commands: np.ndarray,
        terrain_type_ids: np.ndarray,
        *,
        current_yaw: np.ndarray,
        force_resample: bool,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Apply cached source overrides for explicit cold-path terrain IDs."""

        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        heights = np.asarray(height_commands, dtype=self._np_dtype).copy()
        if not self._terrain_commands_cfg.enabled or ids.size == 0:
            return sampled, heights
        commands = np.asarray(sampled["commands"], dtype=self._np_dtype)
        if commands.shape != (ids.size, 3) or heights.shape != (ids.size,):
            raise ValueError(
                "terrain command hook requires subset-shaped commands/heights; got "
                f"{commands.shape} and {heights.shape} for {ids.size} envs"
            )
        type_ids = np.asarray(terrain_type_ids, dtype=np.int32).reshape(-1)
        if type_ids.shape != (ids.size,):
            raise ValueError(
                "terrain command type ids must have one value per env id; got "
                f"{type_ids.shape} for {ids.size} envs"
            )
        if np.any(type_ids < 0) or np.any(type_ids >= len(self._terrain_command_type_names)):
            raise ValueError("terrain command type ids contain an out-of-range profile id")
        changed = type_ids != self._terrain_command_last_type_ids[ids]
        resample = changed | bool(force_resample) | ~self._terrain_command_height_valid[ids]
        profile_for_row = [
            self._terrain_command_profiles_by_type[int(type_id)] for type_id in type_ids
        ]

        disable_special = np.asarray(
            [bool(profile and profile.disable_special_mode) for profile in profile_for_row]
        )
        special = np.asarray(sampled.get("special_mode_id", np.full(ids.size, -1)))
        special_rows = np.flatnonzero(disable_special & (special >= 0))
        self._resample_normal_commands_after_special_disable(
            sampled,
            special_rows,
            np.asarray(current_yaw, dtype=self._np_dtype),
        )
        if "jump_takeoff_request" in sampled:
            np.asarray(sampled["jump_takeoff_request"])[disable_special] = False

        resample_rows = np.flatnonzero(resample)
        if resample_rows.size:
            if len(self._cfg.height_range) != 2:
                raise ValueError(
                    f"height_range must contain two values, got {self._cfg.height_range!r}"
                )
            base_height = _range_spec(
                "height_range",
                ((float(self._cfg.height_range[0]), float(self._cfg.height_range[1])),),
            )
            assert base_height is not None
            for type_id in np.unique(type_ids[resample_rows]):
                local = resample_rows[type_ids[resample_rows] == type_id]
                targets = ids[local]
                profile = self._terrain_command_profiles_by_type[int(type_id)]
                height_ranges = (
                    profile.height_ranges
                    if profile is not None and profile.height_ranges is not None
                    else base_height
                )
                self._terrain_command_height_values[targets] = self._sample_terrain_ranges(
                    height_ranges, local.size
                )
                self._terrain_command_height_valid[targets] = True
                self._terrain_command_override_masks[targets] = False
                self._terrain_command_override_values[targets] = 0.0
                self._terrain_command_yaw_heading_valid[targets] = False
                self._terrain_command_yaw_heading_values[targets] = 0.0
                self._terrain_command_yaw_non_heading_valid[targets] = False
                self._terrain_command_yaw_non_heading_values[targets] = 0.0
                if profile is None:
                    continue
                for column, ranges in (
                    (0, profile.lin_vel_x_ranges),
                    (1, profile.lin_vel_y_ranges),
                ):
                    if ranges is not None:
                        self._terrain_command_override_values[targets, column] = (
                            self._sample_terrain_ranges(ranges, local.size)
                        )
                        self._terrain_command_override_masks[targets, column] = True
                heading_yaw_ranges = profile.ang_vel_z_heading_ranges
                if heading_yaw_ranges is not None:
                    self._terrain_command_yaw_heading_values[targets] = self._sample_terrain_ranges(
                        heading_yaw_ranges, local.size
                    )
                    self._terrain_command_yaw_heading_valid[targets] = True
                non_heading_yaw_ranges = profile.ang_vel_z_non_heading_ranges
                if non_heading_yaw_ranges is not None:
                    self._terrain_command_yaw_non_heading_values[targets] = (
                        self._sample_terrain_ranges(non_heading_yaw_ranges, local.size)
                    )
                    self._terrain_command_yaw_non_heading_valid[targets] = True
            self._terrain_command_last_type_ids[ids[resample]] = type_ids[resample]
            if force_resample:
                self._terrain_command_last_generation[ids] = 0

        heights[:] = self._terrain_command_height_values[ids]
        masks = self._terrain_command_override_masks[ids]
        values = self._terrain_command_override_values[ids]
        commands[masks] = values[masks]
        is_heading = np.asarray(sampled.get("is_heading_env", False), dtype=bool)
        if is_heading.shape == ():
            is_heading = np.full((ids.size,), bool(is_heading), dtype=bool)
        if is_heading.shape != (ids.size,):
            raise ValueError(
                "terrain command heading mask must have one value per env id; got "
                f"{is_heading.shape} for {ids.size} envs"
            )
        heading_yaw = is_heading & self._terrain_command_yaw_heading_valid[ids]
        non_heading_yaw = (~is_heading) & self._terrain_command_yaw_non_heading_valid[ids]
        commands[heading_yaw, 2] = self._terrain_command_yaw_heading_values[ids][heading_yaw]
        commands[non_heading_yaw, 2] = self._terrain_command_yaw_non_heading_values[ids][
            non_heading_yaw
        ]
        for row, profile in enumerate(profile_for_row):
            if (
                profile is not None
                and profile.reset_heading_axis_aligned_only
                and not is_heading[row]
                and not non_heading_yaw[row]
            ):
                commands[row, 2] = 0.0
        if bool(self._terrain_commands_cfg.force_forward):
            # Keep each terrain profile's sampled speed magnitude while
            # removing the signed backward branch.  This is intentionally
            # applied after profile overrides and heading handling so it does
            # not alter the source training command owner.
            commands[:, 0] = np.abs(commands[:, 0])
            commands[:, 0] = np.maximum(commands[:, 0], np.finfo(commands.dtype).eps)
        sampled["commands"] = commands
        return sampled, heights

    def _apply_terrain_command_subset(
        self,
        info: dict[str, Any],
        env_ids: np.ndarray,
        terrain_type_ids: np.ndarray,
        *,
        force_resample: bool,
    ) -> None:
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        if ids.size == 0:
            return
        keys = (
            "commands",
            "heading_commands",
            "is_standing_env",
            "is_heading_env",
            "special_mode_id",
            "jump_takeoff_request",
            "steps",
        )
        sampled = {
            key: np.asarray(info[key])[ids].copy()
            for key in keys
            if key in info and np.asarray(info[key]).shape != ()
        }
        heights_all = np.asarray(info["height_commands"], dtype=self._np_dtype)
        sampled, heights = self._apply_terrain_command_profile_for_type_ids(
            ids,
            sampled,
            heights_all[ids],
            terrain_type_ids,
            current_yaw=np_yaw_from_quat(
                np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype)
            )[ids],
            force_resample=force_resample,
        )
        for key, values in sampled.items():
            if key in info and np.asarray(info[key]).shape != ():
                np.asarray(info[key])[ids] = values
        heights_all[ids] = heights
        info["height_commands"] = heights_all

    def _update_commands(self, info: dict[str, Any]) -> None:
        self._state_machine_super()._update_commands(info)
        if not self._terrain_commands_cfg.enabled:
            return
        generation = np.asarray(
            info.get(
                "command_resample_generation",
                np.zeros((self._num_envs,), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        current_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        type_ids = self._terrain_command_type_ids(positions=current_pos)
        force = (generation != self._terrain_command_last_generation) | (
            type_ids != self._terrain_command_last_type_ids
        )
        forced_ids = np.flatnonzero(force)
        stable_ids = np.flatnonzero(~force)
        self._apply_terrain_command_subset(
            info,
            forced_ids,
            type_ids[forced_ids],
            force_resample=True,
        )
        self._apply_terrain_command_subset(
            info,
            stable_ids,
            type_ids[stable_ids],
            force_resample=False,
        )
        self._terrain_command_last_generation[:] = generation

    def _state_machine_terrain_height(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=self._np_dtype)
        if points.shape[-1:] != (2,):
            raise ValueError(f"terrain query points must end in width 2, got {points.shape}")
        shape = points.shape[:-1]
        if self._terrain_surface_sample_height is None:
            return np.zeros(shape, dtype=self._np_dtype)
        result = np.asarray(
            self._terrain_surface_sample_height(points.reshape(-1, 2)),
            dtype=self._np_dtype,
        )
        expected = (int(np.prod(shape, dtype=np.int64)),)
        if result.shape != expected:
            raise RuntimeError(
                "cached terrain sampler returned the wrong shape: "
                f"queries={points.shape}, result={result.shape}"
            )
        return result.reshape(shape)

    def _state_machine_force_frame(
        self,
    ) -> tuple[
        WheelbipeContactSource,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        machine = self._state_machine
        if machine is None:
            raise RuntimeError("contact-force frame requires an enabled state machine")
        try:
            contact_force = np.asarray(
                self._backend.get_body_contact_force_norm(self._state_machine_contact_body_ids),
                dtype=self._np_dtype,
            )
        except NotImplementedError as exc:
            if not machine.cfg.allow_geometric_contact_fallback:
                raise RuntimeError(
                    "WheelBipe state machine requires the public body contact-force "
                    "contract; select allow_geometric_contact_fallback=true explicitly "
                    "for a bounded geometry-only owner"
                ) from exc
            return WheelbipeContactSource.GEOMETRY, None, None, None
        expected = (self._num_envs, len(self._state_machine_contact_body_ids))
        if contact_force.shape != expected:
            raise RuntimeError(
                "get_body_contact_force_norm(state-machine bodies) must return "
                f"{expected}, got {contact_force.shape}"
            )
        wheel_force = contact_force[:, self._state_machine_wheel_contact_columns]
        undesired_force = contact_force[:, self._state_machine_undesired_contact_columns]
        history = self._state_machine_wheel_force_history
        if self._source_semantics:
            source_history = self._source_contact_history
            expected_source = (
                3,
                self._num_envs,
                # The source V14 reset-contact contract may include bodies
                # (gimbal/guide links) that are not part of the state-machine
                # contact view.  The history is indexed by the owner's full
                # source contact table; only the wheel columns below are
                # projected into the state-machine frame.  Use the cached
                # table width as the source of truth here: this helper is
                # also exercised in isolation by contract tests that build
                # the history buffer without running the full environment
                # materialization that resolves ``_source_contact_body_ids``.
                source_history.shape[2],
            )
            if source_history.shape != expected_source:
                raise RuntimeError(
                    "source_v14 owner contact history must have shape "
                    f"{expected_source}, got {source_history.shape}"
                )
            history[:, 0] = wheel_force
            # The owner captures before each physics substep.  At this point
            # cursor-1 and cursor-2 are the two preceding physics frames; the
            # direct public-backend read above is the just-finished frame.
            for lag in range(1, history.shape[1]):
                source_index = (
                    int(self._source_contact_history_cursor) - lag
                ) % source_history.shape[0]
                history[:, lag] = source_history[source_index][
                    :, self._source_wheel_contact_columns
                ]
        else:
            history[:, 1:] = history[:, :-1]
            history[:, 0] = wheel_force
        return WheelbipeContactSource.FORCE, wheel_force, undesired_force, history.copy()

    def _build_state_machine_sensors(self, info: dict[str, Any]) -> WheelbipeStateMachineSensors:
        machine = self._state_machine
        if machine is None:
            raise RuntimeError("sensor frame requires an enabled state machine")
        changed = self._state_machine_history_generation != machine.reset_generation
        if np.any(changed):
            self._state_machine_wheel_force_history[changed] = 0.0
            self._state_machine_history_generation[changed] = machine.reset_generation[changed]
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        base_vel = np.asarray(self._backend.get_base_lin_vel(), dtype=self._np_dtype)
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype)
        wheel_pos = np.asarray(
            self._backend.get_body_pos_w(self._state_machine_wheel_body_ids),
            dtype=self._np_dtype,
        )
        commands = np.asarray(
            info.get("commands", np.zeros((self._num_envs, 3))),
            dtype=self._np_dtype,
        )
        heights = np.asarray(
            info.get(
                "height_commands",
                np.full(
                    (self._num_envs,),
                    float(self._reward_cfg.base_height_target),
                    dtype=self._np_dtype,
                ),
            ),
            dtype=self._np_dtype,
        )
        yaw = np_yaw_from_quat(base_quat)
        direction = np.where(commands[:, 0] < 0.0, -1.0, 1.0)
        heading = np.stack((np.cos(yaw), np.sin(yaw)), axis=1) * direction[:, None]
        step_points = wheel_pos[:, :, :2] + (
            float(machine.cfg.step_up.forward_offset) * heading[:, None, :]
        )
        stair_points = wheel_pos[:, :, :2] + (
            float(machine.cfg.stair.scan_offset) * heading[:, None, :]
        )
        wheel_ground = self._state_machine_terrain_height(wheel_pos[:, :, :2])
        base_ground = self._state_machine_terrain_height(base_pos[:, :2])
        forward_ground = self._state_machine_terrain_height(step_points)
        stair_ground = self._state_machine_terrain_height(stair_points)
        spatial_difference = np.mean(forward_ground - stair_ground, axis=1)
        slope = (np.abs(spatial_difference) >= float(machine.cfg.slope_height_difference_min)) & (
            np.abs(spatial_difference) < float(machine.cfg.step_up.step_height_min)
        )
        source, wheel_force, base_force, history = self._state_machine_force_frame()
        request = np.asarray(
            info.get("jump_takeoff_request", np.zeros((self._num_envs,), dtype=bool)),
            dtype=bool,
        )
        terrain_profile_id = None
        if machine.cfg.airborne_command_resample.enabled:
            type_ids = self._terrain_command_type_ids(positions=base_pos)
            terrain_profile_id = np.where(
                self._state_machine_airborne_terrain_type[type_ids],
                type_ids,
                -1,
            ).astype(np.int16, copy=False)
        return WheelbipeStateMachineSensors(
            wheel_pos_w=wheel_pos,
            wheel_ground_height=wheel_ground,
            base_pos_w=base_pos,
            base_quat_w=base_quat,
            base_lin_vel_w=base_vel,
            base_ground_height=base_ground,
            wheel_forward_ground_height=forward_ground,
            wheel_stair_ground_height=stair_ground,
            commands=commands,
            height_commands=heights,
            jump_request=request,
            slope=slope,
            contact_source=source,
            wheel_contact_force_norm=wheel_force,
            base_contact_force_norm=base_force,
            wheel_contact_force_history_norm=history,
            terrain_profile_id=terrain_profile_id,
        )

    def _update_state_machine(self, info: dict[str, Any]) -> None:
        machine = self._state_machine
        if machine is None:
            return
        sensors = self._build_state_machine_sensors(info)
        transition = machine.update(sensors)
        commands, heights = machine.apply_command_overrides(
            sensors.commands, sensors.height_commands
        )
        info["commands"] = commands.astype(self._np_dtype, copy=False)
        info["height_commands"] = heights.astype(self._np_dtype, copy=False)
        info["state_machine_state"] = transition["state"]
        info["state_machine_contact"] = transition["contact"]
        info["state_machine_wheel_contact"] = transition["wheel_contact"]
        info["state_machine_base_contact"] = transition["base_contact"]
        info["state_machine_contact_source"] = transition["contact_source"]
        info["state_machine_failure"] = transition["failure"]
        info["state_machine_timeout"] = transition["timeout"]
        info["state_machine_state_time"] = transition["state_time_s"].astype(self._np_dtype)
        info["state_machine_airborne"] = transition["airborne"]
        info["state_machine_landing"] = transition["landing"]
        info["state_machine_jump_phase"] = transition["jump_phase"]
        info["state_machine_jump_target_height"] = transition["jump_target_height"].astype(
            self._np_dtype
        )
        info["state_machine_jump_ref_vel_z"] = transition["jump_ref_vel_z"].astype(self._np_dtype)
        info["state_machine_step_up"] = transition["step_up"]
        info["state_machine_stair"] = transition["stair"]
        info["state_machine_wall_blocked"] = transition["wall_blocked"]
        info["state_machine_stair_success_event"] = transition["stair_success_event"]
        info["state_machine_stair_failure_event"] = transition["stair_failure_event"]
        # Cache the already-read pose frame for the reward hook.  This avoids
        # an extra backend read and keeps the heading-centering term on the
        # exact same control frame as the transition timers.
        info["state_machine_base_pos_w"] = sensors.base_pos_w
        info["state_machine_base_quat_w"] = sensors.base_quat_w
        info["state_machine_wheel_pos_w"] = sensors.wheel_pos_w
        info["control_mode_obs"] = machine.control_mode_obs(state_dtype=self._np_dtype)
        if np.any(transition["jump_trigger_event"]):
            request = np.asarray(sensors.jump_request, dtype=bool).copy()
            request[transition["jump_trigger_event"]] = False
            info["jump_takeoff_request"] = request

    def _state_machine_source_reward_raw(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return only source terms changed by the public v1 configurations."""

        n = self._num_envs
        dtype = self._np_dtype
        machine = self._state_machine
        if machine is None:
            raise RuntimeError("Airborne reward raw terms require an enabled state machine")
        reward_cfg = machine.cfg.airborne_reward
        commands = np.asarray(info["commands"], dtype=dtype)
        actions = np.asarray(info["current_actions"], dtype=dtype)
        last_actions = np.asarray(info["last_actions"], dtype=dtype)
        previous_actions = np.asarray(info["previous_actions"], dtype=dtype)
        qacc = np.asarray(info["qacc"], dtype=dtype)
        torque_native = np.asarray(info["torques"], dtype=dtype)
        policy_torque = np.concatenate(
            (
                torque_native[:, self._native_leg_indices],
                torque_native[:, self._native_wheel_indices],
            ),
            axis=1,
        )
        gravity = np.asarray(projected_gravity, dtype=dtype)
        velocity = np.asarray(linvel, dtype=dtype)
        joint_velocity = np.asarray(dof_vel, dtype=dtype)
        action_second_difference = actions - 2.0 * last_actions + previous_actions
        wheel_power = policy_torque[:, 4:] * joint_velocity[:, 4:]

        # The source privileged stream keeps clipped world-frame root z, but
        # rough/state-machine height rewards subtract terrain after that clip.
        # Reuse the reward-specific signal materialized by the env owner;
        # feeding ``observed_height`` here would reintroduce absolute terrain
        # elevation only in the v1 reward-delta path.
        reward_height = info.get("relative_observed_height", info["observed_height"])
        height_error = np.asarray(reward_height, dtype=dtype).reshape(n) - np.asarray(
            info["height_commands"], dtype=dtype
        ).reshape(n)
        command_x_sq = np.square(commands[:, 0])
        orientation_y_scale = float(self._reward_cfg.orientation_y_amplitude) * np.exp(
            -command_x_sq / float(self._reward_cfg.orientation_y_sigma)
        ) + float(self._reward_cfg.orientation_y_bias)
        _roll, pitch = np_roll_pitch_from_quat(
            np.asarray(info["state_machine_base_quat_w"], dtype=dtype)
        )
        tracking_command_x = commands[:, 0].copy()
        tracking_command_x[
            np.abs(tracking_command_x) < float(self._reward_cfg.stand_still_deadzone)
        ] = 0.0
        lin_error = tracking_command_x - velocity[:, 0] * np.cos(pitch)
        lin_error_limited = np.clip(
            lin_error,
            -float(self._reward_cfg.lin_vel_error_constraint),
            float(self._reward_cfg.lin_vel_error_constraint),
        )

        raw = {
            "action_rate": np.sum(np.square(actions - last_actions), axis=1),
            "action_smoothness_leg": np.sum(np.square(action_second_difference[:, :4]), axis=1),
            "action_smoothness_wheel": np.sum(np.square(action_second_difference[:, 4:]), axis=1),
            "leg_joint_acc": np.sum(np.square(qacc[:, :4]), axis=1),
            "leg_joint_vel": np.sum(np.square(joint_velocity[:, :4]), axis=1),
            "wheel_acc": np.sum(np.square(qacc[:, 4:]), axis=1),
            "wheel_vel": np.sum(np.square(joint_velocity[:, 4:]), axis=1),
            "joint_torque": np.sum(np.square(policy_torque), axis=1),
            "wheel_power": np.sum(np.maximum(wheel_power, 0.0), axis=1),
            "track_lin_vel_xy": np.exp(
                -np.square(lin_error_limited) / float(self._reward_cfg.lin_vel_sigma)
            ),
            "undesired_contact": np.asarray(info["undesired_contact"], dtype=dtype),
            "flat_orientation_y_v": np.square(orientation_y_scale * gravity[:, 0]),
            "termination": np.asarray(info["terminated"], dtype=dtype),
            "track_height_square": np.square(
                height_error * float(self._reward_cfg.height_square_sigma)
            ),
            # The pinned config sets this registered multiplier to zero, but
            # its base reward is absent from the active V14 reward graph.
            "foot_bound_square": np.zeros((n,), dtype=dtype),
        }
        required = set(reward_cfg.base_scale_overrides) | set(reward_cfg.airborne_scale_multipliers)
        missing = required - set(raw)
        if missing:
            raise RuntimeError(f"missing source Airborne reward terms: {sorted(missing)}")
        return raw

    def _state_machine_airborne_additions(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute the five non-zero reward additions enabled by Flat/Rough-v1."""

        machine = self._state_machine
        if machine is None:
            raise RuntimeError("Airborne reward additions require an enabled state machine")
        cfg = machine.cfg.airborne_reward
        dtype = self._np_dtype
        airborne = machine.airborne_state
        wheel_time = machine.wheel_contact_steps.astype(np.float64) * float(machine.cfg.control_dt)
        base_time = machine.base_contact_steps.astype(np.float64) * float(machine.cfg.control_dt)

        rear2_pos = np.asarray(self._backend.get_dof_pos(), dtype=dtype)[
            :, self._state_machine_rear2_pos_indices
        ]
        rear2_pos = (rear2_pos + np.pi) % (2.0 * np.pi) - np.pi
        span = float(cfg.rear2_joint_upper) - float(cfg.rear2_joint_lower)
        soft_lower = float(cfg.rear2_joint_lower) + span * float(cfg.rear2_lower_boundary_ratio)
        soft_upper = float(cfg.rear2_joint_upper) - span * float(cfg.rear2_upper_boundary_ratio)
        joint_limits = airborne * np.sum(
            np.maximum(soft_lower - rear2_pos, 0.0) + np.maximum(rear2_pos - soft_upper, 0.0),
            axis=1,
        )

        base_pos = np.asarray(info["state_machine_base_pos_w"], dtype=dtype)
        base_quat = np.asarray(info["state_machine_base_quat_w"], dtype=dtype)
        wheel_pos = np.asarray(info["state_machine_wheel_pos_w"], dtype=dtype)
        yaw = np_yaw_from_quat(base_quat)
        relative = wheel_pos - base_pos[:, None, :]
        wheel_heading_x = (
            np.cos(yaw)[:, None] * relative[:, :, 0] + np.sin(yaw)[:, None] * relative[:, :, 1]
        )
        heading_window = (
            airborne
            & np.all(wheel_time < float(cfg.wheel_heading_contact_duration_s), axis=1)
            & (base_time < float(cfg.wheel_heading_base_contact_duration_s))
            & np.all(relative[:, :, 2] < float(cfg.wheel_heading_z_max), axis=1)
        )
        heading = heading_window * np.mean(
            np.exp(-np.square(wheel_heading_x) / float(cfg.wheel_heading_sigma)), axis=1
        )

        torque_native = np.asarray(info["torques"], dtype=dtype)
        wheel_torque = torque_native[:, self._native_wheel_indices]
        zero_torque_window = airborne & np.all(
            wheel_time < float(cfg.wheel_zero_torque_before_contact_s), axis=1
        )
        zero_torque = zero_torque_window * np.exp(
            -np.square(np.max(np.abs(wheel_torque), axis=1) / float(cfg.wheel_zero_torque_sigma))
        )

        commands = np.asarray(info["commands"], dtype=dtype)
        root_x = np.asarray(linvel, dtype=dtype)[:, 0]
        positive = (commands[:, 0] > float(cfg.directional_command_x_threshold)) & (
            root_x > float(cfg.directional_root_x_threshold)
        )
        negative = (commands[:, 0] < -float(cfg.directional_command_x_threshold)) & (
            root_x < -float(cfg.directional_root_x_threshold)
        )
        direction = positive.astype(dtype) - negative.astype(dtype)
        gate = positive | negative
        wheel_speed = np.asarray(dof_vel, dtype=dtype)[:, 4:] * direction[:, None]
        wheel_score = np.clip(
            (wheel_speed - float(cfg.directional_speed_start))
            / (float(cfg.directional_speed_full) - float(cfg.directional_speed_start)),
            0.0,
            1.0,
        ).min(axis=1)
        directional_window = airborne & np.all(
            wheel_time < float(cfg.directional_before_contact_s), axis=1
        )
        directional = (directional_window & gate) * wheel_score
        shortfall_window = (
            airborne
            & np.any(wheel_time > 0.0, axis=1)
            & np.all(wheel_time < float(cfg.directional_shortfall_before_contact_s), axis=1)
        )
        shortfall = (shortfall_window & gate) * (1.0 - wheel_score)
        return {
            "airborne_joint_pos_limits": joint_limits,
            "airborne_wheel_heading_x_centering": heading,
            "airborne_air_wheel_zero_torque_exp": zero_torque,
            "airborne_precontact_wheel_directional_speed": directional,
            "airborne_precontact_wheel_directional_speed_shortfall": shortfall,
        }

    def _compute_reward(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        reward = self._state_machine_super()._compute_reward(
            info,
            linvel,
            gyro,
            projected_gravity,
            dof_pos,
            dof_vel,
        )
        machine = self._state_machine
        if machine is None or not machine.cfg.airborne_reward.enabled:
            return reward
        cfg = machine.cfg.airborne_reward
        raw = self._state_machine_source_reward_raw(
            info,
            linvel,
            projected_gravity,
            dof_vel,
        )
        delta = np.zeros_like(reward)
        dt = float(machine.cfg.control_dt)
        base_scales = self._reward_cfg.scales
        for name, target_scale in cfg.base_scale_overrides.items():
            delta += raw[name] * (float(target_scale) - float(base_scales.get(name, 0.0))) * dt
        airborne = machine.airborne_state.astype(self._np_dtype)
        for name, multiplier in cfg.airborne_scale_multipliers.items():
            target_scale = float(cfg.base_scale_overrides.get(name, base_scales.get(name, 0.0)))
            delta += raw[name] * target_scale * (float(multiplier) - 1.0) * airborne * dt
        additions = self._state_machine_airborne_additions(info, linvel, dof_vel)
        weights = {
            "airborne_joint_pos_limits": cfg.joint_pos_limits_weight,
            "airborne_wheel_heading_x_centering": cfg.wheel_heading_x_centering_weight,
            "airborne_air_wheel_zero_torque_exp": cfg.wheel_zero_torque_exp_weight,
            "airborne_precontact_wheel_directional_speed": cfg.wheel_directional_speed_weight,
            "airborne_precontact_wheel_directional_speed_shortfall": (
                cfg.wheel_directional_speed_shortfall_weight
            ),
        }
        for name, value in additions.items():
            delta += value * float(weights[name]) * dt
        numerical_failure = np.asarray(info.get("numerical_safety_failure", False), dtype=bool)
        if numerical_failure.shape != ():
            delta[numerical_failure] = 0.0
        return np.nan_to_num(reward + delta, nan=0.0, posinf=0.0, neginf=0.0).astype(
            reward.dtype,
            copy=False,
        )

    def _compute_truncated(self, state: Any) -> np.ndarray:
        truncated = self._state_machine_super()._compute_truncated(state)
        timeout = np.asarray(state.info.get("state_machine_timeout", False), dtype=bool)
        if timeout.shape != ():
            np.logical_or(truncated, timeout, out=truncated)
        return truncated


__all__ = [
    "WheelbipeAirborneCommandResampleConfig",
    "WheelbipeAirborneRewardConfig",
    "WheelbipeContactSource",
    "WheelbipeJumpPhase",
    "WheelbipeJumpTakeoffConfig",
    "WheelbipeMotionState",
    "WheelbipeStairConfig",
    "WheelbipeStateMachine",
    "WheelbipeStateMachineConfig",
    "WheelbipeStateMachineOwnerMixin",
    "WheelbipeStateMachineSensors",
    "WheelbipeStepUpConfig",
    "WheelbipeTerrainCommandConfig",
    "WheelbipeTerrainCommandProfileConfig",
]
