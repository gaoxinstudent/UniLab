"""Explicit WheelBipe V14 task variants.

The upstream project exposes a small family of task and play identifiers.  A
registry entry is useful even when two entries share the same owner
implementation: it keeps the task selection visible in Hydra and gives the
runtime a place to validate the algorithm/observation contract.  This module
therefore contains *named* variants rather than silently aliasing every
upstream identifier to the normal policy.

The compact variants (DreamWaQ, HIM and NP3O) intentionally emit the source
project's 28-dimensional one-step proprioceptive observation.  Their custom
runner stacks the configured history.  Normal PPO variants retain the public
35-dimensional ROS contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg

from .base import (
    COMPACT_POLICY_OBS_DIM,
    COMPACT_PRIVILEGED_OBS_DIM,
    COMPACT_PRIVILEGED_OBS_HEIGHT_CLIP,
    COMPACT_PRIVILEGED_OBS_HEIGHT_SCALE,
    COMPACT_PRIVILEGED_ROOT_LINVEL_CLIP,
    WheelbipeControlConfig,
    WheelbipeGimbalConfig,
)
from .joystick import (
    WheelbipeCommands,
    WheelbipeDomainRandConfig,
    WheelbipeHIMCurriculumConfig,
    WheelbipeJointFrictionRandomizationConfig,
    WheelbipeNoiseConfig,
    WheelbipeRewardConfig,
    WheelbipeV14Env,
    WheelbipeV14FlatCfg,
)
from .rough import (
    WheelbipeRoughBoundaryResetConfig,
    WheelbipeRoughCommands,
    WheelbipeV14RoughCfg,
    WheelbipeV14RoughEnv,
    wheelbipe_play_scene,
    wheelbipe_rotation_scene,
    wheelbipe_running_scene,
)
from .semantics import (
    SOURCE_V14_FLAT_V1_REWARD_SCALES,
    SOURCE_V14_REWARD_SCALES,
    SOURCE_V14_ROUGH_V0_REWARD_SCALES,
    SOURCE_V14_ROUGH_V1_REWARD_SCALES,
)
from .state_machine import (
    WheelbipeAirborneCommandResampleConfig,
    WheelbipeAirborneRewardConfig,
    WheelbipeStateMachineConfig,
    WheelbipeStateMachineOwnerMixin,
    WheelbipeStepUpConfig,
    WheelbipeTerrainCommandConfig,
    WheelbipeTerrainCommandProfileConfig,
)
from .task_modes import (
    WheelbipeGimbalSpinTranslateConfig,
    WheelbipeGimbalSpinTranslateOwnerMixin,
)


def _pinned_airborne_state_machine() -> WheelbipeStateMachineConfig:
    """Return the Flat-v1/Rough-v1 state owner from the pinned V14 config."""

    return WheelbipeStateMachineConfig(
        enabled=True,
        control_dt=0.02,
        reset_airborne_probability=0.3,
        airborne_enter_steps=1,
        landing_contact_steps=1,
        landing_hold_steps=14,
        max_airborne_steps=50,
        wheel_radius=0.06,
        body_airborne_height=0.30,
        wheel_airborne_clearance=0.08,
        wheel_contact_height=0.15,
        wheel_contact_force_threshold=20.0,
        base_contact_force_threshold=5.0,
        base_contact_steps=13,
        contact_history_steps=3,
        airborne_height_target=0.30,
        landing_height_target=0.24,
        airborne_height_override_enabled=False,
        landing_height_override_enabled=False,
        command_scale_airborne=1.0,
        command_scale_landing=1.0,
        landing_trajectory_enabled=False,
        airborne_reward=WheelbipeAirborneRewardConfig(
            enabled=True,
            base_scale_overrides={
                "action_rate": -0.002,
                "action_smoothness_leg": -0.005,
                "action_smoothness_wheel": -0.001,
                "leg_joint_acc": -1.0e-7,
                "leg_joint_vel": -1.0e-3,
                "wheel_acc": -2.0e-9,
                "wheel_vel": -2.0e-6,
            },
            airborne_scale_multipliers={
                "undesired_contact": 25.0,
                "flat_orientation_y_v": 0.0,
                "termination": 6.0,
                "track_height_square": 0.0,
                "foot_bound_square": 0.0,
            },
            joint_pos_limits_weight=-10.0,
            wheel_heading_x_centering_weight=10.0,
            wheel_zero_torque_exp_weight=20.0,
            wheel_directional_speed_weight=10.0,
            wheel_directional_speed_shortfall_weight=-10.0,
        ),
    )


def _pinned_rough_v1_state_machine() -> WheelbipeStateMachineConfig:
    """Return the final Rough-v1 runtime composition (Airborne + StepUp)."""

    cfg = _pinned_airborne_state_machine()
    cfg.step_up = WheelbipeStepUpConfig(
        enabled=True,
        forward_offset=0.50,
        step_height_min=0.12,
        step_height_max=0.14,
        wall_height=0.14,
        height_command_bias=0.16,
        hold_s=2.0,
        height_command_max=0.40,
    )
    cfg.airborne_command_resample = WheelbipeAirborneCommandResampleConfig(
        enabled=True,
        probability=0.15,
        terrain_names=(
            "high_stair_for_rm",
            "high_speed_stair_for_rm",
            "low_speed_stair_for_rm",
        ),
        lin_vel_x_range=(-1.5, 1.5),
        lin_vel_y_range=(0.0, 0.0),
        ang_vel_z_range=(-1.0, 1.0),
        lin_vel_x_sign_from_current=True,
    )
    cfg.airborne_reward.base_scale_overrides.update(
        {
            "joint_torque": -1.0e-5,
            "wheel_power": -1.0e-5,
            "track_lin_vel_xy": 1.25,
        }
    )
    return cfg


def _pinned_v1_commands() -> WheelbipeCommands:
    """Materialize the inherited Flat-v1 mutually-exclusive command buckets."""

    return WheelbipeCommands(
        vel_limit=[[-2.7, 0.0, -2.0 * np.pi], [2.7, 0.0, 2.0 * np.pi]],
        special_mode_start_iterations=[0, 0, 0],
        special_mode_probabilities=[0.15, 0.15, 0.20],
        gimbal_mode_probability=0.0,
        gimbal_mode_start_iteration=0,
        zero_command_probability=0.10,
        zero_command_start_iteration=0,
    )


def _pinned_rough_v1_commands() -> WheelbipeRoughCommands:
    """Typed Rough-v1 copy of the inherited Flat-v1 command owner."""

    return WheelbipeRoughCommands(
        vel_limit=[[-2.7, 0.0, -2.0 * np.pi], [2.7, 0.0, 2.0 * np.pi]],
        special_mode_start_iterations=[0, 0, 0],
        special_mode_probabilities=[0.15, 0.15, 0.20],
        gimbal_mode_probability=0.0,
        gimbal_mode_start_iteration=0,
        zero_command_probability=0.10,
        zero_command_start_iteration=0,
    )


def _pinned_v2_commands() -> WheelbipeCommands:
    """Materialize the four mutually-exclusive Flat-v2 source buckets."""

    return WheelbipeCommands(
        vel_limit=[[-2.7, 0.0, -2.0 * np.pi], [2.7, 0.0, 2.0 * np.pi]],
        special_mode_start_iterations=[0, 0, 0],
        special_mode_probabilities=[0.10, 0.10, 0.20],
        gimbal_mode_probability=0.20,
        gimbal_mode_start_iteration=0,
        zero_command_probability=0.0,
        zero_command_start_iteration=0,
    )


def _pinned_rough_v2_commands() -> WheelbipeRoughCommands:
    """Typed Rough-v0 copy of the inherited Flat-v2 command owner."""

    return WheelbipeRoughCommands(
        vel_limit=[[-2.7, 0.0, -2.0 * np.pi], [2.7, 0.0, 2.0 * np.pi]],
        special_mode_start_iterations=[0, 0, 0],
        special_mode_probabilities=[0.10, 0.10, 0.20],
        gimbal_mode_probability=0.20,
        gimbal_mode_start_iteration=0,
        zero_command_probability=0.0,
        zero_command_start_iteration=0,
    )


def _pinned_rough_v0_commands() -> WheelbipeRoughCommands:
    """Materialize the normal-only command buckets in the rough checkpoint."""

    return WheelbipeRoughCommands(
        vel_limit=[[-2.7, 0.0, -2.0 * np.pi], [2.7, 0.0, 2.0 * np.pi]],
        special_mode_start_iterations=[0, 0, 0],
        special_mode_probabilities=[0.15, 0.15, 0.30],
        gimbal_mode_probability=0.0,
        gimbal_mode_start_iteration=0,
        zero_command_probability=0.0,
        zero_command_start_iteration=0,
    )


def _pinned_play_v2_commands() -> WheelbipeCommands:
    """Flat-Play-v2 reserves every non-standing sample for gimbal mode."""

    return WheelbipeCommands(
        vel_limit=[[-2.7, 0.0, -2.0 * np.pi], [2.7, 0.0, 2.0 * np.pi]],
        special_mode_start_iterations=[0, 0, 0],
        special_mode_probabilities=[0.0, 0.0, 0.0],
        gimbal_mode_probability=1.0,
        gimbal_mode_start_iteration=0,
        zero_command_probability=0.0,
        zero_command_start_iteration=0,
    )


def _pinned_rough_play_v0_commands() -> WheelbipeRoughCommands:
    """UniformVelocityCommandCfg installed by pinned Rough-Play-v0."""

    return WheelbipeRoughCommands(
        vel_limit=[[2.2, 0.0, -np.pi], [2.2, 0.0, np.pi]],
        heading_range=[-np.pi, np.pi],
        heading_control_stiffness=1.0,
        rel_standing_envs=0.0,
        rel_heading_envs=0.5,
        source_curriculum_enabled=False,
        special_mode_start_iterations=[0, 0, 0],
        special_mode_probabilities=[0.0, 0.0, 0.0],
        gimbal_mode_probability=0.0,
        gimbal_mode_start_iteration=0,
        zero_command_probability=0.0,
        zero_command_start_iteration=0,
    )


def _pinned_rough_play_v1_state_machine() -> WheelbipeStateMachineConfig:
    """Rough-v1 state owner after its play terrain replaces running columns."""

    cfg = _pinned_rough_v1_state_machine()
    cfg.airborne_command_resample.enabled = False
    return cfg


def _profile(
    *,
    height: tuple[float, float],
    x: tuple[tuple[float, float], ...] | None = None,
    y: tuple[tuple[float, float], ...] | None = None,
    yaw_non_heading: tuple[tuple[float, float], ...] | None = None,
    axis_heading: bool = False,
    disable_air: bool = False,
    disable_ground: bool = False,
    disable_special: bool = False,
) -> WheelbipeTerrainCommandProfileConfig:
    return WheelbipeTerrainCommandProfileConfig(
        height_ranges=(height,),
        lin_vel_x_ranges=x,
        lin_vel_y_ranges=y,
        ang_vel_z_non_heading_ranges=yaw_non_heading,
        reset_heading_axis_aligned_only=axis_heading,
        disable_predefined_reset_air=disable_air,
        disable_predefined_reset_ground=disable_ground,
        disable_special_mode=disable_special,
    )


def _pinned_rotation_terrain_commands() -> WheelbipeTerrainCommandConfig:
    """Filtered V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1 for Rough-v0."""

    return WheelbipeTerrainCommandConfig(
        enabled=True,
        profiles={
            "tiny_step_rot": _profile(height=(0.22, 0.42)),
            "slope_for_rm_low": _profile(height=(0.22, 0.42)),
            "inv_slope_for_rm_low": _profile(height=(0.22, 0.42)),
            "stair_slope_for_rm_low": _profile(height=(0.22, 0.42)),
            "inv_stair_slope_for_rm_low": _profile(height=(0.22, 0.42)),
            "plane_for_rm_rot": _profile(height=(0.20, 0.42)),
            "random_uniform_for_rm": _profile(height=(0.22, 0.42)),
        },
    )


def _pinned_running_terrain_commands() -> WheelbipeTerrainCommandConfig:
    """Filtered V14_ROUGH_TERRAIN_COMMAND_OVERRIDES for Rough-v1."""

    return WheelbipeTerrainCommandConfig(
        enabled=True,
        profiles={
            "low_speed_stair_for_rm": _profile(
                height=(0.22, 0.34),
                x=((0.5, 1.5), (-1.5, -0.5)),
                y=((0.0, 0.0),),
                yaw_non_heading=((-0.1, 0.1),),
                axis_heading=True,
                disable_air=True,
                disable_special=True,
            ),
            "tiny_step": _profile(height=(0.22, 0.32), disable_air=True),
            "slope_for_rm_high": _profile(
                height=(0.24, 0.32),
                x=((-2.5, 2.5),),
                y=((0.0, 0.0),),
                disable_air=True,
            ),
            "inv_slope_for_rm_low": _profile(height=(0.22, 0.42)),
            "high_speed_stair_for_rm": _profile(
                height=(0.22, 0.34),
                x=((1.5, 2.7), (-2.7, -1.5)),
                y=((0.0, 0.0),),
                yaw_non_heading=((-0.1, 0.1),),
                axis_heading=True,
                disable_air=True,
                disable_special=True,
            ),
            "stair_slope_for_rm_high": _profile(
                height=(0.24, 0.32),
                x=((-2.5, 2.5),),
                y=((0.0, 0.0),),
                yaw_non_heading=((-np.pi, np.pi),),
                disable_air=True,
            ),
            "inv_stair_slope_for_rm_low": _profile(height=(0.22, 0.42)),
            "inv_stair_slope_for_rm_high": _profile(
                height=(0.24, 0.32),
                x=((1.5, 2.7), (-2.7, -1.5)),
                y=((0.0, 0.0),),
                yaw_non_heading=((-0.1, 0.1),),
                axis_heading=True,
                disable_air=True,
                disable_special=True,
            ),
            "random_uniform_for_rm": _profile(height=(0.22, 0.42)),
            "cliff_inv_stair_slope_short_for_rm": _profile(
                height=(0.24, 0.32),
                x=((2.0, 2.7), (-2.7, -2.0)),
                y=((0.0, 0.0),),
                yaw_non_heading=((-0.1, 0.1),),
                axis_heading=True,
                disable_air=True,
                disable_special=True,
            ),
        },
    )


def _pinned_play_terrain_commands() -> WheelbipeTerrainCommandConfig:
    return WheelbipeTerrainCommandConfig(
        enabled=True,
        profiles={
            "cliff_inv_stair_slope_short_for_rm_play": _profile(
                height=(0.30, 0.30),
                x=((2.5, 2.5),),
                y=((0.0, 0.0),),
                yaw_non_heading=((0.0, 0.0),),
                axis_heading=True,
                disable_air=True,
                disable_ground=True,
                disable_special=True,
            )
        },
    )


def _source_flat_reward_config() -> WheelbipeRewardConfig:
    """Materialize the inherited pinned Flat reward graph for exact aliases."""

    return WheelbipeRewardConfig(scales=dict(SOURCE_V14_REWARD_SCALES))


def _source_flat_v1_reward_config() -> WheelbipeRewardConfig:
    """Materialize the pinned Flat-v1 post-init reward graph."""

    return WheelbipeRewardConfig(scales=dict(SOURCE_V14_FLAT_V1_REWARD_SCALES))


def _source_rough_v0_reward_config() -> WheelbipeRewardConfig:
    """Materialize the pinned Rough-v0 post-init reward graph."""

    return WheelbipeRewardConfig(scales=dict(SOURCE_V14_ROUGH_V0_REWARD_SCALES))


def _source_rough_v1_reward_config() -> WheelbipeRewardConfig:
    """Materialize the pinned Rough-v1 post-init reward graph."""

    return WheelbipeRewardConfig(scales=dict(SOURCE_V14_ROUGH_V1_REWARD_SCALES))


def _source_control_config() -> WheelbipeControlConfig:
    """Return the exact source action owner (runner ``clip_actions: null``).

    The source V14 wheel-velocity controller expands its configured
    ``max_wheel_vel=100`` by 1.5 in ``env.py`` before clipping the decoded
    target, hence the effective target bound is 150 rad/s here.
    """

    return WheelbipeControlConfig(
        clip_actions=np.inf,
        leg_position_target_limit=(-3.14, 3.14),
        wheel_velocity_target_limit=150.0,
    )


def _source_flat_domain_rand() -> WheelbipeDomainRandConfig:
    """Materialize source Flat randomization inherited by compact variants."""

    return WheelbipeDomainRandConfig(
        randomize_body_mass=True,
        random_com=True,
        randomize_body_material=True,
        randomize_kp=True,
        randomize_kd=True,
        joint_friction=WheelbipeJointFrictionRandomizationConfig(enabled=True),
        use_leg_random_start=True,
        use_predefined_leg_random_start=True,
        source_push_velocity_enabled=True,
        source_external_force_enabled=True,
    )


def _pinned_v1_domain_rand() -> WheelbipeDomainRandConfig:
    """Flat/Rough-v1 inherited source DR plus explicit ``air_1`` reset."""

    cfg = _source_flat_domain_rand()
    cfg.predefined_air_enabled = True
    cfg.predefined_air_probability = 0.30
    return cfg


def _pinned_rough_play_v0_domain_rand() -> WheelbipeDomainRandConfig:
    """State-machine-independent ``play_air`` reset declared by Rough-Play-v0."""

    cfg = _source_flat_domain_rand()
    # The pinned Play event owner removes only the global actuator gain
    # randomization terms.  Keep the inherited spring/effort reset events and
    # the remaining source V14 randomization profile intact.
    cfg.randomize_kp = False
    cfg.randomize_kd = False
    cfg.predefined_ground_probability = 0.0
    cfg.predefined_air_enabled = True
    cfg.predefined_air_probability = 1.0
    cfg.predefined_air_pose_z_range = [0.25, 0.25]
    cfg.predefined_air_pose_roll_range = [-0.05, 0.05]
    cfg.predefined_air_pose_pitch_range = [-0.1, 0.1]
    cfg.predefined_air_pose_yaw_range = [-np.pi, np.pi]
    cfg.predefined_air_body_vel_x_range = [1.9, 2.0]
    cfg.predefined_air_body_vel_y_range = [-0.25, 0.25]
    cfg.predefined_air_body_vel_z_range = [0.0, 0.0]
    cfg.predefined_air_body_ang_vel_roll_range = [0.0, 0.0]
    cfg.predefined_air_body_ang_vel_pitch_range = [0.0, 0.0]
    cfg.predefined_air_body_ang_vel_yaw_range = [0.0, 0.0]
    cfg.predefined_air_leg_length_range = [0.20, 0.35]
    cfg.predefined_air_leg_angle_range = [-0.2 * np.pi, 0.2 * np.pi]
    cfg.predefined_air_command_limits_enabled = False
    return cfg


def _source_flat_noise() -> WheelbipeNoiseConfig:
    """Return the source Flat observation-noise profile."""

    # The pinned training checkpoint records unit-scale noise for both leg
    # and wheel joint velocities; keep this effective source profile explicit
    # rather than inheriting the smaller deployment default.
    return WheelbipeNoiseConfig(level=1.0, scale_joint_vel=1.0, scale_wheel_vel=1.0)


def _source_him_curriculum() -> WheelbipeHIMCurriculumConfig:
    return WheelbipeHIMCurriculumConfig(enabled=True)


@dataclass
class WheelbipeVariantCfg(WheelbipeV14FlatCfg):
    """Common metadata and safety knobs for named V14 variants."""

    variant_name: str = "flat-v0"
    # Every class in this module represents an exact pinned upstream ID.  The
    # source reward/observation/reset owner is therefore part of its identity,
    # including variants whose optional task state machine is disabled.
    training_semantics: str = "source_v14"
    reward_config: WheelbipeRewardConfig | None = field(default_factory=_source_flat_reward_config)
    control_config: WheelbipeControlConfig = field(default_factory=_source_control_config)
    # Exact source IDs inherit EventCfgV14 even when their optional task state
    # machine is disabled.  Keep this owner-local so the generic canonical
    # WheelbipeV14FlatCfg can retain its independent legacy DR defaults.
    domain_rand: WheelbipeDomainRandConfig = field(default_factory=_source_flat_domain_rand)
    noise_config: WheelbipeNoiseConfig = field(default_factory=_source_flat_noise)
    # ``normal`` means the ROS 35D contract; ``compact`` is used by custom
    # representation algorithms and is stacked by their runner.
    policy_observation_mode: str = "normal"
    custom_algorithm: str = "ppo"
    num_actor_history: int = 1
    num_estimate: int = 4
    num_costs: int = 0
    # Source-only capabilities are recorded explicitly so named registry
    # entries cannot be mistaken for full Isaac state-machine/gimbal parity.
    # ``implemented`` denotes the backend-neutral bounded owner; it does not
    # assert byte-for-byte Isaac contact/scanner or dynamics equivalence.
    source_state_machine_status: str = "not_used"
    source_gimbal_status: str = "not_used"
    # The source v2 task relies on a gimbal joint.  Gimbal owners explicitly
    # materialize the two extra channels at construction; this flag keeps an
    # exact upstream alias from silently falling back to the eight-actuator
    # mechanism when a custom override disables them.
    require_gimbal_actuators: bool = False
    gimbal_spin_translate: WheelbipeGimbalSpinTranslateConfig = field(
        default_factory=WheelbipeGimbalSpinTranslateConfig
    )
    terrain_commands: WheelbipeTerrainCommandConfig = field(
        default_factory=WheelbipeTerrainCommandConfig
    )
    # Source-only capability metadata is part of the identity of an exact
    # variant, not a user-tunable runtime knob.  Keep the expected tuple on the
    # class (rather than in dataclass fields) so ``registry.make(...,
    # env_cfg_override=...)`` cannot turn a bounded alias into a different
    # capability owner by mutating fields before ``validate()`` runs.
    _SOURCE_CAPABILITY_CONTRACT: ClassVar[tuple[str, str, bool] | None] = None
    _SOURCE_REWARD_SCALES: ClassVar[dict[str, float]] = SOURCE_V14_REWARD_SCALES
    _SOURCE_VEL_HEIGHT_GATE_ENABLED: ClassVar[bool] = False
    _SOURCE_HIM_CURRICULUM_ENABLED: ClassVar[bool] = False
    # ``(control_mode, heading_target_mode, randomize_heading)`` is an
    # immutable reset-state contract for each exact source ID.  The inherited
    # Flat owner uses yaw velocity; v2/rotation owners override this with the
    # manual world-heading controller, and only Flat-Play-v2 fixes heading 0.
    _SOURCE_GIMBAL_RESET_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "velocity",
        "sampled",
        False,
    )

    def validate(self) -> None:
        super().validate()
        if str(self.training_semantics).strip().lower() != "source_v14":
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} is an exact upstream "
                "identity and requires training_semantics='source_v14'; do not "
                "override it to legacy"
            )
        expected_reward_scales = dict(type(self)._SOURCE_REWARD_SCALES)
        actual_reward_scales = (
            None if self.reward_config is None else dict(self.reward_config.scales)
        )
        if actual_reward_scales != expected_reward_scales:
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} has an immutable "
                "source reward graph; reward_config.scales must equal its "
                "pinned enabled profile"
            )
        expected_noise = WheelbipeNoiseConfig(level=1.0, scale_joint_vel=1.0, scale_wheel_vel=1.0)
        if self.noise_config != expected_noise:
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} has an immutable "
                "source actor-noise profile"
            )
        if self.debug_value_diagnosis is not False:
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} has immutable source "
                "debug_value_diagnosis=false; the pinned raw-observation outlier "
                "gate must not be activated with the global step counter"
            )
        expected_height_gate = bool(type(self)._SOURCE_VEL_HEIGHT_GATE_ENABLED)
        if bool(self.vel_height_gate_enabled) != expected_height_gate:
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} has immutable source "
                f"vel_height_gate_enabled={expected_height_gate!r}"
            )
        if (
            str(self.vel_height_gate_mode) != "linear_band"
            or not np.isclose(float(self.vel_height_gate_full_error), 0.05)
            or not np.isclose(float(self.vel_height_gate_zero_error), 0.1)
            or not np.isclose(float(self.vel_height_gate_tracker_sigma), 0.02)
        ):
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} has immutable source "
                "velocity-height gate parameters"
            )
        expected_him_curriculum = WheelbipeHIMCurriculumConfig(
            enabled=bool(type(self)._SOURCE_HIM_CURRICULUM_ENABLED)
        )
        if self.him_curriculum != expected_him_curriculum:
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} has an immutable source "
                "HIM CurriculumCfgV14 profile"
            )
        if not np.isposinf(float(self.control_config.clip_actions)):
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} reproduces source "
                "runner clip_actions=null; do not add a pre-environment action clamp"
            )
        leg_target_limit = self.control_config.leg_position_target_limit
        try:
            canonical_leg_target_limit = (
                ()
                if leg_target_limit is None
                else tuple(float(value) for value in leg_target_limit)
            )
        except (TypeError, ValueError):
            canonical_leg_target_limit = ()
        leg_target_limit_matches = len(canonical_leg_target_limit) == 2 and np.allclose(
            canonical_leg_target_limit,
            (-3.14, 3.14),
            rtol=0.0,
            atol=1.0e-12,
        )
        wheel_target_limit = self.control_config.wheel_velocity_target_limit
        if (
            not leg_target_limit_matches
            or wheel_target_limit is None
            or not np.isclose(float(wheel_target_limit), 150.0)
        ):
            raise ValueError(
                f"Wheelbipe variant {type(self).__name__!r} has immutable source "
                "decoded target limits (leg ±3.14 rad, wheel ±150 rad/s)"
            )
        if isinstance(self.gimbal_spin_translate, dict):
            self.gimbal_spin_translate = WheelbipeGimbalSpinTranslateConfig(
                **self.gimbal_spin_translate
            )
        if not isinstance(self.gimbal_spin_translate, WheelbipeGimbalSpinTranslateConfig):
            raise ValueError("gimbal_spin_translate must be WheelbipeGimbalSpinTranslateConfig")
        self.gimbal_spin_translate.validate()
        if isinstance(self.terrain_commands, dict):
            self.terrain_commands = WheelbipeTerrainCommandConfig(**self.terrain_commands)
        if not isinstance(self.terrain_commands, WheelbipeTerrainCommandConfig):
            raise ValueError("terrain_commands must be WheelbipeTerrainCommandConfig")
        self.terrain_commands.validate()
        if self.gimbal_spin_translate.enabled and not self.gimbal.enabled:
            raise ValueError("gimbal spin/translation mode requires gimbal.enabled=true")
        if self.state_machine.enabled and not np.isclose(
            float(self.state_machine.control_dt), float(self.ctrl_dt)
        ):
            raise ValueError(
                "state_machine.control_dt must match the environment ctrl_dt; "
                f"got {self.state_machine.control_dt} and {self.ctrl_dt}"
            )
        if self.policy_observation_mode not in {"normal", "compact"}:
            raise ValueError(
                "policy_observation_mode must be 'normal' or 'compact', got "
                f"{self.policy_observation_mode!r}"
            )
        if self.num_actor_history < 1:
            raise ValueError("num_actor_history must be positive")
        if self.num_estimate < 1:
            raise ValueError("num_estimate must be positive")
        if self.num_costs < 0:
            raise ValueError("num_costs cannot be negative")
        for field_name in ("source_state_machine_status", "source_gimbal_status"):
            status = str(getattr(self, field_name)).strip().lower()
            if status not in {"not_used", "unported", "required_unavailable", "implemented"}:
                raise ValueError(
                    f"{field_name} must be one of not_used, unported, "
                    f"required_unavailable, implemented; got {status!r}"
                )
        expected_capabilities = getattr(type(self), "_SOURCE_CAPABILITY_CONTRACT", None)
        if expected_capabilities is not None:
            expected_state, expected_gimbal, expected_require_gimbal = expected_capabilities
            actual_state = str(self.source_state_machine_status).strip().lower()
            actual_gimbal = str(self.source_gimbal_status).strip().lower()
            actual_require_gimbal = self.require_gimbal_actuators
            if (
                actual_state != expected_state
                or actual_gimbal != expected_gimbal
                or not isinstance(actual_require_gimbal, bool)
                or actual_require_gimbal != expected_require_gimbal
            ):
                raise ValueError(
                    f"Wheelbipe variant {type(self).__name__!r} has an immutable "
                    "source capability contract (state-machine="
                    f"{expected_state!r}, gimbal={expected_gimbal!r}, "
                    f"require_gimbal_actuators={expected_require_gimbal!r}); got "
                    f"state-machine={actual_state!r}, gimbal={actual_gimbal!r}, "
                    f"require_gimbal_actuators={actual_require_gimbal!r}. "
                    "Do not override source state-machine/gimbal capability "
                    "metadata on an exact upstream variant."
                )
            # An exact upstream variant whose state-machine capability is
            # marked ``implemented`` must actually construct the owner state
            # machine.  Checking only the descriptive status above would let
            # an override such as ``env.state_machine.enabled=false`` keep
            # the metadata while silently reverting Flat-v1/Rough-v1 to the
            # plain eight-actuator lifecycle.  Treat enablement as part of
            # the immutable capability contract, just like gimbal channels.
            if expected_state == "implemented" and not bool(self.state_machine.enabled):
                raise ValueError(
                    f"Wheelbipe variant {type(self).__name__!r} requires its "
                    "immutable source capability contract (implemented "
                    "state-machine owner); do not override "
                    "env.state_machine.enabled=false on an exact upstream variant."
                )
        # Keep the registry boundary fail-closed for genuinely unavailable
        # source capabilities.  Implemented owners below provide an explicit
        # backend-neutral state machine rather than merely relabelling the
        # historical flat task.
        if str(self.source_state_machine_status).strip().lower() == "unported":
            raise ValueError(
                f"Wheelbipe variant {self.variant_name!r} cannot be constructed: "
                "the source state-machine capability is not ported; "
                "select a supported owner or port the state machine first."
            )
        # NP3O's migrated source profile has five independently-trained
        # constraint channels.  Treat that width as part of the owner
        # contract: a Hydra override such as ``env.num_costs=0`` must fail at
        # config validation instead of producing a policy/checkpoint whose
        # cost critic no longer matches the algorithm.
        algorithm = str(self.custom_algorithm).strip().lower()
        if algorithm == "np3o" and int(self.num_costs) != 5:
            raise ValueError(
                "Wheelbipe NP3O owner requires exactly five cost channels; "
                f"got num_costs={self.num_costs}"
            )
        if algorithm in {"him", "dreamwaq", "dream_waq"} and int(self.num_costs) != 0:
            raise ValueError(
                f"Wheelbipe {algorithm} owner does not use cost channels; "
                f"got num_costs={self.num_costs}"
            )
        if self.require_gimbal_actuators and not bool(getattr(self.gimbal, "enabled", False)):
            raise ValueError(
                "This WheelBipe variant requires the physical gimbal channels; "
                "set env.gimbal.enabled=true (the canonical owner does this by default)."
            )
        if self.require_gimbal_actuators:
            expected_mode, expected_target_mode, expected_randomize = type(
                self
            )._SOURCE_GIMBAL_RESET_CONTRACT
            actual_mode = str(self.gimbal.control_mode).strip().lower()
            actual_target_mode = str(self.gimbal.heading_target_mode).strip().lower()
            if (
                actual_mode != expected_mode
                or actual_target_mode != expected_target_mode
                or bool(self.gimbal.randomize_heading) != expected_randomize
                or not np.isclose(float(self.gimbal.pitch_target), -0.5)
                or not np.allclose(
                    np.asarray(self.gimbal.yaw_velocity_range, dtype=np.float64),
                    [-np.pi, np.pi],
                )
                or not np.allclose(
                    np.asarray(self.gimbal.yaw_heading_range, dtype=np.float64),
                    [-np.pi, np.pi],
                )
                or (
                    expected_target_mode == "fixed"
                    and not np.isclose(float(self.gimbal.fixed_heading), 0.0)
                )
            ):
                raise ValueError(
                    f"Wheelbipe variant {type(self).__name__!r} has an immutable source "
                    "gimbal reset contract: pitch=-0.5, yaw ranges=[-pi, pi], "
                    f"control_mode={expected_mode!r}, "
                    f"heading_target_mode={expected_target_mode!r}, "
                    f"randomize_heading={expected_randomize!r}"
                )


@dataclass
class WheelbipeFlatV0Cfg(WheelbipeVariantCfg):
    variant_name: str = "flat-v0"
    _SOURCE_CAPABILITY_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "not_used",
        "implemented",
        True,
    )
    source_gimbal_status: str = "implemented"
    require_gimbal_actuators: bool = True


@dataclass
class WheelbipeFlatV1Cfg(WheelbipeVariantCfg):
    """Flat landing-pretraining profile.

    The owner implements the source airborne/landing lifecycle with public
    per-body contact magnitudes plus wheel/terrain geometry.  The inherited
    physical gimbal remains outside the six-dimensional policy action.
    """

    variant_name: str = "flat-v1"
    _SOURCE_REWARD_SCALES: ClassVar[dict[str, float]] = SOURCE_V14_FLAT_V1_REWARD_SCALES
    reward_config: WheelbipeRewardConfig | None = field(
        default_factory=_source_flat_v1_reward_config
    )
    _SOURCE_CAPABILITY_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "implemented",
        "implemented",
        True,
    )
    source_state_machine_status: str = "implemented"
    source_gimbal_status: str = "implemented"
    require_gimbal_actuators: bool = True
    state_machine: WheelbipeStateMachineConfig = field(
        default_factory=_pinned_airborne_state_machine
    )
    commands: WheelbipeCommands = field(default_factory=_pinned_v1_commands)
    domain_rand: WheelbipeDomainRandConfig = field(default_factory=_pinned_v1_domain_rand)
    height_range: list[float] = field(default_factory=lambda: [0.20, 0.42])
    use_absolute_height: bool = False
    height_obs_clip_enabled: bool = True
    height_obs_clip_range: list[float | None] = field(default_factory=lambda: [0.05, 0.45])
    termination_duration_steps: int = 10
    # Keep source-like physical-step placement explicit.
    use_obs_delay: bool = True
    use_act_delay: bool = True


@dataclass
class WheelbipeFlatV2Cfg(WheelbipeVariantCfg):
    """Flat gimbal-spin profile metadata.

    The owner materializes a ten-actuator XML view and the pinned gimbal-yaw
    spin/translation mode, while the public policy remains 6D/35D compatible.
    """

    variant_name: str = "flat-v2-gimbal-spin"
    _SOURCE_CAPABILITY_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "not_used",
        "implemented",
        True,
    )
    _SOURCE_GIMBAL_RESET_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "heading_pd",
        "sampled",
        True,
    )
    source_gimbal_status: str = "implemented"
    require_gimbal_actuators: bool = True
    heading_command: bool = True
    commands: WheelbipeCommands = field(default_factory=_pinned_v2_commands)
    ctrl_mode_obs_scale: list[float] = field(default_factory=lambda: [1.0] * 7)
    gimbal: WheelbipeGimbalConfig = field(
        default_factory=lambda: WheelbipeGimbalConfig(
            enabled=True,
            control_mode="heading_pd",
            pitch_target=-0.5,
            yaw_kp=20.0,
            yaw_kd=0.1,
            randomize_heading=True,
        )
    )
    gimbal_spin_translate: WheelbipeGimbalSpinTranslateConfig = field(
        default_factory=lambda: WheelbipeGimbalSpinTranslateConfig(enabled=True)
    )

    def __post_init__(self) -> None:
        # ``heading_command`` is retained as a discoverable variant field,
        # while the actual command sampler reads the nested command owner.
        # Set both at construction so this metadata cannot silently disagree
        # with runtime behavior when a gimbal-capable asset is supplied.
        self.commands.heading_command = bool(self.heading_command)


@dataclass
class WheelbipeFlatPlayV0Cfg(WheelbipeFlatV0Cfg):
    variant_name: str = "flat-play-v0"
    play_mode: bool = True


@dataclass
class WheelbipeFlatPlayV2Cfg(WheelbipeFlatV2Cfg):
    variant_name: str = "flat-play-v2-gimbal-spin"
    play_mode: bool = True
    _SOURCE_GIMBAL_RESET_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "heading_pd",
        "fixed",
        False,
    )
    height_range: list[float] = field(default_factory=lambda: [0.20, 0.30])
    commands: WheelbipeCommands = field(default_factory=_pinned_play_v2_commands)
    gimbal: WheelbipeGimbalConfig = field(
        default_factory=lambda: WheelbipeGimbalConfig(
            enabled=True,
            control_mode="heading_pd",
            heading_target_mode="fixed",
            fixed_heading=0.0,
            pitch_target=-0.5,
            yaw_kp=20.0,
            yaw_kd=0.1,
            randomize_heading=False,
        )
    )
    gimbal_spin_translate: WheelbipeGimbalSpinTranslateConfig = field(
        default_factory=lambda: WheelbipeGimbalSpinTranslateConfig(
            enabled=True,
            relative_envs=1.0,
            speed_ranges=((0.6, 0.6),),
            heading_range=(0.0, 0.0),
            project_to_body_command=False,
        )
    )


@dataclass
class WheelbipeRoughV0Cfg(WheelbipeV14RoughCfg, WheelbipeVariantCfg):
    variant_name: str = "rough-v0"
    # ``WheelbipeV14RoughCfg`` precedes the exact-variant base in this
    # multiple-inheritance layout, so restate the immutable source identity.
    training_semantics: str = "source_v14"
    _SOURCE_REWARD_SCALES: ClassVar[dict[str, float]] = SOURCE_V14_ROUGH_V0_REWARD_SCALES
    reward_config: WheelbipeRewardConfig | None = field(
        default_factory=_source_rough_v0_reward_config
    )
    control_config: WheelbipeControlConfig = field(default_factory=_source_control_config)
    # ``WheelbipeV14RoughCfg`` wins the dataclass field lookup in this MRO.
    # Restate EventCfgV14 explicitly so the exact Rough-v0 registry path does
    # not inherit the generic rough owner's disabled randomization defaults.
    domain_rand: WheelbipeDomainRandConfig = field(default_factory=_source_flat_domain_rand)
    noise_config: WheelbipeNoiseConfig = field(default_factory=_source_flat_noise)
    _SOURCE_CAPABILITY_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "not_used",
        "implemented",
        True,
    )
    _SOURCE_GIMBAL_RESET_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "heading_pd",
        "sampled",
        True,
    )
    # Rough-v0 keeps the source v2 physical heading-PD gimbal (including its
    # startup gain randomization), while the released policy/deployment path
    # uses the generic normal-only seven-slot mode tail rather than the
    # optional continuous gimbal-spin feature vector.
    source_gimbal_status: str = "implemented"
    require_gimbal_actuators: bool = True
    source_terrain_preset: str = "rotation"
    # Match the source V14 rough checkpoint's policy input scaling.  The
    # height-target mode component (index 5) is scaled by five.
    ctrl_mode_obs_scale: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]
    )
    scene: SceneCfg = field(default_factory=wheelbipe_rotation_scene)
    commands: WheelbipeRoughCommands = field(default_factory=_pinned_rough_v0_commands)
    terrain_commands: WheelbipeTerrainCommandConfig = field(
        default_factory=_pinned_rotation_terrain_commands
    )
    rough_terrain_boundary_reset: WheelbipeRoughBoundaryResetConfig = field(
        default_factory=lambda: WheelbipeRoughBoundaryResetConfig(
            enabled=True,
            margin=0.5,
            use_inner_terrain_area=False,
        )
    )
    gimbal: WheelbipeGimbalConfig = field(
        default_factory=lambda: WheelbipeGimbalConfig(
            enabled=True,
            control_mode="heading_pd",
            pitch_target=-0.5,
            yaw_kp=20.0,
            yaw_kd=0.1,
            randomize_heading=True,
        )
    )
    # The source rough-v0 checkpoint is normal-only; gimbal remains a
    # physical actuator but does not replace the generic 7D mode tail.
    gimbal_spin_translate: WheelbipeGimbalSpinTranslateConfig = field(
        default_factory=WheelbipeGimbalSpinTranslateConfig
    )


@dataclass
class WheelbipeRoughV1Cfg(WheelbipeV14RoughCfg, WheelbipeVariantCfg):
    variant_name: str = "rough-v1"
    training_semantics: str = "source_v14"
    _SOURCE_REWARD_SCALES: ClassVar[dict[str, float]] = SOURCE_V14_ROUGH_V1_REWARD_SCALES
    reward_config: WheelbipeRewardConfig | None = field(
        default_factory=_source_rough_v1_reward_config
    )
    control_config: WheelbipeControlConfig = field(default_factory=_source_control_config)
    noise_config: WheelbipeNoiseConfig = field(default_factory=_source_flat_noise)
    _SOURCE_CAPABILITY_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "implemented",
        "implemented",
        True,
    )
    source_state_machine_status: str = "implemented"
    source_gimbal_status: str = "implemented"
    require_gimbal_actuators: bool = True
    source_terrain_preset: str = "running"
    scene: SceneCfg = field(default_factory=wheelbipe_running_scene)
    terrain_commands: WheelbipeTerrainCommandConfig = field(
        default_factory=_pinned_running_terrain_commands
    )
    rough_terrain_boundary_reset: WheelbipeRoughBoundaryResetConfig = field(
        default_factory=lambda: WheelbipeRoughBoundaryResetConfig(
            enabled=True,
            margin=0.5,
            use_inner_terrain_area=False,
        )
    )
    state_machine: WheelbipeStateMachineConfig = field(
        default_factory=_pinned_rough_v1_state_machine
    )
    commands: WheelbipeRoughCommands = field(default_factory=_pinned_rough_v1_commands)
    domain_rand: WheelbipeDomainRandConfig = field(default_factory=_pinned_v1_domain_rand)
    height_range: list[float] = field(default_factory=lambda: [0.20, 0.42])
    use_absolute_height: bool = False
    height_obs_clip_enabled: bool = True
    height_obs_clip_range: list[float | None] = field(default_factory=lambda: [0.05, 0.45])
    termination_duration_steps: int = 10
    gimbal_spin_translate: WheelbipeGimbalSpinTranslateConfig = field(
        default_factory=WheelbipeGimbalSpinTranslateConfig
    )
    use_obs_delay: bool = True
    use_act_delay: bool = True


@dataclass
class WheelbipeRoughPlayV0Cfg(WheelbipeRoughV0Cfg):
    variant_name: str = "rough-play-v0"
    play_mode: bool = True
    # The source play owner uses the unscaled mode tail even though the
    # training rough-v0 checkpoint scales its height-target component by five.
    ctrl_mode_obs_scale: list[float] = field(default_factory=lambda: [1.0] * 7)
    source_terrain_preset: str = "play_v0"
    scene: SceneCfg = field(default_factory=lambda: wheelbipe_play_scene(num_rows=10))
    commands: WheelbipeRoughCommands = field(default_factory=_pinned_rough_play_v0_commands)
    domain_rand: WheelbipeDomainRandConfig = field(
        default_factory=_pinned_rough_play_v0_domain_rand
    )
    terrain_commands: WheelbipeTerrainCommandConfig = field(
        default_factory=_pinned_play_terrain_commands
    )
    rough_terrain_boundary_reset: WheelbipeRoughBoundaryResetConfig = field(
        default_factory=lambda: WheelbipeRoughBoundaryResetConfig(
            enabled=True,
            margin=0.5,
            use_inner_terrain_area=True,
        )
    )
    gimbal_spin_translate: WheelbipeGimbalSpinTranslateConfig = field(
        default_factory=WheelbipeGimbalSpinTranslateConfig
    )
    max_episode_seconds: float = 5.0


@dataclass
class WheelbipeRoughPlayV1Cfg(WheelbipeRoughV1Cfg):
    variant_name: str = "rough-play-v1"
    play_mode: bool = True
    source_terrain_preset: str = "play_v1"
    scene: SceneCfg = field(default_factory=lambda: wheelbipe_play_scene(num_rows=1))
    terrain_commands: WheelbipeTerrainCommandConfig = field(
        default_factory=_pinned_play_terrain_commands
    )
    rough_terrain_boundary_reset: WheelbipeRoughBoundaryResetConfig = field(
        default_factory=lambda: WheelbipeRoughBoundaryResetConfig(
            enabled=True,
            margin=0.5,
            use_inner_terrain_area=False,
        )
    )
    max_episode_seconds: float = 5.0
    height_range: list[float] = field(default_factory=lambda: [0.25, 0.25])
    state_machine: WheelbipeStateMachineConfig = field(
        default_factory=_pinned_rough_play_v1_state_machine
    )


@dataclass
class WheelbipeCustomFlatCfg(WheelbipeVariantCfg):
    """Base compact-observation owner for DreamWaQ/HIM/NP3O."""

    policy_observation_mode: str = "compact"
    _SOURCE_CAPABILITY_CONTRACT: ClassVar[tuple[str, str, bool]] = (
        "not_used",
        "implemented",
        True,
    )
    source_gimbal_status: str = "implemented"
    require_gimbal_actuators: bool = True
    ctrl_mode_obs_enabled: bool = False
    ctrl_mode_obs_dim: int = 0
    # DreamWaQ/HIM/NP3O change only the policy-side representation and runner.
    # Their upstream env cfgs inherit the complete Flat owner, including its
    # reward graph, observation noise and domain randomization.  Re-materialize
    # that inheritance here instead of silently falling back to the generic
    # custom-PPO root profile.
    reward_config: WheelbipeRewardConfig | None = field(default_factory=_source_flat_reward_config)
    domain_rand: WheelbipeDomainRandConfig = field(default_factory=_source_flat_domain_rand)
    noise_config: WheelbipeNoiseConfig = field(default_factory=_source_flat_noise)
    # The upstream V14 history algorithms train with 5 ms physics-step delay
    # buffers (obs lags 1--4, action lags 1--3 sampled high-exclusively).  Keep
    # those timing semantics on the compact owner; this still does not claim
    # source dynamics/asset parity.
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    use_obs_delay: bool = True
    use_act_delay: bool = True
    obs_delay_step_unit: str = "physics"
    delay_profile: str = "source_v14_physics"
    delay_range_semantics: str = "exclusive"
    # Isaac's V14 actuator owner limits active leg and wheel actuators to
    # 40 N·m and 5 N·m respectively.  The normal ROS deployment owner keeps
    # its independent command limits (50.9/9.99); custom algorithms use this
    # local owner profile while the MJCF still enforces the same wheel
    # physical ±5 N·m force range at the backend boundary.
    control_config: WheelbipeControlConfig = field(
        default_factory=lambda: WheelbipeControlConfig(
            leg_torque_limit=40.0,
            wheel_torque_limit=5.0,
            clip_actions=np.inf,
            leg_position_target_limit=(-3.14, 3.14),
            wheel_velocity_target_limit=150.0,
        )
    )
    num_actor_history: int = 5
    num_estimate: int = 4
    # Custom variants use the normal source spring/command owner and only
    # change the policy-side representation contract.

    def validate(self) -> None:
        super().validate()
        if self.reward_config is None or dict(self.reward_config.scales) != dict(
            SOURCE_V14_REWARD_SCALES
        ):
            actual = None if self.reward_config is None else sorted(self.reward_config.scales)
            raise ValueError(
                "Exact WheelBipe DreamWaQ/HIM/NP3O owners inherit the source Flat "
                "reward graph; expected SOURCE_V14_REWARD_SCALES, got keys "
                f"{actual!r}"
            )
        required_randomization = (
            "randomize_body_mass",
            "random_com",
            "randomize_body_material",
            "use_leg_random_start",
            "use_predefined_leg_random_start",
            "source_external_force_enabled",
        )
        disabled = [
            name for name in required_randomization if not bool(getattr(self.domain_rand, name))
        ]
        if disabled:
            raise ValueError(
                "Exact WheelBipe DreamWaQ/HIM/NP3O owners inherit source Flat "
                f"domain randomization; disabled fields: {disabled!r}"
            )
        if not np.isclose(float(self.noise_config.level), 1.0):
            raise ValueError(
                "Exact WheelBipe DreamWaQ/HIM/NP3O owners inherit source Flat "
                f"observation noise level 1.0; got {self.noise_config.level!r}"
            )


@dataclass
class WheelbipeDreamWaQCfg(WheelbipeCustomFlatCfg):
    variant_name: str = "flat-dreamwaq-v0"
    custom_algorithm: str = "dreamwaq"
    num_actor_history: int = 5


@dataclass
class WheelbipeDreamWaQPlayCfg(WheelbipeDreamWaQCfg):
    variant_name: str = "flat-dreamwaq-play-v0"
    play_mode: bool = True


@dataclass
class WheelbipeHIMCfg(WheelbipeCustomFlatCfg):
    variant_name: str = "flat-him-v0"
    custom_algorithm: str = "him"
    num_actor_history: int = 5
    _SOURCE_HIM_CURRICULUM_ENABLED: ClassVar[bool] = True
    him_curriculum: WheelbipeHIMCurriculumConfig = field(default_factory=_source_him_curriculum)


@dataclass
class WheelbipeHIMPlayCfg(WheelbipeHIMCfg):
    variant_name: str = "flat-him-play-v0"
    play_mode: bool = True
    _SOURCE_HIM_CURRICULUM_ENABLED: ClassVar[bool] = False
    him_curriculum: WheelbipeHIMCurriculumConfig = field(
        default_factory=WheelbipeHIMCurriculumConfig
    )


@dataclass
class WheelbipeNP3OCfg(WheelbipeCustomFlatCfg):
    variant_name: str = "flat-np3o-barlow-v0"
    custom_algorithm: str = "np3o"
    num_actor_history: int = 10
    num_costs: int = 5
    # The source training owner uses a 20-degree tilt threshold; its separate
    # play config tightens this to 15 degrees (see the subclass below).
    np3o_tilt_limit_deg: float = 20.0
    _SOURCE_VEL_HEIGHT_GATE_ENABLED: ClassVar[bool] = True
    vel_height_gate_enabled: bool = True


@dataclass
class WheelbipeNP3OPlayCfg(WheelbipeNP3OCfg):
    variant_name: str = "flat-np3o-barlow-play-v0"
    play_mode: bool = True
    np3o_tilt_limit_deg: float = 15.0
    _SOURCE_VEL_HEIGHT_GATE_ENABLED: ClassVar[bool] = False
    vel_height_gate_enabled: bool = False


class WheelbipeVariantEnv(
    WheelbipeGimbalSpinTranslateOwnerMixin,
    WheelbipeStateMachineOwnerMixin,
    WheelbipeV14Env,
):
    """Owner env that switches only the declared policy-side profile."""

    _cfg: WheelbipeVariantCfg

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        if getattr(self._cfg, "policy_observation_mode", "normal") == "compact":
            return {"obs": COMPACT_POLICY_OBS_DIM, "critic": COMPACT_PRIVILEGED_OBS_DIM}
        return super().obs_groups_spec

    def _compute_obs(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        *,
        env_ids: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        observations = super()._compute_obs(
            info,
            linvel,
            gyro,
            projected_gravity,
            dof_pos,
            dof_vel,
            env_ids=env_ids,
        )
        if getattr(self._cfg, "policy_observation_mode", "normal") != "compact":
            return observations

        # The parent builds the canonical normal frame to keep sensor scaling
        # in one owner function.  Compact owners remove only the seven mode
        # flags and construct the source privileged frame explicitly as
        # ``[policy_28, root_lin_vel_b_3, observed_height_1]``.  In
        # particular, do not pad/truncate the normal 78D critic: doing so can
        # put torque or command fields where the custom estimator expects the
        # velocity target.
        observations["obs"] = np.asarray(observations["obs"][:, :COMPACT_POLICY_OBS_DIM]).copy()
        linvel = np.asarray(linvel, dtype=observations["obs"].dtype)
        if linvel.ndim != 2 or linvel.shape[1] != 3:
            raise RuntimeError(
                "compact Wheelbipe observation requires body-frame linear velocity shape (N, 3)"
            )
        observed = info.get("observed_height")
        if observed is None:
            # Reset builds the first observation before ``update_state`` has
            # populated ``info``.  Read the backend once on this cold path so
            # the initial privileged target still reflects measured height.
            base_pos = np.asarray(self._backend.get_base_pos(), dtype=observations["obs"].dtype)
            # ``build_reset_observation`` passes sensor arrays for only the
            # selected ``env_ids``, while backend getters return the complete
            # vectorized batch.  Align the cold-path height target with the
            # selected observation rows before constructing the compact
            # privileged frame.  Without this slice a partial reset of a
            # compact owner (e.g. env 0 of a two-env batch) observes a
            # mismatched ``(N,)`` height vector and fails its shape contract.
            if env_ids is not None and base_pos.shape[0] != observations["obs"].shape[0]:
                ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
                if ids.shape != (observations["obs"].shape[0],):
                    raise RuntimeError(
                        "compact Wheelbipe reset env_ids do not match observation batch: "
                        f"ids={ids.shape}, observations={observations['obs'].shape}"
                    )
                if np.any(ids < 0) or np.any(ids >= base_pos.shape[0]):
                    raise IndexError(
                        "compact Wheelbipe reset env_ids are outside the backend batch: "
                        f"ids={ids.tolist()}, batch={base_pos.shape[0]}"
                    )
                base_pos = base_pos[ids]
            observed, _reward_height = self._source_height_signals(base_pos)
        heights = np.asarray(observed, dtype=observations["obs"].dtype).reshape(-1)
        if heights.shape != (observations["obs"].shape[0],):
            raise RuntimeError(
                "compact Wheelbipe observation requires batched info['height_commands']"
            )
        # Match the source V14 ``V14_BASIC_OBS_*`` contract.  The compact
        # privileged frame uses ``root_lin_vel_b`` with an absolute clip of
        # 100 and ``obs_height`` clipped in raw metres to [-10, 10], followed
        # by the source height scale of 5.  Applying the scale after clipping
        # is important: scaling first would turn a 20 m height into the wrong
        # 100-valued feature and would not load source checkpoints faithfully.
        linvel_priv = np.clip(
            linvel,
            -COMPACT_PRIVILEGED_ROOT_LINVEL_CLIP,
            COMPACT_PRIVILEGED_ROOT_LINVEL_CLIP,
        )
        height_priv = (
            np.clip(
                heights,
                COMPACT_PRIVILEGED_OBS_HEIGHT_CLIP[0],
                COMPACT_PRIVILEGED_OBS_HEIGHT_CLIP[1],
            )
            * COMPACT_PRIVILEGED_OBS_HEIGHT_SCALE
        )
        observations["critic"] = np.concatenate(
            (
                observations["obs"],
                linvel_priv.astype(observations["obs"].dtype, copy=False),
                height_priv[:, None].astype(observations["obs"].dtype, copy=False),
            ),
            axis=1,
        ).astype(observations["obs"].dtype, copy=False)
        # The source owner sanitizes the complete privileged frame after
        # clipping/scaling.  Preserve that boundary here so malformed sensor
        # packets cannot poison the custom value function or history buffers.
        np.nan_to_num(observations["critic"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if observations["critic"].shape[1] != COMPACT_PRIVILEGED_OBS_DIM:
            raise RuntimeError(
                "compact Wheelbipe privileged observation contract produced "
                f"{observations['critic'].shape}"
            )
        return observations

    def update_state(self, state: NpEnvState) -> NpEnvState:
        state = super().update_state(state)
        state.info["wheelbipe_variant"] = getattr(self._cfg, "variant_name", "unknown")
        state.info["wheelbipe_source_state_machine"] = getattr(
            self._cfg, "source_state_machine_status", "unknown"
        )
        state.info["wheelbipe_source_gimbal"] = getattr(
            self._cfg, "source_gimbal_status", "unknown"
        )
        return state


class WheelbipeCustomEnv(WheelbipeVariantEnv):
    _cfg: WheelbipeCustomFlatCfg


# ---------------------------------------------------------------------------
# Registry entries.  The exact upstream Gym identifiers are retained as
# strings (including hyphens) so migration tools can map them mechanically;
# UniLab's Hydra owner names below use snake_case.


def _register_pair(
    name: str, cfg_type: type[WheelbipeVariantCfg], env_type: type[WheelbipeV14Env]
) -> None:
    registry.register_env_config(name, cfg_type)
    registry.register_env(name, env_type, "mujoco")
    registry.register_env(name, env_type, "motrix")


_register_pair("WheelbipeV14FlatV0", WheelbipeFlatV0Cfg, WheelbipeVariantEnv)
_register_pair("WheelbipeV14FlatV1", WheelbipeFlatV1Cfg, WheelbipeVariantEnv)
_register_pair("WheelbipeV14FlatV2", WheelbipeFlatV2Cfg, WheelbipeVariantEnv)
_register_pair("WheelbipeV14FlatPlayV0", WheelbipeFlatPlayV0Cfg, WheelbipeVariantEnv)
_register_pair("WheelbipeV14FlatPlayV2", WheelbipeFlatPlayV2Cfg, WheelbipeVariantEnv)
_register_pair("WheelbipeV14RoughV0", WheelbipeRoughV0Cfg, WheelbipeV14RoughEnv)
_register_pair("WheelbipeV14RoughV1", WheelbipeRoughV1Cfg, WheelbipeV14RoughEnv)
_register_pair("WheelbipeV14RoughPlayV0", WheelbipeRoughPlayV0Cfg, WheelbipeV14RoughEnv)
_register_pair("WheelbipeV14RoughPlayV1", WheelbipeRoughPlayV1Cfg, WheelbipeV14RoughEnv)
_register_pair("WheelbipeV14FlatDreamWaQ", WheelbipeDreamWaQCfg, WheelbipeCustomEnv)
_register_pair("WheelbipeV14FlatDreamWaQPlay", WheelbipeDreamWaQPlayCfg, WheelbipeCustomEnv)
_register_pair("WheelbipeV14FlatHIM", WheelbipeHIMCfg, WheelbipeCustomEnv)
_register_pair("WheelbipeV14FlatHIMPlay", WheelbipeHIMPlayCfg, WheelbipeCustomEnv)
_register_pair("WheelbipeV14FlatNP3OBarlow", WheelbipeNP3OCfg, WheelbipeCustomEnv)
_register_pair("WheelbipeV14FlatNP3OBarlowPlay", WheelbipeNP3OPlayCfg, WheelbipeCustomEnv)


# Keep the exact Gymnasium id strings published by SCUTRobotLab available to
# callers migrating task-selection metadata.  These are UniLab registry names
# (and are additionally exposed by the thin Gymnasium adapter), not a second
# implementation: the UniLab environment lifecycle/API remains authoritative.
# The snake/camel-case names above remain canonical registry names for Hydra
# owner configs.  Each exact alias selects its named bounded owner and both
# backends are registered at the same validation boundary.
UPSTREAM_WHEELBIPE_TASK_IDS: dict[str, tuple[type[WheelbipeVariantCfg], type[WheelbipeV14Env]]] = {
    "Robotics-Wheelbipe-V14-Flat-v0": (WheelbipeFlatV0Cfg, WheelbipeVariantEnv),
    "Robotics-Wheelbipe-V14-Flat-v1": (WheelbipeFlatV1Cfg, WheelbipeVariantEnv),
    "Robotics-Wheelbipe-V14-Flat-v2": (WheelbipeFlatV2Cfg, WheelbipeVariantEnv),
    "Robotics-Wheelbipe-V14-Flat-Play-v0": (WheelbipeFlatPlayV0Cfg, WheelbipeVariantEnv),
    "Robotics-Wheelbipe-V14-Flat-Play-v2": (WheelbipeFlatPlayV2Cfg, WheelbipeVariantEnv),
    "Robotics-Wheelbipe-V14-Rough-v0": (WheelbipeRoughV0Cfg, WheelbipeV14RoughEnv),
    "Robotics-Wheelbipe-V14-Rough-v1": (WheelbipeRoughV1Cfg, WheelbipeV14RoughEnv),
    "Robotics-Wheelbipe-V14-Rough-Play-v0": (WheelbipeRoughPlayV0Cfg, WheelbipeV14RoughEnv),
    "Robotics-Wheelbipe-V14-Rough-Play-v1": (WheelbipeRoughPlayV1Cfg, WheelbipeV14RoughEnv),
    "Robotics-Wheelbipe-V14-Flat-DreamWaQ-v0": (WheelbipeDreamWaQCfg, WheelbipeCustomEnv),
    "Robotics-Wheelbipe-V14-Flat-DreamWaQ-Play-v0": (
        WheelbipeDreamWaQPlayCfg,
        WheelbipeCustomEnv,
    ),
    "Robotics-Wheelbipe-V14-Flat-HIM-v0": (WheelbipeHIMCfg, WheelbipeCustomEnv),
    "Robotics-Wheelbipe-V14-Flat-HIM-Play-v0": (WheelbipeHIMPlayCfg, WheelbipeCustomEnv),
    "Robotics-Wheelbipe-V14-Flat-NP3OBarlow-v0": (WheelbipeNP3OCfg, WheelbipeCustomEnv),
    "Robotics-Wheelbipe-V14-Flat-NP3OBarlow-Play-v0": (
        WheelbipeNP3OPlayCfg,
        WheelbipeCustomEnv,
    ),
}

for _upstream_task_id, (_cfg_type, _env_type) in UPSTREAM_WHEELBIPE_TASK_IDS.items():
    _register_pair(_upstream_task_id, _cfg_type, _env_type)


__all__ = [
    "COMPACT_POLICY_OBS_DIM",
    "WheelbipeVariantCfg",
    "WheelbipeFlatV0Cfg",
    "WheelbipeFlatV1Cfg",
    "WheelbipeFlatV2Cfg",
    "WheelbipeFlatPlayV0Cfg",
    "WheelbipeFlatPlayV2Cfg",
    "WheelbipeRoughV0Cfg",
    "WheelbipeRoughV1Cfg",
    "WheelbipeRoughPlayV0Cfg",
    "WheelbipeRoughPlayV1Cfg",
    "WheelbipeCustomFlatCfg",
    "WheelbipeDreamWaQCfg",
    "WheelbipeDreamWaQPlayCfg",
    "WheelbipeHIMCfg",
    "WheelbipeHIMPlayCfg",
    "WheelbipeNP3OCfg",
    "WheelbipeNP3OPlayCfg",
    "WheelbipeVariantEnv",
    "WheelbipeCustomEnv",
    "UPSTREAM_WHEELBIPE_TASK_IDS",
]
