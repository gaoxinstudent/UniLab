"""UniLab Wheelbipe V14 flat locomotion task.

This is the owner-layer implementation of the normal (35 observation, six
action) policy shipped with the public Wheelbipe deployment example.  Isaac
Lab and ROS2 are deliberately not imported here; the only simulator surface
used by the task is :class:`~unilab.base.backend.SimBackend`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import SimBackend, create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import (
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dr.dr_utils import (
    build_interval_push_plan,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import (
    Commands,
    apply_heading_yaw_feedback,
    sample_heading_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.common.terrain_spawn import TerrainCurriculumCfg, TerrainSpawnManager
from unilab.utils.geometry import np_roll_pitch_from_quat
from unilab.utils.rotation import (
    np_quat_apply,
    np_quat_apply_batched,
    np_quat_apply_inverse,
    np_quat_conjugate_batched,
    np_quat_from_euler_xyz,
    np_quat_mul,
    np_wrap_to_pi,
    np_yaw_from_quat,
    np_yaw_to_quat,
)

from .base import (
    COMPACT_POLICY_OBS_DIM,
    COMPACT_PRIVILEGED_OBS_DIM,
    NORMAL_CONTROL_MODE,
    NUM_LEG_ACTIONS,
    NUM_NATIVE_ACTUATORS,
    NUM_POLICY_ACTIONS,
    NUM_WHEEL_ACTIONS,
    POLICY_OBS_DIM,
    PRIVILEGED_OBS_DIM,
    WHEELBIPE_OBS_DELAY_ALIASES,
    WheelbipeControlConfig,
    WheelbipeDelayBuffer,
    WheelbipeGimbalConfig,
    WheelbipeNoiseConfig,
    WheelbipeV14BaseCfg,
    WheelbipeV14BaseEnv,
    build_wheelbipe_policy_observation,
    build_wheelbipe_timing_contract,
    canonical_wheelbipe_delay_range_semantics,
    compute_wheelbipe_motor_ctrl,
    map_policy_action_to_native_targets,
    normalize_wheelbipe_delay_range,
    resolve_wheelbipe_torque_limits,
    sample_wheelbipe_delay_lags,
)
from .gimbal_asset import (
    WHEEL_POSITION_SENSOR_NAMES,
    materialize_wheelbipe_gimbal_asset,
    materialize_wheelbipe_state_machine_asset,
)
from .semantics import (
    SOURCE_V14_LEG_MASS_BODY_NAMES,
    SOURCE_V14_PRIVILEGED_POLICY_PERMUTATION,
    SOURCE_V14_RESET_CONTACT_BODY_NAMES,
    SOURCE_V14_RESET_JOINT_NAMES,
    SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES,
    SOURCE_V14_WHEEL_BODY_NAMES,
    SourceV14HIMCurriculum,
    SourceV14RewardParameters,
    apply_source_v14_termination_duration,
    build_source_v14_critic_observation,
    build_source_v14_height_signals,
    compute_source_v14_reward,
    sample_source_v14_commands,
    sample_source_v14_material_buckets,
    source_v14_inverse_kinematics,
)
from .state_machine import WheelbipeStateMachine, WheelbipeStateMachineConfig
from .task_modes import WheelbipeGimbalSpinTranslateConfig

logger = logging.getLogger(__name__)


@dataclass
class WheelbipeInitState:
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.38])


@dataclass
class WheelbipeCommands(Commands):
    """Command distribution used by the normal deployment policy."""

    vel_limit: list[list[float]] = field(
        # Matches the V14 source command envelope.  The lateral component is
        # reserved by the deployment contract and is clamped to zero by the
        # owner sampler below.
        default_factory=lambda: [[-2.7, 0.0, -2.0 * np.pi], [2.7, 0.0, 2.0 * np.pi]]
    )
    resampling_time: float = 5.0
    resampling_time_range: list[float] = field(default_factory=lambda: [5.0, 15.0])
    heading_command: bool = True
    heading_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    heading_control_stiffness: float = 5.0
    rel_standing_envs: float = 0.1
    rel_heading_envs: float = 0.5
    source_curriculum_enabled: bool = True
    special_mode_min_episode_time: float = 5.0
    special_mode_start_iterations: list[int] = field(default_factory=lambda: [3000, 4000, 2000])
    special_mode_probabilities: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.30])
    gimbal_mode_probability: float = 0.0
    gimbal_mode_start_iteration: int = 0
    zero_command_probability: float = 0.0
    zero_command_start_iteration: int = 0
    training_progress_steps_per_iteration: int = 24
    training_iteration_offset: int = 0


@dataclass
class WheelbipeJointFrictionRandomizationConfig:
    """Pinned EventCfgV14 startup joint-friction distributions.

    MuJoCo and Motrix expose one Coulomb ``frictionloss`` coefficient, so the
    independently sampled PhysX static/dynamic pair is deliberately collapsed
    to the sampled static coefficient.  ``viscous_backend_mode`` makes the
    backend-specific viscous conversion explicit in owner config and the
    runtime DR contract.
    """

    enabled: bool = True
    front_static_range: list[float] = field(default_factory=lambda: [0.25, 1.0])
    front_dynamic_range: list[float] = field(default_factory=lambda: [0.25, 1.0])
    front_viscous_range: list[float] = field(default_factory=lambda: [0.05, 0.2])
    rear_static_range: list[float] = field(default_factory=lambda: [0.25, 1.0])
    rear_dynamic_range: list[float] = field(default_factory=lambda: [0.25, 1.0])
    rear_viscous_range: list[float] = field(default_factory=lambda: [0.05, 0.2])
    wheel_static_range: list[float] = field(default_factory=lambda: [0.05, 0.25])
    wheel_viscous_range: list[float] = field(default_factory=lambda: [0.0, 0.01])
    inactive_static_range: list[float] = field(default_factory=lambda: [0.05, 0.1])
    inactive_viscous_range: list[float] = field(default_factory=lambda: [0.01, 0.025])
    gimbal_static_range: list[float] = field(default_factory=lambda: [0.002, 0.01])
    gimbal_viscous_range: list[float] = field(default_factory=lambda: [0.002, 0.01])
    coulomb_backend_mode: Literal["static_to_frictionloss_dynamic_collapsed"] = (
        "static_to_frictionloss_dynamic_collapsed"
    )
    viscous_backend_mode: Literal["auto", "dof_damping", "unsupported_omitted"] = "auto"


_SOURCE_JOINT_FRICTION_NAMES: dict[str, tuple[str, ...]] = {
    "front": ("left_front1_joint", "right_front1_joint"),
    "rear": ("left_rear1_joint", "right_rear1_joint"),
    "wheel": ("left_wheel_joint", "right_wheel_joint"),
    "inactive": (
        "left_rear2_joint",
        "right_rear2_joint",
        "left_front2_joint",
        "right_front2_joint",
        "left_front3_joint",
        "right_front3_joint",
        "left_front4_joint",
        "right_front4_joint",
        "left_spring1_joint",
        "right_spring1_joint",
    ),
    "gimbal": ("gimbal_yaw_joint", "gimbal_pitch_joint"),
}


@dataclass
class WheelbipeDomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_z_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    init_roll_range: list[float] = field(default_factory=lambda: [-0.15, 0.15])
    init_pitch_range: list[float] = field(default_factory=lambda: [-0.15, 0.15])
    init_yaw_range: list[float] = field(default_factory=lambda: [-3.14, 3.14])
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.75, 1.25])
    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.75, 1.25])
    base_mass_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.3])
    leg_mass_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    wheel_mass_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    com_offset_x: list[float] = field(default_factory=lambda: [-0.04, 0.04])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.02, 0.02])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.02, 0.02])
    randomize_body_material: bool = False
    base_static_friction_range: list[float] = field(default_factory=lambda: [0.01, 0.1])
    base_dynamic_friction_range: list[float] = field(default_factory=lambda: [0.01, 0.1])
    base_restitution_range: list[float] = field(default_factory=lambda: [0.02, 0.2])
    wheel_static_friction_range: list[float] = field(default_factory=lambda: [0.5, 1.2])
    wheel_dynamic_friction_range: list[float] = field(default_factory=lambda: [0.4, 1.0])
    wheel_restitution_range: list[float] = field(default_factory=lambda: [0.02, 0.2])
    material_num_buckets: int = 64
    material_make_consistent: bool = True
    source_guide_material_requested: bool = True
    source_guide_material_missing_target_status: Literal["not_applicable_missing_target"] = (
        "not_applicable_missing_target"
    )
    guide_static_friction_range: list[float] = field(default_factory=lambda: [0.1, 0.7])
    guide_dynamic_friction_range: list[float] = field(default_factory=lambda: [0.1, 0.7])
    guide_restitution_range: list[float] = field(default_factory=lambda: [0.01, 0.1])
    guide_material_num_buckets: int = 8
    joint_friction: WheelbipeJointFrictionRandomizationConfig = field(
        default_factory=WheelbipeJointFrictionRandomizationConfig
    )
    spring_damping_multiplier_range: list[float] = field(default_factory=lambda: [0.5, 1.5])
    leg_effort_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.1])
    wheel_effort_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    control_randomization_min_steps: int = 720
    source_passive_gain_conversion_status: Literal["unsupported_unmaterialized_actuator"] = (
        "unsupported_unmaterialized_actuator"
    )
    use_leg_random_start: bool = False
    use_predefined_leg_random_start: bool = False
    leg_length_range: list[float] = field(default_factory=lambda: [0.13, 0.32])
    leg_angle_range: list[float] = field(default_factory=lambda: [-0.5 * np.pi, 0.75 * np.pi])
    wheel_angle_range: list[float] = field(default_factory=lambda: [-2.0 * np.pi, 2.0 * np.pi])
    leg_joint_vel_range: list[float] = field(default_factory=lambda: [-0.5 * np.pi, 0.5 * np.pi])
    wheel_joint_vel_range: list[float] = field(default_factory=lambda: [-50.0, 50.0])
    # The source reset code uses the nested positive/negative mode weights
    # (0.3 + 0.2 = 0.5) whenever the modes mapping is present.  The legacy
    # top-level ``prob: 0.2`` is only a fallback for a single mode.
    predefined_ground_probability: float = 0.5
    predefined_ground_root_height: float = 0.25
    predefined_ground_zero_torque_seconds: float = 0.2
    predefined_ground_command_seconds: float = 1.5
    predefined_ground_command_x_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    predefined_ground_command_y_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    predefined_ground_command_yaw_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    # Explicit source predefined-air reset owner. Flat/Rough-v1 enables the
    # 0.3 ``air_1`` profile through its state-machine config; Rough-Play-v0
    # installs a state-machine-independent 1.0 ``play_air`` profile. Keeping
    # the ranges in config avoids inferring a reset envelope from an optional
    # task machine in the DR hot path.
    predefined_air_enabled: bool = False
    predefined_air_probability: float = 0.0
    predefined_air_pose_z_range: list[float] = field(default_factory=lambda: [0.12, 0.42])
    predefined_air_pose_roll_range: list[float] = field(default_factory=lambda: [-0.1, 0.1])
    predefined_air_pose_pitch_range: list[float] = field(default_factory=lambda: [-0.2, 0.2])
    predefined_air_pose_yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    predefined_air_body_vel_x_range: list[float] = field(default_factory=lambda: [-2.5, 2.5])
    predefined_air_body_vel_y_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    predefined_air_body_vel_z_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    predefined_air_body_ang_vel_roll_range: list[float] = field(
        default_factory=lambda: [-0.05, 0.05]
    )
    predefined_air_body_ang_vel_pitch_range: list[float] = field(
        default_factory=lambda: [-0.1, 0.1]
    )
    predefined_air_body_ang_vel_yaw_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    predefined_air_leg_length_range: list[float] = field(default_factory=lambda: [0.15, 0.35])
    predefined_air_leg_angle_range: list[float] = field(
        default_factory=lambda: [-0.25 * np.pi, 0.25 * np.pi]
    )
    predefined_air_command_limits_enabled: bool = True
    predefined_air_command_seconds: float = 3.0
    predefined_air_command_x_range: list[float] = field(default_factory=lambda: [-2.5, 2.5])
    predefined_air_command_y_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    predefined_air_command_yaw_range: list[float] = field(
        default_factory=lambda: [-0.5 * np.pi, 0.5 * np.pi]
    )
    predefined_air_height_range: list[float] = field(default_factory=lambda: [0.18, 0.43])
    source_push_velocity_enabled: bool = True
    source_push_interval_range: list[float] = field(default_factory=lambda: [5.0, 10.0])
    source_push_velocity_range: list[list[float]] = field(
        default_factory=lambda: [[-0.25, 0.25], [-0.25, 0.25], [0.0, 0.0]]
    )
    source_external_force_enabled: bool = True
    source_external_force_interval_range: list[float] = field(default_factory=lambda: [5.0, 10.0])
    source_external_force_range: list[float] = field(default_factory=lambda: [-10.0, 10.0])
    source_external_torque_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])


@dataclass
class WheelbipeHIMCurriculumConfig:
    """Pinned ``CurriculumCfgV14`` owner profile used only by HIM training."""

    enabled: bool = False
    reward_key: str = "track_height_exp"
    num_steps_per_env: int = 24
    window_size: int = 64
    min_stage_episodes: int = 64
    normalize_by_episode_length: bool = True
    restore_defaults_after_final_threshold: bool = True
    assist_apply_on_compute: bool = True
    assist_body_name: str = "base_link"
    force_interaction: Literal["shared_wrench_buffer_overwrite"] = "shared_wrench_buffer_overwrite"
    reward_stage_weights: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"track_height_exp": 1.0, "track_height_exp_tight": 1.0},
            {"track_height_exp": 0.8, "track_height_exp_tight": 0.6},
        ]
    )
    assist_force_z_stages: list[float] = field(default_factory=lambda: [160.0, 80.0, 0.0])
    thresholds: list[float] = field(default_factory=lambda: [0.4, 0.4])
    stage_min_episodes: list[int] = field(default_factory=lambda: [500, 500])

    def validate(self, default_reward_scales: Mapping[str, float]) -> None:
        if not bool(self.assist_apply_on_compute):
            raise ValueError("pinned HIM vertical assist requires apply_on_compute=true")
        if str(self.assist_body_name) != "base_link":
            raise ValueError("pinned HIM vertical assist must target base_link")
        if str(self.force_interaction) != "shared_wrench_buffer_overwrite":
            raise ValueError(
                "pinned HIM assist and interval disturbance share one overwrite wrench buffer"
            )
        # Constructing the pure runtime is the single validation path for
        # shape, key and threshold/count contracts.
        SourceV14HIMCurriculum(
            default_reward_scales=default_reward_scales,
            reward_key=self.reward_key,
            num_steps_per_env=self.num_steps_per_env,
            window_size=self.window_size,
            min_stage_episodes=self.min_stage_episodes,
            normalize_by_episode_length=self.normalize_by_episode_length,
            reward_stage_weights=self.reward_stage_weights,
            assist_force_z_stages=self.assist_force_z_stages,
            thresholds=self.thresholds,
            stage_min_episodes=self.stage_min_episodes,
            restore_defaults_after_final_threshold=self.restore_defaults_after_final_threshold,
        )


@dataclass
class WheelbipeRewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_z": -0.5,
            "ang_vel_xy": -0.05,
            "base_height": -2.0,
            "orientation": -1.0,
            "action_rate": -0.01,
            "action_smooth": -0.01,
            "joint_torques_l2": -1.0e-4,
            "wheel_vel": -1.0e-5,
            "wheel_power": -1.0e-4,
            "joint_acc_l2": -5.0e-7,
            "alive": 0.5,
            "upright": 0.25,
        }
    )
    tracking_sigma: float = 0.25
    base_height_target: float = 0.22
    only_positive_rewards: bool = False
    orientation_x_bias: float = 2.0
    orientation_x_sigma: float = 3.0
    orientation_x_amplitude: float = 2.0
    orientation_y_bias: float = 2.0
    orientation_y_sigma: float = 3.0
    orientation_y_amplitude: float = 2.0
    orientation_x_square_sigma: float = 4.0
    orientation_y_square_sigma: float = 2.0
    orientation_x_exp_sigma: float = 0.01
    orientation_y_exp_sigma: float = 0.02
    lin_vel_sigma: float = 0.5
    lin_vel_tight_sigma: float = 0.1
    lin_vel_square_sigma: float = 0.25
    lin_vel_error_constraint: float = 1.0
    ang_vel_sigma: float = 0.25
    ang_vel_square_sigma: float = 0.5
    ang_vel_error_constraint: float = 0.8
    height_sigma: float = 0.025
    height_soft_sigma: float = 0.1
    height_tight_sigma: float = 0.001
    height_square_sigma: float = 10.0
    height_error_constraint: float = 0.15
    no_fork_distance: float = 0.05
    no_fork_square_sigma: float = 5.0
    no_fork_exp_sigma: float = 1.0e-4
    no_fork_z_distance: float = 0.05
    no_fork_z_exp_sigma: float = 1.0e-4
    stand_still_deadzone: float = 0.1
    stand_still_deadzone_enabled: bool = True


@dataclass
class WheelbipeSensor:
    # These names are present in the vendored robot description.  MuJoCo's
    # ``framezaxis``/``upvector`` sensor is retained for the upright
    # termination check.  It is a world-frame body +Z axis, whereas the
    # policy and orientation rewards consume projected gravity in body frame;
    # that latter quantity is derived from the backend base quaternion below.
    local_linvel: str = "local_linvel"
    gyro: str = "gyro"
    upvector: str = "upvector"


@registry.envcfg("WheelbipeV14Flat")
@dataclass
class WheelbipeV14FlatCfg(WheelbipeV14BaseCfg):
    # The source samples settled physics state. Without this refresh MuJoCo
    # sensors lag qpos/qvel by one substep, adding an unintended IMU delay.
    post_step_forward_sensor: bool = True
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(
                ASSETS_ROOT_PATH / "robots" / "wheelbipe_v14_2" / "mjcf" / "scene_flat.xml"
            ),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "wheelbipe_v14_2" / "locomotion_task.xml")
            ],
        )
    )
    init_state: WheelbipeInitState = field(default_factory=WheelbipeInitState)
    commands: WheelbipeCommands = field(default_factory=WheelbipeCommands)
    gimbal_spin_translate: WheelbipeGimbalSpinTranslateConfig = field(
        default_factory=WheelbipeGimbalSpinTranslateConfig
    )
    reward_config: WheelbipeRewardConfig | None = None
    him_curriculum: WheelbipeHIMCurriculumConfig = field(
        default_factory=WheelbipeHIMCurriculumConfig
    )
    sensor: WheelbipeSensor = field(default_factory=WheelbipeSensor)  # type: ignore[assignment]
    domain_rand: WheelbipeDomainRandConfig = field(default_factory=WheelbipeDomainRandConfig)
    height_range: list[float] = field(default_factory=lambda: [0.20, 0.42])
    # Source privileged observations always retain world-frame root z.  Rough
    # and airborne owners set ``use_absolute_height=false`` only for the reward
    # reference, where terrain height is subtracted after the optional v1 clip.
    use_absolute_height: bool = True
    height_obs_clip_enabled: bool = False
    height_obs_clip_range: list[float | None] = field(default_factory=lambda: [None, None])
    # Released body scanner: 20 mm square, 10 mm spacing, aligned to yaw.
    # Cache these offsets at construction; the terrain contract supplies
    # surface heights without asset access in the reward path.
    source_height_scan_xy: list[list[float]] = field(
        default_factory=lambda: [[x, y] for x in (-0.01, 0.0, 0.01) for y in (-0.01, 0.0, 0.01)]
    )
    training_semantics: str = "legacy"
    # The legacy successful V14 run kept RSL-RL's randomized episode length
    # buffer separate from NpEnv's source reset timers.  Keep that lifecycle
    # choice explicit: enabling it is an opt-in experiment because consuming
    # the randomized age changes the first rollout distribution substantially.
    source_episode_age_sync: bool = False
    termination_duration_enabled: bool = True
    termination_duration_steps: int = 20
    termination_roll_deg: float = 40.0
    termination_pitch_deg: float = 40.0
    terminate_joint_vel_abs: float = 500.0
    terminate_root_ang_vel_abs: float = 200.0
    terminate_root_lin_vel_abs: float = 100.0
    terminate_on_obs_outlier: bool = True
    terminate_obs_abs: float = 120.0
    # The pinned V14 observation-outlier path is coupled to its value-debug
    # counter.  All 15 published owners keep this false, so the raw actor/
    # critic cache is still populated but cannot satisfy the ``counter > 10``
    # gate.  Do not substitute the unrelated global environment step counter.
    debug_value_diagnosis: bool = False
    undesired_contact_force_threshold: float = 3.0
    desired_contact_force_threshold: float = 5.0
    ctrl_mode_obs_scale: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]
    )
    # Source velocity-tracking gate.  It is disabled by the common Flat
    # owner and enabled only by the NP3O training profile.
    vel_height_gate_enabled: bool = False
    vel_height_gate_mode: str = "linear_band"
    vel_height_gate_full_error: float = 0.05
    vel_height_gate_zero_error: float = 0.1
    vel_height_gate_tracker_sigma: float = 0.02

    def validate(self) -> None:
        super().validate()
        scan = np.asarray(self.source_height_scan_xy, dtype=np.float64)
        if (
            scan.ndim != 2
            or scan.shape[0] == 0
            or scan.shape[1] != 2
            or not np.isfinite(scan).all()
        ):
            raise ValueError("source_height_scan_xy must be a non-empty finite [N, 2] array")
        if isinstance(self.him_curriculum, dict):
            self.him_curriculum = WheelbipeHIMCurriculumConfig(**self.him_curriculum)
        if not isinstance(self.him_curriculum, WheelbipeHIMCurriculumConfig):
            raise ValueError("him_curriculum must be WheelbipeHIMCurriculumConfig")
        if self.him_curriculum.enabled:
            if self.reward_config is None:
                raise ValueError("enabled him_curriculum requires reward_config")
            self.him_curriculum.validate(self.reward_config.scales)
        # Exercise the pure owner contract here so malformed Hydra bounds fail
        # before a backend or terrain asset is materialized.
        build_source_v14_height_signals(
            np.zeros((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            use_absolute_height=bool(self.use_absolute_height),
            clip_enabled=bool(self.height_obs_clip_enabled),
            clip_range=self.height_obs_clip_range,
        )
        mode = str(self.vel_height_gate_mode).strip().lower()
        if mode not in {"linear_band", "band", "piecewise_linear", "exp", "exponential"}:
            raise ValueError(f"unsupported vel_height_gate_mode {self.vel_height_gate_mode!r}")
        if float(self.vel_height_gate_full_error) < 0.0:
            raise ValueError("vel_height_gate_full_error must be non-negative")
        if float(self.vel_height_gate_zero_error) < float(self.vel_height_gate_full_error):
            raise ValueError("vel_height_gate_zero_error must be >= vel_height_gate_full_error")
        if float(self.vel_height_gate_tracker_sigma) <= 0.0:
            raise ValueError("vel_height_gate_tracker_sigma must be positive")


def _sample_range(values: list[float] | tuple[float, ...], *, name: str) -> tuple[float, float]:
    bounds = np.asarray(values, dtype=np.float64).reshape(-1)
    if bounds.shape != (2,):
        raise ValueError(f"{name} must contain two values, got {bounds.shape}")
    return float(min(bounds)), float(max(bounds))


def _source_reset_profile_mask(
    profile: dict[str, np.ndarray],
    name: str,
    num_reset: int,
) -> np.ndarray:
    if name not in profile:
        raise ValueError(f"source terrain reset profile is missing {name!r}")
    value = np.asarray(profile[name])
    if value.shape != (num_reset,):
        raise ValueError(
            f"source terrain reset profile {name!r} must have shape "
            f"({num_reset},), got {value.shape}"
        )
    return value.astype(bool, copy=False)


def build_wheelbipe_backend_reset_randomization(
    env: Any,
    num_reset: int,
    env_ids: np.ndarray | None = None,
) -> ResetRandomizationPayload | None:
    """Build backend-supported reset randomization, excluding PD gains.

    Gains are owner-level state because the model uses torque motors rather
    than MuJoCo position actuators.  They are sampled separately and consumed
    by the pre-step callback.
    """

    cfg = getattr(env, "cfg", None)
    domain_rand = getattr(cfg, "domain_rand", None)
    if domain_rand is None:
        return None
    payload = ResetRandomizationPayload()
    if bool(getattr(env, "_source_semantics", False)):
        if env_ids is None:
            ids = np.arange(num_reset, dtype=np.intp)
        else:
            ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
            if ids.size != int(num_reset):
                raise ValueError("env_ids size must match num_reset")
        if getattr(domain_rand, "randomize_body_mass", False):
            table = getattr(env, "_source_body_mass", None)
            if table is None:
                raise ValueError("source body-mass randomization was not materialized on init")
            payload.body_mass = np.asarray(table, dtype=np.float64)[ids].copy()
        if getattr(domain_rand, "random_com", False):
            offsets = getattr(env, "_source_base_com_offset", None)
            if offsets is None:
                raise ValueError("source COM randomization was not materialized on init")
            payload.base_com_offset = np.asarray(offsets, dtype=np.float64)[ids].copy()
        if getattr(domain_rand, "randomize_body_material", False):
            friction = getattr(env, "_source_geom_friction", None)
            if friction is None:
                raise ValueError("source material randomization was not materialized on init")
            payload.geom_friction = np.asarray(friction, dtype=np.float64)[ids].copy()
        return None if payload.is_empty() else payload
    if getattr(domain_rand, "randomize_base_mass", False):
        low, high = _sample_range(domain_rand.added_mass_range, name="added_mass_range")
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))
    if getattr(domain_rand, "randomize_body_mass", False):
        # Body-mass randomization is deliberately materialized once during
        # environment construction.  The reset path only scales this cached
        # table, so no model/XML metadata is consulted from the simulation
        # hot path.  Both MuJoCo and Motrix expose this through SimBackend.
        base_body_mass = getattr(env, "_base_body_mass", None)
        if base_body_mass is None:
            raise ValueError("body mass randomization requires a cached body-mass table")
        template = np.asarray(base_body_mass, dtype=np.float64).reshape(-1)
        low, high = _sample_range(
            domain_rand.body_mass_multiplier_range,
            name="body_mass_multiplier_range",
        )
        multipliers = np.random.uniform(low, high, size=(num_reset, template.size))
        body_mass = np.broadcast_to(template, multipliers.shape).copy()
        positive = template > 0.0
        body_mass[:, positive] *= multipliers[:, positive]
        payload.body_mass = body_mass
    if getattr(domain_rand, "random_com", False):
        low, high = _sample_range(domain_rand.com_offset_x, name="com_offset_x")
        offset = np.zeros((num_reset, 3), dtype=np.float64)
        offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        if getattr(domain_rand, "com_offset_y", None) is not None:
            low, high = _sample_range(domain_rand.com_offset_y, name="com_offset_y")
            offset[:, 1] = np.random.uniform(low, high, size=(num_reset,))
        if getattr(domain_rand, "com_offset_z", None) is not None:
            low, high = _sample_range(domain_rand.com_offset_z, name="com_offset_z")
            offset[:, 2] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = offset
    if getattr(domain_rand, "randomize_gravity", False):
        gravity_range = np.asarray(domain_rand.gravity_range, dtype=np.float64)
        if gravity_range.shape != (2, 3):
            raise ValueError(f"gravity_range must have shape (2, 3), got {gravity_range.shape}")
        payload.gravity = np.random.uniform(
            np.minimum(gravity_range[0], gravity_range[1]),
            np.maximum(gravity_range[0], gravity_range[1]),
            size=(num_reset, 3),
        )
    if getattr(domain_rand, "randomize_ground_friction", False):
        base_friction = getattr(env, "_base_geom_friction", None)
        ground_id = getattr(env, "_ground_geom_id", None)
        if base_friction is None or ground_id is None:
            raise ValueError("ground friction randomization requires a cached floor geom")
        low, high = _sample_range(
            domain_rand.ground_friction_multiplier_range,
            name="ground_friction_multiplier_range",
        )
        friction = np.broadcast_to(
            np.asarray(base_friction, dtype=np.float64),
            (num_reset, *np.asarray(base_friction).shape),
        ).copy()
        friction[:, int(ground_id), 0] *= np.random.uniform(low, high, size=(num_reset,))
        payload.geom_friction = friction
    if getattr(domain_rand, "randomize_dof_armature", False):
        base_dof_armature = getattr(env, "_base_dof_armature", None)
        if base_dof_armature is None:
            raise ValueError("dof armature randomization requires a cached dof-armature table")
        template = np.asarray(base_dof_armature, dtype=np.float64).reshape(-1)
        low, high = _sample_range(
            domain_rand.dof_armature_multiplier_range,
            name="dof_armature_multiplier_range",
        )
        armature = np.broadcast_to(template, (num_reset, template.size)).copy()
        positive = template > 0.0
        armature[:, positive] *= np.random.uniform(
            low,
            high,
            size=(num_reset, int(np.count_nonzero(positive))),
        )
        payload.dof_armature = armature
    return None if payload.is_empty() else payload


class WheelbipeV14DomainRandomizationProvider(LocomotionDRProvider):
    def build_init_randomization_plan(self, env: Any) -> InitRandomizationPlan | None:
        if not bool(getattr(env, "_source_semantics", False)):
            return None
        joint_cfg = env.cfg.domain_rand.joint_friction
        if not bool(joint_cfg.enabled):
            return None
        frictionloss = getattr(env, "_source_dof_frictionloss", None)
        if frictionloss is None:
            raise ValueError("source joint friction was not materialized on the cold path")
        return InitRandomizationPlan(
            dof_frictionloss=np.asarray(frictionloss, dtype=np.float64).copy(),
            dof_damping=(
                None
                if env._source_dof_damping is None
                else np.asarray(env._source_dof_damping, dtype=np.float64).copy()
            ),
        )

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        payload = build_wheelbipe_backend_reset_randomization(env, 1)
        if payload is not None:
            unsupported = capabilities.get_unsupported_reset_terms(payload.requested_terms())
            if unsupported:
                # The manager will filter unsupported terms with a warning.  Do
                # not fail construction merely because an optional DR term is
                # unavailable on a backend.
                pass
        validate_interval_push_support(env, capabilities)
        if bool(getattr(env, "_source_semantics", False)):
            domain_rand = env.cfg.domain_rand
            if (
                bool(domain_rand.source_push_velocity_enabled)
                and not capabilities.supports_interval_body_velocity_delta
            ):
                raise NotImplementedError(
                    f"{env._backend.backend_type} does not support source V14 root velocity push"
                )
            if bool(domain_rand.source_external_force_enabled):
                if not capabilities.supports_interval_body_force:
                    raise NotImplementedError(
                        f"{env._backend.backend_type} does not support source V14 interval body force"
                    )
                if not capabilities.supports_interval_body_torque:
                    raise NotImplementedError(
                        f"{env._backend.backend_type} does not support source V14 interval body torque"
                    )
            if (
                bool(env.cfg.him_curriculum.enabled)
                and not capabilities.supports_interval_body_force
            ):
                raise NotImplementedError(
                    f"{env._backend.backend_type} does not support source V14 HIM assist force"
                )

            push_ranges = np.asarray(domain_rand.source_push_velocity_range, dtype=np.float64)
            if push_ranges.shape != (3, 2):
                raise ValueError(
                    f"source_push_velocity_range must have shape (3, 2), got {push_ranges.shape}"
                )
            if np.any(push_ranges[:, 1] < push_ranges[:, 0]):
                raise ValueError("source_push_velocity_range upper bounds must be >= lower bounds")
            for name, values in (
                ("source_push_interval_range", domain_rand.source_push_interval_range),
                (
                    "source_external_force_interval_range",
                    domain_rand.source_external_force_interval_range,
                ),
                ("source_external_force_range", domain_rand.source_external_force_range),
                ("source_external_torque_range", domain_rand.source_external_torque_range),
            ):
                _sample_range(values, name=name)

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        if bool(getattr(env, "_source_semantics", False)):
            domain_rand = env.cfg.domain_rand
            push_enabled = bool(domain_rand.source_push_velocity_enabled)
            force_enabled = bool(domain_rand.source_external_force_enabled)
            assist_enabled = bool(env.cfg.him_curriculum.enabled)
            if push_enabled or force_enabled or assist_enabled:
                velocity_delta = None
                if push_enabled:
                    push_due = int(step_counter) >= np.asarray(env._source_push_next_step)
                    if np.any(push_due):
                        count = int(np.count_nonzero(push_due))
                        ranges = np.asarray(
                            domain_rand.source_push_velocity_range, dtype=np.float64
                        )
                        velocity_delta = np.zeros((env._num_envs, 1, 3), dtype=env._np_dtype)
                        velocity_delta[push_due, 0, :] = np.random.uniform(
                            ranges[:, 0], ranges[:, 1], size=(count, 3)
                        ).astype(env._np_dtype)
                        env._reset_source_push_timers(np.flatnonzero(push_due))

                body_force = None
                body_torque = None
                if force_enabled:
                    force_due = int(step_counter) >= np.asarray(env._source_force_next_step)
                    if np.any(force_due):
                        count = int(np.count_nonzero(force_due))
                        force_low, force_high = _sample_range(
                            domain_rand.source_external_force_range,
                            name="source_external_force_range",
                        )
                        torque_low, torque_high = _sample_range(
                            domain_rand.source_external_torque_range,
                            name="source_external_torque_range",
                        )
                        env._source_body_force[force_due] = np.random.uniform(
                            force_low, force_high, size=(count, 3)
                        ).astype(env._np_dtype)
                        env._source_body_torque[force_due] = np.random.uniform(
                            torque_low, torque_high, size=(count, 3)
                        ).astype(env._np_dtype)
                        # Both pinned manager terms write the articulation's
                        # same external-wrench buffer.  The interval event is
                        # later than the reset/assist write, so it replaces
                        # (rather than adds to) the vertical assist for these
                        # environments until reset or a global assist-stage
                        # transition writes the assist again.
                        env._source_him_assist_latched[force_due] = False
                        env._reset_source_force_timers(np.flatnonzero(force_due))
                    body_force = env._source_body_force[:, None, :].copy()
                    body_torque = env._source_body_torque[:, None, :].copy()

                assist_force_z = float(env.source_him_assist_force_z)
                if assist_enabled:
                    if body_force is None:
                        body_force = np.zeros((env._num_envs, 1, 3), dtype=env._np_dtype)
                    latched = np.asarray(env._source_him_assist_latched, dtype=bool)
                    body_force[latched, 0, :] = 0.0
                    body_force[latched, 0, 2] = assist_force_z
                    if body_torque is not None:
                        body_torque[latched, 0, :] = 0.0

                return IntervalRandomizationPlan(
                    body_ids=env._base_body_ids,
                    body_linear_velocity_delta=velocity_delta,
                    body_force=body_force,
                    body_torque=body_torque,
                )
        return build_interval_push_plan(env, step_counter)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands, threshold=0.08)
        # The normal deployment policy reserves the lateral command slot.
        commands[:, 1] = 0.0
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if getattr(env.cfg.commands, "heading_command", False):
            commands[:, 2] = 0.0
        return commands

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        env_ids = np.asarray(env_ids, dtype=np.int32)
        num_reset = int(env_ids.size)
        if bool(getattr(env, "_source_semantics", False)):
            return self._build_source_reset_plan(env, env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.25, 0.25, size=(num_reset, 2))
        z_low, z_high = _sample_range(env.cfg.domain_rand.init_z_range, name="init_z_range")
        qpos[:, 2] += np.random.uniform(z_low, z_high, size=(num_reset,))
        roll_low, roll_high = _sample_range(
            env.cfg.domain_rand.init_roll_range, name="init_roll_range"
        )
        pitch_low, pitch_high = _sample_range(
            env.cfg.domain_rand.init_pitch_range, name="init_pitch_range"
        )
        yaw_low, yaw_high = _sample_range(env.cfg.domain_rand.init_yaw_range, name="init_yaw_range")
        roll = np.random.uniform(roll_low, roll_high, size=(num_reset,))
        pitch = np.random.uniform(pitch_low, pitch_high, size=(num_reset,))
        yaw = (
            np.random.uniform(yaw_low, yaw_high, size=(num_reset,))
            if env.cfg.domain_rand.randomize_init_yaw
            else np.zeros((num_reset,), dtype=get_global_dtype())
        )
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_quat_from_euler_xyz(roll, pitch, yaw))
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3], yaw=yaw)
        qvel[:, 0:6] = np.random.uniform(-0.25, 0.25, size=(num_reset, 6)).astype(
            get_global_dtype()
        )

        airborne_reset = np.zeros((num_reset,), dtype=bool)
        state_machine = getattr(env, "_state_machine", None)
        if state_machine is not None:
            probability = float(state_machine.cfg.reset_airborne_probability)
            if probability > 0.0:
                airborne_reset = np.random.uniform(size=(num_reset,)) < probability
                if np.any(airborne_reset):
                    # Match the source v1 reset envelope while retaining a
                    # deterministic owner-level geometric contact contract.
                    qpos[airborne_reset, 2] = np.random.uniform(
                        0.12, 0.42, size=int(np.count_nonzero(airborne_reset))
                    )
                    qvel[airborne_reset, 0:6] = np.random.uniform(
                        -0.10,
                        0.10,
                        size=(int(np.count_nonzero(airborne_reset)), 6),
                    ).astype(get_global_dtype())
            state_machine.reset(env_ids, airborne=airborne_reset)

        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)
        env.sample_reset_spring_force(env_ids)
        commands = self._sample_commands(env, num_reset)
        heights = np.random.uniform(
            float(env.cfg.height_range[0]),
            float(env.cfg.height_range[1]),
            size=(num_reset,),
        ).astype(get_global_dtype())
        zeros = zero_actions(num_reset, NUM_POLICY_ACTIONS)
        info_updates: dict[str, Any] = {
            "commands": commands,
            "height_commands": heights,
            "current_actions": zeros.copy(),
            "last_actions": zeros.copy(),
            "previous_actions": zeros.copy(),
            "native_targets": np.zeros(
                (num_reset, int(getattr(env, "_num_native_actuators", NUM_NATIVE_ACTUATORS))),
                dtype=get_global_dtype(),
            ),
            "torques": np.zeros(
                (num_reset, int(getattr(env, "_num_native_actuators", NUM_NATIVE_ACTUATORS))),
                dtype=get_global_dtype(),
            ),
            "qacc": np.zeros((num_reset, NUM_POLICY_ACTIONS), dtype=get_global_dtype()),
            "motor_kp": motor_kp.astype(get_global_dtype()),
            "motor_kd": motor_kd.astype(get_global_dtype()),
            # Keep the mode tail explicit in the state contract.  Variant
            # owners may replace this per-env vector before the next
            # observation; the normal deployment profile remains one-hot
            # ``[1, 0, ..., 0]``.
            "control_mode_obs": np.broadcast_to(
                NORMAL_CONTROL_MODE.astype(get_global_dtype()), (num_reset, 7)
            ).copy(),
        }
        if state_machine is not None:
            mode = state_machine.control_mode_obs(state_dtype=get_global_dtype())
            info_updates["control_mode_obs"] = mode[env_ids]
            info_updates["state_machine_state"] = state_machine.state[env_ids].copy()
            info_updates["state_machine_failure"] = state_machine.failure[env_ids].copy()
        if getattr(env.cfg.commands, "heading_command", False):
            info_updates["heading_commands"] = sample_heading_commands(env, num_reset)
        # Keep the terrain curriculum's distance baseline synchronized with
        # every reset.  BaseSpawnManager is a no-op for flat scenes, while
        # TerrainSpawnManager uses this cold/reset-path record to promote or
        # demote the next episode without any asset lookup in ``step``.
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_wheelbipe_backend_reset_randomization(env, num_reset),
        )

    def _build_source_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        """Build the pinned source V14 reset without simulator-private access."""

        num_reset = int(env_ids.size)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        init_xyz = np.asarray(env.cfg.init_state.pos, dtype=np.float64).reshape(3)
        qpos[:, 0:3] = init_xyz
        qvel.fill(0.0)

        roll_low, roll_high = _sample_range(
            env.cfg.domain_rand.init_roll_range, name="init_roll_range"
        )
        pitch_low, pitch_high = _sample_range(
            env.cfg.domain_rand.init_pitch_range, name="init_pitch_range"
        )
        yaw_low, yaw_high = _sample_range(env.cfg.domain_rand.init_yaw_range, name="init_yaw_range")
        roll = np.random.uniform(roll_low, roll_high, size=num_reset)
        pitch = np.random.uniform(pitch_low, pitch_high, size=num_reset)
        yaw = (
            np.random.uniform(yaw_low, yaw_high, size=num_reset)
            if env.cfg.domain_rand.randomize_init_yaw
            else np.zeros((num_reset,), dtype=np.float64)
        )
        qpos[:, 3:7] = np_quat_from_euler_xyz(roll, pitch, yaw)

        state_machine = getattr(env, "_state_machine", None)
        airborne_reset = np.zeros((num_reset,), dtype=bool)
        ground_reset = np.zeros((num_reset,), dtype=bool)
        reset_mode_sample = np.random.uniform(size=num_reset)
        terrain_reset_profile = env._source_terrain_reset_profile(env_ids)
        disable_airborne_reset = _source_reset_profile_mask(
            terrain_reset_profile,
            "disable_predefined_reset_air",
            num_reset,
        )
        disable_ground_reset = _source_reset_profile_mask(
            terrain_reset_profile,
            "disable_predefined_reset_ground",
            num_reset,
        )
        axis_aligned_reset = _source_reset_profile_mask(
            terrain_reset_profile,
            "reset_heading_axis_aligned_only",
            num_reset,
        )
        configured_air_reset = bool(env.cfg.domain_rand.predefined_air_enabled)
        if configured_air_reset:
            airborne_probability = float(
                np.clip(env.cfg.domain_rand.predefined_air_probability, 0.0, 1.0)
            )
        elif state_machine is not None:
            airborne_probability = float(
                np.clip(state_machine.cfg.reset_airborne_probability, 0.0, 1.0)
            )
        else:
            airborne_probability = 0.0
        airborne_reset = (reset_mode_sample < airborne_probability) & (~disable_airborne_reset)

        # The exact Flat/Rough-v1 identities enable source leg randomization
        # through their state-machine profile.  Keep that identity intact for
        # registry construction even when no Hydra domain-rand override was
        # applied to the inherited compatibility defaults.
        use_leg_random_start = (
            bool(env.cfg.domain_rand.use_leg_random_start)
            or state_machine is not None
            or configured_air_reset
        )
        if use_leg_random_start:
            length_low, length_high = _sample_range(
                env.cfg.domain_rand.leg_length_range, name="leg_length_range"
            )
            angle_low, angle_high = _sample_range(
                env.cfg.domain_rand.leg_angle_range, name="leg_angle_range"
            )
            leg_length = np.random.uniform(length_low, length_high, size=(num_reset, 2))
            leg_angle = np.random.uniform(angle_low, angle_high, size=(num_reset, 2))
            if (
                bool(env.cfg.domain_rand.use_predefined_leg_random_start)
                or state_machine is not None
                or configured_air_reset
            ):
                if state_machine is None:
                    probability = np.clip(
                        float(env.cfg.domain_rand.predefined_ground_probability), 0.0, 1.0
                    )
                    ground_reset = (
                        (reset_mode_sample >= airborne_probability)
                        & (reset_mode_sample < airborne_probability + probability)
                        & (~disable_ground_reset)
                    )
                    positive_probability = 0.6
                else:
                    # Pinned Flat/Rough-v1 samples the 0.3 airborne bucket
                    # first, followed by mutually-exclusive 0.2 positive and
                    # 0.1 negative ground buckets.
                    probability = min(0.3, max(1.0 - airborne_probability, 0.0))
                    ground_reset = (reset_mode_sample >= airborne_probability) & (
                        reset_mode_sample < airborne_probability + probability
                    )
                    ground_reset &= ~disable_ground_reset
                    positive_probability = 2.0 / 3.0
                count = int(np.count_nonzero(ground_reset))
                if count:
                    # The source first assigns a ground-reset env, then samples
                    # positive/negative geometry independently for each leg
                    # with normalized 0.3/0.2 mode weights.
                    positive = np.random.uniform(size=(count, 2)) < positive_probability
                    sampled_length = np.random.uniform(0.14, 0.36, size=(count, 2))
                    sampled_height = np.empty((count, 2), dtype=np.float64)
                    sampled_height[positive] = np.random.uniform(
                        -0.06, 0.12, size=int(np.count_nonzero(positive))
                    )
                    sampled_height[~positive] = np.random.uniform(
                        -0.06, 0.0, size=int(np.count_nonzero(~positive))
                    )
                    ratio = np.clip(
                        sampled_height / np.maximum(sampled_length, 1.0e-6),
                        -1.0 + 1.0e-6,
                        1.0 - 1.0e-6,
                    )
                    leg_length[ground_reset] = sampled_length
                    leg_angle[ground_reset] = np.where(
                        positive, np.arccos(ratio), -np.arccos(ratio)
                    )
                    qpos[ground_reset, 2] = float(env.cfg.domain_rand.predefined_ground_root_height)
                    # The predefined ground reset writes the source asset's
                    # default root orientation, rather than retaining the
                    # ordinary reset event's randomized roll/pitch/yaw.
                    roll[ground_reset] = 0.0
                    pitch[ground_reset] = 0.0
                    yaw[ground_reset] = 0.0
                    qpos[ground_reset, 3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=qpos.dtype)
            if np.any(airborne_reset):
                count = int(np.count_nonzero(airborne_reset))
                # Root linear velocity is sampled in the reset yaw frame,
                # matching source ``reset_root_state_uniform_vel_b``.
                air_leg_length_low, air_leg_length_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_leg_length_range,
                    name="predefined_air_leg_length_range",
                )
                air_leg_angle_low, air_leg_angle_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_leg_angle_range,
                    name="predefined_air_leg_angle_range",
                )
                leg_length[airborne_reset] = np.random.uniform(
                    air_leg_length_low,
                    air_leg_length_high,
                    size=(count, 2),
                )
                leg_angle[airborne_reset] = np.random.uniform(
                    air_leg_angle_low,
                    air_leg_angle_high,
                    size=(count, 2),
                )
                air_roll_low, air_roll_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_pose_roll_range,
                    name="predefined_air_pose_roll_range",
                )
                air_pitch_low, air_pitch_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_pose_pitch_range,
                    name="predefined_air_pose_pitch_range",
                )
                air_yaw_low, air_yaw_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_pose_yaw_range,
                    name="predefined_air_pose_yaw_range",
                )
                air_roll = np.random.uniform(air_roll_low, air_roll_high, size=count)
                air_pitch = np.random.uniform(air_pitch_low, air_pitch_high, size=count)
                air_yaw = np.random.uniform(air_yaw_low, air_yaw_high, size=count)
                roll[airborne_reset] = air_roll
                pitch[airborne_reset] = air_pitch
                yaw[airborne_reset] = air_yaw
                # ``reset_root_state_uniform_vel_b`` interprets its z range
                # as an offset from the asset's default root pose.  Preserve
                # that distinction before the terrain spawn manager adds its
                # own cold-path origin/clearance correction.
                air_z_low, air_z_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_pose_z_range,
                    name="predefined_air_pose_z_range",
                )
                qpos[airborne_reset, 2] = init_xyz[2] + np.random.uniform(
                    air_z_low, air_z_high, size=count
                )
                qpos[airborne_reset, 3:7] = np_quat_from_euler_xyz(air_roll, air_pitch, air_yaw)
                body_x_low, body_x_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_body_vel_x_range,
                    name="predefined_air_body_vel_x_range",
                )
                body_y_low, body_y_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_body_vel_y_range,
                    name="predefined_air_body_vel_y_range",
                )
                body_z_low, body_z_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_body_vel_z_range,
                    name="predefined_air_body_vel_z_range",
                )
                roll_vel_low, roll_vel_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_body_ang_vel_roll_range,
                    name="predefined_air_body_ang_vel_roll_range",
                )
                pitch_vel_low, pitch_vel_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_body_ang_vel_pitch_range,
                    name="predefined_air_body_ang_vel_pitch_range",
                )
                yaw_vel_low, yaw_vel_high = _sample_range(
                    env.cfg.domain_rand.predefined_air_body_ang_vel_yaw_range,
                    name="predefined_air_body_ang_vel_yaw_range",
                )
                body_x = np.random.uniform(body_x_low, body_x_high, size=count)
                body_y = np.random.uniform(body_y_low, body_y_high, size=count)
                cosine = np.cos(air_yaw)
                sine = np.sin(air_yaw)
                qvel[airborne_reset, 0] = cosine * body_x - sine * body_y
                qvel[airborne_reset, 1] = sine * body_x + cosine * body_y
                qvel[airborne_reset, 2] = np.random.uniform(body_z_low, body_z_high, size=count)
                qvel[airborne_reset, 3] = np.random.uniform(roll_vel_low, roll_vel_high, size=count)
                qvel[airborne_reset, 4] = np.random.uniform(
                    pitch_vel_low, pitch_vel_high, size=count
                )
                qvel[airborne_reset, 5] = np.random.uniform(yaw_vel_low, yaw_vel_high, size=count)
            reset_joint_pos = source_v14_inverse_kinematics(leg_length, leg_angle)
            qpos[:, 7 + env._source_reset_pos_indices] = reset_joint_pos
            wheel_low, wheel_high = _sample_range(
                env.cfg.domain_rand.wheel_angle_range, name="wheel_angle_range"
            )
            qpos[:, 7 + env._wheel_pos_indices] = np.random.uniform(
                wheel_low, wheel_high, size=(num_reset, NUM_WHEEL_ACTIONS)
            )
            leg_vel_low, leg_vel_high = _sample_range(
                env.cfg.domain_rand.leg_joint_vel_range, name="leg_joint_vel_range"
            )
            wheel_vel_low, wheel_vel_high = _sample_range(
                env.cfg.domain_rand.wheel_joint_vel_range, name="wheel_joint_vel_range"
            )
            qvel[:, 6 + env._policy_vel_indices[:NUM_LEG_ACTIONS]] = np.random.uniform(
                leg_vel_low, leg_vel_high, size=(num_reset, NUM_LEG_ACTIONS)
            )
            qvel[:, 6 + env._policy_vel_indices[NUM_LEG_ACTIONS:]] = np.random.uniform(
                wheel_vel_low, wheel_vel_high, size=(num_reset, NUM_WHEEL_ACTIONS)
            )

        if np.any(axis_aligned_reset):
            yaw_candidates = np.asarray(
                [0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi],
                dtype=np.float64,
            )
            count = int(np.count_nonzero(axis_aligned_reset))
            yaw[axis_aligned_reset] = yaw_candidates[
                np.random.randint(0, yaw_candidates.size, size=count)
            ]
            qpos[axis_aligned_reset, 3:7] = np_quat_from_euler_xyz(
                roll[axis_aligned_reset],
                pitch[axis_aligned_reset],
                yaw[axis_aligned_reset],
            )

        gimbal_reset_info: dict[str, np.ndarray] = {}
        if env._gimbal_enabled:
            # Source ``_reset_gimbal_joints`` writes the articulation state,
            # not only its controller target buffers.  Express that write in
            # the owner reset plan so the shared ``SimBackend.set_state``
            # boundary applies it identically in MuJoCo and Motrix.
            gimbal_reset_info = env._apply_source_gimbal_reset_to_plan(
                env_ids,
                base_heading=yaw,
                qpos=qpos,
                qvel=qvel,
            )

        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3], yaw=yaw)
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        env._reset_source_episode_buffers(
            env_ids,
            ground_reset=ground_reset,
            airborne_reset=airborne_reset,
        )
        if state_machine is not None:
            state_machine.reset(env_ids, airborne=airborne_reset)

        controls = env.sample_reset_source_control_randomization(env_ids)
        env.set_source_control_randomization(env_ids, **controls)
        env.sample_reset_spring_force(env_ids)
        sampled = env._sample_source_commands(
            current_yaw=np_yaw_from_quat(qpos[:, 3:7]),
            episode_steps=np.zeros((num_reset,), dtype=np.uint32),
        )
        height_low, height_high = _sample_range(env.cfg.height_range, name="height_range")
        heights = np.random.uniform(height_low, height_high, size=num_reset).astype(
            get_global_dtype()
        )
        source_commands = np.asarray(sampled["commands"], dtype=get_global_dtype()).copy()
        if np.any(ground_reset):
            ground_ids = env_ids[ground_reset]
            env._source_ground_restore_command[ground_ids] = source_commands[ground_reset]
            ground_ranges = (
                env.cfg.domain_rand.predefined_ground_command_x_range,
                env.cfg.domain_rand.predefined_ground_command_y_range,
                env.cfg.domain_rand.predefined_ground_command_yaw_range,
            )
            ground_low_high = np.asarray(
                [
                    _sample_range(value, name=f"predefined_ground_command_{axis}_range")
                    for value, axis in zip(ground_ranges, ("x", "y", "yaw"), strict=True)
                ],
                dtype=np.float64,
            )
            override = np.random.uniform(
                ground_low_high[:, 0],
                ground_low_high[:, 1],
                size=(ground_ids.size, 3),
            ).astype(get_global_dtype())
            env._source_ground_override_command[ground_ids] = override
            source_commands[ground_reset] = override
        if np.any(airborne_reset) and bool(
            env.cfg.domain_rand.predefined_air_command_limits_enabled
        ):
            air_ranges = (
                env.cfg.domain_rand.predefined_air_command_x_range,
                env.cfg.domain_rand.predefined_air_command_y_range,
                env.cfg.domain_rand.predefined_air_command_yaw_range,
            )
            air_low_high = np.asarray(
                [
                    _sample_range(value, name=f"predefined_air_command_{axis}_range")
                    for value, axis in zip(air_ranges, ("x", "y", "yaw"), strict=True)
                ],
                dtype=np.float64,
            )
            source_commands[airborne_reset] = np.clip(
                source_commands[airborne_reset],
                air_low_high[:, 0],
                air_low_high[:, 1],
            )
            air_height_low, air_height_high = _sample_range(
                env.cfg.domain_rand.predefined_air_height_range,
                name="predefined_air_height_range",
            )
            heights[airborne_reset] = np.clip(
                heights[airborne_reset], air_height_low, air_height_high
            )
        sampled["commands"] = source_commands
        sampled, heights = env._source_apply_terrain_command_profile(
            env_ids,
            sampled,
            heights,
            current_yaw=np_yaw_from_quat(qpos[:, 3:7]),
            force_resample=True,
        )
        if not isinstance(sampled, dict):
            raise TypeError("source terrain command profile hook must return a command dict")
        heights = np.asarray(heights, dtype=get_global_dtype())
        if heights.shape != (num_reset,):
            raise ValueError(
                "source terrain command profile height_commands must have shape "
                f"({num_reset},), got {heights.shape}"
            )
        commands_after_profile = np.asarray(sampled.get("commands"))
        if commands_after_profile.shape != (num_reset, 3):
            raise ValueError(
                "source terrain command profile commands must have shape "
                f"({num_reset}, 3), got {commands_after_profile.shape}"
            )
        duration_low, duration_high = _sample_range(
            env.cfg.commands.resampling_time_range,
            name="commands.resampling_time_range",
        )
        duration = np.random.uniform(duration_low, duration_high, size=num_reset)
        resample_steps = np.maximum(np.ceil(duration / float(env.cfg.ctrl_dt)).astype(np.int32), 1)
        zeros = zero_actions(num_reset, NUM_POLICY_ACTIONS)
        info_updates: dict[str, Any] = {
            **sampled,
            "height_commands": heights,
            "command_resample_steps_remaining": resample_steps,
            "command_resample_generation": np.zeros((num_reset,), dtype=np.int64),
            "current_actions": zeros.copy(),
            "last_actions": zeros.copy(),
            "previous_actions": zeros.copy(),
            "native_targets": np.zeros(
                (num_reset, env._num_native_actuators), dtype=get_global_dtype()
            ),
            "torques": np.zeros((num_reset, env._num_native_actuators), dtype=get_global_dtype()),
            "qacc": np.zeros((num_reset, NUM_POLICY_ACTIONS), dtype=get_global_dtype()),
            "motor_kp": controls["motor_kp"].astype(get_global_dtype()),
            "motor_kd": controls["motor_kd"].astype(get_global_dtype()),
            "control_mode_obs": np.broadcast_to(
                NORMAL_CONTROL_MODE.astype(get_global_dtype()), (num_reset, 7)
            ).copy(),
            **gimbal_reset_info,
        }
        if state_machine is not None:
            mode = state_machine.control_mode_obs(state_dtype=get_global_dtype())
            info_updates["control_mode_obs"] = mode[env_ids]
            info_updates["state_machine_state"] = state_machine.state[env_ids].copy()
            info_updates["state_machine_failure"] = state_machine.failure[env_ids].copy()
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_wheelbipe_backend_reset_randomization(
                env, num_reset, env_ids=env_ids
            ),
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # ``LocomotionDRProvider`` obtains the configured up-vector sensor as
        # a generic reset input.  For Wheelbipe that sensor is world-frame
        # ``framezaxis`` data, not the body-frame projected gravity expected by
        # the policy.  Recompute from the backend quaternion after reset so
        # roll/pitch starts follow the same contract as the hot path.  The
        # unused argument is kept for the provider hook signature.
        del gravity
        projected_gravity = np.asarray(env.get_projected_gravity(), dtype=get_global_dtype())[
            np.asarray(env_ids, dtype=np.intp)
        ]
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates,
                linvel,
                gyro,
                projected_gravity,
                dof_pos,
                dof_vel,
                env_ids=np.asarray(env_ids, dtype=np.int32),
            ),
        )


@registry.env("WheelbipeV14Flat", sim_backend="mujoco")
@registry.env("WheelbipeV14Flat", sim_backend="motrix")
class WheelbipeV14Env(WheelbipeV14BaseEnv):
    _cfg: WheelbipeV14FlatCfg

    def _source_reset_contact_body_names(self) -> tuple[str, ...]:
        """Return source V14's physical-reset contact body set.

        The upstream owner keeps this set separate from the diagnostic
        ``base_link`` contact feature.  Rough owners override the hook for
        the non-plane gimbal-only reset rule; resolving names remains a
        construction-time operation.
        """

        return SOURCE_V14_RESET_CONTACT_BODY_NAMES

    def _source_terrain_reset_profile(self, env_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Return reset-time terrain masks through an explicit owner hook.

        Flat owners use the all-false identity. Rough state-machine owners may
        override this without adding backend-specific access to the DR provider.
        """

        count = int(np.asarray(env_ids).size)
        return {
            "disable_predefined_reset_air": np.zeros((count,), dtype=bool),
            "disable_predefined_reset_ground": np.zeros((count,), dtype=bool),
            "reset_heading_axis_aligned_only": np.zeros((count,), dtype=bool),
        }

    def _source_apply_terrain_command_profile(
        self,
        env_ids: np.ndarray,
        sampled: dict[str, Any],
        height_commands: np.ndarray,
        *,
        current_yaw: np.ndarray,
        force_resample: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Apply a terrain command profile; flat owners preserve the sample."""

        del env_ids, current_yaw, force_resample
        return sampled, height_commands

    def __init__(self, cfg: WheelbipeV14FlatCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        semantics = str(getattr(cfg, "training_semantics", "legacy")).strip().lower()
        if semantics not in {"legacy", "source_v14"}:
            raise ValueError(
                "training_semantics must be 'legacy' or 'source_v14', "
                f"got {cfg.training_semantics!r}"
            )
        self._source_semantics = semantics == "source_v14"
        joint_friction_cfg = cfg.domain_rand.joint_friction
        if isinstance(joint_friction_cfg, Mapping):
            joint_friction_cfg = WheelbipeJointFrictionRandomizationConfig(
                **dict(joint_friction_cfg)
            )
            cfg.domain_rand.joint_friction = joint_friction_cfg
        if not isinstance(joint_friction_cfg, WheelbipeJointFrictionRandomizationConfig):
            raise ValueError(
                "Wheelbipe domain_rand.joint_friction must be "
                "WheelbipeJointFrictionRandomizationConfig"
            )
        gimbal_cfg = getattr(cfg, "gimbal", WheelbipeGimbalConfig())
        if isinstance(gimbal_cfg, dict):
            gimbal_cfg = WheelbipeGimbalConfig(**gimbal_cfg)
            cfg.gimbal = gimbal_cfg
        if not isinstance(gimbal_cfg, WheelbipeGimbalConfig):
            raise ValueError("Wheelbipe gimbal configuration must be WheelbipeGimbalConfig")
        gimbal_cfg.validate()
        backend_kwargs = env_backend_kwargs(cfg)
        if self._source_semantics or (
            bool(gimbal_cfg.enabled)
            and str(gimbal_cfg.control_mode).strip().lower() == "heading_pd"
        ):
            # The source reward and 78D critic consume wheel body position and
            # body-frame linear velocity; heading-PD additionally consumes
            # gimbal-yaw-link world angular velocity.  Tracking sensors are
            # materialized once on the backend cold path; hot code only calls
            # SimBackend.
            backend_kwargs["add_body_sensors"] = True
        state_machine_cfg = getattr(cfg, "state_machine", WheelbipeStateMachineConfig())
        if isinstance(state_machine_cfg, dict):
            state_machine_cfg = WheelbipeStateMachineConfig(**state_machine_cfg)
            cfg.state_machine = state_machine_cfg
        if not isinstance(state_machine_cfg, WheelbipeStateMachineConfig):
            raise ValueError("Wheelbipe state_machine must be WheelbipeStateMachineConfig")
        state_machine_cfg.validate()
        if backend_type == "motrix":
            backend_kwargs["motrix_disable_equality"] = bool(cfg.motrix_disable_equality)
        materialized_gimbal = None
        scene_cfg = cfg.scene
        if bool(gimbal_cfg.enabled):
            if scene_cfg is None or not scene_cfg.model_file:
                raise ValueError("gimbal-enabled Wheelbipe owners require SceneCfg.model_file")
            materialized_gimbal = materialize_wheelbipe_gimbal_asset(
                scene_cfg.model_file,
                scene_cfg.fragment_files,
            )
            scene_cfg = replace(
                scene_cfg,
                model_file=materialized_gimbal.model_file,
                fragment_files=list(materialized_gimbal.fragment_files),
            )
        elif bool(state_machine_cfg.enabled):
            if scene_cfg is None or not scene_cfg.model_file:
                raise ValueError("state-machine Wheelbipe owners require SceneCfg.model_file")
            materialized_gimbal = materialize_wheelbipe_state_machine_asset(
                scene_cfg.model_file,
                scene_cfg.fragment_files,
            )
            scene_cfg = replace(
                scene_cfg,
                model_file=materialized_gimbal.model_file,
                fragment_files=list(materialized_gimbal.fragment_files),
            )
        try:
            backend = create_backend(
                backend_type,
                scene_cfg,
                num_envs,
                cfg.sim_dt,
                base_name=cfg.asset.base_name,
                push_body_name=cfg.domain_rand.push_body_name,
                **backend_kwargs,
            )
        except Exception:
            if materialized_gimbal is not None:
                materialized_gimbal.cleanup()
            raise
        terrain_spawn_data = backend.get_terrain_spawn_data()
        super().__init__(cfg, backend, num_envs)
        self._materialized_gimbal = materialized_gimbal
        self._materialized_scene = scene_cfg
        self._state_machine = (
            WheelbipeStateMachine(state_machine_cfg, num_envs)
            if bool(state_machine_cfg.enabled)
            else None
        )
        self._state_machine_wheel_body_ids = np.zeros((0,), dtype=np.int32)
        if self._state_machine is not None:
            self._state_machine_wheel_sensor_names = WHEEL_POSITION_SENSOR_NAMES
        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config
        self._source_runtime_reward_scales = dict(self._reward_cfg.scales)
        curriculum_cfg = cfg.him_curriculum
        self._source_him_curriculum = (
            SourceV14HIMCurriculum(
                default_reward_scales=self._reward_cfg.scales,
                reward_key=curriculum_cfg.reward_key,
                num_steps_per_env=curriculum_cfg.num_steps_per_env,
                window_size=curriculum_cfg.window_size,
                min_stage_episodes=curriculum_cfg.min_stage_episodes,
                normalize_by_episode_length=curriculum_cfg.normalize_by_episode_length,
                reward_stage_weights=curriculum_cfg.reward_stage_weights,
                assist_force_z_stages=curriculum_cfg.assist_force_z_stages,
                thresholds=curriculum_cfg.thresholds,
                stage_min_episodes=curriculum_cfg.stage_min_episodes,
                restore_defaults_after_final_threshold=(
                    curriculum_cfg.restore_defaults_after_final_threshold
                ),
            )
            if bool(curriculum_cfg.enabled)
            else None
        )
        if self._source_him_curriculum is not None:
            self._source_runtime_reward_scales = self._source_him_curriculum.reward_scales
        self._enable_reward_log = True
        self._terrain_surface_sample_height = (
            None if terrain_spawn_data is None else terrain_spawn_data.sample_height
        )
        self._source_height_scan_xy = np.asarray(cfg.source_height_scan_xy, dtype=self._np_dtype)
        self._source_height_spawn_margin = 0.0
        terrain_cfg = cfg.scene.terrain.generator if cfg.scene.terrain is not None else None
        if terrain_spawn_data is not None and terrain_cfg is not None:
            spawn_cfg = getattr(cfg, "terrain_curriculum", TerrainCurriculumCfg())
            self._source_height_spawn_margin = float(spawn_cfg.spawn_height_margin)
            self._spawn = TerrainSpawnManager(
                num_envs,
                terrain_spawn_data.terrain_origins,
                cell_size=float(terrain_cfg.size[0]),
                cfg=spawn_cfg,
                sample_height=terrain_spawn_data.sample_height,
            )

        self._last_policy_vel = np.zeros((num_envs, NUM_POLICY_ACTIONS), dtype=self._np_dtype)
        self._motor_kp = np.full(
            (num_envs, NUM_LEG_ACTIONS), float(cfg.control_config.Kp), dtype=np.float64
        )
        self._motor_kd = np.full(
            (num_envs, NUM_LEG_ACTIONS), float(cfg.control_config.Kd), dtype=np.float64
        )
        self._wheel_kd = np.full(
            (num_envs, NUM_WHEEL_ACTIONS), float(cfg.control_config.wheel_Kd), dtype=np.float64
        )
        self._last_motor_ctrl = np.zeros(
            (num_envs, self._num_native_actuators), dtype=self._np_dtype
        )
        self._gimbal_yaw_velocity_target = np.zeros((num_envs,), dtype=self._np_dtype)
        self._gimbal_heading_target = np.zeros((num_envs,), dtype=self._np_dtype)
        self._gimbal_pitch_target = np.full(
            (num_envs,), float(gimbal_cfg.pitch_target), dtype=self._np_dtype
        )
        self._gimbal_control_mode = str(gimbal_cfg.control_mode).strip().lower()
        gimbal_task_cfg = getattr(cfg, "gimbal_spin_translate", None)
        gimbal_v2_enabled = (
            bool(gimbal_task_cfg.get("enabled", False))
            if isinstance(gimbal_task_cfg, dict)
            else bool(getattr(gimbal_task_cfg, "enabled", False))
        )
        # The source v2 owner randomizes the heading-PD gains at startup
        # independently of whether the optional gimbal-spin command mode is
        # enabled.  Rough-v0 inherits that v2 heading controller but its
        # released normal-only checkpoint has no spin-mode feature tail, so
        # tying this randomization to ``gimbal_spin_translate.enabled`` would
        # silently drop a source startup event.
        source_heading_pd_randomization = (
            self._source_semantics
            and self._gimbal_control_mode == "heading_pd"
            and bool(getattr(gimbal_cfg, "randomize_heading", False))
        )
        if self._source_semantics and (gimbal_v2_enabled or source_heading_pd_randomization):
            # V14-v2 owns a separate startup event for the manual gimbal-yaw
            # heading controller; these gains are not the articulation's
            # reset-time actuator gain randomization below.
            self._gimbal_heading_kp = np.random.uniform(20.0, 40.0, size=num_envs)
            self._gimbal_heading_kd = np.random.uniform(0.05, 0.1, size=num_envs)
        else:
            self._gimbal_heading_kp = np.full(
                (num_envs,), float(gimbal_cfg.yaw_kp), dtype=np.float64
            )
            self._gimbal_heading_kd = np.full(
                (num_envs,), float(gimbal_cfg.yaw_kd), dtype=np.float64
            )
        # Spring preload randomization is sampled once per episode (matching
        # the source Isaac owner) and consumed by the pre-step controller.
        self._spring_force_random = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._spring_damping = np.full(
            (num_envs, 2), float(cfg.control_config.spring_damping), dtype=np.float64
        )
        self._leg_effort_scale = np.ones((num_envs, NUM_LEG_ACTIONS), dtype=np.float64)
        self._wheel_effort_scale = np.ones((num_envs, NUM_WHEEL_ACTIONS), dtype=np.float64)
        self._gimbal_yaw_kd = np.full((num_envs,), float(gimbal_cfg.velocity_kd), dtype=np.float64)
        self._gimbal_pitch_kp = np.full((num_envs,), float(gimbal_cfg.pitch_kp), dtype=np.float64)
        self._gimbal_pitch_kd = np.full((num_envs,), float(gimbal_cfg.pitch_kd), dtype=np.float64)
        self._init_delay_buffers()
        # Resolve owner safety limits against the materialized backend range
        # once. The exact V14 owner and the vendored MJCF both bound wheel
        # effort at ±5 N m; the intersection and provenance remain visible in
        # ``torque_contract`` instead of being inferred in the hot path.
        (
            self._ctrl_lower,
            self._ctrl_upper,
            self._torque_contract,
        ) = resolve_wheelbipe_torque_limits(
            cfg.control_config,
            np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64),
            native_leg_indices=self._native_leg_indices,
            native_wheel_indices=self._native_wheel_indices,
            native_spring_indices=self._native_spring_indices,
        )
        if self._gimbal_enabled:
            gimbal_limits = np.asarray(
                [gimbal_cfg.yaw_effort_limit, gimbal_cfg.pitch_effort_limit], dtype=np.float64
            )
            # The generated MJCF declares ±2/±10 ranges.  Intersect those
            # physical bounds with owner limits once on init, just like the
            # leg/wheel/spring groups above.
            actuator_ranges = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
            gimbal_slots = self._native_gimbal_indices
            effective = np.column_stack(
                (
                    np.maximum(actuator_ranges[gimbal_slots, 0], -gimbal_limits),
                    np.minimum(actuator_ranges[gimbal_slots, 1], gimbal_limits),
                )
            )
            if np.any(effective[:, 0] > effective[:, 1]):
                raise ValueError("WheelBipe gimbal owner/backend torque ranges do not intersect")
            self._ctrl_lower[gimbal_slots] = effective[:, 0].astype(self._np_dtype)
            self._ctrl_upper[gimbal_slots] = effective[:, 1].astype(self._np_dtype)
            self._torque_contract["groups"]["gimbal"] = {
                "requested_limits": [float(v) for v in gimbal_limits],
                "backend_range": actuator_ranges[gimbal_slots].tolist(),
                "effective_range": effective.tolist(),
                "slots": [int(v) for v in gimbal_slots],
            }
        self._ctrl_lower = self._ctrl_lower.astype(self._np_dtype, copy=False)
        self._ctrl_upper = self._ctrl_upper.astype(self._np_dtype, copy=False)
        # Cache optional model tables before materialization.  Reset-time
        # randomization scales these arrays without touching backend model
        # metadata, preserving the cold-path asset contract.
        self._base_body_mass = (
            np.asarray(self._backend.get_body_mass(), dtype=np.float64).copy()
            if cfg.domain_rand.randomize_body_mass
            else None
        )
        self._base_dof_armature = (
            np.asarray(self._backend.get_dof_armature(), dtype=np.float64).copy()
            if cfg.domain_rand.randomize_dof_armature
            else None
        )
        self._base_geom_friction = np.asarray(self._backend.get_geom_friction(), dtype=np.float64)
        ground_geom_id: int | None
        try:
            ground_geom_id = int(self._backend.get_geom_id(cfg.asset.ground))
        except (NotImplementedError, ValueError):
            ground_geom_id = None
        self._ground_geom_id = ground_geom_id
        self._init_source_semantics_state()
        self._init_reward_functions()
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        self._init_domain_randomization(WheelbipeV14DomainRandomizationProvider())

    _OBS_DELAY_ALIASES = WHEELBIPE_OBS_DELAY_ALIASES

    def _init_delay_buffers(self) -> None:
        """Materialize optional latency buffers on the environment init path.

        The source V14 task samples delays in physics steps.  Action buffers
        are therefore consumed from ``_pre_step_motor_control`` (which the
        backend invokes once per physics substep).  Observation buffers can be
        selected as ``physics`` for source-like sampling or ``control`` for a
        lower-overhead vectorized profile.  Neither setting claims exact
        source-loop or dynamics parity.
        """

        cfg = self._cfg
        self._delay_range_semantics = canonical_wheelbipe_delay_range_semantics(
            getattr(cfg, "delay_range_semantics", "inclusive")
        )
        self._timing_contract = build_wheelbipe_timing_contract(
            sim_dt=cfg.sim_dt,
            ctrl_dt=cfg.ctrl_dt,
            obs_delay_step_unit=getattr(cfg, "obs_delay_step_unit", "control"),
            use_obs_delay=bool(getattr(cfg, "use_obs_delay", False)),
            use_act_delay=bool(getattr(cfg, "use_act_delay", False)),
            delay_range_semantics=self._delay_range_semantics,
            delay_profile=getattr(cfg, "delay_profile", "local_physics"),
        )
        self._use_obs_delay = bool(getattr(cfg, "use_obs_delay", False))
        self._use_act_delay = bool(getattr(cfg, "use_act_delay", False))
        self._obs_delay_step_unit = str(getattr(cfg, "obs_delay_step_unit", "control")).lower()
        self._obs_delay_buffers: dict[str, WheelbipeDelayBuffer] = {}
        self._obs_delay_ranges: dict[str, tuple[int, int]] = {}
        self._act_delay_buffers: dict[str, WheelbipeDelayBuffer] = {}
        self._act_delay_ranges: dict[str, tuple[int, int]] = {}
        self._physics_delayed_obs: dict[str, np.ndarray] = {}
        # Isaac's DirectRLEnv loop leaves the observation delay cache marked
        # current after reset.  Consequently the first ``_apply_action``
        # substep does not append a frame; the following three substeps and
        # the settled post-step observation append the four source frames.
        # The backend callback has no substep argument, so the owner tracks
        # the count around each control-step call explicitly.
        self._physics_substep_count = 0
        # ``_init_delay_buffers`` is also a useful isolated contract helper in
        # tests and downstream owner subclasses.  Those callers may construct
        # an object with ``__new__`` and populate only the timing fields before
        # invoking this method; retain the historical eight-channel default in
        # that narrow case while normal environments always set the materialized
        # actuator count in ``_init_buffers`` first (eight or ten with gimbal).
        native_actuator_count = int(getattr(self, "_num_native_actuators", NUM_NATIVE_ACTUATORS))
        self._num_native_actuators = native_actuator_count
        self._delayed_native_targets = np.zeros(
            (self._num_envs, native_actuator_count), dtype=self._np_dtype
        )
        self._delay_reset_in_progress = False

        mode = str(getattr(cfg, "policy_observation_mode", "normal")).lower()
        mode_enabled = bool(getattr(cfg, "ctrl_mode_obs_enabled", True))
        mode_dim = int(getattr(cfg, "ctrl_mode_obs_dim", 7))
        if mode == "normal":
            if not mode_enabled or mode_dim != 7:
                raise ValueError(
                    "normal Wheelbipe V14 owners require ctrl_mode_obs_enabled=true "
                    "and ctrl_mode_obs_dim=7 for the 35D policy contract"
                )
        elif mode == "compact":
            if mode_enabled or mode_dim != 0:
                raise ValueError(
                    "compact Wheelbipe V14 owners require ctrl_mode_obs_enabled=false "
                    "and ctrl_mode_obs_dim=0 for the 28D history-policy contract"
                )
        else:
            raise ValueError(f"policy_observation_mode must be 'normal' or 'compact', got {mode!r}")
        if self._obs_delay_step_unit not in {"control", "physics"}:
            raise ValueError(
                "obs_delay_step_unit must be 'control' or 'physics', "
                f"got {self._obs_delay_step_unit!r}"
            )

        if self._use_obs_delay:
            history_len = int(getattr(cfg, "obs_history_len", 10))
            if history_len < 1:
                raise ValueError(f"obs_history_len must be positive, got {history_len}")
            default_lag = int(getattr(cfg, "obs_default_time_lag", 1))
            if default_lag < 0 or default_lag > history_len:
                raise ValueError(
                    f"obs_default_time_lag must be in [0, {history_len}], got {default_lag}"
                )
            raw_cfg = getattr(cfg, "obs_delay_cfg", {}) or {}
            if not isinstance(raw_cfg, dict) or not raw_cfg:
                raise ValueError("use_obs_delay=true requires a non-empty obs_delay_cfg")
            widths = {"gyro": 3, "gravity": 3, "joint_pos": 6, "joint_vel": 6}
            for raw_name, raw_range in raw_cfg.items():
                canonical = self._OBS_DELAY_ALIASES.get(str(raw_name))
                if canonical is None:
                    allowed = ", ".join(sorted(self._OBS_DELAY_ALIASES))
                    raise ValueError(
                        f"unsupported Wheelbipe observation delay group {raw_name!r}; "
                        f"expected one of: {allowed}"
                    )
                if canonical in self._obs_delay_buffers:
                    raise ValueError(f"duplicate observation delay group for {canonical!r}")
                bounds = normalize_wheelbipe_delay_range(
                    raw_range, name=f"obs_delay_cfg.{raw_name}"
                )
                max_lag = (
                    bounds[1]
                    if self._delay_range_semantics == "inclusive"
                    else (bounds[0] if bounds[0] == bounds[1] else bounds[1] - 1)
                )
                if max_lag > history_len:
                    raise ValueError(
                        f"obs_delay_cfg.{raw_name} max lag {max_lag} exceeds "
                        f"obs_history_len {history_len}"
                    )
                self._obs_delay_ranges[canonical] = bounds
                self._obs_delay_buffers[canonical] = WheelbipeDelayBuffer(
                    history_len, self._num_envs, widths[canonical], self._np_dtype
                )
                self._physics_delayed_obs[canonical] = np.zeros(
                    (self._num_envs, widths[canonical]), dtype=self._np_dtype
                )
                # A deterministic initial lag makes the first pre-reset
                # state well-defined; each reset samples the configured range.
                max_lag = (
                    bounds[1]
                    if self._delay_range_semantics == "inclusive"
                    else (bounds[0] if bounds[0] == bounds[1] else bounds[1] - 1)
                )
                initial_lag = min(max(default_lag, bounds[0]), max_lag)
                self._obs_delay_buffers[canonical].set_time_lag(initial_lag)

        if self._use_act_delay:
            if bool(getattr(cfg.control_config, "simulate_action_latency", False)):
                raise ValueError(
                    "use_act_delay and control_config.simulate_action_latency are "
                    "mutually exclusive; choose one latency contract"
                )
            history_len = int(getattr(cfg, "act_history_len", 5))
            if history_len < 1:
                raise ValueError(f"act_history_len must be positive, got {history_len}")
            raw_cfg = getattr(cfg, "act_delay_cfg", {}) or {}
            if not isinstance(raw_cfg, dict) or not raw_cfg:
                raise ValueError("use_act_delay=true requires a non-empty act_delay_cfg")
            widths = {"leg_actions": NUM_LEG_ACTIONS, "wheel_actions": NUM_WHEEL_ACTIONS}
            for group, raw_range in raw_cfg.items():
                group_name = str(group)
                if group_name not in widths:
                    raise ValueError(
                        f"unsupported Wheelbipe action delay group {group_name!r}; "
                        "expected 'leg_actions' or 'wheel_actions'"
                    )
                bounds = normalize_wheelbipe_delay_range(
                    raw_range, name=f"act_delay_cfg.{group_name}"
                )
                max_lag = (
                    bounds[1]
                    if self._delay_range_semantics == "inclusive"
                    else (bounds[0] if bounds[0] == bounds[1] else bounds[1] - 1)
                )
                if max_lag > history_len:
                    raise ValueError(
                        f"act_delay_cfg.{group_name} max lag {max_lag} exceeds "
                        f"act_history_len {history_len}"
                    )
                self._act_delay_ranges[group_name] = bounds
                self._act_delay_buffers[group_name] = WheelbipeDelayBuffer(
                    history_len, self._num_envs, widths[group_name], self._np_dtype
                )

    def _reset_delay_buffers(self, env_ids: np.ndarray) -> None:
        """Clear selected histories and sample one lag per env/episode."""

        raw_ids = np.asarray(env_ids)
        raw_items = raw_ids.reshape(-1).tolist()
        if raw_ids.dtype.kind == "b" or any(
            isinstance(item, (bool, np.bool_)) for item in raw_items
        ):
            raise ValueError("env_ids must contain integer environment indices")
        if raw_ids.dtype.kind not in "iu":
            try:
                numeric_ids = np.asarray(raw_ids, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("env_ids must contain integer environment indices") from exc
            if raw_ids.dtype.kind not in "f" and any(
                not isinstance(item, (int, float, np.integer, np.floating)) for item in raw_items
            ):
                raise ValueError("env_ids must contain integer environment indices")
            if np.any(~np.isfinite(numeric_ids)) or np.any(numeric_ids != np.floor(numeric_ids)):
                raise ValueError("env_ids must contain integer environment indices")
        ids = raw_ids.astype(np.intp, copy=False).reshape(-1)
        if ids.size == 0:
            return
        if np.any(ids < 0) or np.any(ids >= self._num_envs):
            raise IndexError(
                f"env_ids out of range for {self._num_envs} environments: {ids.tolist()}"
            )
        for group, buffer in self._obs_delay_buffers.items():
            low, high = self._obs_delay_ranges[group]
            lags = sample_wheelbipe_delay_lags(
                (low, high),
                self._num_envs,
                name=f"obs_delay_cfg.{group}",
                inclusive=self._delay_range_semantics == "inclusive",
            )
            # Preserve the current lag for envs that are not being reset.
            selected = buffer.lags.copy()
            selected[ids] = lags[ids]
            buffer.set_time_lag(selected)
            buffer.reset(ids)
            self._physics_delayed_obs[group][ids] = 0.0
        for group, buffer in self._act_delay_buffers.items():
            low, high = self._act_delay_ranges[group]
            lags = sample_wheelbipe_delay_lags(
                (low, high),
                self._num_envs,
                name=f"act_delay_cfg.{group}",
                inclusive=self._delay_range_semantics == "inclusive",
            )
            selected = buffer.lags.copy()
            selected[ids] = lags[ids]
            buffer.set_time_lag(selected)
            buffer.reset(ids)
        self._delayed_native_targets[ids] = 0.0

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": POLICY_OBS_DIM, "critic": PRIVILEGED_OBS_DIM}

    @property
    def timing_contract(self) -> dict[str, Any]:
        """Return a detached timing/delay snapshot for diagnostics/export."""

        return deepcopy(self._timing_contract)

    @property
    def torque_contract(self) -> dict[str, Any]:
        """Return the effective owner/backend actuator-limit snapshot."""

        return deepcopy(self._torque_contract)

    @property
    def domain_randomization_contract(self) -> dict[str, Any]:
        """Return explicit source-to-backend DR conversions and known boundaries."""

        snapshot = deepcopy(self._domain_randomization_contract)
        snapshot["him_curriculum"] = self._source_him_curriculum_contract_snapshot()
        return snapshot

    def _source_him_curriculum_contract_snapshot(self) -> dict[str, Any]:
        cfg = self._cfg.him_curriculum
        runtime = self._source_him_curriculum
        runtime_owner = type(self._cfg).__name__
        if runtime is not None:
            source_owner_scope = "WheelbipeV14FlatHIMEnvCfg"
        elif runtime_owner == "WheelbipeHIMPlayCfg":
            source_owner_scope = "WheelbipeV14FlatHIMEnvCfg_Play.curriculum=None"
        else:
            source_owner_scope = "outside_exact_HIM_training_owner"
        result: dict[str, Any] = {
            "status": (
                "implemented_with_backend_lifecycle_conversion"
                if runtime is not None
                else "disabled"
            ),
            "enabled": bool(cfg.enabled),
            "runtime_owner_config": runtime_owner,
            "source_owner_scope": source_owner_scope,
            "enabled_terms": (
                ["track_height_progression", "base_vertical_assist_force_progression"]
                if runtime is not None
                else []
            ),
            "count_semantics": (
                "after the reward buffer exists, one sample per non-empty vectorized reset "
                "batch; raw window/minimum counts are multiplied by num_steps_per_env"
            ),
            "reset_semantics": (
                "the initial pre-reward reset contributes no sample; after first reward "
                "compute, every non-empty reset batch contributes, including a zero-sum "
                "manual repeat reset"
            ),
            "rng_semantics": (
                "curriculum accounting and stage changes consume no RNG; the inherited "
                "interval random-wrench event retains the owner-seeded NumPy stream"
            ),
            "reward_buffer_initialized": bool(self._source_him_reward_buffer_initialized),
            "track_height_progression": {
                "reward_key": str(cfg.reward_key),
                "num_steps_per_env": int(cfg.num_steps_per_env),
                "window_size": int(cfg.window_size),
                "effective_window_samples": int(cfg.window_size * cfg.num_steps_per_env),
                "default_min_stage_episodes": int(cfg.min_stage_episodes),
                "normalize_by_episode_length": bool(cfg.normalize_by_episode_length),
                "restore_defaults_on_last_stage_threshold": bool(
                    cfg.restore_defaults_after_final_threshold
                ),
                "reward_stage_weights": deepcopy(cfg.reward_stage_weights),
                "thresholds": list(cfg.thresholds),
                "stage_min_episodes": list(cfg.stage_min_episodes),
                "effective_stage_min_compute_calls": [
                    int(value * cfg.num_steps_per_env) for value in cfg.stage_min_episodes
                ],
            },
            "base_vertical_assist_force_progression": {
                "reward_key": str(cfg.reward_key),
                "body_name": str(cfg.assist_body_name),
                "requested_force_frame": "world",
                "force_axis": "z",
                "force_z_stages": list(cfg.assist_force_z_stages),
                "thresholds": list(cfg.thresholds),
                "stage_min_episodes": list(cfg.stage_min_episodes),
                "apply_on_compute": bool(cfg.assist_apply_on_compute),
            },
            "force_interaction": str(cfg.force_interaction),
            "force_composition_status": "source_overwrite_preserved_no_addition",
            "backend_application": (
                "current latched source wrench is replayed through the public upcoming-step "
                "body-force contract on every control step"
            ),
            "conversion_boundary": (
                "source Isaac stores a persistent body-local wrench (the assist converts "
                "world +Z at write time); UniLab's public backend contract accepts an "
                "upcoming-step world-frame wrench, so the latched sampled/world +Z value "
                "is replayed in world frame and body-local orientation drift between "
                "source writes is not claimed equivalent"
            ),
            "assist_latched_env_count": int(np.count_nonzero(self._source_him_assist_latched)),
        }
        if runtime is not None:
            result["runtime"] = runtime.contract_snapshot()
        return result

    def _materialize_source_joint_friction(self, domain_rand: WheelbipeDomainRandConfig) -> None:
        """Sample startup joint friction through the public backend tables.

        Isaac/PhysX exposes static, dynamic and viscous coefficients.  The two
        local backends expose one Coulomb coefficient, so the sampled static
        coefficient is added to ``frictionloss`` while the independently
        sampled/clamped dynamic coefficient remains diagnostic evidence.  A
        backend may explicitly omit viscous conversion, but that decision is
        never silent.
        """

        joint_cfg = domain_rand.joint_friction
        self._source_dof_frictionloss = None
        self._source_dof_damping = None
        self._source_joint_friction_samples: dict[str, dict[str, Any]] = {}
        contract: dict[str, Any] = {
            "requested": bool(joint_cfg.enabled),
            "operation": "add",
            "distribution": "uniform",
            "coulomb_conversion": str(joint_cfg.coulomb_backend_mode),
        }
        self._domain_randomization_contract["joint_friction"] = contract
        if not bool(joint_cfg.enabled):
            contract["status"] = "disabled"
            return

        friction_template = np.asarray(
            self._backend.get_dof_frictionloss(), dtype=np.float64
        ).reshape(-1)
        if not friction_template.size or not np.all(np.isfinite(friction_template)):
            raise ValueError("backend dof-frictionloss table must be finite and non-empty")
        if np.any(friction_template < 0.0):
            raise ValueError("backend dof-frictionloss table must be non-negative")
        frictionloss = np.broadcast_to(
            friction_template, (self._num_envs, friction_template.size)
        ).copy()
        self._source_base_dof_frictionloss = friction_template.copy()

        requested_viscous_mode = str(joint_cfg.viscous_backend_mode).strip().lower()
        damping_template: np.ndarray | None = None
        resolved_viscous_mode = requested_viscous_mode
        if requested_viscous_mode in {"auto", "dof_damping"}:
            try:
                damping_template = np.asarray(
                    self._backend.get_dof_damping(), dtype=np.float64
                ).reshape(-1)
            except NotImplementedError:
                if requested_viscous_mode == "dof_damping":
                    raise NotImplementedError(
                        "domain_rand.joint_friction.viscous_backend_mode='dof_damping' "
                        f"is unsupported by {self._backend.backend_type}"
                    ) from None
                resolved_viscous_mode = "unsupported_omitted"
            else:
                resolved_viscous_mode = "dof_damping"
        if resolved_viscous_mode == "dof_damping":
            assert damping_template is not None
            if damping_template.shape != friction_template.shape:
                raise ValueError(
                    "backend dof damping/frictionloss tables must have the same shape, got "
                    f"{damping_template.shape} and {friction_template.shape}"
                )
            if not np.all(np.isfinite(damping_template)) or np.any(damping_template < 0.0):
                raise ValueError("backend dof-damping table must be finite and non-negative")
            damping = np.broadcast_to(
                damping_template, (self._num_envs, damping_template.size)
            ).copy()
            self._source_base_dof_damping = damping_template.copy()
        elif resolved_viscous_mode == "unsupported_omitted":
            damping = None
            self._source_base_dof_damping = None
            logger.warning(
                "WheelBipe source V14 joint viscous friction is requested but omitted on "
                "%s; Coulomb frictionloss remains active and the omission is recorded in "
                "domain_randomization_contract",
                self._backend.backend_type,
            )
        else:  # dataclass Literal is not a runtime validator after Hydra composition.
            raise ValueError(
                "joint_friction.viscous_backend_mode must be auto, dof_damping, or "
                f"unsupported_omitted, got {joint_cfg.viscous_backend_mode!r}"
            )

        ranges = {
            "front": (
                joint_cfg.front_static_range,
                joint_cfg.front_dynamic_range,
                joint_cfg.front_viscous_range,
            ),
            "rear": (
                joint_cfg.rear_static_range,
                joint_cfg.rear_dynamic_range,
                joint_cfg.rear_viscous_range,
            ),
            "wheel": (
                joint_cfg.wheel_static_range,
                joint_cfg.wheel_static_range,
                joint_cfg.wheel_viscous_range,
            ),
            "inactive": (
                joint_cfg.inactive_static_range,
                joint_cfg.inactive_static_range,
                joint_cfg.inactive_viscous_range,
            ),
            "gimbal": (
                joint_cfg.gimbal_static_range,
                joint_cfg.gimbal_static_range,
                joint_cfg.gimbal_viscous_range,
            ),
        }
        covered_indices: list[int] = []
        group_contracts: dict[str, Any] = {}
        for group, names in _SOURCE_JOINT_FRICTION_NAMES.items():
            if group == "gimbal" and not self._gimbal_enabled:
                group_contracts[group] = {"status": "not_applicable_fixed_gimbal"}
                continue
            indices = np.asarray(self._backend.get_joint_dof_indices(names), dtype=np.intp).reshape(
                -1
            )
            if indices.shape != (len(names),):
                raise ValueError(
                    f"source joint-friction group {group!r} resolved {indices.size} DoFs "
                    f"for {len(names)} joints"
                )
            if np.any(indices < 0) or np.any(indices >= friction_template.size):
                raise ValueError(
                    f"source joint-friction group {group!r} resolved out-of-range DoFs"
                )
            covered_indices.extend(int(index) for index in indices)
            static_range, dynamic_range, viscous_range = ranges[group]
            static_low, static_high = _sample_range(
                static_range, name=f"joint_friction.{group}_static_range"
            )
            dynamic_low, dynamic_high = _sample_range(
                dynamic_range, name=f"joint_friction.{group}_dynamic_range"
            )
            viscous_low, viscous_high = _sample_range(
                viscous_range, name=f"joint_friction.{group}_viscous_range"
            )
            sample_shape = (self._num_envs, indices.size)
            static_sample = np.random.uniform(static_low, static_high, size=sample_shape)
            dynamic_sample = np.minimum(
                np.random.uniform(dynamic_low, dynamic_high, size=sample_shape),
                static_sample,
            )
            viscous_sample = np.random.uniform(viscous_low, viscous_high, size=sample_shape)
            frictionloss[:, indices] += static_sample
            if damping is not None:
                damping[:, indices] += viscous_sample
            self._source_joint_friction_samples[group] = {
                "dof_indices": indices.copy(),
                "static": static_sample.copy(),
                "dynamic": dynamic_sample.copy(),
                "viscous": viscous_sample.copy(),
            }
            group_contracts[group] = {
                "joint_names": list(names),
                "dof_indices": [int(index) for index in indices],
                "static_range": [static_low, static_high],
                "dynamic_range": [dynamic_low, dynamic_high],
                "viscous_range": [viscous_low, viscous_high],
            }
        if len(set(covered_indices)) != len(covered_indices):
            raise ValueError("source joint-friction groups resolved overlapping DoFs")

        self._source_dof_frictionloss = frictionloss
        self._source_dof_damping = damping
        contract.update(
            {
                "status": "implemented_with_explicit_conversion",
                "viscous_requested_mode": requested_viscous_mode,
                "viscous_conversion": resolved_viscous_mode,
                "dynamic_coefficient_status": "sampled_clamped_diagnostic_collapsed",
                "groups": group_contracts,
            }
        )

    def _init_source_semantics_state(self) -> None:
        """Resolve and cache every source-V14 simulator dependency.

        Body/joint/geom names and model tables are intentionally resolved only
        here.  The control, reward, observation and termination hot paths below
        consume cached IDs plus public :class:`SimBackend` state methods.
        """

        n = self._num_envs
        dtype = self._np_dtype
        self._source_zero_torque_steps_remaining = np.zeros((n,), dtype=np.int32)
        self._source_zero_torque_active = np.zeros((n,), dtype=bool)
        self._source_ground_command_steps_remaining = np.zeros((n,), dtype=np.int32)
        self._source_ground_restore_command = np.zeros((n, 3), dtype=dtype)
        self._source_ground_override_command = np.zeros((n, 3), dtype=dtype)
        self._source_air_command_steps_remaining = np.zeros((n,), dtype=np.int32)
        air_command_ranges = np.asarray(
            [
                _sample_range(
                    value,
                    name=f"predefined_air_command_{axis}_range",
                )
                for value, axis in zip(
                    (
                        self._cfg.domain_rand.predefined_air_command_x_range,
                        self._cfg.domain_rand.predefined_air_command_y_range,
                        self._cfg.domain_rand.predefined_air_command_yaw_range,
                    ),
                    ("x", "y", "yaw"),
                    strict=True,
                )
            ],
            dtype=dtype,
        )
        self._source_air_command_low = air_command_ranges[:, 0]
        self._source_air_command_high = air_command_ranges[:, 1]
        self._source_air_height_low, self._source_air_height_high = _sample_range(
            self._cfg.domain_rand.predefined_air_height_range,
            name="predefined_air_height_range",
        )
        self._source_termination_counter = np.zeros((n,), dtype=np.int32)
        self._source_value_debug_step = 0
        # Isaac DirectRLEnv evaluates done/reward/reset before it builds the
        # next observation.  Keep a committed previous-observation cache plus
        # a pending current frame so an explicitly enabled debug profile sees
        # the same one-control-step ordering.  Exact V14 owners leave the gate
        # disabled via ``debug_value_diagnosis=false``.
        self._source_obs_safety_nonfinite = np.zeros((n,), dtype=bool)
        self._source_obs_safety_max_abs = np.zeros((n,), dtype=self._np_dtype)
        self._source_obs_safety_pending_nonfinite = np.zeros((n,), dtype=bool)
        self._source_obs_safety_pending_max_abs = np.zeros((n,), dtype=self._np_dtype)
        self._source_control_randomization_initialized = np.zeros((n,), dtype=bool)
        self._source_control_randomization_age = np.zeros((n,), dtype=np.int32)
        self._source_body_force = np.zeros((n, 3), dtype=dtype)
        self._source_body_torque = np.zeros((n, 3), dtype=dtype)
        self._source_force_next_step = np.full((n,), np.iinfo(np.int64).max, dtype=np.int64)
        self._source_push_next_step = np.full((n,), np.iinfo(np.int64).max, dtype=np.int64)
        # Pinned ``_episode_sums`` is ``torch.float`` even though manager
        # statistics become Python floats after ``tolist()``.
        self._source_him_track_height_episode_sum = np.zeros((n,), dtype=np.float32)
        # Source ``_episode_sums`` is absent before the first reward compute,
        # then persists for the rest of the environment lifetime.  Preserve
        # that global existence bit: after creation, every non-empty reset
        # batch contributes one manager sample, even if a caller resets the
        # same environment twice without an intervening step (the second
        # sample is zero in the source).
        self._source_him_reward_buffer_initialized = False
        # The assist term writes the same persistent articulation wrench
        # buffer as the inherited interval-force event.  Constructor/reset
        # writes latch assist; a due interval event clears the selected latch.
        self._source_him_assist_latched = np.full(
            (n,), self._source_him_curriculum is not None, dtype=bool
        )
        self._source_contact_history = np.zeros((3, n, 0), dtype=dtype)
        self._source_contact_history_cursor = 0
        self._source_base_mass_scale = np.ones((n,), dtype=dtype)
        self._source_wheel_material = np.zeros((n, 2, 3), dtype=dtype)
        self._source_reset_pos_indices = np.zeros((0,), dtype=np.intp)
        self._gimbal_yaw_body_ids = np.zeros((0,), dtype=np.int32)
        self._base_body_ids = np.zeros((0,), dtype=np.int32)
        self._wheel_body_ids = np.zeros((0,), dtype=np.int32)
        self._source_contact_body_ids = np.zeros((0,), dtype=np.int32)
        self._source_wheel_contact_columns = np.zeros((0,), dtype=np.intp)
        self._source_undesired_contact_columns = np.zeros((0,), dtype=np.intp)
        self._source_reset_contact_columns = np.zeros((0,), dtype=np.intp)
        self._source_base_contact_column = -1
        self._source_reset_contact = np.zeros((n,), dtype=bool)
        self._source_body_mass = None
        self._source_base_com_offset = None
        self._source_base_com_b = np.zeros((n, 3), dtype=dtype)
        self._source_wheel_com_b = np.zeros((0, 3), dtype=dtype)
        self._source_geom_friction = None
        self._source_guide_material = np.zeros((n, 0, 3), dtype=dtype)
        self._source_guide_body_ids = np.zeros((0,), dtype=np.int32)
        self._source_dof_frictionloss = None
        self._source_dof_damping = None
        self._source_base_dof_frictionloss = None
        self._source_base_dof_damping = None
        self._source_joint_friction_samples = {}
        self._domain_randomization_contract: dict[str, Any] = {
            "profile": "source_v14" if self._source_semantics else "legacy",
            "backend": str(self._backend.backend_type),
        }
        # Preserve the backend's physical range separately from the static
        # owner intersection.  Source effort-limit DR changes that intersection
        # per environment on every reset.
        physical_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        self._backend_ctrl_lower = physical_range[:, 0].copy()
        self._backend_ctrl_upper = physical_range[:, 1].copy()
        self._source_ctrl_lower = np.full((self._num_native_actuators,), -np.inf, dtype=np.float64)
        self._source_ctrl_upper = np.full((self._num_native_actuators,), np.inf, dtype=np.float64)

        if self._gimbal_enabled:
            self._gimbal_yaw_body_ids = np.asarray(
                self._backend.get_body_ids(("gimbal_yaw_link",)), dtype=np.int32
            ).reshape(-1)
            if self._gimbal_yaw_body_ids.shape != (1,):
                raise ValueError(
                    "WheelBipe gimbal owner requires exactly one gimbal_yaw_link body; "
                    f"resolved {self._gimbal_yaw_body_ids.tolist()}"
                )

        if not self._source_semantics:
            return

        base_id = int(self._backend.get_body_id(self._cfg.asset.base_name))
        self._base_body_ids = np.asarray([base_id], dtype=np.int32)
        self._wheel_body_ids = np.asarray(
            self._backend.get_body_ids(SOURCE_V14_WHEEL_BODY_NAMES), dtype=np.int32
        )
        # Isaac root/body linear velocities are measured at each body's COM.
        # Backend link-frame velocities need these cold-path inertial offsets.
        body_ipos = np.asarray(self._backend.get_body_ipos(), dtype=dtype)
        self._source_base_com_b[:] = body_ipos[base_id]
        self._source_wheel_com_b = body_ipos[self._wheel_body_ids].copy()
        reset_contact_names = tuple(self._source_reset_contact_body_names())
        contact_names = tuple(
            dict.fromkeys(
                (
                    *SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES,
                    *SOURCE_V14_WHEEL_BODY_NAMES,
                    *reset_contact_names,
                )
            )
        )
        self._source_contact_body_ids = np.asarray(
            self._backend.get_body_ids(contact_names), dtype=np.int32
        )
        contact_columns = {name: index for index, name in enumerate(contact_names)}
        self._source_wheel_contact_columns = np.asarray(
            [contact_columns[name] for name in SOURCE_V14_WHEEL_BODY_NAMES], dtype=np.intp
        )
        self._source_undesired_contact_columns = np.asarray(
            [contact_columns[name] for name in SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES],
            dtype=np.intp,
        )
        self._source_reset_contact_columns = np.asarray(
            [contact_columns[name] for name in reset_contact_names], dtype=np.intp
        )
        self._source_base_contact_column = int(contact_columns[self._cfg.asset.base_name])
        self._source_contact_history = np.zeros(
            (3, n, self._source_contact_body_ids.size), dtype=dtype
        )
        self._source_reset_pos_indices = np.asarray(
            self._backend.get_joint_dof_pos_indices(SOURCE_V14_RESET_JOINT_NAMES),
            dtype=np.intp,
        )

        domain_rand = self._cfg.domain_rand
        guide_names = tuple(
            name for name in self._backend.get_body_names() if name.endswith("_guide_link")
        )
        guide_requested = bool(domain_rand.source_guide_material_requested)
        if guide_names:
            self._source_guide_body_ids = np.asarray(
                self._backend.get_body_ids(guide_names), dtype=np.int32
            )
            guide_status = "implemented_dynamic_to_sliding"
        elif guide_requested:
            guide_status = str(domain_rand.source_guide_material_missing_target_status)
            if guide_status != "not_applicable_missing_target":
                raise ValueError(
                    "source guide-material target is absent, so "
                    "source_guide_material_missing_target_status must be "
                    "'not_applicable_missing_target'"
                )
            logger.warning(
                "WheelBipe source V14 guide material is requested, but %s exposes no "
                "'*_guide_link' body; status=not_applicable_missing_target is recorded "
                "in domain_randomization_contract",
                self._backend.backend_type,
            )
        else:
            guide_status = "disabled"
        self._domain_randomization_contract.update(
            {
                "interval_root_velocity_push": {
                    "status": (
                        "implemented" if domain_rand.source_push_velocity_enabled else "disabled"
                    ),
                    "interval_range_s": list(domain_rand.source_push_interval_range),
                    "velocity_range_xyz": deepcopy(domain_rand.source_push_velocity_range),
                },
                "interval_external_force_torque": {
                    "status": (
                        "implemented" if domain_rand.source_external_force_enabled else "disabled"
                    ),
                    "target_body": self._cfg.asset.base_name,
                    "interval_range_s": list(domain_rand.source_external_force_interval_range),
                    "force_range_xyz": [list(domain_rand.source_external_force_range)] * 3,
                    "torque_range_xyz": [list(domain_rand.source_external_torque_range)] * 3,
                },
                "guide_material": {
                    "requested": guide_requested,
                    "status": guide_status,
                    "source_body_pattern": ".*_guide_link",
                    "resolved_body_names": list(guide_names),
                    "static_range": list(domain_rand.guide_static_friction_range),
                    "dynamic_range": list(domain_rand.guide_dynamic_friction_range),
                    "restitution_range": list(domain_rand.guide_restitution_range),
                    "num_buckets": int(domain_rand.guide_material_num_buckets),
                },
                "body_material": {
                    "requested": bool(domain_rand.randomize_body_material),
                    "status": (
                        "implemented_with_explicit_conversion"
                        if bool(domain_rand.randomize_body_material)
                        else "disabled"
                    ),
                    "distribution": "uniform_bucketed",
                    "make_consistent": bool(domain_rand.material_make_consistent),
                    "num_buckets": int(domain_rand.material_num_buckets),
                    "groups": {
                        "base": {
                            "source_body_pattern": self._cfg.asset.base_name,
                            "static_range": list(domain_rand.base_static_friction_range),
                            "dynamic_range": list(domain_rand.base_dynamic_friction_range),
                            "restitution_range": list(domain_rand.base_restitution_range),
                            "num_buckets": int(domain_rand.material_num_buckets),
                        },
                        "wheel": {
                            "source_body_pattern": ".*_wheel_link",
                            "static_range": list(domain_rand.wheel_static_friction_range),
                            "dynamic_range": list(domain_rand.wheel_dynamic_friction_range),
                            "restitution_range": list(domain_rand.wheel_restitution_range),
                            "num_buckets": int(domain_rand.material_num_buckets),
                        },
                    },
                    # PhysX owns distinct static/dynamic coefficients plus
                    # restitution.  MuJoCo and Motrix expose one sliding
                    # coefficient through the shared reset payload, so only
                    # the sampled dynamic value drives their contact law.  The
                    # full source triplet is retained for the 78D critic and
                    # diagnostics and this conversion must never be implicit.
                    "backend_conversion": {
                        "sliding_friction": "dynamic_to_single_sliding",
                        "static_friction": "diagnostic_not_contact_law",
                        "restitution": "diagnostic_not_contact_law",
                    },
                },
                "actuator_gain_reset": {
                    "status": "implemented_active_groups_with_boundary",
                    "interval_reset_steps": int(domain_rand.control_randomization_min_steps),
                    "active_groups": ["legs", "wheels", "spring2", "gimbal"],
                    "passive_ideal_pd_damping": {
                        "status": str(domain_rand.source_passive_gain_conversion_status),
                        "joint_names": list(_SOURCE_JOINT_FRICTION_NAMES["inactive"]),
                        "source_baseline": 0.01,
                    },
                },
            }
        )
        if (
            str(domain_rand.source_passive_gain_conversion_status)
            != "unsupported_unmaterialized_actuator"
        ):
            raise ValueError(
                "source_passive_gain_conversion_status must be "
                "'unsupported_unmaterialized_actuator' for the local asset"
            )
        logger.warning(
            "WheelBipe source V14 passive IdealPD damping gain reset is not materialized on "
            "%s; startup viscous joint friction remains independently converted and the "
            "boundary is recorded in domain_randomization_contract",
            self._backend.backend_type,
        )
        self._materialize_source_joint_friction(domain_rand)

        mass_template = np.asarray(self._backend.get_body_mass(), dtype=np.float64).reshape(-1)
        body_mass = np.broadcast_to(mass_template, (n, mass_template.size)).copy()
        positive_mass = mass_template > 0.0
        robot_body_ids = np.asarray(
            self._backend.get_body_subtree_ids(base_id), dtype=np.intp
        ).reshape(-1)
        robot_body_ids = robot_body_ids[
            (robot_body_ids >= 0) & (robot_body_ids < mass_template.size)
        ]
        # Match ``EventCfgV14.add_leg_mass`` exactly.  A subtree-wide
        # complement would also include the 16 passive guide links, although
        # the source event only names front/rear/spring links plus gimbal
        # links.  Resolve the explicit list once here; reset payloads then
        # reuse the cached table without any body-name or XML access.
        source_leg_mass_names = tuple(
            name
            for name in SOURCE_V14_LEG_MASS_BODY_NAMES
            if name in set(self._backend.get_body_names())
        )
        source_leg_mass_ids = (
            np.asarray(self._backend.get_body_ids(source_leg_mass_names), dtype=np.intp).reshape(-1)
            if source_leg_mass_names
            else np.zeros((0,), dtype=np.intp)
        )
        source_leg_mass_ids = source_leg_mass_ids[
            (source_leg_mass_ids >= 0) & (source_leg_mass_ids < mass_template.size)
        ]
        if bool(domain_rand.randomize_body_mass):
            low, high = _sample_range(
                domain_rand.base_mass_multiplier_range,
                name="base_mass_multiplier_range",
            )
            self._source_base_mass_scale = np.random.uniform(low, high, size=n).astype(dtype)
            body_mass[:, base_id] *= self._source_base_mass_scale
            low, high = _sample_range(
                domain_rand.leg_mass_multiplier_range,
                name="leg_mass_multiplier_range",
            )
            randomized_other = source_leg_mass_ids[positive_mass[source_leg_mass_ids]]
            if randomized_other.size:
                body_mass[:, randomized_other] *= np.random.uniform(
                    low, high, size=(n, randomized_other.size)
                )
            low, high = _sample_range(
                domain_rand.wheel_mass_multiplier_range,
                name="wheel_mass_multiplier_range",
            )
            randomized_wheels = self._wheel_body_ids[positive_mass[self._wheel_body_ids]]
            if randomized_wheels.size:
                body_mass[:, randomized_wheels] *= np.random.uniform(
                    low, high, size=(n, randomized_wheels.size)
                )
            self._source_body_mass = body_mass

        if bool(domain_rand.random_com):
            offset = np.zeros((n, 3), dtype=np.float64)
            for column, (field_name, range_name) in enumerate(
                (
                    (domain_rand.com_offset_x, "com_offset_x"),
                    (domain_rand.com_offset_y, "com_offset_y"),
                    (domain_rand.com_offset_z, "com_offset_z"),
                )
            ):
                low, high = _sample_range(field_name, name=range_name)
                offset[:, column] = np.random.uniform(low, high, size=n)
            self._source_base_com_offset = offset
            self._source_base_com_b += offset

        friction_template = np.asarray(self._base_geom_friction, dtype=np.float64)
        geom_friction = np.broadcast_to(friction_template, (n, *friction_template.shape)).copy()
        if bool(domain_rand.randomize_body_material) or self._source_guide_body_ids.size:
            geom_body_ids = np.asarray(self._backend.get_geom_body_ids(), dtype=np.intp)
            geom_contype, geom_conaffinity = self._backend.get_geom_contact_masks()
            collision_geom = (np.asarray(geom_contype) != 0) | (np.asarray(geom_conaffinity) != 0)
            if collision_geom.shape != geom_body_ids.shape:
                raise ValueError("backend geom contact masks must match the geom/body-id table")
            base_geom_ids = np.flatnonzero((geom_body_ids == base_id) & collision_geom)
            wheel_geom_ids = [
                np.flatnonzero((geom_body_ids == int(wheel_id)) & collision_geom)
                for wheel_id in self._wheel_body_ids
            ]
            if bool(domain_rand.randomize_body_material):
                base_material = sample_source_v14_material_buckets(
                    num_samples=n,
                    num_slots=base_geom_ids.size,
                    static_friction_range=domain_rand.base_static_friction_range,
                    dynamic_friction_range=domain_rand.base_dynamic_friction_range,
                    restitution_range=domain_rand.base_restitution_range,
                    num_buckets=int(domain_rand.material_num_buckets),
                    make_consistent=bool(domain_rand.material_make_consistent),
                )
                # MuJoCo/Motrix expose one sliding coefficient rather than the
                # source PhysX static/dynamic pair.  Drive simulation with the
                # sampled dynamic value; retain all three source material fields
                # below for the exact critic contract and report this backend gap.
                if base_geom_ids.size:
                    geom_friction[:, base_geom_ids, 0] = base_material[..., 1]

                self._source_wheel_material[:] = sample_source_v14_material_buckets(
                    num_samples=n,
                    num_slots=2,
                    static_friction_range=domain_rand.wheel_static_friction_range,
                    dynamic_friction_range=domain_rand.wheel_dynamic_friction_range,
                    restitution_range=domain_rand.wheel_restitution_range,
                    num_buckets=int(domain_rand.material_num_buckets),
                    make_consistent=bool(domain_rand.material_make_consistent),
                )
                for wheel_index, geom_ids in enumerate(wheel_geom_ids):
                    if geom_ids.size:
                        geom_friction[:, geom_ids, 0] = self._source_wheel_material[
                            :, wheel_index, 1
                        ][:, None]
            if self._source_guide_body_ids.size:
                guide_geom_ids = np.flatnonzero(
                    np.isin(geom_body_ids, self._source_guide_body_ids) & collision_geom
                )
                self._source_guide_material = sample_source_v14_material_buckets(
                    num_samples=n,
                    num_slots=guide_geom_ids.size,
                    static_friction_range=domain_rand.guide_static_friction_range,
                    dynamic_friction_range=domain_rand.guide_dynamic_friction_range,
                    restitution_range=domain_rand.guide_restitution_range,
                    num_buckets=int(domain_rand.guide_material_num_buckets),
                    make_consistent=True,
                )
                if guide_geom_ids.size:
                    geom_friction[:, guide_geom_ids, 0] = self._source_guide_material[..., 1]
            self._source_geom_friction = geom_friction

        if bool(domain_rand.source_external_force_enabled):
            self._reset_source_force_timers(np.arange(n, dtype=np.intp))
        if bool(domain_rand.source_push_velocity_enabled):
            self._reset_source_push_timers(np.arange(n, dtype=np.intp))

    def _reset_source_force_timers(self, env_ids: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        if not ids.size:
            return
        low, high = _sample_range(
            self._cfg.domain_rand.source_external_force_interval_range,
            name="source_external_force_interval_range",
        )
        duration = np.random.uniform(low, high, size=ids.size)
        intervals = np.maximum(np.ceil(duration / float(self._cfg.ctrl_dt)).astype(np.int64), 1)
        self._source_force_next_step[ids] = int(self.step_counter) + intervals

    def _reset_source_push_timers(self, env_ids: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        if not ids.size:
            return
        low, high = _sample_range(
            self._cfg.domain_rand.source_push_interval_range,
            name="source_push_interval_range",
        )
        duration = np.random.uniform(low, high, size=ids.size)
        intervals = np.maximum(np.ceil(duration / float(self._cfg.ctrl_dt)).astype(np.int64), 1)
        self._source_push_next_step[ids] = int(self.step_counter) + intervals

    def _capture_source_contact_force(self, backend: Any) -> None:
        if not self._source_semantics or self._source_contact_body_ids.size == 0:
            return
        force = np.asarray(
            backend.get_body_contact_force_norm(self._source_contact_body_ids),
            dtype=self._np_dtype,
        )
        expected = (self._num_envs, self._source_contact_body_ids.size)
        if force.shape != expected:
            raise ValueError(
                f"body contact-force contract returned {force.shape}; expected {expected}"
            )
        self._source_contact_history[self._source_contact_history_cursor] = force
        self._source_contact_history_cursor = (
            self._source_contact_history_cursor + 1
        ) % self._source_contact_history.shape[0]

    def _source_contact_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        history_peak = np.max(self._source_contact_history, axis=0)
        latest_index = (
            self._source_contact_history_cursor - 1
        ) % self._source_contact_history.shape[0]
        latest = self._source_contact_history[latest_index]
        wheel_contact = history_peak[:, self._source_wheel_contact_columns] > float(
            self._cfg.desired_contact_force_threshold
        )
        undesired = np.any(
            history_peak[:, self._source_undesired_contact_columns]
            > float(self._cfg.undesired_contact_force_threshold),
            axis=1,
        )
        base_contact = latest[:, self._source_base_contact_column] > 1.0
        # Source V14 termination uses ``_reset_contact_link_idx`` rather than
        # the base-only diagnostic feature.  Keep both signals available: the
        # state machines consume ``base_contact`` while the done owner below
        # consumes this cached latest-frame reset mask.
        if self._source_reset_contact_columns.size:
            self._source_reset_contact = np.any(
                latest[:, self._source_reset_contact_columns] > 1.0, axis=1
            )
        else:
            self._source_reset_contact.fill(False)
        return wheel_contact, undesired, base_contact

    def _source_height_signals(
        self, base_pos: np.ndarray, base_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build distinct privileged and reward height signals from one pose read."""

        positions = np.asarray(base_pos, dtype=self._np_dtype)
        if positions.ndim != 2 or positions.shape[1] < 3:
            raise ValueError(f"base_pos must have shape (N, >=3), got {positions.shape}")
        terrain_height: np.ndarray | None = None
        if (
            not bool(self._cfg.use_absolute_height)
            and self._terrain_surface_sample_height is not None
        ):
            yaw = np_yaw_from_quat(np.asarray(base_quat, dtype=self._np_dtype))
            c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
            dx, dy = self._source_height_scan_xy.T
            offsets = np.stack((c * dx - s * dy, s * dx + c * dy), axis=-1)
            scan_xy = positions[:, None, :2] + offsets
            heights = np.asarray(self._terrain_surface_sample_height(scan_xy), dtype=self._np_dtype)
            valid = np.isfinite(heights)
            terrain_height = np.sum(np.where(valid, heights, 0.0), axis=1) / np.maximum(
                np.sum(valid, axis=1), 1
            )
            # Generated terrain samples are finite. Preserve the source
            # origin fallback if a backend reports no valid surface hits.
            if not np.all(np.any(valid, axis=1)):
                fallback = self._spawn.origins_for(np.arange(positions.shape[0]))[:, 2]
                terrain_height = np.where(
                    np.any(valid, axis=1),
                    terrain_height,
                    fallback - self._source_height_spawn_margin,
                )
        return build_source_v14_height_signals(
            positions[:, 2],
            terrain_height,
            use_absolute_height=bool(self._cfg.use_absolute_height),
            clip_enabled=bool(self._cfg.height_obs_clip_enabled),
            clip_range=self._cfg.height_obs_clip_range,
        )

    def _reset_source_episode_buffers(
        self,
        env_ids: np.ndarray,
        *,
        ground_reset: np.ndarray,
        airborne_reset: np.ndarray,
    ) -> None:
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        self._source_termination_counter[ids] = 0
        self._source_contact_history[:, ids, :] = 0.0
        self._source_reset_contact[ids] = False
        self._source_body_force[ids] = 0.0
        self._source_body_torque[ids] = 0.0
        zero_steps = max(
            int(
                round(
                    float(self._cfg.domain_rand.predefined_ground_zero_torque_seconds)
                    / float(self._cfg.ctrl_dt)
                )
            ),
            0,
        )
        self._source_zero_torque_steps_remaining[ids] = np.where(
            np.asarray(ground_reset, dtype=bool), zero_steps, 0
        )
        ground_command_steps = max(
            int(
                np.ceil(
                    float(self._cfg.domain_rand.predefined_ground_command_seconds)
                    / float(self._cfg.ctrl_dt)
                )
            ),
            0,
        )
        air_command_steps = max(
            int(
                np.ceil(
                    float(self._cfg.domain_rand.predefined_air_command_seconds)
                    / float(self._cfg.ctrl_dt)
                )
            ),
            0,
        )
        self._source_ground_command_steps_remaining[ids] = np.where(
            np.asarray(ground_reset, dtype=bool), ground_command_steps, 0
        )
        self._source_air_command_steps_remaining[ids] = np.where(
            np.asarray(airborne_reset, dtype=bool), air_command_steps, 0
        )
        self._source_ground_restore_command[ids] = 0.0
        self._source_ground_override_command[ids] = 0.0
        if bool(self._cfg.domain_rand.source_external_force_enabled):
            self._reset_source_force_timers(ids)
        if bool(self._cfg.domain_rand.source_push_velocity_enabled):
            self._reset_source_push_timers(ids)

    def sample_reset_source_control_randomization(
        self, env_ids: np.ndarray
    ) -> dict[str, np.ndarray]:
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        cfg = self._cfg.control_config
        domain_rand = self._cfg.domain_rand
        min_steps = max(int(domain_rand.control_randomization_min_steps), 0)
        randomized_mask = (~self._source_control_randomization_initialized[ids]) | (
            self._source_control_randomization_age[ids] >= min_steps
        )
        randomized_count = int(np.count_nonzero(randomized_mask))
        kp = self._motor_kp[ids].copy()
        kd = self._motor_kd[ids].copy()
        wheel_kd = self._wheel_kd[ids].copy()
        spring_damping = self._spring_damping[ids].copy()
        leg_effort_scale = self._leg_effort_scale[ids].copy()
        wheel_effort_scale = self._wheel_effort_scale[ids].copy()
        gimbal_yaw_kd = self._gimbal_yaw_kd[ids].copy()
        gimbal_pitch_kp = self._gimbal_pitch_kp[ids].copy()
        gimbal_pitch_kd = self._gimbal_pitch_kd[ids].copy()
        if randomized_count == 0:
            return {
                "motor_kp": kp,
                "motor_kd": kd,
                "wheel_kd": wheel_kd,
                "spring_damping": spring_damping,
                "leg_effort_scale": leg_effort_scale,
                "wheel_effort_scale": wheel_effort_scale,
                "gimbal_yaw_kd": gimbal_yaw_kd,
                "gimbal_pitch_kp": gimbal_pitch_kp,
                "gimbal_pitch_kd": gimbal_pitch_kd,
                "randomized_mask": randomized_mask,
            }

        sampled_kp = np.full((randomized_count, NUM_LEG_ACTIONS), float(cfg.Kp), dtype=np.float64)
        sampled_kd = np.full((randomized_count, NUM_LEG_ACTIONS), float(cfg.Kd), dtype=np.float64)
        sampled_wheel_kd = np.full(
            (randomized_count, NUM_WHEEL_ACTIONS), float(cfg.wheel_Kd), dtype=np.float64
        )
        if bool(domain_rand.randomize_kp):
            low, high = _sample_range(domain_rand.kp_multiplier_range, name="kp_multiplier_range")
            sampled_kp *= np.random.uniform(low, high, size=sampled_kp.shape)
        if bool(domain_rand.randomize_kd):
            low, high = _sample_range(domain_rand.kd_multiplier_range, name="kd_multiplier_range")
            sampled_kd *= np.random.uniform(low, high, size=sampled_kd.shape)
            sampled_wheel_kd *= np.random.uniform(low, high, size=sampled_wheel_kd.shape)
        # The pinned reset stack first scales every actuator damping by the
        # global 0.75--1.25 gain event, then applies the spring-only 0.5--1.5
        # multiplier.  The fixed IdealPD spring damping remains 50 N s/m.
        gain_low, gain_high = _sample_range(
            domain_rand.kd_multiplier_range, name="kd_multiplier_range"
        )
        low, high = _sample_range(
            domain_rand.spring_damping_multiplier_range,
            name="spring_damping_multiplier_range",
        )
        sampled_spring_damping = (
            float(cfg.spring_damping)
            * np.random.uniform(gain_low, gain_high, size=(randomized_count, 2))
            * np.random.uniform(low, high, size=(randomized_count, 2))
        )
        low, high = _sample_range(
            domain_rand.leg_effort_multiplier_range,
            name="leg_effort_multiplier_range",
        )
        sampled_leg_effort_scale = np.random.uniform(
            low, high, size=(randomized_count, NUM_LEG_ACTIONS)
        )
        low, high = _sample_range(
            domain_rand.wheel_effort_multiplier_range,
            name="wheel_effort_multiplier_range",
        )
        sampled_wheel_effort_scale = np.random.uniform(
            low, high, size=(randomized_count, NUM_WHEEL_ACTIONS)
        )
        gains_low, gains_high = _sample_range(
            domain_rand.kd_multiplier_range, name="kd_multiplier_range"
        )
        kp[randomized_mask] = sampled_kp
        kd[randomized_mask] = sampled_kd
        wheel_kd[randomized_mask] = sampled_wheel_kd
        spring_damping[randomized_mask] = sampled_spring_damping
        leg_effort_scale[randomized_mask] = sampled_leg_effort_scale
        wheel_effort_scale[randomized_mask] = sampled_wheel_effort_scale
        gimbal_yaw_kd[randomized_mask] = float(self._cfg.gimbal.velocity_kd) * np.random.uniform(
            gains_low, gains_high, size=randomized_count
        )
        gimbal_pitch_kp[randomized_mask] = float(self._cfg.gimbal.pitch_kp) * np.random.uniform(
            gains_low, gains_high, size=randomized_count
        )
        gimbal_pitch_kd[randomized_mask] = float(self._cfg.gimbal.pitch_kd) * np.random.uniform(
            gains_low, gains_high, size=randomized_count
        )
        return {
            "motor_kp": kp,
            "motor_kd": kd,
            "wheel_kd": wheel_kd,
            "spring_damping": spring_damping,
            "leg_effort_scale": leg_effort_scale,
            "wheel_effort_scale": wheel_effort_scale,
            "gimbal_yaw_kd": gimbal_yaw_kd,
            "gimbal_pitch_kp": gimbal_pitch_kp,
            "gimbal_pitch_kd": gimbal_pitch_kd,
            "randomized_mask": randomized_mask,
        }

    def set_source_control_randomization(
        self,
        env_ids: np.ndarray,
        *,
        motor_kp: np.ndarray,
        motor_kd: np.ndarray,
        wheel_kd: np.ndarray,
        spring_damping: np.ndarray,
        leg_effort_scale: np.ndarray,
        wheel_effort_scale: np.ndarray,
        gimbal_yaw_kd: np.ndarray,
        gimbal_pitch_kp: np.ndarray,
        gimbal_pitch_kd: np.ndarray,
        randomized_mask: np.ndarray,
    ) -> None:
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        self._motor_kp[ids] = np.asarray(motor_kp, dtype=np.float64)
        self._motor_kd[ids] = np.asarray(motor_kd, dtype=np.float64)
        self._wheel_kd[ids] = np.asarray(wheel_kd, dtype=np.float64)
        self._spring_damping[ids] = np.asarray(spring_damping, dtype=np.float64)
        self._leg_effort_scale[ids] = np.asarray(leg_effort_scale, dtype=np.float64)
        self._wheel_effort_scale[ids] = np.asarray(wheel_effort_scale, dtype=np.float64)
        self._gimbal_yaw_kd[ids] = np.asarray(gimbal_yaw_kd, dtype=np.float64)
        self._gimbal_pitch_kp[ids] = np.asarray(gimbal_pitch_kp, dtype=np.float64)
        self._gimbal_pitch_kd[ids] = np.asarray(gimbal_pitch_kd, dtype=np.float64)
        randomized_ids = ids[np.asarray(randomized_mask, dtype=bool)]
        self._source_control_randomization_initialized[randomized_ids] = True
        self._source_control_randomization_age[randomized_ids] = 0

    def _sample_source_commands(
        self, *, current_yaw: np.ndarray, episode_steps: np.ndarray
    ) -> dict[str, np.ndarray]:
        commands_cfg = self._cfg.commands
        special_starts = tuple(int(value) for value in commands_cfg.special_mode_start_iterations)
        special_probabilities = tuple(
            float(value) for value in commands_cfg.special_mode_probabilities
        )
        gimbal_probability = float(commands_cfg.gimbal_mode_probability)
        zero_command_probability = float(commands_cfg.zero_command_probability)
        if not bool(commands_cfg.source_curriculum_enabled):
            special_starts = (np.iinfo(np.int32).max,) * 3
        return sample_source_v14_commands(
            num_samples=int(np.asarray(current_yaw).size),
            current_yaw=current_yaw,
            episode_steps=episode_steps,
            ctrl_dt=float(self._cfg.ctrl_dt),
            training_iteration=(
                int(self.step_counter)
                // max(int(commands_cfg.training_progress_steps_per_iteration), 1)
                + int(commands_cfg.training_iteration_offset)
            ),
            normal_low=commands_cfg.vel_limit[0],
            normal_high=commands_cfg.vel_limit[1],
            standing_probability=float(commands_cfg.rel_standing_envs),
            heading_probability=float(commands_cfg.rel_heading_envs),
            heading_range=commands_cfg.heading_range,
            heading_stiffness=float(commands_cfg.heading_control_stiffness),
            special_mode_min_episode_time=float(commands_cfg.special_mode_min_episode_time),
            special_mode_start_iterations=special_starts,
            special_mode_probabilities=special_probabilities,
            gimbal_mode_probability=gimbal_probability,
            gimbal_mode_start_iteration=int(commands_cfg.gimbal_mode_start_iteration),
            zero_command_probability=zero_command_probability,
            zero_command_start_iteration=int(commands_cfg.zero_command_start_iteration),
        )

    def sync_training_iteration(self, iteration: int) -> None:
        """Anchor source command curricula when an RSL-RL run is resumed.

        RSL-RL stores the completed learning iteration in its checkpoint, but
        a newly materialized NumPy environment starts ``step_counter`` at
        zero.  Source V14 special-command schedules are expressed in training
        iterations, so leaving the owner offset at zero silently replays the
        early curriculum after every resume.  The generic RSL-RL wrapper calls
        this owner hook after loading a checkpoint; a non-zero explicit config
        offset remains authoritative for callers that intentionally anchor a
        run at another curriculum stage.
        """

        if not self._source_semantics:
            return
        commands_cfg = self._cfg.commands
        configured = int(getattr(commands_cfg, "training_iteration_offset", 0))
        if configured == 0:
            commands_cfg.training_iteration_offset = max(int(iteration), 0)

    def sample_reset_motor_gains(self, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        kp = np.full((num_reset, NUM_LEG_ACTIONS), float(self._cfg.control_config.Kp))
        kd = np.full((num_reset, NUM_LEG_ACTIONS), float(self._cfg.control_config.Kd))
        if self._cfg.domain_rand.randomize_kp:
            low, high = _sample_range(
                self._cfg.domain_rand.kp_multiplier_range, name="kp_multiplier_range"
            )
            kp *= np.random.uniform(low, high, size=(num_reset, 1))
        if self._cfg.domain_rand.randomize_kd:
            low, high = _sample_range(
                self._cfg.domain_rand.kd_multiplier_range, name="kd_multiplier_range"
            )
            kd *= np.random.uniform(low, high, size=(num_reset, 1))
        return kp, kd

    def set_motor_gains(self, env_ids: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        self._motor_kp[ids] = np.asarray(kp, dtype=np.float64)
        self._motor_kd[ids] = np.asarray(kd, dtype=np.float64)

    def sample_reset_spring_force(self, env_ids: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        bounds = _sample_range(
            self._cfg.control_config.spring_random_force,
            name="spring_random_force",
        )
        self._spring_force_random[ids] = np.random.uniform(
            bounds[0], bounds[1], size=(ids.size, 2)
        ).astype(self._np_dtype)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        actions_arr = np.asarray(actions, dtype=self._np_dtype)
        if actions_arr.shape != (self._num_envs, NUM_POLICY_ACTIONS):
            raise ValueError(
                f"Wheelbipe actions must have shape ({self._num_envs}, {NUM_POLICY_ACTIONS}), "
                f"got {actions_arr.shape}"
            )
        clip = float(self._cfg.control_config.clip_actions)
        clipped = np.clip(actions_arr, -clip, clip).astype(self._np_dtype, copy=False)
        previous = np.asarray(
            state.info.get("current_actions", np.zeros_like(clipped)), dtype=self._np_dtype
        )
        state.info["previous_actions"] = np.asarray(state.info.get("last_actions", previous)).copy()
        state.info["last_actions"] = previous.copy()
        state.info["current_actions"] = clipped.copy()
        # ``use_act_delay`` is consumed per physics substep in the callback;
        # the legacy boolean remains a separate one-control-step contract.
        executed = (
            previous
            if self._cfg.control_config.simulate_action_latency and not self._use_act_delay
            else clipped
        )
        targets = map_policy_action_to_native_targets(
            executed,
            action_scale=float(self._cfg.control_config.action_scale),
            wheel_action_scale=float(self._cfg.control_config.wheel_action_scale),
            default_leg_position=self.default_angles[:NUM_LEG_ACTIONS],
            native_leg_indices=self._native_leg_indices,
            native_wheel_indices=self._native_wheel_indices,
            native_spring_indices=self._native_spring_indices,
            num_native_actuators=self._num_native_actuators,
            leg_position_limit=(
                self._cfg.control_config.leg_position_target_limit
                if self._source_semantics
                else None
            ),
            wheel_velocity_limit=(
                self._cfg.control_config.wheel_velocity_target_limit
                if self._source_semantics
                else None
            ),
        )
        state.info["native_targets"] = targets
        # ``SimBackend.step(..., sim_substeps)`` invokes the owner callback
        # once per physics substep.  Reset the counter at the control-step
        # boundary so source-style delayed observations can skip substep 0.
        self._physics_substep_count = 0
        if self._source_semantics:
            self._source_zero_torque_active[:] = self._source_zero_torque_steps_remaining > 0
            self._source_zero_torque_steps_remaining[:] = np.maximum(
                self._source_zero_torque_steps_remaining - 1, 0
            )
        return targets

    def _pre_step_motor_control(self, backend: Any, native_targets: np.ndarray) -> np.ndarray:
        # ``backend`` is the callback argument by contract; never read a
        # captured backend instance here (important for vectorized pools).
        self._capture_source_contact_force(backend)
        full_pos = np.asarray(backend.get_dof_pos(), dtype=self._np_dtype)
        full_vel = np.asarray(backend.get_dof_vel(), dtype=self._np_dtype)
        if self._use_obs_delay and self._obs_delay_step_unit == "physics":
            # Source ``_apply_action`` sees ``obs_update_flag == 1`` on the
            # first substep immediately after reset/observation construction,
            # so that substep does not push a delayed frame.  Push only the
            # remaining pre-step frames; ``update_state`` pushes the settled
            # post-step frame once the backend has completed all substeps.
            if self._physics_substep_count > 0:
                self._capture_physics_delayed_observation(backend, full_pos, full_vel)
            self._physics_substep_count += 1
        targets = np.asarray(native_targets, dtype=self._np_dtype)
        if self._use_act_delay:
            # Delay leg and wheel targets independently, as in the source
            # DelayBuffer setup.  Springs are state-dependent and therefore
            # intentionally remain outside this action queue.
            self._delayed_native_targets[...] = targets
            if "leg_actions" in self._act_delay_buffers:
                delayed_leg = self._act_delay_buffers["leg_actions"].compute(
                    targets[:, self._native_leg_indices]
                )
                self._delayed_native_targets[:, self._native_leg_indices] = delayed_leg
            if "wheel_actions" in self._act_delay_buffers:
                delayed_wheel = self._act_delay_buffers["wheel_actions"].compute(
                    targets[:, self._native_wheel_indices]
                )
                self._delayed_native_targets[:, self._native_wheel_indices] = delayed_wheel
            targets = self._delayed_native_targets
        ctrl = compute_wheelbipe_motor_ctrl(
            targets,
            full_pos,
            full_vel,
            leg_pos_indices=self._leg_pos_indices,
            leg_vel_indices=self._leg_vel_indices,
            wheel_vel_indices=self._wheel_vel_indices,
            spring_pos_indices=self._spring_pos_indices,
            spring_vel_indices=self._spring_vel_indices,
            native_leg_indices=self._native_leg_indices,
            native_wheel_indices=self._native_wheel_indices,
            native_spring_indices=self._native_spring_indices,
            leg_kp=self._motor_kp,
            leg_kd=self._motor_kd,
            wheel_kd=self._wheel_kd,
            spring_force=float(self._cfg.control_config.spring_force),
            spring_damping=float(self._cfg.control_config.spring_damping),
            spring_mode=str(self._cfg.control_config.spring_mode),
            spring_offset=float(self._cfg.control_config.spring_offset),
            spring_linear_up=float(self._cfg.control_config.spring_linear_up),
            spring_linear_down=float(self._cfg.control_config.spring_linear_down),
            spring_linear_length=float(self._cfg.control_config.spring_linear_length),
            spring_force_random=self._spring_force_random,
            lower=self._source_ctrl_lower if self._source_semantics else self._ctrl_lower,
            upper=self._source_ctrl_upper if self._source_semantics else self._ctrl_upper,
            out=self._last_motor_ctrl,
        )
        if self._source_semantics:
            spring_slots = self._native_spring_indices
            spring_vel = full_vel[:, self._spring_vel_indices]
            ctrl[:, spring_slots] -= (
                self._spring_damping - float(self._cfg.control_config.spring_damping)
            ).astype(self._np_dtype) * spring_vel
            ctrl[:, spring_slots] = np.clip(
                ctrl[:, spring_slots],
                self._ctrl_lower[spring_slots],
                self._ctrl_upper[spring_slots],
            )

            # Source ``randomize_actuator_effort_output`` perturbs the
            # explicit actuator's nominal effort and only then applies the
            # actuator effort clip.
            ctrl[:, self._native_leg_indices] *= self._leg_effort_scale
            ctrl[:, self._native_leg_indices] = np.clip(
                ctrl[:, self._native_leg_indices],
                self._ctrl_lower[self._native_leg_indices],
                self._ctrl_upper[self._native_leg_indices],
            )
            ctrl[:, self._native_wheel_indices] *= self._wheel_effort_scale
            ctrl[:, self._native_wheel_indices] = np.clip(
                ctrl[:, self._native_wheel_indices],
                self._ctrl_lower[self._native_wheel_indices],
                self._ctrl_upper[self._native_wheel_indices],
            )
        if self._gimbal_enabled:
            # Gimbal channels are physical low-level controls, not policy
            # outputs.  Compute them from the cached owner targets and the
            # declared backend state arrays on every physics substep.
            gimbal_pos = full_pos[:, self._gimbal_pos_indices]
            gimbal_vel = full_vel[:, self._gimbal_vel_indices]
            gimbal_cfg = self._cfg.gimbal
            if self._gimbal_control_mode == "heading_pd":
                base_yaw = np_yaw_from_quat(
                    np.asarray(backend.get_base_quat(), dtype=self._np_dtype)
                )
                yaw_error = np_wrap_to_pi(self._gimbal_heading_target - base_yaw - gimbal_pos[:, 0])
                yaw_link_rate_w = self._get_gimbal_yaw_link_ang_vel_z_w(backend)
                yaw_ctrl = (
                    self._gimbal_heading_kp * yaw_error - self._gimbal_heading_kd * yaw_link_rate_w
                )
            else:
                yaw_ctrl = float(gimbal_cfg.velocity_kd) * (
                    self._gimbal_yaw_velocity_target - gimbal_vel[:, 0]
                )
            pitch_ctrl = (
                self._gimbal_pitch_kp * (self._gimbal_pitch_target - gimbal_pos[:, 1])
                - self._gimbal_pitch_kd * gimbal_vel[:, 1]
            )
            if self._source_semantics and self._gimbal_control_mode != "heading_pd":
                yaw_ctrl = self._gimbal_yaw_kd * (
                    self._gimbal_yaw_velocity_target - gimbal_vel[:, 0]
                )
            ctrl[:, self._native_gimbal_indices[0]] = yaw_ctrl
            ctrl[:, self._native_gimbal_indices[1]] = pitch_ctrl
            # Integer-array indexing returns a copy in NumPy.  Clipping with
            # ``out=ctrl[:, indices]`` therefore leaves the original control
            # matrix unchanged and can leak an over-limit gimbal torque to the
            # backend.  Clip a temporary view and assign it back explicitly.
            gimbal_ctrl = np.clip(
                ctrl[:, self._native_gimbal_indices],
                self._ctrl_lower[self._native_gimbal_indices],
                self._ctrl_upper[self._native_gimbal_indices],
            )
            ctrl[:, self._native_gimbal_indices] = gimbal_ctrl
        if self._source_semantics and np.any(self._source_zero_torque_active):
            disabled = self._source_zero_torque_active
            disabled_ids = np.flatnonzero(disabled)
            # The source ground-reset grace period disables commanded leg and
            # wheel effort while keeping the passive spring controller active.
            ctrl[np.ix_(disabled_ids, self._native_leg_indices)] = 0.0
            ctrl[np.ix_(disabled_ids, self._native_wheel_indices)] = 0.0
        return ctrl

    def _capture_post_step_delayed_observation(
        self,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> None:
        """Append the settled source sensor frame after a control step.

        This is the owner-side equivalent of the final ``_get_observations``
        call in Isaac's DirectRLEnv loop.  It deliberately consumes the
        already-read post-step arrays so no backend-private access or second
        simulator read is introduced.
        """

        if not self._use_obs_delay or self._obs_delay_step_unit != "physics":
            return
        samples = {
            "gyro": gyro,
            "gravity": projected_gravity,
            # ``update_state`` receives the policy-ordered six-DOF arrays
            # from ``get_dof_pos/get_dof_vel``; unlike the callback's full
            # backend arrays they must not be indexed a second time.
            "joint_pos": dof_pos,
            "joint_vel": dof_vel,
        }
        for group, sample in samples.items():
            buffer = self._obs_delay_buffers.get(group)
            if buffer is not None:
                self._physics_delayed_obs[group] = buffer.compute(
                    np.asarray(sample, dtype=self._np_dtype)
                )

    def _capture_physics_delayed_observation(
        self, backend: Any, full_pos: np.ndarray, full_vel: np.ndarray
    ) -> None:
        """Push one raw sensor frame into the physics-step delay buffers.

        This method is called only from the declared ``SimBackend`` pre-step
        callback.  Joint arrays are supplied by the motor controller to avoid
        a second backend read; IMU/gravity reads use only methods in the
        abstract backend contract.
        """

        if self._delay_reset_in_progress:
            return
        if "gyro" in self._obs_delay_buffers:
            gyro = np.asarray(backend.get_sensor_data(self._cfg.sensor.gyro), dtype=self._np_dtype)
            self._physics_delayed_obs["gyro"] = self._obs_delay_buffers["gyro"].compute(gyro)
        if "gravity" in self._obs_delay_buffers:
            quat = np.asarray(backend.get_base_quat(), dtype=self._np_dtype)
            world_gravity = np.broadcast_to(
                np.asarray([0.0, 0.0, -1.0], dtype=self._np_dtype),
                (self._num_envs, 3),
            )
            gravity = np.asarray(np_quat_apply_inverse(quat, world_gravity), dtype=self._np_dtype)
            self._physics_delayed_obs["gravity"] = self._obs_delay_buffers["gravity"].compute(
                gravity
            )
        if "joint_pos" in self._obs_delay_buffers:
            policy_pos = np.asarray(full_pos[:, self._policy_pos_indices], dtype=self._np_dtype)
            self._physics_delayed_obs["joint_pos"] = self._obs_delay_buffers["joint_pos"].compute(
                policy_pos
            )
        if "joint_vel" in self._obs_delay_buffers:
            policy_vel = np.asarray(full_vel[:, self._policy_vel_indices], dtype=self._np_dtype)
            self._physics_delayed_obs["joint_vel"] = self._obs_delay_buffers["joint_vel"].compute(
                policy_vel
            )

    def _get_delayed_observation(
        self,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the policy sensor stream after the configured delay."""

        if not self._use_obs_delay or self._delay_reset_in_progress:
            return gyro, projected_gravity, dof_pos, dof_vel
        if self._obs_delay_step_unit == "physics":
            # The callback has sampled all substeps that preceded this
            # control observation.  Missing groups intentionally fall back to
            # current values, allowing a profile to delay only one sensor
            # family without silently zeroing the others.
            return (
                self._physics_delayed_obs.get("gyro", gyro),
                self._physics_delayed_obs.get("gravity", projected_gravity),
                self._physics_delayed_obs.get("joint_pos", dof_pos),
                self._physics_delayed_obs.get("joint_vel", dof_vel),
            )

        delayed_gyro = gyro
        delayed_gravity = projected_gravity
        delayed_pos = dof_pos
        delayed_vel = dof_vel
        if "gyro" in self._obs_delay_buffers:
            delayed_gyro = self._obs_delay_buffers["gyro"].compute(gyro)
        if "gravity" in self._obs_delay_buffers:
            delayed_gravity = self._obs_delay_buffers["gravity"].compute(projected_gravity)
        if "joint_pos" in self._obs_delay_buffers:
            delayed_pos = self._obs_delay_buffers["joint_pos"].compute(dof_pos)
        if "joint_vel" in self._obs_delay_buffers:
            delayed_vel = self._obs_delay_buffers["joint_vel"].compute(dof_vel)
        return delayed_gyro, delayed_gravity, delayed_pos, delayed_vel

    def get_local_linvel(self) -> np.ndarray:
        link_velocity = super().get_local_linvel()
        if not self._source_semantics:
            return link_velocity
        gyro = np.asarray(self._backend.get_sensor_data(self._cfg.sensor.gyro))
        return np.asarray(
            link_velocity + np.cross(gyro, self._source_base_com_b), dtype=self._np_dtype
        )

    def _source_wheel_linear_velocity(self) -> np.ndarray:
        """Rotate COM velocity differences, without rotating-frame transport.

        The source subtracts root COM world velocity from wheel COM world
        velocity, then rotates and projects onto the sagittal plane. A backend
        relative-frame velocity also subtracts omega cross displacement and
        therefore is a different signal, even when all COM offsets are zero.
        """
        wheel_quat = self._backend.get_body_quat_w(self._wheel_body_ids)
        offset_w = np_quat_apply_batched(wheel_quat, self._source_wheel_com_b)
        wheel_velocity = self._backend.get_body_lin_vel_w(self._wheel_body_ids)
        wheel_omega = self._backend.get_body_ang_vel_w(self._wheel_body_ids)
        wheel_com_velocity = wheel_velocity + np.cross(wheel_omega, offset_w)
        root_quat = self._backend.get_base_quat()
        root_com_velocity = np_quat_apply(root_quat, self.get_local_linvel())
        relative_velocity = np.asarray(
            np_quat_apply_batched(
                np_quat_conjugate_batched(root_quat[:, None, :]),
                wheel_com_velocity - root_com_velocity[:, None, :],
            ),
            dtype=self._np_dtype,
        )
        relative_velocity[:, :, 1] = 0.0
        return relative_velocity

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # The source DirectRLEnv order is: done -> reward (using the command
        # that drove this action) -> command resample/state-machine update ->
        # observation.  Keep that order here; updating commands before the
        # reward would train against the next command while the action was
        # still conditioned on the previous one.
        self._capture_source_contact_force(self._backend)
        # Keep the measured body height available to compact history owners.
        # Source V14 privileged frames append ``obs_height`` (world-frame
        # root z, optionally clipped by the v1 owner), whereas
        # ``height_commands`` is the target encoded in the policy observation.
        # Rough rewards receive a separate terrain-relative signal below.
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        if self._source_semantics:
            observed_height, reward_height = self._source_height_signals(
                base_pos, self._backend.get_base_quat()
            )
        else:
            observed_height = base_pos[:, 2].copy()
            if self._terrain_surface_sample_height is not None:
                observed_height -= np.asarray(
                    self._terrain_surface_sample_height(base_pos[:, :2]), dtype=self._np_dtype
                )
            reward_height = observed_height
        state.info["observed_height"] = observed_height
        state.info["relative_observed_height"] = reward_height
        linvel = self.get_local_linvel()
        gyro = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.gyro), dtype=self._np_dtype
        )
        upvector = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.upvector), dtype=self._np_dtype
        )
        # The up-vector sensor is world-frame body +Z (needed for the
        # termination threshold).  Policy observations and orientation
        # rewards require gravity projected into the body frame, so derive it
        # from the backend's canonical wxyz base quaternion.
        projected_gravity = self.get_projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        # The source loop appends the settled post-step frame from its final
        # ``_get_observations`` call after the four physics substeps.
        self._capture_post_step_delayed_observation(gyro, projected_gravity, dof_pos, dof_vel)
        state.info["torques"] = self._last_motor_ctrl.copy()
        state.info["qacc"] = self.get_dof_acc()
        self._last_policy_vel[:] = dof_vel
        if self._source_semantics:
            # Isaac's reset event counts control steps between gain/effort
            # randomizations.  Saturating in-place keeps long-running jobs
            # deterministic without int32 rollover.
            np.add(
                self._source_control_randomization_age,
                1,
                out=self._source_control_randomization_age,
                where=(
                    self._source_control_randomization_age
                    < np.iinfo(self._source_control_randomization_age.dtype).max
                ),
            )
            wheel_contact, undesired_contact, base_contact = self._source_contact_features()
            state.info["wheel_contact_state"] = wheel_contact
            state.info["undesired_contact"] = undesired_contact
            state.info["base_contact"] = base_contact
            state.info["reset_contact"] = self._source_reset_contact.copy()
            state.info["wheel_pos_b"] = np.asarray(
                self._backend.get_body_pos_b(self._wheel_body_ids), dtype=self._np_dtype
            )
            state.info["wheel_lin_vel_b"] = self._source_wheel_linear_velocity()
            # Preserve raw, undelayed tensors for the privileged critic.  They
            # are regular owner state, not a second simulator read inside the
            # pure observation builder.
            state.info["_source_raw_gyro"] = gyro
            state.info["_source_raw_gravity"] = projected_gravity
            state.info["_source_raw_dof_pos"] = dof_pos
            state.info["_source_raw_dof_vel"] = dof_vel
        # State-machine transitions are evaluated on the post-physics frame
        # before done/reward construction, so source task-mode reward terms
        # receive the same transition metadata as the original owner.
        self._update_state_machine(state.info)
        state.info["costs"] = self._compute_np3o_costs(
            base_pos=base_pos,
            projected_gravity=projected_gravity,
            gyro=gyro,
            dof_vel=dof_vel,
            torques=self._last_motor_ctrl[
                :, self._native_leg_indices.tolist() + self._native_wheel_indices.tolist()
            ],
        )
        if self._source_semantics:
            terminated = self._compute_source_terminated(
                info=state.info,
                base_pos=base_pos,
                base_quat=np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype),
                linvel=linvel,
                gyro=gyro,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
            )
        else:
            terminated = self._compute_terminated(upvector)
        if self._state_machine is not None:
            np.logical_or(
                terminated,
                np.asarray(state.info.get("state_machine_failure", False), dtype=bool),
                out=terminated,
            )
        state.info["terminated"] = terminated
        reward = self._compute_reward(state.info, linvel, gyro, projected_gravity, dof_pos, dof_vel)

        # Advance the command generator only after reward construction.  The
        # resulting command/state-machine values belong to the observation
        # returned for the next policy action, matching Isaac's
        # ``_get_rewards`` -> ``_get_observations`` lifecycle.
        self._update_commands(state.info)
        self._apply_state_machine_command_overrides(state.info)

        # Only the policy stream is delayed.  Reward, safety and the source
        # privileged critic consume the current simulator state.
        obs_gyro, obs_gravity, obs_dof_pos, obs_dof_vel = self._get_delayed_observation(
            gyro, projected_gravity, dof_pos, dof_vel
        )
        obs = self._compute_obs(
            state.info,
            linvel,
            obs_gyro,
            obs_gravity,
            obs_dof_pos,
            obs_dof_vel,
            env_ids=None,
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _reset_done_envs(self) -> None:
        # NpEnv computes time limits and task boundary truncations *after*
        # update_state. Consume the completed done mask here, before the
        # reset plan replaces the episode's position and distance baseline.
        # Otherwise successful time-limit episodes never promote terrain.
        assert self._state is not None
        done = self._state.terminated | self._state.truncated
        if self.step_counter > 0 and np.any(done):
            stats = self._spawn.update_on_done(
                np.flatnonzero(done).astype(np.int32), self._backend.get_base_pos()[done]
            )
            if stats:
                self._state.info.setdefault("log", {}).update(
                    {f"terrain/{key}": float(value) for key, value in stats.items()}
                )
        super()._reset_done_envs()

    def _update_state_machine(self, info: dict[str, Any]) -> None:
        """Advance the optional owner state machine and apply mode envelopes."""

        state_machine = self._state_machine
        if state_machine is None:
            return
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        wheel_pos = np.stack(
            [
                np.asarray(self._backend.get_sensor_data(sensor_name), dtype=self._np_dtype)
                for sensor_name in self._state_machine_wheel_sensor_names
            ],
            axis=1,
        )
        if self._terrain_surface_sample_height is None:
            terrain = np.zeros((self._num_envs,), dtype=self._np_dtype)
        else:
            terrain = np.asarray(
                self._terrain_surface_sample_height(base_pos[:, :2]), dtype=self._np_dtype
            )
        transition = state_machine.update(wheel_pos, terrain)
        info["state_machine_state"] = transition["state"]
        info["state_machine_contact"] = transition["contact"]
        info["state_machine_failure"] = transition["failure"]
        info["state_machine_state_time"] = transition["state_time"].astype(self._np_dtype)
        info["control_mode_obs"] = state_machine.control_mode_obs(state_dtype=self._np_dtype)
        self._apply_state_machine_command_overrides(info)

    def _apply_state_machine_command_overrides(self, info: dict[str, Any]) -> None:
        """Apply current machine envelopes without advancing its state."""

        state_machine = self._state_machine
        if state_machine is None:
            return
        commands = np.asarray(
            info.get("commands", np.zeros((self._num_envs, 3))), dtype=self._np_dtype
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
        commands, heights = state_machine.apply_command_overrides(commands, heights)
        info["commands"] = commands
        info["height_commands"] = heights

    def _compute_np3o_costs(
        self,
        *,
        base_pos: np.ndarray,
        projected_gravity: np.ndarray,
        gyro: np.ndarray,
        dof_vel: np.ndarray,
        torques: np.ndarray,
    ) -> np.ndarray:
        """Compute the five source NP3O constraint channels in the owner layer."""
        count = int(getattr(self._cfg, "num_costs", 0) or 0)
        if count <= 0:
            return np.zeros((base_pos.shape[0], 0), dtype=self._np_dtype)
        tilt_limit = np.sin(np.deg2rad(float(self._cfg.np3o_tilt_limit_deg)))
        tilt = np.maximum(np.linalg.norm(projected_gravity[:, :2], axis=1) - tilt_limit, 0.0) ** 2
        height = base_pos[:, 2].copy()
        surface_fn = self._terrain_surface_sample_height
        if surface_fn is not None:
            height -= np.asarray(surface_fn(base_pos[:, :2]), dtype=self._np_dtype)
        height_cost = np.maximum(float(self._cfg.np3o_body_height_min) - height, 0.0) ** 2
        height_cost += np.maximum(height - float(self._cfg.np3o_body_height_max), 0.0) ** 2
        ang = (
            np.maximum(
                np.linalg.norm(gyro[:, :2], axis=1)
                / max(float(self._cfg.np3o_ang_vel_xy_limit), 1e-6)
                - 1.0,
                0.0,
            )
            ** 2
        )
        torque = np.mean(
            np.maximum(np.abs(torques) / max(float(self._cfg.np3o_torque_limit), 1e-6) - 1.0, 0.0)
            ** 2,
            axis=1,
        )
        velocity = np.mean(
            np.maximum(
                np.abs(dof_vel) / max(float(self._cfg.np3o_joint_velocity_limit), 1e-6) - 1.0, 0.0
            )
            ** 2,
            axis=1,
        )
        costs = np.stack((tilt, height_cost, ang, torque, velocity), axis=1)[:, :count]
        clip = float(self._cfg.np3o_cost_clip)
        return (
            np.nan_to_num(costs, nan=0.0, posinf=clip, neginf=0.0)
            .clip(0.0, clip)
            .astype(self._np_dtype, copy=False)
        )

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        # Validate the caller's index dtype/values before coercing to int32.
        # A direct ``astype(np.int32)`` would silently turn a boolean mask
        # (``[True]`` -> ``[1]``) or fractional id (``[0.5]`` -> ``[0]``)
        # into a different environment, violating the vectorized reset
        # contract and potentially clearing the wrong delay history.
        raw_env_ids = np.asarray(env_indices)
        self._reset_delay_buffers(raw_env_ids)
        env_ids = raw_env_ids.astype(np.int32, copy=False).reshape(-1)
        self._update_source_him_curriculum_on_reset(env_ids)
        # Reset observations are built synchronously by the DR provider.  Do
        # not consume a latency frame while constructing that initial packet;
        # the first post-reset physics step starts the history exactly once.
        self._delay_reset_in_progress = True
        try:
            obs, info = super().reset(env_ids)
        finally:
            self._delay_reset_in_progress = False
        if env_ids.size:
            self._last_policy_vel[env_ids] = self.get_dof_vel()[env_ids]
            self._reset_gimbal_targets(env_ids)
        return obs, info

    def sync_source_episode_length(self, episode_steps: np.ndarray) -> None:
        """Apply RSL-RL's random-start age to source reset envelopes.

        RSL-RL assigns ``episode_length_buf`` after the initial reset.  The
        source reset envelope is expressed in absolute episode time, so the
        randomized age consumes the corresponding portions of every absolute
        startup window before the first rollout action.  Isaac's source
        runner sets ``episode_length_buf`` after reset, so a reset command or
        zero-torque grace period that has already elapsed must not remain
        active in the migrated owner.
        """

        if not self._source_semantics or not bool(self._cfg.source_episode_age_sync):
            return
        ages = np.asarray(episode_steps, dtype=np.float64).reshape(-1)
        if ages.shape != (self._num_envs,) or not np.all(np.isfinite(ages)):
            raise ValueError(
                "source episode age must contain one finite value per environment; "
                f"got {ages.shape} for {self._num_envs} environments"
            )
        ages = np.maximum(np.floor(ages), 0.0).astype(np.int64)
        for remaining in (
            self._source_zero_torque_steps_remaining,
            self._source_ground_command_steps_remaining,
            self._source_air_command_steps_remaining,
        ):
            remaining[...] = np.maximum(remaining.astype(np.int64) - ages, 0).astype(
                remaining.dtype,
                copy=False,
            )

    @property
    def source_him_assist_force_z(self) -> float:
        curriculum = self._source_him_curriculum
        return 0.0 if curriculum is None else float(curriculum.assist_force_z)

    def _update_source_him_curriculum_on_reset(self, env_ids: np.ndarray) -> None:
        curriculum = self._source_him_curriculum
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        if curriculum is None or ids.size == 0:
            return
        if self._source_him_reward_buffer_initialized:
            advanced = curriculum.record_completed_batch(
                self._source_him_track_height_episode_sum[ids],
                max_episode_length_s=float(self._cfg.max_episode_seconds or 20.0),
            )
            if advanced:
                self._source_runtime_reward_scales = curriculum.reward_scales
                # ``BaseVerticalAssistForceProgression`` applies the new
                # stage to every environment after a transition.
                self._source_him_assist_latched.fill(True)
                logger.info(
                    "WheelBipe HIM curriculum advanced to stage %d: force_z=%.1f N",
                    curriculum.stage,
                    curriculum.assist_force_z,
                )
        # ``apply_on_compute=True`` and the term's reset hook both write the
        # current assist for the reset batch.  This deliberately overwrites a
        # prior interval disturbance on those environments.
        self._source_him_assist_latched[ids] = True
        self._source_him_track_height_episode_sum[ids] = 0.0

    def _reset_gimbal_targets(self, env_ids: np.ndarray) -> None:
        """Sample source-compatible gimbal targets on the reset cold path."""

        if not self._gimbal_enabled or env_ids.size == 0:
            return
        if self._source_semantics:
            # The source reset provider sampled these targets and embedded
            # their matching qpos/qvel values into the ResetPlan before the
            # backend state write.  Resampling here would desynchronize the
            # explicit controller from the physical joint state.
            return
        cfg = self._cfg.gimbal
        velocity_low, velocity_high = (float(v) for v in cfg.yaw_velocity_range)
        self._gimbal_yaw_velocity_target[env_ids] = np.random.uniform(
            velocity_low, velocity_high, size=env_ids.size
        ).astype(self._np_dtype)
        heading_low, heading_high = (float(v) for v in cfg.yaw_heading_range)
        heading_target_mode = str(cfg.heading_target_mode).strip().lower()
        if self._gimbal_control_mode == "heading_pd" and heading_target_mode == "fixed":
            self._gimbal_heading_target[env_ids] = float(cfg.fixed_heading)
        elif bool(cfg.randomize_heading) or self._gimbal_control_mode == "heading_pd":
            self._gimbal_heading_target[env_ids] = np.random.uniform(
                heading_low, heading_high, size=env_ids.size
            ).astype(self._np_dtype)
        else:
            self._gimbal_heading_target[env_ids] = 0.0
        self._gimbal_pitch_target[env_ids] = float(cfg.pitch_target)
        if self._state is not None:
            self._state.info["gimbal_yaw_velocity_target"] = self._gimbal_yaw_velocity_target.copy()
            self._state.info["gimbal_pitch_target"] = self._gimbal_pitch_target.copy()

    def _apply_source_gimbal_reset_to_plan(
        self,
        env_ids: np.ndarray,
        *,
        base_heading: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Embed pinned V14 gimbal reset state in a backend-neutral plan.

        ``qpos``/``qvel`` are complete generalized-state rows consumed by the
        public :meth:`SimBackend.set_state` contract.  Joint index discovery
        happened once during materialization; this reset path only writes the
        cached columns and controller targets.
        """

        if not self._source_semantics:
            raise RuntimeError("source gimbal reset plan requires training_semantics='source_v14'")
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        headings = np.asarray(base_heading, dtype=np.float64).reshape(-1)
        positions = np.asarray(qpos)
        velocities = np.asarray(qvel)
        count = ids.size
        if count == 0:
            return {}
        if headings.shape != (count,):
            raise ValueError(
                f"source gimbal reset base_heading must have shape ({count},), got {headings.shape}"
            )
        if positions.ndim != 2 or positions.shape[0] != count:
            raise ValueError(
                "source gimbal reset qpos must have one row per env id; "
                f"got {positions.shape} for {count} ids"
            )
        if velocities.ndim != 2 or velocities.shape[0] != count:
            raise ValueError(
                "source gimbal reset qvel must have one row per env id; "
                f"got {velocities.shape} for {count} ids"
            )
        if self._gimbal_pos_indices.shape != (2,) or self._gimbal_vel_indices.shape != (2,):
            raise RuntimeError(
                "source gimbal reset requires cached yaw/pitch position and velocity indices"
            )
        pos_columns = 7 + self._gimbal_pos_indices
        vel_columns = 6 + self._gimbal_vel_indices
        if np.any(pos_columns >= positions.shape[1]) or np.any(vel_columns >= velocities.shape[1]):
            raise ValueError(
                "source gimbal reset indices exceed the complete backend state width: "
                f"qpos={positions.shape[1]}, qvel={velocities.shape[1]}"
            )

        cfg = self._cfg.gimbal
        velocity_low, velocity_high = (float(value) for value in cfg.yaw_velocity_range)
        yaw_velocity_target = np.random.uniform(velocity_low, velocity_high, size=count).astype(
            self._np_dtype
        )
        self._gimbal_yaw_velocity_target[ids] = yaw_velocity_target
        pitch_target = np.full((count,), float(cfg.pitch_target), dtype=self._np_dtype)
        self._gimbal_pitch_target[ids] = pitch_target
        positions[:, pos_columns[1]] = pitch_target
        velocities[:, vel_columns[1]] = 0.0

        mode = str(cfg.control_mode).strip().lower()
        target_mode = str(cfg.heading_target_mode).strip().lower()
        if mode == "heading_pd":
            if target_mode == "fixed":
                heading_target = np.full((count,), float(cfg.fixed_heading), dtype=self._np_dtype)
            elif target_mode == "sampled":
                heading_low, heading_high = (float(value) for value in cfg.yaw_heading_range)
                heading_target = np.random.uniform(heading_low, heading_high, size=count).astype(
                    self._np_dtype
                )
            else:  # ``WheelbipeGimbalConfig.validate`` should make this unreachable.
                raise ValueError(
                    "source heading-PD reset requires heading_target_mode='sampled' or 'fixed', "
                    f"got {cfg.heading_target_mode!r}"
                )
            heading_target = np_wrap_to_pi(heading_target).astype(self._np_dtype, copy=False)
            self._gimbal_heading_target[ids] = heading_target
            positions[:, pos_columns[0]] = np_wrap_to_pi(heading_target - headings)
            velocities[:, vel_columns[0]] = 0.0
        elif mode == "velocity":
            # Base/Flat-v0/v1 and all public history-policy variants use the
            # inherited IdealPD velocity owner: yaw starts at zero with qvel
            # equal to the newly sampled velocity target.
            self._gimbal_heading_target[ids] = 0.0
            positions[:, pos_columns[0]] = 0.0
            velocities[:, vel_columns[0]] = yaw_velocity_target
        else:  # ``WheelbipeGimbalConfig.validate`` should make this unreachable.
            raise ValueError(f"unsupported source gimbal reset control mode {cfg.control_mode!r}")

        return {
            "gimbal_yaw_velocity_target": yaw_velocity_target.copy(),
            "gimbal_heading_target": self._gimbal_heading_target[ids].copy(),
            "gimbal_pitch_target": pitch_target.copy(),
        }

    def _get_gimbal_yaw_link_ang_vel_z_w(self, backend: SimBackend) -> np.ndarray:
        """Return source heading-PD derivative state through ``SimBackend``."""

        if self._gimbal_yaw_body_ids.shape != (1,):
            raise RuntimeError("heading-PD requires one cached gimbal_yaw_link body id")
        angular_velocity = np.asarray(
            backend.get_body_ang_vel_w(self._gimbal_yaw_body_ids), dtype=self._np_dtype
        )
        expected = (self._num_envs, 1, 3)
        if angular_velocity.shape != expected:
            raise ValueError(
                "gimbal yaw-link world angular velocity must have shape "
                f"{expected}, got {angular_velocity.shape}"
            )
        # This includes both the floating base's world yaw rate and relative
        # yaw-joint motion, matching source ``body_ang_vel_w[..., 2]``.
        return angular_velocity[:, 0, 2]

    def close(self) -> None:
        materialized = getattr(self, "_materialized_gimbal", None)
        try:
            super().close()
        finally:
            if materialized is not None:
                materialized.cleanup()
                self._materialized_gimbal = None

    def _compute_terminated(self, upvector: np.ndarray) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        finite = np.all(np.isfinite(upvector), axis=1) & np.all(np.isfinite(base_pos), axis=1)
        return (~finite) | (upvector[:, 2] < 0.5) | (base_pos[:, 2] < 0.06)

    def _compute_source_terminated(
        self,
        *,
        info: dict[str, Any],
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        """Apply pinned V14 physical persistence and immediate numeric safety."""

        del base_pos, dof_pos
        arrays = (linvel, gyro, dof_vel)
        finite = np.ones((self._num_envs,), dtype=bool)
        for values in arrays:
            finite &= np.all(np.isfinite(values), axis=1)
        numeric_outlier = (
            np.any(np.abs(dof_vel) > float(self._cfg.terminate_joint_vel_abs), axis=1)
            | np.any(np.abs(gyro) > float(self._cfg.terminate_root_ang_vel_abs), axis=1)
            | np.any(np.abs(linvel) > float(self._cfg.terminate_root_lin_vel_abs), axis=1)
        )

        obs_outlier = (
            bool(self._cfg.terminate_on_obs_outlier)
            & (int(self._source_value_debug_step) > 10)
            & (
                self._source_obs_safety_nonfinite
                | (self._source_obs_safety_max_abs > float(self._cfg.terminate_obs_abs))
            )
        )
        # Commit the frame prepared by ``_compute_obs`` only after the done
        # calculation has consumed the prior cache.  Autoreset observations
        # overwrite committed values for reset envs through the ``env_ids``
        # branch below, matching DirectRLEnv's done->reward->reset->obs order.
        self._source_obs_safety_nonfinite[:] = self._source_obs_safety_pending_nonfinite
        self._source_obs_safety_max_abs[:] = self._source_obs_safety_pending_max_abs
        immediate = (~finite) | numeric_outlier | obs_outlier
        info["numerical_safety_failure"] = immediate.copy()

        roll, pitch = np_roll_pitch_from_quat(base_quat)
        physical = (
            np.asarray(info.get("reset_contact", info.get("base_contact", False)), dtype=bool)
            | (np.abs(roll) > np.deg2rad(float(self._cfg.termination_roll_deg)))
            | (np.abs(pitch) > np.deg2rad(float(self._cfg.termination_pitch_deg)))
        )
        if bool(self._cfg.termination_duration_enabled):
            terminated, counter = apply_source_v14_termination_duration(
                physical | immediate,
                immediate,
                self._source_termination_counter,
                steps=int(self._cfg.termination_duration_steps),
            )
            self._source_termination_counter[:] = counter
            return terminated
        self._source_termination_counter.fill(0)
        return physical | immediate

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
        n = int(gyro.shape[0])
        commands = np.asarray(info["commands"], dtype=self._np_dtype)
        heights = np.asarray(
            info.get("height_commands", np.full((n,), self._reward_cfg.base_height_target)),
            dtype=self._np_dtype,
        )
        actions = np.asarray(
            info.get("current_actions", np.zeros((n, NUM_POLICY_ACTIONS))), dtype=self._np_dtype
        )
        control_mode = np.asarray(
            info.get(
                "control_mode_obs",
                np.broadcast_to(NORMAL_CONTROL_MODE, (n, 7)),
            ),
            dtype=self._np_dtype,
        )
        mode_scale = np.ones((7,), dtype=self._np_dtype)
        if self._source_semantics:
            mode_scale = np.asarray(self._cfg.ctrl_mode_obs_scale, dtype=self._np_dtype).reshape(-1)
            if mode_scale.shape != (7,) or not np.all(np.isfinite(mode_scale)):
                raise ValueError(
                    "ctrl_mode_obs_scale must contain seven finite source input scales"
                )
            gimbal_mode_cfg = getattr(self._cfg, "gimbal_spin_translate", None)
            gimbal_mode_enabled = (
                bool(gimbal_mode_cfg.get("enabled", False))
                if isinstance(gimbal_mode_cfg, dict)
                else bool(getattr(gimbal_mode_cfg, "enabled", False))
            )
            if gimbal_mode_enabled:
                # Source v2 replaces the generic state-machine tail with
                # continuous gimbal features and explicitly resets all seven
                # input scales to one in __post_init__.
                mode_scale = np.ones((7,), dtype=self._np_dtype)
        leg_pos = dof_pos[:, :NUM_LEG_ACTIONS] - self.default_angles[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        projected_gravity = np.asarray(projected_gravity, dtype=self._np_dtype)
        policy_gyro = gyro
        policy_gravity = projected_gravity
        policy_leg_pos = leg_pos
        policy_leg_vel = leg_vel
        policy_wheel_vel = wheel_vel
        if self._source_semantics:
            # Materialize source policy noise exactly once so the actor and
            # raw-observation safety guard inspect the same delayed/noisy
            # values.  Passing ``noise_fn`` to the generic builder would hide
            # those samples from the immediate numerical-reset contract.
            noise_cfg = self._cfg.noise_config
            policy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
            policy_gravity = self._obs_noise(projected_gravity, noise_cfg.scale_gravity)
            policy_leg_pos = self._obs_noise(leg_pos, noise_cfg.scale_joint_angle)
            policy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
            policy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
            raw_policy = np.concatenate(
                (
                    commands,
                    heights.reshape(n, 1),
                    policy_gyro,
                    policy_gravity,
                    policy_leg_pos,
                    np.zeros((n, NUM_WHEEL_ACTIONS), dtype=self._np_dtype),
                    policy_leg_vel,
                    policy_wheel_vel,
                    actions,
                    control_mode,
                ),
                axis=1,
            )
            critic_gyro = np.asarray(info.get("_source_raw_gyro", gyro), dtype=self._np_dtype)
            critic_gravity = np.asarray(
                info.get("_source_raw_gravity", projected_gravity), dtype=self._np_dtype
            )
            critic_pos = np.asarray(info.get("_source_raw_dof_pos", dof_pos), dtype=self._np_dtype)
            critic_vel = np.asarray(info.get("_source_raw_dof_vel", dof_vel), dtype=self._np_dtype)
            raw_critic = np.concatenate(
                (
                    commands,
                    heights.reshape(n, 1),
                    critic_gyro,
                    critic_gravity,
                    critic_pos[:, :NUM_LEG_ACTIONS] - self.default_angles[None, :NUM_LEG_ACTIONS],
                    np.zeros((n, NUM_WHEEL_ACTIONS), dtype=self._np_dtype),
                    critic_vel,
                    actions,
                    np.asarray(linvel, dtype=self._np_dtype),
                    np.asarray(
                        info.get("observed_height", np.zeros((n,), dtype=self._np_dtype)),
                        dtype=self._np_dtype,
                    ).reshape(n, 1),
                    control_mode,
                ),
                axis=1,
            )
            combined_raw = np.concatenate((raw_policy, raw_critic), axis=1)
            current_nonfinite = ~np.all(np.isfinite(combined_raw), axis=1)
            finite_abs = np.where(np.isfinite(combined_raw), np.abs(combined_raw), 0.0)
            current_max_abs = np.max(finite_abs, axis=1).astype(self._np_dtype, copy=False)
            info["_source_obs_safety_nonfinite"] = current_nonfinite
            info["_source_obs_safety_max_abs"] = current_max_abs
            if env_ids is None:
                self._source_obs_safety_pending_nonfinite[:] = current_nonfinite
                self._source_obs_safety_pending_max_abs[:] = current_max_abs
            else:
                ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
                if ids.size != n:
                    raise ValueError(
                        "source reset observation env_ids must match the observation batch; "
                        f"got {ids.size} ids for {n} rows"
                    )
                # Source reset happens before its observation builder, so the
                # reset packet becomes the committed cache immediately.
                self._source_obs_safety_nonfinite[ids] = current_nonfinite
                self._source_obs_safety_max_abs[ids] = current_max_abs
                self._source_obs_safety_pending_nonfinite[ids] = current_nonfinite
                self._source_obs_safety_pending_max_abs[ids] = current_max_abs
        obs = build_wheelbipe_policy_observation(
            commands,
            heights,
            policy_gyro,
            policy_gravity,
            policy_leg_pos,
            policy_leg_vel,
            policy_wheel_vel,
            actions,
            noise_fn=None if self._source_semantics else self._obs_noise,
            noise_config=self._cfg.noise_config,
            control_mode=control_mode,
            control_mode_scale=mode_scale,
            source_training_clips=self._source_semantics,
        )
        if self._source_semantics:
            ids = (
                np.arange(self._num_envs, dtype=np.intp)
                if env_ids is None
                else np.asarray(env_ids, dtype=np.intp).reshape(-1)
            )
            if ids.size != n:
                raise ValueError(
                    f"source critic env_ids has {ids.size} rows for observation batch {n}"
                )
            raw_gyro = np.asarray(info.get("_source_raw_gyro", gyro), dtype=self._np_dtype)
            raw_gravity = np.asarray(
                info.get("_source_raw_gravity", projected_gravity), dtype=self._np_dtype
            )
            raw_pos = np.asarray(info.get("_source_raw_dof_pos", dof_pos), dtype=self._np_dtype)
            raw_vel = np.asarray(info.get("_source_raw_dof_vel", dof_vel), dtype=self._np_dtype)
            torques = np.asarray(
                info.get(
                    "torques",
                    np.zeros((n, self._num_native_actuators), dtype=self._np_dtype),
                ),
                dtype=self._np_dtype,
            )
            policy_torque = np.concatenate(
                (
                    torques[:, self._native_leg_indices],
                    torques[:, self._native_wheel_indices],
                ),
                axis=1,
            )[:, SOURCE_V14_PRIVILEGED_POLICY_PERMUTATION]
            stiffness = np.concatenate(
                (
                    self._motor_kp[ids],
                    np.zeros((n, NUM_WHEEL_ACTIONS), dtype=np.float64),
                ),
                axis=1,
            )[:, SOURCE_V14_PRIVILEGED_POLICY_PERMUTATION]
            damping = np.concatenate((self._motor_kd[ids], self._wheel_kd[ids]), axis=1)[
                :, SOURCE_V14_PRIVILEGED_POLICY_PERMUTATION
            ]

            def selected_lags(
                buffers: dict[str, WheelbipeDelayBuffer], names: tuple[str, ...]
            ) -> np.ndarray:
                columns = []
                for name in names:
                    buffer = buffers.get(name)
                    if buffer is None:
                        columns.append(np.zeros((n,), dtype=self._np_dtype))
                    else:
                        columns.append(np.asarray(buffer.lags[ids], dtype=self._np_dtype))
                return np.stack(columns, axis=1)

            obs_lags = selected_lags(
                self._obs_delay_buffers,
                ("gyro", "gravity", "joint_pos", "joint_vel"),
            )
            act_lags = selected_lags(self._act_delay_buffers, ("leg_actions", "wheel_actions"))
            wheel_lin_vel = info.get("wheel_lin_vel_b")
            if wheel_lin_vel is None:
                wheel_lin_vel = self._source_wheel_linear_velocity()[ids]
            wheel_contact = info.get("wheel_contact_state")
            if wheel_contact is None:
                wheel_contact = self._source_contact_features()[0][ids]
            observed_height = info.get("observed_height")
            if observed_height is None:
                selected_base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)[
                    ids
                ]
                # The critic uses world height only; terrain scanning belongs
                # to the reward update above, including for partial resets.
                observed_height, _reward_height = build_source_v14_height_signals(
                    selected_base_pos[:, 2],
                    None,
                    use_absolute_height=True,
                    clip_enabled=bool(self._cfg.height_obs_clip_enabled),
                    clip_range=self._cfg.height_obs_clip_range,
                )
            critic = build_source_v14_critic_observation(
                commands=commands,
                height_command=heights,
                gyro=raw_gyro,
                projected_gravity=raw_gravity,
                dof_pos=raw_pos,
                dof_vel=raw_vel,
                actions=actions,
                root_lin_vel_b=linvel,
                observed_height=np.asarray(observed_height, dtype=self._np_dtype),
                control_mode=control_mode,
                control_mode_scale=mode_scale,
                joint_stiffness=stiffness,
                joint_damping=damping,
                applied_torque=policy_torque,
                obs_delay_steps=obs_lags,
                act_delay_steps=act_lags,
                wheel_body_lin_vel_b=np.asarray(wheel_lin_vel, dtype=self._np_dtype),
                wheel_contact_state=np.asarray(wheel_contact, dtype=self._np_dtype),
                base_mass_scale=self._source_base_mass_scale[ids],
                wheel_material=self._source_wheel_material[ids],
                default_angles=self.default_angles,
            )
            return {"obs": obs, "critic": critic}
        # Keep the critic's privileged stream fixed at the 78D contract used by
        # the source PPO runs: policy obs (35) + 43 owner-side state values.
        torques = np.asarray(
            info.get(
                "torques",
                np.zeros((n, self._num_native_actuators)),
            ),
            dtype=self._np_dtype,
        )
        extra = np.concatenate(
            (
                np.asarray(linvel, dtype=self._np_dtype),
                np.asarray(dof_pos, dtype=self._np_dtype),
                np.asarray(dof_vel, dtype=self._np_dtype),
                torques[:, self._native_leg_indices],
                np.asarray(heights).reshape(n, 1),
                projected_gravity,
                commands,
            ),
            axis=1,
        )
        if extra.shape[1] < PRIVILEGED_OBS_DIM - POLICY_OBS_DIM:
            extra = np.pad(
                extra,
                ((0, 0), (0, PRIVILEGED_OBS_DIM - POLICY_OBS_DIM - extra.shape[1])),
            )
        elif extra.shape[1] > PRIVILEGED_OBS_DIM - POLICY_OBS_DIM:
            extra = extra[:, : PRIVILEGED_OBS_DIM - POLICY_OBS_DIM]
        critic = np.concatenate((obs, extra), axis=1).astype(get_global_dtype(), copy=False)
        return {"obs": obs, "critic": critic}

    def _compute_reward(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        if self._source_semantics:
            base_quat = np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype)
            _roll, pitch = np_roll_pitch_from_quat(base_quat)
            native_torque = np.asarray(
                info.get(
                    "torques",
                    np.zeros(
                        (self._num_envs, self._num_native_actuators),
                        dtype=self._np_dtype,
                    ),
                ),
                dtype=self._np_dtype,
            )
            policy_torque = np.concatenate(
                (
                    native_torque[:, self._native_leg_indices],
                    native_torque[:, self._native_wheel_indices],
                ),
                axis=1,
            )
            params = SourceV14RewardParameters(
                orientation_x_bias=float(self._reward_cfg.orientation_x_bias),
                orientation_x_sigma=float(self._reward_cfg.orientation_x_sigma),
                orientation_x_amplitude=float(self._reward_cfg.orientation_x_amplitude),
                orientation_y_bias=float(self._reward_cfg.orientation_y_bias),
                orientation_y_sigma=float(self._reward_cfg.orientation_y_sigma),
                orientation_y_amplitude=float(self._reward_cfg.orientation_y_amplitude),
                orientation_x_square_sigma=float(self._reward_cfg.orientation_x_square_sigma),
                orientation_y_square_sigma=float(self._reward_cfg.orientation_y_square_sigma),
                orientation_x_exp_sigma=float(self._reward_cfg.orientation_x_exp_sigma),
                orientation_y_exp_sigma=float(self._reward_cfg.orientation_y_exp_sigma),
                lin_vel_sigma=float(self._reward_cfg.lin_vel_sigma),
                lin_vel_tight_sigma=float(self._reward_cfg.lin_vel_tight_sigma),
                lin_vel_square_sigma=float(self._reward_cfg.lin_vel_square_sigma),
                lin_vel_error_constraint=float(self._reward_cfg.lin_vel_error_constraint),
                ang_vel_sigma=float(self._reward_cfg.ang_vel_sigma),
                ang_vel_square_sigma=float(self._reward_cfg.ang_vel_square_sigma),
                ang_vel_error_constraint=float(self._reward_cfg.ang_vel_error_constraint),
                height_sigma=float(self._reward_cfg.height_sigma),
                height_soft_sigma=float(self._reward_cfg.height_soft_sigma),
                height_tight_sigma=float(self._reward_cfg.height_tight_sigma),
                height_square_sigma=float(self._reward_cfg.height_square_sigma),
                height_error_constraint=float(self._reward_cfg.height_error_constraint),
                no_fork_distance=float(self._reward_cfg.no_fork_distance),
                no_fork_square_sigma=float(self._reward_cfg.no_fork_square_sigma),
                no_fork_exp_sigma=float(self._reward_cfg.no_fork_exp_sigma),
                no_fork_z_distance=float(self._reward_cfg.no_fork_z_distance),
                no_fork_z_exp_sigma=float(self._reward_cfg.no_fork_z_exp_sigma),
                stand_still_deadzone=float(self._reward_cfg.stand_still_deadzone),
                stand_still_deadzone_enabled=bool(self._reward_cfg.stand_still_deadzone_enabled),
                vel_height_gate_enabled=bool(self._cfg.vel_height_gate_enabled),
                vel_height_gate_mode=str(self._cfg.vel_height_gate_mode),
                vel_height_gate_full_error=float(self._cfg.vel_height_gate_full_error),
                vel_height_gate_zero_error=float(self._cfg.vel_height_gate_zero_error),
                vel_height_gate_tracker_sigma=float(self._cfg.vel_height_gate_tracker_sigma),
            )
            reward, terms = compute_source_v14_reward(
                scales=self._source_runtime_reward_scales,
                ctrl_dt=float(self._cfg.ctrl_dt),
                commands=np.asarray(info["commands"], dtype=self._np_dtype),
                linvel=linvel,
                gyro=gyro,
                projected_gravity=projected_gravity,
                pitch=pitch,
                observed_height=np.asarray(
                    info.get("relative_observed_height", info["observed_height"]),
                    dtype=self._np_dtype,
                ),
                height_command=np.asarray(info["height_commands"], dtype=self._np_dtype),
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                qacc=np.asarray(info["qacc"], dtype=self._np_dtype),
                torque=policy_torque,
                actions=np.asarray(info["current_actions"], dtype=self._np_dtype),
                last_actions=np.asarray(info["last_actions"], dtype=self._np_dtype),
                previous_actions=np.asarray(info["previous_actions"], dtype=self._np_dtype),
                wheel_pos_b=np.asarray(info["wheel_pos_b"], dtype=self._np_dtype),
                wheel_contact_state=np.asarray(info["wheel_contact_state"], dtype=bool),
                undesired_contact=np.asarray(info["undesired_contact"], dtype=bool),
                terminated=np.asarray(info["terminated"], dtype=bool),
                params=params,
            )
            numerical_failure = np.asarray(info.get("numerical_safety_failure", False), dtype=bool)
            if np.any(numerical_failure):
                reward[numerical_failure] = 0.0
                for value in terms.values():
                    value[numerical_failure] = 0.0
            if self._source_him_curriculum is not None:
                reward_key = self._source_him_curriculum.reward_key
                tracked = terms.get(reward_key)
                if tracked is not None:
                    self._source_him_track_height_episode_sum += np.asarray(
                        tracked, dtype=np.float32
                    )
                    self._source_him_reward_buffer_initialized = True
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            if self._enable_reward_log and int(steps[0]) % 4 == 0:
                log = info.setdefault("log", {})
                for name, value in terms.items():
                    if float(self._source_runtime_reward_scales.get(name, 0.0)) != 0.0:
                        log[f"reward/{name}"] = float(np.mean(value))
            if bool(self._cfg.debug_value_diagnosis):
                # Pinned ``_debug_print_reward_and_state_stats`` increments
                # this counter only when value diagnosis is enabled.  The raw
                # observation termination gate is intentionally coupled to it.
                self._source_value_debug_step += 1
            return reward
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        base_height = base_pos[:, 2]
        if self._terrain_surface_sample_height is not None:
            surface = np.asarray(self._terrain_surface_sample_height(base_pos[:, :2]))
            base_height = base_height - surface
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            gravity=np.asarray(projected_gravity, dtype=self._np_dtype),
            num_envs=linvel.shape[0],
            default_angles=self.default_angles,
            tracking_sigma=float(self._reward_cfg.tracking_sigma),
            base_height_target=float(self._reward_cfg.base_height_target),
            base_height=base_height,
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=float(self._cfg.ctrl_dt),
            only_positive=bool(self._reward_cfg.only_positive_rewards),
        )

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "orientation": rewards.orientation,
            "action_rate": rewards.action_rate,
            "action_smooth": rewards.action_smooth,
            "alive": rewards.alive,
            "upright": rewards.upright,
            "joint_torques_l2": self._reward_joint_torques_l2,
            "wheel_vel": self._reward_wheel_vel,
            "wheel_power": self._reward_wheel_power,
            "joint_acc_l2": self._reward_joint_acc_l2,
            "stand_still": self._reward_stand_still,
        }

    def _reward_joint_torques_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(ctx.info.get("torques"), dtype=self._np_dtype)
        return np.sum(np.square(torques[:, self._native_leg_indices]), axis=1)

    def _reward_wheel_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.sum(np.square(ctx.dof_vel[:, NUM_LEG_ACTIONS:]), axis=1)

    def _reward_wheel_power(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        torques = np.asarray(ctx.info.get("torques"), dtype=self._np_dtype)
        wheel_torque = torques[:, self._native_wheel_indices]
        return np.sum(np.abs(ctx.dof_vel[:, NUM_LEG_ACTIONS:]) * np.abs(wheel_torque), axis=1)

    def _reward_joint_acc_l2(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(ctx.info.get("qacc"), dtype=self._np_dtype)
        return np.sum(np.square(qacc[:, :NUM_LEG_ACTIONS]), axis=1)

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        stopped = np.linalg.norm(ctx.info["commands"][:, :2], axis=1) < 0.1
        return np.sum(np.abs(ctx.dof_pos - self.default_angles), axis=1) * stopped

    def _apply_source_reset_command_envelopes(
        self,
        commands: np.ndarray,
        info: dict[str, Any],
        *,
        resampled: np.ndarray,
    ) -> None:
        """Apply the source predefined ground/air command modifiers in-place."""

        ground_remaining = self._source_ground_command_steps_remaining
        ground_pending = ground_remaining > 0
        refreshed = ground_pending & np.asarray(resampled, dtype=bool)
        if np.any(refreshed):
            # This matters only for compatibility profiles whose command
            # period is shorter than the canonical 1.5-second modifier.
            # Restore the newest underlying command when the modifier ends.
            self._source_ground_restore_command[refreshed] = commands[refreshed]
        if np.any(ground_pending):
            ground_remaining[ground_pending] -= 1
            ground_active = ground_remaining > 0
            ground_expired = ground_pending & ~ground_active
            commands[ground_active] = self._source_ground_override_command[ground_active]
            commands[ground_expired] = self._source_ground_restore_command[ground_expired]

        air_remaining = self._source_air_command_steps_remaining
        air_pending = air_remaining > 0
        if not np.any(air_pending):
            return
        air_remaining[air_pending] -= 1
        air_active = air_remaining > 0
        if not np.any(air_active):
            return
        commands[air_active] = np.clip(
            commands[air_active],
            self._source_air_command_low,
            self._source_air_command_high,
        )
        heights = info.get("height_commands")
        if heights is not None:
            height_commands = np.asarray(heights, dtype=self._np_dtype).copy()
            height_commands[air_active] = np.clip(
                height_commands[air_active],
                self._source_air_height_low,
                self._source_air_height_high,
            )
            info["height_commands"] = height_commands

    def _update_commands(self, info: dict[str, Any]) -> None:
        commands = info.get("commands")
        if commands is None:
            return
        if self._source_semantics:
            commands_arr = np.asarray(commands, dtype=self._np_dtype)
            remaining = np.asarray(
                info.get(
                    "command_resample_steps_remaining",
                    np.ones((self._num_envs,), dtype=np.int32),
                ),
                dtype=np.int32,
            )
            generation = np.asarray(
                info.get(
                    "command_resample_generation",
                    np.zeros((self._num_envs,), dtype=np.int64),
                ),
                dtype=np.int64,
            )
            remaining -= 1
            due = remaining <= 0
            if np.any(due):
                ids = np.flatnonzero(due)
                base_yaw = np_yaw_from_quat(
                    np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype)
                )
                episode_steps = np.asarray(
                    info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
                )
                sampled = self._sample_source_commands(
                    current_yaw=base_yaw[ids], episode_steps=episode_steps[ids]
                )
                commands_arr[ids] = sampled["commands"]
                for key in (
                    "heading_commands",
                    "is_standing_env",
                    "is_heading_env",
                    "special_mode_id",
                ):
                    current = info.get(key)
                    if current is None:
                        value = np.asarray(sampled[key])
                        current = np.zeros((self._num_envs, *value.shape[1:]), dtype=value.dtype)
                        info[key] = current
                    np.asarray(current)[ids] = sampled[key]
                low, high = _sample_range(
                    self._cfg.commands.resampling_time_range,
                    name="commands.resampling_time_range",
                )
                duration = np.random.uniform(low, high, size=ids.size)
                remaining[ids] = np.maximum(
                    np.ceil(duration / float(self._cfg.ctrl_dt)).astype(np.int32), 1
                )
                generation[ids] += 1

            heading_mask = (
                np.asarray(info.get("is_heading_env", False), dtype=bool)
                & ~np.asarray(info.get("is_standing_env", False), dtype=bool)
                & (np.asarray(info.get("special_mode_id", -1)) < 0)
            )
            if np.any(heading_mask):
                yaw = np_yaw_from_quat(
                    np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype)
                )
                target = np.asarray(info["heading_commands"], dtype=self._np_dtype)
                yaw_error = np_wrap_to_pi(target - yaw)
                yaw_low, yaw_high = sorted(
                    (
                        float(self._cfg.commands.vel_limit[0][2]),
                        float(self._cfg.commands.vel_limit[1][2]),
                    )
                )
                commands_arr[heading_mask, 2] = np.clip(
                    float(self._cfg.commands.heading_control_stiffness) * yaw_error[heading_mask],
                    yaw_low,
                    yaw_high,
                )
            standing = np.asarray(info.get("is_standing_env", False), dtype=bool)
            commands_arr[standing] = 0.0
            commands_arr[:, 1] = 0.0
            self._apply_source_reset_command_envelopes(
                commands_arr,
                info,
                resampled=due,
            )
            info["commands"] = commands_arr
            info["command_resample_steps_remaining"] = remaining
            info["command_resample_generation"] = generation
            return
        commands_arr = np.asarray(commands, dtype=self._np_dtype)
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / float(self._cfg.ctrl_dt))), 1)
            steps = np.asarray(info.get("steps", np.zeros(self._num_envs, dtype=np.uint32)))
            mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(mask):
                count = int(np.count_nonzero(mask))
                low = np.asarray(self._cfg.commands.vel_limit[0], dtype=self._np_dtype)
                high = np.asarray(self._cfg.commands.vel_limit[1], dtype=self._np_dtype)
                sampled = np.random.uniform(low, high, size=(count, 3)).astype(self._np_dtype)
                zero_small_xy_commands(sampled, threshold=0.08)
                sampled[:, 1] = 0.0
                commands_arr[mask] = sampled
                if getattr(self._cfg.commands, "heading_command", False):
                    info["heading_commands"] = sample_heading_commands(self, self._num_envs)
        if getattr(self._cfg.commands, "heading_command", False):
            heading = np.asarray(
                info.get("heading_commands", sample_heading_commands(self, self._num_envs)),
                dtype=self._np_dtype,
            )
            apply_heading_yaw_feedback(
                commands_arr,
                np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype),
                heading,
                stiffness=float(getattr(self._cfg.commands, "heading_control_stiffness", 0.5)),
            )
        info["commands"] = commands_arr

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        return super()._compute_truncated(state)


# Stable aliases make the owner discoverable alongside the existing
# ``*FlatEnv`` naming used by other locomotion tasks, without registering a
# second environment name or duplicating backend classes.
WheelbipeV14FlatEnv = WheelbipeV14Env
WheelbipeEnv = WheelbipeV14Env
# Keep the upstream Isaac task's public configuration spelling available for
# callers migrating imports.  This is an alias only: registry identity stays
# ``WheelbipeV14Flat`` and no second config is registered.
WheelbipeV14FlatEnvCfg = WheelbipeV14FlatCfg


__all__ = [
    "WheelbipeCommands",
    "WheelbipeControlConfig",
    "WheelbipeGimbalConfig",
    "WheelbipeDomainRandConfig",
    "WheelbipeInitState",
    "WheelbipeNoiseConfig",
    "WheelbipeRewardConfig",
    "WheelbipeV14Env",
    "WheelbipeV14FlatEnv",
    "WheelbipeEnv",
    "WheelbipeV14FlatCfg",
    "WheelbipeV14FlatEnvCfg",
    "WheelbipeV14DomainRandomizationProvider",
    "build_wheelbipe_backend_reset_randomization",
]
