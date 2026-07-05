from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    validate_common_reset_randomization,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_mul,
    np_wrap_to_pi,
    np_yaw_from_quat,
    np_yaw_to_quat,
)
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands, zero_small_xy_commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.real68.base import (
    ACTIVE_JOINT_POS_SENSORS,
    CALF_INDICES,
    DEFAULT_ACTIVE_ANGLES,
    HIP_INDICES,
    HOME_BASE_HEIGHT,
    NONWHEEL_CONTACT_SENSORS,
    NUM_ACTIONS,
    POSTURE_INDICES,
    SYMMETRIC_STANDING_ACTIVE_ANGLES,
    WHEEL_CONTACT_SENSORS,
    WHEEL_INDICES,
    Real68BaseCfg,
    Real68BaseEnv,
    compute_real68_motor_ctrl,
    scalarize_contacts,
)

_REAL68_LEFT_POSTURE = np.asarray([0, 1], dtype=np.int32)
_REAL68_RIGHT_POSTURE = np.asarray([2, 3], dtype=np.int32)
_REAL68_MIRROR_SIGNS = np.asarray([-1.0, -1.0], dtype=np.float64)
_REAL68_CURRICULUM_MIN_ABS_COMMAND = 0.05
_REAL68_CURRICULUM_NUM_BINS = 4
_REAL68_LEFT_HIP_INDEX = int(HIP_INDICES[0])
_REAL68_RIGHT_HIP_INDEX = int(HIP_INDICES[1])
_REAL68_LEFT_WHEEL_INDEX = int(WHEEL_INDICES[0])
_REAL68_RIGHT_WHEEL_INDEX = int(WHEEL_INDICES[1])
_REAL68_LEFT_CALF_INDEX = int(CALF_INDICES[0])
_REAL68_RIGHT_CALF_INDEX = int(CALF_INDICES[1])
_REAL68_LEFT_LIANGAN5_CONTACT_INDEX = NONWHEEL_CONTACT_SENSORS.index("left_chuanliangan5_contact")
_REAL68_RIGHT_LIANGAN5_CONTACT_INDEX = NONWHEEL_CONTACT_SENSORS.index("right_liangan5_contact")
_REAL68_FORWARD_AXIS = 1
_REAL68_LATERAL_AXIS = 0
_REAL68_FORWARD_SIGN = 1.0
_REAL68_WHEEL_RADIUS = 0.06
# Left/right wheel bodies sit at y = ±0.215 in real68.xml (lines 54, 110),
# so the differential-drive wheelbase is 2 * 0.215 = 0.43 m. Used to clip the
# commanded (vx, wz) into the reachable diamond |vx| + |wz| * L/2 <= v_wheel_max.
_REAL68_WHEEL_BASE = 0.43


@dataclass
class Real68CommandCurriculumCfg:
    enabled: bool = False
    initial_vel_limit: list[list[float]] = field(
        default_factory=lambda: [[0.1, 0.0, -0.3], [0.35, 0.0, 0.3]]
    )
    final_vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-2.0, 0.0, -0.8], [2.0, 0.0, 0.8]]
    )
    vx_step: float = 0.03
    vx_step_down: float = 0.03
    yaw_step: float = 0.02
    yaw_step_down: float = 0.02
    update_interval_logs: int = 6
    err_mode: bool = False
    min_speed_ratio: float = 0.45
    min_speed_ratio_down: float = 0.2
    max_vx_error: float = 0.25
    vx_error_range: list[float] = field(default_factory=lambda: [0.25, 0.35, 0.6])
    max_wz_error: float = 0.9
    max_wz_error_high: float = 1.1
    max_tilt_rate: float = 0.02
    max_tilt_rate_high: float = 0.05
    max_tilt_angle_deg: float = float("inf")
    max_tilt_angle_deg_high: float = float("inf")
    max_height_violation_rate: float = 0.01
    max_height_violation_rate_high: float = 0.03
    max_nonwheel_contact_rate: float = 0.02
    max_nonwheel_contact_rate_high: float = 0.06
    min_segment_count: int = 64
    yaw_unlock_vx_progress: float = 0.6
    reverse_unlock_vx_progress: float = 0.7
    standing_bootstrap_enabled: bool = False
    standing_bootstrap_min_segments: int = 64
    standing_bootstrap_min_segment_steps: int = 80
    standing_bootstrap_max_abs_vx: float = 0.08
    standing_bootstrap_max_wz_error: float = 0.35
    standing_bootstrap_max_nonwheel_contact: float = 0.02
    standing_prob_initial: float = 0.0
    standing_prob_final: float = 0.0
    standing_decay_vx_progress: float = 1.0
    straight_command_prob: float = 0.0
    yaw_only_command_prob: float = 0.0
    yaw_only_max_abs_vx: float = 0.04
    high_speed_command_prob: float = 0.0
    high_speed_min_abs_vx: float = 0.0


@dataclass
class HeightCommandConfig:
    range: list[float] = field(default_factory=lambda: [0.25, 0.32])
    speed_lift_gain: float = 0.0
    max_speed_lift: float = 0.0


@dataclass
class Real68Commands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.5, 0.0, -0.8], [0.8, 0.0, 0.8]]
    )
    resampling_time: float = 2.0
    rel_standing_envs: float = 0.1


@dataclass
class InitState:
    pos = [0.0, 0.0, HOME_BASE_HEIGHT]


@dataclass
class RewardConfig:
    @dataclass
    class CommandLeanConfig:
        enabled: bool = False
        gravity_x_gain: float = 0.0
        gravity_x_limit: float = 0.0
        hip_gain: float = 0.0
        hip_limit: float = 0.0
        calf_gain: float = 0.0
        calf_limit: float = 0.0

    @dataclass
    class BalanceGateConfig:
        enabled: bool = False
        upright_cos_min: float = 0.75
        upright_cos_full: float = 0.96
        straight_command_yaw_threshold: float = 0.2
        moving_command_vx_threshold: float = 0.08
        heading_error_limit: float = 0.35
        lateral_drift_limit: float = 0.35
        yaw_tracking_sigma: float = 1.5
        standing_roll_pitch_sigma: float = 0.10

    @dataclass
    class ExcessTiltConfig:
        limit_deg: float = 3.0
        scale_deg: float = 3.0

    scales: dict[str, float]
    tracking_sigma: float
    height_tracking_sigma: float = 0.015
    height_safety_margin: float = 0.03
    base_height_target: float = HOME_BASE_HEIGHT
    min_base_height: float = 0.18
    max_base_height: float = 0.38
    max_tilt_cos: float = 0.5
    only_positive_rewards: bool = False
    command_lean: CommandLeanConfig = field(default_factory=CommandLeanConfig)
    balance_gate: BalanceGateConfig = field(default_factory=BalanceGateConfig)
    excess_tilt: ExcessTiltConfig = field(default_factory=ExcessTiltConfig)


@dataclass
class Real68Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    gravity = "upvector"
    accel = "imu_accel"
    quat = "imu_quat"


@dataclass
class Real68DomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    reset_qvel_limit: float = 0.2


@dataclass
class FlatTerminationConfig:
    fall_termination: bool = True
    nonwheel_contact_termination: bool = True
    nonwheel_contact_threshold: float = 0.5
    nonwheel_contact_max_steps: int = 8


@registry.envcfg("Real68BalanceFlat")
@dataclass
class Real68BalanceCfg(Real68BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 15.0
    init_state: InitState = field(default_factory=InitState)
    commands: Real68Commands = field(default_factory=Real68Commands)
    height_command: HeightCommandConfig = field(default_factory=HeightCommandConfig)
    command_curriculum: Real68CommandCurriculumCfg = field(
        default_factory=Real68CommandCurriculumCfg
    )
    reward_config: RewardConfig | None = None
    sensor: Real68Sensor = field(default_factory=Real68Sensor)
    domain_rand: Real68DomainRandConfig = field(default_factory=Real68DomainRandConfig)
    termination_config: FlatTerminationConfig = field(default_factory=FlatTerminationConfig)


class Real68BalanceDomainRandomizationProvider(LocomotionDRProvider):
    def validate(self, env: Any, capabilities) -> None:
        validate_common_reset_randomization(
            env,
            capabilities,
            base_geom_friction=getattr(env, "_base_geom_friction", None),
            ground_geom_id=getattr(env, "_ground_geom_id", None),
        )
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        return build_interval_push_plan(env, step_counter)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.25, 0.25, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        if env.cfg.domain_rand.randomize_init_yaw:
            low, high = env.cfg.domain_rand.init_yaw_range
            yaw = np.random.uniform(low, high, size=(num_reset,))
            qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        limit = float(env.cfg.domain_rand.reset_qvel_limit)
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-limit, limit, size=(num_reset, 6)),
            dtype=get_global_dtype(),
        )
        commands = env.sample_velocity_commands(num_reset)
        if (
            hasattr(env, "_last_command_clip_scale")
            and env._last_command_clip_scale.shape[0] == num_reset
        ):
            env._command_clip_scale[env_ids] = env._last_command_clip_scale
        height_commands = env.sample_height_commands(num_reset, commands=commands)
        info_updates = {
            "commands": commands,
            "height_commands": height_commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "qacc": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "wheel_contacts": np.zeros(
                (num_reset, len(WHEEL_CONTACT_SENSORS)), dtype=get_global_dtype()
            ),
            "nonwheel_contacts": np.zeros(
                (num_reset, len(NONWHEEL_CONTACT_SENSORS)), dtype=get_global_dtype()
            ),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(
                env,
                num_reset,
                base_geom_friction=getattr(env, "_base_geom_friction", None),
                ground_geom_id=getattr(env, "_ground_geom_id", None),
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
        accel = env.get_accel()[env_ids]
        reset_info = dict(info_updates)
        reset_info["wheel_contacts"] = scalarize_contacts(
            env._backend, WHEEL_CONTACT_SENSORS, dtype=get_global_dtype()
        )[env_ids]
        reset_info["nonwheel_contacts"] = scalarize_contacts(
            env._backend, NONWHEEL_CONTACT_SENSORS, dtype=get_global_dtype()
        )[env_ids]
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                reset_info,
                linvel,
                gyro,
                gravity,
                accel,
                dof_pos,
                dof_vel,
            ),
        )


@registry.env("Real68BalanceFlat", sim_backend="mujoco")
class Real68BalanceEnv(Real68BaseEnv):
    _cfg: Real68BalanceCfg

    def __init__(self, cfg: Real68BalanceCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        if ctrl_range.shape != (NUM_ACTIONS, 2):
            raise ValueError(f"Real68 actuator ctrl_range must have shape ({NUM_ACTIONS}, 2)")
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)
        self._ground_geom_id = self._backend.get_geom_id(self._cfg.asset.ground)
        self._base_geom_friction = np.asarray(
            self._backend.get_geom_friction(), dtype=np.float64
        ).copy()
        self._reward_cfg = cfg.reward_config
        self._last_motor_ctrl = np.zeros((num_envs, NUM_ACTIONS), dtype=self._np_dtype)
        self._last_dof_vel_for_acc = np.zeros((num_envs, NUM_ACTIONS), dtype=self._np_dtype)
        self._base_height = np.full((num_envs,), HOME_BASE_HEIGHT, dtype=self._np_dtype)
        self._wheel_contacts = np.zeros(
            (num_envs, len(WHEEL_CONTACT_SENSORS)), dtype=self._np_dtype
        )
        self._nonwheel_contacts = np.zeros(
            (num_envs, len(NONWHEEL_CONTACT_SENSORS)), dtype=self._np_dtype
        )
        self._segment_start_pos_xy = np.zeros((num_envs, 2), dtype=self._np_dtype)
        self._segment_start_yaw = np.zeros((num_envs,), dtype=self._np_dtype)
        self._heading_error = np.zeros((num_envs,), dtype=self._np_dtype)
        self._lateral_drift = np.zeros((num_envs,), dtype=self._np_dtype)
        # Per-env clip scale applied by _clip_to_diamond at command resample
        # time. 1.0 = command was inside the reachable diamond (unchanged);
        # < 1.0 = the (vx, wz) ray was scaled back to the diamond boundary.
        # Read by the metrics writer to expose how often the nominal command
        # box overshoots the differential-drive reachable set.
        self._command_clip_scale = np.ones((num_envs,), dtype=self._np_dtype)
        # Staging buffer written by _clip_to_diamond; sampled by callers that
        # know the resampled env_ids to scatter into _command_clip_scale.
        self._last_command_clip_scale = np.ones((num_envs,), dtype=self._np_dtype)
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        self._init_reward_functions()
        self._init_domain_randomization(self._make_dr_provider())
        self._command_curriculum_vx_progress = 0.0
        self._command_curriculum_yaw_progress = 0.0
        self._command_curriculum_log_count = 0
        ccfg = self._cfg.command_curriculum
        self._standing_bootstrap_complete = not bool(
            ccfg.enabled and ccfg.standing_bootstrap_enabled
        )
        self._command_curriculum_low = np.asarray(
            ccfg.initial_vel_limit[0] if ccfg.enabled else self._cfg.commands.vel_limit[0],
            dtype=self._np_dtype,
        )
        self._command_curriculum_high = np.asarray(
            ccfg.initial_vel_limit[1] if ccfg.enabled else self._cfg.commands.vel_limit[1],
            dtype=self._np_dtype,
        )
        self._curriculum_vx_count = np.zeros((_REAL68_CURRICULUM_NUM_BINS,), dtype=np.int32)
        self._curriculum_vx_signed_speed_ratio_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_vx_error_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_vx_tilt_rate_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_vx_tilt_angle_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_vx_height_violation_rate_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_vx_nonwheel_contact_rate_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_yaw_count = np.zeros((_REAL68_CURRICULUM_NUM_BINS,), dtype=np.int32)
        self._curriculum_yaw_error_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_yaw_tilt_rate_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_yaw_tilt_angle_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_yaw_height_violation_rate_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_yaw_nonwheel_contact_rate_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._nonwheel_contact_steps = np.zeros((num_envs,), dtype=np.int32)
        self._standing_segment_count = 0
        self._standing_segment_steps_sum = 0.0
        self._standing_segment_abs_vx_sum = 0.0
        self._standing_segment_wz_error_sum = 0.0
        self._standing_segment_nonwheel_contact_sum = 0.0
        self._last_standing_bootstrap_eval: dict[str, float] | None = None
        self._segment_cmd_x = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_cmd_yaw = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_abs_vx_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_signed_vx_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_abs_wz_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_vx_error_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_wz_error_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_nonwheel_contact_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_tilt_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_tilt_angle_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_height_violation_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_steps = np.zeros((num_envs,), dtype=np.int32)
        self._curriculum_recorded_segments = 0

    def _make_dr_provider(self) -> Real68BalanceDomainRandomizationProvider:
        return Real68BalanceDomainRandomizationProvider()

    def _forward_linvel(self, linvel: np.ndarray) -> np.ndarray:
        return np.asarray(
            linvel[:, _REAL68_FORWARD_AXIS] * _REAL68_FORWARD_SIGN,
            dtype=self._np_dtype,
        )

    def _lateral_linvel(self, linvel: np.ndarray) -> np.ndarray:
        return np.asarray(linvel[:, _REAL68_LATERAL_AXIS], dtype=self._np_dtype)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 32, "critic": 45}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
        if dof_vel.shape[0] == self._num_envs:
            self._last_dof_vel_for_acc[env_ids] = dof_vel[env_ids]
        commands = np.asarray(
            info.get("commands", np.zeros((len(env_ids), 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        self._nonwheel_contact_steps[env_ids] = 0
        self._reset_command_segments(env_ids, commands)
        return obs, info

    def _command_height_lift(self, cmd_x: np.ndarray) -> np.ndarray:
        hcfg = self._cfg.height_command
        gain = max(float(hcfg.speed_lift_gain), 0.0)
        max_lift = max(float(hcfg.max_speed_lift), 0.0)
        if gain <= 0.0 or max_lift <= 0.0:
            return np.zeros_like(cmd_x, dtype=self._np_dtype)
        lift = np.abs(np.asarray(cmd_x, dtype=self._np_dtype)) * gain
        return np.asarray(np.clip(lift, 0.0, max_lift), dtype=self._np_dtype)

    def sample_height_commands(
        self,
        num_reset: int,
        *,
        commands: np.ndarray | None = None,
    ) -> np.ndarray:
        low, high = self._cfg.height_command.range
        base = np.asarray(np.random.uniform(low, high, size=(num_reset,)), dtype=self._np_dtype)
        if commands is None:
            return base
        lift = self._command_height_lift(np.asarray(commands[:, 0], dtype=self._np_dtype))
        return np.asarray(base + lift, dtype=self._np_dtype)

    def sample_velocity_commands(self, num_samples: int) -> np.ndarray:
        if not self._standing_bootstrap_complete:
            return np.zeros((num_samples, 3), dtype=self._np_dtype)
        low = self._command_curriculum_low
        high = self._command_curriculum_high
        commands = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_samples, 3)),
            dtype=self._np_dtype,
        )
        commands[:, 1] = 0.0
        yaw_only_mask = np.zeros((num_samples,), dtype=bool)
        if self._yaw_curriculum_locked():
            commands[:, 2] = 0.0
        else:
            straight_prob, yaw_only_prob = self._command_mix_probabilities()
            if straight_prob > 0.0 or yaw_only_prob > 0.0:
                mix = np.random.uniform(size=(num_samples,))
                straight_mask = mix < straight_prob
                yaw_only_mask = (mix >= straight_prob) & (mix < straight_prob + yaw_only_prob)
                commands[straight_mask, 2] = 0.0
                if np.any(yaw_only_mask):
                    max_abs_vx = max(float(self._cfg.command_curriculum.yaw_only_max_abs_vx), 0.0)
                    commands[yaw_only_mask, 0] = np.random.uniform(
                        low=-max_abs_vx,
                        high=max_abs_vx,
                        size=(int(np.count_nonzero(yaw_only_mask)),),
                    )
        self._apply_high_speed_command_bias(commands, eligible_mask=~yaw_only_mask)
        zero_small_xy_commands(commands, threshold=0.08)
        self._clip_to_diamond(commands)
        standing_prob = self._standing_command_probability()
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_samples,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        return commands

    def _clip_to_diamond(self, commands: np.ndarray) -> None:
        """Clip commanded (vx, wz) into the differential-drive reachable set, in place.

        The reachable set is the diamond ``|vx| + |wz| * L/2 <= v_wheel_max`` where
        ``v_wheel_max = wheel_velocity_scale * wheel_radius`` is the per-wheel surface
        speed ceiling. Scaling both vx and wz by the same factor keeps the command on
        its original ray (preserving direction and the relative vx/wz mix) while
        pulling it back to the diamond boundary.

        Also stages per-sample clip scales in ``self._last_command_clip_scale`` (1.0 =
        unchanged, <1.0 = scaled back) so callers that know the resampled env_ids can
        scatter them into ``self._command_clip_scale`` for diagnostics.
        """
        wheel_velocity_scale = float(self._cfg.control_config.wheel_velocity_scale)
        v_wheel_max = wheel_velocity_scale * _REAL68_WHEEL_RADIUS
        demand = np.abs(commands[:, 0]) + np.abs(commands[:, 2]) * (0.5 * _REAL68_WHEEL_BASE)
        clip_scale = np.ones_like(demand, dtype=self._np_dtype)
        over = demand > v_wheel_max
        if np.any(over):
            scale = v_wheel_max / np.maximum(demand[over], 1.0e-6)
            clip_scale[over] = scale
            commands[over, 0] *= scale
            commands[over, 2] *= scale
        self._last_command_clip_scale = clip_scale

    def _apply_high_speed_command_bias(
        self,
        commands: np.ndarray,
        *,
        eligible_mask: np.ndarray,
    ) -> None:
        ccfg = self._cfg.command_curriculum
        prob = float(np.clip(ccfg.high_speed_command_prob, 0.0, 1.0))
        min_abs_vx = max(float(ccfg.high_speed_min_abs_vx), 0.0)
        if prob <= 0.0 or min_abs_vx <= 0.0 or commands.shape[0] == 0:
            return

        low_vx = float(self._command_curriculum_low[0])
        high_vx = float(self._command_curriculum_high[0])
        can_sample_negative = low_vx <= -min_abs_vx
        can_sample_positive = high_vx >= min_abs_vx
        if not (can_sample_negative or can_sample_positive):
            return

        eligible = np.asarray(eligible_mask, dtype=bool)
        selected = eligible & (np.random.uniform(size=(commands.shape[0],)) < prob)
        selected_count = int(np.count_nonzero(selected))
        if selected_count == 0:
            return

        choose_negative = np.zeros((selected_count,), dtype=bool)
        if can_sample_negative and can_sample_positive:
            choose_negative = np.random.uniform(size=(selected_count,)) < 0.5
        elif can_sample_negative:
            choose_negative.fill(True)

        sampled = np.empty((selected_count,), dtype=self._np_dtype)
        negative_count = int(np.count_nonzero(choose_negative))
        if negative_count > 0:
            sampled[choose_negative] = np.random.uniform(
                low=low_vx,
                high=-min_abs_vx,
                size=(negative_count,),
            )
        positive_count = selected_count - negative_count
        if positive_count > 0:
            sampled[~choose_negative] = np.random.uniform(
                low=min_abs_vx,
                high=high_vx,
                size=(positive_count,),
            )
        commands[selected, 0] = sampled

    def _command_mix_probabilities(self) -> tuple[float, float]:
        ccfg = self._cfg.command_curriculum
        if not ccfg.enabled:
            return 0.0, 0.0
        straight_prob = float(np.clip(ccfg.straight_command_prob, 0.0, 1.0))
        yaw_only_prob = float(np.clip(ccfg.yaw_only_command_prob, 0.0, 1.0 - straight_prob))
        return straight_prob, yaw_only_prob

    def _yaw_curriculum_locked(self) -> bool:
        ccfg = self._cfg.command_curriculum
        return bool(
            ccfg.enabled
            and self._command_curriculum_vx_progress < float(ccfg.yaw_unlock_vx_progress)
        )

    def _standing_command_probability(self) -> float:
        ccfg = self._cfg.command_curriculum
        if not ccfg.enabled:
            return float(getattr(self._cfg.commands, "rel_standing_envs", 0.0))
        initial = float(ccfg.standing_prob_initial)
        final = float(ccfg.standing_prob_final)
        if ccfg.standing_bootstrap_enabled:
            return final if self._standing_bootstrap_complete else 1.0
        decay = float(ccfg.standing_decay_vx_progress)
        if decay <= 0.0:
            return final
        alpha = float(np.clip(self._command_curriculum_vx_progress / decay, 0.0, 1.0))
        return float((1.0 - alpha) * initial + alpha * final)

    def _update_command_curriculum(self) -> None:
        ccfg = self._cfg.command_curriculum
        if not ccfg.enabled:
            return
        self._command_curriculum_log_count += 1
        interval = max(int(ccfg.update_interval_logs), 1)
        if self._command_curriculum_log_count % interval != 0:
            return
        if not self._standing_bootstrap_complete:
            standing_eval = self._standing_bootstrap_eval()
            standing_ready = bool(
                standing_eval is not None
                and standing_eval["mean_steps"] >= float(ccfg.standing_bootstrap_min_segment_steps)
                and standing_eval["abs_vx"] <= float(ccfg.standing_bootstrap_max_abs_vx)
                and standing_eval["wz_error"] <= float(ccfg.standing_bootstrap_max_wz_error)
                and standing_eval["nonwheel_contact"]
                <= float(ccfg.standing_bootstrap_max_nonwheel_contact)
            )
            if standing_ready:
                self._last_standing_bootstrap_eval = standing_eval
                self._standing_bootstrap_complete = True
                self._reset_command_curriculum_stats()
                self._last_standing_bootstrap_eval = standing_eval
            elif standing_eval is not None:
                self._last_standing_bootstrap_eval = standing_eval
                self._reset_standing_bootstrap_stats()
            return
        vx_eval = self._curriculum_vx_eval()
        progressed = False
        if vx_eval is not None:
            vx_up_threshold = self._vx_curriculum_up_threshold()
            vx_ready = bool(
                vx_eval["speed_ratio"] >= float(ccfg.min_speed_ratio)
                and vx_eval["vx_error"] <= vx_up_threshold
                and vx_eval["tilt_rate"] <= float(ccfg.max_tilt_rate)
                and vx_eval["tilt_angle_deg"] <= float(ccfg.max_tilt_angle_deg)
                and vx_eval["height_violation_rate"] <= float(ccfg.max_height_violation_rate)
                and vx_eval["nonwheel_contact_rate"] <= float(ccfg.max_nonwheel_contact_rate)
            )
            vx_should_backoff = bool(
                vx_eval["speed_ratio"] <= float(ccfg.min_speed_ratio_down)
                or vx_eval["vx_error"] >= float(ccfg.vx_error_range[2])
                or vx_eval["tilt_rate"] >= float(ccfg.max_tilt_rate_high)
                or vx_eval["tilt_angle_deg"] >= float(ccfg.max_tilt_angle_deg_high)
                or vx_eval["height_violation_rate"] >= float(ccfg.max_height_violation_rate_high)
                or vx_eval["nonwheel_contact_rate"] >= float(ccfg.max_nonwheel_contact_rate_high)
            )
            if vx_ready and self._command_curriculum_vx_progress < 1.0:
                self._command_curriculum_vx_progress = min(
                    1.0, self._command_curriculum_vx_progress + float(ccfg.vx_step)
                )
                progressed = True
            elif vx_should_backoff and self._command_curriculum_vx_progress > 0.0:
                self._command_curriculum_vx_progress = max(
                    0.0, self._command_curriculum_vx_progress - float(ccfg.vx_step_down)
                )
                progressed = True
        yaw_eval = self._curriculum_yaw_eval()
        if yaw_eval is not None and self._command_curriculum_vx_progress >= float(
            ccfg.yaw_unlock_vx_progress
        ):
            yaw_ready = bool(
                yaw_eval["wz_error"] <= float(ccfg.max_wz_error)
                and yaw_eval["tilt_rate"] <= float(ccfg.max_tilt_rate)
                and yaw_eval["tilt_angle_deg"] <= float(ccfg.max_tilt_angle_deg)
                and yaw_eval["height_violation_rate"] <= float(ccfg.max_height_violation_rate)
                and yaw_eval["nonwheel_contact_rate"] <= float(ccfg.max_nonwheel_contact_rate)
            )
            yaw_should_backoff = bool(
                yaw_eval["wz_error"] >= float(ccfg.max_wz_error_high)
                or yaw_eval["tilt_rate"] >= float(ccfg.max_tilt_rate_high)
                or yaw_eval["tilt_angle_deg"] >= float(ccfg.max_tilt_angle_deg_high)
                or yaw_eval["height_violation_rate"] >= float(ccfg.max_height_violation_rate_high)
                or yaw_eval["nonwheel_contact_rate"] >= float(ccfg.max_nonwheel_contact_rate_high)
            )
            if yaw_ready and self._command_curriculum_yaw_progress < 1.0:
                self._command_curriculum_yaw_progress = min(
                    1.0, self._command_curriculum_yaw_progress + float(ccfg.yaw_step)
                )
                progressed = True
            elif yaw_should_backoff and self._command_curriculum_yaw_progress > 0.0:
                self._command_curriculum_yaw_progress = max(
                    0.0, self._command_curriculum_yaw_progress - float(ccfg.yaw_step_down)
                )
                progressed = True
        if progressed:
            self._refresh_command_curriculum_limits()
            self._reset_command_curriculum_stats()

    def _vx_curriculum_up_threshold(self) -> float:
        ccfg = self._cfg.command_curriculum
        if bool(ccfg.err_mode):
            return float(ccfg.max_vx_error)
        final_limit = np.asarray(ccfg.final_vel_limit[1], dtype=self._np_dtype)
        final_abs_vx = max(abs(float(final_limit[0])), 1.0e-6)
        current_abs_vx = self._curriculum_abs_limit_x()
        ratio = float(np.clip(current_abs_vx / final_abs_vx, 0.0, 1.0))
        low, high, _ = ccfg.vx_error_range
        return float(low + (high - low) * ratio)

    def _refresh_command_curriculum_limits(self) -> None:
        ccfg = self._cfg.command_curriculum
        initial_low = np.asarray(ccfg.initial_vel_limit[0], dtype=self._np_dtype)
        initial_high = np.asarray(ccfg.initial_vel_limit[1], dtype=self._np_dtype)
        final_low = np.asarray(ccfg.final_vel_limit[0], dtype=self._np_dtype)
        final_high = np.asarray(ccfg.final_vel_limit[1], dtype=self._np_dtype)
        low = initial_low.copy()
        high = initial_high.copy()
        vx_alpha = self._command_curriculum_vx_progress
        yaw_alpha = self._command_curriculum_yaw_progress
        high[0] = (1.0 - vx_alpha) * initial_high[0] + vx_alpha * final_high[0]
        reverse_unlock = float(ccfg.reverse_unlock_vx_progress)
        if reverse_unlock >= 1.0:
            if vx_alpha >= 1.0:
                low[0] = final_low[0]
        elif vx_alpha >= reverse_unlock:
            reverse_alpha = (vx_alpha - reverse_unlock) / max(1.0 - reverse_unlock, 1.0e-6)
            low[0] = (1.0 - reverse_alpha) * initial_low[0] + reverse_alpha * final_low[0]
        low[2] = (1.0 - yaw_alpha) * initial_low[2] + yaw_alpha * final_low[2]
        high[2] = (1.0 - yaw_alpha) * initial_high[2] + yaw_alpha * final_high[2]
        self._command_curriculum_low = low
        self._command_curriculum_high = high

    def get_accel(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.accel), dtype=self._np_dtype
        )

    def get_imu_quat(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.quat), dtype=self._np_dtype
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=self._np_dtype,
        )
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped_actions
        )
        targets = np.zeros_like(exec_actions, dtype=self._np_dtype)
        targets[:, HIP_INDICES] = (
            exec_actions[:, HIP_INDICES] * self._cfg.control_config.hip_velocity_scale
        )
        targets[:, WHEEL_INDICES] = (
            exec_actions[:, WHEEL_INDICES] * self._cfg.control_config.wheel_velocity_scale
        )
        targets[:, CALF_INDICES] = (
            self.default_angles[CALF_INDICES]
            + exec_actions[:, CALF_INDICES] * self._cfg.control_config.calf_action_scale
        )
        return targets

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        active_pos = self.get_dof_pos()
        active_vel = self.get_dof_vel()
        hip_kd = np.full(
            (self._num_envs, len(HIP_INDICES)), self._cfg.control_config.hip_kd, dtype=np.float64
        )
        wheel_kd = np.full(
            (self._num_envs, len(WHEEL_INDICES)),
            self._cfg.control_config.wheel_kd,
            dtype=np.float64,
        )
        calf_kp = np.full(
            (self._num_envs, len(CALF_INDICES)), self._cfg.control_config.calf_kp, dtype=np.float64
        )
        calf_kd = np.full(
            (self._num_envs, len(CALF_INDICES)), self._cfg.control_config.calf_kd, dtype=np.float64
        )
        return compute_real68_motor_ctrl(
            policy_ctrl,
            active_pos,
            active_vel,
            hip_kd=hip_kd,
            wheel_kd=wheel_kd,
            calf_kp=calf_kp,
            calf_kd=calf_kd,
            ctrl_lower=self._ctrl_lower,
            ctrl_upper=self._ctrl_upper,
            out=self._last_motor_ctrl,
        )

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": self._reward_tracking_forward_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "balanced_tracking_lin_vel": self._reward_balanced_tracking_forward_vel,
            "balanced_tracking_ang_vel": self._reward_balanced_tracking_ang_vel,
            "balanced_speed_match": self._reward_balanced_speed_match,
            "balanced_forward_progress": self._reward_balanced_forward_progress,
            "balanced_net_progress": self._reward_balanced_net_progress,
            "forward_progress": self._reward_forward_progress,
            "net_progress": self._reward_net_progress,
            "vx_abs_net_gap": self._reward_vx_abs_net_gap,
            "under_speed": self._reward_under_speed,
            "balanced_under_speed": self._reward_balanced_under_speed,
            "drive_wheel_command": self._reward_drive_wheel_command,
            "yaw_rate_when_uncommanded": rewards.yaw_rate_when_uncommanded,
            "heading_stability": self._reward_heading_stability,
            "lateral_drift": self._reward_lateral_drift,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "orientation": self._reward_orientation,
            "excess_tilt": self._reward_excess_tilt,
            "tilt_termination": self._reward_tilt_termination,
            "action_rate": rewards.action_rate,
            "alive": rewards.alive,
            "torques": self._reward_torques_l2,
            "wheel_vel": self._reward_wheel_vel,
            "posture": self._reward_posture,
            "leg_symmetry": self._reward_leg_symmetry,
            "drive_action_symmetry": self._reward_drive_action_symmetry,
            "calf_action_symmetry": self._reward_calf_action_symmetry,
            "standing_lin_vel": self._reward_standing_lin_vel,
            "standing_wheel_action": self._reward_standing_wheel_action,
            "standing_under_height": self._reward_standing_under_height,
            "standing_liangan5_contact": self._reward_standing_liangan5_contact,
            "standing_orientation": self._reward_standing_orientation,
            "standing_posture": self._reward_standing_posture,
            "standing_leg_symmetry": self._reward_standing_leg_symmetry,
            "height_tracking": self._reward_height_tracking,
            "height_safety": self._reward_height_safety,
            "under_height": self._reward_under_height,
            "joint_pos_penalty": self._reward_joint_pos_penalty,
            "joint_power": self._reward_joint_power,
            "reverse_motion": self._reward_reverse_motion,
            "nonwheel_contact": self._reward_nonwheel_contact,
            "liangan5_contact": self._reward_liangan5_contact,
            "liangan5_contact_asymmetry": self._reward_liangan5_contact_asymmetry,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.gravity), dtype=self._np_dtype
        )
        accel = self.get_accel()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._wheel_contacts[:] = scalarize_contacts(
            self._backend, WHEEL_CONTACT_SENSORS, dtype=self._np_dtype
        )
        self._nonwheel_contacts[:] = scalarize_contacts(
            self._backend, NONWHEEL_CONTACT_SENSORS, dtype=self._np_dtype
        )
        state.info["torques"] = self._last_motor_ctrl.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        state.info["wheel_contacts"] = self._wheel_contacts.copy()
        state.info["nonwheel_contacts"] = self._nonwheel_contacts.copy()
        termination_causes = self._compute_termination_causes(gravity)
        state.info["termination_tilt"] = termination_causes["tilt"]
        state.info["height_violation"] = termination_causes["height"]
        state.info["height_low_violation"] = termination_causes["height_low"]
        state.info["height_high_violation"] = termination_causes["height_high"]
        state.info["termination_nonwheel_contact"] = termination_causes["nonwheel_contact"]
        terminated = np.asarray(termination_causes["terminated"], dtype=bool)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(
            state.info,
            linvel,
            gyro,
            gravity,
            accel,
            dof_pos,
            dof_vel,
        )
        state = state.replace(obs=obs, reward=reward, terminated=terminated)
        self._after_update_state(state, linvel, gyro)
        return state

    def _after_update_state(
        self,
        state: NpEnvState,
        linvel: np.ndarray,
        gyro: np.ndarray,
    ) -> None:
        self._accumulate_command_segments(state.info, linvel, gyro)
        log = state.info.get("log")
        if not isinstance(log, dict):
            return
        self._write_motion_metrics(log, state.info, linvel, gyro)
        self._write_termination_metrics(log, state.info, state.terminated)
        self._update_command_curriculum()
        self._write_command_curriculum_metrics(log)

    def _before_autoreset(self, done: np.ndarray) -> None:
        done_ids = np.flatnonzero(done).astype(np.int32)
        self._finalize_command_segments(done_ids)

    def _accumulate_command_segments(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
    ) -> None:
        commands = np.asarray(
            info.get("commands", np.zeros((self._num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        linvel_forward = self._forward_linvel(linvel)
        gyro_z = np.asarray(gyro[:, 2], dtype=self._np_dtype)
        nonwheel_contacts = np.asarray(
            info.get(
                "nonwheel_contacts",
                np.zeros((self._num_envs, len(NONWHEEL_CONTACT_SENSORS)), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        nonwheel_contact_frac = np.mean(nonwheel_contacts > 0.0, axis=1)
        tilt = np.asarray(
            info.get("termination_tilt", np.zeros((self._num_envs,), dtype=bool)),
            dtype=self._np_dtype,
        )
        gravity = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.gravity), dtype=self._np_dtype
        )
        tilt_angle_deg = np.rad2deg(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0)))
        height_violation = np.asarray(
            info.get("height_violation", np.zeros((self._num_envs,), dtype=bool)),
            dtype=self._np_dtype,
        )
        signed_speed = linvel_forward * np.sign(commands[:, 0])
        self._segment_cmd_x[:] = commands[:, 0]
        self._segment_cmd_yaw[:] = commands[:, 2]
        self._segment_abs_vx_sum += np.abs(linvel_forward)
        self._segment_signed_vx_sum += signed_speed
        self._segment_abs_wz_sum += np.abs(gyro_z)
        self._segment_vx_error_sum += np.abs(commands[:, 0] - linvel_forward)
        self._segment_wz_error_sum += np.abs(commands[:, 2] - gyro_z)
        self._segment_nonwheel_contact_sum += nonwheel_contact_frac
        self._segment_tilt_sum += tilt
        self._segment_tilt_angle_sum += tilt_angle_deg
        self._segment_height_violation_sum += height_violation
        self._segment_steps += 1

    def _reset_command_segments(self, env_ids: np.ndarray, commands: np.ndarray) -> None:
        if env_ids.size == 0:
            return
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype)
        self._segment_cmd_x[env_ids] = commands[:, 0]
        self._segment_cmd_yaw[env_ids] = commands[:, 2]
        self._segment_abs_vx_sum[env_ids] = 0.0
        self._segment_signed_vx_sum[env_ids] = 0.0
        self._segment_abs_wz_sum[env_ids] = 0.0
        self._segment_vx_error_sum[env_ids] = 0.0
        self._segment_wz_error_sum[env_ids] = 0.0
        self._segment_nonwheel_contact_sum[env_ids] = 0.0
        self._segment_tilt_sum[env_ids] = 0.0
        self._segment_tilt_angle_sum[env_ids] = 0.0
        self._segment_height_violation_sum[env_ids] = 0.0
        self._segment_steps[env_ids] = 0
        if base_pos.shape[0] == self._num_envs:
            self._segment_start_pos_xy[env_ids] = base_pos[env_ids, :2]
        if base_quat.shape[0] == self._num_envs:
            self._segment_start_yaw[env_ids] = np_yaw_from_quat(base_quat[env_ids])

    def _reset_command_curriculum_stats(self) -> None:
        self._curriculum_vx_count.fill(0)
        self._curriculum_vx_signed_speed_ratio_sum.fill(0.0)
        self._curriculum_vx_error_sum.fill(0.0)
        self._curriculum_vx_tilt_rate_sum.fill(0.0)
        self._curriculum_vx_tilt_angle_sum.fill(0.0)
        self._curriculum_vx_height_violation_rate_sum.fill(0.0)
        self._curriculum_vx_nonwheel_contact_rate_sum.fill(0.0)
        self._curriculum_yaw_count.fill(0)
        self._curriculum_yaw_error_sum.fill(0.0)
        self._curriculum_yaw_tilt_rate_sum.fill(0.0)
        self._curriculum_yaw_tilt_angle_sum.fill(0.0)
        self._curriculum_yaw_height_violation_rate_sum.fill(0.0)
        self._curriculum_yaw_nonwheel_contact_rate_sum.fill(0.0)
        self._reset_standing_bootstrap_stats()
        self._last_standing_bootstrap_eval = None
        self._curriculum_recorded_segments = 0

    def _reset_standing_bootstrap_stats(self) -> None:
        self._standing_segment_count = 0
        self._standing_segment_steps_sum = 0.0
        self._standing_segment_abs_vx_sum = 0.0
        self._standing_segment_wz_error_sum = 0.0
        self._standing_segment_nonwheel_contact_sum = 0.0

    def _curriculum_abs_limit_x(self) -> float:
        return float(
            max(abs(self._command_curriculum_low[0]), abs(self._command_curriculum_high[0]))
        )

    def _curriculum_abs_limit_yaw(self) -> float:
        return float(
            max(abs(self._command_curriculum_low[2]), abs(self._command_curriculum_high[2]))
        )

    def _curriculum_bin_index(self, magnitude: float, max_abs: float) -> int | None:
        if (
            magnitude <= _REAL68_CURRICULUM_MIN_ABS_COMMAND
            or max_abs <= _REAL68_CURRICULUM_MIN_ABS_COMMAND
        ):
            return None
        span = max(max_abs - _REAL68_CURRICULUM_MIN_ABS_COMMAND, 1.0e-6)
        normalized = np.clip(
            (magnitude - _REAL68_CURRICULUM_MIN_ABS_COMMAND) / span,
            0.0,
            np.nextafter(1.0, 0.0),
        )
        return min(int(normalized * _REAL68_CURRICULUM_NUM_BINS), _REAL68_CURRICULUM_NUM_BINS - 1)

    def _curriculum_bucket_upper(self, index: int, max_abs: float) -> float:
        span = max(max_abs - _REAL68_CURRICULUM_MIN_ABS_COMMAND, 0.0)
        return float(
            _REAL68_CURRICULUM_MIN_ABS_COMMAND
            + span * float(index + 1) / float(_REAL68_CURRICULUM_NUM_BINS)
        )

    def _record_command_segment_stats(
        self,
        *,
        cmd_x: float,
        cmd_yaw: float,
        mean_abs_vx: float,
        mean_signed_vx: float,
        mean_abs_wz: float,
        vx_error: float,
        wz_error: float,
        mean_nonwheel_contact: float,
        mean_tilt: float,
        mean_tilt_angle_deg: float,
        mean_height_violation: float,
        segment_steps: int,
    ) -> None:
        abs_cmd_x = abs(cmd_x)
        abs_cmd_yaw = abs(cmd_yaw)
        if (
            abs_cmd_x <= _REAL68_CURRICULUM_MIN_ABS_COMMAND
            and abs_cmd_yaw <= _REAL68_CURRICULUM_MIN_ABS_COMMAND
        ):
            self._standing_segment_count += 1
            self._standing_segment_steps_sum += float(segment_steps)
            self._standing_segment_abs_vx_sum += mean_abs_vx
            self._standing_segment_wz_error_sum += mean_abs_wz
            self._standing_segment_nonwheel_contact_sum += mean_nonwheel_contact

        vx_max_abs = self._curriculum_abs_limit_x()
        vx_index = self._curriculum_bin_index(abs_cmd_x, vx_max_abs)
        if vx_index is not None:
            self._curriculum_vx_count[vx_index] += 1
            signed_speed_ratio = max(mean_signed_vx, 0.0) / max(abs_cmd_x, 1.0e-6)
            self._curriculum_vx_signed_speed_ratio_sum[vx_index] += signed_speed_ratio
            self._curriculum_vx_error_sum[vx_index] += vx_error
            self._curriculum_vx_tilt_rate_sum[vx_index] += mean_tilt
            self._curriculum_vx_tilt_angle_sum[vx_index] += mean_tilt_angle_deg
            self._curriculum_vx_height_violation_rate_sum[vx_index] += mean_height_violation
            self._curriculum_vx_nonwheel_contact_rate_sum[vx_index] += mean_nonwheel_contact

        yaw_max_abs = self._curriculum_abs_limit_yaw()
        yaw_index = self._curriculum_bin_index(abs_cmd_yaw, yaw_max_abs)
        if yaw_index is not None:
            self._curriculum_yaw_count[yaw_index] += 1
            self._curriculum_yaw_error_sum[yaw_index] += wz_error
            self._curriculum_yaw_tilt_rate_sum[yaw_index] += mean_tilt
            self._curriculum_yaw_tilt_angle_sum[yaw_index] += mean_tilt_angle_deg
            self._curriculum_yaw_height_violation_rate_sum[yaw_index] += mean_height_violation
            self._curriculum_yaw_nonwheel_contact_rate_sum[yaw_index] += mean_nonwheel_contact

        self._curriculum_recorded_segments += 1

    def _finalize_command_segments(self, env_ids: np.ndarray) -> None:
        if env_ids.size == 0:
            return
        valid_ids = env_ids[self._segment_steps[env_ids] > 0]
        if valid_ids.size == 0:
            return
        for env_id in valid_ids:
            steps = int(self._segment_steps[env_id])
            mean_abs_vx = float(self._segment_abs_vx_sum[env_id] / max(steps, 1))
            mean_signed_vx = float(self._segment_signed_vx_sum[env_id] / max(steps, 1))
            mean_abs_wz = float(self._segment_abs_wz_sum[env_id] / max(steps, 1))
            vx_error = float(self._segment_vx_error_sum[env_id] / max(steps, 1))
            wz_error = float(self._segment_wz_error_sum[env_id] / max(steps, 1))
            mean_nonwheel_contact = float(
                self._segment_nonwheel_contact_sum[env_id] / max(steps, 1)
            )
            mean_tilt = float(self._segment_tilt_sum[env_id] / max(steps, 1))
            mean_tilt_angle_deg = float(self._segment_tilt_angle_sum[env_id] / max(steps, 1))
            mean_height_violation = float(
                self._segment_height_violation_sum[env_id] / max(steps, 1)
            )
            self._record_command_segment_stats(
                cmd_x=float(self._segment_cmd_x[env_id]),
                cmd_yaw=float(self._segment_cmd_yaw[env_id]),
                mean_abs_vx=mean_abs_vx,
                mean_signed_vx=mean_signed_vx,
                mean_abs_wz=mean_abs_wz,
                vx_error=vx_error,
                wz_error=wz_error,
                mean_nonwheel_contact=mean_nonwheel_contact,
                mean_tilt=mean_tilt,
                mean_tilt_angle_deg=mean_tilt_angle_deg,
                mean_height_violation=mean_height_violation,
                segment_steps=steps,
            )

    def _standing_bootstrap_eval(self) -> dict[str, float] | None:
        min_count = max(int(self._cfg.command_curriculum.standing_bootstrap_min_segments), 1)
        count = int(self._standing_segment_count)
        if count < min_count:
            return None
        return {
            "count": float(count),
            "mean_steps": float(self._standing_segment_steps_sum / count),
            "abs_vx": float(self._standing_segment_abs_vx_sum / count),
            "wz_error": float(self._standing_segment_wz_error_sum / count),
            "nonwheel_contact": float(self._standing_segment_nonwheel_contact_sum / count),
        }

    def _curriculum_vx_eval(self) -> dict[str, float] | None:
        min_count = max(int(self._cfg.command_curriculum.min_segment_count), 1)
        max_abs = self._curriculum_abs_limit_x()
        for index in range(_REAL68_CURRICULUM_NUM_BINS - 1, -1, -1):
            count = int(self._curriculum_vx_count[index])
            if count < min_count:
                continue
            return {
                "bucket_upper": self._curriculum_bucket_upper(index, max_abs),
                "count": float(count),
                "speed_ratio": float(self._curriculum_vx_signed_speed_ratio_sum[index] / count),
                "vx_error": float(self._curriculum_vx_error_sum[index] / count),
                "tilt_rate": float(self._curriculum_vx_tilt_rate_sum[index] / count),
                "tilt_angle_deg": float(self._curriculum_vx_tilt_angle_sum[index] / count),
                "height_violation_rate": float(
                    self._curriculum_vx_height_violation_rate_sum[index] / count
                ),
                "nonwheel_contact_rate": float(
                    self._curriculum_vx_nonwheel_contact_rate_sum[index] / count
                ),
            }
        return None

    def _curriculum_yaw_eval(self) -> dict[str, float] | None:
        min_count = max(int(self._cfg.command_curriculum.min_segment_count), 1)
        max_abs = self._curriculum_abs_limit_yaw()
        for index in range(_REAL68_CURRICULUM_NUM_BINS - 1, -1, -1):
            count = int(self._curriculum_yaw_count[index])
            if count < min_count:
                continue
            return {
                "bucket_upper": self._curriculum_bucket_upper(index, max_abs),
                "count": float(count),
                "wz_error": float(self._curriculum_yaw_error_sum[index] / count),
                "tilt_rate": float(self._curriculum_yaw_tilt_rate_sum[index] / count),
                "tilt_angle_deg": float(self._curriculum_yaw_tilt_angle_sum[index] / count),
                "height_violation_rate": float(
                    self._curriculum_yaw_height_violation_rate_sum[index] / count
                ),
                "nonwheel_contact_rate": float(
                    self._curriculum_yaw_nonwheel_contact_rate_sum[index] / count
                ),
            }
        return None

    def _write_motion_metrics(
        self,
        log: dict[str, Any],
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
    ) -> None:
        commands = np.asarray(
            info.get("commands", np.zeros((self._num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        cmd_x = commands[:, 0]
        cmd_yaw = commands[:, 2]
        linvel_forward = self._forward_linvel(linvel)
        linvel_lateral = self._lateral_linvel(linvel)
        linvel_raw_x = np.asarray(linvel[:, 0], dtype=self._np_dtype)
        linvel_raw_y = np.asarray(linvel[:, 1], dtype=self._np_dtype)
        gyro_z = gyro[:, 2]
        active = np.abs(cmd_x) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        signed_speed = linvel_forward * np.sign(cmd_x)
        mean_signed_vx = np.where(active, signed_speed, 0.0)
        signed_speed_ratio = np.where(
            active,
            np.maximum(signed_speed, 0.0) / np.maximum(np.abs(cmd_x), 1.0e-6),
            0.0,
        )
        log["metrics/linvel_x"] = float(np.mean(linvel_forward))
        log["metrics/linvel_forward"] = float(np.mean(linvel_forward))
        log["metrics/linvel_lateral"] = float(np.mean(linvel_lateral))
        log["metrics/raw_linvel_x"] = float(np.mean(linvel_raw_x))
        log["metrics/raw_linvel_y"] = float(np.mean(linvel_raw_y))
        log["metrics/gyro_z"] = float(np.mean(gyro_z))
        log["metrics/cmd_x"] = float(np.mean(cmd_x))
        log["metrics/cmd_yaw"] = float(np.mean(cmd_yaw))
        log["metrics/vx_error"] = float(np.mean(np.abs(cmd_x - linvel_forward)))
        log["metrics/wz_error"] = float(np.mean(np.abs(cmd_yaw - gyro_z)))
        log["metrics/heading_error"] = float(np.mean(np.abs(self._compute_heading_error())))
        log["metrics/lateral_drift"] = float(np.mean(np.abs(self._compute_lateral_drift())))
        log["metrics/upright_gate"] = float(np.mean(self._upright_gate()))
        gravity = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.gravity), dtype=self._np_dtype
        )
        log["metrics/upright_cos"] = float(np.mean(gravity[:, 2]))
        log["metrics/tilt_angle_deg"] = float(
            np.rad2deg(np.mean(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))))
        )
        log["metrics/mean_abs_vx"] = float(np.mean(np.abs(linvel_forward)))
        log["metrics/mean_abs_lateral_vel"] = float(np.mean(np.abs(linvel_lateral)))
        log["metrics/mean_signed_vx"] = float(np.mean(mean_signed_vx))
        log["metrics/signed_speed_ratio"] = float(np.mean(signed_speed_ratio))
        log["metrics/vx_abs_net_gap"] = float(
            np.mean(np.abs(linvel_forward)) - np.mean(np.maximum(mean_signed_vx, 0.0))
        )
        log["metrics/mean_abs_wz"] = float(np.mean(np.abs(gyro_z)))
        log["metrics/mean_abs_cmd_x"] = float(np.mean(np.abs(cmd_x)))
        log["metrics/mean_abs_cmd_yaw"] = float(np.mean(np.abs(cmd_yaw)))
        log["metrics/commanded_nonzero_frac"] = float(np.mean(np.abs(cmd_x) > 0.05))
        log["metrics/commanded_yaw_nonzero_frac"] = float(np.mean(np.abs(cmd_yaw) > 0.05))
        # Effective (post-diamond-clip) command magnitudes and clip diagnostics.
        # info["commands"] already holds the clipped commands, so these expose
        # what the policy actually had to track vs. the nominal box.
        log["metrics/mean_abs_cmd_vx_effective"] = float(np.mean(np.abs(cmd_x)))
        log["metrics/mean_abs_cmd_wz_effective"] = float(np.mean(np.abs(cmd_yaw)))
        clip_scale = np.asarray(self._command_clip_scale, dtype=self._np_dtype)
        log["metrics/command_clip_frac"] = float(np.mean(clip_scale < (1.0 - 1.0e-6)))
        log["metrics/command_clip_scale_mean"] = float(np.mean(clip_scale))
        target_lean_forward = self._command_lean_gravity_target(cmd_x)
        log["metrics/target_lean_gx"] = 0.0
        log["metrics/target_lean_gy"] = float(np.mean(target_lean_forward))
        log["metrics/target_lean_forward"] = float(np.mean(target_lean_forward))
        height_commands = np.asarray(
            info.get(
                "height_commands",
                np.full(
                    (self._num_envs,), self._reward_cfg.base_height_target, dtype=self._np_dtype
                ),
            ),
            dtype=self._np_dtype,
        )
        log["metrics/target_height"] = float(np.mean(height_commands))
        base_height = self._reward_base_height_values(self._num_envs)
        log["metrics/base_height"] = float(np.mean(base_height))
        log["metrics/height_error"] = float(np.mean(height_commands - base_height))
        actions = np.asarray(
            info.get("current_actions", np.zeros((self._num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        last_actions = np.asarray(
            info.get("last_actions", np.zeros((self._num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        exec_actions = last_actions if self._cfg.control_config.simulate_action_latency else actions
        wheel_actions = actions[:, WHEEL_INDICES]
        wheel_exec_actions = exec_actions[:, WHEEL_INDICES]
        dof_vel = self.get_dof_vel()
        wheel_vel = dof_vel[:, WHEEL_INDICES]
        wheel_target_vel = wheel_exec_actions * float(self._cfg.control_config.wheel_velocity_scale)
        wheel_surface_speed = -np.mean(wheel_vel, axis=1) * _REAL68_WHEEL_RADIUS
        wheel_target_surface_speed = -np.mean(wheel_target_vel, axis=1) * _REAL68_WHEEL_RADIUS
        wheel_target_error = wheel_target_vel - wheel_vel
        torques = np.asarray(
            info.get("torques", np.zeros((self._num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        wheel_torques = torques[:, WHEEL_INDICES]
        wheel_lower = self._ctrl_lower[WHEEL_INDICES]
        wheel_upper = self._ctrl_upper[WHEEL_INDICES]
        wheel_torque_clipped = (wheel_torques <= (wheel_lower + 1.0e-5)) | (
            wheel_torques >= (wheel_upper - 1.0e-5)
        )
        log["metrics/mean_wheel_action"] = float(np.mean(wheel_actions))
        log["metrics/mean_left_wheel_action"] = float(np.mean(actions[:, WHEEL_INDICES[0]]))
        log["metrics/mean_right_wheel_action"] = float(np.mean(actions[:, WHEEL_INDICES[1]]))
        log["metrics/mean_abs_wheel_action"] = float(np.mean(np.abs(wheel_actions)))
        log["metrics/mean_wheel_exec_action"] = float(np.mean(wheel_exec_actions))
        log["metrics/mean_abs_wheel_exec_action"] = float(np.mean(np.abs(wheel_exec_actions)))
        log["metrics/mean_left_wheel_vel"] = float(np.mean(wheel_vel[:, 0]))
        log["metrics/mean_right_wheel_vel"] = float(np.mean(wheel_vel[:, 1]))
        log["metrics/mean_abs_wheel_vel"] = float(np.mean(np.abs(wheel_vel)))
        log["metrics/mean_wheel_target_vel"] = float(np.mean(wheel_target_vel))
        log["metrics/mean_abs_wheel_target_vel"] = float(np.mean(np.abs(wheel_target_vel)))
        log["metrics/mean_abs_wheel_target_error"] = float(np.mean(np.abs(wheel_target_error)))
        log["metrics/wheel_surface_speed"] = float(np.mean(wheel_surface_speed))
        log["metrics/mean_abs_wheel_surface_speed"] = float(np.mean(np.abs(wheel_surface_speed)))
        log["metrics/wheel_target_surface_speed"] = float(np.mean(wheel_target_surface_speed))
        log["metrics/mean_abs_wheel_torque"] = float(np.mean(np.abs(wheel_torques)))
        log["metrics/wheel_torque_clip_frac"] = float(np.mean(wheel_torque_clipped))
        log["metrics/mean_left_hip_action"] = float(np.mean(actions[:, _REAL68_LEFT_HIP_INDEX]))
        log["metrics/mean_right_hip_action"] = float(np.mean(actions[:, _REAL68_RIGHT_HIP_INDEX]))
        log["metrics/mean_left_calf_action"] = float(np.mean(actions[:, _REAL68_LEFT_CALF_INDEX]))
        log["metrics/mean_right_calf_action"] = float(np.mean(actions[:, _REAL68_RIGHT_CALF_INDEX]))
        log["metrics/hip_action_mirror_error"] = float(
            np.mean(
                np.square(actions[:, _REAL68_LEFT_HIP_INDEX] + actions[:, _REAL68_RIGHT_HIP_INDEX])
            )
        )
        log["metrics/calf_action_mirror_error"] = float(
            np.mean(
                np.square(
                    actions[:, _REAL68_LEFT_CALF_INDEX] + actions[:, _REAL68_RIGHT_CALF_INDEX]
                )
            )
        )
        log["metrics/wheel_action_sync_error"] = float(
            np.mean(
                np.square(
                    actions[:, _REAL68_LEFT_WHEEL_INDEX] - actions[:, _REAL68_RIGHT_WHEEL_INDEX]
                )
            )
        )
        nonwheel_contacts = np.asarray(
            info.get(
                "nonwheel_contacts",
                np.zeros((self._num_envs, len(NONWHEEL_CONTACT_SENSORS)), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        log["metrics/left_liangan5_contact"] = float(
            np.mean(nonwheel_contacts[:, _REAL68_LEFT_LIANGAN5_CONTACT_INDEX])
        )
        log["metrics/right_liangan5_contact"] = float(
            np.mean(nonwheel_contacts[:, _REAL68_RIGHT_LIANGAN5_CONTACT_INDEX])
        )
        log["metrics/liangan5_contact_gap"] = float(
            np.mean(
                np.abs(
                    nonwheel_contacts[:, _REAL68_LEFT_LIANGAN5_CONTACT_INDEX]
                    - nonwheel_contacts[:, _REAL68_RIGHT_LIANGAN5_CONTACT_INDEX]
                )
            )
        )
        if nonwheel_contacts.shape[1] == len(NONWHEEL_CONTACT_SENSORS):
            for idx, sensor_name in enumerate(NONWHEEL_CONTACT_SENSORS):
                log[f"diagnostics/{sensor_name}_count"] = float(
                    np.count_nonzero(nonwheel_contacts[:, idx] > 0.0)
                )

    def _write_command_curriculum_metrics(self, log: dict[str, Any]) -> None:
        vx_eval = self._curriculum_vx_eval()
        yaw_eval = self._curriculum_yaw_eval()
        log["command_curriculum/progress"] = float(self._command_curriculum_vx_progress)
        log["command_curriculum/vx_progress"] = float(self._command_curriculum_vx_progress)
        log["command_curriculum/yaw_progress"] = float(self._command_curriculum_yaw_progress)
        log["command_curriculum/low_vx"] = float(self._command_curriculum_low[0])
        log["command_curriculum/high_vx"] = float(self._command_curriculum_high[0])
        log["command_curriculum/low_wz"] = float(self._command_curriculum_low[2])
        log["command_curriculum/high_wz"] = float(self._command_curriculum_high[2])
        log["command_curriculum/vx_error_up_threshold"] = float(self._vx_curriculum_up_threshold())
        log["command_curriculum/speed_ratio"] = float(
            0.0 if vx_eval is None else vx_eval["speed_ratio"]
        )
        log["command_curriculum/eval_signed_vx_ratio"] = float(
            0.0 if vx_eval is None else vx_eval["speed_ratio"]
        )
        log["command_curriculum/eval_vx_error"] = float(
            0.0 if vx_eval is None else vx_eval["vx_error"]
        )
        log["command_curriculum/eval_tilt_rate"] = float(
            0.0 if vx_eval is None else vx_eval["tilt_rate"]
        )
        log["command_curriculum/eval_tilt_angle_deg"] = float(
            0.0 if vx_eval is None else vx_eval["tilt_angle_deg"]
        )
        log["command_curriculum/eval_height_violation_rate"] = float(
            0.0 if vx_eval is None else vx_eval["height_violation_rate"]
        )
        log["command_curriculum/eval_nonwheel_contact_rate"] = float(
            0.0 if vx_eval is None else vx_eval["nonwheel_contact_rate"]
        )
        log["command_curriculum/eval_wz_error"] = float(
            0.0 if yaw_eval is None else yaw_eval["wz_error"]
        )
        log["command_curriculum/eval_yaw_tilt_rate"] = float(
            0.0 if yaw_eval is None else yaw_eval["tilt_rate"]
        )
        log["command_curriculum/eval_yaw_tilt_angle_deg"] = float(
            0.0 if yaw_eval is None else yaw_eval["tilt_angle_deg"]
        )
        log["command_curriculum/eval_yaw_height_violation_rate"] = float(
            0.0 if yaw_eval is None else yaw_eval["height_violation_rate"]
        )
        log["command_curriculum/eval_yaw_nonwheel_contact_rate"] = float(
            0.0 if yaw_eval is None else yaw_eval["nonwheel_contact_rate"]
        )
        log["command_curriculum/eval_bucket_high_vx"] = float(
            0.0 if vx_eval is None else vx_eval["bucket_upper"]
        )
        log["command_curriculum/eval_bucket_high_wz"] = float(
            0.0 if yaw_eval is None else yaw_eval["bucket_upper"]
        )
        log["command_curriculum/eval_count_vx"] = float(
            0.0 if vx_eval is None else vx_eval["count"]
        )
        log["command_curriculum/eval_count_wz"] = float(
            0.0 if yaw_eval is None else yaw_eval["count"]
        )
        log["command_curriculum/segments_recorded"] = float(self._curriculum_recorded_segments)
        log["command_curriculum/standing_prob"] = float(self._standing_command_probability())
        straight_prob, yaw_only_prob = self._command_mix_probabilities()
        log["command_curriculum/straight_command_prob"] = float(straight_prob)
        log["command_curriculum/yaw_only_command_prob"] = float(yaw_only_prob)
        log["command_curriculum/combo_command_prob"] = float(
            max(0.0, 1.0 - straight_prob - yaw_only_prob)
        )
        standing_eval = self._standing_bootstrap_eval()
        if standing_eval is None:
            standing_eval = self._last_standing_bootstrap_eval
        log["command_curriculum/standing_bootstrap_complete"] = float(
            self._standing_bootstrap_complete
        )
        log["command_curriculum/standing_eval_count"] = float(
            0.0 if standing_eval is None else standing_eval["count"]
        )
        log["command_curriculum/standing_eval_mean_steps"] = float(
            0.0 if standing_eval is None else standing_eval["mean_steps"]
        )
        log["command_curriculum/standing_eval_abs_vx"] = float(
            0.0 if standing_eval is None else standing_eval["abs_vx"]
        )
        log["command_curriculum/standing_eval_wz_error"] = float(
            0.0 if standing_eval is None else standing_eval["wz_error"]
        )
        log["command_curriculum/standing_eval_nonwheel_contact"] = float(
            0.0 if standing_eval is None else standing_eval["nonwheel_contact"]
        )

    def _write_termination_metrics(
        self,
        log: dict[str, Any],
        info: dict[str, Any],
        terminated: np.ndarray,
    ) -> None:
        tilt = np.asarray(
            info.get("termination_tilt", np.zeros((self._num_envs,), dtype=bool)),
            dtype=bool,
        )
        nonwheel_contact = np.asarray(
            info.get("termination_nonwheel_contact", np.zeros((self._num_envs,), dtype=bool)),
            dtype=bool,
        )
        height = np.asarray(
            info.get("height_violation", np.zeros((self._num_envs,), dtype=bool)),
            dtype=bool,
        )
        height_low = np.asarray(
            info.get("height_low_violation", np.zeros((self._num_envs,), dtype=bool)),
            dtype=bool,
        )
        height_high = np.asarray(
            info.get("height_high_violation", np.zeros((self._num_envs,), dtype=bool)),
            dtype=bool,
        )
        terminated = np.asarray(terminated, dtype=bool)
        log["termination/total_count"] = float(np.count_nonzero(terminated))
        log["termination/tilt_count"] = float(np.count_nonzero(terminated & tilt))
        log["termination/nonwheel_contact_count"] = float(
            np.count_nonzero(terminated & nonwheel_contact)
        )
        log["diagnostics/height_violation_count"] = float(np.count_nonzero(height))
        log["diagnostics/height_low_violation_count"] = float(np.count_nonzero(height_low))
        log["diagnostics/height_high_violation_count"] = float(np.count_nonzero(height_high))

    def _compute_termination_causes(self, gravity: np.ndarray) -> dict[str, np.ndarray]:
        base_z = self._reward_base_height_values(gravity.shape[0])
        cfg = self._cfg.termination_config
        tilt = np.zeros((self._num_envs,), dtype=bool)
        if cfg.fall_termination:
            tilt = np.asarray(gravity[:, 2] <= self._reward_cfg.max_tilt_cos, dtype=bool)
        height_low = np.asarray(base_z <= self._reward_cfg.min_base_height, dtype=bool)
        height_high = np.asarray(base_z >= self._reward_cfg.max_base_height, dtype=bool)
        height = np.asarray(
            height_low | height_high,
            dtype=bool,
        )
        contact_threshold = float(cfg.nonwheel_contact_threshold)
        contact_active = np.max(self._nonwheel_contacts, axis=1) > contact_threshold
        if cfg.nonwheel_contact_termination:
            self._nonwheel_contact_steps[contact_active] += 1
            self._nonwheel_contact_steps[~contact_active] = 0
        else:
            self._nonwheel_contact_steps.fill(0)
        nonwheel_contact = np.asarray(
            self._nonwheel_contact_steps >= max(int(cfg.nonwheel_contact_max_steps), 1),
            dtype=bool,
        )
        terminated = np.asarray(tilt | nonwheel_contact, dtype=bool)
        return {
            "tilt": tilt,
            "height": height,
            "height_low": height_low,
            "height_high": height_high,
            "nonwheel_contact": nonwheel_contact,
            "terminated": terminated,
        }

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        return np.asarray(self._compute_termination_causes(gravity)["terminated"], dtype=bool)

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        accel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        posture_diff = dof_pos[:, POSTURE_INDICES] - self.default_angles[POSTURE_INDICES]
        posture_vel = dof_vel[:, POSTURE_INDICES]
        wheel_vel = dof_vel[:, WHEEL_INDICES]
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_accel = self._obs_noise(accel, noise_cfg.scale_accel)
        noisy_linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        noisy_posture_diff = self._obs_noise(posture_diff, noise_cfg.scale_joint_angle)
        noisy_posture_vel = self._obs_noise(posture_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_joint_vel)
        base_height = self._reward_base_height_values(gyro.shape[0])
        height_commands = np.asarray(
            info.get(
                "height_commands", np.full((gyro.shape[0],), self._reward_cfg.base_height_target)
            ),
            dtype=self._np_dtype,
        )
        height_error = (height_commands - base_height)[:, None]
        num_obs = gyro.shape[0]
        last_actions = np.asarray(
            info.get("current_actions", np.zeros((num_obs, self._num_action))),
            dtype=self._np_dtype,
        )
        motor_torque = np.asarray(
            info.get("torques", np.zeros((num_obs, self._num_action))),
            dtype=self._np_dtype,
        )
        wheel_contacts = np.asarray(
            info.get(
                "wheel_contacts",
                np.zeros((num_obs, len(WHEEL_CONTACT_SENSORS)), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        nonwheel_contacts = np.asarray(
            info.get(
                "nonwheel_contacts",
                np.zeros((num_obs, len(NONWHEEL_CONTACT_SENSORS)), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )

        obs = np.concatenate(
            [
                noisy_linvel,
                noisy_gyro,
                -noisy_gravity,
                noisy_accel,
                noisy_posture_diff,
                noisy_posture_vel,
                noisy_wheel_vel,
                last_actions,
                info["commands"],
                height_error,
            ],
            axis=1,
            dtype=self._np_dtype,
        )
        critic = np.concatenate(
            [
                gyro,
                -gravity,
                accel,
                posture_diff,
                posture_vel,
                wheel_vel,
                last_actions,
                info["commands"],
                height_error,
                linvel,
                motor_torque,
                wheel_contacts,
                nonwheel_contacts,
            ],
            axis=1,
            dtype=self._np_dtype,
        )
        return {"obs": obs, "critic": critic}

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=num_obs,
            default_angles=DEFAULT_ACTIVE_ANGLES.astype(self._np_dtype),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._reward_base_height_values(num_obs),
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=True,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )

    def _update_commands(self, info: dict) -> None:
        commands = np.asarray(
            info.get("commands", np.zeros((self._num_envs, 3))), dtype=self._np_dtype
        )
        height_commands = np.asarray(
            info.get(
                "height_commands",
                np.full(
                    (self._num_envs,), self._reward_cfg.base_height_target, dtype=self._np_dtype
                ),
            ),
            dtype=self._np_dtype,
        )
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                env_ids = np.flatnonzero(resample_mask).astype(np.int32)
                self._finalize_command_segments(env_ids)
                sampled_commands = self.sample_velocity_commands(num_resample)
                commands[resample_mask] = sampled_commands
                if self._last_command_clip_scale.shape[0] == num_resample:
                    self._command_clip_scale[env_ids] = self._last_command_clip_scale
                height_commands[resample_mask] = self.sample_height_commands(
                    num_resample,
                    commands=sampled_commands,
                )
                self._reset_command_segments(env_ids, sampled_commands)
        commands[:, 1] = 0.0
        info["commands"] = commands
        info["height_commands"] = height_commands

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray(
            (dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt, dtype=self._np_dtype
        )
        self._last_dof_vel_for_acc[:] = dof_vel
        return qacc

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        if base_pos.shape[0] != num_obs:
            return np.full((num_obs,), HOME_BASE_HEIGHT, dtype=self._np_dtype)
        self._base_height[:] = base_pos[:, 2]
        return self._base_height.copy()

    def _balance_gate_cfg(self) -> RewardConfig.BalanceGateConfig:
        raw_cfg = self._reward_cfg.balance_gate
        if isinstance(raw_cfg, RewardConfig.BalanceGateConfig):
            return raw_cfg
        if isinstance(raw_cfg, dict):
            return RewardConfig.BalanceGateConfig(**raw_cfg)
        raise TypeError(f"Unsupported balance_gate config type: {type(raw_cfg)!r}")

    def _excess_tilt_cfg(self) -> RewardConfig.ExcessTiltConfig:
        raw_cfg = self._reward_cfg.excess_tilt
        if isinstance(raw_cfg, RewardConfig.ExcessTiltConfig):
            return raw_cfg
        if isinstance(raw_cfg, dict):
            return RewardConfig.ExcessTiltConfig(**raw_cfg)
        raise TypeError(f"Unsupported excess_tilt config type: {type(raw_cfg)!r}")

    def _upright_gate(self) -> np.ndarray:
        gate_cfg = self._balance_gate_cfg()
        if not gate_cfg.enabled:
            return np.ones((self._num_envs,), dtype=self._np_dtype)
        gravity = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.gravity), dtype=self._np_dtype
        )
        min_cos = float(gate_cfg.upright_cos_min)
        full_cos = float(gate_cfg.upright_cos_full)
        span = max(full_cos - min_cos, 1.0e-6)
        gate = np.clip((gravity[:, 2] - min_cos) / span, 0.0, 1.0)
        return np.asarray(gate, dtype=self._np_dtype)

    def _compute_heading_error(self) -> np.ndarray:
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=self._np_dtype)
        if base_quat.shape[0] != self._num_envs:
            self._heading_error.fill(0.0)
            return self._heading_error
        yaw = np_yaw_from_quat(base_quat)
        self._heading_error[:] = np_wrap_to_pi(yaw - self._segment_start_yaw)
        return self._heading_error

    def _compute_lateral_drift(self) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=self._np_dtype)
        if base_pos.shape[0] != self._num_envs:
            self._lateral_drift.fill(0.0)
            return self._lateral_drift
        delta = base_pos[:, :2] - self._segment_start_pos_xy
        start_yaw = self._segment_start_yaw
        self._lateral_drift[:] = -np.sin(start_yaw) * delta[:, 0] + np.cos(start_yaw) * delta[:, 1]
        return self._lateral_drift

    def _reward_torques_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        return np.asarray(np.sum(np.square(torques), axis=1), dtype=self._np_dtype)

    def _command_lean_cfg(self) -> RewardConfig.CommandLeanConfig:
        raw_cfg = self._reward_cfg.command_lean
        if isinstance(raw_cfg, RewardConfig.CommandLeanConfig):
            return raw_cfg
        if isinstance(raw_cfg, dict):
            return RewardConfig.CommandLeanConfig(**raw_cfg)
        raise TypeError(f"Unsupported command_lean config type: {type(raw_cfg)!r}")

    def _command_lean_gravity_target(self, cmd_x: np.ndarray) -> np.ndarray:
        lean_cfg = self._command_lean_cfg()
        if not lean_cfg.enabled or lean_cfg.gravity_x_limit <= 0.0:
            return np.zeros_like(cmd_x, dtype=self._np_dtype)
        target = -np.asarray(cmd_x, dtype=self._np_dtype) * float(lean_cfg.gravity_x_gain)
        return np.asarray(
            np.clip(target, -float(lean_cfg.gravity_x_limit), float(lean_cfg.gravity_x_limit)),
            dtype=self._np_dtype,
        )

    def _command_target_posture(self, cmd_x: np.ndarray) -> np.ndarray:
        lean_cfg = self._command_lean_cfg()
        target = np.zeros((cmd_x.shape[0], len(POSTURE_INDICES)), dtype=self._np_dtype)
        if not lean_cfg.enabled:
            return target
        signed_cmd = np.asarray(cmd_x, dtype=self._np_dtype)
        hip = np.asarray(signed_cmd * float(lean_cfg.hip_gain), dtype=self._np_dtype)
        calf = np.asarray(signed_cmd * float(lean_cfg.calf_gain), dtype=self._np_dtype)
        if lean_cfg.hip_limit > 0.0:
            hip = np.asarray(
                np.clip(hip, -float(lean_cfg.hip_limit), float(lean_cfg.hip_limit)),
                dtype=self._np_dtype,
            )
        if lean_cfg.calf_limit > 0.0:
            calf = np.asarray(
                np.clip(calf, -float(lean_cfg.calf_limit), float(lean_cfg.calf_limit)),
                dtype=self._np_dtype,
            )
        target[:, 0] = -hip
        target[:, 1] = calf
        target[:, 2] = hip
        target[:, 3] = -calf
        return target

    def _standing_mask(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        return np.asarray(
            (np.abs(commands[:, 0]) <= _REAL68_CURRICULUM_MIN_ABS_COMMAND)
            & (np.abs(commands[:, 2]) <= _REAL68_CURRICULUM_MIN_ABS_COMMAND),
            dtype=bool,
        )

    def _posture_anchor(self, standing: np.ndarray | None = None) -> np.ndarray:
        anchor = np.asarray(DEFAULT_ACTIVE_ANGLES, dtype=self._np_dtype).copy()
        if standing is None:
            return anchor
        standing = np.asarray(standing, dtype=bool)
        if standing.ndim != 1:
            raise ValueError(f"standing mask must be 1-D, got shape={standing.shape}")
        tiled = np.broadcast_to(anchor, (standing.shape[0], anchor.shape[0])).copy()
        if np.any(standing):
            tiled[standing] = np.asarray(SYMMETRIC_STANDING_ACTIVE_ANGLES, dtype=self._np_dtype)
        return tiled

    def _reward_orientation(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        target_gy = self._command_lean_gravity_target(commands[:, 0])
        gravity_x_error = np.square(ctx.gravity[:, 0])
        gravity_y_error = np.square(ctx.gravity[:, 1] - target_gy)
        return np.asarray(gravity_x_error + gravity_y_error, dtype=self._np_dtype)

    def _reward_excess_tilt(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        tilt_cfg = self._excess_tilt_cfg()
        tilt_deg = np.rad2deg(np.arccos(np.clip(ctx.gravity[:, 2], -1.0, 1.0)))
        excess = np.maximum(tilt_deg - float(tilt_cfg.limit_deg), 0.0)
        scale = max(float(tilt_cfg.scale_deg), 1.0e-6)
        return np.asarray(np.square(excess / scale), dtype=self._np_dtype)

    def _reward_tilt_termination(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(
            ctx.info.get("termination_tilt", np.zeros((ctx.num_envs,), dtype=bool)),
            dtype=self._np_dtype,
        )

    def _straight_motion_mask(self, ctx: RewardContext) -> np.ndarray:
        gate_cfg = self._balance_gate_cfg()
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        return np.asarray(
            (np.abs(commands[:, 0]) > float(gate_cfg.moving_command_vx_threshold))
            & (np.abs(commands[:, 2]) < float(gate_cfg.straight_command_yaw_threshold)),
            dtype=self._np_dtype,
        )

    def _reward_wheel_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.asarray(
            np.sum(np.square(ctx.dof_vel[:, WHEEL_INDICES]), axis=1), dtype=self._np_dtype
        )

    def _reward_posture(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        posture_target = self._command_target_posture(commands[:, 0])
        posture_anchor = self._posture_anchor(self._standing_mask(ctx))
        posture = (
            ctx.dof_pos[:, POSTURE_INDICES]
            - posture_anchor[:, POSTURE_INDICES]
            - posture_target
        )
        return np.asarray(np.sum(np.square(posture), axis=1), dtype=self._np_dtype)

    def _reward_tracking_forward_vel(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        forward_error = np.square(commands[:, 0] - self._forward_linvel(ctx.linvel))
        lateral_error = np.square(self._lateral_linvel(ctx.linvel))
        return np.asarray(
            np.exp(-(forward_error + lateral_error) / ctx.tracking_sigma),
            dtype=self._np_dtype,
        )

    def _reward_balanced_tracking_forward_vel(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(
            self._reward_tracking_forward_vel(ctx) * self._upright_gate(),
            dtype=self._np_dtype,
        )

    def _reward_balanced_tracking_ang_vel(self, ctx: RewardContext) -> np.ndarray:
        gate_cfg = self._balance_gate_cfg()
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        sigma = max(float(gate_cfg.yaw_tracking_sigma), 1.0e-6)
        yaw_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        return np.asarray(
            np.exp(-yaw_error / sigma) * self._upright_gate(),
            dtype=self._np_dtype,
        )

    def _reward_balanced_speed_match(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        cmd_x = np.asarray(commands[:, 0], dtype=self._np_dtype)
        active = np.abs(cmd_x) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        signed_speed = self._forward_linvel(ctx.linvel) * np.sign(cmd_x)
        signed_ratio = signed_speed / np.maximum(np.abs(cmd_x), 1.0e-6)
        ratio_error = np.square(signed_ratio - 1.0)
        match = np.exp(-ratio_error / max(float(self._reward_cfg.tracking_sigma), 1.0e-6))
        return np.asarray(np.where(active, match, 0.0) * self._upright_gate(), dtype=self._np_dtype)

    def _reward_balanced_forward_progress(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(
            self._reward_forward_progress(ctx) * self._upright_gate(), dtype=self._np_dtype
        )

    def _reward_balanced_net_progress(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(
            self._reward_net_progress(ctx) * self._upright_gate(), dtype=self._np_dtype
        )

    def _reward_heading_stability(self, ctx: RewardContext) -> np.ndarray:
        gate_cfg = self._balance_gate_cfg()
        mask = self._straight_motion_mask(ctx)
        limit = max(float(gate_cfg.heading_error_limit), 1.0e-6)
        error = np.abs(self._compute_heading_error()) / limit
        return np.asarray(np.square(error) * mask, dtype=self._np_dtype)

    def _reward_lateral_drift(self, ctx: RewardContext) -> np.ndarray:
        gate_cfg = self._balance_gate_cfg()
        mask = self._straight_motion_mask(ctx)
        limit = max(float(gate_cfg.lateral_drift_limit), 1.0e-6)
        drift = np.abs(self._compute_lateral_drift()) / limit
        return np.asarray(np.square(drift) * mask, dtype=self._np_dtype)

    def _reward_forward_progress(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        cmd_x = np.asarray(commands[:, 0], dtype=self._np_dtype)
        active = np.abs(cmd_x) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        cmd_abs = np.maximum(np.abs(cmd_x), 1.0e-6)
        signed_speed = self._forward_linvel(ctx.linvel) * np.sign(cmd_x)
        progress = np.clip(signed_speed / cmd_abs, 0.0, 1.0)
        return np.asarray(np.where(active, progress, 0.0), dtype=self._np_dtype)

    def _reward_under_speed(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        cmd_x = np.asarray(commands[:, 0], dtype=self._np_dtype)
        active = np.abs(cmd_x) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        cmd_abs = np.maximum(np.abs(cmd_x), 1.0e-6)
        signed_speed = self._forward_linvel(ctx.linvel) * np.sign(cmd_x)
        gap = np.maximum(cmd_abs - signed_speed, 0.0)
        return np.asarray(np.where(active, gap / cmd_abs, 0.0), dtype=self._np_dtype)

    def _reward_balanced_under_speed(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(
            self._reward_under_speed(ctx) * self._upright_gate(), dtype=self._np_dtype
        )

    def _reward_drive_wheel_command(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        actions = np.asarray(
            ctx.info.get("current_actions", np.zeros((ctx.num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        cmd_x = np.asarray(commands[:, 0], dtype=self._np_dtype)
        active = np.abs(cmd_x) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        drive = -np.mean(actions[:, WHEEL_INDICES], axis=1) * np.sign(cmd_x)
        drive = np.clip(drive, 0.0, 1.0)
        return np.asarray(np.where(active, drive, 0.0) * self._upright_gate(), dtype=self._np_dtype)

    def _reward_leg_symmetry(self, ctx: RewardContext) -> np.ndarray:
        posture_anchor = self._posture_anchor(self._standing_mask(ctx))
        posture_diff = ctx.dof_pos[:, POSTURE_INDICES] - posture_anchor[:, POSTURE_INDICES]
        left = posture_diff[:, _REAL68_LEFT_POSTURE]
        right = posture_diff[:, _REAL68_RIGHT_POSTURE]
        mirrored_right = right * _REAL68_MIRROR_SIGNS
        symmetry = np.sum(np.square(left - mirrored_right), axis=1)
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(symmetry * upright, dtype=self._np_dtype)

    def _reward_drive_action_symmetry(self, ctx: RewardContext) -> np.ndarray:
        actions = np.asarray(
            ctx.info.get("current_actions", np.zeros((ctx.num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        active_drive = np.abs(commands[:, 0]) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        hip_mirror = np.square(
            actions[:, _REAL68_LEFT_HIP_INDEX] + actions[:, _REAL68_RIGHT_HIP_INDEX]
        )
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(
            np.where(active_drive, hip_mirror, 0.0) * upright,
            dtype=self._np_dtype,
        )

    def _reward_calf_action_symmetry(self, ctx: RewardContext) -> np.ndarray:
        actions = np.asarray(
            ctx.info.get("current_actions", np.zeros((ctx.num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        active_drive = np.abs(commands[:, 0]) > (0.5 * _REAL68_CURRICULUM_MIN_ABS_COMMAND)
        calf_mirror = np.square(
            actions[:, _REAL68_LEFT_CALF_INDEX] + actions[:, _REAL68_RIGHT_CALF_INDEX]
        )
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(
            np.where(active_drive, calf_mirror, 0.0) * upright,
            dtype=self._np_dtype,
        )

    def _reward_standing_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        standing = self._standing_mask(ctx)
        drift = np.abs(self._forward_linvel(ctx.linvel)) + np.abs(self._lateral_linvel(ctx.linvel))
        return np.asarray(np.where(standing, drift, 0.0), dtype=self._np_dtype)

    def _reward_standing_wheel_action(self, ctx: RewardContext) -> np.ndarray:
        actions = np.asarray(
            ctx.info.get("current_actions", np.zeros((ctx.num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        standing = self._standing_mask(ctx)
        wheel_action = np.mean(np.square(actions[:, WHEEL_INDICES]), axis=1)
        return np.asarray(np.where(standing, wheel_action, 0.0), dtype=self._np_dtype)

    def _reward_standing_under_height(self, ctx: RewardContext) -> np.ndarray:
        targets = np.asarray(
            ctx.info.get(
                "height_commands", np.full((ctx.num_envs,), self._reward_cfg.base_height_target)
            ),
            dtype=self._np_dtype,
        )
        standing = self._standing_mask(ctx)
        margin = max(float(self._reward_cfg.height_safety_margin), 1.0e-6)
        under_height = np.maximum(targets - np.asarray(ctx.base_height, dtype=self._np_dtype), 0.0)
        return np.asarray(np.where(standing, under_height / margin, 0.0), dtype=self._np_dtype)

    def _reward_standing_liangan5_contact(self, ctx: RewardContext) -> np.ndarray:
        standing = self._standing_mask(ctx)
        liangan5_contact = np.asarray(
            np.mean(
                self._nonwheel_contacts[
                    :,
                    [
                        _REAL68_LEFT_LIANGAN5_CONTACT_INDEX,
                        _REAL68_RIGHT_LIANGAN5_CONTACT_INDEX,
                    ],
                ],
                axis=1,
            ),
            dtype=self._np_dtype,
        )
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(np.where(standing, liangan5_contact * upright, 0.0), dtype=self._np_dtype)

    def _reward_standing_orientation(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        standing = self._standing_mask(ctx)
        gate_cfg = self._balance_gate_cfg()
        sigma = max(float(gate_cfg.standing_roll_pitch_sigma), 1.0e-6)
        roll_pitch_error = np.square(ctx.gravity[:, 0]) + np.square(ctx.gravity[:, 1])
        return np.asarray(np.where(standing, roll_pitch_error / sigma, 0.0), dtype=self._np_dtype)

    def _reward_standing_posture(self, ctx: RewardContext) -> np.ndarray:
        standing = self._standing_mask(ctx)
        posture_anchor = self._posture_anchor(standing)
        posture = ctx.dof_pos[:, POSTURE_INDICES] - posture_anchor[:, POSTURE_INDICES]
        return np.asarray(
            np.where(standing, np.sum(np.square(posture), axis=1), 0.0),
            dtype=self._np_dtype,
        )

    def _reward_standing_leg_symmetry(self, ctx: RewardContext) -> np.ndarray:
        standing = self._standing_mask(ctx)
        posture_anchor = self._posture_anchor(standing)
        posture_diff = ctx.dof_pos[:, POSTURE_INDICES] - posture_anchor[:, POSTURE_INDICES]
        left = posture_diff[:, _REAL68_LEFT_POSTURE]
        right = posture_diff[:, _REAL68_RIGHT_POSTURE]
        mirrored_right = right * _REAL68_MIRROR_SIGNS
        symmetry = np.sum(np.square(left - mirrored_right), axis=1)
        return np.asarray(np.where(standing, symmetry, 0.0), dtype=self._np_dtype)

    def _reward_joint_pos_penalty(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        posture_target = self._command_target_posture(commands[:, 0])
        posture_anchor = self._posture_anchor(self._standing_mask(ctx))
        posture = (
            ctx.dof_pos[:, POSTURE_INDICES]
            - posture_anchor[:, POSTURE_INDICES]
            - posture_target
        )
        return np.asarray(np.linalg.norm(posture, axis=1), dtype=self._np_dtype)

    def _reward_joint_power(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        return np.asarray(
            np.sum(np.abs(ctx.dof_vel[:, POSTURE_INDICES] * torques[:, POSTURE_INDICES]), axis=1),
            dtype=self._np_dtype,
        )

    def _reward_net_progress(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        cmd_x = np.asarray(commands[:, 0], dtype=self._np_dtype)
        active = np.abs(cmd_x) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        signed_speed = np.asarray(
            self._forward_linvel(ctx.linvel) * np.sign(cmd_x),
            dtype=self._np_dtype,
        )
        signed_ratio = np.clip(
            signed_speed / np.maximum(np.abs(cmd_x), 1.0e-6),
            -1.0,
            1.0,
        )
        return np.asarray(np.where(active, signed_ratio, 0.0), dtype=self._np_dtype)

    def _reward_reverse_motion(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        cmd_x = np.asarray(commands[:, 0], dtype=self._np_dtype)
        active = np.abs(cmd_x) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        signed_speed = np.asarray(
            self._forward_linvel(ctx.linvel) * np.sign(cmd_x),
            dtype=self._np_dtype,
        )
        penalty = np.where(
            active,
            np.maximum(-signed_speed, 0.0) / np.maximum(np.abs(cmd_x), 1.0e-6),
            0.0,
        )
        return np.asarray(penalty, dtype=self._np_dtype)

    def _reward_vx_abs_net_gap(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        cmd_x = np.asarray(commands[:, 0], dtype=self._np_dtype)
        active = np.abs(cmd_x) > _REAL68_CURRICULUM_MIN_ABS_COMMAND
        linvel_forward = self._forward_linvel(ctx.linvel)
        signed_speed = np.asarray(linvel_forward * np.sign(cmd_x), dtype=self._np_dtype)
        positive_signed = np.maximum(signed_speed, 0.0)
        gap = np.maximum(np.abs(linvel_forward) - positive_signed, 0.0)
        normalized_gap = gap / np.maximum(np.abs(cmd_x), 1.0e-6)
        return np.asarray(np.where(active, normalized_gap, 0.0), dtype=self._np_dtype)

    def _reward_height_tracking(self, ctx: RewardContext) -> np.ndarray:
        targets = np.asarray(
            ctx.info.get(
                "height_commands", np.full((ctx.num_envs,), self._reward_cfg.base_height_target)
            ),
            dtype=self._np_dtype,
        )
        error = targets - ctx.base_height
        sigma = max(float(self._reward_cfg.height_tracking_sigma), 1e-6)
        return np.asarray(np.exp(-np.square(error) / sigma), dtype=self._np_dtype)

    def _reward_height_safety(self, ctx: RewardContext) -> np.ndarray:
        base_height = np.asarray(ctx.base_height, dtype=self._np_dtype)
        margin = max(float(self._reward_cfg.height_safety_margin), 1.0e-6)
        low_band = float(self._reward_cfg.min_base_height) + margin
        high_band = float(self._reward_cfg.max_base_height) - margin
        low_penalty = np.maximum(low_band - base_height, 0.0) / margin
        high_penalty = np.maximum(base_height - high_band, 0.0) / margin
        return np.asarray(low_penalty + high_penalty, dtype=self._np_dtype)

    def _reward_under_height(self, ctx: RewardContext) -> np.ndarray:
        targets = np.asarray(
            ctx.info.get(
                "height_commands", np.full((ctx.num_envs,), self._reward_cfg.base_height_target)
            ),
            dtype=self._np_dtype,
        )
        base_height = np.asarray(ctx.base_height, dtype=self._np_dtype)
        margin = max(float(self._reward_cfg.height_safety_margin), 1.0e-6)
        return np.asarray(np.maximum(targets - base_height, 0.0) / margin, dtype=self._np_dtype)

    def _reward_nonwheel_contact(self, ctx: RewardContext) -> np.ndarray:
        contact = np.asarray(np.max(self._nonwheel_contacts, axis=1), dtype=self._np_dtype)
        return contact

    def _reward_liangan5_contact(self, ctx: RewardContext) -> np.ndarray:
        liangan5_contact = np.asarray(
            np.mean(
                self._nonwheel_contacts[
                    :,
                    [
                        _REAL68_LEFT_LIANGAN5_CONTACT_INDEX,
                        _REAL68_RIGHT_LIANGAN5_CONTACT_INDEX,
                    ],
                ],
                axis=1,
            ),
            dtype=self._np_dtype,
        )
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(liangan5_contact * upright, dtype=self._np_dtype)

    def _reward_liangan5_contact_asymmetry(self, ctx: RewardContext) -> np.ndarray:
        left_contact = np.asarray(
            self._nonwheel_contacts[:, _REAL68_LEFT_LIANGAN5_CONTACT_INDEX], dtype=self._np_dtype
        )
        right_contact = np.asarray(
            self._nonwheel_contacts[:, _REAL68_RIGHT_LIANGAN5_CONTACT_INDEX], dtype=self._np_dtype
        )
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(np.abs(left_contact - right_contact) * upright, dtype=self._np_dtype)
