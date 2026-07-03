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
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
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
    yaw_step: float = 0.02
    update_interval_logs: int = 6
    min_speed_ratio: float = 0.45
    max_vx_error: float = 0.25
    max_wz_error: float = 0.9
    min_segment_count: int = 64
    yaw_unlock_vx_progress: float = 0.6
    reverse_unlock_vx_progress: float = 0.7
    standing_bootstrap_enabled: bool = False
    standing_bootstrap_min_segments: int = 64
    standing_bootstrap_min_segment_steps: int = 80
    standing_bootstrap_max_wz_error: float = 0.35
    standing_bootstrap_max_nonwheel_contact: float = 0.02
    standing_prob_initial: float = 0.0
    standing_prob_final: float = 0.0
    standing_decay_vx_progress: float = 1.0


@dataclass
class HeightCommandConfig:
    range: list[float] = field(default_factory=lambda: [0.25, 0.32])


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

    scales: dict[str, float]
    tracking_sigma: float
    height_tracking_sigma: float = 0.015
    base_height_target: float = HOME_BASE_HEIGHT
    min_base_height: float = 0.18
    max_base_height: float = 0.38
    max_tilt_cos: float = 0.5
    only_positive_rewards: bool = False
    command_lean: CommandLeanConfig = field(default_factory=CommandLeanConfig)


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
        height_commands = env.sample_height_commands(num_reset)
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
        self._curriculum_vx_speed_ratio_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_vx_error_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._curriculum_yaw_count = np.zeros((_REAL68_CURRICULUM_NUM_BINS,), dtype=np.int32)
        self._curriculum_yaw_error_sum = np.zeros(
            (_REAL68_CURRICULUM_NUM_BINS,), dtype=self._np_dtype
        )
        self._standing_segment_count = 0
        self._standing_segment_steps_sum = 0.0
        self._standing_segment_wz_error_sum = 0.0
        self._standing_segment_nonwheel_contact_sum = 0.0
        self._last_standing_bootstrap_eval: dict[str, float] | None = None
        self._segment_cmd_x = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_cmd_yaw = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_abs_vx_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_abs_wz_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_vx_error_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_wz_error_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_nonwheel_contact_sum = np.zeros((num_envs,), dtype=self._np_dtype)
        self._segment_steps = np.zeros((num_envs,), dtype=np.int32)
        self._curriculum_recorded_segments = 0

    def _make_dr_provider(self) -> Real68BalanceDomainRandomizationProvider:
        return Real68BalanceDomainRandomizationProvider()

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
        self._reset_command_segments(env_ids, commands)
        return obs, info

    def sample_height_commands(self, num_reset: int) -> np.ndarray:
        low, high = self._cfg.height_command.range
        return np.asarray(np.random.uniform(low, high, size=(num_reset,)), dtype=self._np_dtype)

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
        if self._yaw_curriculum_locked():
            commands[:, 2] = 0.0
        zero_small_xy_commands(commands, threshold=0.08)
        standing_prob = self._standing_command_probability()
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_samples,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        return commands

    def _yaw_curriculum_locked(self) -> bool:
        ccfg = self._cfg.command_curriculum
        return bool(
            ccfg.enabled and self._command_curriculum_vx_progress < float(ccfg.yaw_unlock_vx_progress)
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
                and standing_eval["wz_error"] <= float(ccfg.standing_bootstrap_max_wz_error)
                and standing_eval["nonwheel_contact"] <= float(
                    ccfg.standing_bootstrap_max_nonwheel_contact
                )
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
        vx_ready = bool(
            vx_eval is not None
            and vx_eval["speed_ratio"] >= float(ccfg.min_speed_ratio)
            and vx_eval["vx_error"] <= float(ccfg.max_vx_error)
        )
        progressed = False
        if vx_ready and self._command_curriculum_vx_progress < 1.0:
            self._command_curriculum_vx_progress = min(
                1.0, self._command_curriculum_vx_progress + float(ccfg.vx_step)
            )
            progressed = True
        yaw_eval = self._curriculum_yaw_eval()
        yaw_ready = bool(
            self._command_curriculum_vx_progress >= float(ccfg.yaw_unlock_vx_progress)
            and yaw_eval is not None
            and yaw_eval["wz_error"] <= float(ccfg.max_wz_error)
        )
        if yaw_ready and self._command_curriculum_yaw_progress < 1.0:
            self._command_curriculum_yaw_progress = min(
                1.0, self._command_curriculum_yaw_progress + float(ccfg.yaw_step)
            )
            progressed = True
        if progressed:
            self._refresh_command_curriculum_limits()
            self._reset_command_curriculum_stats()

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
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": rewards.under_speed,
            "yaw_rate_when_uncommanded": rewards.yaw_rate_when_uncommanded,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "orientation": self._reward_orientation,
            "action_rate": rewards.action_rate,
            "alive": rewards.alive,
            "torques": self._reward_torques_l2,
            "wheel_vel": self._reward_wheel_vel,
            "posture": self._reward_posture,
            "leg_symmetry": self._reward_leg_symmetry,
            "height_tracking": self._reward_height_tracking,
            "joint_pos_penalty": self._reward_joint_pos_penalty,
            "joint_power": self._reward_joint_power,
            "nonwheel_contact": self._reward_nonwheel_contact,
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
        terminated = self._compute_terminated(gravity)
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
        linvel_x = np.asarray(linvel[:, 0], dtype=self._np_dtype)
        gyro_z = np.asarray(gyro[:, 2], dtype=self._np_dtype)
        nonwheel_contacts = np.asarray(
            info.get(
                "nonwheel_contacts",
                np.zeros((self._num_envs, len(NONWHEEL_CONTACT_SENSORS)), dtype=self._np_dtype),
            ),
            dtype=self._np_dtype,
        )
        nonwheel_contact_frac = np.mean(nonwheel_contacts > 0.0, axis=1)
        self._segment_cmd_x[:] = commands[:, 0]
        self._segment_cmd_yaw[:] = commands[:, 2]
        self._segment_abs_vx_sum += np.abs(linvel_x)
        self._segment_abs_wz_sum += np.abs(gyro_z)
        self._segment_vx_error_sum += np.abs(commands[:, 0] - linvel_x)
        self._segment_wz_error_sum += np.abs(commands[:, 2] - gyro_z)
        self._segment_nonwheel_contact_sum += nonwheel_contact_frac
        self._segment_steps += 1

    def _reset_command_segments(self, env_ids: np.ndarray, commands: np.ndarray) -> None:
        if env_ids.size == 0:
            return
        self._segment_cmd_x[env_ids] = commands[:, 0]
        self._segment_cmd_yaw[env_ids] = commands[:, 2]
        self._segment_abs_vx_sum[env_ids] = 0.0
        self._segment_abs_wz_sum[env_ids] = 0.0
        self._segment_vx_error_sum[env_ids] = 0.0
        self._segment_wz_error_sum[env_ids] = 0.0
        self._segment_nonwheel_contact_sum[env_ids] = 0.0
        self._segment_steps[env_ids] = 0

    def _reset_command_curriculum_stats(self) -> None:
        self._curriculum_vx_count.fill(0)
        self._curriculum_vx_speed_ratio_sum.fill(0.0)
        self._curriculum_vx_error_sum.fill(0.0)
        self._curriculum_yaw_count.fill(0)
        self._curriculum_yaw_error_sum.fill(0.0)
        self._reset_standing_bootstrap_stats()
        self._last_standing_bootstrap_eval = None
        self._curriculum_recorded_segments = 0

    def _reset_standing_bootstrap_stats(self) -> None:
        self._standing_segment_count = 0
        self._standing_segment_steps_sum = 0.0
        self._standing_segment_wz_error_sum = 0.0
        self._standing_segment_nonwheel_contact_sum = 0.0

    def _curriculum_abs_limit_x(self) -> float:
        return float(max(abs(self._command_curriculum_low[0]), abs(self._command_curriculum_high[0])))

    def _curriculum_abs_limit_yaw(self) -> float:
        return float(max(abs(self._command_curriculum_low[2]), abs(self._command_curriculum_high[2])))

    def _curriculum_bin_index(self, magnitude: float, max_abs: float) -> int | None:
        if magnitude <= _REAL68_CURRICULUM_MIN_ABS_COMMAND or max_abs <= _REAL68_CURRICULUM_MIN_ABS_COMMAND:
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
        mean_abs_wz: float,
        vx_error: float,
        wz_error: float,
        mean_nonwheel_contact: float,
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
            self._standing_segment_wz_error_sum += mean_abs_wz
            self._standing_segment_nonwheel_contact_sum += mean_nonwheel_contact

        vx_max_abs = self._curriculum_abs_limit_x()
        vx_index = self._curriculum_bin_index(abs_cmd_x, vx_max_abs)
        if vx_index is not None:
            self._curriculum_vx_count[vx_index] += 1
            self._curriculum_vx_speed_ratio_sum[vx_index] += mean_abs_vx / max(abs_cmd_x, 1.0e-6)
            self._curriculum_vx_error_sum[vx_index] += vx_error

        yaw_max_abs = self._curriculum_abs_limit_yaw()
        yaw_index = self._curriculum_bin_index(abs_cmd_yaw, yaw_max_abs)
        if yaw_index is not None:
            self._curriculum_yaw_count[yaw_index] += 1
            self._curriculum_yaw_error_sum[yaw_index] += wz_error

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
            mean_abs_wz = float(self._segment_abs_wz_sum[env_id] / max(steps, 1))
            vx_error = float(self._segment_vx_error_sum[env_id] / max(steps, 1))
            wz_error = float(self._segment_wz_error_sum[env_id] / max(steps, 1))
            mean_nonwheel_contact = float(
                self._segment_nonwheel_contact_sum[env_id] / max(steps, 1)
            )
            self._record_command_segment_stats(
                cmd_x=float(self._segment_cmd_x[env_id]),
                cmd_yaw=float(self._segment_cmd_yaw[env_id]),
                mean_abs_vx=mean_abs_vx,
                mean_abs_wz=mean_abs_wz,
                vx_error=vx_error,
                wz_error=wz_error,
                mean_nonwheel_contact=mean_nonwheel_contact,
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
                "speed_ratio": float(self._curriculum_vx_speed_ratio_sum[index] / count),
                "vx_error": float(self._curriculum_vx_error_sum[index] / count),
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
        linvel_x = linvel[:, 0]
        gyro_z = gyro[:, 2]
        log["metrics/linvel_x"] = float(np.mean(linvel_x))
        log["metrics/gyro_z"] = float(np.mean(gyro_z))
        log["metrics/cmd_x"] = float(np.mean(cmd_x))
        log["metrics/cmd_yaw"] = float(np.mean(cmd_yaw))
        log["metrics/vx_error"] = float(np.mean(np.abs(cmd_x - linvel_x)))
        log["metrics/wz_error"] = float(np.mean(np.abs(cmd_yaw - gyro_z)))
        log["metrics/mean_abs_vx"] = float(np.mean(np.abs(linvel_x)))
        log["metrics/mean_abs_wz"] = float(np.mean(np.abs(gyro_z)))
        log["metrics/mean_abs_cmd_x"] = float(np.mean(np.abs(cmd_x)))
        log["metrics/mean_abs_cmd_yaw"] = float(np.mean(np.abs(cmd_yaw)))
        log["metrics/commanded_nonzero_frac"] = float(np.mean(np.abs(cmd_x) > 0.05))
        log["metrics/commanded_yaw_nonzero_frac"] = float(np.mean(np.abs(cmd_yaw) > 0.05))
        log["metrics/target_lean_gx"] = float(np.mean(self._command_lean_gravity_target(cmd_x)))
        actions = np.asarray(
            info.get("current_actions", np.zeros((self._num_envs, self._num_action))),
            dtype=self._np_dtype,
        )
        wheel_actions = actions[:, WHEEL_INDICES]
        log["metrics/mean_wheel_action"] = float(np.mean(wheel_actions))
        log["metrics/mean_left_wheel_action"] = float(np.mean(actions[:, WHEEL_INDICES[0]]))
        log["metrics/mean_right_wheel_action"] = float(np.mean(actions[:, WHEEL_INDICES[1]]))
        log["metrics/mean_abs_wheel_action"] = float(np.mean(np.abs(wheel_actions)))

    def _write_command_curriculum_metrics(self, log: dict[str, Any]) -> None:
        vx_eval = self._curriculum_vx_eval()
        yaw_eval = self._curriculum_yaw_eval()
        log["command_curriculum/progress"] = float(self._command_curriculum_vx_progress)
        log["command_curriculum/vx_progress"] = float(self._command_curriculum_vx_progress)
        log["command_curriculum/yaw_progress"] = float(self._command_curriculum_yaw_progress)
        log["command_curriculum/speed_ratio"] = float(
            0.0 if vx_eval is None else vx_eval["speed_ratio"]
        )
        log["command_curriculum/eval_vx_error"] = float(0.0 if vx_eval is None else vx_eval["vx_error"])
        log["command_curriculum/eval_wz_error"] = float(0.0 if yaw_eval is None else yaw_eval["wz_error"])
        log["command_curriculum/eval_bucket_high_vx"] = float(
            0.0 if vx_eval is None else vx_eval["bucket_upper"]
        )
        log["command_curriculum/eval_bucket_high_wz"] = float(
            0.0 if yaw_eval is None else yaw_eval["bucket_upper"]
        )
        log["command_curriculum/eval_count_vx"] = float(0.0 if vx_eval is None else vx_eval["count"])
        log["command_curriculum/eval_count_wz"] = float(0.0 if yaw_eval is None else yaw_eval["count"])
        log["command_curriculum/segments_recorded"] = float(self._curriculum_recorded_segments)
        log["command_curriculum/standing_prob"] = float(self._standing_command_probability())
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
        log["command_curriculum/standing_eval_wz_error"] = float(
            0.0 if standing_eval is None else standing_eval["wz_error"]
        )
        log["command_curriculum/standing_eval_nonwheel_contact"] = float(
            0.0 if standing_eval is None else standing_eval["nonwheel_contact"]
        )

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        base_z = self._reward_base_height_values(gravity.shape[0])
        return np.asarray(
            (gravity[:, 2] <= self._reward_cfg.max_tilt_cos)
            | (base_z <= self._reward_cfg.min_base_height)
            | (base_z >= self._reward_cfg.max_base_height),
            dtype=bool,
        )

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
                height_commands[resample_mask] = self.sample_height_commands(num_resample)
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

    def _reward_orientation(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        target_gx = self._command_lean_gravity_target(commands[:, 0])
        gravity_x_error = np.square(ctx.gravity[:, 0] - target_gx)
        gravity_y_error = np.square(ctx.gravity[:, 1])
        return np.asarray(gravity_x_error + gravity_y_error, dtype=self._np_dtype)

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
        posture = (
            ctx.dof_pos[:, POSTURE_INDICES]
            - DEFAULT_ACTIVE_ANGLES[POSTURE_INDICES]
            - posture_target
        )
        return np.asarray(np.sum(np.square(posture), axis=1), dtype=self._np_dtype)

    def _reward_leg_symmetry(self, ctx: RewardContext) -> np.ndarray:
        posture_diff = ctx.dof_pos[:, POSTURE_INDICES] - DEFAULT_ACTIVE_ANGLES[POSTURE_INDICES]
        left = posture_diff[:, _REAL68_LEFT_POSTURE]
        right = posture_diff[:, _REAL68_RIGHT_POSTURE]
        mirrored_right = right * _REAL68_MIRROR_SIGNS
        symmetry = np.sum(np.square(left - mirrored_right), axis=1)
        upright = rewards.upright_scale(ctx.gravity, ctx.num_envs)
        return np.asarray(symmetry * upright, dtype=self._np_dtype)

    def _reward_joint_pos_penalty(self, ctx: RewardContext) -> np.ndarray:
        commands = np.asarray(
            ctx.info.get("commands", np.zeros((ctx.num_envs, 3), dtype=self._np_dtype)),
            dtype=self._np_dtype,
        )
        posture_target = self._command_target_posture(commands[:, 0])
        posture = (
            ctx.dof_pos[:, POSTURE_INDICES]
            - DEFAULT_ACTIVE_ANGLES[POSTURE_INDICES]
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

    def _reward_nonwheel_contact(self, ctx: RewardContext) -> np.ndarray:
        contact = np.asarray(np.max(self._nonwheel_contacts, axis=1), dtype=self._np_dtype)
        return contact
