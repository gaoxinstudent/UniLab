from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
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
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.commands import zero_small_xy_commands
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
    Real68BaseCfg,
    Real68BaseEnv,
    WHEEL_CONTACT_SENSORS,
    WHEEL_INDICES,
    compute_real68_motor_ctrl,
    scalarize_contacts,
)


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
    scales: dict[str, float]
    tracking_sigma: float
    height_tracking_sigma: float = 0.015
    base_height_target: float = HOME_BASE_HEIGHT
    min_base_height: float = 0.18
    max_base_height: float = 0.38
    max_tilt_cos: float = 0.5
    only_positive_rewards: bool = False


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
    reward_config: RewardConfig | None = None
    sensor: Real68Sensor = field(default_factory=Real68Sensor)
    domain_rand: Real68DomainRandConfig = field(default_factory=Real68DomainRandConfig)


class Real68BalanceDomainRandomizationProvider(LocomotionDRProvider):
    def validate(self, env: Any, capabilities) -> None:
        validate_common_reset_randomization(env, capabilities)
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
        commands = self._sample_commands(env, num_reset)
        zero_small_xy_commands(commands, threshold=0.15)
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
            randomization=build_common_reset_randomization(env, num_reset),
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
        passive_pos = env.get_passive_dof_pos()[env_ids]
        passive_vel = env.get_passive_dof_vel()[env_ids]
        quat = env.get_imu_quat()[env_ids]
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
                quat,
                dof_pos,
                dof_vel,
                passive_pos,
                passive_vel,
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
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        if ctrl_range.shape != (NUM_ACTIONS, 2):
            raise ValueError(f"Real68 actuator ctrl_range must have shape ({NUM_ACTIONS}, 2)")
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)
        self._reward_cfg = cfg.reward_config
        self._last_motor_ctrl = np.zeros((num_envs, NUM_ACTIONS), dtype=self._np_dtype)
        self._last_dof_vel_for_acc = np.zeros((num_envs, NUM_ACTIONS), dtype=self._np_dtype)
        self._base_height = np.full((num_envs,), HOME_BASE_HEIGHT, dtype=self._np_dtype)
        self._wheel_contacts = np.zeros((num_envs, len(WHEEL_CONTACT_SENSORS)), dtype=self._np_dtype)
        self._nonwheel_contacts = np.zeros(
            (num_envs, len(NONWHEEL_CONTACT_SENSORS)), dtype=self._np_dtype
        )
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        self._init_reward_functions()
        self._init_domain_randomization(Real68BalanceDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 29, "critic": 65}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
        if dof_vel.shape[0] == self._num_envs:
            self._last_dof_vel_for_acc[env_ids] = dof_vel[env_ids]
        return obs, info

    def sample_height_commands(self, num_reset: int) -> np.ndarray:
        low, high = self._cfg.height_command.range
        return np.asarray(np.random.uniform(low, high, size=(num_reset,)), dtype=self._np_dtype)

    def get_accel(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.accel), dtype=self._np_dtype)

    def get_imu_quat(self) -> np.ndarray:
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.quat), dtype=self._np_dtype)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(actions, -self._cfg.control_config.clip_actions, self._cfg.control_config.clip_actions),
            dtype=self._np_dtype,
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped_actions))
        state.info["current_actions"] = clipped_actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped_actions
        )
        targets = np.zeros_like(exec_actions, dtype=self._np_dtype)
        targets[:, HIP_INDICES] = exec_actions[:, HIP_INDICES] * self._cfg.control_config.hip_velocity_scale
        targets[:, WHEEL_INDICES] = exec_actions[:, WHEEL_INDICES] * self._cfg.control_config.wheel_velocity_scale
        targets[:, CALF_INDICES] = (
            self.default_angles[CALF_INDICES]
            + exec_actions[:, CALF_INDICES] * self._cfg.control_config.calf_action_scale
        )
        return targets

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        active_pos = self.get_dof_pos()
        active_vel = self.get_dof_vel()
        hip_kd = np.full((self._num_envs, len(HIP_INDICES)), self._cfg.control_config.hip_kd, dtype=np.float64)
        wheel_kd = np.full(
            (self._num_envs, len(WHEEL_INDICES)), self._cfg.control_config.wheel_kd, dtype=np.float64
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
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "orientation": rewards.orientation,
            "action_rate": rewards.action_rate,
            "alive": rewards.alive,
            "torques": self._reward_torques_l2,
            "wheel_vel": self._reward_wheel_vel,
            "posture": self._reward_posture,
            "height_tracking": self._reward_height_tracking,
            "nonwheel_contact": self._reward_nonwheel_contact,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = np.asarray(self._backend.get_sensor_data(self._cfg.sensor.gravity), dtype=self._np_dtype)
        accel = self.get_accel()
        quat = self.get_imu_quat()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        passive_pos = self.get_passive_dof_pos()
        passive_vel = self.get_passive_dof_vel()
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
            quat,
            dof_pos,
            dof_vel,
            passive_pos,
            passive_vel,
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

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
        quat: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        passive_pos: np.ndarray,
        passive_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        posture_diff = dof_pos[:, POSTURE_INDICES] - self.default_angles[POSTURE_INDICES]
        posture_vel = dof_vel[:, POSTURE_INDICES]
        wheel_vel = dof_vel[:, WHEEL_INDICES]
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_accel = self._obs_noise(accel, noise_cfg.scale_accel)
        noisy_posture_diff = self._obs_noise(posture_diff, noise_cfg.scale_joint_angle)
        noisy_posture_vel = self._obs_noise(posture_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_joint_vel)
        base_height = self._reward_base_height_values(gyro.shape[0])
        height_commands = np.asarray(
            info.get("height_commands", np.full((gyro.shape[0],), self._reward_cfg.base_height_target)),
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
                passive_pos,
                passive_vel,
                motor_torque,
                quat,
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
        commands = np.asarray(info.get("commands", np.zeros((self._num_envs, 3))), dtype=self._np_dtype)
        height_commands = np.asarray(
            info.get(
                "height_commands",
                np.full((self._num_envs,), self._reward_cfg.base_height_target, dtype=self._np_dtype),
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
                low = np.asarray(self._cfg.commands.vel_limit[0], dtype=self._np_dtype)
                high = np.asarray(self._cfg.commands.vel_limit[1], dtype=self._np_dtype)
                sampled = np.asarray(
                    np.random.uniform(low=low, high=high, size=(num_resample, 3)),
                    dtype=self._np_dtype,
                )
                sampled[:, 1] = 0.0
                zero_small_xy_commands(sampled, threshold=0.15)
                commands[resample_mask] = sampled
                height_commands[resample_mask] = self.sample_height_commands(num_resample)
        commands[:, 1] = 0.0
        info["commands"] = commands
        info["height_commands"] = height_commands

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt, dtype=self._np_dtype)
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

    def _reward_wheel_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.asarray(np.sum(np.square(ctx.dof_vel[:, WHEEL_INDICES]), axis=1), dtype=self._np_dtype)

    def _reward_posture(self, ctx: RewardContext) -> np.ndarray:
        posture = ctx.dof_pos[:, POSTURE_INDICES] - DEFAULT_ACTIVE_ANGLES[POSTURE_INDICES]
        return np.asarray(np.sum(np.square(posture), axis=1), dtype=self._np_dtype)

    def _reward_height_tracking(self, ctx: RewardContext) -> np.ndarray:
        targets = np.asarray(
            ctx.info.get("height_commands", np.full((ctx.num_envs,), self._reward_cfg.base_height_target)),
            dtype=self._np_dtype,
        )
        error = targets - ctx.base_height
        sigma = max(float(self._reward_cfg.height_tracking_sigma), 1e-6)
        return np.asarray(np.exp(-np.square(error) / sigma), dtype=self._np_dtype)

    def _reward_nonwheel_contact(self, ctx: RewardContext) -> np.ndarray:
        contact = np.asarray(np.max(self._nonwheel_contacts, axis=1), dtype=self._np_dtype)
        return contact
