from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)

ACTIVE_JOINT_POS_SENSORS: tuple[str, ...] = (
    "left_hip_bigleg_joint_pos",
    "left_wheel_joint_pos",
    "left_calf_smallleg_joint_pos",
    "right_hip_bigleg_joint_pos",
    "right_wheel_joint_pos",
    "right_calf_smallleg_joint_pos",
)
ACTIVE_JOINT_VEL_SENSORS: tuple[str, ...] = (
    "left_hip_bigleg_joint_vel",
    "left_wheel_joint_vel",
    "left_calf_smallleg_joint_vel",
    "right_hip_bigleg_joint_vel",
    "right_wheel_joint_vel",
    "right_calf_smallleg_joint_vel",
)
PASSIVE_JOINT_POS_SENSORS: tuple[str, ...] = (
    "left_calf_smallleg_liangan_joint_pos",
    "left_chuanliangan2_joint_pos",
    "left_chuanliangan3_joint_pos",
    "left_chuanliangan5_joint_pos",
    "right_calf_smallleg_liangan_joint_pos",
    "right_liangan2_joint_pos",
    "right_liangan3_joint_pos",
    "right_liangan5_joint_pos",
)
PASSIVE_JOINT_VEL_SENSORS: tuple[str, ...] = (
    "left_calf_smallleg_liangan_joint_vel",
    "left_chuanliangan2_joint_vel",
    "left_chuanliangan3_joint_vel",
    "left_chuanliangan5_joint_vel",
    "right_calf_smallleg_liangan_joint_vel",
    "right_liangan2_joint_vel",
    "right_liangan3_joint_vel",
    "right_liangan5_joint_vel",
)
WHEEL_CONTACT_SENSORS: tuple[str, ...] = ("left_wheel_contact", "right_wheel_contact")
NONWHEEL_CONTACT_SENSORS: tuple[str, ...] = (
    "base_link_contact",
    "left_chuanliangan3_contact",
    "left_chuanliangan5_contact",
    "right_liangan3_contact",
    "right_liangan5_contact",
)

NUM_ACTIONS = len(ACTIVE_JOINT_POS_SENSORS)
HIP_INDICES = np.asarray([0, 3], dtype=np.int32)
WHEEL_INDICES = np.asarray([1, 4], dtype=np.int32)
CALF_INDICES = np.asarray([2, 5], dtype=np.int32)
POSTURE_INDICES = np.asarray([0, 2, 3, 5], dtype=np.int32)
SYMMETRIC_STANDING_ACTIVE_ANGLES = np.asarray(
    [
        0.2009805,
        -0.045959,
        -0.517042,
        -0.2009805,
        -0.076009,
        0.517042,
    ],
    dtype=np.float64,
)
DEFAULT_ACTIVE_ANGLES = SYMMETRIC_STANDING_ACTIVE_ANGLES.copy()
HOME_BASE_HEIGHT = 0.257282


@dataclass
class NoiseConfig(BaseNoiseConfig):
    scale_accel: float = 0.2


@dataclass
class ControlConfig(ControlConfigBase):
    clip_actions: float = 1.0
    hip_velocity_scale: float = 6.0
    wheel_velocity_scale: float = 20.0
    calf_action_scale: float = 0.35
    hip_kd: float = 2.0
    wheel_kd: float = 1.0
    calf_kp: float = 45.0
    calf_kd: float = 3.0


@dataclass
class Asset:
    base_name = "base_link"
    ground = "floor"


@dataclass
class Real68BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.001
    ctrl_dt: float = 0.02


def _stack_sensors(backend, names: tuple[str, ...], *, dtype: np.dtype | type) -> np.ndarray:
    return np.asarray(backend.get_sensor_data_batch(names), dtype=dtype)


def scalarize_contacts(backend, names: tuple[str, ...], *, dtype: np.dtype | type) -> np.ndarray:
    values = backend.get_sensor_data_batch(names)
    values = np.asarray(values.reshape(values.shape[0], -1)[:, : len(names)], dtype=dtype)
    return values


def compute_real68_motor_ctrl(
    policy_ctrl: np.ndarray,
    active_pos: np.ndarray,
    active_vel: np.ndarray,
    *,
    hip_kd: np.ndarray,
    wheel_kd: np.ndarray,
    calf_kp: np.ndarray,
    calf_kd: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    out[:, HIP_INDICES] = hip_kd * (policy_ctrl[:, HIP_INDICES] - active_vel[:, HIP_INDICES])
    out[:, WHEEL_INDICES] = (
        wheel_kd * (policy_ctrl[:, WHEEL_INDICES] - active_vel[:, WHEEL_INDICES])
    )
    out[:, CALF_INDICES] = calf_kp * (
        policy_ctrl[:, CALF_INDICES] - active_pos[:, CALF_INDICES]
    ) - calf_kd * active_vel[:, CALF_INDICES]
    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out


class Real68BaseEnv(LocomotionBaseEnv):
    _cfg: Real68BaseCfg

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.default_angles = np.asarray(DEFAULT_ACTIVE_ANGLES, dtype=self._init_qpos.dtype)

    def get_dof_pos(self) -> np.ndarray:
        return _stack_sensors(self._backend, ACTIVE_JOINT_POS_SENSORS, dtype=self.default_angles.dtype)

    def get_dof_vel(self) -> np.ndarray:
        return _stack_sensors(self._backend, ACTIVE_JOINT_VEL_SENSORS, dtype=self.default_angles.dtype)

    def get_passive_dof_pos(self) -> np.ndarray:
        return _stack_sensors(
            self._backend, PASSIVE_JOINT_POS_SENSORS, dtype=self.default_angles.dtype
        )

    def get_passive_dof_vel(self) -> np.ndarray:
        return _stack_sensors(
            self._backend, PASSIVE_JOINT_VEL_SENSORS, dtype=self.default_angles.dtype
        )
