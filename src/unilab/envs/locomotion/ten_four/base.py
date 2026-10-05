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
    "left_rear1_joint_pos",
    "left_wheel_joint_pos",
    "left_front1_joint_pos",
    "right_rear1_joint_pos",
    "right_wheel_joint_pos",
    "right_front1_joint_pos",
)
ACTIVE_JOINT_NAMES: tuple[str, ...] = (
    "left_rear1_joint",
    "left_wheel_joint",
    "left_front1_joint",
    "right_rear1_joint",
    "right_wheel_joint",
    "right_front1_joint",
)
ACTIVE_JOINT_VEL_SENSORS: tuple[str, ...] = (
    "left_rear1_joint_vel",
    "left_wheel_joint_vel",
    "left_front1_joint_vel",
    "right_rear1_joint_vel",
    "right_wheel_joint_vel",
    "right_front1_joint_vel",
)
PASSIVE_JOINT_POS_SENSORS: tuple[str, ...] = (
    "left_rear2_joint_pos",
    "left_front2_joint_pos",
    "left_front3_joint_pos",
    "left_front4_joint_pos",
    "right_rear2_joint_pos",
    "right_front2_joint_pos",
    "right_front3_joint_pos",
    "right_front4_joint_pos",
)
PASSIVE_JOINT_VEL_SENSORS: tuple[str, ...] = (
    "left_rear2_joint_vel",
    "left_front2_joint_vel",
    "left_front3_joint_vel",
    "left_front4_joint_vel",
    "right_rear2_joint_vel",
    "right_front2_joint_vel",
    "right_front3_joint_vel",
    "right_front4_joint_vel",
)
WHEEL_CONTACT_SENSORS: tuple[str, ...] = ("left_wheel_contact", "right_wheel_contact")
NONWHEEL_CONTACT_SENSORS: tuple[str, ...] = (
    "base_link_contact",
    "left_front_guide_contact",
    "left_front1_guide_contact",
    "left_bottom1_guide_contact",
    "left_bottom2_guide_contact",
    "left_bottom3_guide_contact",
    "left_bottom4_guide_contact",
    "right_front_guide_contact",
    "right_front1_guide_contact",
    "right_bottom1_guide_contact",
    "right_bottom2_guide_contact",
    "right_bottom3_guide_contact",
    "right_bottom4_guide_contact",
)
# Keep the critic contact slice checkpoint-compatible while internal reward and
# termination logic observes every non-wheel collision link.
NONWHEEL_CONTACT_OBSERVATION_SENSORS: tuple[str, ...] = (
    "base_link_contact",
    "left_bottom3_guide_contact",
    "left_bottom4_guide_contact",
    "right_bottom3_guide_contact",
    "right_bottom4_guide_contact",
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
# Base height with all joints at zero (the physical power-on / near-zero startup
# state): wheels on the ground, fourbar loop closed with passive joints ~= 0.
# Measured in simulation (see the sim2real startup analysis); the standing
# bootstrap raises the body from here up to HOME_BASE_HEIGHT.
COLD_START_BASE_HEIGHT = 0.192


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
class TenFourBaseCfg(LocomotionBaseCfg):
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


def compute_ten_four_motor_ctrl(
    policy_ctrl: np.ndarray,
    active_pos: np.ndarray,
    active_vel: np.ndarray,
    *,
    hip_kd: np.ndarray,
    wheel_kd: np.ndarray,
    calf_kp: np.ndarray,
    calf_kd: np.ndarray,
    motor_strength: np.ndarray | None,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    out[:, HIP_INDICES] = hip_kd * (policy_ctrl[:, HIP_INDICES] - active_vel[:, HIP_INDICES])
    out[:, WHEEL_INDICES] = wheel_kd * (
        policy_ctrl[:, WHEEL_INDICES] - active_vel[:, WHEEL_INDICES]
    )
    out[:, CALF_INDICES] = (
        calf_kp * (policy_ctrl[:, CALF_INDICES] - active_pos[:, CALF_INDICES])
        - calf_kd * active_vel[:, CALF_INDICES]
    )
    if motor_strength is not None:
        out *= motor_strength
    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out


class TenFourBaseEnv(LocomotionBaseEnv):
    _cfg: TenFourBaseCfg

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
        return _stack_sensors(
            self._backend, ACTIVE_JOINT_POS_SENSORS, dtype=self.default_angles.dtype
        )

    def get_dof_vel(self) -> np.ndarray:
        return _stack_sensors(
            self._backend, ACTIVE_JOINT_VEL_SENSORS, dtype=self.default_angles.dtype
        )

    def get_passive_dof_pos(self) -> np.ndarray:
        return _stack_sensors(
            self._backend, PASSIVE_JOINT_POS_SENSORS, dtype=self.default_angles.dtype
        )

    def get_passive_dof_vel(self) -> np.ndarray:
        return _stack_sensors(
            self._backend, PASSIVE_JOINT_VEL_SENSORS, dtype=self.default_angles.dtype
        )
