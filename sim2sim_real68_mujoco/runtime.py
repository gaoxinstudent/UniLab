from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort

from .config import Sim2SimConfig
from .math_utils import quat_from_euler_xyz, quat_mul


@dataclass
class SensorSlice:
    adr: int
    dim: int


class SensorAccessor:
    def __init__(self, model: mujoco.MjModel):
        self._slices: dict[str, SensorSlice] = {}
        for sensor_id in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            if not name:
                continue
            self._slices[name] = SensorSlice(
                adr=int(model.sensor_adr[sensor_id]),
                dim=int(model.sensor_dim[sensor_id]),
            )

    def read(self, data: mujoco.MjData, name: str) -> np.ndarray:
        sl = self._slices[name]
        return np.asarray(data.sensordata[sl.adr : sl.adr + sl.dim], dtype=np.float64)


class HfieldSampler:
    def __init__(self, model: mujoco.MjModel, geom_name: str):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Geom '{geom_name}' not found")
        hfield_id = int(model.geom_dataid[geom_id])
        if hfield_id < 0:
            raise ValueError(f"Geom '{geom_name}' is not bound to a heightfield")
        self._geom_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64).copy()
        adr = int(model.hfield_adr[hfield_id])
        nrow = int(model.hfield_nrow[hfield_id])
        ncol = int(model.hfield_ncol[hfield_id])
        self._data = np.asarray(model.hfield_data[adr : adr + nrow * ncol], dtype=np.float64).reshape(
            nrow, ncol
        )
        self._half_x = float(model.hfield_size[hfield_id, 0])
        self._half_y = float(model.hfield_size[hfield_id, 1])
        self._z_top = float(model.hfield_size[hfield_id, 2])
        self._nrow = nrow
        self._ncol = ncol

    def sample(self, xy_world: np.ndarray) -> float:
        x = float(xy_world[0] - self._geom_pos[0])
        y = float(xy_world[1] - self._geom_pos[1])
        col = int(np.rint((x + self._half_x) / (2.0 * self._half_x) * (self._ncol - 1)))
        row = int(np.rint((y + self._half_y) / (2.0 * self._half_y) * (self._nrow - 1)))
        col = int(np.clip(col, 0, self._ncol - 1))
        row = int(np.clip(row, 0, self._nrow - 1))
        return float(self._geom_pos[2] + self._data[row, col] * self._z_top)

    def sample_many_max(self, xy_world: np.ndarray) -> float:
        points = np.asarray(xy_world, dtype=np.float64).reshape(-1, 2)
        return max(self.sample(point) for point in points)


class KeyboardCommander:
    def __init__(self, command_limits: np.ndarray):
        self.command = np.zeros((3,), dtype=np.float64)
        self._limits = command_limits.astype(np.float64)
        self.paused = False
        self.single_step = False
        self.reset_requested = False
        self.next_terrain_requested = False
        self.follow_camera = True

    def _clip(self) -> None:
        self.command[:] = np.clip(self.command, self._limits[0], self._limits[1])
        self.command[1] = 0.0

    def handle(self, keycode: int) -> None:
        if keycode in (ord("w"), ord("W")):
            self.command[0] += 0.05
        elif keycode in (ord("s"), ord("S")):
            self.command[0] -= 0.05
        elif keycode in (ord("a"), ord("A")):
            self.command[2] += 0.05
        elif keycode in (ord("d"), ord("D")):
            self.command[2] -= 0.05
        elif keycode in (ord(" "),):
            self.command[:] = 0.0
        elif keycode in (ord("p"), ord("P")):
            self.paused = not self.paused
            print(f"[sim2sim] {'paused' if self.paused else 'resumed'}")
        elif keycode in (ord("n"), ord("N")):
            self.single_step = True
        elif keycode in (ord("r"), ord("R")):
            self.reset_requested = True
        elif keycode in (ord("t"), ord("T")):
            self.next_terrain_requested = True
        elif keycode in (ord("f"), ord("F")):
            self.follow_camera = not self.follow_camera
            print(f"[sim2sim] follow_camera={self.follow_camera}")
        elif keycode in (ord("1"),):
            self.command[:] = np.asarray([0.2, 0.0, 0.0], dtype=np.float64)
        elif keycode in (ord("2"),):
            self.command[:] = np.asarray([0.5, 0.0, 0.0], dtype=np.float64)
        elif keycode in (ord("3"),):
            self.command[:] = np.asarray([0.8, 0.0, 0.0], dtype=np.float64)
        self._clip()
        if keycode not in (ord("p"), ord("P"), ord("n"), ord("N"), ord("f"), ord("F")):
            print(f"[sim2sim] command = vx={self.command[0]:+.2f}, wz={self.command[2]:+.2f}")


class Real68Sim2Sim:
    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        command_override: np.ndarray | None = None,
        random_yaw: bool = False,
        auto_reset: bool = False,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        self.cfg = Sim2SimConfig.load(self.bundle_dir / "sim2sim_config.json")
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.resolve_path("scene_file")))
        self.data = mujoco.MjData(self.model)
        self.sensors = SensorAccessor(self.model)
        self.hfield = HfieldSampler(self.model, self.cfg["terrain"]["geom_name"])
        self.terrain_origins = np.load(self.cfg.resolve_path("terrain_origins_file"))
        self.session = ort.InferenceSession(
            str(self.cfg.resolve_path("policy_file")),
            providers=["CPUExecutionProvider"],
        )
        self.obs_name = self.session.get_inputs()[0].name
        self.action_name = self.session.get_outputs()[0].name
        self.control_cfg = self.cfg["control_config"]
        self.default_angles = np.asarray(self.cfg["default_active_angles"], dtype=np.float64)
        indices = self.cfg["indices"]
        self.hip = np.asarray(indices["hip"], dtype=np.int32)
        self.wheel = np.asarray(indices["wheel"], dtype=np.int32)
        self.calf = np.asarray(indices["calf"], dtype=np.int32)
        self.posture = np.asarray(indices["posture"], dtype=np.int32)
        self.ctrl_lower = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=np.float64)
        self.ctrl_upper = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=np.float64)
        self.command_limits = np.asarray(self.cfg["command_limits"], dtype=np.float64)
        self.command = (
            np.asarray(command_override, dtype=np.float64).copy()
            if command_override is not None
            else np.zeros((3,), dtype=np.float64)
        )
        self.command[1] = 0.0
        self.command[:] = np.clip(self.command, self.command_limits[0], self.command_limits[1])
        self.height_command = float(self.cfg["base_height_target"])
        self.last_action = np.zeros((6,), dtype=np.float64)
        self.last_torque = np.zeros((6,), dtype=np.float64)
        self._policy_action_target = np.zeros((6,), dtype=np.float64)
        self.nonwheel_contact_steps = 0
        self.control_tick = 0
        self.reset_count = 0
        self.random_yaw = random_yaw
        self.auto_reset = auto_reset
        flat_terrain = self.terrain_origins.reshape(-1, 3)
        self._terrain_cell = min(2, flat_terrain.shape[0] - 1)
        self._status_deadline = time.perf_counter()
        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._home_key_id < 0:
            raise ValueError("Keyframe 'home' not found in scene.xml")
        key_qpos = np.asarray(self.model.key_qpos, dtype=np.float64).reshape(self.model.nkey, self.model.nq)
        key_qvel = np.asarray(self.model.key_qvel, dtype=np.float64).reshape(self.model.nkey, self.model.nv)
        self._home_qpos = key_qpos[self._home_key_id].copy()
        self._home_qvel = key_qvel[self._home_key_id].copy()
        self._spawn_footprint_offsets = np.asarray(
            [
                [0.0, 0.0],
                [0.18, 0.0],
                [-0.18, 0.0],
                [0.0, 0.16],
                [0.0, -0.16],
                [0.28, 0.16],
                [0.28, -0.16],
                [-0.28, 0.16],
                [-0.28, -0.16],
                [0.38, 0.0],
                [-0.38, 0.0],
            ],
            dtype=np.float64,
        )
        self.reset()

    def read_vector(self, names: list[str]) -> np.ndarray:
        return np.asarray([self.sensors.read(self.data, name)[0] for name in names], dtype=np.float64)

    def active_dof_pos(self) -> np.ndarray:
        return self.read_vector(self.cfg["active_joint_pos_sensors"])

    def active_dof_vel(self) -> np.ndarray:
        return self.read_vector(self.cfg["active_joint_vel_sensors"])

    def _base_height(self) -> float:
        terrain_z = self.hfield.sample(self.data.qpos[:2])
        return float(self.data.qpos[2] - terrain_z)

    def _compute_obs(self) -> np.ndarray:
        gyro = self.sensors.read(self.data, self.cfg["sensor_names"]["gyro"])
        gravity = self.sensors.read(self.data, self.cfg["sensor_names"]["gravity"])
        accel = self.sensors.read(self.data, self.cfg["sensor_names"]["accel"])
        dof_pos = self.active_dof_pos()
        dof_vel = self.active_dof_vel()
        posture_diff = dof_pos[self.posture] - self.default_angles[self.posture]
        posture_vel = dof_vel[self.posture]
        wheel_vel = dof_vel[self.wheel]
        height_error = np.asarray([self.height_command - self._base_height()], dtype=np.float64)
        obs = np.concatenate(
            [
                gyro,
                -gravity,
                accel,
                posture_diff,
                posture_vel,
                wheel_vel,
                self.last_action,
                self.command,
                height_error,
            ],
            axis=0,
        )
        if obs.shape != (29,):
            raise ValueError(f"Expected obs shape (29,), got {obs.shape}")
        return obs.astype(np.float32, copy=False)

    def _policy_action(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run([self.action_name], {self.obs_name: obs[None, :]})[0]
        action = np.asarray(out[0], dtype=np.float64)
        clip = float(self.control_cfg["clip_actions"])
        return np.clip(action, -clip, clip)

    def _compute_torque(self, action: np.ndarray) -> np.ndarray:
        dof_pos = self.active_dof_pos()
        dof_vel = self.active_dof_vel()
        targets = np.zeros_like(action)
        targets[self.hip] = action[self.hip] * float(self.control_cfg["hip_velocity_scale"])
        targets[self.wheel] = action[self.wheel] * float(self.control_cfg["wheel_velocity_scale"])
        targets[self.calf] = (
            self.default_angles[self.calf] + action[self.calf] * float(self.control_cfg["calf_action_scale"])
        )
        torque = np.zeros_like(action)
        torque[self.hip] = float(self.control_cfg["hip_kd"]) * (
            targets[self.hip] - dof_vel[self.hip]
        )
        torque[self.wheel] = float(self.control_cfg["wheel_kd"]) * (
            targets[self.wheel] - dof_vel[self.wheel]
        )
        torque[self.calf] = float(self.control_cfg["calf_kp"]) * (
            targets[self.calf] - dof_pos[self.calf]
        ) - float(self.control_cfg["calf_kd"]) * dof_vel[self.calf]
        return np.clip(torque, self.ctrl_lower, self.ctrl_upper)

    def _update_policy_action(self) -> None:
        obs = self._compute_obs()
        action = self._policy_action(obs)
        self.last_action[:] = action
        self._policy_action_target[:] = action

    def _apply_motor_control(self) -> None:
        torque = self._compute_torque(self._policy_action_target)
        self.last_torque[:] = torque
        self.data.ctrl[:] = torque

    def _contact_max(self, names: list[str]) -> float:
        values = [float(self.sensors.read(self.data, name)[0]) for name in names]
        return max(values) if values else 0.0

    def _failure_state(self) -> tuple[bool, float]:
        gravity = self.sensors.read(self.data, self.cfg["sensor_names"]["gravity"])
        if gravity[2] <= float(self.cfg["termination_config"]["min_up_proj"]):
            return True, self._contact_max(self.cfg["nonwheel_contact_sensors"])
        if self._base_height() <= float(self.cfg["termination_config"]["min_base_height"]):
            return True, self._contact_max(self.cfg["nonwheel_contact_sensors"])
        nonwheel_max = self._contact_max(self.cfg["nonwheel_contact_sensors"])
        failed = nonwheel_max > float(self.cfg["termination_config"]["nonwheel_contact_threshold"])
        return failed, nonwheel_max

    def _should_reset(self) -> bool:
        failed, nonwheel_max = self._failure_state()
        if failed and nonwheel_max > float(self.cfg["termination_config"]["nonwheel_contact_threshold"]):
            self.nonwheel_contact_steps += 1
        else:
            self.nonwheel_contact_steps = 0
        if self.nonwheel_contact_steps >= int(
            self.cfg["termination_config"]["nonwheel_contact_max_steps"]
        ):
            return True
        gravity = self.sensors.read(self.data, self.cfg["sensor_names"]["gravity"])
        return (
            gravity[2] <= float(self.cfg["termination_config"]["min_up_proj"])
            or self._base_height() <= float(self.cfg["termination_config"]["min_base_height"])
        )

    def _origin_for_current_cell(self) -> np.ndarray:
        flat = self.terrain_origins.reshape(-1, 3)
        return np.asarray(flat[self._terrain_cell % flat.shape[0]], dtype=np.float64).copy()

    def _spawn_surface_height(self, xy: np.ndarray, yaw: float) -> float:
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        rot = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
        footprint_xy = np.asarray(xy, dtype=np.float64)[None, :] + self._spawn_footprint_offsets @ rot.T
        return self.hfield.sample_many_max(footprint_xy)

    def _clear_spawn_penetration(self) -> None:
        threshold = float(self.cfg["termination_config"]["nonwheel_contact_threshold"])
        for _ in range(30):
            mujoco.mj_forward(self.model, self.data)
            if self._contact_max(self.cfg["nonwheel_contact_sensors"]) <= threshold:
                return
            self.data.qpos[2] += 0.01
        mujoco.mj_forward(self.model, self.data)

    def advance_terrain(self) -> None:
        flat = self.terrain_origins.reshape(-1, 3)
        self._terrain_cell = (self._terrain_cell + 1) % flat.shape[0]

    def reset(self, *, advance_terrain: bool = False) -> None:
        if advance_terrain:
            self.advance_terrain()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        qpos = self._home_qpos.copy()
        qvel = self._home_qvel.copy()
        reset_cfg = self.cfg["reset_config"]
        origin = self._origin_for_current_cell()
        xy_jitter = min(float(reset_cfg["xy_jitter"]), 0.15)
        qpos[0] = origin[0] + np.random.uniform(-xy_jitter, xy_jitter)
        qpos[1] = origin[1] + np.random.uniform(-xy_jitter, xy_jitter)
        roll = 0.0
        pitch = 0.0
        yaw = (
            np.random.uniform(*reset_cfg["yaw_range"])
            if self.random_yaw
            else 0.0
        )
        terrain_z = self._spawn_surface_height(qpos[:2], yaw)
        qpos[2] = terrain_z + float(self.cfg["home_base_height"]) + float(reset_cfg["spawn_height_margin"])
        qpos[3:7] = quat_mul(qpos[3:7], quat_from_euler_xyz(roll, pitch, yaw))
        qvel[:6] = 0.0
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = 0.0
        self.last_action[:] = 0.0
        self.last_torque[:] = 0.0
        self._policy_action_target[:] = 0.0
        self.nonwheel_contact_steps = 0
        self.control_tick = 0
        self.reset_count += 1
        mujoco.mj_forward(self.model, self.data)
        self._clear_spawn_penetration()
        self._update_policy_action()
        self._apply_motor_control()

    def step(self) -> None:
        if self.control_tick % int(self.cfg["steps_per_control"]) == 0:
            self._update_policy_action()
        self._apply_motor_control()
        mujoco.mj_step(self.model, self.data)
        self.control_tick += 1
        if self.auto_reset and self._should_reset():
            self.reset()

    def set_terrain_cell(self, cell: int) -> None:
        flat = self.terrain_origins.reshape(-1, 3)
        self._terrain_cell = int(np.clip(cell, 0, flat.shape[0] - 1))

    def update_command(self, command: np.ndarray) -> None:
        self.command[:] = np.asarray(command, dtype=np.float64)
        self.command[1] = 0.0
        self.command[:] = np.clip(self.command, self.command_limits[0], self.command_limits[1])

    def status_line(self) -> str:
        linvel = self.sensors.read(self.data, self.cfg["sensor_names"]["local_linvel"])
        failed, nonwheel_max = self._failure_state()
        return (
            f"cmd(vx={self.command[0]:+.2f}, wz={self.command[2]:+.2f}) "
            f"vel(vx={linvel[0]:+.2f}, vy={linvel[1]:+.2f}) "
            f"base_h={self._base_height():.3f} "
            f"nonwheel={nonwheel_max:.2f} "
            f"failed={failed} "
            f"cell={self._terrain_cell} "
            f"resets={self.reset_count}"
        )

    def maybe_print_status(self) -> None:
        now = time.perf_counter()
        if now >= self._status_deadline:
            print(f"[sim2sim] {self.status_line()}")
            self._status_deadline = now + 0.5

    def set_viewer_camera(self, viewer: Any) -> None:
        if not hasattr(viewer, "cam"):
            return
        base_pos = np.asarray(self.data.qpos[:3], dtype=np.float64)
        viewer.cam.lookat[:] = base_pos
