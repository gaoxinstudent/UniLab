from __future__ import annotations

import os
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
            self._is_hfield = False
            self._plane_z = float(model.geom_pos[geom_id, 2])
            return
        self._is_hfield = True
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
        if not self._is_hfield:
            return self._plane_z
        x = float(xy_world[0] - self._geom_pos[0])
        y = float(xy_world[1] - self._geom_pos[1])
        col = int(np.rint((x + self._half_x) / (2.0 * self._half_x) * (self._ncol - 1)))
        row = int(np.rint((y + self._half_y) / (2.0 * self._half_y) * (self._nrow - 1)))
        col = int(np.clip(col, 0, self._ncol - 1))
        row = int(np.clip(row, 0, self._nrow - 1))
        return float(self._geom_pos[2] + self._data[row, col] * self._z_top)

    def sample_many_max(self, xy_world: np.ndarray) -> float:
        if not self._is_hfield:
            return self._plane_z
        points = np.asarray(xy_world, dtype=np.float64).reshape(-1, 2)
        return max(self.sample(point) for point in points)


class CommanderBase:
    def __init__(self) -> None:
        self.command = np.zeros((3,), dtype=np.float64)
        self.paused = False
        self.single_step = False
        self.reset_requested = False
        self.next_terrain_requested = False
        self.follow_camera = True

    def poll(self) -> None:
        return

    def close(self) -> None:
        return

    def _normalize_command(self) -> None:
        self.command[1] = 0.0

    def _print_command(self) -> None:
        print(f"[sim2sim] command = vx={self.command[0]:+.2f}, wz={self.command[2]:+.2f}")

    def _handle_common_key(self, keycode: int) -> bool:
        if keycode in (ord(" "),):
            self.command[:] = 0.0
            self._normalize_command()
            self._print_command()
            return True
        if keycode in (ord("p"), ord("P")):
            self.paused = not self.paused
            print(f"[sim2sim] {'paused' if self.paused else 'resumed'}")
            return True
        if keycode in (ord("n"), ord("N")):
            self.single_step = True
            return True
        if keycode in (ord("r"), ord("R")):
            self.reset_requested = True
            return True
        if keycode in (ord("t"), ord("T")):
            self.next_terrain_requested = True
            return True
        if keycode in (ord("f"), ord("F")):
            self.follow_camera = not self.follow_camera
            print(f"[sim2sim] follow_camera={self.follow_camera}")
            return True
        return False

    def handle(self, keycode: int) -> None:
        self._handle_common_key(keycode)


class KeyboardCommander(CommanderBase):
    def __init__(self, *, step_size: float = 0.05) -> None:
        super().__init__()
        self._step_size = float(step_size)

    def handle(self, keycode: int) -> None:
        if self._handle_common_key(keycode):
            return
        updated = False
        if keycode in (ord("w"), ord("W")):
            self.command[0] += self._step_size
            updated = True
        elif keycode in (ord("s"), ord("S")):
            self.command[0] -= self._step_size
            updated = True
        elif keycode in (ord("a"), ord("A")):
            self.command[2] += self._step_size
            updated = True
        elif keycode in (ord("d"), ord("D")):
            self.command[2] -= self._step_size
            updated = True
        elif keycode in (ord("1"),):
            self.command[:] = np.asarray([0.2, 0.0, 0.0], dtype=np.float64)
            updated = True
        elif keycode in (ord("2"),):
            self.command[:] = np.asarray([0.5, 0.0, 0.0], dtype=np.float64)
            updated = True
        elif keycode in (ord("3"),):
            self.command[:] = np.asarray([0.8, 0.0, 0.0], dtype=np.float64)
            updated = True
        if updated:
            self._normalize_command()
            self._print_command()


class GamepadCommander(CommanderBase):
    _BUTTONS = {
        "triangle": 0,
        "circle": 1,
        "cross": 2,
        "square": 3,
        "select": 8,
        "start": 9,
    }

    def __init__(
        self,
        *,
        joystick_index: int = 0,
        deadzone: float = 0.12,
        vx_scale: float = 0.8,
        wz_scale: float = 0.4,
        axis_exponent: float = 1.5,
    ) -> None:
        super().__init__()
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for PS2 gamepad control. Install it in the uv environment first."
            ) from exc

        self._pygame = pygame
        self._deadzone = float(np.clip(deadzone, 0.0, 0.95))
        self._vx_scale = float(vx_scale)
        self._wz_scale = float(wz_scale)
        self._axis_exponent = max(float(axis_exponent), 1.0)
        self._report_threshold = 1.0e-3

        pygame.display.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count <= joystick_index:
            raise RuntimeError(f"PS2 gamepad not found at joystick index {joystick_index}; detected {count}")
        self._joystick = pygame.joystick.Joystick(joystick_index)
        self._joystick.init()
        self._button_prev = np.zeros((self._joystick.get_numbuttons(),), dtype=bool)
        self._hat_prev = self._joystick.get_hat(0) if self._joystick.get_numhats() > 0 else (0, 0)
        self._last_reported_command = self.command.copy()
        print(
            "[sim2sim] PS2 gamepad connected: "
            f"{self._joystick.get_name()} axes={self._joystick.get_numaxes()} buttons={self._joystick.get_numbuttons()}"
        )

    def close(self) -> None:
        self._joystick.quit()
        self._pygame.joystick.quit()
        self._pygame.display.quit()

    def _shape_axis(self, raw: float, *, invert: bool = False) -> float:
        value = -float(raw) if invert else float(raw)
        magnitude = abs(value)
        if magnitude <= self._deadzone:
            return 0.0
        scaled = (magnitude - self._deadzone) / (1.0 - self._deadzone)
        shaped = scaled**self._axis_exponent
        return float(np.sign(value) * shaped)

    def _button_edge(self, name: str, pressed: np.ndarray) -> bool:
        index = self._BUTTONS[name]
        return bool(index < pressed.shape[0] and pressed[index] and not self._button_prev[index])

    def _report_command_if_changed(self) -> None:
        if np.allclose(self.command, self._last_reported_command, atol=self._report_threshold):
            return
        self._last_reported_command[:] = self.command
        self._print_command()

    def poll(self) -> None:
        self._pygame.event.pump()
        pressed = np.asarray(
            [bool(self._joystick.get_button(i)) for i in range(self._joystick.get_numbuttons())],
            dtype=bool,
        )

        vx = self._shape_axis(self._joystick.get_axis(1), invert=True) * self._vx_scale
        wz = self._shape_axis(self._joystick.get_axis(2)) * self._wz_scale
        self.command[:] = np.asarray([vx, 0.0, wz], dtype=np.float64)

        if self._button_edge("cross", pressed):
            self.command[:] = 0.0
        if self._button_edge("start", pressed):
            self.reset_requested = True
        if self._button_edge("select", pressed):
            self.paused = not self.paused
            print(f"[sim2sim] {'paused' if self.paused else 'resumed'}")
        if self._button_edge("circle", pressed):
            self.next_terrain_requested = True
        if self._button_edge("triangle", pressed):
            self.follow_camera = not self.follow_camera
            print(f"[sim2sim] follow_camera={self.follow_camera}")
        if self._button_edge("square", pressed):
            self.single_step = True

        if self._joystick.get_numhats() > 0:
            hat = self._joystick.get_hat(0)
            if hat != self._hat_prev:
                self._hat_prev = hat

        self._report_command_if_changed()
        self._button_prev = pressed


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
        obs_input = self.session.get_inputs()[0]
        self.obs_name = obs_input.name
        self.obs_dim = int(obs_input.shape[-1])
        if self.obs_dim not in (29, 32):
            raise ValueError(f"Unsupported Real68 policy obs dim: {self.obs_dim}")
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
        self.forward_axis = int(self.cfg.raw.get("forward_axis", 0))
        self.lateral_axis = int(self.cfg.raw.get("lateral_axis", 1 if self.forward_axis == 0 else 0))
        self.forward_sign = float(self.cfg.raw.get("forward_sign", 1.0))
        self.command = (
            np.asarray(command_override, dtype=np.float64).copy()
            if command_override is not None
            else np.zeros((3,), dtype=np.float64)
        )
        self.command[1] = 0.0
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
        linvel = self.sensors.read(self.data, self.cfg["sensor_names"]["local_linvel"])
        gyro = self.sensors.read(self.data, self.cfg["sensor_names"]["gyro"])
        gravity = self.sensors.read(self.data, self.cfg["sensor_names"]["gravity"])
        accel = self.sensors.read(self.data, self.cfg["sensor_names"]["accel"])
        dof_pos = self.active_dof_pos()
        dof_vel = self.active_dof_vel()
        posture_diff = dof_pos[self.posture] - self.default_angles[self.posture]
        posture_vel = dof_vel[self.posture]
        wheel_vel = dof_vel[self.wheel]
        height_error = np.asarray([self.height_command - self._base_height()], dtype=np.float64)
        parts = [
            gyro,
            -gravity,
            accel,
            posture_diff,
            posture_vel,
            wheel_vel,
            self.last_action,
            self.command,
            height_error,
        ]
        if self.obs_dim == 32:
            parts.insert(0, linvel)
        obs = np.concatenate(parts, axis=0)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Expected obs shape ({self.obs_dim},), got {obs.shape}")
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

    def status_line(self) -> str:
        linvel = self.sensors.read(self.data, self.cfg["sensor_names"]["local_linvel"])
        forward_vel = float(linvel[self.forward_axis] * self.forward_sign)
        lateral_vel = float(linvel[self.lateral_axis])
        failed, nonwheel_max = self._failure_state()
        return (
            f"cmd(vx={self.command[0]:+.2f}, wz={self.command[2]:+.2f}) "
            f"vel(forward={forward_vel:+.2f}, lateral={lateral_vel:+.2f}) "
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
