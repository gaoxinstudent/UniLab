"""Pinned WheelBipe V14 gimbal spin/translation task mode.

The physical gimbal controller and mutually-exclusive source command-mode
assignment remain owned by ``joystick.py``.  This module consumes reserved
``special_mode_id == 3`` assignments and owns the source-v2 task semantics
layered above them: translation targets expressed in the gimbal-yaw frame,
the seven policy mode features, and the matching reward substitution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from unilab.base.backend.base import SimBackend
from unilab.utils.geometry import np_roll_pitch_from_quat

from ._random import GLOBAL_NUMPY_RANDOM
from .base import WheelbipeV14BaseCfg

if TYPE_CHECKING:
    from .joystick import WheelbipeRewardConfig


def _range(name: str, value: tuple[float, float]) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values, got {value!r}")
    return float(min(array)), float(max(array))


def _segments(name: str, value: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    if not value:
        raise ValueError(f"{name} must contain at least one range")
    return tuple(_range(name, segment) for segment in value)


@dataclass
class WheelbipeGimbalSpinTranslateConfig:
    """Source v2 gimbal-frame command distribution and reward weights."""

    enabled: bool = False
    relative_envs: float = 0.20
    min_episode_time_s: float = 5.0
    yaw_rate_ranges: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: (
            (2.4 * np.pi, 3.6 * np.pi),
            (-3.6 * np.pi, -2.4 * np.pi),
        )
    )
    speed_ranges: tuple[tuple[float, float], ...] = field(default_factory=lambda: ((0.0, 0.75),))
    speed_deadzone: float = 0.05
    heading_range: tuple[float, float] = (-np.pi, np.pi)
    height_range: tuple[float, float] = (0.20, 0.40)
    project_to_body_command: bool = False
    use_sampled_heading_obs: bool = False
    zero_heading_in_deadzone: bool = False
    lin_vel_yaw_scale: float = 1.0
    lin_vel_yaw_sigma: float = 0.25
    lin_speed_scale: float = 1.0
    lin_speed_sigma: float = 0.25
    lin_heading_scale: float = 5.0
    lin_heading_sigma: float = 0.025
    heading_cmd_speed_min: float = 0.10
    heading_measured_speed_min: float = 0.0
    lin_vel_yaw_square_sigma: float = 0.50
    lin_vel_yaw_square_scale: float = -0.20
    lin_speed_overshoot_sigma: float = 0.50
    lin_speed_overshoot_scale: float = 0.0
    heading_error_square_sigma: float = 4.0
    heading_error_square_scale: float = -0.20
    stand_still_scale: float = -1.0
    stand_still_speed_threshold: float = 0.05

    def validate(self) -> None:
        probability = float(self.relative_envs)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"gimbal_spin_translate.relative_envs must be in [0, 1], got {probability!r}"
            )
        _segments("gimbal_spin_translate.yaw_rate_ranges", self.yaw_rate_ranges)
        speed_segments = _segments("gimbal_spin_translate.speed_ranges", self.speed_ranges)
        if any(low < 0.0 for low, _ in speed_segments):
            raise ValueError("gimbal_spin_translate speed ranges cannot be negative")
        _range("gimbal_spin_translate.heading_range", self.heading_range)
        _range("gimbal_spin_translate.height_range", self.height_range)
        for name in (
            "min_episode_time_s",
            "speed_deadzone",
            "lin_vel_yaw_sigma",
            "lin_speed_sigma",
            "lin_heading_sigma",
            "heading_cmd_speed_min",
            "heading_measured_speed_min",
            "lin_vel_yaw_square_sigma",
            "lin_speed_overshoot_sigma",
            "heading_error_square_sigma",
            "stand_still_speed_threshold",
        ):
            number = float(getattr(self, name))
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"gimbal_spin_translate.{name} must be finite and non-negative")
        for name in (
            "lin_vel_yaw_scale",
            "lin_speed_scale",
            "lin_heading_scale",
            "lin_vel_yaw_square_scale",
            "lin_speed_overshoot_scale",
            "heading_error_square_scale",
            "stand_still_scale",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"gimbal_spin_translate.{name} must be finite")


class _WheelbipeTaskModeSuper(Protocol):
    """Next cooperative owner in the concrete WheelBipe environment MRO."""

    def _update_commands(self, info: dict[str, Any]) -> None: ...

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def _update_state_machine(self, info: dict[str, Any]) -> None: ...

    def _compute_reward(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray: ...


class WheelbipeGimbalSpinTranslateOwnerMixin:
    """Cooperative env adapter for source v2 spin/translation semantics."""

    _cfg: WheelbipeV14BaseCfg
    _backend: SimBackend
    _num_envs: int
    _np_dtype: np.dtype[Any]
    _reward_cfg: WheelbipeRewardConfig
    _source_semantics: bool
    _gimbal_enabled: bool
    _gimbal_pos_indices: np.ndarray

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = self._gimbal_spin_translate_cfg()
        cfg.validate()
        if cfg.enabled and not self._gimbal_enabled:
            raise ValueError("gimbal spin/translation mode requires physical gimbal actuators")
        n = self._num_envs
        self._gimbal_spin_active = np.zeros((n,), dtype=bool)
        self._gimbal_spin_last_command_generation = np.full((n,), -1, dtype=np.int64)
        self._gimbal_spin_velocity_yaw = np.zeros((n, 2), dtype=self._np_dtype)
        self._gimbal_spin_heading = np.zeros((n,), dtype=self._np_dtype)
        self._gimbal_spin_height = np.zeros((n,), dtype=self._np_dtype)
        self._gimbal_spin_rng = GLOBAL_NUMPY_RANDOM

    def _task_mode_super(self) -> _WheelbipeTaskModeSuper:
        """Return the next owner while preserving Python's cooperative MRO."""

        return cast(_WheelbipeTaskModeSuper, super())

    def _gimbal_spin_translate_cfg(self) -> WheelbipeGimbalSpinTranslateConfig:
        cfg = getattr(
            self._cfg,
            "gimbal_spin_translate",
            WheelbipeGimbalSpinTranslateConfig(),
        )
        if isinstance(cfg, dict):
            cfg = WheelbipeGimbalSpinTranslateConfig(**cfg)
            setattr(self._cfg, "gimbal_spin_translate", cfg)
        if not isinstance(cfg, WheelbipeGimbalSpinTranslateConfig):
            raise ValueError("gimbal_spin_translate must be WheelbipeGimbalSpinTranslateConfig")
        return cfg

    def _sample_segmented(self, ranges: tuple[tuple[float, float], ...], count: int) -> np.ndarray:
        segments = np.asarray(_segments("gimbal spin ranges", ranges), dtype=np.float64)
        # Pinned V14 first chooses a segment uniformly, then samples uniformly
        # inside it.  This intentionally does not weight wider intervals more.
        selected = self._gimbal_spin_rng.integers(0, len(segments), size=count)
        low = segments[selected, 0]
        high = segments[selected, 1]
        return self._gimbal_spin_rng.uniform(low, high)

    def _resample_gimbal_spin_mode(self, mask: np.ndarray) -> None:
        cfg = self._gimbal_spin_translate_cfg()
        ids = np.flatnonzero(mask)
        if ids.size == 0:
            return
        self._gimbal_spin_velocity_yaw[ids] = 0.0
        self._gimbal_spin_heading[ids] = 0.0
        self._gimbal_spin_height[ids] = 0.0
        speed = self._sample_segmented(cfg.speed_ranges, ids.size)
        speed = np.where(speed < float(cfg.speed_deadzone), 0.0, speed)
        heading_low, heading_high = _range("gimbal_spin_translate.heading_range", cfg.heading_range)
        heading = self._gimbal_spin_rng.uniform(heading_low, heading_high, size=ids.size)
        height_low, height_high = _range("gimbal_spin_translate.height_range", cfg.height_range)
        self._gimbal_spin_heading[ids] = heading
        self._gimbal_spin_velocity_yaw[ids, 0] = speed * np.cos(heading)
        self._gimbal_spin_velocity_yaw[ids, 1] = speed * np.sin(heading)
        self._gimbal_spin_height[ids] = self._gimbal_spin_rng.uniform(
            height_low, height_high, size=ids.size
        )

    def _gimbal_yaw_angle(self) -> np.ndarray:
        if not self._gimbal_enabled:
            return np.zeros((self._num_envs,), dtype=self._np_dtype)
        full_pos = np.asarray(self._backend.get_dof_pos(), dtype=self._np_dtype)
        return full_pos[:, self._gimbal_pos_indices[0]]

    def _apply_gimbal_spin_mode(self, info: dict[str, Any]) -> None:
        cfg = self._gimbal_spin_translate_cfg()
        if not cfg.enabled:
            return
        commands = np.asarray(info["commands"], dtype=self._np_dtype).copy()
        if commands.shape != (self._num_envs, 3):
            raise ValueError(
                "gimbal spin/translation commands must have shape "
                f"{(self._num_envs, 3)}, got {commands.shape}"
            )
        if "special_mode_id" not in info or "command_resample_generation" not in info:
            raise RuntimeError(
                "enabled gimbal spin/translation requires source command owner fields "
                "special_mode_id and command_resample_generation"
            )
        special_mode = np.asarray(info["special_mode_id"], dtype=np.int8)
        generation = np.asarray(info["command_resample_generation"], dtype=np.int64)
        expected = (self._num_envs,)
        if special_mode.shape != expected or generation.shape != expected:
            raise ValueError(
                "gimbal source command owner fields must both have shape "
                f"{expected}; got {special_mode.shape} and {generation.shape}"
            )

        # The source command owner allocates modes 0..3 in one categorical
        # sample.  Consuming id 3 directly keeps gimbal mutually exclusive
        # with spin-low, spin-mid and dash; no second Bernoulli draw belongs
        # in this task owner.
        active = special_mode == 3
        resample = active & (generation != self._gimbal_spin_last_command_generation)
        inactive = ~active
        if np.any(inactive):
            self._gimbal_spin_velocity_yaw[inactive] = 0.0
            self._gimbal_spin_heading[inactive] = 0.0
            self._gimbal_spin_height[inactive] = 0.0
            self._gimbal_spin_last_command_generation[inactive] = -1
        self._resample_gimbal_spin_mode(resample)
        self._gimbal_spin_last_command_generation[resample] = generation[resample]
        self._gimbal_spin_active[:] = active
        yaw_angle = self._gimbal_yaw_angle()
        if np.any(active):
            # ``commands[:, 2]`` is the reserved mode's yaw rate sampled by
            # the source command owner.  Preserve it verbatim so assignment
            # and command generation remain one atomic, mutually-exclusive
            # operation.
            if cfg.project_to_body_command:
                velocity = self._gimbal_spin_velocity_yaw[active]
                cosine = np.cos(yaw_angle[active])
                sine = np.sin(yaw_angle[active])
                commands[active, 0] = cosine * velocity[:, 0] - sine * velocity[:, 1]
                commands[active, 1] = sine * velocity[:, 0] + cosine * velocity[:, 1]
                # The pinned source applies the sampled height only on the
                # projection path.  ``project_to_body_command=false`` zeros
                # XY and returns before this write, so Flat-Play-v2 keeps the
                # ordinary command owner's height rather than activating the
                # separately sampled gimbal-spin height.
                heights = np.asarray(info["height_commands"], dtype=self._np_dtype).copy()
                heights[active] = self._gimbal_spin_height[active]
                info["height_commands"] = heights
            else:
                commands[active, :2] = 0.0
        mode = np.zeros((self._num_envs, 7), dtype=self._np_dtype)
        mode[:, 0] = ~active
        mode[active, 1] = 1.0
        speed = np.linalg.norm(self._gimbal_spin_velocity_yaw, axis=1)
        mode[active, 2] = speed[active]
        heading = self._gimbal_spin_heading.copy()
        if not cfg.use_sampled_heading_obs:
            heading = np.arctan2(
                self._gimbal_spin_velocity_yaw[:, 1],
                self._gimbal_spin_velocity_yaw[:, 0],
            )
        mode[active, 3] = np.sin(heading[active])
        mode[active, 4] = np.cos(heading[active])
        if cfg.zero_heading_in_deadzone:
            dead = active & (speed <= float(cfg.speed_deadzone))
            mode[dead, 3:5] = 0.0
        mode[active, 5] = np.sin(yaw_angle[active])
        mode[active, 6] = np.cos(yaw_angle[active])
        info["commands"] = commands
        info["control_mode_obs"] = mode
        info["gimbal_spin_translate_active"] = active.copy()
        info["gimbal_spin_translate_velocity_yaw"] = self._gimbal_spin_velocity_yaw.copy()
        info["gimbal_spin_translate_heading"] = self._gimbal_spin_heading.copy()

    def _update_commands(self, info: dict[str, Any]) -> None:
        self._task_mode_super()._update_commands(info)
        self._apply_gimbal_spin_mode(info)

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observations, info = self._task_mode_super().reset(env_indices)
        ids = np.asarray(env_indices, dtype=np.intp).reshape(-1)
        if ids.size:
            self._gimbal_spin_active[ids] = False
            self._gimbal_spin_last_command_generation[ids] = -1
            self._gimbal_spin_velocity_yaw[ids] = 0.0
            self._gimbal_spin_heading[ids] = 0.0
            self._gimbal_spin_height[ids] = 0.0
        return observations, info

    def _update_state_machine(self, info: dict[str, Any]) -> None:
        # The state-machine owner runs after command resampling.  Preserve its
        # diagnostics, but restore the source v2 continuous mode tail for
        # active spin/translation environments so both owners can coexist.
        gimbal_mode = np.asarray(
            info.get("control_mode_obs", np.zeros((self._num_envs, 7))),
            dtype=self._np_dtype,
        ).copy()
        self._task_mode_super()._update_state_machine(info)
        cfg = self._gimbal_spin_translate_cfg()
        if not cfg.enabled:
            return
        active = self._gimbal_spin_active
        if not np.any(active):
            return
        mode = np.asarray(info["control_mode_obs"], dtype=self._np_dtype).copy()
        mode[active] = gimbal_mode[active]
        info["control_mode_obs"] = mode

    def _suppressed_xy_reward(
        self,
        commands: np.ndarray,
        velocity_body: np.ndarray,
    ) -> np.ndarray:
        """Return global XY terms masked by pinned gimbal v2."""

        scales = self._reward_cfg.scales
        if self._source_semantics:
            base_quat = np.asarray(self._backend.get_base_quat(), dtype=np.float64)
            _roll, pitch = np_roll_pitch_from_quat(base_quat)
            command_x = commands[:, 0].copy()
            stand = np.abs(command_x) < float(self._reward_cfg.stand_still_deadzone)
            command_x[stand] = 0.0
            error = command_x - velocity_body[:, 0] * np.cos(pitch)
            limited = np.clip(
                error,
                -float(self._reward_cfg.lin_vel_error_constraint),
                float(self._reward_cfg.lin_vel_error_constraint),
            )
            return (
                float(scales.get("track_lin_vel_xy", 0.0))
                * np.exp(-np.square(limited) / float(self._reward_cfg.lin_vel_sigma))
                + float(scales.get("track_lin_vel_xy_tight", 0.0))
                * np.exp(-np.square(limited) / float(self._reward_cfg.lin_vel_tight_sigma))
                + float(scales.get("track_lin_vel_xy_square", 0.0))
                * np.square(error * float(self._reward_cfg.lin_vel_square_sigma))
                + float(scales.get("stand_still_lin_vel", 0.0))
                * np.sum(np.abs(velocity_body), axis=1)
                * stand
            )
        error_sq = np.sum(np.square(commands[:, :2] - velocity_body), axis=1)
        return float(scales.get("tracking_lin_vel", 0.0)) * np.exp(
            -error_sq / float(self._reward_cfg.tracking_sigma)
        )

    def _compute_reward(
        self,
        info: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        reward = self._task_mode_super()._compute_reward(
            info, linvel, gyro, projected_gravity, dof_pos, dof_vel
        )
        cfg = self._gimbal_spin_translate_cfg()
        active = self._gimbal_spin_active if cfg.enabled else np.zeros_like(reward, dtype=bool)
        if not np.any(active):
            return reward
        dtype = reward.dtype
        velocity_body = np.asarray(linvel, dtype=np.float64)[:, :2]
        yaw_angle = self._gimbal_yaw_angle().astype(np.float64)
        cosine = np.cos(yaw_angle)
        sine = np.sin(yaw_angle)
        measured = np.empty_like(velocity_body)
        measured[:, 0] = cosine * velocity_body[:, 0] + sine * velocity_body[:, 1]
        measured[:, 1] = -sine * velocity_body[:, 0] + cosine * velocity_body[:, 1]
        target = self._gimbal_spin_velocity_yaw.astype(np.float64)
        error_sq = np.sum(np.square(target - measured), axis=1)
        target_speed = np.linalg.norm(target, axis=1)
        measured_speed = np.linalg.norm(measured, axis=1)
        speed_error = target_speed - measured_speed
        target_heading = np.arctan2(target[:, 1], target[:, 0])
        measured_heading = np.arctan2(measured[:, 1], measured[:, 0])
        heading_error = (target_heading - measured_heading + np.pi) % (2.0 * np.pi) - np.pi
        heading_gate = (target_speed > float(cfg.heading_cmd_speed_min)) & (
            measured_speed > float(cfg.heading_measured_speed_min)
        )
        stand = target_speed <= float(cfg.stand_still_speed_threshold)
        custom = (
            float(cfg.lin_vel_yaw_scale)
            * np.exp(-error_sq / max(float(cfg.lin_vel_yaw_sigma), 1.0e-6))
            + float(cfg.lin_speed_scale)
            * np.exp(-np.square(speed_error) / max(float(cfg.lin_speed_sigma), 1.0e-6))
            + float(cfg.lin_heading_scale)
            * np.exp(-np.square(heading_error) / max(float(cfg.lin_heading_sigma), 1.0e-6))
            * heading_gate
            * ~stand
            + float(cfg.lin_vel_yaw_square_scale)
            * np.square(float(cfg.lin_vel_yaw_square_sigma))
            * error_sq
            + float(cfg.lin_speed_overshoot_scale)
            * np.square(
                np.maximum(measured_speed - target_speed, 0.0)
                * float(cfg.lin_speed_overshoot_sigma)
            )
            * ~stand
            + float(cfg.heading_error_square_scale)
            * np.square(float(cfg.heading_error_square_sigma) * heading_error)
            * heading_gate
            * ~stand
            + float(cfg.stand_still_scale) * np.sum(np.abs(measured), axis=1) * stand
        )
        # Suppress ordinary body-frame XY tracking exactly where the source
        # replaces it with the gimbal-yaw-frame terms above.
        commands = np.asarray(info["commands"], dtype=np.float64)
        suppressed = self._suppressed_xy_reward(commands, velocity_body)
        delta = custom - suppressed
        valid_active = active & ~np.asarray(
            info.get("numerical_safety_failure", np.zeros_like(active)), dtype=bool
        )
        reward[valid_active] += (delta[valid_active] * float(self._cfg.ctrl_dt)).astype(
            dtype, copy=False
        )
        if np.any(valid_active):
            info.setdefault("log", {})["reward/gimbal_spin_translate"] = float(
                np.mean(custom[valid_active])
            )
        return reward


__all__ = [
    "WheelbipeGimbalSpinTranslateConfig",
    "WheelbipeGimbalSpinTranslateOwnerMixin",
]
