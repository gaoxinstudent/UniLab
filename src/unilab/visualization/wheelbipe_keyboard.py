# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Adapted from wheeled-legged_RL/scripts/rsl_rl/keyboard_controller.py at the
# pinned revision recorded in THIRD_PARTY_NOTICES.md.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================
"""Backend-neutral WheelBipe keyboard command owner.

The source controller is tied to Omniverse keyboard events.  This module keeps
its command semantics independent of any GUI: callers with press/release
events use :meth:`key_press` and :meth:`key_release`; MuJoCo's keycode-only
viewer uses :meth:`viewer_key`, whose deliberately latched behavior is
described on that method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WheelbipeKeyboardController:
    """Source-compatible velocity, height, and jump keyboard state.

    ``W/S`` set forward velocity, ``A/D`` set yaw rate, and their release
    zeros the corresponding axis.  ``Z/X`` set a height rate that is
    integrated by :meth:`advance` and clamped.  ``Q`` queues one jump request;
    :meth:`apply` transfers that request to the state-machine-owned
    ``jump_takeoff_request`` flag exactly once.
    """

    low: np.ndarray
    high: np.ndarray
    default_height: float
    height_range: tuple[float, float] = (0.15, 0.55)
    pos_sensitivity: float = 0.8
    rot_sensitivity: float = 1.0
    height_sensitivity: float = 0.1
    command: np.ndarray = field(init=False)
    current_height: float = field(init=False)
    _height_rate: float = field(init=False, default=0.0)
    _pressed: set[str] = field(init=False, default_factory=set)
    _jump_pending: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=np.float64)
        self.high = np.asarray(self.high, dtype=np.float64)
        if self.low.shape != (3,) or self.high.shape != (3,):
            raise ValueError("WheelBipe keyboard velocity limits must each have shape (3,)")
        if np.any(~np.isfinite(self.low)) or np.any(~np.isfinite(self.high)):
            raise ValueError("WheelBipe keyboard velocity limits must be finite")
        if np.any(self.low > self.high):
            raise ValueError("WheelBipe keyboard velocity limits must be ordered")
        height_bounds = np.asarray(self.height_range, dtype=np.float64)
        if (
            height_bounds.shape != (2,)
            or np.any(~np.isfinite(height_bounds))
            or height_bounds[0] > height_bounds[1]
        ):
            raise ValueError("WheelBipe keyboard height_range must be an ordered finite pair")
        self.height_range = (float(height_bounds[0]), float(height_bounds[1]))
        for name in ("pos_sensitivity", "rot_sensitivity", "height_sensitivity"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"WheelBipe keyboard {name} must be finite and non-negative")
        if not np.isfinite(float(self.default_height)):
            raise ValueError("WheelBipe keyboard default_height must be finite")
        self.command = np.zeros(3, dtype=np.float64)
        self.current_height = float(np.clip(self.default_height, *self.height_range))

    @classmethod
    def from_vel_limit(
        cls,
        vel_limit: Any,
        *,
        default_height: float,
        height_range: tuple[float, float] = (0.15, 0.55),
        pos_sensitivity: float = 0.8,
        rot_sensitivity: float = 1.0,
        height_sensitivity: float = 0.1,
    ) -> "WheelbipeKeyboardController":
        """Construct from the task's ``commands.vel_limit`` contract."""

        limit = np.asarray(vel_limit, dtype=np.float64)
        if limit.shape != (2, 3):
            raise ValueError(f"commands.vel_limit must have shape (2, 3), got {limit.shape}")
        return cls(
            low=limit[0],
            high=limit[1],
            default_height=float(default_height),
            height_range=height_range,
            pos_sensitivity=float(pos_sensitivity),
            rot_sensitivity=float(rot_sensitivity),
            height_sensitivity=float(height_sensitivity),
        )

    @staticmethod
    def _normalize_key(key: str) -> str:
        value = str(key).strip().upper()
        if len(value) != 1:
            raise ValueError(f"WheelBipe keyboard key must be one character, got {key!r}")
        return value

    def key_press(self, key: str) -> bool:
        """Apply a source-style key-press event and return whether it was handled."""

        name = self._normalize_key(key)
        if name == "L":
            self.reset()
        elif name == "W":
            self.command[0] = float(np.clip(self.pos_sensitivity, self.low[0], self.high[0]))
            self._pressed.add(name)
        elif name == "S":
            self.command[0] = float(np.clip(-self.pos_sensitivity, self.low[0], self.high[0]))
            self._pressed.add(name)
        elif name == "A":
            self.command[2] = float(np.clip(self.rot_sensitivity, self.low[2], self.high[2]))
            self._pressed.add(name)
        elif name == "D":
            self.command[2] = float(np.clip(-self.rot_sensitivity, self.low[2], self.high[2]))
            self._pressed.add(name)
        elif name == "Z":
            self._height_rate = self.height_sensitivity
            self._pressed.add(name)
        elif name == "X":
            self._height_rate = -self.height_sensitivity
            self._pressed.add(name)
        elif name == "Q":
            self._jump_pending = True
        else:
            return False
        return True

    def key_release(self, key: str) -> bool:
        """Apply the source release rule for velocity/yaw/height keys."""

        name = self._normalize_key(key)
        if name in {"W", "S"}:
            if name in self._pressed:
                self.command[0] = 0.0
                self._pressed.discard(name)
            return True
        if name in {"A", "D"}:
            if name in self._pressed:
                self.command[2] = 0.0
                self._pressed.discard(name)
            return True
        if name in {"Z", "X"}:
            if name in self._pressed:
                self._height_rate = 0.0
                self._pressed.discard(name)
            return True
        return False

    def viewer_key(self, key: str) -> bool:
        """Handle a MuJoCo keycode-only event.

        MuJoCo's passive viewer does not expose key releases.  Velocity and
        yaw therefore latch at the source magnitude until the opposite key or
        ``L`` is pressed.  Height keys are safe discrete nudges instead of a
        rate that could remain stuck forever.  The full press/release API
        above remains available to frontends that provide both event types.
        """

        name = self._normalize_key(key)
        if name in {"Z", "X"}:
            sign = 1.0 if name == "Z" else -1.0
            self.current_height = float(
                np.clip(
                    self.current_height + sign * self.height_sensitivity,
                    *self.height_range,
                )
            )
            return True
        return self.key_press(name)

    def advance(self, dt: float) -> None:
        """Integrate a held height key for one control step."""

        value = float(dt)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"WheelBipe keyboard dt must be finite and non-negative, got {dt!r}")
        self.current_height = float(
            np.clip(self.current_height + self._height_rate * value, *self.height_range)
        )

    def apply(self, info: dict[str, Any], *, env_id: int | None = None) -> None:
        """Write commands into public env info and transfer a pending jump.

        ``env_id=None`` applies teleoperation to every row (interactive play
        normally has one environment).  A queued Q event is written as a
        one-shot boolean flag; the WheelBipe state-machine owner decides
        whether the jump is enabled/accepted and clears accepted requests.
        """

        commands = info.get("commands")
        heights = info.get("height_commands")
        if not isinstance(commands, np.ndarray) or commands.ndim != 2 or commands.shape[1] != 3:
            raise ValueError("WheelBipe keyboard requires batched info['commands'] with width 3")
        if not isinstance(heights, np.ndarray) or heights.shape != (commands.shape[0],):
            raise ValueError(
                "WheelBipe keyboard requires info['height_commands'] with one value per env"
            )
        if env_id is None:
            selected: int | slice = slice(None)
        else:
            index = int(env_id)
            if index < 0 or index >= commands.shape[0]:
                raise IndexError(f"WheelBipe keyboard env_id {index} is out of range")
            selected = index
        commands[selected] = self.command.astype(commands.dtype, copy=False)
        heights[selected] = np.asarray(self.current_height, dtype=heights.dtype)
        if self._jump_pending:
            raw_request = info.get("jump_takeoff_request")
            if raw_request is None:
                request = np.zeros((commands.shape[0],), dtype=bool)
            else:
                request = np.asarray(raw_request, dtype=bool)
                if request.shape != (commands.shape[0],):
                    raise ValueError("WheelBipe jump_takeoff_request must have one boolean per env")
                request = request.copy()
            request[selected] = True
            info["jump_takeoff_request"] = request
            self._jump_pending = False

    def reset(self) -> None:
        """Reset velocity, height, held keys, and any unconsumed jump."""

        self.command[:] = 0.0
        self.current_height = float(np.clip(self.default_height, *self.height_range))
        self._height_rate = 0.0
        self._pressed.clear()
        self._jump_pending = False

    def zero(self) -> None:
        """Compatibility alias used by the shared interactive viewer."""

        self.reset()

    def describe(self) -> str:
        """Return a compact viewer status line."""

        return (
            f"cmd vx={self.command[0]:+.2f} vy=+0.00 vyaw={self.command[2]:+.2f} "
            f"height={self.current_height:.2f}"
        )


__all__ = ["WheelbipeKeyboardController"]
