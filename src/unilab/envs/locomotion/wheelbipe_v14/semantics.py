"""Pinned WheelBipe V14 training semantics expressed as pure NumPy helpers.

The functions in this module intentionally have no simulator dependency.  The
environment owner resolves body/joint IDs and reads :class:`SimBackend` state;
this module only defines the source observation, reward, command, reset and
termination math.  Keeping that boundary explicit makes the 78D critic layout
and reward graph inexpensive to regression-test.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype

SOURCE_V14_RESET_JOINT_NAMES: tuple[str, ...] = (
    "left_front1_joint",
    "right_front1_joint",
    "left_rear1_joint",
    "right_rear1_joint",
    "left_front2_joint",
    "right_front2_joint",
    "left_front3_joint",
    "right_front3_joint",
    "left_front4_joint",
    "right_front4_joint",
    "left_rear2_joint",
    "right_rear2_joint",
)

# ``privileged_extra_joint_names`` in the pinned source resolves these regexes
# in tuple order: rear pair, front pair, then wheels.  This differs from the
# public policy action order and therefore remains an explicit permutation.
SOURCE_V14_PRIVILEGED_POLICY_PERMUTATION = np.asarray([1, 3, 0, 2, 4, 5], dtype=np.intp)

SOURCE_V14_WHEEL_BODY_NAMES: tuple[str, ...] = (
    "left_wheel_link",
    "right_wheel_link",
)
# The source V14 contact sensor is configured with the ``.*_guide_link``
# pattern.  Body-id lookup in the backend contract is exact-name based, so
# materialize the pattern once on the frozen V14 asset rather than passing a
# regex into a backend.  This list is also used by the source flat reset set.
SOURCE_V14_GUIDE_BODY_NAMES: tuple[str, ...] = (
    "left_rear2_guide_link",
    "right_rear2_guide_link",
    "left_bottom1_guide_link",
    "left_bottom2_guide_link",
    "left_bottom3_guide_link",
    "left_bottom4_guide_link",
    "left_front1_guide_link",
    "left_front_guide_link",
    "left_rear_guide_link",
    "right_bottom1_guide_link",
    "right_bottom2_guide_link",
    "right_bottom3_guide_link",
    "right_bottom4_guide_link",
    "right_front1_guide_link",
    "right_front_guide_link",
    "right_rear_guide_link",
)
# ``EventCfgV14.add_leg_mass`` names these bodies explicitly.  In particular,
# it does *not* match the mechanism's ``*_guide_link`` bodies; keeping this
# list separate from the contact groups prevents a subtree-wide randomizer
# from silently changing guide-link inertias on the migrated owner.
SOURCE_V14_LEG_MASS_BODY_NAMES: tuple[str, ...] = (
    "left_front1_link",
    "right_front1_link",
    "left_front2_link",
    "right_front2_link",
    "left_front3_link",
    "right_front3_link",
    "left_front4_link",
    "right_front4_link",
    "left_rear1_link",
    "right_rear1_link",
    "left_rear2_link",
    "right_rear2_link",
    "left_spring1_link",
    "right_spring1_link",
    "left_spring2_link",
    "right_spring2_link",
    "gimbal_yaw_link",
    "gimbal_pitch_link",
)
SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES: tuple[str, ...] = (
    "base_link",
    "left_rear1_link",
    "right_rear1_link",
    "left_rear2_link",
    "right_rear2_link",
    "left_front1_link",
    "right_front1_link",
    "left_front2_link",
    "right_front2_link",
    "left_front3_link",
    "right_front3_link",
    "left_front4_link",
    "right_front4_link",
    "gimbal_yaw_link",
    "gimbal_pitch_link",
    *SOURCE_V14_GUIDE_BODY_NAMES,
)

# ``WheelbipeV14Env.__init__`` overrides the inherited V13/V25 reset set.  On
# the flat (plane) owner it uses base, gimbal and all guide contacts; the
# rough owner overrides this to the two gimbal links for non-plane terrain.
# Keep this separate from the diagnostic undesired-contact set because the
# two source lists have different lifecycle meanings.
SOURCE_V14_RESET_CONTACT_BODY_NAMES: tuple[str, ...] = (
    "base_link",
    "gimbal_yaw_link",
    "gimbal_pitch_link",
    *SOURCE_V14_GUIDE_BODY_NAMES,
)

SOURCE_V14_REWARD_SCALES: dict[str, float] = {
    "termination": -200.0,
    "leg_joint_acc": -5.0e-7,
    "leg_joint_vel": -5.0e-3,
    "leg_joint_pair_pos_diff": -0.0,
    "joint_torque": -1.0e-4,
    "wheel_acc": -1.0e-8,
    "wheel_vel": -1.0e-5,
    "wheel_power": -1.0e-4,
    "wheel_air_spin": 0.0,
    "lin_vel_z": -0.5,
    "ang_vel_xy": -0.05,
    "action_smoothness_leg": -0.05,
    "action_rate": -0.01,
    "action_smoothness_wheel": -0.01,
    "flat_orientation_y": -0.0,
    "flat_orientation_y_v": -2.0,
    "flat_orientation_y_exp": 1.0,
    "flat_orientation_x": -0.0,
    "flat_orientation_x_v": -2.0,
    "flat_orientation_x_exp": 1.0,
    "track_lin_vel_xy": 1.0,
    "track_lin_vel_xy_tight": 0.0,
    # Effective values recorded in the upstream checkpoint's params/env.yaml.
    # The source class defaults are -1/-1/-1, but the verified training run
    # materializes these runtime overrides and the checkpoint was optimized
    # against them.
    "track_lin_vel_xy_square": -0.1,
    "track_ang_vel_z": 1.0,
    "track_ang_vel_z_square": -0.1,
    "stand_still_lin_vel": -2.0,
    "stand_still": -0.0,
    "track_height_exp": 0.0,
    "track_height_exp_soft": 0.0,
    "track_height_exp_tight": 1.0,
    "track_height_square": -1.0,
    "track_height_exp_both_wheels_contact": 0.0,
    "no_fork": -1.0,
    "no_fork_square": -1.0,
    "no_fork_exp": -0.0,
    "no_fork_z_exp": -0.0,
    "undesired_contact": -2.0,
}

# Exact subclasses mutate the inherited Flat reward map in their pinned
# ``__post_init__`` methods.  Keep the resulting enabled graphs as immutable
# owner profiles instead of relying on runtime state-machine code to repair a
# wrong base scale after the reward has already been accumulated.
SOURCE_V14_FLAT_V1_REWARD_SCALES: dict[str, float] = {
    **SOURCE_V14_REWARD_SCALES,
    "action_rate": -0.002,
    "action_smoothness_leg": -0.005,
    "action_smoothness_wheel": -0.001,
    "leg_joint_acc": -1.0e-7,
    "leg_joint_vel": -1.0e-3,
    "wheel_acc": -2.0e-9,
    "wheel_vel": -2.0e-6,
}
SOURCE_V14_ROUGH_V0_REWARD_SCALES: dict[str, float] = {
    **SOURCE_V14_REWARD_SCALES,
    "wheel_power": -1.0e-5,
    "joint_torque": -1.0e-5,
    # The rough rotation/stair checkpoints materialize the source command
    # owner's stand-still penalty at -1 (the flat owner uses -2 in its
    # verified run).
    "stand_still_lin_vel": -1.0,
    # Exact source Rough-v0 defaults and the 10:19:59 snapshot use -1.
    # Canonical PPO instead selects the 16:23:21 snapshot (-0.1) in YAML.
    "track_lin_vel_xy_square": -1.0,
    "track_ang_vel_z_square": -1.0,
}
SOURCE_V14_ROUGH_V1_REWARD_SCALES: dict[str, float] = {
    **SOURCE_V14_FLAT_V1_REWARD_SCALES,
    "wheel_power": -1.0e-5,
    "joint_torque": -1.0e-5,
    "track_lin_vel_xy": 1.25,
    # Exact source Rough-v1 class defaults, separate from canonical PPO's
    # released 16:23:21 snapshot.
    "track_lin_vel_xy_square": -1.0,
    "track_ang_vel_z_square": -1.0,
    "stand_still_lin_vel": -1.0,
}


class SourceV14HIMCurriculum:
    """Backend-neutral runtime for pinned ``CurriculumCfgV14``.

    The two upstream manager terms consume the same completed-episode batch
    mean, window and thresholds, so their reward-weight and vertical-assist
    stages advance in lockstep.  This owner keeps that enabled-path behavior
    together while simulation force application remains on ``SimBackend``.
    """

    def __init__(
        self,
        *,
        default_reward_scales: Mapping[str, float],
        reward_key: str,
        num_steps_per_env: int,
        window_size: int,
        min_stage_episodes: int,
        normalize_by_episode_length: bool,
        reward_stage_weights: Sequence[Mapping[str, float]],
        assist_force_z_stages: Sequence[float],
        thresholds: Sequence[float],
        stage_min_episodes: Sequence[int],
        restore_defaults_after_final_threshold: bool,
    ) -> None:
        self.reward_key = str(reward_key)
        self.num_steps_per_env = int(num_steps_per_env)
        if self.num_steps_per_env <= 0:
            raise ValueError("HIM curriculum num_steps_per_env must be positive")
        self.window_size = max(int(window_size), 1) * self.num_steps_per_env
        self.min_stage_episodes = max(int(min_stage_episodes), 1) * self.num_steps_per_env
        self.normalize_by_episode_length = bool(normalize_by_episode_length)
        self.restore_defaults_after_final_threshold = bool(restore_defaults_after_final_threshold)
        self._default_reward_scales = {
            str(key): float(value) for key, value in default_reward_scales.items()
        }
        self._reward_stage_weights = [
            {str(key): float(value) for key, value in stage.items()}
            for stage in reward_stage_weights
        ]
        self._assist_force_z_stages = tuple(float(value) for value in assist_force_z_stages)
        self._thresholds = tuple(float(value) for value in thresholds)
        self._stage_min_episodes = tuple(
            max(int(value), 0) * self.num_steps_per_env for value in stage_min_episodes
        )
        if not self._reward_stage_weights:
            raise ValueError("HIM curriculum requires at least one reward stage")
        missing_reward_keys = sorted(
            {
                key
                for stage in self._reward_stage_weights
                for key in stage
                if key not in self._default_reward_scales
            }
        )
        if missing_reward_keys:
            raise ValueError(
                "HIM curriculum reward keys are absent from the immutable source reward graph: "
                f"{missing_reward_keys}"
            )
        if self.reward_key not in self._default_reward_scales:
            raise ValueError(
                f"HIM curriculum reward_key {self.reward_key!r} is absent from the reward graph"
            )
        expected_force_stages = len(self._reward_stage_weights) + int(
            self.restore_defaults_after_final_threshold
        )
        if len(self._assist_force_z_stages) != expected_force_stages:
            raise ValueError(
                "HIM curriculum assist force stages must cover every reward stage "
                "and the optional restored-default stage"
            )
        expected_transitions = expected_force_stages - 1
        if len(self._thresholds) != expected_transitions:
            raise ValueError("HIM curriculum thresholds must cover every stage transition")
        if len(self._stage_min_episodes) != expected_transitions:
            raise ValueError("HIM curriculum stage_min_episodes must cover every transition")
        if any(force < 0.0 for force in self._assist_force_z_stages):
            raise ValueError("HIM curriculum assist forces must be non-negative")
        self._stage = 0
        self._recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self._batches_since_stage_change = 0
        self._total_batches = 0
        self._last_batch_mean = 0.0
        self._last_window_mean = 0.0

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def assist_force_z(self) -> float:
        return self._assist_force_z_stages[self._stage]

    @property
    def reward_scales(self) -> dict[str, float]:
        scales = dict(self._default_reward_scales)
        if self._stage < len(self._reward_stage_weights):
            scales.update(
                {
                    key: value
                    for key, value in self._reward_stage_weights[self._stage].items()
                    if key in scales
                }
            )
        return scales

    def record_completed_batch(
        self,
        episode_reward_sums: np.ndarray,
        *,
        max_episode_length_s: float,
    ) -> bool:
        """Record one source manager compute call and maybe advance a stage."""

        # The source ``_episode_sums`` buffer is ``torch.float``.  Retain its
        # float32 accumulation/normalization before converting each completed
        # value to a Python float, exactly as ``Tensor.tolist()`` does there.
        values = np.asarray(episode_reward_sums, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return False
        if self.normalize_by_episode_length:
            duration = float(max_episode_length_s)
            if duration <= 0.0:
                raise ValueError("HIM curriculum max_episode_length_s must be positive")
            values = values / np.float32(duration)
        rewards_cpu = values.tolist()
        self._last_batch_mean = float(sum(rewards_cpu) / len(rewards_cpu))
        self._recent_rewards.append(self._last_batch_mean)
        self._batches_since_stage_change += 1
        self._total_batches += 1
        self._last_window_mean = float(sum(self._recent_rewards) / len(self._recent_rewards))

        if self._stage >= len(self._assist_force_z_stages) - 1:
            return False
        if len(self._recent_rewards) < self.window_size:
            return False
        min_batches = (
            self._stage_min_episodes[self._stage]
            if self._stage < len(self._stage_min_episodes)
            else self.min_stage_episodes
        )
        if self._batches_since_stage_change < min_batches:
            return False
        window_values = list(self._recent_rewards)[-self.window_size :]
        window_mean = float(sum(window_values) / self.window_size)
        if window_mean < self._thresholds[self._stage]:
            return False
        self._stage += 1
        self._batches_since_stage_change = 0
        self._recent_rewards.clear()
        self._last_window_mean = window_mean
        return True

    def contract_snapshot(self) -> dict[str, Any]:
        """Return the two source manager-term states without hiding their shapes.

        The enabled source terms use identical samples and gates, so their
        transitions are observably lock-step.  The reward term nevertheless
        remains at stage 1 when it restores defaults, while the assist term
        advances to its distinct third (zero-force) stage.  Exposing both
        views avoids reporting the combined runtime index as a source manager
        state that never existed.
        """

        managed_keys = {key for stage in self._reward_stage_weights for key in stage}
        reward_stage = min(self._stage, len(self._reward_stage_weights) - 1)
        defaults_restored = bool(
            self.restore_defaults_after_final_threshold
            and self._stage >= len(self._reward_stage_weights)
        )
        common: dict[str, float | int] = {
            "last_window_mean": float(self._last_window_mean),
            "last_batch_mean": float(self._last_batch_mean),
            "total_episodes": int(self._total_batches // self.num_steps_per_env),
            "total_compute_calls": int(self._total_batches),
            "window_samples": int(self.window_size),
        }
        reward_state: dict[str, float | int] = {
            **common,
            "stage": int(reward_stage),
            "stage_count": int(len(self._reward_stage_weights)),
            "defaults_restored": int(defaults_restored),
        }
        for key in managed_keys:
            reward_state[f"weight_{key}"] = float(self.reward_scales[key])
        assist_state: dict[str, float | int] = {
            **common,
            "stage": int(self._stage),
            "stage_count": int(len(self._assist_force_z_stages)),
            "force_z": float(self.assist_force_z),
        }
        if self._stage < len(self._thresholds):
            threshold = float(self._thresholds[self._stage])
            min_calls = int(self._stage_min_episodes[self._stage])
            reward_state["next_threshold"] = threshold
            reward_state["min_episodes"] = int(min_calls // self.num_steps_per_env)
            reward_state["min_compute_calls"] = min_calls
            assist_state["next_threshold"] = threshold
            assist_state["min_episodes"] = int(min_calls // self.num_steps_per_env)
            assist_state["min_compute_calls"] = min_calls
        return {
            "track_height_progression": reward_state,
            "base_vertical_assist_force_progression": assist_state,
        }


def source_v14_inverse_kinematics(
    leg_length: np.ndarray,
    leg_angle: np.ndarray,
    *,
    # These are the effective constants serialized by the verified source V14
    # training run (``params/env.yaml``).  ``cfg_utils.py`` also contains a
    # newer candidate V14 table, but ``WheelbipeV14FlatEnvCfg`` leaves those
    # fields commented out and inherits the following values in the actual
    # source reset path.  Keep the effective owner values here so a migrated
    # reset produces the same closed-chain pose as the source checkpoint.
    links_length: Sequence[float] = (
        0.1134,
        0.135,
        0.21,
    ),
    alpha_offset: Sequence[float] = (
        np.deg2rad(-6.61),
        np.pi,
        np.deg2rad(29.7),
        np.deg2rad(180.0 - 6.61 - 2.0 * 29.7),
        np.deg2rad(29.7),
        np.deg2rad(29.7),
    ),
) -> np.ndarray:
    """Return the pinned V14 closed-chain IK in source reset order.

    ``leg_length`` and ``leg_angle`` have shape ``(N, 2)`` for left/right
    legs.  The result has shape ``(N, 12)`` and matches the source
    ``transpose(-2, -1).reshape(N, -1)`` ordering consumed by
    :data:`SOURCE_V14_RESET_JOINT_NAMES`.
    """

    length = np.asarray(leg_length, dtype=np.float64)
    angle = np.asarray(leg_angle, dtype=np.float64)
    if length.ndim != 2 or length.shape[1] != 2 or angle.shape != length.shape:
        raise ValueError(
            "leg_length and leg_angle must have the same shape (N, 2), "
            f"got {length.shape} and {angle.shape}"
        )
    links = np.asarray(links_length, dtype=np.float64).reshape(-1)
    offsets = np.asarray(alpha_offset, dtype=np.float64).reshape(-1)
    if links.shape != (3,) or offsets.shape != (6,):
        raise ValueError("links_length and alpha_offset must contain 3 and 6 values")
    if np.any(length <= 0.0) or np.any(links <= 0.0):
        raise ValueError("leg and link lengths must be positive")

    links_sq = np.square(links)
    length_sq = np.square(length)
    solve_arg = (links_sq[0] * length_sq + links_sq[2] * (links_sq[0] - links_sq[1])) / (
        2.0 * links[2] * links_sq[0] * length
    )
    solve_triangle = np.arccos(np.clip(solve_arg, -1.0, 1.0))

    alpha = np.zeros((*length.shape, 6), dtype=np.float64)
    alpha[..., 0] = angle + 0.5 * np.pi - solve_triangle
    alpha[..., 1] = angle + 0.5 * np.pi + solve_triangle
    passive_arg = (links_sq[0] + links_sq[1] - links_sq[0] * length_sq / links_sq[2]) / (
        2.0 * links[0] * links[1]
    )
    alpha[..., 2] = np.arccos(np.clip(passive_arg, -1.0, 1.0))
    alpha[..., 3] = 2.0 * np.pi - (alpha[..., 1] - alpha[..., 0]) - 2.0 * alpha[..., 2]
    alpha[..., 4] = alpha[..., 2]
    alpha[..., 5] = alpha[..., 2]

    alpha[..., 0] -= offsets[0]
    alpha[..., 1] -= offsets[1]
    alpha[..., 2] = -(alpha[..., 2] - offsets[2])
    alpha[..., 3] = -(alpha[..., 3] - offsets[3])
    alpha[..., 4] = -(alpha[..., 4] - offsets[4])
    alpha[..., 5] -= offsets[5]
    return alpha.transpose(0, 2, 1).reshape(length.shape[0], 12)


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(values), low, high)


def build_source_v14_height_signals(
    root_height_w: np.ndarray,
    terrain_height_w: np.ndarray | None = None,
    *,
    use_absolute_height: bool,
    clip_enabled: bool = False,
    clip_range: Sequence[float | None] = (None, None),
) -> tuple[np.ndarray, np.ndarray]:
    """Return the source critic height and reward-reference height.

    The pinned V14 owner first applies its optional observation-height clamp to
    the robot's *world-frame* root z.  That absolute signal is written to the
    privileged observation.  Only the reward path subtracts the estimated
    terrain height when ``use_absolute_height`` is false.  Keeping the two
    arrays separate prevents rough-terrain owners from silently feeding a
    terrain-relative height into source PPO/HIM/DreamWaQ critics.
    """

    observed = np.asarray(root_height_w, dtype=get_global_dtype()).reshape(-1).copy()
    if clip_enabled:
        bounds = tuple(clip_range)
        if len(bounds) != 2:
            raise ValueError("height_obs_clip_range must contain exactly two bounds")
        lower_raw, upper_raw = bounds
        lower = -np.inf if lower_raw is None else float(lower_raw)
        upper = np.inf if upper_raw is None else float(upper_raw)
        if not np.isfinite(lower) and lower != -np.inf:
            raise ValueError("height_obs_clip_range lower bound must be finite or None")
        if not np.isfinite(upper) and upper != np.inf:
            raise ValueError("height_obs_clip_range upper bound must be finite or None")
        if upper < lower:
            raise ValueError("height_obs_clip_range upper bound must be >= lower bound")
        np.clip(observed, lower, upper, out=observed)

    reward_height = observed.copy()
    if not bool(use_absolute_height):
        if terrain_height_w is None:
            terrain = np.zeros_like(observed)
        else:
            terrain = np.asarray(terrain_height_w, dtype=observed.dtype).reshape(-1)
            if terrain.shape != observed.shape:
                raise ValueError(
                    "terrain_height_w must match root_height_w; "
                    f"got {terrain.shape} and {observed.shape}"
                )
        reward_height -= terrain
    return observed, reward_height


def build_source_v14_critic_observation(
    *,
    commands: np.ndarray,
    height_command: np.ndarray,
    gyro: np.ndarray,
    projected_gravity: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    actions: np.ndarray,
    root_lin_vel_b: np.ndarray,
    observed_height: np.ndarray,
    control_mode: np.ndarray,
    joint_stiffness: np.ndarray,
    joint_damping: np.ndarray,
    applied_torque: np.ndarray,
    obs_delay_steps: np.ndarray,
    act_delay_steps: np.ndarray,
    wheel_body_lin_vel_b: np.ndarray,
    wheel_contact_state: np.ndarray,
    base_mass_scale: np.ndarray,
    wheel_material: np.ndarray,
    default_angles: np.ndarray,
    control_mode_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Build the exact 78D privileged critic field order of pinned V14."""

    dtype = get_global_dtype()
    commands = np.asarray(commands, dtype=dtype)
    n = int(commands.shape[0])

    def batch(value: np.ndarray, width: int, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=dtype).reshape(n, -1)
        if result.shape != (n, width):
            raise ValueError(f"{name} must have shape ({n}, {width}), got {result.shape}")
        return result

    commands = batch(commands, 3, "commands")
    height = batch(np.asarray(height_command).reshape(n, 1), 1, "height_command")
    gyro = batch(gyro, 3, "gyro")
    gravity = batch(projected_gravity, 3, "projected_gravity")
    pos = batch(dof_pos, 6, "dof_pos")
    vel = batch(dof_vel, 6, "dof_vel")
    actions = batch(actions, 6, "actions")
    root_lin = batch(root_lin_vel_b, 3, "root_lin_vel_b")
    obs_height = batch(np.asarray(observed_height).reshape(n, 1), 1, "observed_height")
    mode = batch(control_mode, 7, "control_mode")
    if control_mode_scale is None:
        mode_scale = np.ones((7,), dtype=dtype)
    else:
        mode_scale = np.asarray(control_mode_scale, dtype=dtype).reshape(-1)
        if mode_scale.shape != (7,) or not np.all(np.isfinite(mode_scale)):
            raise ValueError("control_mode_scale must contain seven finite values")
    stiffness = batch(joint_stiffness, 6, "joint_stiffness")
    damping = batch(joint_damping, 6, "joint_damping")
    torque = batch(applied_torque, 6, "applied_torque")
    obs_lags = batch(obs_delay_steps, 4, "obs_delay_steps")
    act_lags = batch(act_delay_steps, 2, "act_delay_steps")
    wheel_velocity = batch(wheel_body_lin_vel_b, 6, "wheel_body_lin_vel_b")
    contact = batch(wheel_contact_state, 2, "wheel_contact_state")
    mass_scale = batch(np.asarray(base_mass_scale).reshape(n, 1), 1, "base_mass_scale")
    material = batch(wheel_material, 6, "wheel_material")
    defaults = np.asarray(default_angles, dtype=dtype).reshape(-1)
    if defaults.shape != (6,):
        raise ValueError(f"default_angles must have shape (6,), got {defaults.shape}")

    # Base critic frame: the same 28 semantic fields as policy, but current,
    # noise-free and undelayed, followed by root velocity, observed height,
    # and the seven-element control-mode block.
    leg_position = pos[:, :4] - defaults[:4]
    muted_wheel_position = np.zeros((n, 2), dtype=dtype)
    base = np.concatenate(
        (
            _clip(commands, -100.0, 100.0),
            _clip(height, 0.0, 1.0) * 5.0,
            _clip(gyro, -100.0, 100.0) * 0.5,
            _clip(gravity, -100.0, 100.0),
            _clip(leg_position, -100.0, 100.0),
            muted_wheel_position,
            _clip(vel[:, :4], -200.0, 200.0) * 0.1,
            _clip(vel[:, 4:], -200.0, 200.0) * 0.1,
            _clip(actions, -100.0, 100.0),
            _clip(root_lin, -100.0, 100.0),
            _clip(obs_height, -10.0, 10.0) * 5.0,
            mode * mode_scale[None, :],
        ),
        axis=1,
    )
    privileged_extra = np.concatenate(
        (
            _clip(stiffness, -100.0, 100.0),
            _clip(damping, -100.0, 100.0),
            _clip(torque, -100.0, 100.0) * 0.05,
            _clip(obs_lags, -100.0, 100.0),
            _clip(act_lags, -100.0, 100.0),
            _clip(wheel_velocity, -100.0, 100.0),
            _clip(contact, -1.0, 1.0),
            _clip(mass_scale, -10.0, 10.0),
            _clip(material, -100.0, 100.0),
        ),
        axis=1,
    )
    critic = np.concatenate((base, privileged_extra), axis=1)
    if base.shape[1] != 39 or privileged_extra.shape[1] != 39 or critic.shape != (n, 78):
        raise RuntimeError(
            "source V14 critic layout must be base39 + privileged39 = 78, "
            f"got {base.shape[1]} + {privileged_extra.shape[1]}"
        )
    return np.nan_to_num(critic, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype, copy=False)


@dataclass(frozen=True)
class SourceV14RewardParameters:
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
    vel_height_gate_enabled: bool = False
    vel_height_gate_mode: str = "linear_band"
    vel_height_gate_full_error: float = 0.05
    vel_height_gate_zero_error: float = 0.1
    vel_height_gate_tracker_sigma: float = 0.02


def compute_source_v14_reward(
    *,
    scales: Mapping[str, float],
    ctrl_dt: float,
    commands: np.ndarray,
    linvel: np.ndarray,
    gyro: np.ndarray,
    projected_gravity: np.ndarray,
    pitch: np.ndarray,
    observed_height: np.ndarray,
    height_command: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    qacc: np.ndarray,
    torque: np.ndarray,
    actions: np.ndarray,
    last_actions: np.ndarray,
    previous_actions: np.ndarray,
    wheel_pos_b: np.ndarray,
    wheel_contact_state: np.ndarray,
    undesired_contact: np.ndarray,
    terminated: np.ndarray,
    params: SourceV14RewardParameters = SourceV14RewardParameters(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute the pinned V14 reward graph and return weighted, dt-scaled terms."""

    dtype = get_global_dtype()
    commands = np.asarray(commands, dtype=dtype)
    n = commands.shape[0]
    linvel = np.asarray(linvel, dtype=dtype)
    gyro = np.asarray(gyro, dtype=dtype)
    gravity = np.asarray(projected_gravity, dtype=dtype)
    dof_pos = np.asarray(dof_pos, dtype=dtype)
    dof_vel = np.asarray(dof_vel, dtype=dtype)
    qacc = np.asarray(qacc, dtype=dtype)
    torque = np.asarray(torque, dtype=dtype)
    actions = np.asarray(actions, dtype=dtype)
    last_actions = np.asarray(last_actions, dtype=dtype)
    previous_actions = np.asarray(previous_actions, dtype=dtype)
    wheel_pos = np.asarray(wheel_pos_b, dtype=dtype).reshape(n, 2, 3)
    wheel_contact = np.asarray(wheel_contact_state, dtype=bool).reshape(n, 2)

    tracking_commands = commands.copy()
    if bool(params.stand_still_deadzone_enabled):
        stand_lin = np.abs(commands[:, 0]) < float(params.stand_still_deadzone)
        stand_yaw = np.abs(commands[:, 2]) < float(params.stand_still_deadzone)
    else:
        stand_lin = np.zeros((n,), dtype=bool)
        stand_yaw = np.zeros((n,), dtype=bool)
    tracking_commands[stand_lin, 0] = 0.0
    tracking_commands[stand_yaw, 2] = 0.0

    pgb_x = gravity[:, 0]
    pgb_y = gravity[:, 1]
    command_x_sq = np.square(commands[:, 0])
    forward_horizontal = linvel[:, 0] * np.cos(np.asarray(pitch, dtype=dtype))
    lin_error = tracking_commands[:, 0] - forward_horizontal
    lin_error_limited = np.clip(
        lin_error,
        -float(params.lin_vel_error_constraint),
        float(params.lin_vel_error_constraint),
    )
    yaw_error = tracking_commands[:, 2] - gyro[:, 2]
    yaw_error_limited = np.clip(
        yaw_error,
        -float(params.ang_vel_error_constraint),
        float(params.ang_vel_error_constraint),
    )
    height_error = np.asarray(observed_height).reshape(n) - np.asarray(height_command).reshape(n)
    height_error_limited = np.clip(
        height_error,
        -float(params.height_error_constraint),
        float(params.height_error_constraint),
    )
    action_second_difference = actions - 2.0 * last_actions + previous_actions
    wheel_power = torque[:, 4:] * dof_vel[:, 4:]
    wheel_x_difference = wheel_pos[:, 0, 0] - wheel_pos[:, 1, 0]
    wheel_z_difference = wheel_pos[:, 0, 2] - wheel_pos[:, 1, 2]
    no_fork_over = np.maximum(np.abs(wheel_x_difference) - float(params.no_fork_distance), 0.0)
    no_fork_z_over = np.maximum(np.abs(wheel_z_difference) - float(params.no_fork_z_distance), 0.0)

    raw: dict[str, np.ndarray] = {
        "termination": np.asarray(terminated, dtype=dtype),
        "leg_joint_acc": np.sum(np.square(qacc[:, :4]), axis=1),
        "leg_joint_vel": np.sum(np.square(dof_vel[:, :4]), axis=1),
        "leg_joint_pair_pos_diff": np.square(dof_pos[:, 0] - dof_pos[:, 2])
        + np.square(dof_pos[:, 1] - dof_pos[:, 3]),
        "joint_torque": np.sum(np.square(torque), axis=1),
        "wheel_acc": np.sum(np.square(qacc[:, 4:]), axis=1),
        "wheel_vel": np.sum(np.square(dof_vel[:, 4:]), axis=1),
        "wheel_power": np.sum(np.maximum(wheel_power, 0.0), axis=1),
        "wheel_air_spin": np.zeros((n,), dtype=dtype),
        "lin_vel_z": np.square(linvel[:, 2]),
        "ang_vel_xy": np.sum(np.square(gyro[:, :2]), axis=1),
        "action_smoothness_leg": np.sum(np.square(action_second_difference[:, :4]), axis=1),
        "action_rate": np.sum(np.square(actions - last_actions), axis=1),
        "action_smoothness_wheel": np.sum(np.square(action_second_difference[:, 4:]), axis=1),
        "flat_orientation_y": np.square(float(params.orientation_y_square_sigma) * pgb_x),
        "flat_orientation_y_v": np.square(
            (
                float(params.orientation_y_amplitude)
                * np.exp(-command_x_sq / float(params.orientation_y_sigma))
                + float(params.orientation_y_bias)
            )
            * pgb_x
        ),
        "flat_orientation_y_exp": np.exp(-np.square(pgb_x) / float(params.orientation_y_exp_sigma)),
        "flat_orientation_x": np.square(float(params.orientation_x_square_sigma) * pgb_y),
        "flat_orientation_x_v": np.square(
            (
                float(params.orientation_x_amplitude)
                * np.exp(-command_x_sq / float(params.orientation_x_sigma))
                + float(params.orientation_x_bias)
            )
            * pgb_y
        ),
        "flat_orientation_x_exp": np.exp(-np.square(pgb_y) / float(params.orientation_x_exp_sigma)),
        "track_lin_vel_xy": np.exp(-np.square(lin_error_limited) / float(params.lin_vel_sigma)),
        "track_lin_vel_xy_tight": np.exp(
            -np.square(lin_error_limited) / float(params.lin_vel_tight_sigma)
        ),
        "track_lin_vel_xy_square": np.square(lin_error * float(params.lin_vel_square_sigma)),
        "track_ang_vel_z": np.exp(-np.square(yaw_error_limited) / float(params.ang_vel_sigma)),
        "track_ang_vel_z_square": np.square(yaw_error * float(params.ang_vel_square_sigma)),
        "stand_still_lin_vel": np.sum(np.abs(linvel[:, :2]), axis=1) * stand_lin,
        "stand_still": (
            np.sum(np.square(linvel[:, :2]), axis=1) * stand_lin + np.square(gyro[:, 2]) * stand_yaw
        ),
        "track_height_exp": np.exp(-np.square(height_error_limited) / float(params.height_sigma)),
        "track_height_exp_soft": np.exp(
            -np.square(height_error_limited) / float(params.height_soft_sigma)
        ),
        "track_height_exp_tight": np.exp(
            -np.square(height_error_limited) / float(params.height_tight_sigma)
        ),
        "track_height_square": np.square(height_error * float(params.height_square_sigma)),
        "no_fork": (np.abs(wheel_x_difference) > float(params.no_fork_distance)).astype(dtype),
        "no_fork_square": np.square(wheel_x_difference * float(params.no_fork_square_sigma)),
        "no_fork_exp": 1.0 - np.exp(-no_fork_over / max(float(params.no_fork_exp_sigma), 1.0e-8)),
        "no_fork_z_exp": 1.0
        - np.exp(-no_fork_z_over / max(float(params.no_fork_z_exp_sigma), 1.0e-8)),
        "undesired_contact": np.asarray(undesired_contact, dtype=dtype),
    }
    if bool(params.vel_height_gate_enabled):
        gate_mode = str(params.vel_height_gate_mode).strip().lower()
        if gate_mode in {"linear_band", "band", "piecewise_linear"}:
            height_abs_error = np.abs(height_error)
            full_error = max(float(params.vel_height_gate_full_error), 0.0)
            zero_error = max(float(params.vel_height_gate_zero_error), full_error)
            width = zero_error - full_error
            if width <= 0.0:
                velocity_height_gate = (height_abs_error <= full_error).astype(dtype)
            else:
                velocity_height_gate = np.clip(
                    (zero_error - height_abs_error) / width,
                    0.0,
                    1.0,
                ).astype(dtype, copy=False)
        else:
            sigma = max(float(params.vel_height_gate_tracker_sigma), 1.0e-6)
            velocity_height_gate = np.exp(-np.square(height_error_limited) / sigma)
        for name in (
            "track_lin_vel_xy",
            "track_lin_vel_xy_tight",
            "track_lin_vel_xy_square",
            "track_ang_vel_z",
            "track_ang_vel_z_square",
        ):
            raw[name] *= velocity_height_gate
    raw["track_height_exp_both_wheels_contact"] = raw["track_height_exp_soft"] * np.all(
        wheel_contact, axis=1
    )

    weighted: dict[str, np.ndarray] = {}
    total = np.zeros((n,), dtype=dtype)
    dt = float(ctrl_dt)
    for name, scale in scales.items():
        value = raw.get(name)
        if value is None:
            value = np.zeros((n,), dtype=dtype)
        term = np.nan_to_num(
            np.asarray(value, dtype=dtype) * float(scale) * dt,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(dtype, copy=False)
        weighted[str(name)] = term
        total += term
    return total, weighted


def apply_source_v14_termination_duration(
    raw_terminate: np.ndarray,
    immediate_terminate: np.ndarray,
    counter: np.ndarray,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the source consecutive-step gate while bypassing numeric failures."""

    raw = np.asarray(raw_terminate, dtype=bool)
    immediate = np.asarray(immediate_terminate, dtype=bool)
    previous = np.asarray(counter, dtype=np.int32)
    if raw.shape != immediate.shape or previous.shape != raw.shape:
        raise ValueError("termination masks and counter must have identical shapes")
    required = max(int(steps), 1)
    delayed = raw & ~immediate
    next_counter = np.where(
        delayed,
        np.minimum(previous + 1, required),
        np.zeros_like(previous),
    ).astype(np.int32, copy=False)
    return (next_counter >= required) | immediate, next_counter


def sample_source_v14_interval(
    intervals: Sequence[Sequence[float]], num_samples: int
) -> np.ndarray:
    """Sample a union of intervals with probability proportional to width."""

    ranges = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    lower = np.minimum(ranges[:, 0], ranges[:, 1])
    upper = np.maximum(ranges[:, 0], ranges[:, 1])
    widths = upper - lower
    if ranges.shape[0] == 0 or np.sum(widths) <= 0.0:
        raise ValueError("intervals must contain at least one non-zero-width interval")
    choices = np.random.choice(ranges.shape[0], size=int(num_samples), p=widths / np.sum(widths))
    return np.random.uniform(lower[choices], upper[choices], size=int(num_samples))


def sample_source_v14_material_buckets(
    *,
    num_samples: int,
    num_slots: int,
    static_friction_range: Sequence[float],
    dynamic_friction_range: Sequence[float],
    restitution_range: Sequence[float],
    num_buckets: int = 64,
    make_consistent: bool = True,
) -> np.ndarray:
    """Sample source material buckets, then assign one bucket per shape slot."""

    ranges = np.asarray(
        (static_friction_range, dynamic_friction_range, restitution_range),
        dtype=np.float64,
    )
    if ranges.shape != (3, 2) or not np.all(np.isfinite(ranges)):
        raise ValueError("source material ranges must contain three finite low/high pairs")
    bucket_count = int(num_buckets)
    if bucket_count <= 0:
        raise ValueError("source material num_buckets must be positive")
    n = int(num_samples)
    slots = int(num_slots)
    if n < 0 or slots < 0:
        raise ValueError("source material sample and slot counts must be non-negative")
    lower = np.minimum(ranges[:, 0], ranges[:, 1])
    upper = np.maximum(ranges[:, 0], ranges[:, 1])
    buckets = np.random.uniform(lower, upper, size=(bucket_count, 3))
    if make_consistent:
        buckets[:, 1] = np.minimum(buckets[:, 0], buckets[:, 1])
    bucket_ids = np.random.randint(0, bucket_count, size=(n, slots))
    return buckets[bucket_ids].astype(get_global_dtype(), copy=False)


def sample_source_v14_commands(
    *,
    num_samples: int,
    current_yaw: np.ndarray,
    episode_steps: np.ndarray,
    ctrl_dt: float,
    training_iteration: int,
    normal_low: Sequence[float],
    normal_high: Sequence[float],
    standing_probability: float,
    heading_probability: float,
    heading_range: Sequence[float],
    heading_stiffness: float,
    special_mode_min_episode_time: float,
    special_mode_start_iterations: Sequence[int],
    special_mode_probabilities: Sequence[float] = (0.15, 0.15, 0.30),
    gimbal_mode_probability: float = 0.0,
    gimbal_mode_start_iteration: int = 0,
    zero_command_probability: float = 0.0,
    zero_command_start_iteration: int = 0,
) -> dict[str, np.ndarray]:
    """Sample pinned standing/heading/spin/dash/gimbal command semantics.

    Mode ids ``0..2`` are the ordinary spin-low, spin-mid and dash buckets.
    Id ``3`` is profile-specific: source-v1 uses it for its zero-command
    bucket, while source-v2 reserves it for gimbal spin/translation.  The two
    meanings are mutually exclusive.  For v2 its yaw-rate command is sampled
    here and the task-mode owner fills only the gimbal-frame translation,
    heading and height fields.
    """

    n = int(num_samples)
    dtype = get_global_dtype()
    yaw = np.asarray(current_yaw, dtype=dtype).reshape(n)
    steps = np.asarray(episode_steps).reshape(n)
    low = np.asarray(normal_low, dtype=np.float64).reshape(3)
    high = np.asarray(normal_high, dtype=np.float64).reshape(3)
    lower = np.minimum(low, high)
    upper = np.maximum(low, high)
    heading_bounds = np.sort(np.asarray(heading_range, dtype=np.float64).reshape(2))
    starts = tuple(int(value) for value in special_mode_start_iterations)
    if len(starts) != 3:
        raise ValueError("special_mode_start_iterations must contain spin-low, spin-mid, dash")
    probabilities = np.asarray(special_mode_probabilities, dtype=np.float64).reshape(-1)
    if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "special_mode_probabilities must contain three finite values for "
            "spin-low, spin-mid and dash"
        )
    gimbal_probability = float(gimbal_mode_probability)
    zero_probability = float(zero_command_probability)
    if (
        np.any(probabilities < 0.0)
        or not np.isfinite(gimbal_probability)
        or gimbal_probability < 0.0
        or not np.isfinite(zero_probability)
        or zero_probability < 0.0
    ):
        raise ValueError("source special-mode probabilities must be finite and non-negative")
    if gimbal_probability > 0.0 and zero_probability > 0.0:
        raise ValueError("source mode id 3 cannot be both gimbal and zero-command")
    if float(np.sum(probabilities)) + gimbal_probability + zero_probability > 1.0 + 1.0e-9:
        raise ValueError("source special-mode probabilities cannot sum to more than one")
    mode3_is_zero_command = zero_probability > 0.0
    mode3_probability = zero_probability if mode3_is_zero_command else gimbal_probability
    mode3_start_iteration = (
        int(zero_command_start_iteration)
        if mode3_is_zero_command
        else int(gimbal_mode_start_iteration)
    )

    commands = np.zeros((n, 3), dtype=dtype)
    heading_targets = np.asarray(
        np.random.uniform(heading_bounds[0], heading_bounds[1], size=n), dtype=dtype
    )
    standing = np.random.uniform(size=n) <= np.clip(float(standing_probability), 0.0, 1.0)
    heading = np.zeros((n,), dtype=bool)
    special_mode = np.full((n,), -1, dtype=np.int8)

    nonstanding_ids = np.flatnonzero(~standing)
    ready = np.asarray(steps[nonstanding_ids], dtype=np.float64) * float(ctrl_dt) >= float(
        special_mode_min_episode_time
    )
    eligible_ids = nonstanding_ids[ready]
    mode_specs = (
        # id, probability, iteration start
        (0, float(probabilities[0]), starts[0]),
        (1, float(probabilities[1]), starts[1]),
        (2, float(probabilities[2]), starts[2]),
        (3, mode3_probability, mode3_start_iteration),
    )
    active = [spec for spec in mode_specs if spec[1] > 0.0 and int(training_iteration) >= spec[2]]
    if eligible_ids.size and active:
        assignment = np.random.uniform(size=eligible_ids.size)
        cursor = 0.0
        for index in np.random.permutation(len(active)):
            mode_id, probability, _start = active[int(index)]
            mask = (assignment >= cursor) & (assignment < cursor + probability)
            special_mode[eligible_ids[mask]] = int(mode_id)
            cursor += probability

    for mode_id in range(4):
        ids = np.flatnonzero(special_mode == mode_id)
        if not ids.size:
            continue
        if mode_id == 0:
            commands[ids, 0] = np.random.uniform(-0.1, 0.1, size=ids.size)
            commands[ids, 2] = sample_source_v14_interval(
                ((2.0 * np.pi, 3.25 * np.pi), (-3.25 * np.pi, -2.0 * np.pi)),
                ids.size,
            )
        elif mode_id == 1:
            commands[ids, 0] = np.random.uniform(-0.1, 0.1, size=ids.size)
            commands[ids, 2] = sample_source_v14_interval(
                ((3.25 * np.pi, 4.5 * np.pi), (-4.5 * np.pi, -3.25 * np.pi)),
                ids.size,
            )
        elif mode_id == 2:
            commands[ids, 0] = sample_source_v14_interval(((2.0, 3.0), (-3.0, -2.0)), ids.size)
            commands[ids, 2] = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, size=ids.size)
        elif not mode3_is_zero_command:
            commands[ids, 2] = sample_source_v14_interval(
                ((2.4 * np.pi, 3.6 * np.pi), (-3.6 * np.pi, -2.4 * np.pi)),
                ids.size,
            )

    normal_ids = np.flatnonzero((~standing) & (special_mode < 0))
    if normal_ids.size:
        commands[normal_ids] = np.random.uniform(lower, upper, size=(normal_ids.size, 3))
        commands[normal_ids, 1] = 0.0
        heading[normal_ids] = np.random.uniform(size=normal_ids.size) <= np.clip(
            float(heading_probability), 0.0, 1.0
        )
        heading_ids = normal_ids[heading[normal_ids]]
        if heading_ids.size:
            error = (heading_targets[heading_ids] - yaw[heading_ids] + np.pi) % (
                2.0 * np.pi
            ) - np.pi
            commands[heading_ids, 2] = np.clip(float(heading_stiffness) * error, lower[2], upper[2])

    commands[standing] = 0.0
    return {
        "commands": commands.astype(dtype, copy=False),
        "heading_commands": heading_targets,
        "is_standing_env": standing,
        "is_heading_env": heading,
        "special_mode_id": special_mode,
    }


__all__ = [
    "SOURCE_V14_PRIVILEGED_POLICY_PERMUTATION",
    "SOURCE_V14_RESET_JOINT_NAMES",
    "SOURCE_V14_FLAT_V1_REWARD_SCALES",
    "SOURCE_V14_REWARD_SCALES",
    "SOURCE_V14_ROUGH_V0_REWARD_SCALES",
    "SOURCE_V14_ROUGH_V1_REWARD_SCALES",
    "SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES",
    "SOURCE_V14_RESET_CONTACT_BODY_NAMES",
    "SOURCE_V14_GUIDE_BODY_NAMES",
    "SOURCE_V14_WHEEL_BODY_NAMES",
    "SOURCE_V14_LEG_MASS_BODY_NAMES",
    "SourceV14RewardParameters",
    "apply_source_v14_termination_duration",
    "build_source_v14_height_signals",
    "build_source_v14_critic_observation",
    "compute_source_v14_reward",
    "sample_source_v14_commands",
    "sample_source_v14_interval",
    "sample_source_v14_material_buckets",
    "source_v14_inverse_kinematics",
]
