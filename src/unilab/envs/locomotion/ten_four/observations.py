from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTOR_ONE_STEP_DIM = 28
CRITIC_ONE_STEP_DIM = 59
OBSERVATION_SCHEMA = "ten_four_balance_v2"
ACTION_SCHEMA = "ten_four_direct_mixed_v1"


@dataclass(frozen=True)
class TenFourObservationSlices:
    local_linvel: slice = slice(0, 3)
    clean_actor: slice = slice(3, 31)
    qacc: slice = slice(31, 37)
    contacts: slice = slice(37, 44)
    dynamics: slice = slice(44, 59)


CRITIC_SLICES = TenFourObservationSlices()


def build_actor_observation(
    *,
    gyro: np.ndarray,
    gravity: np.ndarray,
    accel: np.ndarray,
    posture_diff: np.ndarray,
    posture_vel: np.ndarray,
    wheel_vel: np.ndarray,
    last_actions: np.ndarray,
    commands: np.ndarray,
    height_command: np.ndarray,
    dtype: np.dtype | type,
) -> np.ndarray:
    actor = np.concatenate(
        [
            gyro,
            -gravity,
            accel,
            posture_diff,
            posture_vel,
            wheel_vel,
            last_actions,
            commands[:, (0, 2)],
            height_command,
        ],
        axis=1,
        dtype=dtype,
    )
    if actor.shape[1] != ACTOR_ONE_STEP_DIM:
        raise ValueError(
            f"{OBSERVATION_SCHEMA} actor observation must have "
            f"{ACTOR_ONE_STEP_DIM} columns, got {actor.shape[1]}"
        )
    return actor


def build_critic_observation(
    *,
    local_linvel: np.ndarray,
    clean_actor: np.ndarray,
    qacc: np.ndarray,
    wheel_contacts: np.ndarray,
    nonwheel_contacts: np.ndarray,
    dynamics: np.ndarray,
    dtype: np.dtype | type,
) -> np.ndarray:
    critic = np.concatenate(
        [
            local_linvel,
            clean_actor,
            qacc,
            wheel_contacts,
            nonwheel_contacts,
            dynamics,
        ],
        axis=1,
        dtype=dtype,
    )
    if critic.shape[1] != CRITIC_ONE_STEP_DIM:
        raise ValueError(
            f"{OBSERVATION_SCHEMA} critic observation must have "
            f"{CRITIC_ONE_STEP_DIM} columns, got {critic.shape[1]}"
        )
    return critic


def append_history(history: np.ndarray, frame: np.ndarray, frame_dim: int) -> np.ndarray:
    if history.shape[1] == frame_dim:
        history[:] = frame
        return history
    history[:, :-frame_dim] = history[:, frame_dim:]
    history[:, -frame_dim:] = frame
    return history


def fill_history(history: np.ndarray, frame: np.ndarray, frame_dim: int) -> np.ndarray:
    history[:] = np.tile(frame, (1, history.shape[1] // frame_dim))
    return history
