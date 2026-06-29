from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext


def _ctx(
    commands: np.ndarray,
    linvel_x: np.ndarray,
    gyro_z: np.ndarray | None = None,
) -> RewardContext:
    num_envs = commands.shape[0]
    linvel = np.zeros((num_envs, 3), dtype=np.float32)
    linvel[:, 0] = linvel_x
    gyro = np.zeros((num_envs, 3), dtype=np.float32)
    if gyro_z is not None:
        gyro[:, 2] = gyro_z
    return RewardContext(
        info={"commands": commands.astype(np.float32)},
        linvel=linvel,
        gyro=gyro,
        dof_pos=np.zeros((num_envs, 1), dtype=np.float32),
        num_envs=num_envs,
    )


def test_forward_progress_and_under_speed_follow_signed_x_command():
    ctx = _ctx(
        np.asarray(
            [
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        np.asarray([1.0, -1.0, -1.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(
        rewards.forward_progress(ctx),
        np.asarray([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        rewards.under_speed(ctx),
        np.asarray([0.5, 0.5, 1.5, 1.0, 0.0], dtype=np.float32),
    )


def test_yaw_rate_when_uncommanded_only_penalizes_near_zero_yaw_command():
    ctx = _ctx(
        np.asarray(
            [
                [0.2, 0.0, 0.0],
                [0.2, 0.0, 0.04],
                [0.2, 0.0, 0.2],
            ]
        ),
        np.asarray([0.0, 0.0, 0.0]),
        gyro_z=np.asarray([1.0, -2.0, 3.0]),
    )

    np.testing.assert_allclose(
        rewards.yaw_rate_when_uncommanded(ctx),
        np.asarray([1.0, 4.0, 0.0], dtype=np.float32),
    )
