from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext


def _ctx(commands: np.ndarray, linvel_x: np.ndarray) -> RewardContext:
    num_envs = commands.shape[0]
    linvel = np.zeros((num_envs, 3), dtype=np.float32)
    linvel[:, 0] = linvel_x
    return RewardContext(
        info={"commands": commands.astype(np.float32)},
        linvel=linvel,
        gyro=np.zeros((num_envs, 3), dtype=np.float32),
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
