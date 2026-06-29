from __future__ import annotations

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries


pytest.importorskip("mujoco", reason="mujoco not installed")


def test_real68_rough_env_reset_and_step_contract():
    ensure_registries()
    env = registry.make(
        "Real68BalanceRough",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        state = env.init_state()
        critic_dim = 45 + env._height_scan_dim
        assert env._height_scan_dim > 0
        assert set(state.obs) == {"obs", "critic"}
        assert state.obs["obs"].shape == (2, 29)
        assert state.obs["critic"].shape == (2, critic_dim)
        reset_obs, _ = env.reset(np.asarray([0], dtype=np.int32))
        assert reset_obs["obs"].shape == (1, 29)
        assert reset_obs["critic"].shape == (1, critic_dim)
        step_state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert step_state.obs["obs"].shape == (2, 29)
        assert step_state.obs["critic"].shape == (2, critic_dim)
        assert step_state.reward.shape == (2,)
        assert step_state.terminated.shape == (2,)
        assert step_state.truncated.shape == (2,)
    finally:
        env.close()
