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
        log = step_state.info.get("log", {})
        assert "metrics/mean_abs_vx" in log
        assert "metrics/mean_abs_cmd_x" in log
        assert "command_curriculum/progress" in log
        assert "terrain_curriculum/mean_level" in log
    finally:
        env.close()


def test_real68_rough_curriculum_logs_appear_after_done():
    ensure_registries()
    env = registry.make(
        "Real68BalanceRough",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "command_curriculum": {"enabled": False},
            "terrain_curriculum": {"enabled": True},
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        state = env.init_state()
        env.apply_action(np.zeros((2, 6), dtype=np.float32), state)
        state.truncated[:] = True
        out = env.update_state(state)
        log = out.info.get("log", {})
        for key in (
            "terrain_curriculum/mean_level",
            "terrain_curriculum/max_level",
            "terrain_curriculum/min_level",
            "terrain_curriculum/mean_walked",
            "terrain_curriculum/num_promoted",
            "terrain_curriculum/num_demoted",
            "terrain_curriculum/num_skipped",
        ):
            assert key in log
    finally:
        env.close()


def test_real68_command_curriculum_starts_small_and_expands():
    ensure_registries()
    env = registry.make(
        "Real68BalanceRough",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "command_curriculum": {
                "enabled": True,
                "initial_vel_limit": [[0.1, 0.0, -0.3], [0.35, 0.0, 0.3]],
                "final_vel_limit": [[-2.0, 0.0, -5.0], [3.0, 0.0, 5.0]],
                "vx_step": 0.25,
                "yaw_step": 0.25,
                "update_interval_logs": 1,
                "min_speed_ratio": 0.45,
                "max_vx_error": 0.3,
                "max_wz_error": 0.9,
                "yaw_unlock_vx_progress": 0.6,
                "reverse_unlock_vx_progress": 0.7,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        commands = env.sample_velocity_commands(256)
        assert float(commands[:, 0].min()) >= 0.1 - 1.0e-6
        assert float(commands[:, 0].max()) <= 0.35 + 1.0e-6
        assert float(commands[:, 2].min()) >= -0.3 - 1.0e-6
        assert float(commands[:, 2].max()) <= 0.3 + 1.0e-6

        env._update_command_curriculum(
            mean_abs_vx=0.12,
            mean_abs_cmd_x=0.2,
            vx_error=0.2,
            wz_error=0.7,
        )
        assert env._command_curriculum_vx_progress == pytest.approx(0.25)
        assert env._command_curriculum_yaw_progress == pytest.approx(0.0)
        assert env._command_curriculum_low[0] == pytest.approx(0.1)
        assert env._command_curriculum_high[0] > 0.35
        assert env._command_curriculum_high[2] == pytest.approx(0.3)

        env._command_curriculum_vx_progress = 0.75
        env._update_command_curriculum(
            mean_abs_vx=0.12,
            mean_abs_cmd_x=0.2,
            vx_error=0.2,
            wz_error=0.7,
        )
        assert env._command_curriculum_yaw_progress == pytest.approx(0.25)
        assert env._command_curriculum_low[0] < 0.1
        assert env._command_curriculum_high[2] > 0.3
    finally:
        env.close()
