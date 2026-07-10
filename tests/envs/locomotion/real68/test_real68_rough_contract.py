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
        assert state.obs["obs"].shape == (2, 32)
        assert state.obs["critic"].shape == (2, critic_dim)
        reset_obs, _ = env.reset(np.asarray([0], dtype=np.int32))
        assert reset_obs["obs"].shape == (1, 32)
        assert reset_obs["critic"].shape == (1, critic_dim)
        step_state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert step_state.obs["obs"].shape == (2, 32)
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


def test_real68_rough_uses_rough_termination_thresholds():
    ensure_registries()
    env = registry.make(
        "Real68BalanceRough",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "termination_config": {
                "fall_termination": True,
                "min_up_proj": 0.2,
                "min_base_height": 0.0,
                "nonwheel_contact_termination": False,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
                "max_tilt_cos": 0.9,
            },
        },
    )
    try:
        env._nonwheel_contacts.fill(0.0)
        upright_enough = env._compute_termination_causes(
            np.asarray([[0.0, 0.0, 0.25], [0.0, 0.0, 0.25]], dtype=np.float32)
        )
        assert not np.any(upright_enough["terminated"])

        tilted = env._compute_termination_causes(
            np.asarray([[0.0, 0.0, 0.15], [0.0, 0.0, 0.15]], dtype=np.float32)
        )
        assert np.all(tilted["terminated"])
    finally:
        env.close()


def test_real68_rough_recovery_protects_fallen_contact_and_low_height():
    ensure_registries()
    env = registry.make(
        "Real68BalanceRough",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={
            "recovery": {"enabled": True, "fall_detect_cos": 0.9, "timeout_seconds": 1.0},
            "termination_config": {
                "fall_termination": True,
                "min_up_proj": 0.2,
                "min_base_height": 10.0,
                "nonwheel_contact_termination": True,
                "nonwheel_contact_max_steps": 1,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        gravity = np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32)
        info = {
            "commands": np.asarray([[0.5, 0.0, 1.0]], dtype=np.float32),
            "tracking_commands": np.asarray([[0.5, 0.0, 1.0]], dtype=np.float32),
        }
        env._nonwheel_contacts.fill(1.0)
        env._recovery_eligible[:] = True
        env._update_recovery_state(info, gravity)
        causes = env._compute_termination_causes(gravity)

        assert env._recovery_active[0]
        assert not causes["terminated"][0]
        np.testing.assert_allclose(info["commands"], np.zeros((1, 3)))
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
                "min_segment_count": 1,
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
        np.testing.assert_allclose(commands[:, 2], 0.0)

        env._record_command_segment_stats(
            cmd_x=0.3,
            cmd_yaw=0.0,
            mean_abs_vx=0.18,
            mean_signed_vx=0.18,
            mean_abs_wz=0.7,
            vx_error=0.2,
            wz_error=0.7,
            mean_nonwheel_contact=0.0,
            mean_tilt=0.0,
            mean_tilt_angle_deg=0.0,
            mean_height_violation=0.0,
            segment_steps=100,
        )
        env._update_command_curriculum()
        assert env._command_curriculum_vx_progress == pytest.approx(0.25)
        assert env._command_curriculum_yaw_progress == pytest.approx(0.0)
        assert env._command_curriculum_low[0] == pytest.approx(0.1)
        assert env._command_curriculum_high[0] > 0.35
        assert env._command_curriculum_high[2] == pytest.approx(0.3)

        env._command_curriculum_vx_progress = 0.75
        env._refresh_command_curriculum_limits()
        env._reset_command_curriculum_stats()
        env._record_command_segment_stats(
            cmd_x=1.5,
            cmd_yaw=1.5,
            mean_abs_vx=0.9,
            mean_signed_vx=0.9,
            mean_abs_wz=0.7,
            vx_error=0.2,
            wz_error=0.7,
            mean_nonwheel_contact=0.0,
            mean_tilt=0.0,
            mean_tilt_angle_deg=0.0,
            mean_height_violation=0.0,
            segment_steps=100,
        )
        env._update_command_curriculum()
        assert env._command_curriculum_yaw_progress == pytest.approx(0.25)
        assert env._command_curriculum_low[0] < 0.1
        assert env._command_curriculum_high[2] > 0.3

        commands = env.sample_velocity_commands(256)
        assert float(commands[:, 2].min()) >= -5.0
        assert float(commands[:, 2].max()) <= 5.0
        assert np.any(np.abs(commands[:, 2]) > 0.0)
    finally:
        env.close()


def test_real68_command_curriculum_can_delay_reverse_until_completion():
    ensure_registries()
    env = registry.make(
        "Real68BalanceRough",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "command_curriculum": {
                "enabled": True,
                "initial_vel_limit": [[0.35, 0.0, 0.0], [0.45, 0.0, 0.0]],
                "final_vel_limit": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "reverse_unlock_vx_progress": 1.0,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        env._command_curriculum_vx_progress = 0.8
        env._refresh_command_curriculum_limits()
        assert env._command_curriculum_low[0] == pytest.approx(0.35)

        env._command_curriculum_vx_progress = 1.0
        env._refresh_command_curriculum_limits()
        assert env._command_curriculum_low[0] == pytest.approx(-1.0)
    finally:
        env.close()


def test_real68_command_curriculum_can_schedule_standing_commands():
    ensure_registries()
    env = registry.make(
        "Real68BalanceRough",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "command_curriculum": {
                "enabled": True,
                "initial_vel_limit": [[0.35, 0.0, 0.0], [0.45, 0.0, 0.0]],
                "final_vel_limit": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "standing_prob_initial": 0.5,
                "standing_prob_final": 0.3,
                "standing_decay_vx_progress": 1.0,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        env._command_curriculum_vx_progress = 0.0
        assert env._standing_command_probability() == pytest.approx(0.5)

        env._command_curriculum_vx_progress = 0.3
        assert env._standing_command_probability() == pytest.approx(0.44)

        env._command_curriculum_vx_progress = 1.0
        assert env._standing_command_probability() == pytest.approx(0.3)
    finally:
        env.close()
