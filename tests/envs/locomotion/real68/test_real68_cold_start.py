from __future__ import annotations

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.real68.base import (
    COLD_START_BASE_HEIGHT,
    DEFAULT_ACTIVE_ANGLES,
)

pytest.importorskip("mujoco", reason="mujoco not installed")


def _make_env(cold_start_fraction: float, num_envs: int = 4):
    return registry.make(
        "Real68BalanceFlat",
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override={
            "recovery": {"enabled": False},
            "cold_start": {
                "enabled": True,
                "fraction": cold_start_fraction,
                "joint_noise": 0.0,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )


def test_real68_cold_start_reset_starts_near_zero_joints_on_ground():
    """With fraction=1.0 every reset must start at the power-on pose: joints ~0,
    wheels on the ground (base at COLD_START_BASE_HEIGHT), and at rest."""
    ensure_registries()
    env = _make_env(cold_start_fraction=1.0)
    try:
        env.reset(np.arange(4, dtype=np.int32))
        assert np.all(env._cold_start_for_env)

        # Active joints near zero (the startup pose), not the keyframe stance.
        dof_pos = env.get_dof_pos()
        np.testing.assert_allclose(dof_pos, np.zeros_like(dof_pos), atol=1e-4)
        # Joints at rest (power-on).
        dof_vel = env.get_dof_vel()
        np.testing.assert_allclose(dof_vel, np.zeros_like(dof_vel), atol=1e-4)
        # Wheels on the ground at the measured cold-start height.
        base_z = np.asarray(env._backend.get_base_pos())[:, 2]
        np.testing.assert_allclose(base_z, np.full((4,), COLD_START_BASE_HEIGHT), atol=1e-3)
    finally:
        env.close()


def test_real68_cold_start_reset_plan_carries_flag_and_override():
    """build_reset_plan must mark cold-start envs in info_updates and produce a
    near-zero qpos / zero qvel for them, while fraction=0 keeps the keyframe."""
    ensure_registries()
    env = _make_env(cold_start_fraction=1.0)
    try:
        env_ids = np.arange(4, dtype=np.int32)
        plan = env._dr_manager._provider.build_reset_plan(env, env_ids)
        assert np.all(plan.info_updates["cold_start"])
        # All 14 joints (active + passive) zeroed.
        np.testing.assert_allclose(plan.qpos[:, 7:], np.zeros_like(plan.qpos[:, 7:]), atol=1e-6)
        np.testing.assert_allclose(
            plan.qpos[:, 2], np.full((4,), COLD_START_BASE_HEIGHT), atol=1e-6
        )
        # Power-on at rest.
        np.testing.assert_allclose(plan.qvel, np.zeros_like(plan.qvel), atol=1e-6)
    finally:
        env.close()

    env0 = _make_env(cold_start_fraction=0.0)
    try:
        plan0 = env0._dr_manager._provider.build_reset_plan(env0, np.arange(4, dtype=np.int32))
        assert not np.any(plan0.info_updates["cold_start"])
        # Default (non-cold) resets keep the keyframe stance.
        active = env0._active_qpos_indices
        np.testing.assert_allclose(
            plan0.qpos[:, active],
            np.tile(DEFAULT_ACTIVE_ANGLES, (4, 1)),
            atol=1e-6,
        )
    finally:
        env0.close()


def test_real68_cold_start_metric_appears_in_log():
    ensure_registries()
    env = _make_env(cold_start_fraction=1.0, num_envs=2)
    try:
        env.reset(np.arange(2, dtype=np.int32))
        state = env.init_state()
        env.apply_action(np.zeros((2, 6), dtype=np.float32), state)
        out = env.update_state(state)
        log = out.info.get("log", {})
        assert "metrics/cold_start_stand_ratio" in log
        assert "metrics/cold_start_base_height" in log
        assert log["metrics/cold_start_n"] == 2
    finally:
        env.close()


def test_real68_rough_cold_start_reset_starts_near_zero_joints():
    """The unified/rough env (what the real68_balance task trains) must also
    apply the cold-start channel in its own build_reset_plan."""
    ensure_registries()
    env = registry.make(
        "Real68BalanceRough",
        num_envs=4,
        sim_backend="mujoco",
        env_cfg_override={
            "recovery": {"enabled": False},
            "cold_start": {"enabled": True, "fraction": 1.0, "joint_noise": 0.0},
            "command_curriculum": {"enabled": False},
            "terrain_curriculum": {"enabled": False},
            "reward_config": {"scales": {"alive": 1.0}, "tracking_sigma": 0.25},
        },
    )
    try:
        env.reset(np.arange(4, dtype=np.int32))
        assert np.all(env._cold_start_for_env)
        dof_pos = env.get_dof_pos()
        np.testing.assert_allclose(dof_pos, np.zeros_like(dof_pos), atol=1e-4)
        dof_vel = env.get_dof_vel()
        np.testing.assert_allclose(dof_vel, np.zeros_like(dof_vel), atol=1e-4)
        # Base is placed on the terrain ground at the cold-start height; joints
        # (not absolute base height) are what define the startup pose.
        base_z = np.asarray(env._backend.get_base_pos())[:, 2]
        origins = env._spawn.origins_for(np.arange(4, dtype=np.int32))
        np.testing.assert_allclose(base_z, origins[:, 2] + COLD_START_BASE_HEIGHT, atol=1e-3)
    finally:
        env.close()
