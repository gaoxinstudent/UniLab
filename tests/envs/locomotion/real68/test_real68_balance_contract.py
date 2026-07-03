from __future__ import annotations

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.real68.base import POSTURE_INDICES, WHEEL_INDICES

pytest.importorskip("mujoco", reason="mujoco not installed")


def test_real68_scene_compiles_and_has_expected_counts():
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
    )
    assert model.nu == 6
    assert model.nsensor >= 30
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home") >= 0


def test_real68_balance_env_reset_and_step_contract():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        state = env.init_state()
        assert set(state.obs) == {"obs", "critic"}
        assert state.obs["obs"].shape == (2, 32)
        assert state.obs["critic"].shape == (2, 45)
        reset_obs, _ = env.reset(np.asarray([0], dtype=np.int32))
        assert reset_obs["obs"].shape == (1, 32)
        assert reset_obs["critic"].shape == (1, 45)
        step_state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert step_state.obs["obs"].shape == (2, 32)
        assert step_state.reward.shape == (2,)
        assert step_state.terminated.shape == (2,)
        log = step_state.info.get("log", {})
        assert "metrics/mean_abs_vx" in log
        assert "metrics/mean_abs_cmd_x" in log
    finally:
        env.close()


def test_real68_joint_penalties_ignore_wheel_positions_and_power():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        default = np.broadcast_to(env.default_angles, (2, env._num_action)).copy()
        wheel_shifted = default.copy()
        wheel_shifted[:, WHEEL_INDICES] += np.asarray([10.0, -10.0])
        ctx_default = RewardContext(
            info={"commands": np.zeros((2, 3))},
            linvel=np.zeros((2, 3)),
            gyro=np.zeros((2, 3)),
            dof_pos=default,
            dof_vel=np.zeros_like(default),
            num_envs=2,
            default_angles=env.default_angles,
        )
        ctx_wheel_shifted = RewardContext(
            info={"commands": np.zeros((2, 3))},
            linvel=np.zeros((2, 3)),
            gyro=np.zeros((2, 3)),
            dof_pos=wheel_shifted,
            dof_vel=np.zeros_like(default),
            num_envs=2,
            default_angles=env.default_angles,
        )
        np.testing.assert_allclose(
            env._reward_joint_pos_penalty(ctx_default),
            env._reward_joint_pos_penalty(ctx_wheel_shifted),
        )

        wheel_vel = np.zeros_like(default)
        wheel_vel[:, WHEEL_INDICES] = 100.0
        wheel_torque = np.zeros_like(default)
        wheel_torque[:, WHEEL_INDICES] = 100.0
        ctx_wheel_power = RewardContext(
            info={"commands": np.zeros((2, 3)), "torques": wheel_torque},
            linvel=np.zeros((2, 3)),
            gyro=np.zeros((2, 3)),
            dof_pos=default,
            dof_vel=wheel_vel,
            num_envs=2,
            default_angles=env.default_angles,
        )
        np.testing.assert_allclose(env._reward_joint_power(ctx_wheel_power), np.zeros((2,)))
    finally:
        env.close()


def test_real68_command_lean_targets_reduce_orientation_and_posture_penalties():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
                "command_lean": {
                    "enabled": True,
                    "gravity_x_gain": 0.12,
                    "gravity_x_limit": 0.12,
                    "hip_gain": 0.18,
                    "hip_limit": 0.18,
                    "calf_gain": 0.12,
                    "calf_limit": 0.12,
                },
            },
        },
    )
    try:
        cmd_x = np.asarray([0.8], dtype=np.float32)
        commands = np.asarray([[0.8, 0.0, 0.0]], dtype=np.float32)
        target_gx = env._command_lean_gravity_target(cmd_x)
        target_posture = env._command_target_posture(cmd_x)
        default = env.default_angles[POSTURE_INDICES][None, :].copy()

        upright_ctx = RewardContext(
            info={"commands": commands},
            linvel=np.zeros((1, 3), dtype=np.float32),
            gyro=np.zeros((1, 3), dtype=np.float32),
            gravity=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            dof_pos=np.broadcast_to(env.default_angles, (1, env._num_action)).copy(),
            dof_vel=np.zeros((1, env._num_action), dtype=np.float32),
            num_envs=1,
            default_angles=env.default_angles,
        )
        leaned_dof = np.broadcast_to(env.default_angles, (1, env._num_action)).copy()
        leaned_dof[:, POSTURE_INDICES] = default + target_posture
        leaned_ctx = RewardContext(
            info={"commands": commands},
            linvel=np.zeros((1, 3), dtype=np.float32),
            gyro=np.zeros((1, 3), dtype=np.float32),
            gravity=np.asarray([[target_gx[0], 0.0, 1.0]], dtype=np.float32),
            dof_pos=leaned_dof,
            dof_vel=np.zeros((1, env._num_action), dtype=np.float32),
            num_envs=1,
            default_angles=env.default_angles,
        )

        assert float(env._reward_orientation(leaned_ctx)[0]) < float(env._reward_orientation(upright_ctx)[0])
        assert float(env._reward_posture(leaned_ctx)[0]) < float(env._reward_posture(upright_ctx)[0])
        assert float(env._reward_joint_pos_penalty(leaned_ctx)[0]) < float(
            env._reward_joint_pos_penalty(upright_ctx)[0]
        )
    finally:
        env.close()


def test_real68_command_curriculum_bootstraps_standing_before_velocity_commands():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=8,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "command_curriculum": {
                "enabled": True,
                "initial_vel_limit": [[0.35, 0.0, 0.0], [0.45, 0.0, 0.0]],
                "final_vel_limit": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "update_interval_logs": 1,
                "standing_bootstrap_enabled": True,
                "standing_bootstrap_min_segments": 2,
                "standing_bootstrap_min_segment_steps": 4,
                "standing_bootstrap_max_wz_error": 0.2,
                "standing_bootstrap_max_nonwheel_contact": 0.01,
                "standing_prob_initial": 0.0,
                "standing_prob_final": 0.0,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        assert env._standing_command_probability() == pytest.approx(1.0)
        np.testing.assert_allclose(env.sample_velocity_commands(8), np.zeros((8, 3)))

        env._standing_segment_count = 2
        env._standing_segment_steps_sum = 8.0
        env._standing_segment_wz_error_sum = 0.2
        env._standing_segment_nonwheel_contact_sum = 0.0
        env._update_command_curriculum()

        assert env._standing_bootstrap_complete is True
        assert env._standing_command_probability() == pytest.approx(0.0)
        commands = env.sample_velocity_commands(8)
        assert np.all(commands[:, 0] >= 0.35)
        assert np.all(commands[:, 0] <= 0.45)
        np.testing.assert_allclose(commands[:, 1:], np.zeros((8, 2)))
    finally:
        env.close()
