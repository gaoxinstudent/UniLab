from __future__ import annotations

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.real68.base import (
    DEFAULT_ACTIVE_ANGLES,
    POSTURE_INDICES,
    SYMMETRIC_STANDING_ACTIVE_ANGLES,
    WHEEL_INDICES,
)

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


def test_real68_deployment_observation_uses_wheel_odometry_when_enabled():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "sensor": {"use_wheel_odometry": True},
            "reward_config": {"scales": {"alive": 1.0}, "tracking_sigma": 0.25},
        },
    )
    try:
        dof_pos = np.broadcast_to(env.default_angles, (1, env._num_action)).copy()
        dof_vel = np.zeros_like(dof_pos)
        dof_vel[:, WHEEL_INDICES] = -10.0
        obs = env._compute_obs(
            {"commands": np.zeros((1, 3), dtype=np.float32)},
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            dof_pos,
            dof_vel,
        )
        np.testing.assert_allclose(obs["obs"][:, :3], [[0.0, 0.6, 0.0]])
    finally:
        env.close()


def test_real68_actor_height_command_does_not_require_measured_base_height():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "height_command": {"observation_reference_height": 0.23},
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
                "base_height_target": 0.23,
            },
        },
    )
    try:
        dof_pos = np.broadcast_to(env.default_angles, (1, env._num_action)).copy()
        obs = env._compute_obs(
            {
                "commands": np.zeros((1, 3), dtype=np.float32),
                "height_commands": np.asarray([0.25], dtype=np.float32),
            },
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            dof_pos,
            np.zeros_like(dof_pos),
        )
        assert obs["obs"][0, -1] == pytest.approx(0.02)
        assert obs["critic"][0, 28] != pytest.approx(obs["obs"][0, -1])
    finally:
        env.close()


def test_real68_recovery_zeros_commands_and_suppresses_fall_termination():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "recovery": {
                "enabled": True,
                "fall_detect_cos": 0.9,
                "upright_cos": 0.96,
                "upright_hold_seconds": 0.04,
                "timeout_seconds": 1.0,
            },
            "termination_config": {
                "fall_termination": True,
                "nonwheel_contact_termination": True,
                "nonwheel_contact_max_steps": 1,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
                "max_tilt_cos": 0.9,
            },
        },
    )
    try:
        tracking = np.asarray([[0.8, 0.0, 1.2]], dtype=np.float32)
        info = {"commands": tracking.copy(), "tracking_commands": tracking.copy()}
        fallen_gravity = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        env._nonwheel_contacts.fill(1.0)
        env._recovery_eligible[:] = True

        env._update_recovery_state(info, fallen_gravity)
        assert env._recovery_active[0]
        np.testing.assert_allclose(info["commands"], np.zeros((1, 3)))
        causes = env._compute_termination_causes(fallen_gravity)
        assert not causes["terminated"][0]

        upright_gravity = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        env._nonwheel_contacts.fill(0.0)
        env._update_recovery_state(info, upright_gravity)
        assert env._recovery_active[0]
        env._update_recovery_state(info, upright_gravity)
        assert not env._recovery_active[0]
        assert info["recovery_completed"][0]
        np.testing.assert_allclose(info["commands"], tracking)

        env._recovery_eligible[:] = False
        env._update_recovery_state(info, fallen_gravity)
        assert not env._recovery_active[0]
        assert env._compute_termination_causes(fallen_gravity)["terminated"][0]
    finally:
        env.close()


def test_real68_recovery_reset_uses_configured_height_and_arbitrary_orientation():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=4,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "recovery": {
                "enabled": True,
                "initial_base_height": 0.23,
                "initial_recovery_probability": 1.0,
                "initial_roll_range": [1.0, 1.0],
                "initial_pitch_range": [0.5, 0.5],
            },
            "reward_config": {"scales": {"alive": 1.0}, "tracking_sigma": 0.25},
        },
    )
    try:
        state = env.init_state()
        base_pos = np.asarray(env._backend.get_base_pos())
        np.testing.assert_allclose(base_pos[:, 2], np.full((4,), 0.23), atol=1.0e-6)
        assert np.all(np.asarray(state.info["recovery_active"], dtype=bool))
        np.testing.assert_allclose(state.info["commands"], np.zeros((4, 3)))
        assert not np.allclose(np.asarray(env._backend.get_base_quat())[:, 1:3], np.zeros((4, 2)))
    finally:
        env.close()


def test_real68_command_lean_targets_reduce_orientation_and_posture_penalties():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=8,
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
        target_gy = env._command_lean_gravity_target(cmd_x)
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
            gravity=np.asarray([[0.0, target_gy[0], 1.0]], dtype=np.float32),
            dof_pos=leaned_dof,
            dof_vel=np.zeros((1, env._num_action), dtype=np.float32),
            num_envs=1,
            default_angles=env.default_angles,
        )

        assert float(env._reward_orientation(leaned_ctx)[0]) < float(
            env._reward_orientation(upright_ctx)[0]
        )
        assert float(env._reward_posture(leaned_ctx)[0]) < float(
            env._reward_posture(upright_ctx)[0]
        )
        assert float(env._reward_joint_pos_penalty(leaned_ctx)[0]) < float(
            env._reward_joint_pos_penalty(upright_ctx)[0]
        )
    finally:
        env.close()


def test_real68_symmetric_standing_anchor_is_mirrored():
    hip_left, wheel_left, calf_left, hip_right, wheel_right, calf_right = (
        SYMMETRIC_STANDING_ACTIVE_ANGLES
    )
    assert hip_left == pytest.approx(-hip_right)
    assert calf_left == pytest.approx(-calf_right)
    assert wheel_left == pytest.approx(-0.045959)
    assert wheel_right == pytest.approx(-0.076009)


def test_real68_default_active_angles_match_symmetric_anchor():
    np.testing.assert_allclose(DEFAULT_ACTIVE_ANGLES, SYMMETRIC_STANDING_ACTIVE_ANGLES)


def test_real68_standing_rewards_prefer_symmetric_upright_anchor():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=8,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
                "balance_gate": {
                    "enabled": True,
                    "standing_roll_pitch_sigma": 0.03,
                },
            },
        },
    )
    try:
        commands = np.zeros((1, 3), dtype=np.float32)
        symmetric_dof = np.broadcast_to(
            SYMMETRIC_STANDING_ACTIVE_ANGLES, (1, env._num_action)
        ).copy()
        asymmetric_dof = symmetric_dof.copy()
        asymmetric_dof[:, POSTURE_INDICES] += np.asarray([[0.025, 0.04, -0.03, -0.05]])
        upright_ctx = RewardContext(
            info={"commands": commands},
            linvel=np.zeros((1, 3), dtype=np.float32),
            gyro=np.zeros((1, 3), dtype=np.float32),
            gravity=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            dof_pos=symmetric_dof,
            dof_vel=np.zeros((1, env._num_action), dtype=np.float32),
            num_envs=1,
            default_angles=env.default_angles,
        )
        tilted_ctx = RewardContext(
            info={"commands": commands},
            linvel=np.zeros((1, 3), dtype=np.float32),
            gyro=np.zeros((1, 3), dtype=np.float32),
            gravity=np.asarray([[0.08, -0.06, 1.0]], dtype=np.float32),
            dof_pos=asymmetric_dof,
            dof_vel=np.zeros((1, env._num_action), dtype=np.float32),
            num_envs=1,
            default_angles=env.default_angles,
        )

        assert float(env._reward_standing_orientation(upright_ctx)[0]) < float(
            env._reward_standing_orientation(tilted_ctx)[0]
        )
        assert float(env._reward_standing_posture(upright_ctx)[0]) < float(
            env._reward_standing_posture(tilted_ctx)[0]
        )
        assert float(env._reward_standing_leg_symmetry(upright_ctx)[0]) < float(
            env._reward_standing_leg_symmetry(tilted_ctx)[0]
        )
        assert float(env._reward_posture(upright_ctx)[0]) < float(
            env._reward_posture(tilted_ctx)[0]
        )
    finally:
        env.close()


def test_real68_home_keyframe_uses_symmetric_active_angles():
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
    )
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    key_qpos = np.asarray(model.key_qpos, dtype=np.float64).reshape(model.nkey, model.nq)[key_id]
    active_joint_names = (
        "left_hip_bigleg_joint",
        "left_wheel_joint",
        "left_calf_smallleg_joint",
        "right_hip_bigleg_joint",
        "right_wheel_joint",
        "right_calf_smallleg_joint",
    )
    active_qpos = []
    for joint_name in active_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert joint_id >= 0
        qpos_adr = int(model.jnt_qposadr[joint_id])
        active_qpos.append(float(key_qpos[qpos_adr]))
    np.testing.assert_allclose(np.asarray(active_qpos), SYMMETRIC_STANDING_ACTIVE_ANGLES)


def test_real68_home_keyframe_is_geometrically_level_left_to_right():
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
    )
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    left_wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    right_wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")
    left_liangan5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_chuanliangan5")
    right_liangan5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_liangan5")

    assert left_wheel >= 0 and right_wheel >= 0
    assert left_liangan5 >= 0 and right_liangan5 >= 0
    assert float(data.xpos[left_wheel, 2] - data.xpos[right_wheel, 2]) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(data.xpos[left_liangan5, 2] - data.xpos[right_liangan5, 2]) == pytest.approx(
        0.0, abs=1e-6
    )


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


def test_real68_command_curriculum_can_separate_straight_and_yaw_only_commands():
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
                "initial_vel_limit": [[0.5, 0.0, 1.0], [0.5, 0.0, 1.0]],
                "final_vel_limit": [[0.5, 0.0, 1.0], [0.5, 0.0, 1.0]],
                "straight_command_prob": 1.0,
                "yaw_only_command_prob": 0.0,
                "yaw_unlock_vx_progress": 0.0,
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
        straight = env.sample_velocity_commands(16)
        np.testing.assert_allclose(straight[:, 0], np.full((16,), 0.5))
        np.testing.assert_allclose(straight[:, 1:], np.zeros((16, 2)))

        env._cfg.command_curriculum.straight_command_prob = 0.0
        env._cfg.command_curriculum.yaw_only_command_prob = 1.0
        yaw_only = env.sample_velocity_commands(16)
        np.testing.assert_allclose(yaw_only[:, :2], np.zeros((16, 2)))
        np.testing.assert_allclose(yaw_only[:, 2], np.full((16,), 1.0))
    finally:
        env.close()


def test_real68_command_curriculum_can_bias_high_speed_commands():
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
                "initial_vel_limit": [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
                "final_vel_limit": [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
                "standing_prob_initial": 0.0,
                "standing_prob_final": 0.0,
                "high_speed_command_prob": 1.0,
                "high_speed_min_abs_vx": 1.0,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        commands = env.sample_velocity_commands(128)
        assert np.all(np.abs(commands[:, 0]) >= 1.0)
        assert np.all(np.abs(commands[:, 0]) <= 1.5)
        np.testing.assert_allclose(commands[:, 1:], np.zeros((128, 2)))
    finally:
        env.close()


def test_real68_command_curriculum_reports_velocity_safety_rates():
    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "command_curriculum": {
                "min_segment_count": 1,
            },
            "reward_config": {
                "scales": {"alive": 1.0},
                "tracking_sigma": 0.25,
            },
        },
    )
    try:
        env._record_command_segment_stats(
            cmd_x=0.4,
            cmd_yaw=0.0,
            mean_abs_vx=0.3,
            mean_signed_vx=0.28,
            mean_abs_wz=0.0,
            vx_error=0.12,
            wz_error=0.0,
            mean_nonwheel_contact=0.02,
            mean_tilt=0.03,
            mean_tilt_angle_deg=4.5,
            mean_height_violation=0.01,
            segment_steps=25,
        )

        vx_eval = env._curriculum_vx_eval()
        assert vx_eval is not None
        assert vx_eval["speed_ratio"] == pytest.approx(0.7)
        assert vx_eval["vx_error"] == pytest.approx(0.12)
        assert vx_eval["tilt_rate"] == pytest.approx(0.03)
        assert vx_eval["tilt_angle_deg"] == pytest.approx(4.5)
        assert vx_eval["height_violation_rate"] == pytest.approx(0.01)
        assert vx_eval["nonwheel_contact_rate"] == pytest.approx(0.02)

        log: dict[str, float] = {}
        env._write_command_curriculum_metrics(log)
        assert log["command_curriculum/eval_tilt_rate"] == pytest.approx(0.03)
        assert log["command_curriculum/eval_tilt_angle_deg"] == pytest.approx(4.5)
        assert log["command_curriculum/eval_height_violation_rate"] == pytest.approx(0.01)
        assert log["command_curriculum/eval_nonwheel_contact_rate"] == pytest.approx(0.02)
    finally:
        env.close()


def test_real68_diamond_clip_preserves_high_speed_bias_with_nonzero_yaw():
    """Diamond clip on combo (vx, wz) commands: reachability + bias semantics.

    Feeds deterministic corner combos directly to ``_clip_to_diamond`` so the
    contract does not depend on sampler randomness. Asserts:
      (a) every clipped command lies inside the reachable diamond
          ``|vx| + |wz|*L/2 <= v_wheel_max``;
      (b) high-speed bias survives in low/medium |wz| bands where the corner is
          already reachable (clip is a no-op, |vx| unchanged);
      (c) extreme |wz| combos are scaled back, with vx and wz scaled by the
          same factor (ray preserved), landing exactly on the diamond boundary.
    """
    from unilab.envs.locomotion.real68.balance import (
        _REAL68_WHEEL_BASE,
        _REAL68_WHEEL_RADIUS,
    )

    ensure_registries()
    env = registry.make(
        "Real68BalanceFlat",
        num_envs=8,
        sim_backend="mujoco",
        env_cfg_override={
            "scene": SceneCfg(
                model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "scene_flat.xml")
            ),
            "control_config": {"wheel_velocity_scale": 28.0},
            "reward_config": {"scales": {"alive": 1.0}, "tracking_sigma": 0.25},
        },
    )
    try:
        v_wheel_max = float(env._cfg.control_config.wheel_velocity_scale) * _REAL68_WHEEL_RADIUS
        half_L = 0.5 * _REAL68_WHEEL_BASE
        high_speed_min_abs_vx = 0.9  # production high_speed_min_abs_vx

        # Deterministic combo grid: (|vx|, |wz|) corners spanning reachable and
        # unreachable combos. Rows are (vx, wz) signed values.
        combos = np.asarray(
            [
                # --- low/medium |wz|, high-speed bias should be preserved (no clip) ---
                [0.90, 0.0],  # pure forward, well inside
                [0.90, 1.0],  # 0.9 + 1.0*0.215 = 1.115 < 1.68
                [0.90, 3.0],  # 0.9 + 3.0*0.215 = 1.545 < 1.68 (last reachable corner)
                [-0.90, 2.0],
                # --- extreme |wz|, high-speed bias must be clipped back ---
                [0.90, 5.0],  # 0.9 + 5.0*0.215 = 1.975 > 1.68 (unreachable)
                [0.90, 7.5],  # nominal pure-spin corner paired with high vx: ultramax
                [-0.90, 7.5],
                [1.60, 7.5],  # the (vx_max, wz_max) box corner, most over-demanded
            ],
            dtype=env._np_dtype,
        )
        cmds = np.zeros((combos.shape[0], 3), dtype=env._np_dtype)
        cmds[:, 0] = combos[:, 0]
        cmds[:, 2] = combos[:, 1]

        env._clip_to_diamond(cmds)
        clip_scale = np.asarray(env._last_command_clip_scale, dtype=np.float64)

        demand = np.abs(cmds[:, 0]) + np.abs(cmds[:, 2]) * half_L

        # (a) Reachability invariant.
        assert np.all(demand <= v_wheel_max + 1.0e-4), (
            f"clipped command exceeds reachable set: max demand {demand.max():.4f} > {v_wheel_max:.4f}"
        )

        # (b) Low/medium |wz| rows (indices 0..3): clip must be a no-op, |vx| unchanged.
        for i in range(0, 4):
            assert clip_scale[i] == pytest.approx(1.0), (
                f"row {i} (|vx|={abs(combos[i, 0])}, |wz|={abs(combos[i, 1])}) should not be clipped"
            )
            assert abs(cmds[i, 0]) == pytest.approx(abs(combos[i, 0]), abs=1.0e-5)
            assert abs(cmds[i, 2]) == pytest.approx(abs(combos[i, 1]), abs=1.0e-5)
        # The high-speed-bias floor (|vx| >= 0.9) is preserved on these rows.
        assert np.all(np.abs(cmds[:4, 0]) >= high_speed_min_abs_vx - 1.0e-5)

        # (c) Extreme |wz| rows (indices 4..7): clip must fire, ray preserved,
        # landing exactly on the diamond boundary.
        for i in range(4, 8):
            assert clip_scale[i] < 1.0 - 1.0e-5, (
                f"row {i} (|vx|={abs(combos[i, 0])}, |wz|={abs(combos[i, 1])}) must be clipped"
            )
            # Ray preservation: both components scaled by the same factor.
            sx = cmds[i, 0] / combos[i, 0] if combos[i, 0] != 0 else 1.0
            sz = cmds[i, 2] / combos[i, 1] if combos[i, 1] != 0 else 1.0
            np.testing.assert_allclose(sx, sz, rtol=1.0e-5)
            # Lands on the boundary (demand == v_wheel_max).
            np.testing.assert_allclose(
                abs(cmds[i, 0]) + abs(cmds[i, 2]) * half_L, v_wheel_max, rtol=1.0e-4
            )
            # High-speed bias necessarily reduced below the floor (physically unreachable).
            assert abs(cmds[i, 0]) < high_speed_min_abs_vx, (
                f"row {i}: clipped |vx|={abs(cmds[i, 0]):.4f} should be < {high_speed_min_abs_vx}"
            )

        # (d) Diagnostic scatter: the clip scale batch is distributed into the
        # per-env buffer verbatim by both production callers
        # (_update_commands, build_reset_plan), so clip_frac / clip_scale_mean
        # reduce to mean statistics over the buffer.
        env._command_clip_scale[: clip_scale.shape[0]] = clip_scale
        expected_clip_frac = float(np.mean(clip_scale < 1.0 - 1.0e-6))
        expected_clip_scale_mean = float(np.mean(clip_scale))
        observed_clip_frac = float(
            np.mean(env._command_clip_scale[: clip_scale.shape[0]] < 1.0 - 1.0e-6)
        )
        observed_clip_scale_mean = float(np.mean(env._command_clip_scale[: clip_scale.shape[0]]))
        assert observed_clip_frac == pytest.approx(expected_clip_frac)
        assert observed_clip_scale_mean == pytest.approx(expected_clip_scale_mean)
        # 4 of 8 combos are clipped (rows 4..7), so frac == 0.5.
        assert observed_clip_frac == pytest.approx(0.5)
        assert 0.0 < observed_clip_scale_mean < 1.0
    finally:
        env.close()
