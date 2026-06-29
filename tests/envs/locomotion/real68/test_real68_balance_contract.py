from __future__ import annotations

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.base.scene import SceneCfg
from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.real68.base import WHEEL_INDICES


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
        assert state.obs["obs"].shape == (2, 29)
        assert state.obs["critic"].shape == (2, 65)
        reset_obs, _ = env.reset(np.asarray([0], dtype=np.int32))
        assert reset_obs["obs"].shape == (1, 29)
        assert reset_obs["critic"].shape == (1, 65)
        step_state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert step_state.obs["obs"].shape == (2, 29)
        assert step_state.reward.shape == (2,)
        assert step_state.terminated.shape == (2,)
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
