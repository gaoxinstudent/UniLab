from __future__ import annotations

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.base.scene import SceneCfg
from unilab.assets import ASSETS_ROOT_PATH


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
