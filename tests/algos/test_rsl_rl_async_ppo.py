from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict

from unilab.algos.torch.rsl_rl_async_ppo.buffer import RslRlPpoRolloutBuffer
from unilab.algos.torch.rsl_rl_async_ppo.runner import (
    ppo_construct_cfg,
    validate_async_ppo_v1_config,
)
from unilab.algos.torch.rsl_rl_async_ppo.staging import stage_ppo_rollout
from unilab.algos.torch.rsl_rl_async_ppo.storage_adapter import fill_rollout_storage
from unilab.algos.torch.rsl_rl_async_ppo.worker import _record_reward_components

ROOT_DIR = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT_DIR / "conf" / "ppo"


def _ppo_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=overrides or [])


class _FakeStorage:
    def __init__(self) -> None:
        self.num_transitions_per_env = 2
        self.num_envs = 3
        self.observations = TensorDict(
            {"policy": torch.zeros(2, 3, 4)},
            batch_size=[2, 3],
        )
        self.actions = torch.zeros(2, 3, 2)
        self.rewards = torch.zeros(2, 3, 1)
        self.dones = torch.zeros(2, 3, 1, dtype=torch.bool)
        self.values = torch.zeros(2, 3, 1)
        self.actions_log_prob = torch.zeros(2, 3, 1)
        self.distribution_params = None
        self.step = 99
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1
        self.step = 0


def test_async_ppo_config_defaults_are_opt_in() -> None:
    cfg = _ppo_cfg()

    assert getattr(cfg.training, "async") is False
    assert cfg.training.async_queue_size == 1
    assert cfg.training.async_policy_lag_max == 1
    assert cfg.training.async_stale_rollout_policy == "block"
    assert cfg.training.async_strict_onpolicy is True


def test_async_ppo_v1_rejects_backlog() -> None:
    cfg = _ppo_cfg(["training.async=true", "training.async_queue_size=2"])

    with pytest.raises(ValueError, match="async_queue_size"):
        validate_async_ppo_v1_config(cfg, {"algorithm": {}})


def test_ppo_construct_cfg_adds_runner_multi_gpu_default_without_mutating_input() -> None:
    train_cfg = {"algorithm": {}, "actor": {}, "critic": {}}

    constructed = ppo_construct_cfg(train_cfg)

    assert constructed["multi_gpu"] is None
    assert "multi_gpu" not in train_cfg
    assert constructed is not train_cfg


def test_async_worker_records_reward_components_from_env_log() -> None:
    sink: defaultdict[str, list[float]] = defaultdict(list)

    _record_reward_components(
        sink,
        {
            "log": {
                "Episode/rew_tracking_lin_vel": torch.tensor([1.0, 3.0]),
                "rew_action_rate": -0.25,
                "reward/base_height": -0.5,
                "Episode/length": 12.0,
            }
        },
    )

    assert dict(sink) == {
        "tracking_lin_vel": [2.0],
        "action_rate": [-0.25],
        "base_height": [-0.5],
    }


def test_ppo_rollout_schema_includes_values_and_distribution_params() -> None:
    buffer = RslRlPpoRolloutBuffer(
        num_envs=3,
        num_steps=2,
        obs_shapes={"policy": (4,), "critic": (5,)},
        action_dim=2,
        distribution_param_shapes=((2,), (2,)),
        num_slots=1,
        create=True,
    )
    try:
        shapes = buffer.slot_shapes
        assert shapes["values"] == (3, 2, 1)
        assert shapes["actions_log_prob"] == (3, 2, 1)
        assert shapes["distribution_params/0"] == (3, 2, 2)
        assert shapes["distribution_params/1"] == (3, 2, 2)
        assert shapes["last_obs/policy"] == (3, 4)
    finally:
        buffer.cleanup()


def test_stage_and_storage_adapter_fill_rsl_rl_storage_contract() -> None:
    raw: dict[str, np.ndarray] = {
        "obs/policy": np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
        "last_obs/policy": np.ones((3, 4), dtype=np.float32),
        "actions": np.ones((3, 2, 2), dtype=np.float32),
        "rewards": np.ones((3, 2), dtype=np.float32),
        "dones": np.zeros((3, 2), dtype=np.float32),
        "values": np.full((3, 2, 1), 2.0, dtype=np.float32),
        "actions_log_prob": np.full((3, 2, 1), -0.5, dtype=np.float32),
        "distribution_params/0": np.full((3, 2, 2), 0.25, dtype=np.float32),
        "distribution_params/1": np.full((3, 2, 2), 0.75, dtype=np.float32),
        "policy_version_start": np.array([1.0], dtype=np.float32),
        "policy_version_end": np.array([1.0], dtype=np.float32),
        "rollout_created_time_ns": np.array([123.0], dtype=np.float32),
        "rollout_collect_time": np.array([0.25], dtype=np.float32),
    }
    rollout = stage_ppo_rollout(
        raw,
        obs_shapes={"policy": (4,)},
        distribution_param_count=2,
        device="cpu",
    )
    storage: Any = _FakeStorage()

    fill_rollout_storage(storage, rollout)

    assert storage.clear_calls == 1
    assert storage.step == storage.num_transitions_per_env
    assert torch.equal(
        storage.observations["policy"],
        torch.from_numpy(raw["obs/policy"]).transpose(0, 1),
    )
    assert torch.allclose(storage.values, torch.full((2, 3, 1), 2.0))
    assert storage.distribution_params is not None
    assert len(storage.distribution_params) == 2
    assert torch.allclose(storage.distribution_params[0], torch.full((2, 3, 2), 0.25))
