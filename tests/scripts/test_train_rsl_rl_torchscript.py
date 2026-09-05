"""Normal PPO playback tests for inference-only TorchScript actors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from unilab.training.wheelbipe import WheelbipeTorchScriptPolicy

ROOT = Path(__file__).resolve().parents[2]


def _load_train_script() -> Any:
    name = "train_rsl_rl_torchscript_playback_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "train_rsl_rl.py")
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive branch
        raise RuntimeError("could not load scripts/train_rsl_rl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _save_policy(path: Path) -> Path:
    actor = torch.nn.Sequential(torch.nn.Linear(35, 6), torch.nn.Tanh()).eval()
    torch.jit.trace(actor, torch.zeros((1, 35), dtype=torch.float32)).save(str(path))
    return path


class _FakeEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeWrapper:
    num_obs = 35
    num_actions = 6

    def __init__(self, env: _FakeEnv, device: str) -> None:
        self.env = env
        self.device = device

    @staticmethod
    def _observations() -> TensorDict:
        return TensorDict(
            {
                "actor": torch.zeros((1, 35)),
                "critic": torch.zeros((1, 78)),
            },
            batch_size=[1],
        )

    def reset(self) -> tuple[TensorDict, dict[str, object]]:
        return self._observations(), {}

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert actions.shape == (1, 6)
        return self._observations(), torch.ones(1), torch.zeros(1, dtype=torch.bool), {}


def test_play_helper_runs_jit_policy_without_constructing_rsl_runner(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    module = _load_train_script()
    model_file = _save_policy(tmp_path / "policy.pt")
    policy = WheelbipeTorchScriptPolicy(model_file)
    env = _FakeEnv()

    class ShouldNotConstructRunner:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("TorchScript playback must not construct a second RSL runner")

    monkeypatch.setattr(module, "OnPolicyRunner", ShouldNotConstructRunner)
    cfg = OmegaConf.create({"training": {"play_render_mode": "none", "play_steps": 2}})

    result = module._play_rsl_rl_with_env(
        cfg,
        "cpu",
        env,
        _FakeWrapper,
        {},
        model_file,
        tmp_path,
        torchscript_policy=policy,
    )

    assert result is None
    assert env.closed is True
    assert "PPO numerical playback complete: steps=2" in capsys.readouterr().out


def test_play_helper_rejects_jit_environment_dimension_mismatch(tmp_path: Path) -> None:
    module = _load_train_script()
    model_file = _save_policy(tmp_path / "policy.pt")
    policy = WheelbipeTorchScriptPolicy(model_file)
    env = _FakeEnv()

    class BadWrapper(_FakeWrapper):
        num_obs = 16

    cfg = OmegaConf.create({"training": {"play_render_mode": "none", "play_steps": 1}})
    with pytest.raises(ValueError, match="requires a 35D actor observation"):
        module._play_rsl_rl_with_env(
            cfg,
            "cpu",
            env,
            BadWrapper,
            {},
            model_file,
            tmp_path,
            torchscript_policy=policy,
        )
    assert env.closed is True
