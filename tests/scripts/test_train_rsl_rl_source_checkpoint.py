"""Entrypoint wiring for upstream Wheelbipe vanilla-PPO playback."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def _load_train_rsl_rl() -> Any:
    name = "train_rsl_rl_source_checkpoint_test"
    path = ROOT / "scripts" / "train_rsl_rl.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ELU(),
            torch.nn.Linear(3, 2),
        )
        self.distribution = torch.nn.Module()
        self.distribution.register_parameter("std_param", torch.nn.Parameter(torch.ones(2)))


class _Critic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(5, 3),
            torch.nn.ELU(),
            torch.nn.Linear(3, 1),
        )


def _source_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    actor = _Actor()
    critic = _Critic()
    state: dict[str, torch.Tensor] = {
        f"actor.{key}": value.detach().clone() for key, value in actor.mlp.state_dict().items()
    }
    state.update(
        {f"critic.{key}": value.detach().clone() for key, value in critic.mlp.state_dict().items()}
    )
    state["std"] = torch.tensor([0.25, 0.5])
    torch.save({"model_state_dict": state, "iter": 10}, path)
    return actor.mlp.state_dict(), critic.mlp.state_dict()


def test_play_helper_loads_upstream_checkpoint_without_runner_resume(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_train_rsl_rl()
    checkpoint = tmp_path / "model_10.pt"
    expected_actor, expected_critic = _source_checkpoint(checkpoint)

    actor = _Actor()
    critic = _Critic()

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            del wrapped_env, train_cfg, log_dir, device
            self.alg = types.SimpleNamespace(actor=actor, critic=critic)
            self.load_called = False

        def load(self, path, **kwargs):
            del path, kwargs
            self.load_called = True
            raise AssertionError("source playback must not use native runner.load")

        def get_inference_policy(self, device):
            del device
            return lambda observations: torch.zeros((observations.shape[0], 2))

    class FakeWrapper:
        num_obs = 4
        num_actions = 2

        def __init__(self, env, device):
            self.env = env
            self.device = device

        def reset(self):
            return torch.zeros((1, 4)), {}

        def step(self, actions):
            assert actions.shape == (1, 2)
            return torch.zeros((1, 4)), torch.ones(1), torch.zeros(1, dtype=torch.bool), {}

    class FakeEnv:
        cfg = types.SimpleNamespace(render_spacing=1.0, render_offset_mode="grid")

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    env = FakeEnv()
    runner_holder: dict[str, FakeRunner] = {}

    class CapturingRunner(FakeRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            runner_holder["runner"] = self

    monkeypatch.setattr(module, "OnPolicyRunner", CapturingRunner)
    cfg = OmegaConf.create(
        {
            "training": {
                "play_render_mode": "none",
                "play_steps": 1,
            }
        }
    )

    result = module._play_rsl_rl_with_env(
        cfg,
        "cpu",
        env,
        FakeWrapper,
        {},
        checkpoint,
        tmp_path,
        source_checkpoint=True,
    )

    assert result is None
    assert env.closed is True
    assert runner_holder["runner"].load_called is False
    for key, value in expected_actor.items():
        torch.testing.assert_close(actor.mlp.state_dict()[key], value)
    for key, value in expected_critic.items():
        torch.testing.assert_close(critic.mlp.state_dict()[key], value)
    torch.testing.assert_close(actor.distribution.std_param, torch.tensor([0.25, 0.5]))


def test_play_rejects_malformed_source_checkpoint_before_env_creation(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A foreign ``model_state_dict`` must fail closed before backend setup."""

    module = _load_train_rsl_rl()
    checkpoint = tmp_path / "malformed.pt"
    torch.save({"model_state_dict": {"std": torch.ones(2)}}, checkpoint)

    cfg = OmegaConf.create(
        {
            "algo": {},
            "training": {"task_name": "wheelbipe_v14_flat"},
        }
    )
    env_created = False

    monkeypatch.setattr(module, "_resolve_ppo_wrapper_cls", lambda _rl_cfg: object())
    monkeypatch.setattr(module, "get_log_root", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(
        module,
        "parse_checkpoint_path",
        lambda *_args, **_kwargs: (checkpoint, tmp_path),
    )

    def fail_create_env(*_args, **_kwargs):
        nonlocal env_created
        env_created = True
        raise AssertionError("malformed source checkpoints must not create an environment")

    monkeypatch.setattr(module, "create_env", fail_create_env)

    with pytest.raises(
        ValueError,
        match="not a valid upstream Wheelbipe vanilla-PPO checkpoint",
    ):
        module.play_rsl_rl(cfg, device="cpu")
    assert env_created is False


def test_exact_wheelbipe_play_latest_falls_back_to_canonical_checkpoint_root(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Play aliases keep their env variant while finding canonical latest runs."""

    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    module = _load_train_rsl_rl()
    cfg = OmegaConf.create(
        {
            "algo": {"load_run": "-1", "checkpoint": -1, "algo_log_name": "rsl_rl_ppo"},
            "training": {
                "task_name": "WheelbipeV14FlatPlayV0",
                "play_only": True,
                "sim_backend": "mujoco",
                "play_env_num": 1,
                "sim2sim_strict": True,
            },
        }
    )
    canonical_run = (
        tmp_path / "logs" / "rsl_rl_ppo" / "WheelbipeV14Flat" / "2026-01-01_00-00-00_mujoco"
    )
    canonical_run.mkdir(parents=True)
    checkpoint = canonical_run / "model_7.pt"
    torch.save({"actor_state_dict": {}}, checkpoint)
    # The newest run can be present while still collecting its first
    # checkpoint.  The targeted Play fallback must continue to the newest
    # *usable* predecessor instead of returning a silent no-op.
    (tmp_path / "logs" / "rsl_rl_ppo" / "WheelbipeV14Flat" / "2026-02-01_00-00-00_mujoco").mkdir(
        parents=True
    )

    captured: dict[str, Any] = {}

    class FakeEnv:
        pass

    def fake_create_env(config, **kwargs):
        captured["task_name"] = str(config.training.task_name)
        captured["env_kwargs"] = kwargs
        return FakeEnv()

    def fake_play_helper(_cfg, _device, _env, _wrapper, _rl_cfg, load_path, load_dir, **kwargs):
        del _cfg, _device, _env, _wrapper, _rl_cfg, load_dir, kwargs
        captured["load_path"] = load_path
        return None

    original_root = module.ROOT_DIR
    module.ROOT_DIR = tmp_path
    try:
        monkeypatch.setattr(module, "_resolve_ppo_wrapper_cls", lambda _cfg: object)
        monkeypatch.setattr(module, "create_env", fake_create_env)
        monkeypatch.setattr(module, "build_ppo_play_env_cfg_override", lambda _cfg: {})
        monkeypatch.setattr(module, "resolve_sim2sim_config", lambda _run, config, **_kw: config)
        monkeypatch.setattr(module, "_play_rsl_rl_with_env", fake_play_helper)
        assert module.play_rsl_rl(cfg, device="cpu") is None
    finally:
        module.ROOT_DIR = original_root

    assert captured["load_path"] == checkpoint
    assert captured["task_name"] == "WheelbipeV14FlatPlayV0"
    assert "Using latest non-Play checkpoint" in capsys.readouterr().out


def test_exact_wheelbipe_play_explicit_load_run_does_not_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    """An explicit run selection remains fail-closed at the requested root."""

    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    module = _load_train_rsl_rl()
    cfg = OmegaConf.create(
        {
            "algo": {"load_run": "explicit-run", "checkpoint": -1, "algo_log_name": "rsl_rl_ppo"},
            "training": {
                "task_name": "WheelbipeV14FlatPlayV0",
                "play_only": True,
                "sim_backend": "mujoco",
                "play_env_num": 1,
            },
        }
    )
    calls: list[dict[str, Any]] = []

    def fake_parse(*args, **kwargs):
        del args
        calls.append(dict(kwargs))
        return None, None

    original_root = module.ROOT_DIR
    module.ROOT_DIR = tmp_path
    try:
        monkeypatch.setattr(module, "_resolve_ppo_wrapper_cls", lambda _cfg: object)
        monkeypatch.setattr(module, "parse_checkpoint_path", fake_parse)
        with pytest.raises(FileNotFoundError, match="Could not resolve a checkpoint"):
            module.play_rsl_rl(cfg, device="cpu")
    finally:
        module.ROOT_DIR = original_root

    assert len(calls) == 1
    assert "task_name" not in calls[0]
