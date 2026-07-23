from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unilab.algos.torch.him_ppo.runner import HIMOnPolicyRunner, _HIMLogger


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.std = torch.nn.Parameter(torch.tensor([0.01]))
        self.min_noise_std = 0.03

    def disable_empirical_normalization(self) -> None:
        raise AssertionError("normalization fallback should not be used")

    def clamp_action_std_(self) -> None:
        with torch.no_grad():
            self.std.clamp_(min=self.min_noise_std)


class _TrainingEnv:
    def __init__(self, progress: float) -> None:
        self.progress = progress
        self.loaded_states: list[dict] = []

    def training_state_dict(self) -> dict:
        return {"version": 1, "progress": self.progress}

    def load_training_state_dict(self, state: dict) -> None:
        self.loaded_states.append(state)
        self.progress = float(state["progress"])


def _make_runner(env: _TrainingEnv) -> HIMOnPolicyRunner:
    runner = HIMOnPolicyRunner.__new__(HIMOnPolicyRunner)
    runner.actor_critic = _Actor()
    runner.alg = SimpleNamespace(
        optimizer=torch.optim.Adam(runner.actor_critic.parameters(), lr=1.0e-3)
    )
    runner.current_learning_iteration = 0
    runner.logger = _HIMLogger()
    runner.env = env
    runner.device = "cpu"
    return runner


def test_him_checkpoint_round_trip_restores_runner_and_env_state(tmp_path):
    source = _make_runner(_TrainingEnv(progress=0.65))
    source.current_learning_iteration = 123
    source.logger.tot_timesteps = 456
    source.logger.rewbuffer.extend([1.0, 2.0])
    source.logger.lenbuffer.extend([10.0, 20.0])
    checkpoint = tmp_path / "model_123.pt"
    source.save(str(checkpoint))

    restored_env = _TrainingEnv(progress=0.0)
    restored = _make_runner(restored_env)
    restored.load(str(checkpoint))

    assert restored.current_learning_iteration == 123
    assert restored.logger.tot_timesteps == 456
    assert list(restored.logger.rewbuffer) == [1.0, 2.0]
    assert list(restored.logger.lenbuffer) == [10.0, 20.0]
    assert restored_env.progress == pytest.approx(0.65)
    assert len(restored_env.loaded_states) == 1
    torch.testing.assert_close(restored.actor_critic.std, torch.tensor([0.03]))


def test_him_play_load_skips_training_state(tmp_path):
    source = _make_runner(_TrainingEnv(progress=0.8))
    source.current_learning_iteration = 50
    source.logger.tot_timesteps = 999
    checkpoint = tmp_path / "model_50.pt"
    source.save(str(checkpoint))

    play_env = _TrainingEnv(progress=0.0)
    play_runner = _make_runner(play_env)
    play_runner.load(str(checkpoint), restore_training_state=False)

    assert play_runner.current_learning_iteration == 50
    assert play_runner.logger.tot_timesteps == 0
    assert play_env.progress == 0.0
    assert play_env.loaded_states == []


def test_him_legacy_checkpoint_warns_that_curriculum_cannot_resume(tmp_path):
    source = _make_runner(_TrainingEnv(progress=0.0))
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "actor_state_dict": source.actor_critic.state_dict(),
            "optimizer_state_dict": source.alg.optimizer.state_dict(),
            "iteration": 7,
        },
        checkpoint,
    )

    restored = _make_runner(_TrainingEnv(progress=0.0))
    with pytest.warns(UserWarning, match="curricula will restart"):
        restored.load(str(checkpoint))
    assert restored.current_learning_iteration == 7
