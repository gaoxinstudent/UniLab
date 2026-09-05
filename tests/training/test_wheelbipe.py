"""Boundary tests for the Wheelbipe ONNX rollout helper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.wheelbipe_v14.base import POLICY_OBS_CLIP
from unilab.training.wheelbipe import (
    hydrate_wheelbipe_play_config,
    run_wheelbipe_policy,
    wheelbipe_play_checkpoint_task_candidates,
)


def test_exact_play_owner_lists_variant_then_canonical_checkpoint_roots() -> None:
    assert wheelbipe_play_checkpoint_task_candidates("WheelbipeV14FlatPlayV0") == (
        "WheelbipeV14FlatV0",
        "WheelbipeV14Flat",
    )
    assert wheelbipe_play_checkpoint_task_candidates("WheelbipeV14RoughPlayV1") == (
        "WheelbipeV14RoughV1",
        "WheelbipeV14Rough",
    )
    assert wheelbipe_play_checkpoint_task_candidates("WheelbipeV14FlatHIMPlay") == (
        "WheelbipeV14FlatHIM",
    )
    assert wheelbipe_play_checkpoint_task_candidates("WheelbipeV14FlatDreamWaQPlay") == (
        "WheelbipeV14FlatDreamWaQ",
    )
    assert wheelbipe_play_checkpoint_task_candidates("WheelbipeV14FlatNP3OBarlowPlay") == (
        "WheelbipeV14FlatNP3OBarlow",
    )
    assert wheelbipe_play_checkpoint_task_candidates("WheelbipeV14FlatHIM") == ()


def test_rough_canonical_play_prefers_source_v1_checkpoint_root() -> None:
    assert wheelbipe_play_checkpoint_task_candidates("WheelbipeV14Rough")[:2] == (
        "WheelbipeV14RoughV1",
        "WheelbipeV14RoughV0",
    )


def test_hydrate_canonical_play_from_rough_v1_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": {"task": "WheelbipeV14RoughV1"},
                "config": {
                    "env": {
                        "state_machine": {"enabled": True},
                        "gimbal": {"control_mode": "velocity", "randomize_heading": False},
                    },
                    "reward": {"scales": {"track_lin_vel_xy": 1.25}},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "WheelbipeV14Rough"},
            "env": {
                "gimbal": {"control_mode": "heading_pd", "randomize_heading": True},
            },
            "reward": {"scales": {"track_lin_vel_xy": 1.0}},
        }
    )

    hydrated = hydrate_wheelbipe_play_config(run_dir, cfg)

    assert hydrated.training.task_name == "WheelbipeV14RoughV1"
    assert hydrated.env.state_machine.enabled is True
    assert hydrated.env.gimbal.control_mode == "velocity"
    assert hydrated.reward.scales.track_lin_vel_xy == pytest.approx(1.25)
    assert cfg.training.task_name == "WheelbipeV14Rough"


class _RecordingPolicy:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []

    def predict(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(np.asarray(observation).copy())
        return np.zeros((observation.shape[0], 6), dtype=np.float32)


class _ResettingEnv:
    def __init__(self) -> None:
        self.calls = 0

    def _state(self) -> NpEnvState:
        # Deliberately return a fresh/random command on every reset.  The
        # helper must overwrite it before the next policy invocation.
        return NpEnvState(
            obs={
                "obs": np.zeros((1, 35), dtype=np.float32),
                "critic": np.zeros((1, 78), dtype=np.float32),
            },
            reward=np.zeros((1,), dtype=np.float32),
            terminated=np.zeros((1,), dtype=bool),
            truncated=np.zeros((1,), dtype=bool),
            info={
                "commands": np.asarray([[9.0, 9.0, 9.0]], dtype=np.float32),
                "height_commands": np.asarray([9.0], dtype=np.float32),
            },
        )

    def init_state(self) -> NpEnvState:
        return self._state()

    def step(self, actions: np.ndarray) -> NpEnvState:
        assert actions.shape == (1, 6)
        self.calls += 1
        state = self._state()
        # Exercise the autoreset path on every step.
        state.terminated[:] = True
        return state


def test_rollout_command_overrides_update_obs_and_autoresets() -> None:
    env = _ResettingEnv()
    policy = _RecordingPolicy()

    diagnostics = run_wheelbipe_policy(
        env,
        policy,  # type: ignore[arg-type]
        steps=3,
        command=(0.3, 0.0, -0.4),
        height=0.27,
    )

    assert diagnostics["done_count"] == 3.0
    assert len(policy.observations) == 3
    for observation in policy.observations:
        np.testing.assert_allclose(observation[0, :3], [0.3, 0.0, -0.4])
        np.testing.assert_allclose(observation[0, 3], 1.35)


def test_rollout_publishes_completed_post_step_states() -> None:
    env = _ResettingEnv()
    policy = _RecordingPolicy()
    published: list[tuple[NpEnvState, int]] = []

    run_wheelbipe_policy(
        env,
        policy,  # type: ignore[arg-type]
        steps=3,
        command=(0.2, 0.0, 0.1),
        step_callback=lambda state, step: published.append((state, step)),
    )

    assert [step for _state, step in published] == [1, 2, 3]
    for state, _step in published:
        np.testing.assert_allclose(state.info["commands"], [[0.2, 0.0, 0.1]])


def test_rollout_command_overrides_keep_observation_clip_contract() -> None:
    env = _ResettingEnv()
    policy = _RecordingPolicy()

    run_wheelbipe_policy(
        env,
        policy,  # type: ignore[arg-type]
        steps=1,
        command=(1.0e6, -1.0e6, 1.0e6),
        height=1.0e6,
    )

    np.testing.assert_array_equal(
        policy.observations[0][0, :4],
        [POLICY_OBS_CLIP, -POLICY_OBS_CLIP, POLICY_OBS_CLIP, POLICY_OBS_CLIP],
    )


@pytest.mark.parametrize(
    ("command", "height"),
    [((np.nan, 0.0, 0.0), None), ((np.inf, 0.0, 0.0), None), ((0.0, 0.0, 0.0), np.inf)],
)
def test_rollout_rejects_nonfinite_command_overrides(
    command: tuple[float, float, float], height: float | None
) -> None:
    env = _ResettingEnv()
    policy = _RecordingPolicy()

    with pytest.raises(ValueError, match="finite"):
        run_wheelbipe_policy(
            env,
            policy,  # type: ignore[arg-type]
            steps=1,
            command=command,
            height=height,
        )

    assert env.calls == 0
    assert policy.observations == []
