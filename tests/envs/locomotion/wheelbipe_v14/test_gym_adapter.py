"""Gymnasium source-ID compatibility tests for WheelBipe."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.wheelbipe_v14.gym_adapter import (
    WheelbipeGymEnv,
    register_wheelbipe_gym_envs,
)


def test_upstream_ids_are_registered_as_gym_specs() -> None:
    ensure_registries()
    ids = register_wheelbipe_gym_envs()
    assert len(ids) == 15
    assert "Robotics-Wheelbipe-V14-Flat-v0" in ids
    assert gym.spec("Robotics-Wheelbipe-V14-Flat-v0").entry_point == (
        "unilab.envs.locomotion.wheelbipe_v14.gym_adapter:make_wheelbipe_gym_env"
    )


def test_gym_facade_exposes_standard_reset_step_api() -> None:
    pytest.importorskip("mujoco")
    ensure_registries()
    env = gym.make("Robotics-Wheelbipe-V14-Flat-v0", sim_backend="mujoco")
    try:
        assert isinstance(env.unwrapped, WheelbipeGymEnv)
        obs, info = env.reset(seed=7)
        assert obs.shape == (35,)
        assert obs.dtype == np.float32
        assert info["wheelbipe_task_id"] == "Robotics-Wheelbipe-V14-Flat-v0"
        raw = np.asarray([20.0, -20.0, 8.0, -8.0, 20.0, -20.0], dtype=np.float32)
        assert env.action_space.contains(raw)
        next_obs, reward, terminated, truncated, step_info = env.step(raw)
        assert next_obs.shape == (35,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["unilab_sim_backend"] == "mujoco"
        np.testing.assert_array_equal(step_info["current_actions"], raw)
    finally:
        env.close()


def test_gym_facade_supports_compact_custom_source_id() -> None:
    pytest.importorskip("mujoco")
    ensure_registries()
    env = gym.make("Robotics-Wheelbipe-V14-Flat-HIM-v0", sim_backend="mujoco")
    try:
        obs, _info = env.reset()
        assert obs.shape == (28,)
        obs, _reward, _terminated, _truncated, _info = env.step(np.zeros((6,), dtype=np.float32))
        assert obs.shape == (28,)
    finally:
        env.close()


def test_gym_facade_supports_state_machine_source_id() -> None:
    pytest.importorskip("mujoco")
    ensure_registries()
    env = gym.make("Robotics-Wheelbipe-V14-Flat-v1", sim_backend="mujoco")
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (35,)
        assert info["wheelbipe_task_id"] == "Robotics-Wheelbipe-V14-Flat-v1"
        obs, reward, terminated, truncated, _info = env.step(np.zeros((6,), dtype=np.float32))
        assert obs.shape == (35,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_gym_facade_reset_seed_replays_interleaved_trajectory() -> None:
    """Gym's per-env seed must survive the owner's global NumPy draws.

    The owner currently uses ``np.random`` in reset, command resampling,
    delay sampling and backend perturbation paths.  Two facades seeded alike
    therefore need independent snapshots; stepping one env must not perturb
    the other env's next observation.
    """

    pytest.importorskip("mujoco")
    ensure_registries()
    env_a = gym.make("Robotics-Wheelbipe-V14-Flat-v0", sim_backend="mujoco")
    env_b = gym.make("Robotics-Wheelbipe-V14-Flat-v0", sim_backend="mujoco")
    try:
        obs_a, _ = env_a.reset(seed=123)
        obs_b, _ = env_b.reset(seed=123)
        np.testing.assert_array_equal(obs_a, obs_b)

        zero = np.zeros((6,), dtype=np.float32)
        # Interleave calls deliberately.  Without facade-local RNG snapshots,
        # env_b consumes env_a's process-global stream and diverges here.
        for _ in range(3):
            next_a = env_a.step(zero)
            next_b = env_b.step(zero)
            np.testing.assert_array_equal(next_a[0], next_b[0])
            assert next_a[1:4] == next_b[1:4]
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.parametrize("sim_backend", ["mujoco", "motrix"])
def test_gym_facade_rough_seed_replays_static_terrain(
    sim_backend: str,
) -> None:
    """A reset seed must also replay rough owners' cold-path heightfield.

    Rough terrain is generated while ``gym.make`` constructs the owner, before
    Gymnasium can call ``reset(seed=...)``.  The facade therefore gives that
    cold path a stable task/backend seed; this regression catches accidental
    dependence on construction order while still exercising both backends.
    """

    if sim_backend == "mujoco":
        pytest.importorskip("mujoco")
    else:
        pytest.importorskip("motrixsim", reason="motrixsim is not installed")
    ensure_registries()
    task_id = "Robotics-Wheelbipe-V14-Rough-v0"
    env_a = gym.make(task_id, sim_backend=sim_backend)
    env_b = gym.make(task_id, sim_backend=sim_backend)
    try:
        obs_a, _ = env_a.reset(seed=5150)
        obs_b, _ = env_b.reset(seed=5150)
        np.testing.assert_array_equal(obs_a, obs_b)
        zero = np.zeros((6,), dtype=np.float32)
        for _ in range(2):
            step_a = env_a.step(zero)
            step_b = env_b.step(zero)
            # Motrix can differ by a few ulps across independently-created
            # solver instances even with identical state/terrain; this is
            # still deterministic at the Gym contract's numeric tolerance.
            np.testing.assert_allclose(step_a[0], step_b[0], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(step_a[1], step_b[1], rtol=1e-6, atol=1e-7)
            assert step_a[2:4] == step_b[2:4]
    finally:
        env_a.close()
        env_b.close()


def test_gym_facade_seed_does_not_mutate_caller_numpy_rng() -> None:
    """A Gym reset/step should not consume the process-global caller stream."""

    pytest.importorskip("mujoco")
    ensure_registries()
    env = gym.make("Robotics-Wheelbipe-V14-Flat-v0", sim_backend="mujoco")
    try:
        np.random.seed(991)
        expected = np.random.rand(5)
        np.random.seed(991)
        env.reset(seed=17)
        env.step(np.zeros((6,), dtype=np.float32))
        observed = np.random.rand(5)
        np.testing.assert_array_equal(observed, expected)
    finally:
        env.close()


def test_gym_facade_construction_does_not_mutate_caller_numpy_rng() -> None:
    """Cold-path rough terrain generation must be isolated from the caller."""

    pytest.importorskip("mujoco")
    ensure_registries()
    np.random.seed(2027)
    expected = np.random.rand(5)
    np.random.seed(2027)
    env = gym.make("Robotics-Wheelbipe-V14-Rough-v0", sim_backend="mujoco")
    try:
        observed = np.random.rand(5)
        np.testing.assert_array_equal(observed, expected)
    finally:
        env.close()
