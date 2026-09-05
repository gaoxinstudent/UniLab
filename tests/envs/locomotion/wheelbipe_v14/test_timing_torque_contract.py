"""Pure tests for the Wheelbipe timing and actuator-limit contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.locomotion.wheelbipe_v14.base import (
    WHEELBIPE_DELAY_PROFILE_SOURCE_V14,
    build_wheelbipe_timing_contract,
    resolve_wheelbipe_torque_limits,
    sample_wheelbipe_delay_lags,
    wheelbipe_delay_profile_overrides,
)
from unilab.envs.locomotion.wheelbipe_v14.joystick import WheelbipeV14Env, WheelbipeV14FlatCfg
from unilab.envs.locomotion.wheelbipe_v14.variants import (
    WheelbipeFlatV0Cfg,
    WheelbipeHIMCfg,
    WheelbipeVariantEnv,
)


def test_source_delay_sampler_uses_high_exclusive_endpoint() -> None:
    np.random.seed(11)
    sampled = sample_wheelbipe_delay_lags((1, 4), 512, name="obs_delay", inclusive=False)
    assert int(sampled.min()) >= 1
    assert int(sampled.max()) < 4
    np.testing.assert_array_equal(
        sample_wheelbipe_delay_lags((3, 3), 4, name="fixed", inclusive=False),
        np.full(4, 3, dtype=np.intp),
    )


def test_source_timing_profile_is_explicit_and_not_full_parity() -> None:
    contract = build_wheelbipe_timing_contract(
        sim_dt=0.005,
        ctrl_dt=0.02,
        obs_delay_step_unit="physics",
        use_obs_delay=True,
        use_act_delay=True,
        delay_range_semantics="exclusive",
        delay_profile=WHEELBIPE_DELAY_PROFILE_SOURCE_V14,
    )
    assert contract["substeps"] == 4
    assert contract["source_timing_match"] is True
    assert contract["source_delay_sampling_match"] is True
    assert contract["full_source_parity"] is False


def test_source_timing_profile_fails_closed_on_local_timing() -> None:
    with pytest.raises(ValueError, match="source_v14_physics delay profile contract mismatch"):
        build_wheelbipe_timing_contract(
            sim_dt=0.001,
            ctrl_dt=0.02,
            obs_delay_step_unit="physics",
            use_obs_delay=True,
            use_act_delay=True,
            delay_range_semantics="exclusive",
            delay_profile=WHEELBIPE_DELAY_PROFILE_SOURCE_V14,
        )


def test_delay_reset_clears_only_selected_env_and_samples_exclusive_lag() -> None:
    cfg = WheelbipeV14FlatCfg()
    cfg.use_obs_delay = True
    cfg.use_act_delay = True
    cfg.obs_delay_step_unit = "physics"
    cfg.delay_range_semantics = "exclusive"
    cfg.obs_delay_cfg = {"gyro": (1, 4)}
    cfg.act_delay_cfg = {"leg_actions": (1, 3)}

    # The delay owner is backend-independent; construct only its cold-path
    # buffers so reset semantics can be tested without a simulator.
    env = object.__new__(WheelbipeV14Env)
    env._cfg = cfg
    env._num_envs = 3
    env._np_dtype = np.dtype(np.float32)
    env._init_delay_buffers()
    obs_buffer = env._obs_delay_buffers["gyro"]
    act_buffer = env._act_delay_buffers["leg_actions"]
    obs_buffer.set_time_lag(np.asarray([1, 2, 3], dtype=np.intp))
    act_buffer.set_time_lag(np.asarray([1, 2, 3], dtype=np.intp))
    obs_buffer.compute(np.ones((3, 3), dtype=np.float32))
    act_buffer.compute(np.ones((3, 4), dtype=np.float32))
    env._delayed_native_targets[0] = 2.0
    env._delayed_native_targets[1] = 2.0

    np.random.seed(23)
    env._reset_delay_buffers(np.asarray([1], dtype=np.int32))

    # Non-selected environments retain both their lag and history.  The
    # selected environment is cleared and receives a source-style [1, high)
    # lag; its delayed target scratch is reset as well.
    assert int(obs_buffer.lags[0]) == 1
    assert int(obs_buffer.lags[2]) == 3
    assert 1 <= int(obs_buffer.lags[1]) < 4
    assert int(act_buffer.lags[0]) == 1
    assert int(act_buffer.lags[2]) == 3
    assert 1 <= int(act_buffer.lags[1]) < 3
    np.testing.assert_allclose(obs_buffer._history[:, 1], 0.0)
    np.testing.assert_allclose(act_buffer._history[:, 1], 0.0)
    assert np.all(env._delayed_native_targets[1] == 0.0)


def test_source_physics_observation_delay_pushes_settled_post_step_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source loop samples three pre-step and one settled frame per control step.

    ``_apply_action`` updates the source cache after each physics step.  Since
    the cache is already marked current after the previous ``_get_observations``
    call, the first substep skips an append; the remaining three substeps
    provide the four frames consumed by the delayed observation ring.
    """

    pytest.importorskip("mujoco")
    cfg = WheelbipeFlatV0Cfg()
    env = WheelbipeVariantEnv(cfg, num_envs=1, backend_type="mujoco")
    try:
        calls: list[tuple[np.ndarray, np.ndarray]] = []
        original = env._capture_physics_delayed_observation  # noqa: SLF001

        def capture(backend, full_pos, full_vel):
            calls.append((np.asarray(full_pos).copy(), np.asarray(full_vel).copy()))
            return original(backend, full_pos, full_vel)

        monkeypatch.setattr(env, "_capture_physics_delayed_observation", capture)
        settled: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        original_settled = env._capture_post_step_delayed_observation  # noqa: SLF001

        def capture_settled(gyro, gravity, dof_pos, dof_vel):
            settled.append(
                (
                    np.asarray(gyro).copy(),
                    np.asarray(gravity).copy(),
                    np.asarray(dof_pos).copy(),
                    np.asarray(dof_vel).copy(),
                )
            )
            return original_settled(gyro, gravity, dof_pos, dof_vel)

        monkeypatch.setattr(env, "_capture_post_step_delayed_observation", capture_settled)
        env.init_state()
        calls.clear()
        env.step(np.zeros((1, 6), dtype=np.float32))
        assert len(calls) == 3
        assert len(settled) == 1
    finally:
        env.close()


def test_delay_reset_rejects_negative_or_fractional_environment_ids() -> None:
    cfg = WheelbipeV14FlatCfg()
    env = object.__new__(WheelbipeV14Env)
    env._cfg = cfg
    env._num_envs = 2
    env._np_dtype = np.dtype(np.float32)
    env._init_delay_buffers()
    with pytest.raises(IndexError, match="out of range"):
        env._reset_delay_buffers(np.asarray([-1], dtype=np.int32))
    with pytest.raises(ValueError, match="integer environment indices"):
        env._reset_delay_buffers(np.asarray([0.5], dtype=np.float64))


def test_profile_overrides_are_explicit() -> None:
    overrides = wheelbipe_delay_profile_overrides(WHEELBIPE_DELAY_PROFILE_SOURCE_V14)
    assert overrides == {
        "delay_profile": WHEELBIPE_DELAY_PROFILE_SOURCE_V14,
        "delay_range_semantics": "exclusive",
        "sim_dt": 0.005,
        "ctrl_dt": 0.02,
        "obs_delay_step_unit": "physics",
        "use_obs_delay": True,
        "obs_history_len": 10,
        "obs_default_time_lag": 1,
        "obs_delay_cfg": {
            "root_ang_vel_b": [1, 4],
            "projected_gravity_b": [1, 4],
            "joint_pos": [1, 4],
            "joint_vel": [1, 4],
        },
        "use_act_delay": True,
        "act_history_len": 5,
        "act_delay_cfg": {"leg_actions": [1, 3], "wheel_actions": [1, 3]},
    }


def test_local_profile_override_is_complete_and_does_not_inherit_source_timing() -> None:
    overrides = wheelbipe_delay_profile_overrides("local_physics")
    assert overrides["delay_profile"] == "local_physics"
    assert overrides["delay_range_semantics"] == "inclusive"
    assert overrides["sim_dt"] == pytest.approx(0.001)
    assert overrides["ctrl_dt"] == pytest.approx(0.02)
    assert overrides["obs_delay_step_unit"] == "control"
    assert overrides["use_obs_delay"] is False
    assert overrides["use_act_delay"] is False
    assert overrides["obs_history_len"] == 10
    assert overrides["act_history_len"] == 5
    assert overrides["obs_default_time_lag"] == 1
    assert overrides["obs_delay_cfg"] == {
        "root_ang_vel_b": [1, 4],
        "projected_gravity_b": [1, 4],
        "joint_pos": [1, 4],
        "joint_vel": [1, 4],
    }
    assert overrides["act_delay_cfg"] == {
        "leg_actions": [1, 3],
        "wheel_actions": [1, 3],
    }


def test_local_profile_override_normalizes_compact_custom_owner() -> None:
    cfg = WheelbipeHIMCfg()
    for key, value in wheelbipe_delay_profile_overrides("local_physics").items():
        setattr(cfg, key, value)
    assert cfg.delay_profile == "local_physics"
    assert cfg.sim_dt == pytest.approx(0.001)
    assert cfg.ctrl_dt == pytest.approx(0.02)
    assert cfg.obs_delay_step_unit == "control"
    assert cfg.use_obs_delay is False
    assert cfg.use_act_delay is False
    cfg.validate()


def test_source_profile_overrides_validate_at_owner_boundary() -> None:
    cfg = WheelbipeV14FlatCfg()
    for key, value in wheelbipe_delay_profile_overrides(WHEELBIPE_DELAY_PROFILE_SOURCE_V14).items():
        setattr(cfg, key, value)
    cfg.validate()


def test_torque_limits_intersect_backend_ranges_and_report_narrowing() -> None:
    cfg = SimpleNamespace(
        leg_torque_limit=50.9,
        wheel_torque_limit=9.99,
        spring_torque_limit=1000.0,
    )
    backend_ranges = np.asarray(
        [
            [-54.0, 54.0],
            [-54.0, 54.0],
            [-5.0, 5.0],
            [-1000.0, 1000.0],
            [-54.0, 54.0],
            [-54.0, 54.0],
            [-5.0, 5.0],
            [-1000.0, 1000.0],
        ],
        dtype=np.float64,
    )
    lower, upper, contract = resolve_wheelbipe_torque_limits(
        cfg,  # type: ignore[arg-type]
        backend_ranges,
        native_leg_indices=np.asarray([0, 1, 4, 5]),
        native_wheel_indices=np.asarray([2, 6]),
        native_spring_indices=np.asarray([3, 7]),
    )
    np.testing.assert_allclose(lower[[0, 1, 4, 5]], -50.9)
    np.testing.assert_allclose(upper[[0, 1, 4, 5]], 50.9)
    np.testing.assert_allclose(lower[[2, 6]], -5.0)
    np.testing.assert_allclose(upper[[2, 6]], 5.0)
    assert contract["groups"]["wheel"]["backend_clips_owner"] is True
    assert contract["groups"]["leg"]["backend_clips_owner"] is False


def test_torque_limits_reject_empty_intersection() -> None:
    cfg = SimpleNamespace(
        leg_torque_limit=1.0,
        wheel_torque_limit=9.99,
        spring_torque_limit=1000.0,
    )
    ranges = np.asarray([[-1.0, 1.0]] * 8, dtype=np.float64)
    ranges[2] = [10.0, 11.0]
    with pytest.raises(ValueError, match="wheel torque contract has no intersection"):
        resolve_wheelbipe_torque_limits(
            cfg,  # type: ignore[arg-type]
            ranges,
            native_leg_indices=[0, 1, 4, 5],
            native_wheel_indices=[2, 6],
            native_spring_indices=[3, 7],
        )
