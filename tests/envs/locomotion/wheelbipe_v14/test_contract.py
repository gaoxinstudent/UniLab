"""Unit tests for the Wheelbipe V14 policy/control contract.

The observation and action adapters are deliberately pure NumPy functions.
Keeping their tests independent of a simulator makes it inexpensive to catch
the two most dangerous deployment regressions: a reordered observation field
or an action written to the wrong native actuator slot.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.wheelbipe_v14.base import (
    COMPACT_PRIVILEGED_OBS_HEIGHT_CLIP,
    COMPACT_PRIVILEGED_OBS_HEIGHT_SCALE,
    COMPACT_PRIVILEGED_ROOT_LINVEL_CLIP,
    POLICY_OBS_CLIP,
    POLICY_OBS_DIM,
    WheelbipeDelayBuffer,
    build_wheelbipe_policy_observation,
    compute_wheelbipe_motor_ctrl,
    map_policy_action_to_native_targets,
    sample_wheelbipe_delay_lags,
)
from unilab.envs.locomotion.wheelbipe_v14.joystick import (
    WheelbipeV14FlatCfg,
    build_wheelbipe_backend_reset_randomization,
)


def test_policy_observation_layout_and_scales() -> None:
    """The 35 fields remain in the public ROS deployment order."""

    commands = np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)
    height = np.asarray([0.4, -0.2], dtype=np.float32)
    gyro = np.asarray([[4.0, 5.0, 6.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    gravity = np.asarray([[7.0, 8.0, 9.0], [-7.0, -8.0, -9.0]], dtype=np.float32)
    leg_pos = np.asarray([[10.0, 11.0, 12.0, 13.0], [-10.0, -11.0, -12.0, -13.0]])
    leg_vel = np.asarray([[14.0, 15.0, 16.0, 17.0], [-14.0, -15.0, -16.0, -17.0]])
    wheel_vel = np.asarray([[18.0, 19.0], [-18.0, -19.0]])
    previous_actions = np.asarray(
        [[20.0, 21.0, 22.0, 23.0, 24.0, 25.0], [-20.0, -21.0, -22.0, -23.0, -24.0, -25.0]],
        dtype=np.float32,
    )

    obs = build_wheelbipe_policy_observation(
        commands,
        height,
        gyro,
        gravity,
        leg_pos,
        leg_vel,
        wheel_vel,
        previous_actions,
    )

    expected = np.concatenate(
        (
            commands,
            height[:, None] * 5.0,
            gyro * 0.5,
            gravity,
            leg_pos,
            np.zeros((2, 2), dtype=np.float32),  # wheel position is reserved/zero
            leg_vel * 0.1,
            wheel_vel * 0.1,
            previous_actions,
            np.tile(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (2, 1)),
        ),
        axis=1,
    )

    assert obs.shape == (2, POLICY_OBS_DIM)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs, expected)


def test_policy_observation_rejects_mismatched_batch() -> None:
    components = [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),  # height has a different batch size
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 6), dtype=np.float32),
    ]
    with pytest.raises(ValueError, match="same batch size"):
        build_wheelbipe_policy_observation(*components)


def test_policy_observation_clips_scaled_fields_to_ros_contract() -> None:
    """Extreme sensor values cannot escape the deployment [-100, 100] bound."""

    obs = build_wheelbipe_policy_observation(
        np.full((1, 3), 1.0e6, dtype=np.float32),
        np.asarray([1.0e6], dtype=np.float32),
        np.full((1, 3), -1.0e6, dtype=np.float32),
        np.full((1, 3), 1.0e6, dtype=np.float32),
        np.full((1, 4), -1.0e6, dtype=np.float32),
        np.full((1, 4), 1.0e6, dtype=np.float32),
        np.full((1, 2), -1.0e6, dtype=np.float32),
        np.full((1, 6), 1.0e6, dtype=np.float32),
    )

    assert np.all(np.isfinite(obs))
    assert float(np.max(obs)) == POLICY_OBS_CLIP
    assert float(np.min(obs)) == -POLICY_OBS_CLIP
    # The deployment helper scales height before clipping (1e6 * 5 -> 100),
    # rather than applying the training-only raw-height [0, 1] bound first.
    assert float(obs[0, 3]) == POLICY_OBS_CLIP


def test_source_training_policy_observation_clips_raw_components_before_scale() -> None:
    """Pinned Isaac actor bounds differ intentionally from the ROS wrapper."""

    obs = build_wheelbipe_policy_observation(
        np.asarray([[150.0, -150.0, 0.0]], dtype=np.float32),
        np.asarray([-3.0], dtype=np.float32),
        np.asarray([[150.0, -150.0, 0.0]], dtype=np.float32),
        np.asarray([[150.0, -150.0, 0.0]], dtype=np.float32),
        np.asarray([[150.0, -150.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[300.0, -300.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[300.0, -300.0]], dtype=np.float32),
        np.asarray([[150.0, -150.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        control_mode=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 150.0, 0.0]], dtype=np.float32),
        control_mode_scale=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]),
        source_training_clips=True,
    )

    np.testing.assert_array_equal(obs[0, 0:3], [100.0, -100.0, 0.0])
    assert float(obs[0, 3]) == 0.0
    np.testing.assert_array_equal(obs[0, 4:7], [50.0, -50.0, 0.0])
    np.testing.assert_array_equal(obs[0, 7:10], [100.0, -100.0, 0.0])
    np.testing.assert_array_equal(obs[0, 10:14], [100.0, -100.0, 0.0, 0.0])
    np.testing.assert_array_equal(obs[0, 16:20], [20.0, -20.0, 0.0, 0.0])
    np.testing.assert_array_equal(obs[0, 20:22], [20.0, -20.0])
    np.testing.assert_array_equal(obs[0, 22:28], [100.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    # ctrl_mode_obs has a scale config but no clip config in pinned V14.
    assert float(obs[0, 33]) == 750.0


def test_source_training_policy_observation_sanitizes_only_assembled_actor() -> None:
    """Pinned source replaces actor NaNs after concat without mutating actions."""

    actions = np.asarray([[np.nan, np.inf, -np.inf, 1.0, 2.0, 3.0]], dtype=np.float32)
    original_actions = actions.copy()
    obs = build_wheelbipe_policy_observation(
        np.asarray([[np.nan, np.inf, -np.inf]], dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([[np.nan, np.inf, -np.inf]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        actions,
        control_mode=np.asarray([[np.inf] * 7], dtype=np.float32),
        source_training_clips=True,
    )

    assert np.all(np.isfinite(obs))
    # NaNs survive np.clip and are replaced only at the final actor boundary.
    assert float(obs[0, 0]) == 0.0
    assert float(obs[0, 3]) == 0.0
    assert float(obs[0, 4]) == 0.0
    assert float(obs[0, 22]) == 0.0
    assert np.all(obs[0, -7:] == 0.0)
    # Infinities are already consumed by the source component clamp.
    np.testing.assert_array_equal(obs[0, 1:3], [100.0, -100.0])
    np.testing.assert_array_equal(obs[0, 23:25], [100.0, -100.0])
    # Observation sanitization must not become an action-input sanitizer.
    np.testing.assert_array_equal(actions, original_actions)


def test_policy_observation_accepts_owner_control_mode_tail() -> None:
    """State-machine owners can replace the normal one-hot tail in-place."""

    mode = np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    obs = build_wheelbipe_policy_observation(
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 6), dtype=np.float32),
        control_mode=mode,
    )
    np.testing.assert_array_equal(obs[0, -7:], mode[0])


def test_compact_partial_reset_uses_selected_backend_height_batch() -> None:
    """A compact reset must build a privileged row for only selected envs."""

    pytest.importorskip("mujoco")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14FlatHIM",
        sim_backend="mujoco",
        num_envs=2,
    )
    try:
        initial = env.init_state()
        assert initial.obs["obs"].shape == (2, 28)
        obs, _info = env.reset(np.asarray([0], dtype=np.int32))
        assert obs["obs"].shape == (1, 28)
        assert obs["critic"].shape == (1, 32)
    finally:
        env.close()


def test_compact_critic_tail_matches_source_clip_and_scale_contract() -> None:
    """Compact privileged velocity/height fields retain source V14 scaling."""

    pytest.importorskip("mujoco")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14FlatHIM",
        sim_backend="mujoco",
        num_envs=1,
    )
    try:
        # Call the owner observation boundary with deliberately out-of-range
        # privileged values.  Source V14 clips root_lin_vel_b to +/-100 and
        # clips raw obs_height to [-10, 10] before multiplying it by five.
        out = env._compute_obs(  # noqa: SLF001 - contract regression at owner boundary
            {
                "commands": np.zeros((1, 3), dtype=np.float32),
                "height_commands": np.zeros((1,), dtype=np.float32),
                "current_actions": np.zeros((1, 6), dtype=np.float32),
                "observed_height": np.asarray([20.0], dtype=np.float32),
                "torques": np.zeros((1, 8), dtype=np.float32),
            },
            np.asarray([[150.0, -125.0, np.nan]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
            np.zeros((1, 6), dtype=np.float32),
            np.zeros((1, 6), dtype=np.float32),
            env_ids=None,
        )
        expected_velocity = np.asarray(
            [
                COMPACT_PRIVILEGED_ROOT_LINVEL_CLIP,
                -COMPACT_PRIVILEGED_ROOT_LINVEL_CLIP,
                0.0,
            ],
            dtype=np.float32,
        )
        expected_height = (
            COMPACT_PRIVILEGED_OBS_HEIGHT_CLIP[1] * COMPACT_PRIVILEGED_OBS_HEIGHT_SCALE
        )
        np.testing.assert_allclose(out["critic"][0, -4:-1], expected_velocity)
        np.testing.assert_allclose(out["critic"][0, -1], expected_height)
    finally:
        env.close()


def test_wheelbipe_delay_buffer_returns_per_env_lag_and_resets_history() -> None:
    buffer = WheelbipeDelayBuffer(history_len=3, num_envs=2, width=1, dtype=np.dtype(np.float32))
    buffer.set_time_lag(np.asarray([0, 2], dtype=np.intp))
    np.testing.assert_allclose(
        # Isaac Lab's CircularBuffer fills an environment's history on its
        # first append, so a positive lag is immediately usable after reset.
        buffer.compute(np.asarray([[1.0], [1.0]], dtype=np.float32)), [[1.0], [1.0]]
    )
    np.testing.assert_allclose(
        buffer.compute(np.asarray([[2.0], [2.0]], dtype=np.float32)), [[2.0], [1.0]]
    )
    np.testing.assert_allclose(
        buffer.compute(np.asarray([[3.0], [3.0]], dtype=np.float32)), [[3.0], [1.0]]
    )
    buffer.reset(np.asarray([1], dtype=np.int32))
    np.testing.assert_allclose(
        buffer.compute(np.asarray([[4.0], [4.0]], dtype=np.float32)), [[4.0], [4.0]]
    )


def test_wheelbipe_delay_range_sampling_is_inclusive_for_fixed_and_random_bounds() -> None:
    np.testing.assert_array_equal(
        sample_wheelbipe_delay_lags((2, 2), 4, name="delay"), np.asarray([2, 2, 2, 2])
    )
    np.random.seed(7)
    sampled = sample_wheelbipe_delay_lags((0, 2), 256, name="delay")
    assert sampled.min() >= 0
    assert sampled.max() <= 2


def test_wheelbipe_cfg_validates_enabled_delay_contract() -> None:
    cfg = WheelbipeV14FlatCfg()
    cfg.use_obs_delay = True
    # The source profile samples an exclusive upper bound, so ``12`` makes
    # lag 11 reachable and therefore exceeds the ten-frame history.
    cfg.obs_delay_cfg = {"joint_pos": (0, 12)}
    with pytest.raises(ValueError, match="exceeds obs_history_len"):
        cfg.validate()

    cfg = WheelbipeV14FlatCfg()
    cfg.use_act_delay = True
    cfg.control_config.simulate_action_latency = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.validate()


def test_policy_action_mapping_keeps_leg_wheel_and_spring_slots_distinct() -> None:
    actions = np.asarray([[1.0, -1.0, 0.5, -0.5, 0.2, -0.2]], dtype=np.float32)
    targets = map_policy_action_to_native_targets(
        actions,
        action_scale=0.5,
        wheel_action_scale=10.0,
        default_leg_position=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        native_leg_indices=np.asarray([0, 1, 4, 5]),
        native_wheel_indices=np.asarray([2, 6]),
        native_spring_indices=np.asarray([3, 7]),
        spring_target=0.07,
    )

    expected = np.asarray([[0.6, -0.3, 2.0, 0.07, 0.55, 0.15, -2.0, 0.07]], dtype=np.float32)
    assert targets.shape == (1, 8)
    np.testing.assert_allclose(targets, expected)


def test_source_policy_action_mapping_clamps_decoded_physical_targets_only() -> None:
    actions = np.asarray([[20.0, -20.0, 8.0, -8.0, 20.0, -20.0]], dtype=np.float32)
    targets = map_policy_action_to_native_targets(
        actions,
        action_scale=0.5,
        wheel_action_scale=10.0,
        default_leg_position=np.zeros(4, dtype=np.float32),
        native_leg_indices=np.asarray([0, 1, 4, 5]),
        native_wheel_indices=np.asarray([2, 6]),
        native_spring_indices=np.asarray([3, 7]),
        leg_position_limit=(-3.14, 3.14),
        wheel_velocity_limit=100.0,
    )

    np.testing.assert_allclose(targets[0, [0, 1, 4, 5]], [3.14, -3.14, 3.14, -3.14])
    np.testing.assert_allclose(targets[0, [2, 6]], [100.0, -100.0])


def test_motor_control_uses_native_slots_and_clips_each_actuator_group() -> None:
    native_targets = np.asarray([[1.0, 2.0, 30.0, 0.0, 4.0, 5.0, -20.0, 0.0]], dtype=np.float32)
    full_pos = np.asarray([[0.1, -0.2, 0.0, 0.0, 0.3, -0.4, 0.0, 0.0]], dtype=np.float32)
    full_vel = np.asarray([[0.2, -0.1, 1.0, -2.0, 0.3, -0.4, 3.0, -4.0]], dtype=np.float32)
    out = np.empty((1, 8), dtype=np.float32)

    ctrl = compute_wheelbipe_motor_ctrl(
        native_targets,
        full_pos,
        full_vel,
        leg_pos_indices=np.asarray([0, 1, 4, 5]),
        leg_vel_indices=np.asarray([0, 1, 4, 5]),
        wheel_vel_indices=np.asarray([2, 6]),
        spring_pos_indices=np.asarray([3, 7]),
        spring_vel_indices=np.asarray([3, 7]),
        native_leg_indices=np.asarray([0, 1, 4, 5]),
        native_wheel_indices=np.asarray([2, 6]),
        native_spring_indices=np.asarray([3, 7]),
        leg_kp=np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        leg_kd=np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        wheel_kd=np.asarray([[0.5, 0.75]], dtype=np.float32),
        spring_force=240.0,
        spring_damping=50.0,
        lower=np.asarray([-100, -100, -10, -400, -100, -100, -10, -400], dtype=np.float32),
        upper=np.asarray([100, 100, 10, 400, 100, 100, 10, 400], dtype=np.float32),
        out=out,
    )

    expected = np.asarray([[8.8, 44.2, 10.0, 340.0, 100.0, 100.0, -10.0, 400.0]], dtype=np.float32)
    assert ctrl is out
    np.testing.assert_allclose(ctrl, expected, rtol=1e-5, atol=1e-5)


def test_linear_spring_matches_v14_force_curve_and_preload_randomization() -> None:
    """V14 uses 400--600 N over compressed spring travel plus episode preload."""

    out = np.empty((1, 8), dtype=np.float32)
    ctrl = compute_wheelbipe_motor_ctrl(
        np.zeros((1, 8), dtype=np.float32),
        np.asarray([[0, 0, 0, 0.010, 0, 0, 0, 0.06076]], dtype=np.float32),
        np.zeros((1, 8), dtype=np.float32),
        leg_pos_indices=np.asarray([0, 1, 4, 5]),
        leg_vel_indices=np.asarray([0, 1, 4, 5]),
        wheel_vel_indices=np.asarray([2, 6]),
        spring_pos_indices=np.asarray([3, 7]),
        spring_vel_indices=np.asarray([3, 7]),
        native_leg_indices=np.asarray([0, 1, 4, 5]),
        native_wheel_indices=np.asarray([2, 6]),
        native_spring_indices=np.asarray([3, 7]),
        leg_kp=np.zeros((1, 4), dtype=np.float32),
        leg_kd=np.zeros((1, 4), dtype=np.float32),
        wheel_kd=np.zeros((1, 2), dtype=np.float32),
        spring_force=240.0,
        spring_damping=0.0,
        spring_mode="linear",
        spring_offset=0.06076,
        spring_linear_up=600.0,
        spring_linear_down=400.0,
        spring_linear_length=0.07,
        spring_force_random=np.asarray([[10.0, -10.0]], dtype=np.float32),
        lower=np.full(8, -1000.0, dtype=np.float32),
        upper=np.full(8, 1000.0, dtype=np.float32),
        out=out,
    )
    # left: 400 + (600-400)/.07*(.06076-.010) + 10; right: 400 - 10.
    np.testing.assert_allclose(ctrl[0, [3, 7]], [555.0286, 390.0], rtol=1e-5, atol=1e-4)


def test_reset_randomization_scales_cached_body_mass_and_armature() -> None:
    """Optional generic DR terms remain owner-layer and cold-path safe."""

    env = SimpleNamespace(
        _base_body_mass=np.asarray([0.0, 2.0, 4.0]),
        _base_dof_armature=np.asarray([0.0, 0.1, 0.2]),
        cfg=SimpleNamespace(
            domain_rand=SimpleNamespace(
                randomize_base_mass=False,
                randomize_body_mass=True,
                body_mass_multiplier_range=[2.0, 2.0],
                random_com=False,
                randomize_gravity=False,
                randomize_ground_friction=False,
                randomize_dof_armature=True,
                dof_armature_multiplier_range=[3.0, 3.0],
            )
        ),
    )
    payload = build_wheelbipe_backend_reset_randomization(env, 2)
    assert payload is not None
    np.testing.assert_allclose(payload.body_mass, [[0.0, 4.0, 8.0]] * 2)
    np.testing.assert_allclose(payload.dof_armature, [[0.0, 0.3, 0.6]] * 2)
