"""Pinned source-V14 training semantics regression tests."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.envs.locomotion.wheelbipe_v14 import joystick as joystick_module
from unilab.envs.locomotion.wheelbipe_v14.joystick import (
    WheelbipeCommands,
    WheelbipeV14Env,
)
from unilab.envs.locomotion.wheelbipe_v14.semantics import (
    SOURCE_V14_GUIDE_BODY_NAMES,
    SOURCE_V14_RESET_CONTACT_BODY_NAMES,
    SOURCE_V14_REWARD_SCALES,
    SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES,
    SourceV14HIMCurriculum,
    SourceV14RewardParameters,
    apply_source_v14_termination_duration,
    build_source_v14_critic_observation,
    build_source_v14_height_signals,
    compute_source_v14_reward,
    sample_source_v14_commands,
    sample_source_v14_material_buckets,
    source_v14_inverse_kinematics,
)


def test_source_v14_contact_sets_match_v14_owner() -> None:
    """V14 adds gimbal/guide contacts to both diagnostics and flat reset."""

    assert SOURCE_V14_GUIDE_BODY_NAMES
    assert SOURCE_V14_RESET_CONTACT_BODY_NAMES == (
        "base_link",
        "gimbal_yaw_link",
        "gimbal_pitch_link",
        *SOURCE_V14_GUIDE_BODY_NAMES,
    )
    assert all(
        name in SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES for name in SOURCE_V14_GUIDE_BODY_NAMES
    )
    assert "gimbal_yaw_link" in SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES
    assert "gimbal_pitch_link" in SOURCE_V14_UNDESIRED_CONTACT_BODY_NAMES


from unilab.envs.locomotion.wheelbipe_v14.variants import (
    WheelbipeCustomEnv,
    WheelbipeFlatV0Cfg,
    WheelbipeFlatV1Cfg,
    WheelbipeHIMCfg,
    WheelbipeHIMPlayCfg,
    WheelbipeNP3OCfg,
    WheelbipeNP3OPlayCfg,
    WheelbipeRoughV0Cfg,
    WheelbipeRoughV1Cfg,
    WheelbipeVariantEnv,
)


def test_source_critic_has_exact_base39_and_privileged39_layout() -> None:
    n = 2
    critic = build_source_v14_critic_observation(
        commands=np.full((n, 3), 1.0),
        height_command=np.full((n,), 0.2),
        gyro=np.full((n, 3), 2.0),
        projected_gravity=np.full((n, 3), 3.0),
        dof_pos=np.arange(12, dtype=np.float32).reshape(n, 6),
        dof_vel=np.full((n, 6), 4.0),
        actions=np.full((n, 6), 5.0),
        root_lin_vel_b=np.full((n, 3), 6.0),
        observed_height=np.full((n,), 0.3),
        control_mode=np.tile(np.arange(7, dtype=np.float32), (n, 1)),
        joint_stiffness=np.full((n, 6), 7.0),
        joint_damping=np.full((n, 6), 8.0),
        applied_torque=np.full((n, 6), 9.0),
        obs_delay_steps=np.tile(np.arange(4, dtype=np.float32), (n, 1)),
        act_delay_steps=np.tile(np.arange(2, dtype=np.float32), (n, 1)),
        wheel_body_lin_vel_b=np.full((n, 2, 3), 10.0),
        wheel_contact_state=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        base_mass_scale=np.asarray([1.1, 1.2]),
        wheel_material=np.arange(12, dtype=np.float32).reshape(n, 2, 3),
        default_angles=np.zeros((6,), dtype=np.float32),
        control_mode_scale=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]),
    )

    assert critic.shape == (2, 78)
    np.testing.assert_allclose(critic[:, 3], 1.0)  # height * 5
    np.testing.assert_allclose(critic[:, 4:7], 1.0)  # gyro * 0.5
    np.testing.assert_allclose(critic[:, 14:16], 0.0)  # muted wheel position
    np.testing.assert_allclose(critic[:, 28:31], 6.0)
    np.testing.assert_allclose(critic[:, 31], 1.5)  # observed height * 5
    np.testing.assert_allclose(critic[:, 37], 25.0)  # raw mode[5], then *5 (no clip entry)
    np.testing.assert_allclose(critic[:, 39:45], 7.0)
    np.testing.assert_allclose(critic[:, 45:51], 8.0)
    np.testing.assert_allclose(critic[:, 51:57], 0.45)  # torque * 0.05
    np.testing.assert_allclose(critic[:, 57:61], np.tile(np.arange(4), (n, 1)))
    np.testing.assert_allclose(critic[:, 63:69], 10.0)
    np.testing.assert_allclose(critic[:, 69:71], [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(critic[:, 71], [1.1, 1.2])
    np.testing.assert_allclose(critic[:, 72:78], np.arange(12).reshape(2, 6))


def test_source_critic_sanitizes_concatenated_base_and_privileged_fields() -> None:
    """Pinned ``priv``/``priv_latent`` replace non-finite concat outputs."""

    n = 1
    zeros2 = np.zeros((n, 2), dtype=np.float32)
    zeros3 = np.zeros((n, 3), dtype=np.float32)
    zeros4 = np.zeros((n, 4), dtype=np.float32)
    zeros6 = np.zeros((n, 6), dtype=np.float32)
    critic = build_source_v14_critic_observation(
        commands=np.asarray([[np.nan, np.inf, -np.inf]], dtype=np.float32),
        height_command=np.asarray([np.nan], dtype=np.float32),
        gyro=zeros3,
        projected_gravity=zeros3,
        dof_pos=zeros6,
        dof_vel=zeros6,
        actions=zeros6,
        root_lin_vel_b=zeros3,
        observed_height=np.asarray([np.nan], dtype=np.float32),
        control_mode=np.asarray([[np.inf] * 7], dtype=np.float32),
        joint_stiffness=np.asarray([[np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        joint_damping=zeros6,
        applied_torque=zeros6,
        obs_delay_steps=zeros4,
        act_delay_steps=zeros2,
        wheel_body_lin_vel_b=zeros6,
        wheel_contact_state=zeros2,
        base_mass_scale=np.asarray([np.nan], dtype=np.float32),
        wheel_material=zeros6,
        default_angles=np.zeros((6,), dtype=np.float32),
    )

    assert critic.shape == (1, 78)
    assert np.all(np.isfinite(critic))
    assert float(critic[0, 0]) == 0.0
    assert float(critic[0, 3]) == 0.0
    assert float(critic[0, 31]) == 0.0
    assert float(critic[0, 39]) == 0.0
    assert float(critic[0, 71]) == 0.0
    assert np.all(critic[0, 32:39] == 0.0)
    # Source component clamps consume infinities before final nan_to_num.
    np.testing.assert_array_equal(critic[0, 1:3], [100.0, -100.0])


def test_np3o_velocity_tracking_uses_source_linear_height_gate() -> None:
    zeros3 = np.zeros((3, 3), dtype=np.float32)
    zeros6 = np.zeros((3, 6), dtype=np.float32)
    _reward, terms = compute_source_v14_reward(
        scales={"track_lin_vel_xy": 1.0, "track_ang_vel_z": 1.0},
        ctrl_dt=1.0,
        commands=zeros3,
        linvel=zeros3,
        gyro=zeros3,
        projected_gravity=np.tile([0.0, 0.0, -1.0], (3, 1)),
        pitch=np.zeros((3,), dtype=np.float32),
        observed_height=np.asarray([0.35, 0.375, 0.40], dtype=np.float32),
        height_command=np.full((3,), 0.30, dtype=np.float32),
        dof_pos=zeros6,
        dof_vel=zeros6,
        qacc=zeros6,
        torque=zeros6,
        actions=zeros6,
        last_actions=zeros6,
        previous_actions=zeros6,
        wheel_pos_b=np.zeros((3, 2, 3), dtype=np.float32),
        wheel_contact_state=np.zeros((3, 2), dtype=bool),
        undesired_contact=np.zeros((3,), dtype=bool),
        terminated=np.zeros((3,), dtype=bool),
        params=SourceV14RewardParameters(
            vel_height_gate_enabled=True,
            vel_height_gate_mode="linear_band",
            vel_height_gate_full_error=0.05,
            vel_height_gate_zero_error=0.1,
        ),
    )

    # Zero velocity error makes both ungated tracking terms one.  Height
    # errors at the full/mid/zero thresholds expose the gate directly.
    np.testing.assert_allclose(terms["track_lin_vel_xy"], [1.0, 0.5, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(terms["track_ang_vel_z"], [1.0, 0.5, 0.0], atol=1.0e-6)


def test_np3o_train_and_play_own_distinct_source_height_gate_profiles() -> None:
    train = WheelbipeNP3OCfg()
    play = WheelbipeNP3OPlayCfg()

    assert train.vel_height_gate_enabled
    assert not play.vel_height_gate_enabled
    train.validate()
    play.validate()
    train.vel_height_gate_zero_error = 0.2
    with pytest.raises(ValueError, match="immutable source velocity-height gate parameters"):
        train.validate()


def test_him_curriculum_reproduces_reward_and_assist_force_stage_transitions() -> None:
    curriculum = SourceV14HIMCurriculum(
        default_reward_scales=SOURCE_V14_REWARD_SCALES,
        reward_key="track_height_exp",
        num_steps_per_env=1,
        window_size=2,
        min_stage_episodes=1,
        normalize_by_episode_length=True,
        reward_stage_weights=[
            {"track_height_exp": 1.0, "track_height_exp_tight": 1.0},
            {"track_height_exp": 0.8, "track_height_exp_tight": 0.6},
        ],
        assist_force_z_stages=[160.0, 80.0, 0.0],
        thresholds=[0.4, 0.4],
        stage_min_episodes=[2, 2],
        restore_defaults_after_final_threshold=True,
    )

    assert curriculum.stage == 0
    assert curriculum.assist_force_z == 160.0
    assert curriculum.reward_scales["track_height_exp"] == 1.0
    assert curriculum.reward_scales["track_height_exp_tight"] == 1.0
    assert not curriculum.record_completed_batch(np.asarray([0.5]), max_episode_length_s=1.0)
    assert curriculum.record_completed_batch(np.asarray([0.5]), max_episode_length_s=1.0)
    assert curriculum.stage == 1
    assert curriculum.assist_force_z == 80.0
    assert curriculum.reward_scales["track_height_exp"] == 0.8
    assert curriculum.reward_scales["track_height_exp_tight"] == 0.6
    assert not curriculum.record_completed_batch(np.asarray([0.5]), max_episode_length_s=1.0)
    assert curriculum.record_completed_batch(np.asarray([0.5]), max_episode_length_s=1.0)
    assert curriculum.stage == 2
    assert curriculum.assist_force_z == 0.0
    assert curriculum.reward_scales["track_height_exp"] == 0.0
    assert curriculum.reward_scales["track_height_exp_tight"] == 1.0

    snapshot = curriculum.contract_snapshot()
    reward_state = snapshot["track_height_progression"]
    assist_state = snapshot["base_vertical_assist_force_progression"]
    assert reward_state["stage"] == 1
    assert reward_state["stage_count"] == 2
    assert reward_state["defaults_restored"] == 1
    assert assist_state["stage"] == 2
    assert assist_state["stage_count"] == 3
    assert assist_state["force_z"] == 0.0


def test_him_curriculum_exact_source_count_boundaries_and_rng_identity() -> None:
    before = np.random.get_state()
    curriculum = SourceV14HIMCurriculum(
        default_reward_scales=SOURCE_V14_REWARD_SCALES,
        reward_key="track_height_exp",
        num_steps_per_env=24,
        window_size=64,
        min_stage_episodes=64,
        normalize_by_episode_length=True,
        reward_stage_weights=[
            {"track_height_exp": 1.0, "track_height_exp_tight": 1.0},
            {"track_height_exp": 0.8, "track_height_exp_tight": 0.6},
        ],
        assist_force_z_stages=[160.0, 80.0, 0.0],
        thresholds=[0.4, 0.4],
        stage_min_episodes=[500, 500],
        restore_defaults_after_final_threshold=True,
    )
    # Source multiplies 64 and 500 by 24.  A vectorized reset batch is one
    # sample regardless of how many environment ids it contains.
    for _ in range(11_999):
        assert not curriculum.record_completed_batch(
            np.asarray([10.0, 10.0]), max_episode_length_s=20.0
        )
    assert curriculum.stage == 0
    assert curriculum.record_completed_batch(np.asarray([10.0, 10.0]), max_episode_length_s=20.0)
    assert curriculum.stage == 1
    first = curriculum.contract_snapshot()["track_height_progression"]
    assert first["window_samples"] == 64 * 24
    assert first["min_compute_calls"] == 500 * 24
    for _ in range(11_999):
        assert not curriculum.record_completed_batch(np.asarray([10.0]), max_episode_length_s=20.0)
    assert curriculum.record_completed_batch(np.asarray([10.0]), max_episode_length_s=20.0)
    assert curriculum.stage == 2

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_him_assist_force_uses_source_wrench_buffer_overwrite_order() -> None:
    provider = joystick_module.WheelbipeV14DomainRandomizationProvider()
    next_steps = np.asarray([0, 10], dtype=np.int64)
    env = SimpleNamespace(
        _source_semantics=True,
        _num_envs=2,
        _np_dtype=np.dtype(np.float32),
        _base_body_ids=np.asarray([3], dtype=np.int32),
        _source_force_next_step=next_steps,
        _source_body_force=np.zeros((2, 3), dtype=np.float32),
        _source_body_torque=np.zeros((2, 3), dtype=np.float32),
        _source_him_assist_latched=np.ones((2,), dtype=bool),
        source_him_assist_force_z=160.0,
        _reset_source_force_timers=lambda ids: next_steps.__setitem__(ids, 10),
        cfg=SimpleNamespace(
            him_curriculum=SimpleNamespace(enabled=True),
            domain_rand=SimpleNamespace(
                source_push_velocity_enabled=False,
                source_external_force_enabled=True,
                source_external_force_range=[2.0, 2.0],
                source_external_torque_range=[3.0, 3.0],
            ),
        ),
    )

    plan = provider.build_interval_randomization_plan(env, step_counter=0)

    assert plan is not None
    np.testing.assert_array_equal(plan.body_ids, [3])
    assert plan.body_force is not None
    assert plan.body_torque is not None
    # The due interval event overwrites env 0's assist with its sampled
    # wrench.  Env 1 keeps the force last written by the assist manager term;
    # the two forces are never added together.
    np.testing.assert_array_equal(
        plan.body_force,
        np.asarray([[[2.0, 2.0, 2.0]], [[0.0, 0.0, 160.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        plan.body_torque,
        np.asarray([[[3.0, 3.0, 3.0]], [[0.0, 0.0, 0.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(env._source_him_assist_latched, [False, True])


def test_him_curriculum_reset_counts_zero_batch_after_reward_buffer_exists() -> None:
    curriculum = SourceV14HIMCurriculum(
        default_reward_scales=SOURCE_V14_REWARD_SCALES,
        reward_key="track_height_exp",
        num_steps_per_env=1,
        window_size=2,
        min_stage_episodes=1,
        normalize_by_episode_length=True,
        reward_stage_weights=[
            {"track_height_exp": 1.0, "track_height_exp_tight": 1.0},
            {"track_height_exp": 0.8, "track_height_exp_tight": 0.6},
        ],
        assist_force_z_stages=[160.0, 80.0, 0.0],
        thresholds=[0.4, 0.4],
        stage_min_episodes=[2, 2],
        restore_defaults_after_final_threshold=True,
    )
    owner = cast(Any, WheelbipeV14Env.__new__(WheelbipeV14Env))
    owner._source_him_curriculum = curriculum
    owner._source_him_track_height_episode_sum = np.zeros((2,), dtype=np.float64)
    owner._source_him_reward_buffer_initialized = False
    owner._source_him_assist_latched = np.zeros((2,), dtype=bool)
    owner._source_runtime_reward_scales = curriculum.reward_scales
    owner._cfg = SimpleNamespace(max_episode_seconds=1.0)

    # Initial reset matches the source's missing ``_episode_sums`` key and
    # therefore applies assist without creating a curriculum sample.
    owner._update_source_him_curriculum_on_reset(np.asarray([0, 1], dtype=np.int32))
    initial = curriculum.contract_snapshot()["track_height_progression"]
    assert initial["total_compute_calls"] == 0
    np.testing.assert_array_equal(owner._source_him_assist_latched, [True, True])

    owner._source_him_reward_buffer_initialized = True
    owner._source_him_track_height_episode_sum[:] = 0.5
    owner._update_source_him_curriculum_on_reset(np.asarray([0, 1], dtype=np.int32))
    assert curriculum.contract_snapshot()["track_height_progression"]["total_compute_calls"] == 1

    # Once the source reward buffer exists, a second manual reset without a
    # step still contributes one zero-valued batch sample; it is not filtered
    # by a local per-environment "active episode" heuristic.
    owner._update_source_him_curriculum_on_reset(np.asarray([0], dtype=np.int32))
    second = curriculum.contract_snapshot()["track_height_progression"]
    assert second["total_compute_calls"] == 2
    assert second["last_batch_mean"] == 0.0
    assert curriculum.stage == 0


def test_him_curriculum_stage_advance_relatches_assist_for_every_environment() -> None:
    curriculum = SourceV14HIMCurriculum(
        default_reward_scales=SOURCE_V14_REWARD_SCALES,
        reward_key="track_height_exp",
        num_steps_per_env=1,
        window_size=1,
        min_stage_episodes=1,
        normalize_by_episode_length=True,
        reward_stage_weights=[
            {"track_height_exp": 1.0, "track_height_exp_tight": 1.0},
            {"track_height_exp": 0.8, "track_height_exp_tight": 0.6},
        ],
        assist_force_z_stages=[160.0, 80.0, 0.0],
        thresholds=[0.4, 0.4],
        stage_min_episodes=[1, 1],
        restore_defaults_after_final_threshold=True,
    )
    owner = cast(Any, WheelbipeV14Env.__new__(WheelbipeV14Env))
    owner._source_him_curriculum = curriculum
    owner._source_him_track_height_episode_sum = np.asarray([0.5, 0.0], dtype=np.float64)
    owner._source_him_reward_buffer_initialized = True
    owner._source_him_assist_latched = np.zeros((2,), dtype=bool)
    owner._source_runtime_reward_scales = curriculum.reward_scales
    owner._cfg = SimpleNamespace(max_episode_seconds=1.0)

    # Only env 0 resets, but a source assist-stage transition writes the new
    # force to every environment (``_apply_assist_force(None)``).
    owner._update_source_him_curriculum_on_reset(np.asarray([0], dtype=np.int32))

    assert curriculum.stage == 1
    assert owner._source_runtime_reward_scales["track_height_exp"] == 0.8
    np.testing.assert_array_equal(owner._source_him_assist_latched, [True, True])


def test_source_v1_clips_world_height_before_rough_reward_subtracts_terrain() -> None:
    observed, reward_height = build_source_v14_height_signals(
        np.asarray([0.80, 0.01], dtype=np.float32),
        np.asarray([0.20, -0.10], dtype=np.float32),
        use_absolute_height=False,
        clip_enabled=True,
        clip_range=(0.05, 0.45),
    )

    # The privileged critic sees clipped world-frame root z.  Only the reward
    # path then subtracts terrain; clipping the relative height instead would
    # incorrectly produce [0.45, 0.11] for this fixture.
    np.testing.assert_allclose(observed, [0.45, 0.05])
    np.testing.assert_allclose(reward_height, [0.25, 0.15])


def test_exact_v1_height_owner_is_distinct_from_v0_and_rough_reward_owner() -> None:
    flat_v0 = WheelbipeFlatV0Cfg()
    flat_v1 = WheelbipeFlatV1Cfg()
    rough_v0 = WheelbipeRoughV0Cfg()
    rough_v1 = WheelbipeRoughV1Cfg()

    assert flat_v0.use_absolute_height
    assert not flat_v0.height_obs_clip_enabled
    assert not rough_v0.use_absolute_height
    assert not rough_v0.height_obs_clip_enabled
    for cfg in (flat_v1, rough_v1):
        assert not cfg.use_absolute_height
        assert cfg.height_obs_clip_enabled
        assert cfg.height_obs_clip_range == pytest.approx([0.05, 0.45])


def test_v1_state_machine_reward_delta_uses_terrain_relative_height_signal() -> None:
    pytest.importorskip("mujoco")
    env = WheelbipeVariantEnv(WheelbipeFlatV1Cfg(), num_envs=1, backend_type="mujoco")
    try:
        zeros3 = np.zeros((1, 3), dtype=np.float32)
        zeros6 = np.zeros((1, 6), dtype=np.float32)
        info = {
            "commands": zeros3,
            "current_actions": zeros6,
            "last_actions": zeros6,
            "previous_actions": zeros6,
            "qacc": zeros6,
            "torques": np.zeros((1, env._num_native_actuators), dtype=np.float32),  # noqa: SLF001
            "observed_height": np.asarray([0.45], dtype=np.float32),
            "relative_observed_height": np.asarray([0.25], dtype=np.float32),
            "height_commands": np.asarray([0.30], dtype=np.float32),
            "undesired_contact": np.asarray([False]),
            "terminated": np.asarray([False]),
            "state_machine_base_quat_w": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        }
        raw = env._state_machine_source_reward_raw(  # noqa: SLF001
            info,
            zeros3,
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
            zeros6,
        )

        # (0.25 - 0.30)^2 * source height_square_sigma(10)^2.
        np.testing.assert_allclose(raw["track_height_square"], [0.25], atol=1.0e-6)
    finally:
        env.close()


def test_height_clip_contract_rejects_reversed_bounds_before_backend_creation() -> None:
    cfg = WheelbipeFlatV1Cfg(height_obs_clip_range=[0.45, 0.05])
    with pytest.raises(ValueError, match="upper bound must be >= lower bound"):
        cfg.validate()


def test_source_reward_uses_dt_scaled_termination_and_positive_wheel_power() -> None:
    zeros3 = np.zeros((2, 3), dtype=np.float32)
    zeros6 = np.zeros((2, 6), dtype=np.float32)
    torque = zeros6.copy()
    velocity = zeros6.copy()
    torque[:, 4:] = [2.0, -3.0]
    velocity[:, 4:] = [4.0, 5.0]
    reward, terms = compute_source_v14_reward(
        scales={"termination": -200.0, "wheel_power": -1.0e-4},
        ctrl_dt=0.02,
        commands=zeros3,
        linvel=zeros3,
        gyro=zeros3,
        projected_gravity=np.tile([0.0, 0.0, -1.0], (2, 1)),
        pitch=np.zeros((2,)),
        observed_height=np.full((2,), 0.3),
        height_command=np.full((2,), 0.3),
        dof_pos=zeros6,
        dof_vel=velocity,
        qacc=zeros6,
        torque=torque,
        actions=zeros6,
        last_actions=zeros6,
        previous_actions=zeros6,
        wheel_pos_b=np.zeros((2, 2, 3)),
        wheel_contact_state=np.zeros((2, 2), dtype=bool),
        undesired_contact=np.zeros((2,), dtype=bool),
        terminated=np.asarray([True, False]),
    )

    # Only +8 W is consumption; the -15 W regenerative channel is clamped.
    np.testing.assert_allclose(terms["wheel_power"], -8.0e-4 * 0.02)
    assert terms["termination"][0] == pytest.approx(-4.0)
    assert terms["termination"][1] == pytest.approx(0.0)
    np.testing.assert_allclose(reward, terms["termination"] + terms["wheel_power"])


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_normal_source_reward_reads_backend_physics_acceleration_contract(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    captured: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    source_reward = compute_source_v14_reward

    def capture_reward(**kwargs):
        reward, terms = source_reward(**kwargs)
        captured.append(
            (
                np.asarray(kwargs["qacc"]).copy(),
                {name: np.asarray(value).copy() for name, value in terms.items()},
            )
        )
        return reward, terms

    monkeypatch.setattr(joystick_module, "compute_source_v14_reward", capture_reward)
    cfg = WheelbipeFlatV0Cfg(motrix_disable_equality=backend == "motrix")
    env = WheelbipeVariantEnv(cfg, num_envs=1, backend_type=backend)
    try:
        state = env.init_state()
        backend_acc = np.asarray(env._backend.get_dof_acc())  # noqa: SLF001
        assert backend_acc.shape == np.asarray(env._backend.get_dof_vel()).shape  # noqa: SLF001
        assert np.all(backend_acc == 0.0)

        state = env.step(np.zeros((1, 6), dtype=np.float32))
        assert state.obs["obs"].shape == (1, 35)
        assert state.obs["critic"].shape == (1, 78)
        np.testing.assert_allclose(state.info["qacc"], env.get_dof_acc())
        assert np.all(np.isfinite(state.info["qacc"]))

        sentinel = np.arange(1.0, 7.0, dtype=np.float32).reshape(1, 6)
        full_acc = np.zeros_like(env._backend.get_dof_vel())  # noqa: SLF001
        full_acc[:, env._policy_vel_indices] = sentinel  # noqa: SLF001
        monkeypatch.setattr(env._backend, "get_dof_acc", lambda: full_acc)  # noqa: SLF001
        state = env.step(np.zeros((1, 6), dtype=np.float32))

        passed_qacc, terms = captured[-1]
        np.testing.assert_allclose(state.info["qacc"], sentinel)
        np.testing.assert_allclose(passed_qacc, sentinel)
        expected_leg = (
            np.sum(np.square(sentinel[:, :4]), axis=1)
            * SOURCE_V14_REWARD_SCALES["leg_joint_acc"]
            * cfg.ctrl_dt
        )
        expected_wheel = (
            np.sum(np.square(sentinel[:, 4:]), axis=1)
            * SOURCE_V14_REWARD_SCALES["wheel_acc"]
            * cfg.ctrl_dt
        )
        np.testing.assert_allclose(terms["leg_joint_acc"], expected_leg)
        np.testing.assert_allclose(terms["wheel_acc"], expected_wheel)
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_source_active_dr_contract_is_materialized_and_explicit(
    backend: str, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    cfg = WheelbipeFlatV0Cfg(motrix_disable_equality=backend == "motrix")
    with caplog.at_level(logging.WARNING):
        env = WheelbipeVariantEnv(cfg, num_envs=3, backend_type=backend)
    try:
        contract = env.domain_randomization_contract
        assert contract["interval_root_velocity_push"]["status"] == "implemented"
        assert contract["interval_external_force_torque"]["status"] == "implemented"
        assert contract["guide_material"]["requested"] is True
        assert contract["guide_material"]["status"] == "implemented_dynamic_to_sliding"
        assert len(contract["guide_material"]["resolved_body_names"]) == 16
        body_material = contract["body_material"]
        assert body_material == {
            "requested": True,
            "status": "implemented_with_explicit_conversion",
            "distribution": "uniform_bucketed",
            "make_consistent": True,
            "num_buckets": 64,
            "groups": {
                "base": {
                    "source_body_pattern": "base_link",
                    "static_range": [0.01, 0.1],
                    "dynamic_range": [0.01, 0.1],
                    "restitution_range": [0.02, 0.2],
                    "num_buckets": 64,
                },
                "wheel": {
                    "source_body_pattern": ".*_wheel_link",
                    "static_range": [0.5, 1.2],
                    "dynamic_range": [0.4, 1.0],
                    "restitution_range": [0.02, 0.2],
                    "num_buckets": 64,
                },
            },
            "backend_conversion": {
                "sliding_friction": "dynamic_to_single_sliding",
                "static_friction": "diagnostic_not_contact_law",
                "restitution": "diagnostic_not_contact_law",
            },
        }
        assert (
            contract["actuator_gain_reset"]["passive_ideal_pd_damping"]["status"]
            == "unsupported_unmaterialized_actuator"
        )
        friction_contract = contract["joint_friction"]
        assert friction_contract["status"] == "implemented_with_explicit_conversion"
        expected_viscous = "dof_damping" if backend == "mujoco" else "unsupported_omitted"
        assert friction_contract["viscous_conversion"] == expected_viscous
        assert "not_applicable_missing_target" not in caplog.text
        assert "passive IdealPD damping gain reset is not materialized" in caplog.text
        if backend == "motrix":
            assert "joint viscous friction is requested but omitted" in caplog.text

        base_friction = env._source_base_dof_frictionloss  # noqa: SLF001
        sampled_friction = env._source_dof_frictionloss  # noqa: SLF001
        assert base_friction is not None
        assert sampled_friction is not None
        assert sampled_friction.shape == (3, base_friction.size)
        for group, samples in env._source_joint_friction_samples.items():  # noqa: SLF001
            indices = samples["dof_indices"]
            np.testing.assert_allclose(
                sampled_friction[:, indices] - base_friction[indices], samples["static"]
            )
            assert np.all(samples["dynamic"] <= samples["static"])
            ranges = friction_contract["groups"][group]
            assert np.all(samples["static"] >= ranges["static_range"][0])
            assert np.all(samples["static"] <= ranges["static_range"][1])
            assert np.all(samples["viscous"] >= ranges["viscous_range"][0])
            assert np.all(samples["viscous"] <= ranges["viscous_range"][1])
        if backend == "mujoco":
            base_damping = env._source_base_dof_damping  # noqa: SLF001
            sampled_damping = env._source_dof_damping  # noqa: SLF001
            assert base_damping is not None
            assert sampled_damping is not None
            for samples in env._source_joint_friction_samples.values():  # noqa: SLF001
                indices = samples["dof_indices"]
                np.testing.assert_allclose(
                    sampled_damping[:, indices] - base_damping[indices], samples["viscous"]
                )
            for index, model in enumerate(env._backend._pool.get_all_models()):  # noqa: SLF001
                np.testing.assert_allclose(model.dof_frictionloss, sampled_friction[index])
                np.testing.assert_allclose(model.dof_damping, sampled_damping[index])
        else:
            assert env._source_dof_damping is None  # noqa: SLF001
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_him_curriculum_runtime_contract_and_zero_force_clear_both_backends(
    backend: str,
) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    cfg = WheelbipeHIMCfg(motrix_disable_equality=backend == "motrix")
    env = WheelbipeCustomEnv(cfg, num_envs=1, backend_type=backend)
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (1, 28)
        contract = env.domain_randomization_contract["him_curriculum"]
        assert contract["status"] == "implemented_with_backend_lifecycle_conversion"
        assert contract["enabled_terms"] == [
            "track_height_progression",
            "base_vertical_assist_force_progression",
        ]
        assert contract["reward_buffer_initialized"] is False
        assert "initial pre-reward reset" in contract["reset_semantics"]
        assert "consume no RNG" in contract["rng_semantics"]
        assert contract["force_interaction"] == "shared_wrench_buffer_overwrite"
        assert contract["force_composition_status"] == "source_overwrite_preserved_no_addition"
        assert "body-local orientation drift" in contract["conversion_boundary"]
        # Diagnostics and run metadata may persist this owner snapshot without
        # reaching into a backend or custom encoder.
        json.dumps(contract, sort_keys=True)
        assert contract["track_height_progression"]["effective_window_samples"] == 64 * 24
        assert contract["track_height_progression"]["effective_stage_min_compute_calls"] == [
            500 * 24,
            500 * 24,
        ]
        assert contract["base_vertical_assist_force_progression"]["force_z_stages"] == [
            160.0,
            80.0,
            0.0,
        ]
        assert contract["runtime"]["base_vertical_assist_force_progression"]["force_z"] == 160.0

        state = env.step(np.zeros((1, 6), dtype=np.float32))
        assert np.all(np.isfinite(state.reward))
        curriculum = env._source_him_curriculum  # noqa: SLF001 - runtime contract regression
        assert curriculum is not None
        curriculum._stage = 2  # noqa: SLF001 - exercise the source terminal force stage
        env._source_runtime_reward_scales = curriculum.reward_scales  # noqa: SLF001
        env._source_him_assist_latched.fill(True)  # noqa: SLF001
        env._source_body_force.fill(0.0)  # noqa: SLF001
        env._source_body_torque.fill(0.0)  # noqa: SLF001
        env._source_force_next_step.fill(np.iinfo(np.int64).max)  # noqa: SLF001

        state = env.step(np.zeros((1, 6), dtype=np.float32))
        assert np.all(np.isfinite(state.reward))
        terminal = env.domain_randomization_contract["him_curriculum"]
        assert terminal["runtime"]["base_vertical_assist_force_progression"]["force_z"] == 0.0
        if backend == "motrix":
            base_id = int(env._base_body_ids[0])  # noqa: SLF001
            applied = env._backend._applied_body_forces[base_id]  # noqa: SLF001
            np.testing.assert_array_equal(applied, np.zeros((1, 3), dtype=np.float64))
        else:
            # MuJoCo consumes and clears the upcoming-step xfrc buffer.
            assert not np.any(env._backend._pending_xfrc_applied)  # noqa: SLF001
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_him_play_owner_has_no_curriculum_force_on_both_backends(backend: str) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    cfg = WheelbipeHIMPlayCfg(motrix_disable_equality=backend == "motrix")
    env = WheelbipeCustomEnv(cfg, num_envs=1, backend_type=backend)
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (1, 28)
        assert env.source_him_assist_force_z == 0.0
        contract = env.domain_randomization_contract["him_curriculum"]
        assert contract["status"] == "disabled"
        assert contract["enabled"] is False
        assert contract["source_owner_scope"] == ("WheelbipeV14FlatHIMEnvCfg_Play.curriculum=None")
        assert contract["enabled_terms"] == []
        assert "runtime" not in contract
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_source_interval_wrench_plan_has_pinned_ranges_and_base_target(backend: str) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    env = WheelbipeVariantEnv(
        WheelbipeFlatV0Cfg(motrix_disable_equality=backend == "motrix"),
        num_envs=4,
        backend_type=backend,
    )
    try:
        env._source_push_next_step[:] = 0  # noqa: SLF001
        env._source_force_next_step[:] = 0  # noqa: SLF001
        plan = env._dr_manager._provider.build_interval_randomization_plan(  # noqa: SLF001
            env, env.step_counter
        )
        assert plan is not None
        np.testing.assert_array_equal(plan.body_ids, env._base_body_ids)  # noqa: SLF001
        assert plan.body_linear_velocity_delta is not None
        assert plan.body_force is not None
        assert plan.body_torque is not None
        assert plan.body_linear_velocity_delta.shape == (4, 1, 3)
        assert plan.body_force.shape == (4, 1, 3)
        assert plan.body_torque.shape == (4, 1, 3)
        assert np.all(np.abs(plan.body_linear_velocity_delta[..., :2]) <= 0.25)
        np.testing.assert_allclose(plan.body_linear_velocity_delta[..., 2], 0.0)
        assert np.all(np.abs(plan.body_force) <= 10.0)
        assert np.all(np.abs(plan.body_torque) <= 1.0)
        env._backend.apply_interval_randomization(plan)  # noqa: SLF001
    finally:
        env.close()


def test_source_termination_duration_bypasses_immediate_numeric_failure() -> None:
    counter = np.zeros((2,), dtype=np.int32)
    for _ in range(19):
        terminated, counter = apply_source_v14_termination_duration(
            np.asarray([True, True]),
            np.asarray([False, True]),
            counter,
            steps=20,
        )
        assert not terminated[0]
        assert terminated[1]
    terminated, counter = apply_source_v14_termination_duration(
        np.asarray([True, False]), np.asarray([False, False]), counter, steps=20
    )
    assert terminated.tolist() == [True, False]
    assert counter.tolist() == [20, 0]


def test_source_observation_outlier_gate_uses_debug_counter_and_previous_cache() -> None:
    """Match the pinned active gate and DirectRLEnv post-step ordering.

    Pinned evidence: ``wheelbipe25_v3/env.py:4091-4099`` increments
    ``_value_debug_step`` only while value diagnosis is enabled;
    ``wheelbipe_V14/env_cfg.py:769`` and ``cfg_utils.py:641`` keep it disabled.
    The outlier gate at ``wheelbipe25_v3/env.py:4971`` therefore stays inactive
    for all exact IDs. DirectRLEnv computes done before the next observation,
    so an explicitly enabled diagnostic profile consumes the committed prior
    raw-observation cache rather than the frame currently being assembled.
    """

    pytest.importorskip("mujoco")
    cfg = WheelbipeFlatV0Cfg()
    env = WheelbipeVariantEnv(cfg, num_envs=1, backend_type="mujoco")
    try:
        env.init_state()
        env.step(np.zeros((1, 6), dtype=np.float32))
        assert env._source_value_debug_step == 0  # noqa: SLF001

        safe_info: dict[str, Any] = {"base_contact": np.asarray([False])}
        safe_args = {
            "base_pos": np.asarray([[0.0, 0.0, 0.3]], dtype=np.float32),
            "base_quat": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "linvel": np.zeros((1, 3), dtype=np.float32),
            "gyro": np.zeros((1, 3), dtype=np.float32),
            "dof_pos": np.zeros((1, 6), dtype=np.float32),
            "dof_vel": np.zeros((1, 6), dtype=np.float32),
        }

        # A global step count and even a committed bad raw frame cannot open
        # the exact profile's gate while its source debug counter stays zero.
        env.step_counter = 1_000
        env._source_obs_safety_nonfinite[:] = True  # noqa: SLF001
        env._source_obs_safety_pending_nonfinite[:] = False  # noqa: SLF001
        terminated = env._compute_source_terminated(info=safe_info, **safe_args)  # noqa: SLF001
        assert not terminated[0]

        # Exercise the dormant opt-in diagnostic path without weakening the
        # exact cfg validation: current raw stats become visible on the next
        # done calculation, not the current one.
        env._cfg.debug_value_diagnosis = True  # noqa: SLF001
        env.step(np.zeros((1, 6), dtype=np.float32))
        assert env._source_value_debug_step == 1  # noqa: SLF001
        env._source_value_debug_step = 11  # noqa: SLF001
        env._source_obs_safety_nonfinite[:] = False  # noqa: SLF001
        env._source_obs_safety_pending_nonfinite[:] = True  # noqa: SLF001
        terminated = env._compute_source_terminated(info=safe_info, **safe_args)  # noqa: SLF001
        assert not terminated[0]

        env._source_obs_safety_pending_nonfinite[:] = False  # noqa: SLF001
        terminated = env._compute_source_terminated(info=safe_info, **safe_args)  # noqa: SLF001
        assert terminated[0]
        assert safe_info["numerical_safety_failure"][0]
    finally:
        env.close()


def test_source_command_curriculum_gates_disjoint_spin_and_dash_modes() -> None:
    np.random.seed(7)
    n = 20_000
    common = dict(
        num_samples=n,
        current_yaw=np.zeros((n,)),
        episode_steps=np.full((n,), 300, dtype=np.uint32),
        ctrl_dt=0.02,
        normal_low=(-2.7, 0.0, -2.0 * np.pi),
        normal_high=(2.7, 0.0, 2.0 * np.pi),
        standing_probability=0.0,
        heading_probability=0.0,
        heading_range=(-np.pi, np.pi),
        heading_stiffness=5.0,
        special_mode_min_episode_time=5.0,
        special_mode_start_iterations=(3000, 4000, 2000),
    )
    before = sample_source_v14_commands(training_iteration=1999, **common)
    assert np.all(before["special_mode_id"] == -1)

    active = sample_source_v14_commands(training_iteration=4000, **common)
    mode = active["special_mode_id"]
    commands = active["commands"]
    proportions = [np.mean(mode == index) for index in range(3)]
    np.testing.assert_allclose(proportions, [0.15, 0.15, 0.30], atol=0.015)
    assert np.all(np.abs(commands[mode == 0, 2]) >= 2.0 * np.pi)
    assert np.all(np.abs(commands[mode == 0, 2]) <= 3.25 * np.pi)
    assert np.all(np.abs(commands[mode == 1, 2]) >= 3.25 * np.pi)
    assert np.all(np.abs(commands[mode == 1, 2]) <= 4.5 * np.pi)
    assert np.all(np.abs(commands[mode == 2, 0]) >= 2.0)
    assert np.all(np.abs(commands[mode == 2, 0]) <= 3.0)


def test_source_v2_command_buckets_are_mutually_exclusive_point_one_mix() -> None:
    np.random.seed(11)
    n = 30_000
    sampled = sample_source_v14_commands(
        num_samples=n,
        current_yaw=np.zeros((n,)),
        episode_steps=np.full((n,), 300, dtype=np.uint32),
        ctrl_dt=0.02,
        training_iteration=0,
        normal_low=(-2.7, 0.0, -2.0 * np.pi),
        normal_high=(2.7, 0.0, 2.0 * np.pi),
        standing_probability=0.0,
        heading_probability=0.0,
        heading_range=(-np.pi, np.pi),
        heading_stiffness=5.0,
        special_mode_min_episode_time=5.0,
        special_mode_start_iterations=(0, 0, 0),
        special_mode_probabilities=(0.1, 0.1, 0.2),
        gimbal_mode_probability=0.2,
        gimbal_mode_start_iteration=0,
    )

    mode = sampled["special_mode_id"]
    proportions = [np.mean(mode == index) for index in range(4)]
    np.testing.assert_allclose(proportions, [0.1, 0.1, 0.2, 0.2], atol=0.012)
    assert np.all(np.abs(sampled["commands"][mode == 3, 2]) >= 2.4 * np.pi)
    assert np.all(np.abs(sampled["commands"][mode == 3, 2]) <= 3.6 * np.pi)
    assert np.all(sampled["commands"][mode == 3, :2] == 0.0)


def test_source_v1_zero_command_is_special_not_standing() -> None:
    np.random.seed(13)
    n = 30_000
    sampled = sample_source_v14_commands(
        num_samples=n,
        current_yaw=np.zeros((n,)),
        episode_steps=np.full((n,), 300, dtype=np.uint32),
        ctrl_dt=0.02,
        training_iteration=0,
        normal_low=(-2.7, 0.0, -2.0 * np.pi),
        normal_high=(2.7, 0.0, 2.0 * np.pi),
        standing_probability=0.1,
        heading_probability=0.5,
        heading_range=(-np.pi, np.pi),
        heading_stiffness=5.0,
        special_mode_min_episode_time=5.0,
        special_mode_start_iterations=(0, 0, 0),
        special_mode_probabilities=(0.15, 0.15, 0.2),
        zero_command_probability=0.1,
        zero_command_start_iteration=0,
    )

    nonstanding = ~sampled["is_standing_env"]
    mode = sampled["special_mode_id"]
    assert np.mean(~nonstanding) == pytest.approx(0.1, abs=0.012)
    proportions = [np.mean(mode[nonstanding] == index) for index in range(4)]
    np.testing.assert_allclose(proportions, [0.15, 0.15, 0.2, 0.1], atol=0.012)
    zero_mode = mode == 3
    assert np.any(zero_mode)
    assert not np.any(sampled["is_standing_env"][zero_mode])
    assert np.all(sampled["commands"][zero_mode] == 0.0)


def test_owner_command_sampler_is_config_first(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_sampler(**kwargs):
        captured.append(kwargs)
        count = int(kwargs["num_samples"])
        return {
            "commands": np.zeros((count, 3), dtype=np.float32),
            "heading_commands": np.zeros((count,), dtype=np.float32),
            "is_heading_env": np.zeros((count,), dtype=bool),
            "is_standing_env": np.zeros((count,), dtype=bool),
            "special_mode_id": np.full((count,), -1, dtype=np.int8),
        }

    monkeypatch.setattr(joystick_module, "sample_source_v14_commands", fake_sampler)
    commands = WheelbipeCommands(
        special_mode_start_iterations=[8, 9, 10],
        special_mode_probabilities=[0.11, 0.12, 0.13],
        gimbal_mode_probability=0.0,
        zero_command_probability=0.07,
        zero_command_start_iteration=6,
    )
    env = cast(Any, WheelbipeV14Env.__new__(WheelbipeV14Env))
    env._cfg = SimpleNamespace(  # noqa: SLF001
        commands=commands,
        ctrl_dt=0.02,
        # These owners must not rewrite the explicit command distribution.
        gimbal_spin_translate=SimpleNamespace(enabled=True, relative_envs=0.9),
    )
    env._state_machine = object()  # noqa: SLF001
    env.step_counter = 0

    env._sample_source_commands(  # noqa: SLF001
        current_yaw=np.zeros((2,), dtype=np.float32),
        episode_steps=np.zeros((2,), dtype=np.uint32),
    )
    assert captured[-1]["special_mode_start_iterations"] == (8, 9, 10)
    assert captured[-1]["special_mode_probabilities"] == pytest.approx((0.11, 0.12, 0.13))
    assert captured[-1]["gimbal_mode_probability"] == pytest.approx(0.0)
    assert captured[-1]["zero_command_probability"] == pytest.approx(0.07)
    assert captured[-1]["zero_command_start_iteration"] == 6

    commands.source_curriculum_enabled = False
    commands.gimbal_mode_probability = 0.2
    commands.zero_command_probability = 0.0
    env._sample_source_commands(  # noqa: SLF001
        current_yaw=np.zeros((1,), dtype=np.float32),
        episode_steps=np.zeros((1,), dtype=np.uint32),
    )
    assert captured[-1]["special_mode_start_iterations"] == (np.iinfo(np.int32).max,) * 3
    assert captured[-1]["gimbal_mode_probability"] == pytest.approx(0.2)


def test_source_material_randomization_uses_consistent_64_bucket_table() -> None:
    np.random.seed(19)
    material = sample_source_v14_material_buckets(
        num_samples=4096,
        num_slots=2,
        static_friction_range=(0.5, 1.2),
        dynamic_friction_range=(0.4, 1.0),
        restitution_range=(0.02, 0.2),
        num_buckets=64,
        make_consistent=True,
    )

    assert material.shape == (4096, 2, 3)
    assert np.unique(material.reshape(-1, 3), axis=0).shape[0] <= 64
    assert np.all(material[..., 1] <= material[..., 0])


def test_source_reset_ik_is_batched_and_left_right_symmetric() -> None:
    result = source_v14_inverse_kinematics(
        np.asarray([[0.2, 0.2], [0.25, 0.25]]),
        np.asarray([[0.1, 0.1], [-0.2, -0.2]]),
    )
    assert result.shape == (2, 12)
    np.testing.assert_allclose(result[:, 0::2], result[:, 1::2])
    assert np.all(np.isfinite(result))


def test_source_reset_ik_uses_effective_source_linkage_constants() -> None:
    """The reset pose matches the constants in the verified source run sidecar."""

    length = np.asarray([[0.2, 0.2]], dtype=np.float64)
    angle = np.asarray([[0.1, 0.1]], dtype=np.float64)
    result = source_v14_inverse_kinematics(length, angle)

    links = np.asarray([0.1134, 0.135, 0.21], dtype=np.float64)
    offsets = np.deg2rad(
        [
            -6.61,
            180.0,
            29.7,
            180.0 - 6.61 - 2.0 * 29.7,
            29.7,
            29.7,
        ]
    )
    links_sq = np.square(links)
    length_sq = np.square(length)
    triangle = np.arccos(
        np.clip(
            (links_sq[0] * length_sq + links_sq[2] * (links_sq[0] - links_sq[1]))
            / (2.0 * links[2] * links_sq[0] * length),
            -1.0,
            1.0,
        )
    )
    alpha = np.zeros((1, 2, 6), dtype=np.float64)
    alpha[..., 0] = angle + 0.5 * np.pi - triangle
    alpha[..., 1] = angle + 0.5 * np.pi + triangle
    alpha[..., 2] = np.arccos(
        np.clip(
            (links_sq[0] + links_sq[1] - links_sq[0] * length_sq / links_sq[2])
            / (2.0 * links[0] * links[1]),
            -1.0,
            1.0,
        )
    )
    alpha[..., 3] = 2.0 * np.pi - (alpha[..., 1] - alpha[..., 0]) - 2.0 * alpha[..., 2]
    alpha[..., 4] = alpha[..., 2]
    alpha[..., 5] = alpha[..., 2]
    alpha[..., 0] -= offsets[0]
    alpha[..., 1] -= offsets[1]
    alpha[..., 2] = -(alpha[..., 2] - offsets[2])
    alpha[..., 3] = -(alpha[..., 3] - offsets[3])
    alpha[..., 4] = -(alpha[..., 4] - offsets[4])
    alpha[..., 5] -= offsets[5]
    expected = alpha.transpose(0, 2, 1).reshape(1, 12)
    np.testing.assert_allclose(result, expected, atol=1.0e-12, rtol=1.0e-12)


def test_source_reset_materializes_airborne_state_and_gain_age_gate() -> None:
    pytest.importorskip("mujoco")
    cfg = WheelbipeFlatV1Cfg()
    # Flat-v1 now materializes the source ``air_1`` reset independently of
    # the task machine.  Exercise that config-first owner directly instead of
    # relying on the legacy state-machine fallback probability.
    cfg.domain_rand.predefined_air_probability = 1.0
    env = WheelbipeVariantEnv(cfg, num_envs=2, backend_type="mujoco")
    try:
        state = env.init_state()
        assert np.all(env._state_machine.airborne_state)  # noqa: SLF001
        height = env._backend.get_base_pos()[:, 2]  # noqa: SLF001
        default_height = float(cfg.init_state.pos[2])
        assert np.all((height >= default_height + 0.12) & (height <= default_height + 0.42))
        commands = np.asarray(state.info["commands"])
        assert np.all((commands[:, 0] >= -2.5) & (commands[:, 0] <= 2.5))
        assert np.all(commands[:, 1] == 0.0)
        assert np.all((commands[:, 2] >= -0.5 * np.pi) & (commands[:, 2] <= 0.5 * np.pi))
        assert np.all(
            (state.info["height_commands"] >= 0.18) & (state.info["height_commands"] <= 0.43)
        )
        assert np.all(env._source_air_command_steps_remaining == 150)  # noqa: SLF001

        immediate = env.sample_reset_source_control_randomization(
            np.asarray([0, 1], dtype=np.int32)
        )
        assert not np.any(immediate["randomized_mask"])
        env._source_control_randomization_age[:] = 720  # noqa: SLF001
        eligible = env.sample_reset_source_control_randomization(np.asarray([0, 1], dtype=np.int32))
        assert np.all(eligible["randomized_mask"])
    finally:
        env.close()


def test_source_ground_command_modifier_restores_underlying_command() -> None:
    pytest.importorskip("mujoco")
    cfg = WheelbipeFlatV1Cfg()
    env = WheelbipeVariantEnv(cfg, num_envs=1, backend_type="mujoco")
    try:
        env.init_state()
        env._source_air_command_steps_remaining[:] = 0  # noqa: SLF001
        env._source_ground_command_steps_remaining[:] = 2  # noqa: SLF001
        env._source_ground_override_command[:] = [0.5, 0.0, -0.25]  # noqa: SLF001
        env._source_ground_restore_command[:] = [-0.75, 0.0, 0.4]  # noqa: SLF001
        info = {"height_commands": np.asarray([0.3], dtype=np.float32)}
        commands = np.zeros((1, 3), dtype=np.float32)

        env._apply_source_reset_command_envelopes(  # noqa: SLF001
            commands, info, resampled=np.asarray([False])
        )
        np.testing.assert_allclose(commands, [[0.5, 0.0, -0.25]])
        env._apply_source_reset_command_envelopes(  # noqa: SLF001
            commands, info, resampled=np.asarray([False])
        )
        np.testing.assert_allclose(commands, [[-0.75, 0.0, 0.4]])
    finally:
        env.close()


def test_source_state_machine_control_mode_slot_five_is_scaled_by_five() -> None:
    pytest.importorskip("mujoco")
    cfg = WheelbipeFlatV1Cfg()
    env = WheelbipeVariantEnv(cfg, num_envs=1, backend_type="mujoco")
    try:
        state = env.init_state()
        info = dict(state.info)
        mode = np.zeros((1, 7), dtype=np.float32)
        mode[0, 5] = 0.25
        info["control_mode_obs"] = mode
        output = env._compute_obs(  # noqa: SLF001
            info,
            env.get_local_linvel(),
            env._backend.get_sensor_data(env.cfg.sensor.gyro),  # noqa: SLF001
            env.get_projected_gravity(),
            env.get_dof_pos(),
            env.get_dof_vel(),
            env_ids=None,
        )
        assert output["obs"][0, 33] == pytest.approx(1.25)
        assert output["critic"][0, 37] == pytest.approx(1.25)
    finally:
        env.close()
