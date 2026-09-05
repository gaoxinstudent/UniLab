from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.envs.locomotion.wheelbipe_v14.joystick import WheelbipeRewardConfig
from unilab.envs.locomotion.wheelbipe_v14.state_machine import (
    WheelbipeAirborneCommandResampleConfig,
    WheelbipeAirborneRewardConfig,
    WheelbipeContactSource,
    WheelbipeJumpPhase,
    WheelbipeJumpTakeoffConfig,
    WheelbipeMotionState,
    WheelbipeStairConfig,
    WheelbipeStateMachine,
    WheelbipeStateMachineConfig,
    WheelbipeStateMachineOwnerMixin,
    WheelbipeStateMachineSensors,
    WheelbipeStepUpConfig,
)
from unilab.envs.locomotion.wheelbipe_v14.task_modes import (
    WheelbipeGimbalSpinTranslateConfig,
    WheelbipeGimbalSpinTranslateOwnerMixin,
)


def _frame(
    *,
    wheel_z: float = 0.06,
    base_z: float = 0.25,
    forward_height: float = 0.0,
    stair_height: float = 0.0,
    wheel_force: float = 0.0,
    base_force: float = 0.0,
    force_history: float | None = None,
    jump_request: bool = False,
    height_command: float = 0.22,
    command: tuple[float, float, float] = (0.5, 0.0, 0.1),
    terrain_profile_id: int | None = None,
    contact_source: WheelbipeContactSource = WheelbipeContactSource.FORCE,
) -> WheelbipeStateMachineSensors:
    wheel = np.asarray([[[0.0, 0.2, wheel_z], [0.0, -0.2, wheel_z]]])
    history = None
    if force_history is not None:
        history = np.full((1, 3, 2), force_history, dtype=np.float64)
    return WheelbipeStateMachineSensors(
        wheel_pos_w=wheel,
        wheel_ground_height=np.zeros((1, 2)),
        base_pos_w=np.asarray([[0.0, 0.0, base_z]]),
        base_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        base_lin_vel_w=np.zeros((1, 3)),
        base_ground_height=np.zeros((1,)),
        wheel_forward_ground_height=np.full((1, 2), forward_height),
        wheel_stair_ground_height=np.full((1, 2), stair_height),
        commands=np.asarray([command]),
        height_commands=np.asarray([height_command]),
        jump_request=np.asarray([jump_request]),
        slope=np.zeros((1,), dtype=bool),
        contact_source=contact_source,
        wheel_contact_force_norm=(
            np.full((1, 2), wheel_force) if contact_source is WheelbipeContactSource.FORCE else None
        ),
        base_contact_force_norm=(
            np.full((1, 1), base_force) if contact_source is WheelbipeContactSource.FORCE else None
        ),
        wheel_contact_force_history_norm=history,
        terrain_profile_id=(
            None if terrain_profile_id is None else np.asarray([terrain_profile_id], dtype=np.int16)
        ),
    )


def test_force_sensor_contract_is_explicit_and_shape_checked() -> None:
    sensors = _frame()
    sensors.validate(1)
    with pytest.raises(ValueError, match="current wheel and base force"):
        replace(sensors, wheel_contact_force_norm=None).validate(1)
    with pytest.raises(ValueError, match=r"\(N, H, 2\)"):
        replace(
            sensors,
            wheel_contact_force_history_norm=np.zeros((1, 3, 2, 1)),
        ).validate(1)


def test_source_owner_builds_three_final_physics_force_samples() -> None:
    class ContactBackend:
        def __init__(self, force: np.ndarray):
            self.force = force
            self.calls: list[np.ndarray] = []

        def get_body_contact_force_norm(self, body_ids: np.ndarray) -> np.ndarray:
            self.calls.append(np.asarray(body_ids).copy())
            return self.force.copy()

    num_envs = 2
    num_bodies = 11
    wheel_columns = np.asarray([9, 10], dtype=np.intp)
    current = np.arange(num_envs * num_bodies, dtype=np.float64).reshape(num_envs, num_bodies)
    owner = object.__new__(WheelbipeStateMachineOwnerMixin)
    owner._state_machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(enabled=True, contact_history_steps=3),
        num_envs=num_envs,
    )
    owner._backend = ContactBackend(current)
    owner._num_envs = num_envs
    owner._np_dtype = np.dtype(np.float64)
    owner._source_semantics = True
    owner._state_machine_contact_body_ids = np.arange(num_bodies, dtype=np.int32)
    owner._state_machine_wheel_contact_columns = wheel_columns
    owner._state_machine_undesired_contact_columns = np.arange(9, dtype=np.intp)
    owner._state_machine_wheel_force_history = np.zeros((num_envs, 3, 2), dtype=np.float64)
    owner._source_contact_history = np.zeros((3, num_envs, num_bodies), dtype=np.float64)
    # Cursor zero means slots 2 and 1 are the two newest pre-step physics
    # frames.  The direct backend read above supplies the final post-step one.
    owner._source_contact_history[2][:, wheel_columns] = np.asarray([[31.0, 32.0], [131.0, 132.0]])
    owner._source_contact_history[1][:, wheel_columns] = np.asarray([[21.0, 22.0], [121.0, 122.0]])
    owner._source_contact_history_cursor = 0
    owner._source_wheel_contact_columns = wheel_columns

    source, wheel, undesired, history = owner._state_machine_force_frame()

    assert source is WheelbipeContactSource.FORCE
    assert len(owner._backend.calls) == 1
    np.testing.assert_array_equal(owner._backend.calls[0], np.arange(num_bodies))
    np.testing.assert_array_equal(wheel, current[:, wheel_columns])
    np.testing.assert_array_equal(undesired, current[:, :9])
    np.testing.assert_array_equal(
        history,
        np.asarray(
            [
                [[9.0, 10.0], [31.0, 32.0], [21.0, 22.0]],
                [[20.0, 21.0], [131.0, 132.0], [121.0, 122.0]],
            ]
        ),
    )


def test_force_stair_requires_configured_history_window() -> None:
    machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(
            enabled=True,
            airborne_enabled=False,
            contact_history_steps=3,
            stair=WheelbipeStairConfig(enabled=True),
        ),
        num_envs=1,
    )
    with pytest.raises(ValueError, match="enabled Stair machine requires"):
        machine.update(_frame(force_history=None))
    with pytest.raises(ValueError, match="history length must match"):
        machine.update(
            replace(
                _frame(force_history=10.0),
                wheel_contact_force_history_norm=np.zeros((1, 2, 2)),
            )
        )


def test_gimbal_spin_mode_uses_yaw_frame_target_and_fixed_policy_tail() -> None:
    class Backend:
        def get_dof_pos(self) -> np.ndarray:
            return np.asarray([[np.pi / 2.0, 0.0], [0.0, 0.0]])

    owner = object.__new__(WheelbipeGimbalSpinTranslateOwnerMixin)
    owner._cfg = SimpleNamespace(
        ctrl_dt=0.02,
        commands=SimpleNamespace(resampling_time=5.0),
        gimbal_spin_translate=WheelbipeGimbalSpinTranslateConfig(
            enabled=True,
            # Assignment is authoritative: task_modes must not perform a
            # second probability draw from this metadata field.
            relative_envs=0.0,
            min_episode_time_s=0.0,
            speed_ranges=((0.6, 0.6),),
            heading_range=(0.0, 0.0),
            height_range=(0.3, 0.3),
            project_to_body_command=False,
        ),
    )
    owner._num_envs = 2
    owner._np_dtype = np.dtype(np.float64)
    owner._gimbal_enabled = True
    owner._gimbal_pos_indices = np.asarray([0, 1], dtype=np.intp)
    owner._backend = Backend()
    owner._gimbal_spin_active = np.zeros((2,), dtype=bool)
    owner._gimbal_spin_last_command_generation = np.full((2,), -1, dtype=np.int64)
    owner._gimbal_spin_velocity_yaw = np.zeros((2, 2), dtype=np.float64)
    owner._gimbal_spin_heading = np.zeros((2,), dtype=np.float64)
    owner._gimbal_spin_height = np.zeros((2,), dtype=np.float64)
    owner._gimbal_spin_rng = np.random.default_rng(7)
    info = {
        "commands": np.asarray([[0.0, 0.0, 3.25], [0.1, 0.0, 7.0]]),
        "height_commands": np.asarray([0.2, 0.2]),
        "steps": np.zeros((2,), dtype=np.uint32),
        "special_mode_id": np.asarray([3, 0], dtype=np.int8),
        "command_resample_generation": np.asarray([4, 9], dtype=np.int64),
    }

    owner._apply_gimbal_spin_mode(info)

    np.testing.assert_array_equal(owner._gimbal_spin_active, [True, False])
    np.testing.assert_allclose(info["commands"][0], [0.0, 0.0, 3.25])
    np.testing.assert_allclose(info["commands"][1], [0.1, 0.0, 7.0])
    # Source ``_apply_gimbal_spin_translate_command`` returns after zeroing
    # XY when projection is disabled; the ordinary height command is not
    # replaced by the separately sampled gimbal-spin height.
    assert info["height_commands"][0] == pytest.approx(0.2)
    np.testing.assert_allclose(
        info["control_mode_obs"][0],
        [0.0, 1.0, 0.6, 0.0, 1.0, 1.0, 0.0],
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(info["control_mode_obs"][1], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # A new source command generation resamples the gimbal-frame target even
    # when two consecutive categorical draws both select reserved id 3.
    previous = owner._gimbal_spin_velocity_yaw.copy()
    info["command_resample_generation"][0] += 1
    owner._apply_gimbal_spin_mode(info)
    assert owner._gimbal_spin_last_command_generation[0] == 5
    assert owner._gimbal_spin_velocity_yaw[0, 0] == pytest.approx(previous[0, 0])


@pytest.mark.parametrize("special_mode_id", [-1, 0, 1, 2])
def test_gimbal_spin_mode_only_consumes_reserved_categorical_assignment(
    special_mode_id: int,
) -> None:
    class Backend:
        def get_dof_pos(self) -> np.ndarray:
            return np.zeros((1, 2))

    owner = object.__new__(WheelbipeGimbalSpinTranslateOwnerMixin)
    owner._cfg = SimpleNamespace(
        ctrl_dt=0.02,
        commands=SimpleNamespace(resampling_time=5.0),
        gimbal_spin_translate=WheelbipeGimbalSpinTranslateConfig(
            enabled=True,
            relative_envs=1.0,
            min_episode_time_s=0.0,
        ),
    )
    owner._num_envs = 1
    owner._np_dtype = np.dtype(np.float64)
    owner._gimbal_enabled = True
    owner._gimbal_pos_indices = np.asarray([0, 1], dtype=np.intp)
    owner._backend = Backend()
    owner._gimbal_spin_active = np.ones((1,), dtype=bool)
    owner._gimbal_spin_last_command_generation = np.zeros((1,), dtype=np.int64)
    owner._gimbal_spin_velocity_yaw = np.ones((1, 2), dtype=np.float64)
    owner._gimbal_spin_heading = np.ones((1,), dtype=np.float64)
    owner._gimbal_spin_height = np.ones((1,), dtype=np.float64)
    owner._gimbal_spin_rng = np.random.default_rng(3)
    info = {
        "commands": np.asarray([[1.0, 0.0, 2.0]]),
        "height_commands": np.asarray([0.2]),
        "special_mode_id": np.asarray([special_mode_id], dtype=np.int8),
        "command_resample_generation": np.asarray([1], dtype=np.int64),
    }

    owner._apply_gimbal_spin_mode(info)

    assert not owner._gimbal_spin_active[0]
    np.testing.assert_array_equal(info["commands"], [[1.0, 0.0, 2.0]])
    np.testing.assert_array_equal(info["control_mode_obs"], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def test_gimbal_mode_cooperative_command_and_reset_hooks_dispatch_once() -> None:
    class Parent:
        def _update_commands(self, info: dict[str, object]) -> None:
            self.update_calls += 1
            info["parent_updated"] = True

        def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
            self.reset_calls += 1
            return {"obs": np.ones((env_indices.size, 1))}, {"parent_reset": True}

    class Owner(WheelbipeGimbalSpinTranslateOwnerMixin, Parent):
        pass

    owner = object.__new__(Owner)
    owner._cfg = SimpleNamespace(
        gimbal_spin_translate=WheelbipeGimbalSpinTranslateConfig(enabled=False)
    )
    owner._num_envs = 2
    owner._np_dtype = np.dtype(np.float64)
    owner._gimbal_spin_active = np.ones((2,), dtype=bool)
    owner._gimbal_spin_last_command_generation = np.ones((2,), dtype=np.int64)
    owner._gimbal_spin_velocity_yaw = np.ones((2, 2), dtype=np.float64)
    owner._gimbal_spin_heading = np.ones((2,), dtype=np.float64)
    owner._gimbal_spin_height = np.ones((2,), dtype=np.float64)
    owner.update_calls = 0
    owner.reset_calls = 0

    info: dict[str, object] = {}
    owner._update_commands(info)
    observations, reset_info = owner.reset(np.asarray([1], dtype=np.intp))

    assert owner.update_calls == 1
    assert owner.reset_calls == 1
    assert info["parent_updated"] is True
    assert reset_info["parent_reset"] is True
    np.testing.assert_array_equal(observations["obs"], [[1.0]])
    assert owner._gimbal_spin_active.tolist() == [True, False]
    assert owner._gimbal_spin_last_command_generation.tolist() == [1, -1]


def test_gimbal_mode_tail_has_priority_when_state_machine_coexists() -> None:
    class StateParent:
        def _update_state_machine(self, info: dict[str, np.ndarray]) -> None:
            info["control_mode_obs"] = np.asarray(
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )

    class ModeOwner(WheelbipeGimbalSpinTranslateOwnerMixin, StateParent):
        pass

    owner = object.__new__(ModeOwner)
    owner._cfg = SimpleNamespace(
        gimbal_spin_translate=WheelbipeGimbalSpinTranslateConfig(enabled=True)
    )
    owner._num_envs = 2
    owner._np_dtype = np.dtype(np.float64)
    owner._gimbal_spin_active = np.asarray([True, False])
    gimbal_mode = np.asarray(
        [
            [0.0, 1.0, 0.6, 0.0, 1.0, 0.5, 0.8],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    info = {"control_mode_obs": gimbal_mode.copy()}

    owner._update_state_machine(info)

    np.testing.assert_array_equal(info["control_mode_obs"][0], gimbal_mode[0])
    np.testing.assert_array_equal(info["control_mode_obs"][1], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_gimbal_reward_replaces_all_source_xy_terms() -> None:
    class RewardParent:
        def _compute_reward(self, *args: object, **kwargs: object) -> np.ndarray:
            del args, kwargs
            return self.parent_reward.copy()

    class RewardOwner(WheelbipeGimbalSpinTranslateOwnerMixin, RewardParent):
        pass

    class Backend:
        def get_dof_pos(self) -> np.ndarray:
            return np.zeros((1, 2), dtype=np.float64)

        def get_base_quat(self) -> np.ndarray:
            return np.asarray([[1.0, 0.0, 0.0, 0.0]])

    cfg = WheelbipeGimbalSpinTranslateConfig(
        enabled=True,
        lin_vel_yaw_scale=2.0,
        lin_speed_scale=3.0,
        lin_heading_scale=4.0,
    )
    owner = object.__new__(RewardOwner)
    owner._cfg = SimpleNamespace(ctrl_dt=0.02, gimbal_spin_translate=cfg)
    owner._reward_cfg = SimpleNamespace(
        scales={
            "track_lin_vel_xy": 2.0,
            "track_lin_vel_xy_tight": 3.0,
            "track_lin_vel_xy_square": 4.0,
            "stand_still_lin_vel": 5.0,
        },
        stand_still_deadzone=0.1,
        lin_vel_error_constraint=1.0,
        lin_vel_sigma=1.0,
        lin_vel_tight_sigma=2.0,
        lin_vel_square_sigma=2.0,
        tracking_sigma=1.0,
    )
    owner._source_semantics = True
    owner._num_envs = 1
    owner._np_dtype = np.dtype(np.float64)
    owner._gimbal_enabled = True
    owner._gimbal_pos_indices = np.asarray([0, 1], dtype=np.intp)
    owner._backend = Backend()
    owner._gimbal_spin_active = np.asarray([True])
    owner._gimbal_spin_velocity_yaw = np.asarray([[0.5, 0.0]])
    owner._gimbal_spin_heading = np.zeros((1,))
    owner._gimbal_spin_height = np.zeros((1,))
    velocity = np.asarray([[0.5, 0.0, 0.0]])
    error = -0.5
    suppressed = (
        2.0 * np.exp(-(error**2) / 1.0)
        + 3.0 * np.exp(-(error**2) / 2.0)
        + 4.0 * (error * 2.0) ** 2
        + 5.0 * 0.5
    )
    owner.parent_reward = np.asarray([100.0 + suppressed * 0.02])
    info = {
        "commands": np.asarray([[0.0, 0.0, 3.0]]),
        "numerical_safety_failure": np.asarray([False]),
    }

    reward = owner._compute_reward(
        info,
        velocity,
        np.zeros((1, 3)),
        np.asarray([[0.0, 0.0, -1.0]]),
        np.zeros((1, 6)),
        np.zeros((1, 6)),
    )

    # Perfect yaw-frame velocity matches all three configured positive custom
    # terms: 2 + 3 + 4, with zero square/stand-still penalties.
    np.testing.assert_allclose(reward, [100.0 + 9.0 * 0.02])


def test_airborne_uses_force_and_height_then_holds_landing() -> None:
    machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(
            enabled=True,
            airborne_enter_steps=1,
            landing_contact_steps=1,
            landing_hold_steps=1,
            max_airborne_steps=20,
            wheel_contact_force_threshold=20.0,
            command_scale_airborne=1.0,
            command_scale_landing=1.0,
        ),
        num_envs=1,
    )
    transition = machine.update(_frame(wheel_z=0.40, base_z=0.45))
    assert transition["state"][0] == WheelbipeMotionState.AIRBORNE
    # Low wheel geometry alone cannot start landing in an explicit force frame.
    transition = machine.update(_frame(wheel_z=0.06, base_z=0.30, wheel_force=0.0))
    assert transition["state"][0] == WheelbipeMotionState.AIRBORNE
    transition = machine.update(_frame(wheel_z=0.06, base_z=0.30, wheel_force=25.0))
    assert transition["state"][0] == WheelbipeMotionState.LANDING
    assert transition["contact_source"][0] == WheelbipeContactSource.FORCE
    transition = machine.update(_frame(wheel_z=0.06, base_z=0.30, wheel_force=25.0))
    assert transition["state"][0] == WheelbipeMotionState.NORMAL


def test_airborne_max_duration_is_plain_exit_and_base_timer_freezes() -> None:
    max_machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(
            enabled=True,
            airborne_enter_steps=1,
            max_airborne_steps=2,
            landing_contact_steps=10,
            base_contact_steps=10,
        ),
        num_envs=1,
    )
    assert max_machine.update(_frame(wheel_z=0.40, base_z=0.45))["airborne"][0]
    transition = max_machine.update(_frame(wheel_z=0.40, base_z=0.45))
    assert transition["state"][0] == WheelbipeMotionState.NORMAL
    assert not transition["failure"][0]

    base_machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(
            enabled=True,
            airborne_enter_steps=1,
            max_airborne_steps=20,
            landing_contact_steps=10,
            base_contact_steps=2,
            base_contact_force_threshold=5.0,
        ),
        num_envs=1,
    )
    base_machine.update(_frame(wheel_z=0.40, base_z=0.45))
    assert base_machine.update(_frame(base_force=6.0))["airborne"][0]
    assert base_machine.update(_frame(base_force=0.0))["airborne"][0]
    transition = base_machine.update(_frame(base_force=6.0))
    assert transition["state"][0] == WheelbipeMotionState.NORMAL


def test_airborne_terrain_command_override_is_entry_sampled_and_persistent() -> None:
    machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(
            enabled=True,
            airborne_enter_steps=1,
            max_airborne_steps=100,
            command_scale_airborne=1.0,
            airborne_command_resample=WheelbipeAirborneCommandResampleConfig(
                enabled=True,
                probability=1.0,
                terrain_names=("low_speed_stair_for_rm",),
                lin_vel_x_range=(-1.5, 1.5),
                lin_vel_y_range=(0.0, 0.0),
                ang_vel_z_range=(-1.0, 1.0),
            ),
        ),
        num_envs=1,
    )
    machine._rng = np.random.default_rng(17)
    enter = _frame(
        wheel_z=0.40,
        base_z=0.45,
        command=(0.5, 0.0, 0.0),
        terrain_profile_id=0,
    )
    transition = machine.update(enter)
    assert transition["airborne"][0]
    first, _ = machine.apply_command_overrides(enter.commands, enter.height_commands)
    assert 0.0 <= first[0, 0] <= 1.5
    assert first[0, 1] == 0.0
    assert -1.0 <= first[0, 2] <= 1.0

    later = replace(enter, commands=np.asarray([[-0.5, 0.0, 0.0]]))
    machine.update(later)
    persistent, _ = machine.apply_command_overrides(later.commands, later.height_commands)
    np.testing.assert_array_equal(persistent, first)

    machine.reset(np.asarray([0], dtype=np.intp))
    assert not machine.airborne_command_override_active[0]


def test_airborne_reward_applies_global_overrides_and_state_multipliers() -> None:
    class RewardParent:
        def _compute_reward(self, *args: object, **kwargs: object) -> np.ndarray:
            del args, kwargs
            return self.parent_reward.copy()

    class RewardOwner(WheelbipeStateMachineOwnerMixin, RewardParent):
        pass

    class Backend:
        def get_dof_pos(self) -> np.ndarray:
            return np.zeros((2, 2), dtype=np.float64)

    reward_cfg = WheelbipeAirborneRewardConfig(
        enabled=True,
        base_scale_overrides={"action_rate": -0.002},
        airborne_scale_multipliers={"termination": 6.0, "undesired_contact": 25.0},
    )
    machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(enabled=True, airborne_reward=reward_cfg),
        num_envs=2,
    )
    machine.airborne_state[:] = [True, False]
    owner = object.__new__(RewardOwner)
    owner._state_machine = machine
    owner._backend = Backend()
    owner._num_envs = 2
    owner._np_dtype = np.dtype(np.float64)
    owner._native_leg_indices = np.arange(4, dtype=np.intp)
    owner._native_wheel_indices = np.arange(4, 6, dtype=np.intp)
    owner._state_machine_rear2_pos_indices = np.zeros((0,), dtype=np.intp)
    owner._reward_cfg = WheelbipeRewardConfig(
        scales={"action_rate": -0.01, "termination": -200.0, "undesired_contact": -2.0}
    )
    # Parent reward is the ordinary source graph for raw action_rate=6,
    # termination=1 and undesired_contact=1.
    owner.parent_reward = np.full((2,), (6.0 * -0.01 - 200.0 - 2.0) * 0.02)
    zeros3 = np.zeros((2, 3), dtype=np.float64)
    zeros6 = np.zeros((2, 6), dtype=np.float64)
    info = {
        "commands": zeros3.copy(),
        "height_commands": np.full((2,), 0.2),
        "observed_height": np.full((2,), 0.2),
        "current_actions": np.ones((2, 6)),
        "last_actions": zeros6.copy(),
        "previous_actions": zeros6.copy(),
        "qacc": zeros6.copy(),
        "torques": zeros6.copy(),
        "undesired_contact": np.ones((2,), dtype=bool),
        "terminated": np.ones((2,), dtype=bool),
        "state_machine_base_pos_w": zeros3.copy(),
        "state_machine_base_quat_w": np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        "state_machine_wheel_pos_w": np.zeros((2, 2, 3)),
        "numerical_safety_failure": np.zeros((2,), dtype=bool),
    }

    reward = owner._compute_reward(info, zeros3, zeros3, zeros3, zeros6, zeros6)

    # Global action target is -0.002 for both rows.  Airborne then applies
    # termination x6 and undesired-contact x25; the normal row is unchanged.
    np.testing.assert_allclose(reward, [-25.00024, -4.04024], atol=1.0e-10)


def test_airborne_reward_additions_follow_contact_timer_windows() -> None:
    class Backend:
        def get_dof_pos(self) -> np.ndarray:
            return np.asarray([[0.0, 0.0]])

    cfg = WheelbipeAirborneRewardConfig(enabled=True)
    machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(enabled=True, airborne_reward=cfg), num_envs=1
    )
    machine.airborne_state[0] = True
    owner = object.__new__(WheelbipeStateMachineOwnerMixin)
    owner._state_machine = machine
    owner._backend = Backend()
    owner._np_dtype = np.dtype(np.float64)
    owner._num_envs = 1
    owner._native_wheel_indices = np.asarray([4, 5], dtype=np.intp)
    owner._state_machine_rear2_pos_indices = np.asarray([0, 1], dtype=np.intp)
    info = {
        "commands": np.asarray([[1.5, 0.0, 0.0]]),
        "torques": np.zeros((1, 6)),
        "state_machine_base_pos_w": np.asarray([[0.0, 0.0, 0.3]]),
        "state_machine_base_quat_w": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        "state_machine_wheel_pos_w": np.asarray([[[0.0, 0.2, 0.1], [0.0, -0.2, 0.1]]]),
    }
    linvel = np.asarray([[1.5, 0.0, 0.0]])

    before = owner._state_machine_airborne_additions(
        info, linvel, np.asarray([[0.0, 0.0, 0.0, 0.0, 10.0, 10.0]])
    )
    assert before["airborne_wheel_heading_x_centering"][0] == pytest.approx(1.0)
    assert before["airborne_air_wheel_zero_torque_exp"][0] == pytest.approx(1.0)
    assert before["airborne_precontact_wheel_directional_speed"][0] == pytest.approx(1.0)
    assert before["airborne_precontact_wheel_directional_speed_shortfall"][0] == 0.0
    assert before["airborne_joint_pos_limits"][0] > 0.0

    machine.wheel_contact_steps[0, 0] = 1
    after = owner._state_machine_airborne_additions(
        info, linvel, np.asarray([[0.0, 0.0, 0.0, 0.0, 5.0, 5.0]])
    )
    assert after["airborne_wheel_heading_x_centering"][0] == 0.0
    assert after["airborne_air_wheel_zero_torque_exp"][0] == 0.0
    assert after["airborne_precontact_wheel_directional_speed"][0] == 0.0
    assert after["airborne_precontact_wheel_directional_speed_shortfall"][0] == pytest.approx(0.5)


def test_jump_takeoff_push_tuck_reference_preserves_commands() -> None:
    machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(
            enabled=True,
            control_dt=0.02,
            airborne_enabled=False,
            jump_takeoff=WheelbipeJumpTakeoffConfig(
                enabled=True,
                peak_height_range=(0.50, 0.50),
                tuck_timing_mode="fixed_tuck_time",
                fixed_tuck_time_s=0.04,
                min_duration_s=0.12,
                enter_airborne_on_exit=False,
            ),
        ),
        num_envs=1,
    )
    sensors = _frame(wheel_z=0.06, base_z=0.20, jump_request=True)
    transition = machine.update(sensors)
    assert transition["jump_phase"][0] == WheelbipeJumpPhase.PUSH
    assert transition["jump_trigger_event"][0]
    mode = machine.control_mode_obs(state_dtype=np.float32)
    assert mode[0, 4] == 1.0
    assert mode[0, 5] == pytest.approx(0.50)
    commands, heights = machine.apply_command_overrides(sensors.commands, sensors.height_commands)
    np.testing.assert_array_equal(commands, sensors.commands)
    np.testing.assert_array_equal(heights, sensors.height_commands)
    transition = machine.update(replace(sensors, jump_request=np.asarray([False])))
    transition = machine.update(replace(sensors, jump_request=np.asarray([False])))
    assert transition["jump_phase"][0] == WheelbipeJumpPhase.TUCK
    assert np.isfinite(transition["jump_ref_vel_z"][0])


def test_step_up_temporal_probe_holds_height_and_wall_times_out() -> None:
    machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(
            enabled=True,
            control_dt=0.02,
            airborne_enabled=False,
            step_up=WheelbipeStepUpConfig(
                enabled=True,
                step_height_min=0.10,
                step_height_max=0.20,
                wall_height=0.25,
                height_command_bias=0.10,
                hold_s=0.04,
                height_command_max=0.40,
            ),
        ),
        num_envs=1,
    )
    machine.update(_frame(forward_height=0.0))
    transition = machine.update(_frame(forward_height=0.15))
    assert transition["step_detect_event"][0]
    assert transition["state"][0] == WheelbipeMotionState.STEP_UP
    _, height = machine.apply_command_overrides(np.zeros((1, 3)), np.asarray([0.22]))
    assert height[0] == pytest.approx(0.32)
    transition = machine.update(_frame(forward_height=0.50))
    assert transition["wall_blocked"][0]
    assert transition["timeout"][0]
    assert transition["state"][0] == WheelbipeMotionState.WALL_BLOCKED


def test_stair_uses_three_sample_force_peak_and_success_window() -> None:
    machine = WheelbipeStateMachine(
        WheelbipeStateMachineConfig(
            enabled=True,
            control_dt=0.02,
            airborne_enabled=False,
            stair=WheelbipeStairConfig(
                enabled=True,
                step_height_min=0.10,
                step_height_max=0.20,
                height_command_range=(0.30, 0.30),
                contact_force_threshold=5.0,
                success_duration_s=0.04,
                timeout_s=1.0,
            ),
        ),
        num_envs=1,
    )
    machine.update(_frame(stair_height=0.0, force_history=10.0))
    transition = machine.update(_frame(stair_height=0.15, base_z=0.45, force_history=10.0))
    assert transition["stair_detect_event"][0]
    assert transition["state"][0] == WheelbipeMotionState.STAIR
    mode = machine.control_mode_obs(state_dtype=np.float32)
    np.testing.assert_array_equal(mode[0, :5], np.asarray([0, 1, 0, 0, 0]))
    transition = machine.update(_frame(stair_height=0.15, base_z=0.45, force_history=10.0))
    assert transition["stair_success_event"][0]
    assert not transition["failure"][0]
    assert transition["state"][0] == WheelbipeMotionState.NORMAL
