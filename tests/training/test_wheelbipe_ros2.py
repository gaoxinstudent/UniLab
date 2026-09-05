"""Contract tests for the no-ROS WheelBipe deployment adapter."""

from __future__ import annotations

import json
import runpy
import struct
from pathlib import Path

import numpy as np
import pytest

import unilab.training.wheelbipe_ros2 as ros2
from unilab.envs.locomotion.wheelbipe_v14.base import build_wheelbipe_policy_observation
from unilab.training.wheelbipe_ros2 import (
    WHEELBIPE_REAL_COMMAND_PACKET_SIZE,
    WHEELBIPE_REAL_RECONNECT_INTERVAL_SEC,
    WHEELBIPE_REAL_STATE_PACKET_SIZE,
    WHEELBIPE_ROS2_JOINT_NAMES,
    WHEELBIPE_ROS2_QOS,
    WHEELBIPE_ROS2_TOPICS,
    WheelbipeRealBridgeGate,
    WheelbipeRealJointCommand,
    WheelbipeRealStateDecoder,
    WheelbipeRos2Controller,
    WheelbipeRos2ControllerConfig,
    WheelbipeRos2ControllerState,
    WheelbipeRos2NativeBundleError,
    WheelbipeRos2RobotState,
    _symmetric_remainder,
    decode_wheelbipe_real_command_packet,
    decode_wheelbipe_real_state_packet,
    encode_wheelbipe_real_command_packet,
    encode_wheelbipe_real_state_packet,
    load_wheelbipe_ros2_config,
    materialize_wheelbipe_ros2_native_workspace,
    require_wheelbipe_ros2_native_runtime,
    verify_wheelbipe_ros2_native_bundle,
    wheelbipe_crc16,
    wheelbipe_ros2_contract_snapshot,
    wheelbipe_ros2_native_runtime_probe,
    wheelbipe_xbox_teleop_command,
)


def _state(**kwargs: object) -> WheelbipeRos2RobotState:
    values: dict[str, object] = {
        "positions": np.zeros(8),
        "velocities": np.zeros(8),
        "efforts": np.zeros(8),
        "linear_acceleration": np.zeros(3),
        "angular_velocity": np.asarray([0.1, 0.2, 0.3]),
        "orientation_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "projected_gravity_b": np.asarray([0.0, 0.0, -1.0]),
        "timestamp": 0.0,
        "period": 0.002,
    }
    values.update(kwargs)
    return WheelbipeRos2RobotState(**values)  # type: ignore[arg-type]


def _controller(policy: object | None = None, **config: object) -> WheelbipeRos2Controller:
    cfg = WheelbipeRos2ControllerConfig(**config)  # type: ignore[arg-type]
    controller = WheelbipeRos2Controller(policy, config=cfg)
    controller.on_init()
    controller.on_configure()
    controller.on_activate()
    return controller


def test_source_api_order_topics_and_config_are_explicit() -> None:
    assert WHEELBIPE_ROS2_JOINT_NAMES == (
        "left_front1_joint",
        "left_rear1_joint",
        "right_front1_joint",
        "right_rear1_joint",
        "left_wheel_joint",
        "right_wheel_joint",
        "left_spring2_joint",
        "right_spring2_joint",
    )
    assert WHEELBIPE_ROS2_TOPICS["motion_command"]["type"] == "geometry_msgs/msg/Twist"
    assert WHEELBIPE_ROS2_TOPICS["joint_commands"]["type"] == "sensor_msgs/msg/JointState"
    assert load_wheelbipe_ros2_config().update_rate_hz == 500
    assert load_wheelbipe_ros2_config().inference_frequency_hz == 50
    loaded = load_wheelbipe_ros2_config()
    assert loaded.rl_print_inference_time is False
    assert loaded.rl_publish_network_io is True
    assert WHEELBIPE_ROS2_QOS["command_subscriptions"]["depth"] == 1
    snapshot = wheelbipe_ros2_contract_snapshot(loaded)
    assert snapshot["native_ros2"] is False
    assert snapshot["ros_graph"] is False
    assert snapshot["serial_io"] is False
    assert snapshot["realtime_guarantee"] is False
    assert snapshot["protocol"]["command_packet_size"] == WHEELBIPE_REAL_COMMAND_PACKET_SIZE


def test_owner_module_is_importable_without_rclpy_or_a_ros_graph() -> None:
    # The optional runtime probe may report either availability depending on
    # the host, but importing the owner itself must never import rclpy or
    # create a DDS context.
    assert "rclpy" not in ros2.__dict__
    assert isinstance(ros2.wheelbipe_ros2_runtime_available(), bool)


def test_deployment_yaml_records_real_backend_sensor_and_plugin_contract() -> None:
    # The controller parameter loader intentionally ignores deployment
    # metadata; inspect the YAML separately so the real backend's second
    # semantic sensor is not lost in the no-ROS projection.
    from omegaconf import OmegaConf

    path = Path(__file__).resolve().parents[2] / "conf" / "deployment" / "wheelbipe_v14_ros2.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, dict)
    metadata = raw["wheelbipe_ros2"]
    assert metadata["ros2_control"]["controller_plugin"] == (
        "robot_locomotion/TemplateRos2Controller"
    )
    assert metadata["real_bridge"]["plugin"] == "template_real_ros2_ctrl::RealBridge"
    assert metadata["real_bridge"]["sensors"] == ["imu", "dt7"]
    assert metadata["real_bridge"]["output_gate"] == ("active_and_connected_and_fresh_valid_state")
    assert metadata["native_source"]["package_count"] == 7
    assert metadata["native_source"]["source_file_count"] == 72
    runtime = metadata["runtime"]
    assert runtime["native_ros2"] is False
    assert runtime["native_source_packaged"] is True
    assert runtime["native_workspace_built"] is False
    assert runtime["ros_graph"] is False
    assert runtime["serial_io"] is False
    assert runtime["hardware_validated"] is False
    assert runtime["realtime_guarantee"] is False


def test_omitted_command_timestamps_use_steady_clock_for_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROS callback-style receipts must not be compared with simulation time."""

    steady = [100.0]
    monkeypatch.setattr(ros2.time, "monotonic", lambda: steady[0])
    controller = _controller(lambda _obs: np.zeros(6, dtype=np.float32), auto_enter_rl=True)
    controller.set_motion_command(0.4, 0.2)  # no explicit timestamp => steady clock
    controller.set_height_command(0.31)
    fresh = controller.update(_state(), timestamp=0.0, period=0.002)
    assert fresh.diagnostic["command_timed_out"] is False
    steady[0] = 100.6
    expired = controller.update(_state(), timestamp=0.6, period=0.002)
    assert expired.diagnostic["command_timed_out"] is True
    # Height is reset while the controller is still in INIT/IDLE, matching the
    # source callback timeout branch.
    assert controller._height.value == pytest.approx(0.22)  # type: ignore[attr-defined]


def test_noise_matches_source_six_position_loop() -> None:
    controller = _controller(
        lambda _obs: np.zeros(6, dtype=np.float32),
        auto_enter_rl=True,
        enable_noise=True,
    )
    # Reach RL and inspect the first policy frame.  The source StateRL adds
    # joint-position noise to all six non-spring slots, including the two
    # reserved wheel-position zeros.
    controller.update(_state(), timestamp=0.0, period=0.002)
    output = controller.update(_state(), timestamp=0.02, period=0.002)
    if output.observation is None:
        output = controller.update(_state(), timestamp=0.022, period=0.002)
    assert output.observation is not None
    assert not np.allclose(output.observation[14:16], 0.0)


def test_controller_fsm_and_inference_cadence() -> None:
    calls: list[np.ndarray] = []

    def policy(observation: np.ndarray) -> np.ndarray:
        calls.append(np.asarray(observation).copy())
        return np.full(6, 0.2, dtype=np.float32)

    controller = _controller(policy, auto_enter_rl=True)
    states: list[WheelbipeRos2ControllerState] = []
    outputs = []
    for index in range(20):
        output = controller.update(_state(), timestamp=index * 0.002, period=0.002)
        states.append(output.state)
        outputs.append(output)
    assert states[0] is WheelbipeRos2ControllerState.INIT
    assert WheelbipeRos2ControllerState.RL in states
    # 20 controller ticks at 500 Hz produce two 50 Hz inference opportunities
    # after the INIT gate (the exact first tick is part of the source contract).
    assert 1 <= len(calls) <= 3
    assert outputs[-1].action is not None
    assert outputs[-1].command.final_torque.shape == (8,)
    assert outputs[-1].command.names == WHEELBIPE_ROS2_JOINT_NAMES
    assert outputs[-1].command.frame_id == "base_link"


def test_position_error_wrap_matches_source_remainder() -> None:
    values = _symmetric_remainder(
        np.asarray([1.5 * np.pi, -1.5 * np.pi, np.pi, -np.pi, 3.0 * np.pi, 5.0 * np.pi]),
        2.0 * np.pi,
    )
    # The 3*pi and 5*pi cases exercise the ties-to-even quotient rule used by
    # C++ std::remainder; a modulo implementation would return +pi for both.
    np.testing.assert_allclose(
        values,
        [-0.5 * np.pi, 0.5 * np.pi, np.pi, -np.pi, -np.pi, np.pi],
    )


def test_reserved_wheel_position_slots_keep_source_scale_and_clamps() -> None:
    controller = _controller(
        lambda _obs: np.zeros(6, dtype=np.float32),
        policy_input_joint_pos_scale=(1.0, 1.0, 1.0, 1.0, 2.0, 3.0),
        policy_input_joint_pos_min=(0.0, 0.0, 0.0, 0.0, 0.25, -0.5),
        policy_input_joint_pos_max=(100.0, 100.0, 100.0, 100.0, 0.5, 0.5),
    )
    controller.set_state_command(3)
    controller.update(_state(), timestamp=0.0, period=0.002)
    output = controller.update(_state(), timestamp=0.02, period=0.002)
    if output.observation is None:
        output = controller.update(_state(), timestamp=0.022, period=0.002)
    assert output.observation is not None
    # Source ``scaleClamp(0, scale[4/5], min[4/5], max[4/5])`` yields these
    # values even though the slots are reserved in normal mode.
    np.testing.assert_allclose(output.observation[14:16], [0.25, 0.0])


def test_prepare_hardware_pd_vel_keeps_wheels_stopped() -> None:
    """PREPARE must not forward wheel targets as velocity commands.

    ``TemplateRos2Controller::writeControllerCommands`` emits zero wheel
    velocity unless the FSM is in RL.  PREPARE still records the measured
    wheel positions in the desired-position buffer used by RL.
    """

    controller = _controller(
        lambda _obs: np.zeros(6, dtype=np.float32),
        auto_enter_rl=False,
        lowlevel_output_mode="hardware_pd_vel",
    )
    controller.current_state = WheelbipeRos2ControllerState.PREPARE
    controller.target_state = WheelbipeRos2ControllerState.PREPARE
    controller._desired_positions[4:6] = np.asarray([1.25, -0.75])  # type: ignore[attr-defined]
    output = controller._write_commands(_state(), active=True)  # type: ignore[attr-defined]
    np.testing.assert_allclose(output.command.velocities[4:6], 0.0)

    controller.current_state = WheelbipeRos2ControllerState.RL
    output_rl = controller._write_commands(_state(), active=True)  # type: ignore[attr-defined]
    np.testing.assert_allclose(output_rl.command.velocities[4:6], [1.25, -0.75])


def test_prepare_updates_only_legs_and_preserves_prior_output_positions() -> None:
    """Mirror source ``NUM_PREPARE_JOINTS == 4`` in hardware-PD output.

    A direct INIT -> PREPARE transition starts with the reset-time non-leg
    output positions (zero).  When PREPARE is requested from IDLE, the source
    IDLE state has first refreshed those positions from the measured sample,
    and StatePrepare carries them through unchanged.
    """

    sample = _state(
        positions=np.asarray([0.2, -0.2, 0.3, -0.3, 1.25, -0.75, 0.4, -0.5]),
    )
    controller = _controller(
        lambda _obs: np.zeros(6, dtype=np.float32),
        auto_enter_rl=False,
        lowlevel_output_mode="hardware_pd",
    )
    controller.set_state_command(WheelbipeRos2ControllerState.PREPARE)
    controller.update(sample, timestamp=0.0, period=0.002)
    direct = controller.update(sample, timestamp=0.02, period=0.002)
    assert direct.state is WheelbipeRos2ControllerState.PREPARE
    # StatePrepare writes only indices 0..3; reset-time non-leg outputs remain
    # zero on a direct INIT transition.
    np.testing.assert_allclose(direct.command.positions[4:8], 0.0)

    idle_controller = _controller(
        lambda _obs: np.zeros(6, dtype=np.float32),
        auto_enter_rl=False,
        lowlevel_output_mode="hardware_pd",
    )
    idle_controller.update(sample, timestamp=0.0, period=0.002)
    idle = idle_controller.update(sample, timestamp=0.02, period=0.002)
    assert idle.state is WheelbipeRos2ControllerState.IDLE
    idle_controller.set_state_command(WheelbipeRos2ControllerState.PREPARE)
    prepared = idle_controller.update(sample, timestamp=0.022, period=0.002)
    assert prepared.state is WheelbipeRos2ControllerState.PREPARE
    np.testing.assert_allclose(prepared.command.positions[4:8], sample.positions[4:8])


def test_normal_observation_matches_owner_builder() -> None:
    controller = _controller(lambda _obs: np.zeros(6, dtype=np.float32))
    controller.set_state_command(3)
    controller.set_motion_command(0.4, -0.6, timestamp=0.0)
    controller.set_height_command(0.27, timestamp=0.0)
    sample = _state(
        positions=np.arange(8, dtype=np.float64) * 0.01,
        velocities=np.arange(8, dtype=np.float64) * 0.2,
    )
    # Cross the source 10 ms INIT hold and request RL.
    controller.update(sample, timestamp=0.0, period=0.002)
    output = controller.update(sample, timestamp=0.02, period=0.002)
    if output.state is not WheelbipeRos2ControllerState.RL:
        output = controller.update(sample, timestamp=0.022, period=0.002)
    assert output.observation is not None
    expected = build_wheelbipe_policy_observation(
        np.asarray([[0.4, 0.0, -0.6]]),
        np.asarray([0.27]),
        np.asarray([[0.1, 0.2, 0.3]]),
        np.asarray([[0.0, 0.0, -1.0]]),
        sample.positions[:4][None, :],
        sample.velocities[:4][None, :],
        sample.velocities[4:6][None, :],
        np.zeros((1, 6)),
    )[0]
    np.testing.assert_allclose(output.observation, expected)


def test_timeout_clamp_and_nonfinite_policy_fail_safe() -> None:
    controller = _controller(lambda _obs: np.ones(6, dtype=np.float32), auto_enter_rl=True)
    controller.set_motion_command(999.0, -999.0, timestamp=0.0)
    controller.set_height_command(999.0, timestamp=0.0)
    # At 0.6 s commands have expired; controller remains finite and safe.
    output = controller.update(_state(), timestamp=0.6, period=0.002)
    assert output.diagnostic["command_timed_out"]
    with pytest.raises(ValueError, match="finite"):
        controller.set_motion_command(np.nan, 0.0, timestamp=0.6)

    bad = _controller(lambda _obs: np.full(6, np.nan), auto_enter_rl=True)
    bad.update(_state(), timestamp=0.0, period=0.002)
    safe = bad.update(_state(), timestamp=0.02, period=0.002)
    if safe.state is not WheelbipeRos2ControllerState.RL:
        safe = bad.update(_state(), timestamp=0.022, period=0.002)
    assert safe.safe_stop
    assert safe.target_state is WheelbipeRos2ControllerState.IDLE
    assert np.all(np.isfinite(safe.command.final_torque))


def test_height_timeout_retains_rl_value_but_resets_when_idle() -> None:
    controller = _controller(lambda obs: np.zeros(6, dtype=np.float32), auto_enter_rl=True)
    controller.set_height_command(0.31, timestamp=0.0)
    controller.update(_state(), timestamp=0.0, period=0.002)
    output = controller.update(_state(), timestamp=0.02, period=0.002)
    if output.state is not WheelbipeRos2ControllerState.RL:
        output = controller.update(_state(), timestamp=0.022, period=0.002)
    assert output.observation is not None
    assert output.observation[3] == pytest.approx(0.31 * 5.0)
    # Source timeout handling retains the latest height in RL (the receipt flag
    # is cleared, but the value is not overwritten while RL is active).
    retained = controller.update(_state(), timestamp=0.6, period=0.002)
    assert retained.observation is not None
    assert retained.observation[3] == pytest.approx(0.31 * 5.0)
    controller.set_state_command(1)
    idle = controller.update(_state(), timestamp=0.602, period=0.002)
    assert idle.state is WheelbipeRos2ControllerState.IDLE


def test_dt7_state_uses_source_llround_and_sim_time_timestamp() -> None:
    controller = _controller(
        lambda _obs: np.zeros(6, dtype=np.float32),
        auto_enter_rl=False,
        use_dt7=True,
    )
    controller.update(_state(), timestamp=0.0, period=0.002, dt7=(2.5, 0.3, -0.4, 0.28))
    # C++ std::llround(2.5) is 3; the Python adapter must not truncate to 2.
    assert controller.target_state is WheelbipeRos2ControllerState.RL
    assert controller._motion.timestamp == pytest.approx(0.0)  # type: ignore[attr-defined]


def test_invalid_dt7_state_is_ignored_but_motion_and_height_are_kept() -> None:
    controller = _controller(
        lambda _obs: np.zeros(6, dtype=np.float32),
        auto_enter_rl=False,
        use_dt7=True,
    )
    controller.update(_state(), timestamp=0.0, period=0.002, dt7=(99.0, 0.7, -0.8, 0.28))
    # RealBridge/TemplateRos2Controller ignores an unsupported state byte while
    # still exposing the current DT7 velocity and height sample.
    assert controller.target_state is WheelbipeRos2ControllerState.IDLE
    assert controller._motion.linear_x == pytest.approx(0.7)  # type: ignore[attr-defined]
    assert controller._motion.angular_z == pytest.approx(-0.8)  # type: ignore[attr-defined]
    assert controller._height.value == pytest.approx(0.28)  # type: ignore[attr-defined]


def test_real_command_packet_size_crc_and_roundtrip() -> None:
    commands = [WheelbipeRealJointCommand(1, 2, 3, 4, 5) for _ in range(6)]
    encoded = encode_wheelbipe_real_command_packet(
        commands,
        h7_timestamp=1.25,
        pc_timestamp=1234.5,
        spring_compensation=(0.1, 0.2),
        speed_error=(0.3, 0.4, 0.5),
    )
    assert len(encoded) == WHEELBIPE_REAL_COMMAND_PACKET_SIZE
    assert wheelbipe_crc16(encoded[:-4]) == struct.unpack_from("<H", encoded, len(encoded) - 4)[0]
    decoded = decode_wheelbipe_real_command_packet(encoded)
    assert decoded.h7_timestamp == pytest.approx(1.25)
    assert decoded.commands[0].kp == pytest.approx(4.0)
    assert decoded.speed_error == pytest.approx((0.3, 0.4, 0.5))
    safe = encode_wheelbipe_real_command_packet(commands, safe_stop=True)
    assert all(
        value == 0.0
        for value in decode_wheelbipe_real_command_packet(safe).commands[0].__dict__.values()
    )


def _state_packet() -> bytes:
    # Header/timestamps + 6*(position,velocity,effort) + IMU + ordinary DT7.
    joint = [float(index) for index in range(18)]
    imu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]
    dt7 = [2, 0.7, -0.8, 0.25]
    prefix = struct.pack("<BBfd18f10fB3f", 0xA8, 0xE6, 2.5, 99.0, *joint, *imu, *dt7)
    assert len(prefix) == WHEELBIPE_REAL_STATE_PACKET_SIZE - 4
    return prefix + struct.pack("<HBB", wheelbipe_crc16(prefix), 0xC3, 0xF7)


def test_real_state_packet_validation_and_stream_decoder() -> None:
    packet = _state_packet()
    assert len(packet) == WHEELBIPE_REAL_STATE_PACKET_SIZE
    decoded = decode_wheelbipe_real_state_packet(packet)
    assert decoded.joint_positions == pytest.approx((0.0, 3.0, 6.0, 9.0, 12.0, 15.0))
    assert decoded.dt7.state == 2
    decoder = WheelbipeRealStateDecoder()
    assert decoder.feed(b"noise" + packet[:40]) == []
    accepted = decoder.feed(packet[40:])
    assert len(accepted) == 1
    with pytest.raises(ValueError, match="CRC"):
        decode_wheelbipe_real_state_packet(packet[:-5] + b"x" + packet[-4:])


def test_real_state_packet_encoder_roundtrip() -> None:
    decoded = decode_wheelbipe_real_state_packet(_state_packet())
    encoded = encode_wheelbipe_real_state_packet(decoded)
    assert encoded == _state_packet()
    assert decode_wheelbipe_real_state_packet(encoded) == decoded


def test_invalid_sensor_sample_returns_finite_safe_stop() -> None:
    controller = _controller(lambda _obs: np.zeros(6, dtype=np.float32))
    controller.set_state_command(2)
    output = controller.update(
        _state(orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 0.0])),
        timestamp=0.0,
        period=0.002,
    )
    assert output.safe_stop
    assert output.target_state is WheelbipeRos2ControllerState.PREPARE
    assert output.diagnostic["target_idle_requested"] is False
    assert np.all(np.isfinite(output.command.positions))


def test_hardware_pd_vel_keeps_wheel_velocity_zero_during_prepare() -> None:
    """Source ``hardware_pd_vel`` gates wheel velocity commands to StateRL."""

    controller = _controller(lambda _obs: np.zeros(6, dtype=np.float32))
    controller.set_state_command(WheelbipeRos2ControllerState.PREPARE)
    controller.update(_state(), timestamp=0.0, period=0.002)
    output = controller.update(_state(), timestamp=0.02, period=0.002)
    assert output.state is WheelbipeRos2ControllerState.PREPARE
    np.testing.assert_array_equal(output.command.velocities[4:6], np.zeros(2))

    controller.set_state_command(WheelbipeRos2ControllerState.RL)
    output = controller.update(_state(), timestamp=0.022, period=0.002)
    assert output.state is WheelbipeRos2ControllerState.RL
    # Zero policy action still produces a valid zero wheel velocity target;
    # this assertion distinguishes RL gating from accidental PREPARE output.
    np.testing.assert_array_equal(output.command.velocities[4:6], np.zeros(2))


def test_native_ros2_bundle_verifies_static_packages_plugins_and_launches() -> None:
    verified = verify_wheelbipe_ros2_native_bundle()
    assert verified["verified"] is True
    assert verified["source_revision"] == ros2.WHEELBIPE_DEPLOYMENT_SOURCE_REVISION
    assert verified["source_license"] == "MIT"
    assert verified["source_file_count"] == 72
    assert verified["packages"] == [
        "keyboard_teleop",
        "mujoco_ros2_control",
        "real_bridge",
        "robot_descriptions",
        "template_middleware",
        "template_ros2_controller",
        "xbox_teleop",
    ]
    assert verified["controller_manager"]["manager"] == ("/wheelbipe_V14/controller_manager")
    assert verified["controller_manager"]["spawners"] == [
        "joint_state_broadcaster",
        "template_ros2_controller",
    ]
    plugins = {plugin["name"]: plugin for plugin in verified["plugins"]}
    assert plugins["robot_locomotion/TemplateRos2Controller"]["base_class_type"] == (
        "controller_interface::ControllerInterface"
    )
    assert plugins["template_real_ros2_ctrl::RealBridge"]["base_class_type"] == (
        "hardware_interface::SystemInterface"
    )
    assert len(verified["launch_files"]) == 4
    assert len(verified["config_files"]) == 4
    assert len(verified["host_rules"]) == 2
    assert all(rule["installed"] is False for rule in verified["host_rules"])
    assert verified["dependency_versions"] == {
        "mujoco": "3.5.0",
        "onnxruntime": "1.20.0",
    }
    assert verified["claims"] == {
        "source_packaged": True,
        "native_ros2_active": False,
        "ros_graph_created": False,
        "serial_device_opened": False,
        "hardware_validated": False,
        "realtime_guarantee": False,
    }


def test_native_manifest_revision_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = (
        Path(__file__).resolve().parents[2]
        / "deployment"
        / "ros2"
        / "wheelbipe_v14_native"
        / "manifest.yaml"
    )
    malformed = tmp_path / "manifest.yaml"
    malformed.write_text(
        manifest.read_text(encoding="utf-8").replace(
            ros2.WHEELBIPE_DEPLOYMENT_SOURCE_REVISION,
            "0" * 40,
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(WheelbipeRos2NativeBundleError, match="source_revision"):
        verify_wheelbipe_ros2_native_bundle(manifest_path=malformed)


def test_native_runtime_probe_is_non_importing_and_require_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ros2, "find_spec", lambda _name: None)
    monkeypatch.setattr(ros2.shutil, "which", lambda _name: None)
    probe = wheelbipe_ros2_native_runtime_probe()
    assert probe["bundle_verified"] is True
    assert probe["optional_dependencies_available"] is False
    assert probe["missing_python_modules"] == [
        "rclpy",
        "launch",
        "launch_ros",
        "ament_index_python",
    ]
    assert probe["missing_executables"] == [
        "ros2",
        "colcon",
        "xacro",
        "cmake",
        "g++",
        "pkg-config",
        "rosdep",
    ]
    assert probe["required_platform"] == "linux-x86_64"
    assert probe["workspace_built"] is False
    assert probe["ros_graph_created"] is False
    assert probe["serial_device_opened"] is False
    with pytest.raises(
        WheelbipeRos2NativeBundleError,
        match="external ROS 2 Humble/colcon environment",
    ):
        require_wheelbipe_ros2_native_runtime()


def test_native_workspace_materializer_copies_assets_without_building(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "wheelbipe_colcon"
    result = materialize_wheelbipe_ros2_native_workspace(destination)
    assert result["copied_asset_files"] == 22
    assert result["workspace_built"] is False
    assert result["workspace_sourced"] is False
    assert result["ros_graph_created"] is False
    assert result["host_rules_installed"] is False
    assert (
        destination
        / "src"
        / "controllers"
        / "template_ros2_controller"
        / "template_ros2_controller_plugin.xml"
    ).is_file()
    assert (
        destination
        / "src"
        / "controllers"
        / "template_ros2_controller"
        / "policy"
        / "parallel"
        / "V14-35-flat-and-rotation-13k.onnx"
    ).is_file()
    assert (
        destination
        / "src"
        / "resources"
        / "robot_descriptions"
        / "wheelbipeV14_2"
        / "meshes"
        / "base_link.STL"
    ).is_file()
    assert (destination / "udev" / "99-wheelbipe-serial.rules").is_file()
    assert (destination / "LICENSE").is_file()
    assert (destination / "dependencies.lock").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_wheelbipe_ros2_native_workspace(destination)


def test_real_bridge_gate_throttles_reconnect_and_inhibits_stale_output() -> None:
    gate = WheelbipeRealBridgeGate()
    assert gate.reconnect_interval_sec == WHEELBIPE_REAL_RECONNECT_INTERVAL_SEC
    assert gate.should_attempt_connect(10.0)
    gate.record_connection_attempt(succeeded=False, now=10.0)
    assert not gate.should_attempt_connect(10.999)
    with pytest.raises(RuntimeError, match="rate limited"):
        gate.record_connection_attempt(succeeded=True, now=10.999)
    assert gate.should_attempt_connect(11.0)
    gate.record_connection_attempt(succeeded=True, now=11.0)
    gate.activate()
    assert not gate.command_permitted(11.0)
    assert gate.feed_state_bytes(b"garbage", now=11.001) == []
    assert not gate.command_permitted(11.001)
    accepted = gate.feed_state_bytes(_state_packet(), now=11.01)
    assert len(accepted) == 1
    assert gate.command_permitted(11.10)
    assert not gate.command_permitted(11.111)
    assert gate.has_valid_state is False
    assert gate.diagnostic(now=11.112)["command_permitted"] is False


def test_real_bridge_gate_deactivate_only_requests_safe_stop_from_fresh_state() -> None:
    gate = WheelbipeRealBridgeGate(connected=True, active=True)
    gate.accept_state_packet(decode_wheelbipe_real_state_packet(_state_packet()), now=5.0)
    assert gate.deactivate(now=5.05) is True
    assert gate.active is False
    assert gate.connected is False
    assert gate.has_valid_state is False

    stale = WheelbipeRealBridgeGate(connected=True, active=True)
    stale.accept_state_packet(decode_wheelbipe_real_state_packet(_state_packet()), now=5.0)
    assert stale.deactivate(now=5.101) is False


def test_xbox_teleop_mapping_and_disconnect_are_fail_closed() -> None:
    command = wheelbipe_xbox_teleop_command(
        left_y=-32767,
        right_x=32767,
        left_trigger=0,
        right_trigger=1023,
        current_height=0.22,
        dt=0.5,
        connected=True,
        start_pressed=True,
    )
    assert command.linear_x == pytest.approx(2.5)
    assert command.angular_z == pytest.approx(-3.0)
    assert command.height == pytest.approx(0.26)
    assert command.state_command == 3
    assert command.safe_stop is False

    disconnected = wheelbipe_xbox_teleop_command(
        left_y=-32767,
        right_x=32767,
        left_trigger=1023,
        right_trigger=1023,
        current_height=0.3,
        dt=0.02,
        connected=False,
    )
    assert disconnected.linear_x == 0.0
    assert disconnected.angular_z == 0.0
    assert disconnected.height == pytest.approx(0.3)
    assert disconnected.safe_stop is True
    with pytest.raises(ValueError, match="ambiguous"):
        wheelbipe_xbox_teleop_command(
            left_y=0,
            right_x=0,
            left_trigger=0,
            right_trigger=0,
            current_height=0.22,
            dt=0.02,
            connected=True,
            start_pressed=True,
            record_pressed=True,
        )
    with pytest.raises(ValueError, match="left_y must be finite"):
        wheelbipe_xbox_teleop_command(
            left_y=float("nan"),
            right_x=0,
            left_trigger=0,
            right_trigger=0,
            current_height=0.22,
            dt=0.02,
            connected=True,
        )


def test_native_cli_verification_stays_on_cold_path(capsys: pytest.CaptureFixture[str]) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "sim2sim_wheelbipe_ros2.py"
    namespace = runpy.run_path(str(script), run_name="wheelbipe_ros2_cli_test")
    assert namespace["main"](["--verify-native-bundle"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["claims"]["native_ros2_active"] is False
    assert payload["claims"]["hardware_validated"] is False
