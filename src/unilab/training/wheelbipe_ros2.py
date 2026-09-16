"""Owner-layer adapter for the WheelBipe V14 ROS 2 deployment contract.

The upstream ``wheelbipe_ros2_sim2sim`` project is a ROS 2/ros2_control
controller.  ROS 2 Humble, ``controller_manager`` and the hardware plugins are
not runtime dependencies of UniLab, so importing this module deliberately does
not import ``rclpy`` or any ROS message package.  Instead it keeps the
deployment boundary executable in a normal Python process:

* :class:`WheelbipeRos2Controller` mirrors the source controller's public
  inputs, 8-joint state, INIT/IDLE/PREPARE/RL state machine, 500 Hz update and
  50 Hz policy cadence, observation layout, low-level output modes, and
  fail-safe behavior.
* The packet helpers and :class:`WheelbipeRealBridgeGate` mirror the
  MIT-licensed ``RealBridge`` byte layout, CRC, reconnect throttle and stale
  output gate without opening a serial device.
* A pinned native colcon source bundle can be verified and materialized on a
  cold path.  ROS system dependencies, builds and launches remain external.

This is an owner adapter, not a claim that UniLab provides a ROS graph,
``ros2_control`` lifecycle, source dynamics, or certified hardware safety.
The source API/message/timing constants are kept explicit below so tests and
callers can audit the boundary without depending on C++ headers.
"""

from __future__ import annotations

import math
import platform
import shutil
import struct
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

import numpy as np

from unilab.envs.locomotion.wheelbipe_v14.base import (
    NORMAL_CONTROL_MODE,
    NUM_NATIVE_ACTUATORS,
    NUM_POLICY_ACTIONS,
    POLICY_OBS_CLIP,
    POLICY_OBS_DIM,
)
from unilab.utils.rotation import np_quat_apply_inverse

# ---------------------------------------------------------------------------
# Source ROS API and timing contract
# ---------------------------------------------------------------------------

WHEELBIPE_ROS2_NAMESPACE = "wheelbipe_V14"
WHEELBIPE_ROS2_CONTROLLER_NAME = "template_ros2_controller"
WHEELBIPE_ROS2_UPDATE_RATE_HZ = 500
WHEELBIPE_ROS2_INFERENCE_RATE_HZ = 50
WHEELBIPE_ROS2_INIT_HOLD_SEC = 0.01
WHEELBIPE_TRAINING_SOURCE_REVISION = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
WHEELBIPE_DEPLOYMENT_SOURCE_REVISION = "daa34f54d56cab91b3989d8152a7ce7b61092994"
WHEELBIPE_ROS2_NATIVE_MANIFEST_RELATIVE_PATH = Path(
    "deployment/ros2/wheelbipe_v14_native/manifest.yaml"
)
WHEELBIPE_ROS2_NATIVE_SOURCE_RELATIVE_PATH = Path("deployment/ros2/wheelbipe_v14_native/src")
WHEELBIPE_ROS2_NATIVE_SOURCE_TREE_SHA256 = (
    "67f0628533f6f8cc849159426fe6769636c304ab4596d34cbfeed4a2f4e53b27"
)
WHEELBIPE_ROS2_NATIVE_SOURCE_FILE_COUNT = 72
WHEELBIPE_ROS2_NATIVE_REQUIRED_MODULES: tuple[str, ...] = (
    "rclpy",
    "launch",
    "launch_ros",
    "ament_index_python",
)
WHEELBIPE_ROS2_NATIVE_REQUIRED_EXECUTABLES: tuple[str, ...] = (
    "ros2",
    "colcon",
    "xacro",
    "cmake",
    "g++",
    "pkg-config",
    "rosdep",
)
WHEELBIPE_ROS2_NATIVE_PLATFORM = "linux-x86_64"

WHEELBIPE_ROS2_JOINT_NAMES: tuple[str, ...] = (
    "left_front1_joint",
    "left_rear1_joint",
    "right_front1_joint",
    "right_rear1_joint",
    "left_wheel_joint",
    "right_wheel_joint",
    "left_spring2_joint",
    "right_spring2_joint",
)
WHEELBIPE_ROS2_COMMUNICATED_JOINT_NAMES: tuple[str, ...] = WHEELBIPE_ROS2_JOINT_NAMES[:6]
WHEELBIPE_ROS2_COMMAND_INTERFACES: tuple[str, ...] = (
    "position",
    "velocity",
    "effort",
    "kp",
    "kd",
)
WHEELBIPE_ROS2_STATE_INTERFACES: tuple[str, ...] = ("position", "velocity", "effort")
WHEELBIPE_ROS2_SENSOR_NAMES: tuple[str, ...] = ("imu",)
WHEELBIPE_ROS2_DT7_INTERFACES: tuple[str, ...] = (
    "cmd_state",
    "cmd_vel_x",
    "cmd_omega_z",
    "cmd_height",
)

# ``template_ros2_controller`` uses a one-element best-effort subscription
# queue for the three command topics.  Publishers use the ordinary ROS system
# default queue depth of ten.  Keep this as data rather than importing
# ``rclpy.qos`` so a no-ROS process can inspect the contract.
WHEELBIPE_ROS2_QOS: dict[str, dict[str, Any]] = {
    "command_subscriptions": {
        "history": "keep_last",
        "depth": 1,
        "reliability": "best_effort",
        "durability": "system_default",
    },
    "state_publishers": {
        "history": "keep_last",
        "depth": 10,
        # The source uses the ``create_publisher(topic, 10)`` overload, whose
        # rmw default profile is reliable/volatile (unlike the explicit
        # SystemDefaultsQoS command subscriptions above).
        "reliability": "reliable",
        "durability": "volatile",
        "source_constructor": "create_publisher(topic, 10)",
    },
}

# A serializable snapshot of the source launch/config contract.  Keeping this
# at module scope makes it possible for a CLI or test to audit the API without
# constructing a controller or importing ROS message classes.
WHEELBIPE_ROS2_API_CONTRACT: dict[str, Any] = {
    "namespace": WHEELBIPE_ROS2_NAMESPACE,
    "controller": WHEELBIPE_ROS2_CONTROLLER_NAME,
    "joints": WHEELBIPE_ROS2_JOINT_NAMES,
    "communicated_joints": WHEELBIPE_ROS2_COMMUNICATED_JOINT_NAMES,
    "command_interfaces": WHEELBIPE_ROS2_COMMAND_INTERFACES,
    "state_interfaces": WHEELBIPE_ROS2_STATE_INTERFACES,
    "sensors": WHEELBIPE_ROS2_SENSOR_NAMES,
    "dt7_interfaces": WHEELBIPE_ROS2_DT7_INTERFACES,
    "update_rate_hz": WHEELBIPE_ROS2_UPDATE_RATE_HZ,
    "inference_rate_hz": WHEELBIPE_ROS2_INFERENCE_RATE_HZ,
    "policy_input_shape": (1, POLICY_OBS_DIM),
    "policy_output_shape": (1, NUM_POLICY_ACTIONS),
    "qos": WHEELBIPE_ROS2_QOS,
}

# The values are intentionally plain dictionaries: they can be serialized in
# diagnostics and compared against the source launch/config files without
# importing a ROS message class.
WHEELBIPE_ROS2_TOPICS: dict[str, dict[str, str]] = {
    "motion_command": {
        "direction": "input",
        "type": "geometry_msgs/msg/Twist",
        "name": f"/{WHEELBIPE_ROS2_NAMESPACE}/motion_command",
        "fields": "linear.x,angular.z",
    },
    "height_command": {
        "direction": "input",
        "type": "std_msgs/msg/Float64",
        "name": f"/{WHEELBIPE_ROS2_NAMESPACE}/height_command",
        "fields": "data",
    },
    "state_command": {
        "direction": "input",
        "type": "std_msgs/msg/Int32",
        "name": f"/{WHEELBIPE_ROS2_NAMESPACE}/state_command",
        "fields": "data (0..3)",
    },
    "current_state": {
        "direction": "output",
        "type": "std_msgs/msg/Int32",
        "name": f"/{WHEELBIPE_ROS2_NAMESPACE}/current_state",
        "fields": "data",
    },
    "joint_commands": {
        "direction": "output",
        "type": "sensor_msgs/msg/JointState",
        "name": f"/{WHEELBIPE_ROS2_NAMESPACE}/joint_commands",
        "fields": "name,position,velocity,effort",
    },
    "joint_final_torque": {
        "direction": "output",
        "type": "std_msgs/msg/Float64MultiArray",
        "name": f"/{WHEELBIPE_ROS2_NAMESPACE}/joint_final_torque",
        "fields": "data[8]",
    },
    "rl_network_input": {
        "direction": "debug-output",
        "type": "std_msgs/msg/Float64MultiArray",
        "name": f"/{WHEELBIPE_ROS2_NAMESPACE}/{WHEELBIPE_ROS2_CONTROLLER_NAME}/rl_network_input",
        "fields": "data[35]",
    },
    "rl_network_output": {
        "direction": "debug-output",
        "type": "std_msgs/msg/Float64MultiArray",
        "name": f"/{WHEELBIPE_ROS2_NAMESPACE}/{WHEELBIPE_ROS2_CONTROLLER_NAME}/rl_network_output",
        "fields": "data[6]",
    },
}

WHEELBIPE_ROS2_TELEOP_CONTRACT: dict[str, Any] = {
    "topics": {
        "motion": f"/{WHEELBIPE_ROS2_NAMESPACE}/motion_command",
        "height": f"/{WHEELBIPE_ROS2_NAMESPACE}/height_command",
        "state": f"/{WHEELBIPE_ROS2_NAMESPACE}/state_command",
        "current_state": f"/{WHEELBIPE_ROS2_NAMESPACE}/current_state",
    },
    "qos": {
        "history": "keep_last",
        "depth": 1,
        "reliability": "best_effort",
        "durability": "volatile",
    },
    "keyboard": {
        "motion_keys": {
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "stop": "space",
        },
        "height_keys": {"up": "t", "down": "g", "reset": "r"},
        "state_keys": {"INIT": "0", "IDLE": "1", "PREPARE": "2", "RL": "3"},
        "exit_key": "x",
        "stdin_must_be_tty": True,
    },
    "xbox": {
        "device_api": "linux_evdev",
        "reconnect_interval_ms": 1000,
        "start_button_code": 315,
        "record_button_code": 167,
        "axes": {"linear_x": 1, "angular_z": 3, "height_down": 2, "height_up": 5},
        "disconnect_behavior": "immediate_zero_motion_and_clear_axes",
    },
}


class WheelbipeRos2NativeBundleError(RuntimeError):
    """Raised when the optional native ROS 2 source/runtime boundary is invalid."""


def _wheelbipe_repository_root(repository_root: str | Path | None = None) -> Path:
    root = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    if not root.is_dir():
        raise WheelbipeRos2NativeBundleError(f"repository root does not exist: {root}")
    return root


def _wheelbipe_safe_child(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WheelbipeRos2NativeBundleError(f"{name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise WheelbipeRos2NativeBundleError(f"{name} must be relative, got {value!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise WheelbipeRos2NativeBundleError(f"{name} escapes its owner root: {value!r}") from exc
    return candidate


def _wheelbipe_native_manifest_data(
    *,
    repository_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    root = _wheelbipe_repository_root(repository_root)
    path = (
        root / WHEELBIPE_ROS2_NATIVE_MANIFEST_RELATIVE_PATH
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    if not path.is_file():
        raise WheelbipeRos2NativeBundleError(f"native ROS2 manifest does not exist: {path}")
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - Hydra is a project dependency
        raise WheelbipeRos2NativeBundleError(
            "OmegaConf is required to inspect the native ROS2 manifest"
        ) from exc
    try:
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise WheelbipeRos2NativeBundleError(
            f"unable to parse native ROS2 manifest {path}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise WheelbipeRos2NativeBundleError("native ROS2 manifest must contain a mapping")
    return root, path, cast(dict[str, Any], loaded)


def load_wheelbipe_ros2_native_manifest(
    *,
    repository_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the static native manifest without importing or starting ROS 2."""

    _root, _path, loaded = _wheelbipe_native_manifest_data(
        repository_root=repository_root,
        manifest_path=manifest_path,
    )
    return deepcopy(loaded)


def _wheelbipe_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WheelbipeRos2NativeBundleError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _wheelbipe_sequence(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise WheelbipeRos2NativeBundleError(f"{name} must be a list")
    return value


def _wheelbipe_tree_digest(root: Path, files: Sequence[Path]) -> str:
    digest = sha256()
    for file_path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _wheelbipe_verify_package(source_root: Path, package_name: str, relative: object) -> None:
    package_root = _wheelbipe_safe_child(source_root, relative, name=f"packages.{package_name}")
    if not package_root.is_dir():
        raise WheelbipeRos2NativeBundleError(
            f"native ROS2 package directory is missing: {package_root}"
        )
    package_xml = package_root / "package.xml"
    cmake = package_root / "CMakeLists.txt"
    if not package_xml.is_file() or not cmake.is_file():
        raise WheelbipeRos2NativeBundleError(
            f"native ROS2 package {package_name!r} must contain package.xml and CMakeLists.txt"
        )
    try:
        declared_name = ElementTree.parse(package_xml).getroot().findtext("name")
    except (OSError, ElementTree.ParseError) as exc:
        raise WheelbipeRos2NativeBundleError(
            f"invalid package.xml for {package_name!r}: {exc}"
        ) from exc
    if declared_name != package_name:
        raise WheelbipeRos2NativeBundleError(
            f"package path for {package_name!r} declares {declared_name!r}"
        )


def _wheelbipe_verify_plugin(source_root: Path, value: object, index: int) -> dict[str, str]:
    plugin = _wheelbipe_mapping(value, name=f"plugins[{index}]")
    required = (
        "package",
        "descriptor",
        "cmake",
        "category",
        "library",
        "name",
        "type",
        "base_class_type",
    )
    for field_name in required:
        if not isinstance(plugin.get(field_name), str) or not plugin[field_name]:
            raise WheelbipeRos2NativeBundleError(
                f"plugins[{index}].{field_name} must be a non-empty string"
            )
    descriptor = _wheelbipe_safe_child(
        source_root, plugin["descriptor"], name=f"plugins[{index}].descriptor"
    )
    cmake = _wheelbipe_safe_child(source_root, plugin["cmake"], name=f"plugins[{index}].cmake")
    if not descriptor.is_file() or not cmake.is_file():
        raise WheelbipeRos2NativeBundleError(
            f"plugin {plugin['name']!r} is missing its descriptor or CMake registration"
        )
    try:
        xml_root = ElementTree.parse(descriptor).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise WheelbipeRos2NativeBundleError(
            f"invalid plugin descriptor {descriptor}: {exc}"
        ) from exc
    if xml_root.tag != "library" or xml_root.attrib.get("path") != plugin["library"]:
        raise WheelbipeRos2NativeBundleError(
            f"plugin descriptor {descriptor} has the wrong library"
        )
    matching = [
        node for node in xml_root.findall("class") if node.attrib.get("name") == plugin["name"]
    ]
    if len(matching) != 1:
        raise WheelbipeRos2NativeBundleError(
            f"plugin descriptor {descriptor} must declare {plugin['name']!r} exactly once"
        )
    declaration = matching[0].attrib
    if (
        declaration.get("type") != plugin["type"]
        or declaration.get("base_class_type") != plugin["base_class_type"]
    ):
        raise WheelbipeRos2NativeBundleError(
            f"plugin descriptor {descriptor} has an incompatible class contract"
        )
    cmake_text = cmake.read_text(encoding="utf-8")
    if (
        "pluginlib_export_plugin_description_file" not in cmake_text
        or plugin["category"] not in cmake_text
        or Path(plugin["descriptor"]).name not in cmake_text
    ):
        raise WheelbipeRos2NativeBundleError(
            f"plugin {plugin['name']!r} is not registered through pluginlib in {cmake}"
        )
    return {field_name: str(plugin[field_name]) for field_name in required}


def verify_wheelbipe_ros2_native_bundle(
    *,
    repository_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the pinned native source bundle without importing ROS modules.

    Verification covers package manifests, pluginlib descriptors and CMake
    registration, launch syntax, YAML configs, pinned source hashes and the
    binary asset overlays used by the cold-path workspace materializer.  It
    does not build a colcon workspace, create a ROS graph, open a serial port,
    or establish hardware/realtime safety.
    """

    root, path, manifest = _wheelbipe_native_manifest_data(
        repository_root=repository_root,
        manifest_path=manifest_path,
    )
    if manifest.get("schema_version") != 1:
        raise WheelbipeRos2NativeBundleError("native ROS2 manifest schema_version must be 1")
    bundle = _wheelbipe_mapping(manifest.get("bundle"), name="bundle")
    expected_bundle_values = {
        "id": "wheelbipe_v14_native_ros2",
        "source_revision": WHEELBIPE_DEPLOYMENT_SOURCE_REVISION,
        "source_root": WHEELBIPE_ROS2_NATIVE_SOURCE_RELATIVE_PATH.as_posix(),
        "source_file_count": WHEELBIPE_ROS2_NATIVE_SOURCE_FILE_COUNT,
        "source_tree_sha256": WHEELBIPE_ROS2_NATIVE_SOURCE_TREE_SHA256,
    }
    for field_name, expected in expected_bundle_values.items():
        if bundle.get(field_name) != expected:
            raise WheelbipeRos2NativeBundleError(
                f"bundle.{field_name} must be {expected!r}, got {bundle.get(field_name)!r}"
            )
    if bundle.get("source_license") != "MIT":
        raise WheelbipeRos2NativeBundleError("bundle.source_license must be 'MIT'")
    license_path = _wheelbipe_safe_child(
        root, bundle.get("license_file"), name="bundle.license_file"
    )
    if not license_path.is_file() or not isinstance(bundle.get("license_sha256"), str):
        raise WheelbipeRos2NativeBundleError("native source license file/hash is invalid")
    license_digest = sha256(license_path.read_bytes()).hexdigest()
    if license_digest != bundle["license_sha256"]:
        raise WheelbipeRos2NativeBundleError(
            f"native source license hash mismatch: {license_digest}"
        )
    source_root = _wheelbipe_safe_child(root, bundle["source_root"], name="bundle.source_root")
    if not source_root.is_dir():
        raise WheelbipeRos2NativeBundleError(
            f"native ROS2 source bundle does not exist: {source_root}"
        )
    entries = sorted(source_root.rglob("*"))
    symlinks = [entry for entry in entries if entry.is_symlink()]
    if symlinks:
        raise WheelbipeRos2NativeBundleError(
            f"native ROS2 source bundle must not contain symlinks: {symlinks[0]}"
        )
    source_files = [entry for entry in entries if entry.is_file()]
    if len(source_files) != WHEELBIPE_ROS2_NATIVE_SOURCE_FILE_COUNT:
        raise WheelbipeRos2NativeBundleError(
            "native ROS2 source file count mismatch: "
            f"expected {WHEELBIPE_ROS2_NATIVE_SOURCE_FILE_COUNT}, got {len(source_files)}"
        )
    source_digest = _wheelbipe_tree_digest(source_root, source_files)
    if source_digest != WHEELBIPE_ROS2_NATIVE_SOURCE_TREE_SHA256:
        raise WheelbipeRos2NativeBundleError(
            "native ROS2 source tree hash mismatch: "
            f"expected {WHEELBIPE_ROS2_NATIVE_SOURCE_TREE_SHA256}, got {source_digest}"
        )

    expected_packages = {
        "template_ros2_controller": "controllers/template_ros2_controller",
        "mujoco_ros2_control": "interfaces/mujoco_bridge",
        "real_bridge": "interfaces/real_bridge",
        "template_middleware": "middlewares/template_middleware",
        "robot_descriptions": "resources/robot_descriptions",
        "keyboard_teleop": "tools/keyboard_teleop",
        "xbox_teleop": "tools/xbox_teleop",
    }
    packages = _wheelbipe_mapping(manifest.get("packages"), name="packages")
    if packages != expected_packages:
        raise WheelbipeRos2NativeBundleError(
            "native ROS2 package map is incomplete or does not match the pinned layout"
        )
    for package_name, relative in packages.items():
        _wheelbipe_verify_package(source_root, package_name, relative)

    expected_controller_manager = {
        "namespace": WHEELBIPE_ROS2_NAMESPACE,
        "manager": f"/{WHEELBIPE_ROS2_NAMESPACE}/controller_manager",
        "update_rate_hz": WHEELBIPE_ROS2_UPDATE_RATE_HZ,
        "spawners": ["joint_state_broadcaster", WHEELBIPE_ROS2_CONTROLLER_NAME],
        "controller_type": "robot_locomotion/TemplateRos2Controller",
        "joints": list(WHEELBIPE_ROS2_JOINT_NAMES),
        "command_interfaces": list(WHEELBIPE_ROS2_COMMAND_INTERFACES),
        "state_interfaces": list(WHEELBIPE_ROS2_STATE_INTERFACES),
        "sensors": list(WHEELBIPE_ROS2_SENSOR_NAMES),
        "supported_backends": ["sim", "real"],
    }
    controller_manager_contract = _wheelbipe_mapping(
        manifest.get("controller_manager_contract"), name="controller_manager_contract"
    )
    if controller_manager_contract != expected_controller_manager:
        raise WheelbipeRos2NativeBundleError(
            "controller_manager_contract does not match the pinned source boundary"
        )

    plugin_values = _wheelbipe_sequence(manifest.get("plugins"), name="plugins")
    if len(plugin_values) != 3:
        raise WheelbipeRos2NativeBundleError("native ROS2 manifest must declare three plugins")
    plugins = [
        _wheelbipe_verify_plugin(source_root, plugin, index)
        for index, plugin in enumerate(plugin_values)
    ]

    launch_values = _wheelbipe_sequence(manifest.get("launch_files"), name="launch_files")
    if len(launch_values) != 4:
        raise WheelbipeRos2NativeBundleError("native ROS2 manifest must declare four launch files")
    launch_files: list[str] = []
    for index, relative in enumerate(launch_values):
        launch_path = _wheelbipe_safe_child(source_root, relative, name=f"launch_files[{index}]")
        if not launch_path.is_file():
            raise WheelbipeRos2NativeBundleError(f"native launch file is missing: {launch_path}")
        try:
            compile(launch_path.read_text(encoding="utf-8"), str(launch_path), "exec")
        except (OSError, SyntaxError) as exc:
            raise WheelbipeRos2NativeBundleError(
                f"native launch file has invalid Python syntax: {launch_path}: {exc}"
            ) from exc
        launch_files.append(str(relative))

    config_values = _wheelbipe_sequence(manifest.get("config_files"), name="config_files")
    if len(config_values) != 4:
        raise WheelbipeRos2NativeBundleError("native ROS2 manifest must declare four config files")
    from omegaconf import OmegaConf

    config_files: list[str] = []
    parsed_configs: dict[str, dict[str, Any]] = {}
    for index, relative in enumerate(config_values):
        config_path = _wheelbipe_safe_child(source_root, relative, name=f"config_files[{index}]")
        if not config_path.is_file():
            raise WheelbipeRos2NativeBundleError(f"native config file is missing: {config_path}")
        try:
            parsed_config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        except Exception as exc:
            raise WheelbipeRos2NativeBundleError(
                f"native config file is invalid: {config_path}: {exc}"
            ) from exc
        if not isinstance(parsed_config, dict):
            raise WheelbipeRos2NativeBundleError(
                f"native config file must contain a mapping: {config_path}"
            )
        config_files.append(str(relative))
        parsed_configs[str(relative)] = cast(dict[str, Any], parsed_config)

    middleware_config = parsed_configs["middlewares/template_middleware/config/wheelbipe_V14.yaml"]
    try:
        wildcard = cast(dict[str, Any], middleware_config["/**"])
        manager_parameters = cast(
            dict[str, Any],
            cast(dict[str, Any], wildcard["controller_manager"])["ros__parameters"],
        )
        controller_parameters = cast(
            dict[str, Any],
            cast(dict[str, Any], wildcard[WHEELBIPE_ROS2_CONTROLLER_NAME])["ros__parameters"],
        )
        static_manager_values = {
            "update_rate_hz": manager_parameters["update_rate"],
            "joint_state_broadcaster": cast(
                dict[str, Any], manager_parameters["joint_state_broadcaster"]
            )["type"],
            "controller_type": cast(
                dict[str, Any], manager_parameters[WHEELBIPE_ROS2_CONTROLLER_NAME]
            )["type"],
            "controller_update_rate_hz": controller_parameters["update_rate"],
            "joints": controller_parameters["joints"],
            "command_interfaces": controller_parameters["command_interfaces"],
            "state_interfaces": controller_parameters["state_interfaces"],
            "sensors": controller_parameters["sensors"],
        }
    except (KeyError, TypeError) as exc:
        raise WheelbipeRos2NativeBundleError(
            "middleware config is missing the controller_manager interface contract"
        ) from exc
    expected_static_manager_values = {
        "update_rate_hz": WHEELBIPE_ROS2_UPDATE_RATE_HZ,
        "joint_state_broadcaster": "joint_state_broadcaster/JointStateBroadcaster",
        "controller_type": expected_controller_manager["controller_type"],
        "controller_update_rate_hz": WHEELBIPE_ROS2_UPDATE_RATE_HZ,
        "joints": expected_controller_manager["joints"],
        "command_interfaces": expected_controller_manager["command_interfaces"],
        "state_interfaces": expected_controller_manager["state_interfaces"],
        "sensors": expected_controller_manager["sensors"],
    }
    if static_manager_values != expected_static_manager_values:
        raise WheelbipeRos2NativeBundleError(
            "middleware config does not match controller_manager_contract"
        )

    host_rule_values = _wheelbipe_sequence(manifest.get("host_rules"), name="host_rules")
    if len(host_rule_values) != 2:
        raise WheelbipeRos2NativeBundleError("native ROS2 manifest must declare two udev rules")
    host_rules: list[dict[str, Any]] = []
    for index, raw_rule in enumerate(host_rule_values):
        rule = _wheelbipe_mapping(raw_rule, name=f"host_rules[{index}]")
        rule_path = _wheelbipe_safe_child(root, rule.get("path"), name=f"host_rules[{index}].path")
        if not rule_path.is_file() or not isinstance(rule.get("sha256"), str):
            raise WheelbipeRos2NativeBundleError(f"host_rules[{index}] is invalid")
        rule_digest = sha256(rule_path.read_bytes()).hexdigest()
        if rule_digest != rule["sha256"]:
            raise WheelbipeRos2NativeBundleError(
                f"udev rule hash mismatch for {rule_path}: {rule_digest}"
            )
        if rule.get("installed") is not False:
            raise WheelbipeRos2NativeBundleError(
                "native bundle must not claim that host udev rules are installed"
            )
        host_rules.append({"path": str(rule_path), "sha256": rule_digest, "installed": False})

    overlay_values = _wheelbipe_sequence(manifest.get("asset_overlays"), name="asset_overlays")
    if len(overlay_values) != 2:
        raise WheelbipeRos2NativeBundleError("native ROS2 manifest must declare two asset overlays")
    overlay_snapshots: list[dict[str, Any]] = []
    for index, raw_overlay in enumerate(overlay_values):
        overlay = _wheelbipe_mapping(raw_overlay, name=f"asset_overlays[{index}]")
        kind = overlay.get("kind")
        source = _wheelbipe_safe_child(
            root, overlay.get("source"), name=f"asset_overlays[{index}].source"
        )
        destination = _wheelbipe_safe_child(
            source_root,
            overlay.get("destination"),
            name=f"asset_overlays[{index}].destination",
        )
        if kind == "file":
            if not source.is_file() or not isinstance(overlay.get("sha256"), str):
                raise WheelbipeRos2NativeBundleError(
                    f"asset_overlays[{index}] file source/hash is invalid"
                )
            digest = sha256(source.read_bytes()).hexdigest()
            if digest != overlay["sha256"]:
                raise WheelbipeRos2NativeBundleError(
                    f"asset overlay hash mismatch for {source}: {digest}"
                )
            count = 1
        elif kind == "directory_glob":
            pattern = overlay.get("pattern")
            if not source.is_dir() or not isinstance(pattern, str) or not pattern:
                raise WheelbipeRos2NativeBundleError(
                    f"asset_overlays[{index}] directory glob is invalid"
                )
            files = sorted(source.glob(pattern))
            if any(item.is_symlink() or not item.is_file() for item in files):
                raise WheelbipeRos2NativeBundleError(
                    f"asset_overlays[{index}] matched a non-regular file"
                )
            if len(files) != overlay.get("file_count"):
                raise WheelbipeRos2NativeBundleError(f"asset_overlays[{index}] file count mismatch")
            digest = _wheelbipe_tree_digest(source, files)
            if digest != overlay.get("tree_sha256"):
                raise WheelbipeRos2NativeBundleError(
                    f"asset overlay tree hash mismatch for {source}: {digest}"
                )
            count = len(files)
        else:
            raise WheelbipeRos2NativeBundleError(
                f"asset_overlays[{index}].kind must be 'file' or 'directory_glob'"
            )
        overlay_snapshots.append(
            {
                "kind": kind,
                "source": str(source),
                "destination": str(destination.relative_to(source_root)),
                "file_count": count,
                "sha256": digest,
            }
        )

    native_runtime = _wheelbipe_mapping(manifest.get("native_runtime"), name="native_runtime")
    if tuple(native_runtime.get("python_modules", ())) != WHEELBIPE_ROS2_NATIVE_REQUIRED_MODULES:
        raise WheelbipeRos2NativeBundleError("native_runtime.python_modules is incomplete")
    if tuple(native_runtime.get("executables", ())) != WHEELBIPE_ROS2_NATIVE_REQUIRED_EXECUTABLES:
        raise WheelbipeRos2NativeBundleError("native_runtime.executables is incomplete")
    if native_runtime.get("platform") != WHEELBIPE_ROS2_NATIVE_PLATFORM:
        raise WheelbipeRos2NativeBundleError(
            f"native_runtime.platform must be {WHEELBIPE_ROS2_NATIVE_PLATFORM!r}"
        )
    dependency_lock = _wheelbipe_safe_child(
        root, native_runtime.get("dependency_lock"), name="native_runtime.dependency_lock"
    )
    if not dependency_lock.is_file() or not isinstance(
        native_runtime.get("dependency_lock_sha256"), str
    ):
        raise WheelbipeRos2NativeBundleError("native runtime dependency lock is invalid")
    dependency_lock_digest = sha256(dependency_lock.read_bytes()).hexdigest()
    if dependency_lock_digest != native_runtime["dependency_lock_sha256"]:
        raise WheelbipeRos2NativeBundleError(
            f"native runtime dependency lock hash mismatch: {dependency_lock_digest}"
        )
    dependency_values = OmegaConf.to_container(OmegaConf.load(dependency_lock), resolve=True)
    if not isinstance(dependency_values, dict):
        raise WheelbipeRos2NativeBundleError("native runtime dependency lock must be a mapping")
    if dependency_values.get("platform") != WHEELBIPE_ROS2_NATIVE_PLATFORM:
        raise WheelbipeRos2NativeBundleError(
            "native runtime dependency lock has an incompatible platform"
        )
    expected_dependency_versions = {"mujoco": "3.5.0", "onnxruntime": "1.20.0"}
    try:
        dependencies = cast(dict[str, Any], dependency_values["dependencies"])
        dependency_versions = {
            name: cast(dict[str, Any], dependencies[name])["version"]
            for name in expected_dependency_versions
        }
    except (KeyError, TypeError) as exc:
        raise WheelbipeRos2NativeBundleError(
            "native runtime dependency lock is missing pinned dependencies"
        ) from exc
    if dependency_versions != expected_dependency_versions:
        raise WheelbipeRos2NativeBundleError(
            "native runtime dependency versions do not match the pinned source"
        )

    serial_contract = _wheelbipe_mapping(manifest.get("serial_contract"), name="serial_contract")
    expected_serial = {
        "plugin": "template_real_ros2_ctrl::RealBridge",
        "port": "/dev/wheelbipe_h7",
        "baudrate": 2_000_000,
        "reconnect_interval_ms": 1000,
        "stale_state_timeout_ms": 100,
        "joint_count": 8,
        "communicated_joint_count": 6,
        "sensors": ["imu", "dt7"],
        "output_gate": "active_and_connected_and_fresh_valid_state",
        "deactivate_safe_stop": True,
        "realtime_transport": False,
    }
    if serial_contract != expected_serial:
        raise WheelbipeRos2NativeBundleError(
            "serial_contract does not match the pinned RealBridge fail-closed boundary"
        )
    teleop_contract = _wheelbipe_mapping(manifest.get("teleop_contract"), name="teleop_contract")
    if teleop_contract != WHEELBIPE_ROS2_TELEOP_CONTRACT:
        raise WheelbipeRos2NativeBundleError(
            "teleop_contract does not match the pinned keyboard/Xbox boundary"
        )
    claims = _wheelbipe_mapping(manifest.get("claims"), name="claims")
    expected_claims = {
        "source_packaged": True,
        "native_ros2_active": False,
        "ros_graph_created": False,
        "serial_device_opened": False,
        "hardware_validated": False,
        "realtime_guarantee": False,
    }
    if claims != expected_claims:
        raise WheelbipeRos2NativeBundleError(
            "native ROS2 bundle claims must remain explicit and fail closed"
        )
    return {
        "manifest": str(path),
        "source_root": str(source_root),
        "source_revision": WHEELBIPE_DEPLOYMENT_SOURCE_REVISION,
        "source_license": "MIT",
        "license": str(license_path),
        "source_file_count": len(source_files),
        "source_bytes": sum(item.stat().st_size for item in source_files),
        "source_tree_sha256": source_digest,
        "packages": sorted(packages),
        "controller_manager": deepcopy(controller_manager_contract),
        "plugins": plugins,
        "launch_files": launch_files,
        "config_files": config_files,
        "host_rules": host_rules,
        "dependency_lock": str(dependency_lock),
        "dependency_versions": dependency_versions,
        "asset_overlays": overlay_snapshots,
        "claims": deepcopy(claims),
        "verified": True,
    }


def wheelbipe_ros2_native_runtime_probe(
    *,
    repository_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Probe optional ROS tooling without importing it or creating a graph."""

    bundle_error: str | None = None
    try:
        bundle = verify_wheelbipe_ros2_native_bundle(
            repository_root=repository_root,
            manifest_path=manifest_path,
        )
    except WheelbipeRos2NativeBundleError as exc:
        bundle = None
        bundle_error = str(exc)
    modules: dict[str, bool] = {}
    for module_name in WHEELBIPE_ROS2_NATIVE_REQUIRED_MODULES:
        try:
            modules[module_name] = find_spec(module_name) is not None
        except (AttributeError, ImportError, ValueError):
            modules[module_name] = False
    executables = {
        executable: shutil.which(executable) is not None
        for executable in WHEELBIPE_ROS2_NATIVE_REQUIRED_EXECUTABLES
    }
    missing_modules = [name for name, available in modules.items() if not available]
    missing_executables = [name for name, available in executables.items() if not available]
    detected_platform = f"{platform.system().lower()}-{platform.machine().lower()}"
    platform_supported = detected_platform == WHEELBIPE_ROS2_NATIVE_PLATFORM
    tooling_available = not missing_modules and not missing_executables and platform_supported
    return {
        "bundle_verified": bundle is not None,
        "bundle_error": bundle_error,
        "source_revision": WHEELBIPE_DEPLOYMENT_SOURCE_REVISION,
        "python_modules": modules,
        "executables": executables,
        "missing_python_modules": missing_modules,
        "missing_executables": missing_executables,
        "required_platform": WHEELBIPE_ROS2_NATIVE_PLATFORM,
        "detected_platform": detected_platform,
        "platform_supported": platform_supported,
        "optional_dependencies_available": tooling_available,
        # Presence of tools is not evidence of a built/sourced workspace or a
        # functional ROS graph.  Keep operational claims false by construction.
        "workspace_built": False,
        "workspace_sourced": False,
        "ros_graph_created": False,
        "serial_device_opened": False,
        "hardware_validated": False,
        "realtime_guarantee": False,
    }


def require_wheelbipe_ros2_native_runtime(
    *,
    repository_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the bundle and minimum external ROS tooling exist."""

    probe = wheelbipe_ros2_native_runtime_probe(
        repository_root=repository_root,
        manifest_path=manifest_path,
    )
    if not probe["bundle_verified"]:
        raise WheelbipeRos2NativeBundleError(
            f"native WheelBipe ROS2 bundle is invalid: {probe['bundle_error']}"
        )
    if not probe["optional_dependencies_available"]:
        missing = [
            *(f"Python module {name}" for name in probe["missing_python_modules"]),
            *(f"executable {name}" for name in probe["missing_executables"]),
        ]
        if not probe["platform_supported"]:
            missing.append(
                f"platform {probe['required_platform']} (detected {probe['detected_platform']})"
            )
        raise WheelbipeRos2NativeBundleError(
            "native WheelBipe ROS2 requires an external ROS 2 Humble/colcon environment; "
            "missing "
            + ", ".join(missing)
            + ". UniLab does not install ROS system dependencies, build the workspace, "
            "create a ROS graph, or open hardware from this adapter."
        )
    return probe


def materialize_wheelbipe_ros2_native_workspace(
    destination: str | Path,
    *,
    repository_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy the verified source/assets into a new, otherwise absent workspace.

    This cold-path operation only packages files.  It never invokes colcon,
    sources a setup file, launches ROS, or touches a serial device.  Existing
    destinations are rejected so callers cannot accidentally overwrite a
    workspace.
    """

    verification = verify_wheelbipe_ros2_native_bundle(
        repository_root=repository_root,
        manifest_path=manifest_path,
    )
    root, source_manifest, manifest = _wheelbipe_native_manifest_data(
        repository_root=repository_root,
        manifest_path=manifest_path,
    )
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(
            f"native ROS2 workspace destination already exists: {destination_path}"
        )
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(
            f"native ROS2 workspace parent does not exist: {destination_path.parent}"
        )
    source_root = Path(verification["source_root"])
    workspace_source = destination_path / "src"
    shutil.copytree(source_root, workspace_source)
    copied_assets = 0
    for index, raw_overlay in enumerate(
        _wheelbipe_sequence(manifest.get("asset_overlays"), name="asset_overlays")
    ):
        overlay = _wheelbipe_mapping(raw_overlay, name=f"asset_overlays[{index}]")
        source = _wheelbipe_safe_child(
            root, overlay.get("source"), name=f"asset_overlays[{index}].source"
        )
        target = _wheelbipe_safe_child(
            workspace_source,
            overlay.get("destination"),
            name=f"asset_overlays[{index}].destination",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if overlay["kind"] == "file":
            shutil.copy2(source, target)
            copied_assets += 1
        else:
            target.mkdir(parents=True, exist_ok=True)
            for asset in sorted(source.glob(str(overlay["pattern"]))):
                shutil.copy2(asset, target / asset.name)
                copied_assets += 1
    shutil.copy2(source_manifest, destination_path / "wheelbipe_v14_native_manifest.yaml")
    license_source = _wheelbipe_safe_child(
        root,
        _wheelbipe_mapping(manifest.get("bundle"), name="bundle").get("license_file"),
        name="bundle.license_file",
    )
    shutil.copy2(license_source, destination_path / "LICENSE")
    dependency_lock_source = _wheelbipe_safe_child(
        root,
        _wheelbipe_mapping(manifest.get("native_runtime"), name="native_runtime").get(
            "dependency_lock"
        ),
        name="native_runtime.dependency_lock",
    )
    shutil.copy2(dependency_lock_source, destination_path / "dependencies.lock")
    host_rules_destination = destination_path / "udev"
    host_rules_destination.mkdir()
    for raw_rule in _wheelbipe_sequence(manifest.get("host_rules"), name="host_rules"):
        rule = _wheelbipe_mapping(raw_rule, name="host_rules entry")
        rule_source = _wheelbipe_safe_child(root, rule.get("path"), name="host_rules.path")
        shutil.copy2(rule_source, host_rules_destination / rule_source.name)
    return {
        **verification,
        "workspace": str(destination_path),
        "workspace_source": str(workspace_source),
        "copied_asset_files": copied_assets,
        "host_rules_installed": False,
        "workspace_built": False,
        "workspace_sourced": False,
        "ros_graph_created": False,
    }


def wheelbipe_ros2_runtime_available() -> bool:
    """Return whether the optional ROS 2 Python runtime is importable.

    The owner adapter intentionally remains usable when this returns
    ``False``.  This probe only consults package metadata and never imports a
    ROS module or creates a DDS context, which keeps it safe for cold-path
    diagnostics and tests.
    """

    return find_spec("rclpy") is not None


def wheelbipe_ros2_contract_snapshot(
    config: "WheelbipeRos2ControllerConfig | None" = None,
) -> dict[str, Any]:
    """Return a JSON-compatible snapshot of the no-ROS source boundary.

    ``config`` is optional so callers can inspect the source API without
    loading YAML.  When supplied, its validated timing/parameter values are
    included under ``controller``.  The returned object is detached from the
    module-level dictionaries and can therefore be serialized or annotated by
    a CLI without mutating global contract data.
    """

    snapshot: dict[str, Any] = {
        "api": deepcopy(WHEELBIPE_ROS2_API_CONTRACT),
        "topics": deepcopy(WHEELBIPE_ROS2_TOPICS),
        "state_ids": dict(WHEELBIPE_ROS2_STATE_IDS),
        "protocol": {
            "header": list(WHEELBIPE_REAL_HEADER),
            "end": list(WHEELBIPE_REAL_END),
            "command_packet_size": WHEELBIPE_REAL_COMMAND_PACKET_SIZE,
            "state_packet_size": WHEELBIPE_REAL_STATE_PACKET_SIZE,
            "crc_polynomial": "0x8005",
            "crc_initial": "0xffff",
        },
        "teleop": deepcopy(WHEELBIPE_ROS2_TELEOP_CONTRACT),
        "native_source_bundle": {
            "manifest": WHEELBIPE_ROS2_NATIVE_MANIFEST_RELATIVE_PATH.as_posix(),
            "source_root": WHEELBIPE_ROS2_NATIVE_SOURCE_RELATIVE_PATH.as_posix(),
            "source_file_count": WHEELBIPE_ROS2_NATIVE_SOURCE_FILE_COUNT,
            "source_tree_sha256": WHEELBIPE_ROS2_NATIVE_SOURCE_TREE_SHA256,
            "source_packaged": True,
            "workspace_built": False,
            "workspace_sourced": False,
        },
        "source_revision": WHEELBIPE_DEPLOYMENT_SOURCE_REVISION,
        "training_source_revision": WHEELBIPE_TRAINING_SOURCE_REVISION,
        "ros2_runtime_available": wheelbipe_ros2_runtime_available(),
        # These flags describe this owner implementation, not the optional
        # ROS package probe above.  Even on a machine where ``rclpy`` happens
        # to be installed, the adapter never creates a node/DDS graph,
        # registers a ros2_control plugin, or opens a serial transport.
        "native_ros2": False,
        "ros_graph": False,
        "serial_io": False,
        "realtime_guarantee": False,
    }
    if config is not None:
        validated = config.validate()
        snapshot["controller"] = {
            "update_rate_hz": int(validated.update_rate_hz),
            "inference_rate_hz": int(validated.inference_frequency_hz),
            "lowlevel_output_mode": validated.lowlevel_output_mode,
            "action_filter_type": validated.action_filter_type,
            "observation_delay_steps": int(validated.observation_delay_steps),
            "use_dt7": bool(validated.use_dt7),
        }
    return snapshot


class WheelbipeRos2ControllerState(IntEnum):
    """State IDs published by the source ``current_state`` topic."""

    INIT = 0
    IDLE = 1
    PREPARE = 2
    RL = 3


# Short aliases are useful to callers porting source code that used the C++
# enum name.  They do not create another state representation.
ControllerState = WheelbipeRos2ControllerState
WHEELBIPE_ROS2_STATE_IDS = {state.name: int(state) for state in WheelbipeRos2ControllerState}


# ---------------------------------------------------------------------------
# Small immutable command/state records
# ---------------------------------------------------------------------------


def _array(value: Any, shape: tuple[int, ...], *, name: str, dtype: Any = np.float64) -> np.ndarray:
    """Convert a public vector to an owned array with an exact shape."""

    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return np.array(result, dtype=dtype, copy=True)


def _finite_array(value: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class WheelbipeRos2MotionCommand:
    """Fields consumed from ``geometry_msgs/msg/Twist``."""

    linear_x: float = 0.0
    angular_z: float = 0.0
    timestamp: float | None = None


@dataclass(frozen=True)
class WheelbipeRos2HeightCommand:
    value: float = 0.22
    timestamp: float | None = None


@dataclass(frozen=True)
class WheelbipeRos2StateCommand:
    state: int = int(WheelbipeRos2ControllerState.IDLE)


@dataclass(frozen=True)
class WheelbipeRos2Dt7Command:
    """The ordinary DT7 fields exposed by the source RealBridge.

    The source intentionally omits the old ``jump`` field.  Keeping this
    record to exactly four values prevents accidentally reintroducing it.
    """

    state: int = int(WheelbipeRos2ControllerState.IDLE)
    linear_x: float = 0.0
    angular_z: float = 0.0
    height: float = 0.22


@dataclass
class WheelbipeRos2RobotState:
    """Eight-joint state and IMU sample consumed by the Python controller.

    ``orientation_xyzw`` follows ROS sensor message order.  The source
    controller internally changes it to ``wxyz`` before projecting gravity.
    ``projected_gravity_b`` is an optional owner-only shortcut for reconstructed
    simulator observations; when present it is validated and used directly.
    """

    positions: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    efforts: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    orientation_xyzw: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    timestamp: float = 0.0
    period: float = 1.0 / WHEELBIPE_ROS2_UPDATE_RATE_HZ
    projected_gravity_b: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.positions = _array(self.positions, (8,), name="positions")
        self.velocities = _array(self.velocities, (8,), name="velocities")
        self.efforts = _array(self.efforts, (8,), name="efforts")
        self.linear_acceleration = _array(
            self.linear_acceleration, (3,), name="linear_acceleration"
        )
        self.angular_velocity = _array(self.angular_velocity, (3,), name="angular_velocity")
        self.orientation_xyzw = _array(self.orientation_xyzw, (4,), name="orientation_xyzw")
        if self.projected_gravity_b is not None:
            self.projected_gravity_b = _array(
                self.projected_gravity_b, (3,), name="projected_gravity_b"
            )

    def validate(self) -> None:
        for name in (
            "positions",
            "velocities",
            "efforts",
            "linear_acceleration",
            "angular_velocity",
            "orientation_xyzw",
        ):
            _finite_array(getattr(self, name), name=name)
        if self.projected_gravity_b is not None:
            _finite_array(self.projected_gravity_b, name="projected_gravity_b")
        if not math.isfinite(float(self.timestamp)) or not math.isfinite(float(self.period)):
            raise ValueError("robot timestamp and period must be finite")
        if float(self.period) <= 0.0:
            raise ValueError("robot period must be positive")
        quat_norm_sq = float(np.dot(self.orientation_xyzw, self.orientation_xyzw))
        if quat_norm_sq <= 1.0e-12:
            raise ValueError("orientation quaternion norm must be positive")


@dataclass
class WheelbipeRos2JointCommand:
    """The five command interfaces plus the final-torque diagnostic."""

    positions: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    efforts: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    kp: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    final_torque: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    # ``sensor_msgs/JointState`` metadata emitted by the source writer.  Keep
    # it on the owner record so a thin ROS shim can publish without inventing
    # a second ordering contract.
    joint_names: tuple[str, ...] = WHEELBIPE_ROS2_JOINT_NAMES
    frame_id: str = "base_link"

    def __post_init__(self) -> None:
        for name in ("positions", "velocities", "efforts", "kp", "kd", "final_torque"):
            setattr(self, name, _array(getattr(self, name), (8,), name=name))
        self.joint_names = tuple(str(name) for name in self.joint_names)
        if self.joint_names != WHEELBIPE_ROS2_JOINT_NAMES:
            raise ValueError("joint_names must match the source ros2_control order exactly")

    def validate(self) -> None:
        for name in ("positions", "velocities", "efforts", "kp", "kd", "final_torque"):
            _finite_array(getattr(self, name), name=name)

    @property
    def names(self) -> tuple[str, ...]:
        """ROS ``JointState.name`` spelling alias."""

        return self.joint_names


@dataclass
class WheelbipeRos2ControlOutput:
    """Output of one controller update, suitable for an env or a ROS shim."""

    state: WheelbipeRos2ControllerState
    target_state: WheelbipeRos2ControllerState
    command: WheelbipeRos2JointCommand
    observation: np.ndarray | None = None
    raw_action: np.ndarray | None = None
    action: np.ndarray | None = None
    safe_stop: bool = False
    diagnostic: dict[str, Any] = field(default_factory=dict)

    @property
    def joint_commands(self) -> WheelbipeRos2JointCommand:
        """Source-topic naming alias for :attr:`command`."""

        return self.command


# ---------------------------------------------------------------------------
# Source parameter contract
# ---------------------------------------------------------------------------


def _tuple_default(values: Sequence[float], size: int) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != size:
        raise ValueError(f"expected {size} defaults, got {len(result)}")
    return result


@dataclass(frozen=True)
class WheelbipeRos2ControllerConfig:
    """Validated subset of ``template_ros2_controller_parameters.yaml``.

    Names use Python-friendly ``*_hz`` spellings, while
    :meth:`from_mapping` accepts the exact source parameter names too.
    """

    update_rate_hz: int = WHEELBIPE_ROS2_UPDATE_RATE_HZ
    inference_frequency_hz: int = WHEELBIPE_ROS2_INFERENCE_RATE_HZ
    motion_command_timeout_sec: float = 0.5
    motion_linear_x_min: float = -2.5
    motion_linear_x_max: float = 2.5
    motion_angular_z_min: float = -3.0
    motion_angular_z_max: float = 3.0
    default_command_height: float = 0.22
    command_height_min: float = 0.20
    command_height_max: float = 0.40
    auto_enter_rl: bool = False
    use_dt7: bool = False
    # These two source parameters only affect ROS-side diagnostics/publication
    # in the C++ controller.  The no-ROS owner keeps them in the validated
    # config so a source YAML round-trip cannot silently drop contract fields.
    print_inference_time: bool = False
    publish_network_io: bool = True
    lowlevel_output_mode: str = "hardware_pd_vel"
    action_filter_type: str = "none"
    action_filter_window: int = 3
    action_filter_alpha: float = 0.8
    joint_stiffness: tuple[float, ...] = (60.0, 60.0, 60.0, 60.0, 0.0, 0.0, 0.0, 0.0)
    joint_damping: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.0, 0.0)
    joint_action_scale: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 10.0, 10.0, 0.0, 0.0)
    joint_output_max: tuple[float, ...] = (
        50.9,
        50.9,
        50.9,
        50.9,
        9.99,
        9.99,
        1000.0,
        1000.0,
    )
    joint_output_min: tuple[float, ...] = (
        -50.9,
        -50.9,
        -50.9,
        -50.9,
        -9.99,
        -9.99,
        -1000.0,
        -1000.0,
    )
    joint_bias: tuple[float, ...] = (0.0,) * 8
    default_dof_pos: tuple[float, ...] = (0.0,) * 8
    prepare_dof_pos: tuple[float, ...] = (0.0,) * 4
    prepare_kp: tuple[float, ...] = (80.0,) * 4
    prepare_kd: tuple[float, ...] = (2.0,) * 4
    prepare_max_velocity: float = 1.0
    enable_noise: bool = False
    imu_gyro_noise_stddev: tuple[float, ...] = (0.01,) * 3
    imu_accel_noise_stddev: tuple[float, ...] = (0.01,) * 3
    joint_position_noise_stddev: float = 0.01
    joint_velocity_noise_stddev: float = 0.5
    enable_delay: bool = False
    observation_delay_steps: int = 0
    policy_input_cmd_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 5.0)
    policy_input_cmd_max: tuple[float, ...] = (100.0,) * 4
    policy_input_cmd_min: tuple[float, ...] = (-100.0,) * 4
    policy_input_ang_vel_scale: tuple[float, ...] = (0.5,) * 3
    policy_input_ang_vel_max: tuple[float, ...] = (100.0,) * 3
    policy_input_ang_vel_min: tuple[float, ...] = (-100.0,) * 3
    policy_input_gravity_scale: tuple[float, ...] = (1.0,) * 3
    policy_input_gravity_max: tuple[float, ...] = (100.0,) * 3
    policy_input_gravity_min: tuple[float, ...] = (-100.0,) * 3
    policy_input_joint_pos_scale: tuple[float, ...] = (1.0,) * 6
    policy_input_joint_pos_max: tuple[float, ...] = (100.0,) * 6
    policy_input_joint_pos_min: tuple[float, ...] = (-100.0,) * 6
    policy_input_joint_vel_scale: tuple[float, ...] = (0.1,) * 4
    policy_input_joint_vel_max: tuple[float, ...] = (100.0,) * 4
    policy_input_joint_vel_min: tuple[float, ...] = (-100.0,) * 4
    policy_input_wheel_vel_scale: tuple[float, ...] = (0.1,) * 2
    policy_input_wheel_vel_max: tuple[float, ...] = (100.0,) * 2
    policy_input_wheel_vel_min: tuple[float, ...] = (-100.0,) * 2
    policy_input_action_scale: tuple[float, ...] = (1.0,) * 6
    policy_input_action_max: tuple[float, ...] = (100.0,) * 6
    policy_input_action_min: tuple[float, ...] = (-100.0,) * 6
    rl_model_path: str = "policy/parallel/V14-35-flat-and-rotation-13k.onnx"

    # Source spellings remain available as read-only properties for adapters
    # that translate parameter names mechanically.
    @property
    def rl_inference_frequency(self) -> int:
        return self.inference_frequency_hz

    @property
    def rl_action_filter_type(self) -> str:
        return self.action_filter_type

    @property
    def rl_action_filter_window(self) -> int:
        return self.action_filter_window

    @property
    def rl_action_filter_alpha(self) -> float:
        return self.action_filter_alpha

    @property
    def rl_print_inference_time(self) -> bool:
        return self.print_inference_time

    @property
    def rl_publish_network_io(self) -> bool:
        return self.publish_network_io

    @property
    def update_rate(self) -> int:
        return self.update_rate_hz

    def validate(self) -> "WheelbipeRos2ControllerConfig":
        """Fail closed on the same dimensions/bounds checked by the C++ node."""

        integer_fields = {
            "update_rate_hz": self.update_rate_hz,
            "inference_frequency_hz": self.inference_frequency_hz,
            "action_filter_window": self.action_filter_window,
            "observation_delay_steps": self.observation_delay_steps,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if not 1 <= int(self.update_rate_hz) <= 2000:
            raise ValueError("update_rate_hz must be in [1, 2000]")
        # The source generated parameter definition bounds this independently
        # to [1, 500] Hz (rather than tying it to a user-overridden update
        # rate).  Keep that exact validation boundary here.
        if not 1 <= int(self.inference_frequency_hz) <= 500:
            raise ValueError("inference_frequency_hz must be in [1, 500]")
        if not 1 <= int(self.action_filter_window) <= 100:
            raise ValueError("action_filter_window must be in [1, 100]")
        if not 0 <= int(self.observation_delay_steps) <= 20:
            raise ValueError("observation_delay_steps must be in [0, 20]")

        for name in (
            "auto_enter_rl",
            "use_dt7",
            "print_inference_time",
            "publish_network_io",
            "enable_noise",
            "enable_delay",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

        scalar_fields = (
            "motion_command_timeout_sec",
            "motion_linear_x_min",
            "motion_linear_x_max",
            "motion_angular_z_min",
            "motion_angular_z_max",
            "default_command_height",
            "command_height_min",
            "command_height_max",
            "action_filter_alpha",
            "prepare_max_velocity",
            "joint_position_noise_stddev",
            "joint_velocity_noise_stddev",
        )
        for name in scalar_fields:
            scalar_value = float(getattr(self, name))
            if not math.isfinite(scalar_value):
                raise ValueError(f"{name} must be finite")
        if not 0.01 <= self.motion_command_timeout_sec <= 10.0:
            raise ValueError("motion_command_timeout_sec must be in [0.01, 10]")
        if not 0.0 <= self.action_filter_alpha <= 1.0:
            raise ValueError("action_filter_alpha must be in [0, 1]")
        if not 0.01 <= self.prepare_max_velocity <= 20.0:
            raise ValueError("prepare_max_velocity must be in [0.01, 20]")
        if self.joint_position_noise_stddev < 0.0 or self.joint_velocity_noise_stddev < 0.0:
            raise ValueError("joint noise standard deviations must be nonnegative")
        if self.motion_linear_x_min > self.motion_linear_x_max:
            raise ValueError("motion_linear_x_min must be <= motion_linear_x_max")
        if self.motion_angular_z_min > self.motion_angular_z_max:
            raise ValueError("motion_angular_z_min must be <= motion_angular_z_max")
        if self.command_height_min > self.command_height_max:
            raise ValueError("command_height_min must be <= command_height_max")
        if not self.command_height_min <= self.default_command_height <= self.command_height_max:
            raise ValueError("default_command_height must lie within command height bounds")
        if self.lowlevel_output_mode not in {"torque", "hardware_pd", "hardware_pd_vel"}:
            raise ValueError("lowlevel_output_mode is not supported")
        if self.action_filter_type not in {"none", "moving_avg", "lowpass"}:
            raise ValueError("action_filter_type is not supported")

        expected_lengths = {
            "joint_stiffness": 8,
            "joint_damping": 8,
            "joint_action_scale": 8,
            "joint_output_max": 8,
            "joint_output_min": 8,
            "joint_bias": 8,
            "default_dof_pos": 8,
            "prepare_dof_pos": 4,
            "prepare_kp": 4,
            "prepare_kd": 4,
            "imu_gyro_noise_stddev": 3,
            "imu_accel_noise_stddev": 3,
            "policy_input_cmd_scale": 4,
            "policy_input_cmd_max": 4,
            "policy_input_cmd_min": 4,
            "policy_input_ang_vel_scale": 3,
            "policy_input_ang_vel_max": 3,
            "policy_input_ang_vel_min": 3,
            "policy_input_gravity_scale": 3,
            "policy_input_gravity_max": 3,
            "policy_input_gravity_min": 3,
            "policy_input_joint_pos_scale": 6,
            "policy_input_joint_pos_max": 6,
            "policy_input_joint_pos_min": 6,
            "policy_input_joint_vel_scale": 4,
            "policy_input_joint_vel_max": 4,
            "policy_input_joint_vel_min": 4,
            "policy_input_wheel_vel_scale": 2,
            "policy_input_wheel_vel_max": 2,
            "policy_input_wheel_vel_min": 2,
            "policy_input_action_scale": 6,
            "policy_input_action_max": 6,
            "policy_input_action_min": 6,
        }
        nonnegative = {
            "joint_stiffness",
            "joint_damping",
            "prepare_kp",
            "prepare_kd",
            "imu_gyro_noise_stddev",
            "imu_accel_noise_stddev",
        }
        bounds: list[tuple[str, str, str]] = [
            ("joint_output_min", "joint_output_max", "joint_output"),
            ("policy_input_cmd_min", "policy_input_cmd_max", "policy_input_cmd"),
            ("policy_input_ang_vel_min", "policy_input_ang_vel_max", "policy_input_ang_vel"),
            ("policy_input_gravity_min", "policy_input_gravity_max", "policy_input_gravity"),
            ("policy_input_joint_pos_min", "policy_input_joint_pos_max", "policy_input_joint_pos"),
            ("policy_input_joint_vel_min", "policy_input_joint_vel_max", "policy_input_joint_vel"),
            ("policy_input_wheel_vel_min", "policy_input_wheel_vel_max", "policy_input_wheel_vel"),
            ("policy_input_action_min", "policy_input_action_max", "policy_input_action"),
        ]
        for name, size in expected_lengths.items():
            values = tuple(getattr(self, name))
            if len(values) != size:
                raise ValueError(f"{name} must contain exactly {size} values; got {len(values)}")
            for index, value in enumerate(values):
                if not math.isfinite(float(value)):
                    raise ValueError(f"{name}[{index}] must be finite")
                if name in nonnegative and float(value) < 0.0:
                    raise ValueError(f"{name}[{index}] must be nonnegative")
        for low_name, high_name, label in bounds:
            low = tuple(getattr(self, low_name))
            high = tuple(getattr(self, high_name))
            if any(a > b for a, b in zip(low, high, strict=True)):
                raise ValueError(f"{label}_min must be <= the corresponding maximum")
        return self

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "WheelbipeRos2ControllerConfig":
        """Build a config from source YAML or a nested owner mapping.

        Unknown keys are ignored intentionally: deployment/runtime metadata can
        live beside controller parameters without becoming accidental Python
        controller behavior.  Values are copied into tuples before validation.
        """

        current: Mapping[str, Any] = mapping
        for key in ("wheelbipe_ros2", "controller", "template_ros2_controller"):
            nested = current.get(key) if isinstance(current, Mapping) else None
            if isinstance(nested, Mapping):
                current = nested

        aliases: dict[str, str] = {
            "update_rate": "update_rate_hz",
            "rl_inference_frequency": "inference_frequency_hz",
            "rl_action_filter_type": "action_filter_type",
            "rl_action_filter_window": "action_filter_window",
            "rl_action_filter_alpha": "action_filter_alpha",
            "enable_observation_delay": "enable_delay",
            "rl_print_inference_time": "print_inference_time",
            "rl_publish_network_io": "publish_network_io",
        }
        fields_by_name = {field_name for field_name in cls.__dataclass_fields__}
        values: dict[str, Any] = {}
        for raw_name, raw_value in current.items():
            name = aliases.get(str(raw_name), str(raw_name))
            if name not in fields_by_name:
                continue
            if name in {
                field_name
                for field_name, field_info in cls.__dataclass_fields__.items()
                if "tuple" in str(field_info.type)
            }:
                try:
                    raw_value = tuple(float(item) for item in raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{raw_name} must be a numeric sequence") from exc
            values[name] = raw_value
        result = cls(**values)
        return result.validate()


def load_wheelbipe_ros2_config(path: str | Path | None = None) -> WheelbipeRos2ControllerConfig:
    """Load the checked-in deployment YAML on the cold path.

    ``OmegaConf`` is imported lazily so the pure packet/controller module stays
    usable in minimal Python environments and the hot update path never parses
    YAML.
    """

    if path is None:
        path = (
            Path(__file__).resolve().parents[2]
            / ".."
            / "conf"
            / "deployment"
            / ("wheelbipe_v14_ros2.yaml")
        )
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Wheelbipe ROS2 deployment config does not exist: {config_path}")
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - hydra is a project dependency
        raise RuntimeError("OmegaConf is required to load the Wheelbipe ROS2 config") from exc
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Wheelbipe ROS2 config must contain a mapping: {config_path}")
    return WheelbipeRos2ControllerConfig.from_mapping(loaded)


# ---------------------------------------------------------------------------
# Controller implementation
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), float(low)), float(high)))


def _symmetric_remainder(values: np.ndarray, period: float) -> np.ndarray:
    """Match C++ ``std::remainder`` for a vector of finite angles.

    ``numpy.remainder`` returns ``[0, period)`` for a positive period, while
    the source controller's ``positionError`` uses the symmetric interval
    around zero.  Rounding the quotient with NumPy's ties-to-even ``rint``
    mirrors the C/C++ ``remainder`` rule at *both* half-period ties (for
    example ``remainder(3*pi, 2*pi) == -pi``), which a modulo-and-subtract
    implementation gets wrong for odd positive multiples.
    """

    period = float(period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError(f"period must be finite and positive, got {period!r}")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite angles")
    quotient = np.rint(values / period)
    return values - quotient * period


class WheelbipeRos2Controller:
    """Executable Python owner for the source ROS2 controller behavior.

    The class intentionally accepts a policy callable rather than importing
    ONNX Runtime.  ``WheelbipeOnnxPolicy`` from :mod:`unilab.training.wheelbipe`
    can be passed directly, while tests and lightweight adapters can provide a
    deterministic callable.
    """

    def __init__(
        self,
        policy: Callable[[np.ndarray], np.ndarray] | Any | None = None,
        *,
        config: WheelbipeRos2ControllerConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = (config or WheelbipeRos2ControllerConfig()).validate()
        self._policy_predict: Callable[[np.ndarray], Any] | None = None
        if policy is not None:
            candidate = getattr(policy, "predict", None)
            self._policy_predict = candidate if callable(candidate) else policy
            if not callable(self._policy_predict):
                raise TypeError("policy must be callable or expose a callable predict method")
        self._rng = rng or np.random.default_rng()
        self._lifecycle = "unconfigured"
        self.current_state = WheelbipeRos2ControllerState.INIT
        self.target_state = WheelbipeRos2ControllerState.IDLE
        self._state_entry_time: float | None = None
        self._first_update = True
        self._auto_enter_rl_pending = bool(self.config.auto_enter_rl)
        self._last_update_time: float | None = None
        self._last_inference_time: float | None = None
        self.inference_count = 0
        self._last_action = np.zeros(NUM_POLICY_ACTIONS, dtype=np.float64)
        self._last_raw_action = np.zeros(NUM_POLICY_ACTIONS, dtype=np.float64)
        self._desired_positions = np.zeros(NUM_NATIVE_ACTUATORS, dtype=np.float64)
        self._prepare_positions: np.ndarray | None = None
        self._observation_delay: deque[np.ndarray] = deque()
        self._action_ma_buffers = [deque[float]() for _ in range(NUM_POLICY_ACTIONS)]
        self._action_lp_last = np.zeros(NUM_POLICY_ACTIONS, dtype=np.float64)
        self._motion = WheelbipeRos2MotionCommand()
        self._height = WheelbipeRos2HeightCommand(self.config.default_command_height)
        self._motion_received = False
        self._height_received = False
        # ROS callbacks timestamp command receipt with a steady clock, while
        # deterministic simulation callers usually pass the simulator clock
        # explicitly.  Keep the clock domain alongside each receipt so an
        # omitted timestamp does not get compared with a ROS/simulation epoch
        # (which would otherwise disable or prematurely trigger timeout).
        self._motion_timestamp_monotonic = False
        self._height_timestamp_monotonic = False
        self._last_diagnostic: dict[str, Any] = {}
        self.last_observation: np.ndarray | None = None
        self.last_raw_action: np.ndarray | None = None
        self.last_output: WheelbipeRos2ControlOutput | None = None
        self._timing_contract = {
            "update_rate_hz": int(self.config.update_rate_hz),
            "inference_rate_hz": int(self.config.inference_frequency_hz),
            "update_period_sec": 1.0 / float(self.config.update_rate_hz),
            "inference_period_sec": 1.0 / float(self.config.inference_frequency_hz),
            "source": f"wheelbipe_ros2_sim2sim@{WHEELBIPE_DEPLOYMENT_SOURCE_REVISION}",
            "ros_graph": False,
        }

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def timing_contract(self) -> dict[str, Any]:
        return dict(self._timing_contract)

    @property
    def policy_available(self) -> bool:
        return self._policy_predict is not None

    def on_init(self) -> bool:
        self._lifecycle = "unconfigured"
        self.reset_runtime()
        return True

    def on_configure(self) -> bool:
        if self._lifecycle not in {"unconfigured", "inactive"}:
            raise RuntimeError(f"cannot configure controller from {self._lifecycle}")
        self.config.validate()
        self._lifecycle = "inactive"
        return True

    def on_activate(self) -> bool:
        if self._lifecycle != "inactive":
            raise RuntimeError(f"cannot activate controller from {self._lifecycle}")
        self.reset_runtime()
        self._lifecycle = "active"
        return True

    def on_deactivate(self) -> bool:
        if self._lifecycle != "active":
            return True
        self._lifecycle = "inactive"
        self.reset_runtime()
        return True

    def on_cleanup(self) -> bool:
        if self._lifecycle == "active":
            self.on_deactivate()
        self._lifecycle = "unconfigured"
        self.reset_runtime()
        return True

    def on_error(self) -> bool:
        self._lifecycle = "error"
        self.reset_runtime()
        return True

    def on_shutdown(self) -> bool:
        self._lifecycle = "finalized"
        self.reset_runtime()
        return True

    # Source-compatible aliases for callers that use a compact lifecycle API.
    configure = on_configure
    activate = on_activate
    deactivate = on_deactivate

    def reset_runtime(self, timestamp: float | None = None) -> None:
        """Reset FSM, filters and delay state at an episode/clock boundary."""

        self.current_state = WheelbipeRos2ControllerState.INIT
        self.target_state = WheelbipeRos2ControllerState.IDLE
        self._state_entry_time = None if timestamp is None else float(timestamp)
        self._first_update = True
        self._auto_enter_rl_pending = bool(self.config.auto_enter_rl)
        self._last_update_time = None
        self._last_inference_time = None
        self.inference_count = 0
        self._last_action.fill(0.0)
        self._last_raw_action.fill(0.0)
        self._desired_positions.fill(0.0)
        self._prepare_positions = None
        self._observation_delay.clear()
        for buffer in self._action_ma_buffers:
            buffer.clear()
        self._action_lp_last.fill(0.0)
        self._motion = WheelbipeRos2MotionCommand()
        self._height = WheelbipeRos2HeightCommand(self.config.default_command_height)
        self._motion_received = False
        self._height_received = False
        self._motion_timestamp_monotonic = False
        self._height_timestamp_monotonic = False
        self.last_observation = None
        self.last_raw_action = None
        self.last_output = None
        self._last_diagnostic = {}

    reset = reset_runtime

    def set_motion_command(
        self, linear_x: float, angular_z: float, *, timestamp: float | None = None
    ) -> WheelbipeRos2MotionCommand:
        values = (float(linear_x), float(angular_z))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("motion command must contain only finite values")
        command = WheelbipeRos2MotionCommand(
            linear_x=_clamp(
                values[0], self.config.motion_linear_x_min, self.config.motion_linear_x_max
            ),
            angular_z=_clamp(
                values[1], self.config.motion_angular_z_min, self.config.motion_angular_z_max
            ),
            timestamp=self._command_time(timestamp),
        )
        self._motion = command
        self._motion_received = True
        self._motion_timestamp_monotonic = timestamp is None
        return command

    def set_height_command(self, height: float, *, timestamp: float | None = None) -> float:
        value = float(height)
        if not math.isfinite(value):
            raise ValueError("height command must be finite")
        limited = _clamp(value, self.config.command_height_min, self.config.command_height_max)
        self._height = WheelbipeRos2HeightCommand(limited, self._command_time(timestamp))
        self._height_received = True
        self._height_timestamp_monotonic = timestamp is None
        return limited

    def set_state_command(self, raw_state: int) -> WheelbipeRos2ControllerState:
        try:
            numeric = int(raw_state)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"state command must be an integer in 0..3, got {raw_state!r}"
            ) from exc
        if isinstance(raw_state, bool) or numeric != raw_state or numeric not in range(4):
            raise ValueError(f"state command must be one of 0, 1, 2, 3; got {raw_state!r}")
        target = WheelbipeRos2ControllerState(numeric)
        if target is WheelbipeRos2ControllerState.RL and not self.policy_available:
            raise ValueError("RL state requires a policy callable")
        self.target_state = target
        return target

    def resolve_state_command(self, raw_state: int) -> WheelbipeRos2ControllerState | None:
        """Resolve a source ``state_command`` value without mutating runtime.

        The C++ subscriber ignores malformed values (and ignores ``RL`` when
        the ONNX session is unavailable) rather than changing the target to a
        guessed state.  ``set_state_command`` remains the strict convenience
        API used by deterministic callers; this non-mutating resolver lets a
        ROS/message shim reproduce the source callback's fail-closed behavior.
        """

        try:
            numeric = int(raw_state)
        except (TypeError, ValueError):
            return None
        if isinstance(raw_state, bool) or numeric != raw_state or numeric not in range(4):
            return None
        target = WheelbipeRos2ControllerState(numeric)
        if target is WheelbipeRos2ControllerState.RL and not self.policy_available:
            return None
        return target

    def try_set_state_command(self, raw_state: int) -> bool:
        """Apply a source-style state command, returning ``False`` if ignored."""

        target = self.resolve_state_command(raw_state)
        if target is None:
            return False
        self.target_state = target
        return True

    def set_target_state(
        self, state: WheelbipeRos2ControllerState | int
    ) -> WheelbipeRos2ControllerState:
        """Set a target using the source ``StateMachine`` naming."""

        return self.set_state_command(int(state))

    def get_current_state(self) -> WheelbipeRos2ControllerState:
        return self.current_state

    def get_target_state(self) -> WheelbipeRos2ControllerState:
        return self.target_state

    def get_state_name(self, state: WheelbipeRos2ControllerState | int | None = None) -> str:
        value = self.current_state if state is None else WheelbipeRos2ControllerState(int(state))
        return value.name

    def set_dt7_command(
        self,
        command: WheelbipeRos2Dt7Command | Sequence[float],
        *,
        timestamp: float | None = None,
    ) -> None:
        """Apply ordinary DT7 state/velocity/yaw/height fields (no jump)."""

        if isinstance(command, WheelbipeRos2Dt7Command):
            value = command
        else:
            fields = tuple(command)
            if len(fields) != 4:
                raise ValueError(
                    "DT7 command must contain exactly state, linear_x, angular_z, height"
                )
            try:
                raw_state = float(fields[0])
            except (TypeError, ValueError) as exc:
                raise ValueError("DT7 state must be finite") from exc
            if not math.isfinite(raw_state):
                raise ValueError("DT7 state must be finite")
            # The C++ bridge reads a floating-point state interface and applies
            # ``std::llround`` before resolving 0..3 (half values round away
            # from zero).  Do not truncate toward zero here.
            rounded_state = (
                math.floor(raw_state + 0.5) if raw_state >= 0.0 else math.ceil(raw_state - 0.5)
            )
            value = WheelbipeRos2Dt7Command(
                state=int(rounded_state),
                linear_x=float(fields[1]),
                angular_z=float(fields[2]),
                height=float(fields[3]),
            )
        self.set_motion_command(value.linear_x, value.angular_z, timestamp=timestamp)
        self.set_height_command(value.height, timestamp=timestamp)
        # ``TemplateRos2Controller::updateRobotState`` resolves the DT7 state
        # through its non-throwing callback path: an out-of-range value (or an
        # RL request without an initialized policy) is logged and ignored,
        # while the velocity/height fields from the same sample remain usable.
        # Keep the strict ``set_state_command`` API for direct callers, but do
        # not turn a malformed DT7 state byte into a process-level failsafe.
        self.try_set_state_command(value.state)

    def update(
        self,
        robot_state: WheelbipeRos2RobotState,
        timestamp: float | None = None,
        period: float | None = None,
        dt7: WheelbipeRos2Dt7Command | Sequence[float] | None = None,
    ) -> WheelbipeRos2ControlOutput:
        """Run one source-equivalent controller update.

        ``timestamp`` and ``period`` are explicit to support deterministic
        simulation.  If omitted, the robot sample's values are used.  A
        backwards clock resets the policy/FSM runtime just like the ROS node.
        """

        if self._lifecycle != "active":
            raise RuntimeError("controller must be active before update")
        if not isinstance(robot_state, WheelbipeRos2RobotState):
            raise TypeError("robot_state must be WheelbipeRos2RobotState")
        try:
            robot_state.validate()
        except ValueError as exc:
            # Match the source update boundary: malformed sensor data yields a
            # safe command rather than entering the policy graph.  The C++
            # controller's ``updateRobotState`` branch writes a safe command
            # and returns an error without changing the FSM target; reserve
            # the target->IDLE request for StateRL/policy failures below.
            safe_now = float(timestamp) if timestamp is not None else float(robot_state.timestamp)
            safe_dt = float(period) if period is not None else float(robot_state.period)
            return self._failsafe(
                robot_state,
                str(exc),
                now=safe_now,
                dt=safe_dt,
                request_idle=False,
            )
        now = float(robot_state.timestamp if timestamp is None else timestamp)
        dt = float(robot_state.period if period is None else period)
        if not math.isfinite(now) or not math.isfinite(dt) or dt <= 0.0:
            return self._failsafe(
                robot_state,
                "timestamp/period is invalid",
                now=now,
                dt=dt,
                request_idle=False,
            )
        if self._last_update_time is not None and now < self._last_update_time:
            self.reset_runtime(now)
        self._last_update_time = now
        if self._state_entry_time is None:
            self._state_entry_time = now
        if self._first_update:
            self._first_update = False

        if self.config.use_dt7 and dt7 is not None:
            try:
                self.set_dt7_command(dt7, timestamp=now)
            except (TypeError, ValueError) as exc:
                return self._failsafe(robot_state, str(exc), now=now, dt=dt)
        self._expire_commands(now)

        # Source ``auto_enter_rl_pending`` waits until INIT has elapsed before
        # requesting RL.  A target command received during INIT follows the
        # same transition gate below.
        if (
            self._auto_enter_rl_pending
            and self.policy_available
            and self.current_state is not WheelbipeRos2ControllerState.INIT
        ):
            self.target_state = WheelbipeRos2ControllerState.RL
            self._auto_enter_rl_pending = False

        if self.current_state is WheelbipeRos2ControllerState.INIT:
            assert self._state_entry_time is not None
            if now - self._state_entry_time > WHEELBIPE_ROS2_INIT_HOLD_SEC:
                if self.target_state is WheelbipeRos2ControllerState.INIT:
                    self.target_state = WheelbipeRos2ControllerState.IDLE
                self._transition(self.target_state, robot_state, now)
        elif self.target_state is not self.current_state:
            self._transition(self.target_state, robot_state, now)

        try:
            if self.current_state is WheelbipeRos2ControllerState.PREPARE:
                self._run_prepare(robot_state, dt)
                output = self._write_commands(robot_state, active=True)
                action = None
                observation = None
                raw_action = None
            elif self.current_state is WheelbipeRos2ControllerState.RL:
                if self.target_state is WheelbipeRos2ControllerState.IDLE:
                    output = self._safe_commands(robot_state)
                    action = None
                    observation = None
                    raw_action = None
                else:
                    observation = self._build_observation(robot_state)
                    self.last_observation = observation.copy()
                    raw_action = self._infer_if_due(observation, now)
                    output = self._write_commands(robot_state, active=True)
                    action = self._last_action.copy()
            else:
                output = self._safe_commands(robot_state)
                action = None
                observation = None
                raw_action = None
        except (TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
            return self._failsafe(robot_state, str(exc), now=now, dt=dt)

        output.state = self.current_state
        output.target_state = self.target_state
        output.observation = None if observation is None else observation.copy()
        output.raw_action = None if raw_action is None else raw_action.copy()
        output.action = None if action is None else action.copy()
        output.diagnostic.update(
            {
                "timestamp": now,
                "period": dt,
                "controller_rate_hz": int(self.config.update_rate_hz),
                "inference_rate_hz": int(self.config.inference_frequency_hz),
                "inference_count": int(self.inference_count),
                "ros_graph": False,
                "command_timed_out": not self._motion_received,
            }
        )
        self.last_output = output
        return output

    def _command_time(self, timestamp: float | None) -> float:
        value = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(value):
            raise ValueError("command timestamp must be finite")
        return value

    def _expire_commands(self, now: float) -> None:
        timeout = float(self.config.motion_command_timeout_sec)
        if self._motion_received and (
            self._motion.timestamp is None
            or (
                (time.monotonic() if self._motion_timestamp_monotonic else now)
                - float(self._motion.timestamp)
                > timeout
            )
        ):
            self._motion = WheelbipeRos2MotionCommand(timestamp=None)
            self._motion_received = False
            self._motion_timestamp_monotonic = False
        if self._height_received and (
            self._height.timestamp is None
            or (
                (time.monotonic() if self._height_timestamp_monotonic else now)
                - float(self._height.timestamp)
                > timeout
            )
        ):
            if self.current_state in {
                WheelbipeRos2ControllerState.INIT,
                WheelbipeRos2ControllerState.IDLE,
            }:
                self._height = WheelbipeRos2HeightCommand(self.config.default_command_height)
            self._height_received = False
            self._height_timestamp_monotonic = False

    def _transition(
        self,
        target: WheelbipeRos2ControllerState,
        robot_state: WheelbipeRos2RobotState,
        now: float,
    ) -> None:
        if target is WheelbipeRos2ControllerState.RL and not self.policy_available:
            self.target_state = WheelbipeRos2ControllerState.IDLE
            target = WheelbipeRos2ControllerState.IDLE
        if target is self.current_state:
            return
        if target is WheelbipeRos2ControllerState.PREPARE:
            self._prepare_positions = robot_state.positions[:4].copy()
            # ``StatePrepare`` in the source controller owns a four-element
            # target vector (``NUM_PREPARE_JOINTS == 4``).  It writes
            # ``output_position`` only for the leg joints; wheel/spring
            # output positions therefore carry over from the state we are
            # leaving.  IDLE is the one state that refreshes every output
            # position from the measured sample, so capture that snapshot at
            # the transition boundary.  INIT keeps the reset-time values
            # (normally zero), and RL keeps its last policy/spring targets.
            if self.current_state is WheelbipeRos2ControllerState.IDLE:
                self._desired_positions[4:8] = robot_state.positions[4:8]
        if target is WheelbipeRos2ControllerState.RL:
            self._last_inference_time = None
            self._last_action.fill(0.0)
            self._desired_positions.fill(0.0)
            self._observation_delay.clear()
            for buffer in self._action_ma_buffers:
                buffer.clear()
            self._action_lp_last.fill(0.0)
        self.current_state = target
        self._state_entry_time = now

    def _run_prepare(self, robot_state: WheelbipeRos2RobotState, dt: float) -> None:
        if self._prepare_positions is None:
            self._prepare_positions = robot_state.positions[:4].copy()
        assert self._prepare_positions is not None
        target = np.asarray(self.config.prepare_dof_pos, dtype=np.float64)
        max_step = float(self.config.prepare_max_velocity) * dt
        delta = target - self._prepare_positions
        self._prepare_positions += np.clip(delta, -max_step, max_step)
        # Match source ``StatePrepare::run``: only the first four joints are
        # interpolated and assigned an output position.  Wheel/spring slots
        # intentionally remain whatever the previous FSM state produced.
        self._desired_positions[:4] = self._prepare_positions

    def _build_observation(self, robot_state: WheelbipeRos2RobotState) -> np.ndarray:
        cfg = self.config
        command = np.asarray(
            [
                self._motion.linear_x if self._motion_received else 0.0,
                0.0,
                self._motion.angular_z if self._motion_received else 0.0,
                # The source keeps the last height while RL is active even
                # after the topic timeout; only INIT/IDLE timeout handling
                # restores ``default_command_height``.  ``_height`` is always
                # initialized to that default, so it is the authoritative
                # value for the observation rather than the receipt flag.
                self._height.value,
            ],
            dtype=np.float64,
        )
        command = np.clip(
            command * np.asarray(cfg.policy_input_cmd_scale),
            np.asarray(cfg.policy_input_cmd_min),
            np.asarray(cfg.policy_input_cmd_max),
        )
        angular = np.clip(
            robot_state.angular_velocity * np.asarray(cfg.policy_input_ang_vel_scale),
            np.asarray(cfg.policy_input_ang_vel_min),
            np.asarray(cfg.policy_input_ang_vel_max),
        )
        if robot_state.projected_gravity_b is None:
            # ROS reports xyzw, while UniLab's rotation helper expects wxyz.
            xyzw = robot_state.orientation_xyzw
            quat_wxyz = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
            gravity = np.asarray(
                np_quat_apply_inverse(quat_wxyz, np.asarray([0.0, 0.0, -1.0])), dtype=np.float64
            )
        else:
            gravity = robot_state.projected_gravity_b.copy()
        gravity = np.clip(
            gravity * np.asarray(cfg.policy_input_gravity_scale),
            np.asarray(cfg.policy_input_gravity_min),
            np.asarray(cfg.policy_input_gravity_max),
        )
        leg_pos = np.clip(
            robot_state.positions[:4] * np.asarray(cfg.policy_input_joint_pos_scale)[:4],
            np.asarray(cfg.policy_input_joint_pos_min)[:4],
            np.asarray(cfg.policy_input_joint_pos_max)[:4],
        )
        # The source still runs ``scaleClamp`` for the two reserved wheel
        # position slots before writing zero.  Defaults leave them at zero,
        # but preserving their configured scale/clamps matters for an
        # alternate source-compatible parameter file (e.g. non-zero minima).
        wheel_pos = np.clip(
            np.zeros(2, dtype=np.float64) * np.asarray(cfg.policy_input_joint_pos_scale)[4:6],
            np.asarray(cfg.policy_input_joint_pos_min)[4:6],
            np.asarray(cfg.policy_input_joint_pos_max)[4:6],
        )
        leg_vel = np.clip(
            robot_state.velocities[:4] * np.asarray(cfg.policy_input_joint_vel_scale),
            np.asarray(cfg.policy_input_joint_vel_min),
            np.asarray(cfg.policy_input_joint_vel_max),
        )
        wheel_vel = np.clip(
            robot_state.velocities[4:6] * np.asarray(cfg.policy_input_wheel_vel_scale),
            np.asarray(cfg.policy_input_wheel_vel_min),
            np.asarray(cfg.policy_input_wheel_vel_max),
        )
        if cfg.enable_noise:
            angular = angular + self._rng.normal(0.0, np.asarray(cfg.imu_gyro_noise_stddev))
            gravity = gravity + self._rng.normal(0.0, np.asarray(cfg.imu_accel_noise_stddev))
            leg_pos = leg_pos + self._rng.normal(0.0, cfg.joint_position_noise_stddev, size=4)
            # The source loop applies joint-position noise to all six
            # non-spring positions, including the two reserved wheel slots.
            # Those slots are normally zero, but retaining the same noise
            # sequence matters when ``enable_noise`` is explicitly selected.
            wheel_pos = wheel_pos + self._rng.normal(0.0, cfg.joint_position_noise_stddev, size=2)
            leg_vel = leg_vel + self._rng.normal(0.0, cfg.joint_velocity_noise_stddev, size=4)
        sensor = np.concatenate((angular, gravity, leg_pos, wheel_pos, leg_vel, wheel_vel))
        if cfg.enable_delay and cfg.observation_delay_steps > 0:
            delay = int(cfg.observation_delay_steps)
            self._observation_delay.append(sensor.copy())
            if len(self._observation_delay) <= delay:
                sensor = self._observation_delay[0].copy()
            else:
                sensor = self._observation_delay.popleft()
        action_tail = np.clip(
            self._last_action * np.asarray(cfg.policy_input_action_scale),
            np.asarray(cfg.policy_input_action_min),
            np.asarray(cfg.policy_input_action_max),
        )
        result = np.concatenate(
            (command, sensor, action_tail, NORMAL_CONTROL_MODE.astype(np.float64))
        )
        if result.shape != (POLICY_OBS_DIM,):
            raise RuntimeError(f"source normal observation must be 35D, got {result.shape}")
        np.clip(result, -POLICY_OBS_CLIP, POLICY_OBS_CLIP, out=result)
        if not np.all(np.isfinite(result)):
            raise ValueError("policy observation is not finite")
        return result.astype(np.float32, copy=False)

    # Public owner-layer hook for adapters that need to inspect the exact
    # source vector without entering the FSM.  It performs the same finite
    # checks and uses only cached config/runtime state.
    def build_observation(self, robot_state: WheelbipeRos2RobotState) -> np.ndarray:
        robot_state.validate()
        return self._build_observation(robot_state)

    def _infer_if_due(self, observation: np.ndarray, now: float) -> np.ndarray:
        period = 1.0 / float(self.config.inference_frequency_hz)
        if self._last_inference_time is not None and now >= self._last_inference_time:
            if now - self._last_inference_time + 1.0e-9 < period:
                return (
                    self._last_raw_action.copy()
                    if self._last_raw_action is not None
                    else self._last_action.copy()
                )
        if self._policy_predict is None:
            raise RuntimeError("RL state has no policy callable")
        # The source ONNX graph is static ``[1, 35]``.  Use a batch-shaped
        # call first; a small fallback keeps the owner convenient for simple
        # one-vector test/callback policies without weakening output checks.
        try:
            raw_value = self._policy_predict(observation[None, :])
        except (TypeError, IndexError, ValueError):
            raw_value = self._policy_predict(observation)
        raw = np.asarray(raw_value, dtype=np.float64)
        if raw.ndim == 2 and raw.shape == (1, NUM_POLICY_ACTIONS):
            raw = raw[0]
        if raw.shape != (NUM_POLICY_ACTIONS,):
            raise ValueError(f"policy output must have shape (6,), got {raw.shape}")
        if not np.all(np.isfinite(raw)):
            raise ValueError("policy output contains non-finite values")
        filtered = np.empty(NUM_POLICY_ACTIONS, dtype=np.float64)
        for index, value in enumerate(raw):
            filtered[index] = self._filter_action(index, float(value))
        scaled = (
            filtered * np.asarray(self.config.joint_action_scale)[:6]
            + np.asarray(self.config.default_dof_pos)[:6]
        )
        if not np.all(np.isfinite(scaled)):
            raise ValueError("scaled policy action is not finite")
        self._last_raw_action = raw.copy()
        self.last_raw_action = raw.copy()
        self._last_action = filtered
        self._desired_positions[:6] = scaled
        self._last_inference_time = now
        self.inference_count += 1
        return raw

    def _filter_action(self, index: int, value: float) -> float:
        kind = self.config.action_filter_type
        if kind == "moving_avg":
            buffer = self._action_ma_buffers[index]
            buffer.append(value)
            while len(buffer) > int(self.config.action_filter_window):
                buffer.popleft()
            return float(sum(buffer) / len(buffer))
        if kind == "lowpass":
            alpha = float(self.config.action_filter_alpha)
            result = alpha * value + (1.0 - alpha) * self._action_lp_last[index]
            self._action_lp_last[index] = result
            return float(result)
        return value

    def _apply_lowlevel(self, robot_state: WheelbipeRos2RobotState) -> None:
        cfg = self.config
        # Legs use position PD; wheels use velocity targets in hardware_pd_vel;
        # springs hold measured position with zero active torque.
        self._desired_positions[6:8] = robot_state.positions[6:8]
        self._output_torque = np.zeros(8, dtype=np.float64)
        for index in range(8):
            if index < 4:
                self._output_torque[index] = (
                    cfg.joint_stiffness[index]
                    * (self._desired_positions[index] - robot_state.positions[index])
                    - cfg.joint_damping[index] * robot_state.velocities[index]
                )
            elif index < 6:
                if cfg.lowlevel_output_mode == "hardware_pd_vel":
                    self._output_torque[index] = 0.0
                elif cfg.lowlevel_output_mode == "hardware_pd":
                    self._output_torque[index] = self._desired_positions[index]
                else:
                    self._output_torque[index] = (
                        self._desired_positions[index]
                        - cfg.joint_damping[index] * robot_state.velocities[index]
                    )
            else:
                self._output_torque[index] = 0.0
        if not np.all(np.isfinite(self._output_torque)):
            raise ValueError("low-level control output is not finite")
        self._output_torque = np.clip(
            self._output_torque,
            np.asarray(cfg.joint_output_min),
            np.asarray(cfg.joint_output_max),
        )
        # StateRL applies the configured per-joint bias after limiting the
        # controller torque (the bias is part of the ros2_control effort
        # interface, not the learned action).
        self._output_torque = self._output_torque + np.asarray(cfg.joint_bias)

    def _write_commands(
        self, robot_state: WheelbipeRos2RobotState, *, active: bool
    ) -> WheelbipeRos2ControlOutput:
        cfg = self.config
        if self.current_state is WheelbipeRos2ControllerState.PREPARE:
            self._output_torque = np.zeros(8, dtype=np.float64)
            for index in range(4):
                self._output_torque[index] = (
                    cfg.prepare_kp[index]
                    * (self._desired_positions[index] - robot_state.positions[index])
                    - cfg.prepare_kd[index] * robot_state.velocities[index]
                )
            self._output_torque[6:8] = np.asarray(cfg.joint_bias)[6:8]
        else:
            self._apply_lowlevel(robot_state)
        positions = robot_state.positions.copy()
        velocities = np.zeros(8, dtype=np.float64)
        efforts = np.zeros(8, dtype=np.float64)
        kp = np.zeros(8, dtype=np.float64)
        kd = np.zeros(8, dtype=np.float64)
        hardware_pd = cfg.lowlevel_output_mode == "hardware_pd"
        hardware_pd_vel = cfg.lowlevel_output_mode == "hardware_pd_vel"
        if active:
            if hardware_pd_vel:
                positions[:] = self._desired_positions
                positions[4:6] = 0.0
                # The source controller exposes wheel velocity commands only
                # while StateRL is active.  During PREPARE the wheel target
                # is initialized from the measured position for the
                # position-command snapshot, but the velocity interface must
                # stay zero until policy inference is running; otherwise a
                # hardware_pd_vel deployment can spin wheels during the
                # standing transition.
                if self.current_state is WheelbipeRos2ControllerState.RL:
                    velocities[4:6] = self._desired_positions[4:6]
            elif hardware_pd:
                positions[:] = self._desired_positions
            else:
                velocities[:] = 0.0
                efforts[:] = self._output_torque
            if hardware_pd or hardware_pd_vel:
                kp[:] = np.asarray(cfg.joint_stiffness)
                kd[:] = np.asarray(cfg.joint_damping)
                if self.current_state is WheelbipeRos2ControllerState.PREPARE:
                    kp[:4] = np.asarray(cfg.prepare_kp)
                    kd[:4] = np.asarray(cfg.prepare_kd)
                # Spring effort is explicit in hardware modes.  Wheel effort is
                # explicit only for hardware_pd, matching the C++ write path.
                efforts[6:8] = self._output_torque[6:8]
                if hardware_pd:
                    efforts[4:6] = self._output_torque[4:6]
        else:
            positions[:] = robot_state.positions

        # ``writeControllerCommands`` computes final torque from the effort
        # feed-forward plus hardware PD, then applies the configured limit. In
        # hardware modes the effort interface receives only the correction
        # needed to make that limited value exact.
        raw_final = efforts.copy()
        if active and (hardware_pd or hardware_pd_vel):
            position_error = positions - robot_state.positions
            position_error[:4] = _symmetric_remainder(position_error[:4], 2.0 * math.pi)
            raw_final += kp * position_error + kd * (velocities - robot_state.velocities)
        raw_finite = np.nan_to_num(raw_final, nan=0.0, posinf=0.0, neginf=0.0)
        final = np.clip(
            raw_finite,
            np.asarray(cfg.joint_output_min),
            np.asarray(cfg.joint_output_max),
        )
        if active and (hardware_pd or hardware_pd_vel):
            efforts += final - raw_finite
        else:
            efforts = final
        command = WheelbipeRos2JointCommand(
            positions=positions,
            velocities=velocities,
            efforts=efforts,
            kp=kp,
            kd=kd,
            final_torque=final,
        )
        command.validate()
        return WheelbipeRos2ControlOutput(
            state=self.current_state,
            target_state=self.target_state,
            command=command,
            safe_stop=False,
            diagnostic={"mode": cfg.lowlevel_output_mode},
        )

    def _safe_commands(self, robot_state: WheelbipeRos2RobotState) -> WheelbipeRos2ControlOutput:
        # A safe command must itself be finite even when the triggering sample
        # was malformed (the source ros2_control writer uses finiteOrZero).
        command = WheelbipeRos2JointCommand(
            positions=np.nan_to_num(robot_state.positions.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        )
        return WheelbipeRos2ControlOutput(
            state=self.current_state,
            target_state=self.target_state,
            command=command,
            action=None,
            safe_stop=True,
            diagnostic={"mode": "safe_stop"},
        )

    def _failsafe(
        self,
        robot_state: WheelbipeRos2RobotState,
        reason: str,
        *,
        now: float,
        dt: float,
        request_idle: bool = True,
    ) -> WheelbipeRos2ControlOutput:
        self._last_action.fill(0.0)
        self._last_raw_action.fill(0.0)
        self.last_raw_action = None
        self.last_observation = None
        self._last_inference_time = None
        self._observation_delay.clear()
        for buffer in self._action_ma_buffers:
            buffer.clear()
        self._action_lp_last.fill(0.0)
        # ``writeSafeJointCommands`` in the source refreshes every
        # output-position interface from the finite current sample, even
        # when the sample that triggered the error was malformed.  Keep the
        # position buffer coherent so a subsequent PREPARE transition does
        # not resurrect stale wheel/spring targets.
        self._desired_positions[:] = np.nan_to_num(
            robot_state.positions.copy(), nan=0.0, posinf=0.0, neginf=0.0
        )
        # StateRL's source failSafe requests IDLE but leaves the current state
        # transition to the next state-machine tick.  Mirroring that detail
        # keeps state/current_state diagnostics useful to ROS shims while the
        # output is already a safe hold command.
        if request_idle:
            self.target_state = WheelbipeRos2ControllerState.IDLE
        output = self._safe_commands(robot_state)
        output.diagnostic.update(
            {
                "safe_stop_reason": str(reason),
                "timestamp": now,
                "period": dt,
                "ros_graph": False,
                "target_idle_requested": bool(request_idle),
            }
        )
        output.safe_stop = True
        self._last_diagnostic = dict(output.diagnostic)
        self.last_output = output
        return output


# Public owner-layer spelling for callers that refer to this implementation as
# an adapter.  Keep one class and one lifecycle state, rather than maintaining
# a wrapper that could drift from ``WheelbipeRos2Controller``.
WheelbipeRos2ControllerAdapter = WheelbipeRos2Controller


# ---------------------------------------------------------------------------
# RealBridge packet contract (pure Python, no serial I/O)
# ---------------------------------------------------------------------------


WHEELBIPE_REAL_HEADER = bytes((0xA8, 0xE6))
WHEELBIPE_REAL_END = bytes((0xC3, 0xF7))
WHEELBIPE_REAL_COMMUNICATED_JOINTS = 6
WHEELBIPE_REAL_COMMAND_PACKET_SIZE = 158
WHEELBIPE_REAL_STATE_PACKET_SIZE = 143
WHEELBIPE_REAL_RECONNECT_INTERVAL_SEC = 1.000
WHEELBIPE_REAL_STATE_TIMEOUT_SEC = 0.100


def wheelbipe_crc16(data: bytes | bytearray | memoryview, polynomial: int = 0x8005) -> int:
    """Source-compatible reflected CRC16 (initial value ``0xFFFF``)."""

    crc = 0xFFFF
    for byte in bytes(data):
        crc ^= int(byte)
        for _ in range(8):
            crc = ((crc >> 1) ^ int(polynomial)) & 0xFFFF if crc & 1 else crc >> 1
    return int(crc)


@dataclass(frozen=True)
class WheelbipeRealJointCommand:
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0
    kp: float = 0.0
    kd: float = 0.0

    @property
    def position_command(self) -> float:
        return self.position

    @property
    def velocity_command(self) -> float:
        return self.velocity

    @property
    def effort_command(self) -> float:
        return self.effort


@dataclass(frozen=True)
class WheelbipeRealDt7CommandPacket:
    state: int = 1
    linear_x: float = 0.0
    angular_z: float = 0.0
    height: float = 0.22


@dataclass(frozen=True)
class WheelbipeRealImuPacket:
    linear_acceleration: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name, size in (
            ("linear_acceleration", 3),
            ("angular_velocity", 3),
            ("orientation_xyzw", 4),
        ):
            try:
                values = tuple(float(value) for value in getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a numeric sequence") from exc
            if len(values) != size:
                raise ValueError(f"{name} must contain exactly {size} values")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        if float(np.dot(self.orientation_xyzw, self.orientation_xyzw)) <= 1.0e-12:
            raise ValueError("orientation_xyzw quaternion norm must be positive")


@dataclass(frozen=True)
class WheelbipeRealStatePacket:
    h7_timestamp: float
    pc_timestamp: float
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    joint_efforts: tuple[float, ...]
    imu: WheelbipeRealImuPacket
    dt7: WheelbipeRealDt7CommandPacket

    def __post_init__(self) -> None:
        for name in ("h7_timestamp", "pc_timestamp"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("joint_positions", "joint_velocities", "joint_efforts"):
            try:
                values = tuple(float(value) for value in getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a numeric sequence") from exc
            if len(values) != WHEELBIPE_REAL_COMMUNICATED_JOINTS:
                raise ValueError(
                    f"{name} must contain exactly {WHEELBIPE_REAL_COMMUNICATED_JOINTS} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        if not isinstance(self.imu, WheelbipeRealImuPacket):
            raise TypeError("imu must be WheelbipeRealImuPacket")
        if not isinstance(self.dt7, WheelbipeRealDt7CommandPacket):
            raise TypeError("dt7 must be WheelbipeRealDt7CommandPacket")
        state = int(self.dt7.state)
        if isinstance(self.dt7.state, bool) or state != self.dt7.state or not 0 <= state <= 255:
            raise ValueError("dt7.state must be an integer in [0, 255]")
        object.__setattr__(self.dt7, "state", state)
        for name in ("linear_x", "angular_z", "height"):
            value = float(getattr(self.dt7, name))
            if not math.isfinite(value):
                raise ValueError(f"dt7.{name} must be finite")
            object.__setattr__(self.dt7, name, value)

    def as_robot_state(
        self, *, period: float = 1.0 / WHEELBIPE_ROS2_UPDATE_RATE_HZ
    ) -> WheelbipeRos2RobotState:
        positions = np.zeros(8, dtype=np.float64)
        velocities = np.zeros(8, dtype=np.float64)
        efforts = np.zeros(8, dtype=np.float64)
        # RealBridge wraps each communicated joint to the shortest revolute
        # interval before exporting state interfaces.
        # Match the RealBridge/MuJoCo ``normalizeAngle`` fmod semantics,
        # including the sign of an exact +/-pi tie.
        positions[:6] = _symmetric_remainder(
            np.asarray(self.joint_positions, dtype=np.float64), 2.0 * math.pi
        )
        velocities[:6] = self.joint_velocities
        efforts[:6] = self.joint_efforts
        # RealBridge applies this fixed frame transform before exporting the
        # ROS IMU interfaces.  Preserve it here instead of silently treating a
        # wire packet as already being in controller coordinates.
        wire_xyzw = np.asarray(self.imu.orientation_xyzw, dtype=np.float64)
        bridge_xyzw = np.asarray(
            [-wire_xyzw[1], wire_xyzw[0], wire_xyzw[2], wire_xyzw[3]], dtype=np.float64
        )
        return WheelbipeRos2RobotState(
            positions=positions,
            velocities=velocities,
            efforts=efforts,
            linear_acceleration=np.asarray(self.imu.linear_acceleration),
            angular_velocity=np.asarray(self.imu.angular_velocity),
            orientation_xyzw=bridge_xyzw,
            timestamp=float(self.pc_timestamp),
            period=period,
        )


@dataclass(frozen=True)
class WheelbipeRealCommandPacket:
    """Decoded source ``RealMsgCommand`` payload (six communicated joints)."""

    h7_timestamp: float
    pc_timestamp: float
    commands: tuple[WheelbipeRealJointCommand, ...]
    spring_compensation: tuple[float, float]
    speed_error: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("h7_timestamp", "pc_timestamp"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        commands = tuple(self.commands)
        if len(commands) != WHEELBIPE_REAL_COMMUNICATED_JOINTS:
            raise ValueError(
                f"commands must contain exactly {WHEELBIPE_REAL_COMMUNICATED_JOINTS} joints"
            )
        if not all(isinstance(command, WheelbipeRealJointCommand) for command in commands):
            raise TypeError("commands must contain WheelbipeRealJointCommand values")
        for index, command in enumerate(commands):
            for field_name in ("position", "velocity", "effort", "kp", "kd"):
                value = float(getattr(command, field_name))
                if not math.isfinite(value):
                    raise ValueError(f"commands[{index}].{field_name} must be finite")
        object.__setattr__(self, "commands", commands)
        for name, size in (("spring_compensation", 2), ("speed_error", 3)):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != size:
                raise ValueError(f"{name} must contain exactly {size} values")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)


def _finite_float32(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
        raise ValueError(f"{name} must be finite and representable as float32")
    return value


def encode_wheelbipe_real_command_packet(
    commands: Sequence[WheelbipeRealJointCommand | Sequence[float]] | None = None,
    *,
    h7_timestamp: float = 0.0,
    pc_timestamp: float = 0.0,
    spring_compensation: Sequence[float] = (0.0, 0.0),
    speed_error: Sequence[float] = (0.0, 0.0, 0.0),
    safe_stop: bool = False,
) -> bytes:
    """Encode a 158-byte source ``RealMsgCommand`` packet.

    The two spring and three speed-error fields are retained for wire
    compatibility and default to zero, matching the source bridge.
    """

    if commands is None:
        command_values = [WheelbipeRealJointCommand() for _ in range(6)]
    else:
        if len(commands) != 6:
            raise ValueError("RealBridge command packet requires exactly six communicated joints")
        command_values = []
        for index, value in enumerate(commands):
            if isinstance(value, WheelbipeRealJointCommand):
                command_values.append(value)
            else:
                fields = tuple(value)
                if len(fields) != 5:
                    raise ValueError(f"commands[{index}] must contain five float fields")
                command_values.append(WheelbipeRealJointCommand(*map(float, fields)))
    if safe_stop:
        command_values = [WheelbipeRealJointCommand() for _ in range(6)]
    spring = tuple(float(value) for value in spring_compensation)
    speed = tuple(float(value) for value in speed_error)
    if len(spring) != 2 or len(speed) != 3:
        raise ValueError("spring_compensation requires two and speed_error requires three values")
    if safe_stop:
        # ``RealBridge::transmit(true)`` starts from a zero-initialized packed
        # struct, so compatibility/reserved fields are zeroed as well.
        spring = (0.0, 0.0)
        speed = (0.0, 0.0, 0.0)
    values: list[float] = [_finite_float32(h7_timestamp, name="h7_timestamp"), float(pc_timestamp)]
    if not math.isfinite(float(pc_timestamp)):
        raise ValueError("pc_timestamp must be finite")
    for index, command in enumerate(command_values):
        values.extend(
            _finite_float32(getattr(command, field_name), name=f"commands[{index}].{field_name}")
            for field_name in ("position", "velocity", "effort", "kp", "kd")
        )
    values.extend(_finite_float32(value, name="spring_compensation") for value in spring)
    values.extend(_finite_float32(value, name="speed_error") for value in speed)
    # Header + timestamp fields, 35 float payload fields, CRC and end markers.
    packet_without_crc = struct.pack("<BBfd35f", 0xA8, 0xE6, values[0], values[1], *values[2:])
    if len(packet_without_crc) != WHEELBIPE_REAL_COMMAND_PACKET_SIZE - 4:
        raise AssertionError(f"unexpected command prefix size {len(packet_without_crc)}")
    crc = wheelbipe_crc16(packet_without_crc)
    packet = packet_without_crc + struct.pack("<HBB", crc, 0xC3, 0xF7)
    if len(packet) != WHEELBIPE_REAL_COMMAND_PACKET_SIZE:
        raise AssertionError(f"unexpected command packet size {len(packet)}")
    return packet


def encode_wheelbipe_real_state_packet(
    packet: WheelbipeRealStatePacket | None = None,
    *,
    h7_timestamp: float = 0.0,
    pc_timestamp: float = 0.0,
    joint_positions: Sequence[float] = (0.0,) * WHEELBIPE_REAL_COMMUNICATED_JOINTS,
    joint_velocities: Sequence[float] = (0.0,) * WHEELBIPE_REAL_COMMUNICATED_JOINTS,
    joint_efforts: Sequence[float] = (0.0,) * WHEELBIPE_REAL_COMMUNICATED_JOINTS,
    linear_acceleration: Sequence[float] = (0.0, 0.0, 0.0),
    angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    orientation_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    dt7_state: int = 1,
    dt7_linear_x: float = 0.0,
    dt7_angular_z: float = 0.0,
    dt7_height: float = 0.22,
) -> bytes:
    """Encode a source-compatible 143-byte ``RealMsgState`` frame.

    ``RealBridge`` only receives this frame from H7 at runtime.  The helper is
    intentionally provided for deterministic protocol fixtures and loopback
    tests; it never opens a serial device.  Pass a decoded
    :class:`WheelbipeRealStatePacket` as ``packet`` for a lossless wire-format
    round trip, or use the keyword fields to construct a fixture.
    """

    if packet is None:
        try:
            state_value = int(dt7_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("dt7_state must be an integer in [0, 255]") from exc
        if isinstance(dt7_state, bool) or state_value != dt7_state or not 0 <= state_value <= 255:
            raise ValueError("dt7_state must be an integer in [0, 255]")
        packet = WheelbipeRealStatePacket(
            h7_timestamp=float(h7_timestamp),
            pc_timestamp=float(pc_timestamp),
            joint_positions=tuple(joint_positions),
            joint_velocities=tuple(joint_velocities),
            joint_efforts=tuple(joint_efforts),
            imu=WheelbipeRealImuPacket(
                linear_acceleration=cast(
                    tuple[float, float, float],
                    tuple(float(value) for value in linear_acceleration),
                ),
                angular_velocity=cast(
                    tuple[float, float, float],
                    tuple(float(value) for value in angular_velocity),
                ),
                orientation_xyzw=cast(
                    tuple[float, float, float, float],
                    tuple(float(value) for value in orientation_xyzw),
                ),
            ),
            dt7=WheelbipeRealDt7CommandPacket(
                state=state_value,
                linear_x=float(dt7_linear_x),
                angular_z=float(dt7_angular_z),
                height=float(dt7_height),
            ),
        )
    elif not isinstance(packet, WheelbipeRealStatePacket):
        raise TypeError("packet must be WheelbipeRealStatePacket or None")

    joint_values: list[float] = []
    for index in range(WHEELBIPE_REAL_COMMUNICATED_JOINTS):
        joint_values.extend(
            _finite_float32(value, name=f"joint[{index}].{field_name}")
            for field_name, value in (
                ("position", packet.joint_positions[index]),
                ("velocity", packet.joint_velocities[index]),
                ("effort", packet.joint_efforts[index]),
            )
        )
    imu_values = [
        _finite_float32(value, name=f"imu.{field_name}")
        for field_name, values in (
            ("linear_acceleration", packet.imu.linear_acceleration),
            ("angular_velocity", packet.imu.angular_velocity),
            ("orientation_xyzw", packet.imu.orientation_xyzw),
        )
        for value in values
    ]
    dt7_state_value = int(packet.dt7.state)
    if not 0 <= dt7_state_value <= 255:
        raise ValueError("dt7.state must be in [0, 255]")
    prefix = struct.pack(
        "<BBfd18f10fB3f",
        0xA8,
        0xE6,
        _finite_float32(packet.h7_timestamp, name="h7_timestamp"),
        float(packet.pc_timestamp),
        *joint_values,
        *imu_values,
        dt7_state_value,
        _finite_float32(packet.dt7.linear_x, name="dt7.linear_x"),
        _finite_float32(packet.dt7.angular_z, name="dt7.angular_z"),
        _finite_float32(packet.dt7.height, name="dt7.height"),
    )
    if len(prefix) != WHEELBIPE_REAL_STATE_PACKET_SIZE - 4:
        raise AssertionError(f"unexpected state prefix size {len(prefix)}")
    crc = wheelbipe_crc16(prefix)
    result = prefix + struct.pack("<HBB", crc, 0xC3, 0xF7)
    if len(result) != WHEELBIPE_REAL_STATE_PACKET_SIZE:
        raise AssertionError(f"unexpected state packet size {len(result)}")
    return result


def decode_wheelbipe_real_command_packet(
    payload: bytes | bytearray | memoryview, *, validate_crc: bool = True
) -> WheelbipeRealCommandPacket:
    """Decode one source 158-byte command packet without touching serial I/O."""

    data = bytes(payload)
    if len(data) != WHEELBIPE_REAL_COMMAND_PACKET_SIZE:
        raise ValueError(
            f"RealBridge command packet must be {WHEELBIPE_REAL_COMMAND_PACKET_SIZE} bytes, got {len(data)}"
        )
    if data[:2] != WHEELBIPE_REAL_HEADER or data[-2:] != WHEELBIPE_REAL_END:
        raise ValueError("RealBridge command packet markers are invalid")
    expected_crc = struct.unpack_from("<H", data, WHEELBIPE_REAL_COMMAND_PACKET_SIZE - 4)[0]
    actual_crc = wheelbipe_crc16(data[:-4])
    if validate_crc and expected_crc != actual_crc:
        raise ValueError(
            f"RealBridge command packet CRC mismatch: expected {expected_crc:#06x}, got {actual_crc:#06x}"
        )
    unpacked = struct.unpack("<BBfd35fHBB", data)
    h7_timestamp = float(unpacked[2])
    pc_timestamp = float(unpacked[3])
    floats = tuple(float(value) for value in unpacked[4 : 4 + 35])
    if not math.isfinite(h7_timestamp) or not math.isfinite(pc_timestamp):
        raise ValueError("RealBridge command timestamps must be finite")
    if not all(math.isfinite(value) for value in floats):
        raise ValueError("RealBridge command payload contains non-finite values")
    commands = tuple(
        WheelbipeRealJointCommand(*floats[index * 5 : (index + 1) * 5]) for index in range(6)
    )
    return WheelbipeRealCommandPacket(
        h7_timestamp=h7_timestamp,
        pc_timestamp=pc_timestamp,
        commands=commands,
        spring_compensation=(floats[30], floats[31]),
        speed_error=(floats[32], floats[33], floats[34]),
    )


def decode_wheelbipe_real_state_packet(
    payload: bytes | bytearray | memoryview, *, validate_crc: bool = True
) -> WheelbipeRealStatePacket:
    """Decode and validate one source 143-byte ``RealMsgState`` packet."""

    data = bytes(payload)
    if len(data) != WHEELBIPE_REAL_STATE_PACKET_SIZE:
        raise ValueError(
            f"RealBridge state packet must be {WHEELBIPE_REAL_STATE_PACKET_SIZE} bytes, got {len(data)}"
        )
    if data[:2] != WHEELBIPE_REAL_HEADER or data[-2:] != WHEELBIPE_REAL_END:
        raise ValueError("RealBridge state packet markers are invalid")
    expected_crc = struct.unpack_from("<H", data, WHEELBIPE_REAL_STATE_PACKET_SIZE - 4)[0]
    actual_crc = wheelbipe_crc16(data[:-4])
    if validate_crc and expected_crc != actual_crc:
        raise ValueError(
            f"RealBridge state packet CRC mismatch: expected {expected_crc:#06x}, got {actual_crc:#06x}"
        )
    unpacked = struct.unpack("<BBfd18f10fB3fHBB", data)
    h7_timestamp = float(unpacked[2])
    pc_timestamp = float(unpacked[3])
    joint_values = unpacked[4 : 4 + 18]
    imu_values = unpacked[22 : 22 + 10]
    dt7_values = unpacked[32 : 32 + 4]
    if not math.isfinite(h7_timestamp) or not math.isfinite(pc_timestamp):
        raise ValueError("RealBridge timestamps must be finite")
    if not all(
        math.isfinite(float(value)) for value in (*joint_values, *imu_values, *dt7_values[1:])
    ):
        raise ValueError("RealBridge state payload contains non-finite values")
    linear_acceleration: tuple[float, float, float] = (
        float(imu_values[0]),
        float(imu_values[1]),
        float(imu_values[2]),
    )
    angular_velocity: tuple[float, float, float] = (
        float(imu_values[3]),
        float(imu_values[4]),
        float(imu_values[5]),
    )
    orientation: tuple[float, float, float, float] = (
        float(imu_values[6]),
        float(imu_values[7]),
        float(imu_values[8]),
        float(imu_values[9]),
    )
    if float(np.dot(orientation, orientation)) <= 1.0e-12:
        raise ValueError("RealBridge IMU quaternion norm must be positive")
    return WheelbipeRealStatePacket(
        h7_timestamp=h7_timestamp,
        pc_timestamp=pc_timestamp,
        joint_positions=tuple(float(value) for value in joint_values[0:18:3]),
        joint_velocities=tuple(float(value) for value in joint_values[1:18:3]),
        joint_efforts=tuple(float(value) for value in joint_values[2:18:3]),
        imu=WheelbipeRealImuPacket(
            linear_acceleration=linear_acceleration,
            angular_velocity=angular_velocity,
            orientation_xyzw=orientation,
        ),
        dt7=WheelbipeRealDt7CommandPacket(
            state=int(dt7_values[0]),
            linear_x=float(dt7_values[1]),
            angular_z=float(dt7_values[2]),
            height=float(dt7_values[3]),
        ),
    )


class WheelbipeRealStateDecoder:
    """Incremental source-like decoder for noisy serial byte streams.

    No file descriptor is opened.  ``feed`` scans for the two-byte header and
    consumes one byte after an invalid candidate, matching ``RealBridge``.
    """

    def __init__(self, *, validate_crc: bool = True) -> None:
        self._buffer = bytearray()
        self.validate_crc = bool(validate_crc)

    def feed(self, payload: bytes | bytearray | memoryview) -> list[WheelbipeRealStatePacket]:
        self._buffer.extend(bytes(payload))
        packets: list[WheelbipeRealStatePacket] = []
        size = WHEELBIPE_REAL_STATE_PACKET_SIZE
        while len(self._buffer) >= size:
            marker = self._buffer.find(WHEELBIPE_REAL_HEADER)
            if marker < 0:
                del self._buffer[:-1]
                break
            if marker:
                del self._buffer[:marker]
            if len(self._buffer) < size:
                break
            candidate = bytes(self._buffer[:size])
            try:
                packet = decode_wheelbipe_real_state_packet(
                    candidate, validate_crc=self.validate_crc
                )
            except ValueError:
                del self._buffer[:1]
                continue
            packets.append(packet)
            del self._buffer[:size]
        return packets


def wheelbipe_real_state_is_fresh(
    last_valid_timestamp: float | None,
    now: float,
    *,
    timeout_sec: float = WHEELBIPE_REAL_STATE_TIMEOUT_SEC,
) -> bool:
    """Return whether a valid RealBridge state sample is within its timeout."""

    if last_valid_timestamp is None:
        return False
    now = float(now)
    timeout_sec = float(timeout_sec)
    return (
        math.isfinite(now)
        and math.isfinite(float(last_valid_timestamp))
        and math.isfinite(timeout_sec)
        and timeout_sec > 0.0
        and now >= float(last_valid_timestamp)
        and now - float(last_valid_timestamp) <= timeout_sec
    )


@dataclass
class WheelbipeRealBridgeGate:
    """Pure state gate for the native RealBridge reconnect/write contract.

    This class owns no file descriptor and performs no serial I/O.  It mirrors
    the security-relevant decisions around reconnect throttling, valid packet
    receipt, stale-state invalidation and command inhibition so they remain
    testable on hosts without ROS 2 or hardware.
    """

    reconnect_interval_sec: float = WHEELBIPE_REAL_RECONNECT_INTERVAL_SEC
    state_timeout_sec: float = WHEELBIPE_REAL_STATE_TIMEOUT_SEC
    active: bool = False
    connected: bool = False
    has_valid_state: bool = False
    last_connect_attempt: float | None = None
    last_valid_state: float | None = None
    latest_state: WheelbipeRealStatePacket | None = None
    _decoder: WheelbipeRealStateDecoder = field(default_factory=WheelbipeRealStateDecoder)

    def __post_init__(self) -> None:
        for name in ("reconnect_interval_sec", "state_timeout_sec"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        for name in ("active", "connected", "has_valid_state"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

    @staticmethod
    def _time(value: float, *, name: str = "timestamp") -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def should_attempt_connect(self, now: float) -> bool:
        """Return whether an unavailable port may be retried at ``now``."""

        timestamp = self._time(now)
        if self.connected:
            return False
        if self.last_connect_attempt is None:
            return True
        return timestamp >= self.last_connect_attempt and (
            timestamp - self.last_connect_attempt >= self.reconnect_interval_sec
        )

    def record_connection_attempt(self, *, succeeded: bool, now: float) -> None:
        """Record one actual open attempt; early/unannounced attempts are rejected."""

        if not isinstance(succeeded, bool):
            raise ValueError("succeeded must be a boolean")
        timestamp = self._time(now)
        if not self.should_attempt_connect(timestamp):
            raise RuntimeError("serial reconnect attempt is still rate limited")
        self.last_connect_attempt = timestamp
        if succeeded:
            self.connected = True
            # A connection alone never authorizes output.  The source waits
            # for a fresh CRC-valid finite state packet first.
            self.has_valid_state = False
            self.last_valid_state = None
            self.latest_state = None
            self._decoder = WheelbipeRealStateDecoder()

    def disconnect(self) -> None:
        """Invalidate transport and state exactly as ``close_connection`` does."""

        self.connected = False
        self.has_valid_state = False
        self.last_valid_state = None
        self.latest_state = None
        self._decoder = WheelbipeRealStateDecoder()

    def activate(self) -> None:
        self.active = True

    def accept_state_packet(self, packet: WheelbipeRealStatePacket, *, now: float) -> None:
        if not self.connected:
            raise RuntimeError("cannot accept a RealBridge state packet while disconnected")
        if not isinstance(packet, WheelbipeRealStatePacket):
            raise TypeError("packet must be WheelbipeRealStatePacket")
        timestamp = self._time(now)
        self.latest_state = packet
        self.last_valid_state = timestamp
        self.has_valid_state = True

    def feed_state_bytes(
        self, payload: bytes | bytearray | memoryview, *, now: float
    ) -> list[WheelbipeRealStatePacket]:
        """Decode available frames and mark only validated frames as fresh."""

        if not self.connected:
            raise RuntimeError("cannot receive RealBridge state bytes while disconnected")
        timestamp = self._time(now)
        packets = self._decoder.feed(payload)
        if packets:
            self.accept_state_packet(packets[-1], now=timestamp)
        return packets

    def state_is_fresh(self, now: float) -> bool:
        timestamp = self._time(now)
        fresh = (
            self.connected
            and self.has_valid_state
            and wheelbipe_real_state_is_fresh(
                self.last_valid_state,
                timestamp,
                timeout_sec=self.state_timeout_sec,
            )
        )
        if not fresh and self.has_valid_state:
            self.has_valid_state = False
            self.last_valid_state = None
        return bool(fresh)

    def command_permitted(self, now: float) -> bool:
        """Mirror native ``write``: active + connected + fresh valid state."""

        return self.active and self.state_is_fresh(now)

    def deactivate(self, *, now: float) -> bool:
        """Return whether native code would request a final safe-stop packet.

        The return value is only an intent flag; this pure gate never transmits
        it.  Deactivation always invalidates and closes the modeled transport.
        """

        safe_stop_requested = self.command_permitted(now)
        self.active = False
        self.disconnect()
        return safe_stop_requested

    def diagnostic(self, *, now: float) -> dict[str, Any]:
        fresh = self.state_is_fresh(now)
        return {
            "active": self.active,
            "connected": self.connected,
            "has_valid_state": self.has_valid_state,
            "state_fresh": fresh,
            "command_permitted": self.active and fresh,
            "last_connect_attempt": self.last_connect_attempt,
            "last_valid_state": self.last_valid_state,
            "serial_io": False,
            "realtime_guarantee": False,
        }


@dataclass(frozen=True)
class WheelbipeRos2XboxTeleopConfig:
    """Pinned evdev normalization used by the packaged Xbox teleop node."""

    stick_deadzone: int = 3000
    axis_abs_max: float = 32767.0
    trigger_abs_max: float = 1023.0
    trigger_deadzone: float = 2.0
    linear_x_axis_sign: float = -1.0
    angular_z_axis_sign: float = -1.0
    max_linear_x: float = 2.5
    max_angular_z: float = 3.0
    min_height: float = 0.20
    max_height: float = 0.40
    default_height: float = 0.22
    height_down_rate: float = 0.2
    height_up_rate: float = 0.2
    idle_state_id: int = 1
    rl_state_id: int = 3

    def validate(self) -> "WheelbipeRos2XboxTeleopConfig":
        if isinstance(self.stick_deadzone, bool) or self.stick_deadzone < 0:
            raise ValueError("stick_deadzone must be a non-negative integer")
        for name in (
            "axis_abs_max",
            "trigger_abs_max",
            "trigger_deadzone",
            "linear_x_axis_sign",
            "angular_z_axis_sign",
            "max_linear_x",
            "max_angular_z",
            "min_height",
            "max_height",
            "default_height",
            "height_down_rate",
            "height_up_rate",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.axis_abs_max <= 0.0 or self.trigger_abs_max <= 0.0:
            raise ValueError("axis maxima must be positive")
        if self.trigger_deadzone < 0.0:
            raise ValueError("trigger_deadzone must be non-negative")
        if self.max_linear_x < 0.0 or self.max_angular_z < 0.0:
            raise ValueError("motion maxima must be non-negative")
        if self.height_down_rate < 0.0 or self.height_up_rate < 0.0:
            raise ValueError("height rates must be non-negative")
        if self.min_height > self.max_height:
            raise ValueError("min_height must not exceed max_height")
        if not self.min_height <= self.default_height <= self.max_height:
            raise ValueError("default_height must be inside the configured height range")
        return self


@dataclass(frozen=True)
class WheelbipeRos2TeleopCommand:
    linear_x: float
    angular_z: float
    height: float
    state_command: int | None = None
    safe_stop: bool = False


def wheelbipe_xbox_teleop_command(
    *,
    left_y: float,
    right_x: float,
    left_trigger: float,
    right_trigger: float,
    current_height: float,
    dt: float,
    connected: bool,
    start_pressed: bool = False,
    record_pressed: bool = False,
    config: WheelbipeRos2XboxTeleopConfig | None = None,
) -> WheelbipeRos2TeleopCommand:
    """Apply the source Xbox mapping without opening evdev or publishing ROS."""

    cfg = (config or WheelbipeRos2XboxTeleopConfig()).validate()
    if not isinstance(connected, bool):
        raise ValueError("connected must be a boolean")
    if not isinstance(start_pressed, bool) or not isinstance(record_pressed, bool):
        raise ValueError("button states must be booleans")
    if start_pressed and record_pressed:
        raise ValueError("simultaneous START and RECORD state commands are ambiguous")
    raw_values = {
        "left_y": left_y,
        "right_x": right_x,
        "left_trigger": left_trigger,
        "right_trigger": right_trigger,
        "current_height": current_height,
        "dt": dt,
    }
    values: dict[str, float] = {}
    for name, raw_value in raw_values.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        values[name] = value
    elapsed = min(max(values["dt"], 0.0), 0.2)
    height = _clamp(values["current_height"], cfg.min_height, cfg.max_height)
    if not connected:
        return WheelbipeRos2TeleopCommand(
            linear_x=0.0,
            angular_z=0.0,
            height=height,
            safe_stop=True,
        )

    def normalize_stick(raw_value: float, sign: float, maximum: float) -> float:
        if abs(raw_value) < cfg.stick_deadzone:
            return 0.0
        return _clamp(sign * raw_value / cfg.axis_abs_max, -1.0, 1.0) * maximum

    def normalize_trigger(raw_value: float) -> float:
        if raw_value <= cfg.trigger_deadzone:
            return 0.0
        return _clamp(raw_value / cfg.trigger_abs_max, 0.0, 1.0)

    linear_x = normalize_stick(values["left_y"], cfg.linear_x_axis_sign, cfg.max_linear_x)
    angular_z = normalize_stick(values["right_x"], cfg.angular_z_axis_sign, cfg.max_angular_z)
    down = normalize_trigger(values["left_trigger"])
    up = normalize_trigger(values["right_trigger"])
    height = _clamp(
        height + (up * cfg.height_up_rate - down * cfg.height_down_rate) * elapsed,
        cfg.min_height,
        cfg.max_height,
    )
    state_command = (
        cfg.rl_state_id if start_pressed else cfg.idle_state_id if record_pressed else None
    )
    return WheelbipeRos2TeleopCommand(
        linear_x=linear_x,
        angular_z=angular_z,
        height=height,
        state_command=state_command,
        safe_stop=False,
    )


# ---------------------------------------------------------------------------
# Observation reconstruction and public exports
# ---------------------------------------------------------------------------


def wheelbipe_robot_state_from_policy_observation(
    observation: Sequence[float] | np.ndarray,
    *,
    timestamp: float = 0.0,
    period: float = 1.0 / WHEELBIPE_ROS2_UPDATE_RATE_HZ,
) -> WheelbipeRos2RobotState:
    """Reconstruct a controller sample from UniLab's normal 35D env vector.

    This helper is for the in-process CLI adapter only.  It preserves the
    source feature order and marks gravity as an already projected vector;
    it cannot recover raw ROS IMU orientation or spring sensor values from a
    policy observation.
    """

    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (POLICY_OBS_DIM,):
        raise ValueError(f"normal Wheelbipe observation must have shape (35,), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("normal Wheelbipe observation must be finite")
    positions = np.zeros(8, dtype=np.float64)
    velocities = np.zeros(8, dtype=np.float64)
    positions[:4] = values[10:14]
    velocities[:4] = values[16:20] / 0.1
    velocities[4:6] = values[20:22] / 0.1
    return WheelbipeRos2RobotState(
        positions=positions,
        velocities=velocities,
        efforts=np.zeros(8, dtype=np.float64),
        linear_acceleration=np.zeros(3, dtype=np.float64),
        angular_velocity=values[4:7] / 0.5,
        orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        timestamp=float(timestamp),
        period=float(period),
        projected_gravity_b=values[7:10].copy(),
    )


__all__ = [
    "ControllerState",
    "WHEELBIPE_REAL_COMMAND_PACKET_SIZE",
    "WHEELBIPE_REAL_COMMUNICATED_JOINTS",
    "WHEELBIPE_REAL_END",
    "WHEELBIPE_REAL_HEADER",
    "WHEELBIPE_REAL_RECONNECT_INTERVAL_SEC",
    "WHEELBIPE_REAL_STATE_PACKET_SIZE",
    "WHEELBIPE_REAL_STATE_TIMEOUT_SEC",
    "WHEELBIPE_ROS2_COMMUNICATED_JOINT_NAMES",
    "WHEELBIPE_ROS2_COMMAND_INTERFACES",
    "WHEELBIPE_ROS2_CONTROLLER_NAME",
    "WHEELBIPE_ROS2_DT7_INTERFACES",
    "WHEELBIPE_ROS2_INIT_HOLD_SEC",
    "WHEELBIPE_ROS2_INFERENCE_RATE_HZ",
    "WHEELBIPE_ROS2_JOINT_NAMES",
    "WHEELBIPE_ROS2_NAMESPACE",
    "WHEELBIPE_ROS2_NATIVE_MANIFEST_RELATIVE_PATH",
    "WHEELBIPE_ROS2_NATIVE_PLATFORM",
    "WHEELBIPE_ROS2_NATIVE_REQUIRED_EXECUTABLES",
    "WHEELBIPE_ROS2_NATIVE_REQUIRED_MODULES",
    "WHEELBIPE_ROS2_NATIVE_SOURCE_FILE_COUNT",
    "WHEELBIPE_ROS2_NATIVE_SOURCE_RELATIVE_PATH",
    "WHEELBIPE_ROS2_NATIVE_SOURCE_TREE_SHA256",
    "WHEELBIPE_ROS2_API_CONTRACT",
    "WHEELBIPE_ROS2_QOS",
    "WHEELBIPE_ROS2_SENSOR_NAMES",
    "WHEELBIPE_ROS2_STATE_IDS",
    "WHEELBIPE_ROS2_STATE_INTERFACES",
    "WHEELBIPE_ROS2_TOPICS",
    "WHEELBIPE_ROS2_TELEOP_CONTRACT",
    "WHEELBIPE_ROS2_UPDATE_RATE_HZ",
    "WHEELBIPE_DEPLOYMENT_SOURCE_REVISION",
    "WHEELBIPE_TRAINING_SOURCE_REVISION",
    "WheelbipeRealDt7CommandPacket",
    "WheelbipeRealBridgeGate",
    "WheelbipeRealCommandPacket",
    "WheelbipeRealImuPacket",
    "WheelbipeRealJointCommand",
    "WheelbipeRealStateDecoder",
    "WheelbipeRealStatePacket",
    "WheelbipeRos2Controller",
    "WheelbipeRos2ControllerAdapter",
    "WheelbipeRos2ControllerConfig",
    "WheelbipeRos2ControllerState",
    "WheelbipeRos2ControlOutput",
    "WheelbipeRos2Dt7Command",
    "WheelbipeRos2HeightCommand",
    "WheelbipeRos2JointCommand",
    "WheelbipeRos2MotionCommand",
    "WheelbipeRos2NativeBundleError",
    "WheelbipeRos2RobotState",
    "WheelbipeRos2TeleopCommand",
    "WheelbipeRos2XboxTeleopConfig",
    "decode_wheelbipe_real_state_packet",
    "decode_wheelbipe_real_command_packet",
    "encode_wheelbipe_real_command_packet",
    "encode_wheelbipe_real_state_packet",
    "load_wheelbipe_ros2_config",
    "load_wheelbipe_ros2_native_manifest",
    "materialize_wheelbipe_ros2_native_workspace",
    "require_wheelbipe_ros2_native_runtime",
    "verify_wheelbipe_ros2_native_bundle",
    "wheelbipe_crc16",
    "wheelbipe_real_state_is_fresh",
    "wheelbipe_ros2_contract_snapshot",
    "wheelbipe_ros2_native_runtime_probe",
    "wheelbipe_ros2_runtime_available",
    "wheelbipe_robot_state_from_policy_observation",
    "wheelbipe_xbox_teleop_command",
]
