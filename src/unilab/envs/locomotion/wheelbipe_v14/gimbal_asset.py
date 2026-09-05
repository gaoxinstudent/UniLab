"""Cold-path materialization of the WheelBipe V14 gimbal mechanism.

The vendored WheelBipe MJCF intentionally describes the robot without the
camera gimbal joints so the historical eight-actuator owners keep their
original ``nq=25, nu=8`` contract.  The upstream Isaac asset, however, adds a
one-DOF yaw and pitch joint plus two PD actuators.  This module materializes a
separate, temporary XML view for owners that explicitly opt into that
mechanism.  It never parses XML from ``step``/``reset`` and it keeps all
temporary files beside the source assets so relative mesh/include paths stay
valid for both MuJoCo and Motrix.

The returned paths are owned by the caller and must be cleaned up when the env
closes.  Keeping this lifecycle explicit prevents a gimbal-enabled owner from
mutating the eight-actuator source asset in-place or leaking source capability
metadata into a normal run.
"""

from __future__ import annotations

import os
import tempfile
import weakref
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Sequence

GIMBAL_YAW_JOINT = "gimbal_yaw_joint"
GIMBAL_PITCH_JOINT = "gimbal_pitch_joint"
GIMBAL_YAW_ACTUATOR = "gimbal_yaw_joint_ctrl"
GIMBAL_PITCH_ACTUATOR = "gimbal_pitch_joint_ctrl"
WHEEL_POSITION_SENSOR_NAMES: tuple[str, str] = ("wheel_pos_left", "wheel_pos_right")
WHEEL_POSITION_SENSOR_BODIES: tuple[str, str] = ("left_wheel_link", "right_wheel_link")

# The source Isaac owner has two extra scalar joints/actuators.  These values
# are used only to extend the task fragment's cold-path keyframe vectors.
GIMBAL_QPOS_WIDTH = 2
GIMBAL_CTRL_WIDTH = 2


def _cleanup_paths(paths: Sequence[str], *, suppress_errors: bool = True) -> None:
    """Unlink materialized files, including when a caller skips ``close``.

    The finalizer path must never raise during garbage collection or Python
    interpreter shutdown.  Explicit ``WheelbipeGimbalAsset.cleanup`` calls
    retain the historical fail-fast behavior for unexpected filesystem
    errors, while still attempting every path so one stale file cannot leave
    the remaining generated XML behind.
    """

    first_error: OSError | None = None
    for raw_path in reversed(tuple(paths)):
        try:
            Path(raw_path).unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None and not suppress_errors:
        raise first_error


@dataclass
class WheelbipeGimbalAsset:
    """Materialized XML paths and deterministic cleanup ownership."""

    model_file: str
    fragment_files: tuple[str, ...]
    cleanup_paths: tuple[str, ...]
    # ``weakref.finalize`` invokes the same owner cleanup when an embedding
    # caller abandons an env without an explicit ``close``.  The callback only
    # captures immutable path strings, not ``self``, so it does not create a
    # reference cycle and remains runnable during interpreter shutdown.
    _finalizer: weakref.finalize = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._finalizer = weakref.finalize(
            self,
            _cleanup_paths,
            tuple(self.cleanup_paths),
        )

    def cleanup(self) -> None:
        """Remove temporary XML files created for this owner."""

        _cleanup_paths(self.cleanup_paths, suppress_errors=False)
        # Detach after successful unlinking.  If an unexpected filesystem
        # error occurs, retaining the finalizer gives interpreter shutdown a
        # second chance to remove paths that could not be unlinked now.
        self._finalizer.detach()


def _new_sibling_path(source: Path, *, label: str) -> Path:
    fd, raw = tempfile.mkstemp(
        prefix=f".unilab_wheelbipe_{label}_",
        suffix=source.suffix or ".xml",
        dir=str(source.parent),
    )
    os.close(fd)
    return Path(raw)


def _write_tree(tree: ET.ElementTree[ET.Element], destination: Path) -> None:
    root = tree.getroot()
    ET.indent(root, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def _body_by_name(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"WheelBipe gimbal asset is missing body {name!r}")


def _joint_names(root: ET.Element) -> set[str]:
    return {str(joint.get("name")) for joint in root.iter("joint") if joint.get("name")}


def _add_gimbal_joints_and_actuators(tree: ET.ElementTree[ET.Element], source: Path) -> None:
    root = tree.getroot()
    names = _joint_names(root)
    if GIMBAL_YAW_JOINT in names or GIMBAL_PITCH_JOINT in names:
        raise ValueError(
            f"WheelBipe source XML {source} already contains one of the gimbal joints; "
            "refusing to duplicate actuator channels"
        )

    yaw_body = _body_by_name(root, "gimbal_yaw_link")
    pitch_body = _body_by_name(root, "gimbal_pitch_link")
    # The source USD uses an unconstrained yaw axis and a pitch axis normal to
    # the camera bracket.  Finite XML ranges are an owner safety bound; they do
    # not alter the public six-action policy contract.
    yaw_joint = ET.Element(
        "joint",
        {
            "name": GIMBAL_YAW_JOINT,
            "type": "hinge",
            "axis": "0 0 1",
            "armature": "0.0001",
            "limited": "true",
            "range": "-3.141592653589793 3.141592653589793",
        },
    )
    pitch_joint = ET.Element(
        "joint",
        {
            "name": GIMBAL_PITCH_JOINT,
            "type": "hinge",
            "axis": "0 1 0",
            "armature": "0.0001",
            "limited": "true",
            "range": "-1.5707963267948966 1.5707963267948966",
        },
    )
    # Isaac's ``IdealPDActuatorCfg`` stiffness/damping are active-controller
    # gains.  The WheelBipe owner below computes those torques explicitly on
    # every physics substep, so encoding the same values as MJCF joint
    # stiffness/damping would add an uncommanded passive spring/damper and
    # count both gains twice.  Startup joint-friction randomization may still
    # add its independently configured passive viscous term through the
    # public backend contract.
    # Joints must precede the site's/geoms' child elements in MJCF.  Inserting
    # at index zero also makes the generated order deterministic for Motrix.
    yaw_body.insert(0, yaw_joint)
    pitch_body.insert(0, pitch_joint)

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    ET.SubElement(
        actuator,
        "motor",
        {
            "name": GIMBAL_YAW_ACTUATOR,
            "joint": GIMBAL_YAW_JOINT,
            "ctrllimited": "true",
            "ctrlrange": "-2 2",
            "forcelimited": "true",
            "forcerange": "-2 2",
        },
    )
    ET.SubElement(
        actuator,
        "motor",
        {
            "name": GIMBAL_PITCH_ACTUATOR,
            "joint": GIMBAL_PITCH_JOINT,
            "ctrllimited": "true",
            "ctrlrange": "-10 10",
            "forcelimited": "true",
            "forcerange": "-10 10",
        },
    )


def _vector_tokens(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return str(raw).replace("\n", " ").split()


def _extend_keyframe_vectors(tree: ET.ElementTree[ET.Element], source: Path) -> None:
    root = tree.getroot()
    keyframes = list(root.findall("keyframe"))
    for keyframe in keyframes:
        for key in keyframe.findall("key"):
            qpos = _vector_tokens(key.get("qpos"))
            if qpos is not None:
                # The source V14 asset has 41 qpos values after the guide
                # chain is materialized (free base + 34 scalar joints).  Keep
                # compatibility with the pre-guide 25-value fragment because
                # the helper is also used by downstream reduced assets.
                if len(qpos) in {25, 41}:
                    qpos.extend(["0", "0"])
                elif len(qpos) not in {27, 43}:
                    raise ValueError(
                        f"WheelBipe gimbal keyframe in {source} has {len(qpos)} qpos values; "
                        "expected 25/41 (source) or 27/43 (already materialized)"
                    )
                key.set("qpos", " ".join(qpos))
            ctrl = _vector_tokens(key.get("ctrl"))
            if ctrl is not None:
                if len(ctrl) == 8:
                    ctrl.extend(["0", "0"])
                elif len(ctrl) != 10:
                    raise ValueError(
                        f"WheelBipe gimbal keyframe in {source} has {len(ctrl)} ctrl values; "
                        "expected 8 (source) or 10 (already materialized)"
                    )
                key.set("ctrl", " ".join(ctrl))


def _add_wheel_position_sensors(tree: ET.ElementTree[ET.Element], source: Path) -> None:
    """Add declared SimBackend frame-position sensors for state-machine use."""

    root = tree.getroot()
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    existing = {elem.get("name") for elem in sensor}
    for name, body in zip(WHEEL_POSITION_SENSOR_NAMES, WHEEL_POSITION_SENSOR_BODIES, strict=True):
        if name in existing:
            continue
        # ``xbody`` is the world-frame body pose used by both backend sensor
        # adapters.  Missing body names are rejected now, before any backend
        # is constructed, so an owner cannot silently lose contact signals.
        _body_by_name(root, body)
        ET.SubElement(
            sensor,
            "framepos",
            {"name": name, "objtype": "xbody", "objname": body},
        )


def _materialize_robot(
    source: Path,
    cleanup: list[str],
    *,
    add_gimbal: bool = True,
    add_wheel_position_sensors: bool = False,
) -> Path:
    tree = ET.parse(source)
    if add_gimbal:
        _add_gimbal_joints_and_actuators(tree, source)
    if add_wheel_position_sensors:
        _add_wheel_position_sensors(tree, source)
    destination = _new_sibling_path(source, label="gimbal_robot")
    try:
        _write_tree(tree, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    cleanup.append(str(destination))
    return destination


def _materialize_scene_with_robot(source: Path, robot: Path, cleanup: list[str]) -> Path:
    tree = ET.parse(source)
    root = tree.getroot()
    includes = list(root.findall("include"))
    if not includes:
        raise ValueError(f"Expected a robot <include> in WheelBipe scene {source}")
    # Replace only the include that names the robot.  Other includes (if a
    # future scene adds one) retain their source-relative paths.
    replaced = False
    for include in includes:
        include_file = include.get("file")
        if not include_file:
            continue
        candidate = (source.parent / include_file).resolve()
        if candidate.exists() and candidate.name.startswith("wheelbipeV14"):
            include.set("file", robot.name)
            replaced = True
            break
    if not replaced:
        # The current flat scene has exactly one include; fail closed if a
        # renamed/custom scene would otherwise attach the wrong XML.
        if len(includes) != 1:
            raise ValueError(f"Could not identify the robot include in WheelBipe scene {source}")
        includes[0].set("file", robot.name)
    destination = _new_sibling_path(source, label="gimbal_scene")
    try:
        _write_tree(tree, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    cleanup.append(str(destination))
    return destination


def _materialize_fragment(source: Path, cleanup: list[str]) -> Path:
    tree = ET.parse(source)
    if not tree.getroot().find("keyframe"):
        return source
    _extend_keyframe_vectors(tree, source)
    destination = _new_sibling_path(source, label="gimbal_fragment")
    try:
        _write_tree(tree, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    cleanup.append(str(destination))
    return destination


def materialize_wheelbipe_gimbal_asset(
    model_file: str,
    fragment_files: Sequence[str] = (),
) -> WheelbipeGimbalAsset:
    """Create a gimbal-capable XML view without mutating checked-in assets.

    The operation is intentionally explicit and cold-path only.  Both the
    robot XML and any scene fragment carrying a keyframe are copied beside
    their source, preserving relative mesh/include resolution for MuJoCo and
    Motrix.  Call :meth:`WheelbipeGimbalAsset.cleanup` after backend teardown.
    """

    source_model = Path(model_file).resolve()
    if not source_model.is_file():
        raise FileNotFoundError(f"WheelBipe model XML does not exist: {source_model}")
    cleanup: list[str] = []
    try:
        root = ET.parse(source_model).getroot()
        includes = list(root.findall("include"))
        if includes:
            robot_path = _materialize_robot(
                (source_model.parent / str(includes[0].get("file"))).resolve(),
                cleanup,
                add_wheel_position_sensors=True,
            )
            materialized_model = _materialize_scene_with_robot(source_model, robot_path, cleanup)
        else:
            materialized_model = _materialize_robot(
                source_model, cleanup, add_wheel_position_sensors=True
            )

        materialized_fragments = tuple(
            str(_materialize_fragment(Path(fragment).resolve(), cleanup))
            for fragment in fragment_files
        )
        return WheelbipeGimbalAsset(
            model_file=str(materialized_model),
            fragment_files=materialized_fragments,
            cleanup_paths=tuple(cleanup),
        )
    except Exception:
        for path in reversed(cleanup):
            Path(path).unlink(missing_ok=True)
        raise


def materialize_wheelbipe_state_machine_asset(
    model_file: str,
    fragment_files: Sequence[str] = (),
) -> WheelbipeGimbalAsset:
    """Materialize wheel pose sensors without adding gimbal DOFs.

    Flat-v1/Rough-v1 use the eight-actuator mechanism but need wheel geometry
    for their landing state machine.  This companion keeps their ``nq=25,
    nu=8`` contract while adding only two frame-position sensors on the cold
    path.
    """

    source_model = Path(model_file).resolve()
    if not source_model.is_file():
        raise FileNotFoundError(f"WheelBipe model XML does not exist: {source_model}")
    cleanup: list[str] = []
    try:
        root = ET.parse(source_model).getroot()
        includes = list(root.findall("include"))
        if includes:
            include_file = includes[0].get("file")
            if not include_file:
                raise ValueError(f"WheelBipe scene {source_model} has an empty robot include")
            robot_path = _materialize_robot(
                (source_model.parent / include_file).resolve(),
                cleanup,
                add_gimbal=False,
                add_wheel_position_sensors=True,
            )
            materialized_model = _materialize_scene_with_robot(source_model, robot_path, cleanup)
        else:
            tree = ET.parse(source_model)
            _add_wheel_position_sensors(tree, source_model)
            destination = _new_sibling_path(source_model, label="state_machine_robot")
            _write_tree(tree, destination)
            cleanup.append(str(destination))
            materialized_model = destination
        return WheelbipeGimbalAsset(
            model_file=str(materialized_model),
            fragment_files=tuple(str(Path(fragment).resolve()) for fragment in fragment_files),
            cleanup_paths=tuple(cleanup),
        )
    except Exception:
        for path in reversed(cleanup):
            Path(path).unlink(missing_ok=True)
        raise


__all__ = [
    "GIMBAL_PITCH_ACTUATOR",
    "GIMBAL_PITCH_JOINT",
    "GIMBAL_YAW_ACTUATOR",
    "GIMBAL_YAW_JOINT",
    "WHEEL_POSITION_SENSOR_NAMES",
    "WheelbipeGimbalAsset",
    "materialize_wheelbipe_gimbal_asset",
    "materialize_wheelbipe_state_machine_asset",
]
