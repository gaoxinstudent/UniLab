"""Executable owner-boundary checks for WheelBipe gimbal and landing modes."""

from __future__ import annotations

import gc
import weakref
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.locomotion.wheelbipe_v14 import joystick as joystick_module
from unilab.envs.locomotion.wheelbipe_v14.base import (
    WheelbipeControlConfig,
    WheelbipeGimbalConfig,
)
from unilab.envs.locomotion.wheelbipe_v14.gimbal_asset import (
    GIMBAL_PITCH_ACTUATOR,
    GIMBAL_PITCH_JOINT,
    GIMBAL_YAW_ACTUATOR,
    GIMBAL_YAW_JOINT,
    WHEEL_POSITION_SENSOR_NAMES,
    materialize_wheelbipe_gimbal_asset,
)
from unilab.envs.locomotion.wheelbipe_v14.rough import WheelbipeV14RoughEnv
from unilab.envs.locomotion.wheelbipe_v14.semantics import (
    SOURCE_V14_WHEEL_BODY_NAMES,
)
from unilab.envs.locomotion.wheelbipe_v14.state_machine import (
    WheelbipeMotionState,
    WheelbipeStateMachine,
    WheelbipeStateMachineConfig,
)
from unilab.envs.locomotion.wheelbipe_v14.variants import (
    WheelbipeDreamWaQCfg,
    WheelbipeDreamWaQPlayCfg,
    WheelbipeFlatPlayV0Cfg,
    WheelbipeFlatPlayV2Cfg,
    WheelbipeFlatV0Cfg,
    WheelbipeFlatV1Cfg,
    WheelbipeFlatV2Cfg,
    WheelbipeHIMCfg,
    WheelbipeHIMPlayCfg,
    WheelbipeNP3OCfg,
    WheelbipeNP3OPlayCfg,
    WheelbipeRoughPlayV0Cfg,
    WheelbipeRoughPlayV1Cfg,
    WheelbipeRoughV0Cfg,
    WheelbipeRoughV1Cfg,
    WheelbipeVariantEnv,
)
from unilab.utils.rotation import np_wrap_to_pi, np_yaw_from_quat

ASSET_DIR = ASSETS_ROOT_PATH / "robots" / "wheelbipe_v14_2"
SCENE_XML = ASSET_DIR / "mjcf" / "scene_flat.xml"
TASK_XML = ASSET_DIR / "locomotion_task.xml"


def test_gimbal_materializer_produces_real_joint_actuator_and_keyframe_contract() -> None:
    mujoco = pytest.importorskip("mujoco")
    materialized = materialize_wheelbipe_gimbal_asset(str(SCENE_XML), [str(TASK_XML)])
    try:
        model = mujoco.MjModel.from_xml_path(materialized.model_file)
        assert (model.nq, model.nv, model.nu) == (43, 42, 10)
        joint_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(model.njnt)
        }
        actuator_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(model.nu)
        }
        assert {GIMBAL_YAW_JOINT, GIMBAL_PITCH_JOINT}.issubset(joint_names)
        assert {GIMBAL_YAW_ACTUATOR, GIMBAL_PITCH_ACTUATOR}.issubset(actuator_names)
        for joint_name in (GIMBAL_YAW_JOINT, GIMBAL_PITCH_JOINT):
            joint_id = int(model.joint(joint_name).id)
            dof_id = int(model.jnt_dofadr[joint_id])
            # Isaac IdealPD gains are applied by the env owner.  Generated
            # MJCF joints must not add the same values again as passive
            # stiffness/damping; source joint-friction DR is a separate term.
            assert float(model.jnt_stiffness[joint_id]) == 0.0
            assert float(model.dof_damping[dof_id]) == 0.0
        assert set(WHEEL_POSITION_SENSOR_NAMES).issubset(
            {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, idx)
                for idx in range(model.nsensor)
            }
        )

        fragment_root = ET.parse(materialized.fragment_files[0]).getroot()
        key = fragment_root.find("./keyframe/key")
        assert key is not None
        assert np.fromstring(key.attrib["qpos"], sep=" ").size == 43
        assert np.fromstring(key.attrib["ctrl"], sep=" ").size == 10
    finally:
        paths = materialized.cleanup_paths
        materialized.cleanup()
        assert all(not Path(path).exists() for path in paths)


def test_gimbal_materializer_gc_finalizer_cleans_abandoned_asset() -> None:
    """Abandoned cold-path assets must not leave generated XML in the checkout."""

    materialized = materialize_wheelbipe_gimbal_asset(str(SCENE_XML), [str(TASK_XML)])
    paths = tuple(materialized.cleanup_paths)
    assert all(Path(path).is_file() for path in paths)
    reference = weakref.ref(materialized)

    # ``WheelbipeGimbalAsset`` is normally owned by an env and cleaned by
    # ``env.close``.  This exercises the defensive owner finalizer used when an
    # embedding caller drops an env during an exception or interpreter exit.
    del materialized
    gc.collect()
    try:
        assert reference() is None
        assert all(not Path(path).exists() for path in paths)
    finally:
        # Keep the test leak-free even if a runtime changes finalizer timing.
        for path in paths:
            Path(path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("gimbal_cfg", "base_heading", "expected_yaw_pos", "expected_yaw_vel"),
    [
        (
            WheelbipeGimbalConfig(
                control_mode="velocity",
                pitch_target=-0.5,
                yaw_velocity_range=(1.25, 1.25),
            ),
            np.asarray([0.2, -0.4]),
            np.asarray([0.0, 0.0]),
            np.asarray([1.25, 1.25]),
        ),
        (
            WheelbipeGimbalConfig(
                control_mode="heading_pd",
                heading_target_mode="sampled",
                pitch_target=-0.5,
                yaw_velocity_range=(1.25, 1.25),
                yaw_heading_range=(0.7, 0.7),
            ),
            np.asarray([0.2, -0.4]),
            np.asarray([0.5, 1.1]),
            np.asarray([0.0, 0.0]),
        ),
        (
            WheelbipeGimbalConfig(
                control_mode="heading_pd",
                heading_target_mode="fixed",
                fixed_heading=0.0,
                pitch_target=-0.5,
                yaw_velocity_range=(1.25, 1.25),
            ),
            np.asarray([0.2, -0.4]),
            np.asarray([-0.2, 0.4]),
            np.asarray([0.0, 0.0]),
        ),
    ],
)
def test_source_gimbal_reset_plan_writes_physical_qpos_and_qvel(
    gimbal_cfg: WheelbipeGimbalConfig,
    base_heading: np.ndarray,
    expected_yaw_pos: np.ndarray,
    expected_yaw_vel: np.ndarray,
) -> None:
    owner = object.__new__(WheelbipeVariantEnv)
    owner._source_semantics = True
    owner._np_dtype = np.dtype(np.float64)
    owner._cfg = SimpleNamespace(gimbal=gimbal_cfg)
    owner._gimbal_pos_indices = np.asarray([0, 1], dtype=np.intp)
    owner._gimbal_vel_indices = np.asarray([0, 1], dtype=np.intp)
    owner._gimbal_yaw_velocity_target = np.zeros((4,), dtype=np.float64)
    owner._gimbal_heading_target = np.zeros((4,), dtype=np.float64)
    owner._gimbal_pitch_target = np.zeros((4,), dtype=np.float64)
    env_ids = np.asarray([1, 3], dtype=np.intp)
    qpos = np.zeros((2, 9), dtype=np.float64)
    qvel = np.zeros((2, 8), dtype=np.float64)

    info = owner._apply_source_gimbal_reset_to_plan(
        env_ids,
        base_heading=base_heading,
        qpos=qpos,
        qvel=qvel,
    )

    np.testing.assert_allclose(qpos[:, 7], expected_yaw_pos)
    np.testing.assert_allclose(qpos[:, 8], -0.5)
    np.testing.assert_allclose(qvel[:, 6], expected_yaw_vel)
    np.testing.assert_allclose(qvel[:, 7], 0.0)
    np.testing.assert_allclose(info["gimbal_yaw_velocity_target"], 1.25)
    np.testing.assert_allclose(info["gimbal_pitch_target"], -0.5)


def test_heading_pd_uses_yaw_link_world_rate_not_relative_joint_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend:
        def __init__(self) -> None:
            self.body_requests: list[np.ndarray] = []

        def get_dof_pos(self) -> np.ndarray:
            return np.asarray([[0.25, 0.0]])

        def get_dof_vel(self) -> np.ndarray:
            # Relative yaw joint rate deliberately differs from the world
            # yaw-link rate below (which includes the base yaw rate).
            return np.asarray([[0.5, 0.0]])

        def get_base_quat(self) -> np.ndarray:
            return np.asarray([[1.0, 0.0, 0.0, 0.0]])

        def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
            self.body_requests.append(np.asarray(body_ids).copy())
            return np.asarray([[[0.0, 0.0, 2.5]]])

    def zero_motor_ctrl(*args: object, out: np.ndarray, **kwargs: object) -> np.ndarray:
        del args, kwargs
        out.fill(0.0)
        return out

    monkeypatch.setattr(joystick_module, "compute_wheelbipe_motor_ctrl", zero_motor_ctrl)
    owner = object.__new__(WheelbipeVariantEnv)
    owner._num_envs = 1
    owner._np_dtype = np.dtype(np.float64)
    owner._source_semantics = False
    owner._use_obs_delay = False
    owner._use_act_delay = False
    owner._gimbal_enabled = True
    owner._gimbal_control_mode = "heading_pd"
    owner._gimbal_pos_indices = np.asarray([0, 1], dtype=np.intp)
    owner._gimbal_vel_indices = np.asarray([0, 1], dtype=np.intp)
    owner._gimbal_yaw_body_ids = np.asarray([7], dtype=np.int32)
    owner._gimbal_heading_target = np.asarray([0.25])
    owner._gimbal_heading_kp = np.asarray([20.0])
    owner._gimbal_heading_kd = np.asarray([0.1])
    owner._gimbal_pitch_target = np.asarray([-0.5])
    owner._gimbal_pitch_kp = np.asarray([20.0])
    owner._gimbal_pitch_kd = np.asarray([0.5])
    owner._gimbal_yaw_velocity_target = np.zeros((1,))
    owner._gimbal_yaw_kd = np.asarray([0.5])
    owner._native_gimbal_indices = np.asarray([0, 1], dtype=np.intp)
    owner._native_leg_indices = np.zeros((0,), dtype=np.intp)
    owner._native_wheel_indices = np.zeros((0,), dtype=np.intp)
    owner._native_spring_indices = np.zeros((0,), dtype=np.intp)
    owner._leg_pos_indices = np.zeros((0,), dtype=np.intp)
    owner._leg_vel_indices = np.zeros((0,), dtype=np.intp)
    owner._wheel_vel_indices = np.zeros((0,), dtype=np.intp)
    owner._spring_pos_indices = np.zeros((0,), dtype=np.intp)
    owner._spring_vel_indices = np.zeros((0,), dtype=np.intp)
    owner._motor_kp = np.zeros((1, 0))
    owner._motor_kd = np.zeros((1, 0))
    owner._wheel_kd = np.zeros((1, 0))
    owner._spring_force_random = np.zeros((1, 0))
    owner._last_motor_ctrl = np.zeros((1, 2))
    owner._ctrl_lower = np.asarray([-10.0, -10.0])
    owner._ctrl_upper = np.asarray([10.0, 10.0])
    owner._cfg = SimpleNamespace(
        control_config=WheelbipeControlConfig(),
        gimbal=WheelbipeGimbalConfig(control_mode="heading_pd"),
    )
    owner._capture_source_contact_force = lambda backend: None
    backend = Backend()

    ctrl = owner._pre_step_motor_control(backend, np.zeros((1, 2)))

    # Heading error is zero, so source torque is purely -kd * world link
    # rate = -0.1 * 2.5.  Using relative joint qvel would incorrectly yield
    # -0.05 instead.
    # Pitch is likewise one explicit IdealPD contribution:
    # 20 * (-0.5 - 0.0) - 0.5 * 0 = -10.  The generated MJCF test above
    # establishes that no second passive spring/damper is present.
    np.testing.assert_allclose(ctrl, [[-0.25, -10.0]])
    assert len(backend.body_requests) == 1
    np.testing.assert_array_equal(backend.body_requests[0], [7])


@pytest.mark.parametrize(
    ("cfg_type", "mode", "target_mode"),
    [
        (WheelbipeFlatV0Cfg, "velocity", "sampled"),
        (WheelbipeFlatV1Cfg, "velocity", "sampled"),
        (WheelbipeFlatV2Cfg, "heading_pd", "sampled"),
        (WheelbipeFlatPlayV0Cfg, "velocity", "sampled"),
        (WheelbipeFlatPlayV2Cfg, "heading_pd", "fixed"),
        (WheelbipeRoughV0Cfg, "heading_pd", "sampled"),
        (WheelbipeRoughV1Cfg, "velocity", "sampled"),
        (WheelbipeRoughPlayV0Cfg, "heading_pd", "sampled"),
        (WheelbipeRoughPlayV1Cfg, "velocity", "sampled"),
        (WheelbipeDreamWaQCfg, "velocity", "sampled"),
        (WheelbipeDreamWaQPlayCfg, "velocity", "sampled"),
        (WheelbipeHIMCfg, "velocity", "sampled"),
        (WheelbipeHIMPlayCfg, "velocity", "sampled"),
        (WheelbipeNP3OCfg, "velocity", "sampled"),
        (WheelbipeNP3OPlayCfg, "velocity", "sampled"),
    ],
)
def test_all_exact_ids_freeze_source_gimbal_reset_mode(
    cfg_type: type,
    mode: str,
    target_mode: str,
) -> None:
    cfg = cfg_type()
    cfg.validate()
    assert cfg.gimbal.pitch_target == pytest.approx(-0.5)
    assert cfg.gimbal.control_mode == mode
    assert cfg.gimbal.heading_target_mode == target_mode

    cfg.gimbal.control_mode = "velocity" if mode == "heading_pd" else "heading_pd"
    with pytest.raises(ValueError, match="immutable source gimbal reset contract"):
        cfg.validate()


def _assert_backend_gimbal_reset_state(env: WheelbipeVariantEnv, mode: str) -> None:
    qpos = np.asarray(env._backend.get_dof_pos())
    qvel = np.asarray(env._backend.get_dof_vel())
    yaw_pos = qpos[:, env._gimbal_pos_indices[0]]
    pitch_pos = qpos[:, env._gimbal_pos_indices[1]]
    yaw_vel = qvel[:, env._gimbal_vel_indices[0]]
    pitch_vel = qvel[:, env._gimbal_vel_indices[1]]
    np.testing.assert_allclose(pitch_pos, -0.5, atol=2.0e-5)
    np.testing.assert_allclose(pitch_vel, 0.0, atol=2.0e-5)
    if mode == "heading_pd":
        base_heading = np_yaw_from_quat(np.asarray(env._backend.get_base_quat()))
        expected = np_wrap_to_pi(env._gimbal_heading_target - base_heading)
        np.testing.assert_allclose(yaw_pos, expected, atol=2.0e-5)
        np.testing.assert_allclose(yaw_vel, 0.0, atol=2.0e-5)
    else:
        np.testing.assert_allclose(yaw_pos, 0.0, atol=2.0e-5)
        np.testing.assert_allclose(
            yaw_vel,
            env._gimbal_yaw_velocity_target,
            atol=2.0e-5,
        )


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
@pytest.mark.parametrize(
    ("cfg_type", "mode"),
    [
        (WheelbipeFlatV0Cfg, "velocity"),
        (WheelbipeFlatV2Cfg, "heading_pd"),
        (WheelbipeFlatPlayV2Cfg, "heading_pd"),
    ],
)
def test_source_gimbal_physical_state_is_written_on_init_and_partial_reset(
    backend: str,
    cfg_type: type,
    mode: str,
) -> None:
    pytest.importorskip("mujoco" if backend == "mujoco" else "motrixsim")
    env = WheelbipeVariantEnv(cfg_type(), num_envs=2, backend_type=backend)
    try:
        env.init_state()
        _assert_backend_gimbal_reset_state(env, mode)
        env.reset(np.asarray([0], dtype=np.int32))
        _assert_backend_gimbal_reset_state(env, mode)
    finally:
        env.close()


def test_state_machine_airborne_landing_recovery_transitions() -> None:
    cfg = WheelbipeStateMachineConfig(
        enabled=True,
        airborne_enter_steps=2,
        landing_contact_steps=2,
        landing_hold_steps=2,
        max_airborne_steps=20,
    )
    machine = WheelbipeStateMachine(cfg, num_envs=1)
    wheel_air = np.asarray([[[0.0, -0.2, 0.4], [0.0, 0.2, 0.4]]])
    wheel_ground = np.asarray([[[0.0, -0.2, 0.06], [0.0, 0.2, 0.06]]])
    terrain = np.zeros((1,))
    machine.update(wheel_air, terrain)
    transition = machine.update(wheel_air, terrain)
    assert transition["state"][0] == WheelbipeMotionState.AIRBORNE
    transition = machine.update(wheel_ground, terrain)
    assert transition["state"][0] == WheelbipeMotionState.AIRBORNE
    transition = machine.update(wheel_ground, terrain)
    assert transition["state"][0] == WheelbipeMotionState.LANDING
    machine.update(wheel_ground, terrain)
    transition = machine.update(wheel_ground, terrain)
    assert transition["state"][0] == WheelbipeMotionState.NORMAL


@pytest.mark.parametrize(
    ("cfg_type", "env_type", "backend", "expected_actuators"),
    [
        (WheelbipeFlatV1Cfg, WheelbipeVariantEnv, "mujoco", 10),
        (WheelbipeFlatV1Cfg, WheelbipeVariantEnv, "motrix", 10),
        (WheelbipeFlatV2Cfg, WheelbipeVariantEnv, "mujoco", 10),
        (WheelbipeFlatV2Cfg, WheelbipeVariantEnv, "motrix", 10),
        (WheelbipeRoughV0Cfg, WheelbipeV14RoughEnv, "mujoco", 10),
        (WheelbipeRoughV0Cfg, WheelbipeV14RoughEnv, "motrix", 10),
        (WheelbipeRoughV1Cfg, WheelbipeV14RoughEnv, "mujoco", 10),
        (WheelbipeRoughV1Cfg, WheelbipeV14RoughEnv, "motrix", 10),
    ],
)
def test_source_variant_owner_constructs_and_steps(
    cfg_type: type, env_type: type, backend: str, expected_actuators: int
) -> None:
    pytest.importorskip("mujoco" if backend == "mujoco" else "motrixsim")
    cfg = cfg_type()
    env = env_type(cfg, num_envs=1, backend_type=backend)
    try:
        state = env.init_state()
        assert env._backend.num_actuators == expected_actuators
        assert state.obs["obs"].shape == (1, 35)
        state = env.step(np.zeros((1, 6), dtype=np.float32))
        assert np.all(np.isfinite(state.obs["obs"]))
        wheel_ids = env._backend.get_body_ids(SOURCE_V14_WHEEL_BODY_NAMES)
        force = env._backend.get_body_contact_force_norm(wheel_ids)
        assert np.asarray(force).shape == (1, 2)
        if cfg.state_machine.enabled:
            assert "state_machine_state" in state.info
            assert "state_machine_contact" in state.info
        if cfg.gimbal_spin_translate.enabled:
            assert "special_mode_id" in state.info
            assert "command_resample_generation" in state.info
    finally:
        env.close()
