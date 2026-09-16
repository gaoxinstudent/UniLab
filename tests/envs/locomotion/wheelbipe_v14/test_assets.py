"""Integrity checks for the vendored Wheelbipe model and deployment graph."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.wheelbipe_v14.base import NATIVE_ACTUATOR_NAMES
from unilab.envs.locomotion.wheelbipe_v14.joystick import (
    WheelbipeNoiseConfig,
    WheelbipeRewardConfig,
)
from unilab.envs.locomotion.wheelbipe_v14.rough import WheelbipeV14RoughEnv
from unilab.envs.locomotion.wheelbipe_v14.semantics import (
    SOURCE_V14_LEG_MASS_BODY_NAMES,
    SOURCE_V14_REWARD_SCALES,
)
from unilab.training.wheelbipe import (
    DEFAULT_WHEELBIPE_POLICY,
    WheelbipeOnnxPolicy,
    inspect_wheelbipe_onnx,
)

ASSET_DIR = ASSETS_ROOT_PATH / "robots" / "wheelbipe_v14_2"
ROBOT_XML = ASSET_DIR / "mjcf" / "wheelbipeV14_2.xml"
SCENE_XML = ASSET_DIR / "mjcf" / "scene_flat.xml"
TASK_XML = ASSET_DIR / "locomotion_task.xml"
POLICY_SHA256 = "a1244761f7ede02f8c80d076d4315a25f014df43df3f7f0d20c2ca5bcd518719"


def test_state_machine_config_owners_are_public_package_exports() -> None:
    import unilab.envs.locomotion.wheelbipe_v14 as wheelbipe_v14

    expected = {
        "WheelbipeAirborneCommandResampleConfig",
        "WheelbipeAirborneRewardConfig",
        "WheelbipeTerrainCommandConfig",
        "WheelbipeTerrainCommandProfileConfig",
    }
    assert expected <= set(wheelbipe_v14.__all__)
    assert all(getattr(wheelbipe_v14, name) is not None for name in expected)


def test_vendored_robot_assets_and_scene_are_present() -> None:
    assert ROBOT_XML.is_file()
    assert SCENE_XML.is_file()
    assert TASK_XML.is_file()
    assert DEFAULT_WHEELBIPE_POLICY.is_file()

    robot_root = ET.parse(ROBOT_XML).getroot()
    scene_root = ET.parse(SCENE_XML).getroot()
    assert scene_root.find("./include[@file='wheelbipeV14_2.xml']") is not None
    assert scene_root.find("./worldbody/geom[@name='floor']") is not None
    assert scene_root.find("./option").get("timestep") == "0.001"

    # Every mesh referenced by the pure robot description is vendored beside it.
    for mesh in robot_root.findall("./asset/mesh"):
        mesh_path = (ROBOT_XML.parent / mesh.attrib["file"]).resolve()
        assert mesh_path.is_file(), f"missing mesh asset: {mesh_path}"


def test_rough_registry_entries_use_rough_owner() -> None:
    """Every rough/play rough identifier must retain terrain-boundary logic."""

    ensure_registries()
    for name in (
        "WheelbipeV14RoughV0",
        "WheelbipeV14RoughV1",
        "WheelbipeV14RoughPlayV0",
        "WheelbipeV14RoughPlayV1",
    ):
        assert registry._envs[name].env_cls_dict["mujoco"] is WheelbipeV14RoughEnv
        assert registry._envs[name].env_cls_dict["motrix"] is WheelbipeV14RoughEnv


def test_source_variant_registry_surface_is_complete() -> None:
    """Keep the 15 upstream identifiers present without implying parity."""

    ensure_registries()
    expected = {
        "WheelbipeV14FlatV0",
        "WheelbipeV14FlatV1",
        "WheelbipeV14FlatV2",
        "WheelbipeV14FlatPlayV2",
        "WheelbipeV14RoughV0",
        "WheelbipeV14RoughV1",
        "WheelbipeV14FlatDreamWaQ",
        "WheelbipeV14FlatHIM",
        "WheelbipeV14FlatNP3OBarlow",
        "WheelbipeV14FlatPlayV0",
        "WheelbipeV14FlatDreamWaQPlay",
        "WheelbipeV14FlatHIMPlay",
        "WheelbipeV14FlatNP3OBarlowPlay",
        "WheelbipeV14RoughPlayV0",
        "WheelbipeV14RoughPlayV1",
    }
    assert expected.issubset(registry.list_registered_envs())


def test_upstream_gymnasium_ids_are_explicit_registry_aliases() -> None:
    """Source callers can keep their published task ids during migration."""

    ensure_registries()
    from unilab.envs.locomotion.wheelbipe_v14.variants import UPSTREAM_WHEELBIPE_TASK_IDS

    registered = registry.list_registered_envs()
    assert set(UPSTREAM_WHEELBIPE_TASK_IDS).issubset(registered)
    assert len(UPSTREAM_WHEELBIPE_TASK_IDS) == 15
    for task_id, (cfg_type, _) in UPSTREAM_WHEELBIPE_TASK_IDS.items():
        assert set(registered[task_id]["available_backends"]) == {"mujoco", "motrix"}
        cfg = cfg_type()
        assert cfg.training_semantics == "source_v14"
        assert cfg.reward_config is not None
        assert dict(cfg.reward_config.scales) == dict(cfg_type._SOURCE_REWARD_SCALES)
        assert cfg.noise_config == WheelbipeNoiseConfig(
            level=1.0, scale_joint_vel=1.0, scale_wheel_vel=1.0
        )
        assert cfg.debug_value_diagnosis is False
        assert np.isposinf(float(cfg.control_config.clip_actions))
        assert cfg.control_config.leg_position_target_limit == (-3.14, 3.14)
        assert float(cfg.control_config.wheel_velocity_target_limit) == pytest.approx(150.0)
        assert cfg.domain_rand.randomize_body_mass, task_id
        assert cfg.domain_rand.random_com, task_id
        assert cfg.domain_rand.randomize_body_material, task_id
        assert cfg.domain_rand.use_leg_random_start, task_id
        assert cfg.domain_rand.use_predefined_leg_random_start, task_id
        assert cfg.domain_rand.source_external_force_enabled, task_id
        assert cfg.domain_rand.com_offset_x == pytest.approx([-0.04, 0.04]), task_id
        assert cfg.domain_rand.com_offset_y == pytest.approx([-0.02, 0.02]), task_id
        assert cfg.domain_rand.com_offset_z == pytest.approx([-0.02, 0.02]), task_id
        assert cfg.domain_rand.init_yaw_range == pytest.approx([-3.14, 3.14]), task_id
        cfg.validate()

    him = registry._envs["WheelbipeV14FlatHIM"].env_cfg_cls()
    him_play = registry._envs["WheelbipeV14FlatHIMPlay"].env_cfg_cls()
    assert him.him_curriculum.enabled
    assert him.him_curriculum.reward_stage_weights == [
        {"track_height_exp": 1.0, "track_height_exp_tight": 1.0},
        {"track_height_exp": 0.8, "track_height_exp_tight": 0.6},
    ]
    assert him.him_curriculum.assist_force_z_stages == [160.0, 80.0, 0.0]
    assert him.him_curriculum.thresholds == [0.4, 0.4]
    assert him.him_curriculum.stage_min_episodes == [500, 500]
    assert him.him_curriculum.num_steps_per_env == 24
    assert him.him_curriculum.window_size == 64
    assert him.him_curriculum.min_stage_episodes == 64
    assert him.him_curriculum.normalize_by_episode_length
    assert him.him_curriculum.assist_apply_on_compute
    assert him.him_curriculum.assist_body_name == "base_link"
    assert him.him_curriculum.force_interaction == "shared_wrench_buffer_overwrite"
    assert not him_play.him_curriculum.enabled


def test_exact_him_curriculum_profile_fails_closed() -> None:
    ensure_registries()
    cfg = registry._envs["WheelbipeV14FlatHIM"].env_cfg_cls()
    cfg.him_curriculum.assist_force_z_stages[0] = 120.0
    with pytest.raises(ValueError, match="immutable source HIM CurriculumCfgV14 profile"):
        cfg.validate()


def test_source_capabilities_are_explicit() -> None:
    ensure_registries()
    v1 = registry._envs["WheelbipeV14FlatV1"].env_cfg_cls()
    rough_v1 = registry._envs["WheelbipeV14RoughV1"].env_cfg_cls()
    v2 = registry._envs["WheelbipeV14FlatV2"].env_cfg_cls()
    assert v1.source_state_machine_status == "implemented"
    assert v1.source_gimbal_status == "implemented"
    assert rough_v1.source_state_machine_status == "implemented"
    assert rough_v1.source_gimbal_status == "implemented"
    assert v2.source_gimbal_status == "implemented"


def test_v1_exact_state_and_command_profiles_match_pinned_runtime() -> None:
    ensure_registries()
    flat = registry._envs["WheelbipeV14FlatV1"].env_cfg_cls()
    rough = registry._envs["WheelbipeV14RoughV1"].env_cfg_cls()
    rough_play = registry._envs["WheelbipeV14RoughPlayV1"].env_cfg_cls()

    # Exercise the immutable capability tuple directly, independently of the
    # registry.make integration tests below. A malformed tuple must not hide
    # behind successful metadata construction.
    flat.validate()
    rough.validate()
    rough_play.validate()

    assert flat.termination_duration_steps == 10
    assert rough.termination_duration_steps == 10
    assert flat.height_range == pytest.approx([0.20, 0.42])
    assert rough.height_range == pytest.approx([0.20, 0.42])
    assert rough_play.height_range == pytest.approx([0.25, 0.25])
    assert flat.state_machine.airborne_enabled
    assert not flat.state_machine.step_up.enabled
    assert not flat.state_machine.jump_takeoff.enabled
    assert not flat.state_machine.stair.enabled
    assert rough.state_machine.airborne_enabled
    assert rough.state_machine.step_up.enabled
    assert not rough.state_machine.jump_takeoff.enabled
    assert not rough.state_machine.stair.enabled
    assert rough.state_machine.step_up.forward_offset == pytest.approx(0.50)
    assert rough.state_machine.step_up.step_height_min == pytest.approx(0.12)
    assert rough.state_machine.step_up.step_height_max == pytest.approx(0.14)
    assert rough.state_machine.step_up.wall_height == pytest.approx(0.14)
    assert rough.state_machine.step_up.height_command_bias == pytest.approx(0.16)
    assert rough.state_machine.step_up.hold_s == pytest.approx(2.0)
    assert rough.state_machine.step_up.height_command_max == pytest.approx(0.40)
    for cfg in (flat, rough):
        assert cfg.commands.special_mode_start_iterations == [0, 0, 0]
        assert cfg.commands.special_mode_probabilities == pytest.approx([0.15, 0.15, 0.20])
        assert cfg.commands.gimbal_mode_probability == 0.0
        assert not cfg.gimbal_spin_translate.enabled
        assert cfg.state_machine.airborne_reward.airborne_scale_multipliers == {
            "undesired_contact": 25.0,
            "flat_orientation_y_v": 0.0,
            "termination": 6.0,
            "track_height_square": 0.0,
            "foot_bound_square": 0.0,
        }


def test_exact_custom_variants_inherit_source_flat_env_semantics() -> None:
    """Compact algorithms change policy representation, not the Flat task."""

    ensure_registries()
    for name in (
        "WheelbipeV14FlatDreamWaQ",
        "WheelbipeV14FlatDreamWaQPlay",
        "WheelbipeV14FlatHIM",
        "WheelbipeV14FlatHIMPlay",
        "WheelbipeV14FlatNP3OBarlow",
        "WheelbipeV14FlatNP3OBarlowPlay",
    ):
        cfg = registry._envs[name].env_cfg_cls()
        assert cfg.policy_observation_mode == "compact", name
        assert cfg.training_semantics == "source_v14", name
        assert cfg.reward_config is not None, name
        assert dict(cfg.reward_config.scales) == dict(SOURCE_V14_REWARD_SCALES), name
        assert float(cfg.noise_config.level) == pytest.approx(1.0), name
        for field_name in (
            "randomize_body_mass",
            "random_com",
            "randomize_body_material",
            "use_leg_random_start",
            "use_predefined_leg_random_start",
            "source_external_force_enabled",
        ):
            assert bool(getattr(cfg.domain_rand, field_name)), (name, field_name)
        cfg.validate()


def test_exact_custom_variant_rejects_legacy_reward_graph() -> None:
    ensure_registries()
    cfg = registry._envs["WheelbipeV14FlatHIM"].env_cfg_cls(reward_config=WheelbipeRewardConfig())
    with pytest.raises(ValueError, match="immutable source reward graph"):
        cfg.validate()


@pytest.mark.parametrize(
    "override",
    [
        {"reward_config": WheelbipeRewardConfig()},
        {"noise_config": WheelbipeNoiseConfig(level=0.0)},
        {"control_config": {"clip_actions": 1.0}},
        {"control_config": {"leg_position_target_limit": [-1.0, 1.0]}},
        {"debug_value_diagnosis": True},
    ],
)
def test_exact_variant_rejects_non_source_reward_noise_and_action_profiles(
    override: dict[str, object],
) -> None:
    ensure_registries()
    with pytest.raises(ValueError, match="immutable source|clip_actions=null"):
        registry.make(
            "Robotics-Wheelbipe-V14-Flat-v0",
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override=override,
        )


def test_exact_variant_rejects_legacy_training_semantics() -> None:
    ensure_registries()
    cfg = registry._envs["WheelbipeV14FlatV0"].env_cfg_cls(training_semantics="legacy")
    with pytest.raises(ValueError, match="exact upstream identity"):
        cfg.validate()


def test_exact_source_action_owner_preserves_raw_action_and_clamps_decoded_targets() -> None:
    pytest.importorskip("mujoco")
    ensure_registries()
    env = registry.make(
        "Robotics-Wheelbipe-V14-Flat-v0",
        sim_backend="mujoco",
        num_envs=1,
    )
    try:
        state = env.init_state()
        raw = np.asarray([[20.0, -20.0, 8.0, -8.0, 20.0, -20.0]], dtype=np.float32)
        assert env.action_space.contains(raw[0])
        targets = env.apply_action(raw, state)
        np.testing.assert_array_equal(state.info["current_actions"], raw)
        np.testing.assert_allclose(
            targets[0, env._native_leg_indices],  # noqa: SLF001
            [3.14, -3.14, 3.14, -3.14],
        )
        np.testing.assert_allclose(
            targets[0, env._native_wheel_indices],  # noqa: SLF001
            [150.0, -150.0],
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    ("task_name", "expected_actuators", "state_machine", "gimbal"),
    [
        ("WheelbipeV14FlatV1", 10, True, True),
        ("WheelbipeV14FlatV2", 10, False, True),
        ("WheelbipeV14RoughV1", 10, True, True),
        ("WheelbipeV14RoughV0", 10, False, True),
        ("Robotics-Wheelbipe-V14-Flat-v1", 10, True, True),
        ("Robotics-Wheelbipe-V14-Flat-v2", 10, False, True),
        ("Robotics-Wheelbipe-V14-Rough-v1", 10, True, True),
        ("Robotics-Wheelbipe-V14-Rough-v0", 10, False, True),
    ],
)
def test_implemented_source_variants_materialize_owner_capabilities(
    task_name: str, expected_actuators: int, state_machine: bool, gimbal: bool
) -> None:
    """Exact source IDs execute their named bounded owner on MuJoCo."""

    pytest.importorskip("mujoco")
    ensure_registries()
    env = registry.make(
        task_name,
        sim_backend="mujoco",
        num_envs=1,
    )
    try:
        assert int(env._num_native_actuators) == expected_actuators
        assert bool(env._state_machine is not None) is state_machine
        assert bool(env._gimbal_enabled) is gimbal
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_all_15_exact_source_ids_construct_reset_and_step_on_both_backends(
    backend: str,
) -> None:
    """Every published alias owns a runnable source profile on both owners."""

    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    ensure_registries()
    from unilab.envs.locomotion.wheelbipe_v14.variants import UPSTREAM_WHEELBIPE_TASK_IDS

    for task_name in UPSTREAM_WHEELBIPE_TASK_IDS:
        env = registry.make(
            task_name,
            sim_backend=backend,
            num_envs=1,
            env_cfg_override={
                "motrix_disable_equality": backend == "motrix",
            },
        )
        try:
            assert env._source_semantics, task_name
            assert env._gimbal_enabled, task_name
            assert env._num_native_actuators == 10, task_name
            expected_obs = 28 if env.cfg.policy_observation_mode == "compact" else 35
            expected_critic = 32 if env.cfg.policy_observation_mode == "compact" else 78
            state = env.init_state()
            assert state.obs["obs"].shape == (1, expected_obs), task_name
            assert state.obs["critic"].shape == (1, expected_critic), task_name
            state = env.step(np.zeros((1, 6), dtype=np.float32))
            assert state.obs["obs"].shape == (1, expected_obs), task_name
            assert state.obs["critic"].shape == (1, expected_critic), task_name
            assert np.all(np.isfinite(state.obs["obs"])), task_name
            assert np.all(np.isfinite(state.obs["critic"])), task_name
        finally:
            env.close()


def test_exact_flat_v0_capability_identity_fails_closed() -> None:
    """Exact Flat-v0 cannot rewrite its inherited gimbal/source identity."""
    ensure_registries()
    with pytest.raises(ValueError, match="immutable source capability contract"):
        registry.make(
            "WheelbipeV14FlatV0",
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override={
                "source_state_machine_status": "unported",
            },
        )


@pytest.mark.parametrize(
    ("task_name", "capability_override"),
    [
        (
            "WheelbipeV14FlatV1",
            {"source_state_machine_status": "unported"},
        ),
        (
            "Robotics-Wheelbipe-V14-Flat-v1",
            {"source_state_machine_status": "unported"},
        ),
        (
            "WheelbipeV14FlatV2",
            {"require_gimbal_actuators": False},
        ),
        (
            "Robotics-Wheelbipe-V14-Rough-v0",
            {"require_gimbal_actuators": False},
        ),
        (
            "WheelbipeV14FlatV1",
            {"state_machine": {"enabled": False}},
        ),
        (
            "Robotics-Wheelbipe-V14-Rough-v1",
            {"state_machine": {"enabled": False}},
        ),
    ],
)
def test_source_capability_contract_cannot_be_overridden(
    task_name: str, capability_override: dict[str, object]
) -> None:
    """Internal metadata overrides must not enable an unported source alias."""

    ensure_registries()
    with pytest.raises(ValueError, match="immutable source capability contract"):
        registry.make(
            task_name,
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override={
                **capability_override,
            },
        )


def test_keyframe_is_task_level_and_matches_model_dimensions() -> None:
    robot_root = ET.parse(ROBOT_XML).getroot()
    task_root = ET.parse(TASK_XML).getroot()

    # A robot fragment must stay reusable and therefore cannot own a task pose.
    assert robot_root.find(".//keyframe") is None
    keys = task_root.findall("./keyframe/key")
    assert [key.attrib["name"] for key in keys] == ["home"]
    qpos = np.fromstring(keys[0].attrib["qpos"], sep=" ")
    ctrl = np.fromstring(keys[0].attrib["ctrl"], sep=" ")
    assert qpos.size == 41
    assert ctrl.size == len(NATIVE_ACTUATOR_NAMES) == 8
    np.testing.assert_allclose(qpos[:3], [0.0, 0.0, 0.38])
    np.testing.assert_allclose(qpos[3:7], [1.0, 0.0, 0.0, 0.0])


def test_native_actuator_order_and_limits_are_frozen_in_robot_xml() -> None:
    root = ET.parse(ROBOT_XML).getroot()
    actuators = root.findall("./actuator/motor")
    assert [motor.attrib["name"] for motor in actuators] == list(NATIVE_ACTUATOR_NAMES)
    assert [motor.attrib["joint"] for motor in actuators] == [
        name.removesuffix("_ctrl") for name in NATIVE_ACTUATOR_NAMES
    ]

    expected_limits = ([(-54.0, 54.0)] * 2) + [(-5.0, 5.0), (-1000.0, 1000.0)]
    expected_limits += [(-54.0, 54.0)] * 2 + [(-5.0, 5.0), (-1000.0, 1000.0)]
    limits = [tuple(np.fromstring(motor.attrib["ctrlrange"], sep=" ")) for motor in actuators]
    assert limits == expected_limits

    sensor_names = {sensor.attrib["name"] for sensor in root.findall("./sensor/*")}
    assert {"gyro", "local_linvel", "upvector"}.issubset(sensor_names)


def test_mujoco_can_materialize_the_flat_scene() -> None:
    mujoco = pytest.importorskip("mujoco", reason="mujoco is not installed")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    assert model.nq == 41
    assert model.nv == 40
    assert model.nu == len(NATIVE_ACTUATOR_NAMES) == 8
    assert model.sensor("gyro").id >= 0
    assert model.sensor("local_linvel").id >= 0
    assert model.sensor("upvector").id >= 0
    guide_body_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx)
        for idx in range(model.nbody)
        if "guide" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx) or "")
    }
    guide_joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
        for idx in range(model.njnt)
        if "guide" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) or "")
    }
    assert len(guide_body_names) == 16
    assert len(guide_joint_names) == 16

    # Source USD mass and IdealPD actuator armatures. Startup DR adds source
    # friction samples to zero, without the ROS bridge's extra front friction.
    base_id = model.body("base_link").id
    assert float(model.body_mass[base_id]) == pytest.approx(15.96301746, rel=0.0, abs=1.0e-7)
    expected_armature = {
        "left_front1_joint": 0.015795,
        "right_front1_joint": 0.015795,
        "left_rear1_joint": 0.015795,
        "right_rear1_joint": 0.015795,
        "left_wheel_joint": 0.0,
        "right_wheel_joint": 0.0,
    }
    expected_frictionloss = {
        "left_front1_joint": 0.0,
        "right_front1_joint": 0.0,
        "left_rear1_joint": 0.0,
        "right_rear1_joint": 0.0,
        "left_wheel_joint": 0.0,
        "right_wheel_joint": 0.0,
    }
    for joint_name, armature in expected_armature.items():
        joint_id = model.joint(joint_name).id
        dof_id = int(model.jnt_dofadr[joint_id])
        assert float(model.dof_armature[dof_id]) == pytest.approx(armature, rel=0.0, abs=1.0e-9)
        assert float(model.dof_frictionloss[dof_id]) == pytest.approx(
            expected_frictionloss[joint_name], rel=0.0, abs=1.0e-9
        )


def test_source_instance_collision_primitives_and_touch_volumes() -> None:
    """USD instance proxies contain real rollers, not empty inertial bodies."""
    mujoco = pytest.importorskip("mujoco")
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/source_collision_primitives.json").read_text()
    )
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    primitives = fixture["primitives"]
    assert len(primitives) == 33
    assert sum("guide" in item["body"] for item in primitives) == 16
    for item in primitives:
        body = item["body"]
        geom = model.geom(f"{body}_collision")
        site = model.site(f"contact_site_{body}")
        expected_type = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        }[item["type"]]
        assert geom.type[0] == expected_type
        assert geom.bodyid[0] == model.body(body).id
        assert geom.conaffinity[0] != 0
        for name in ("pos", "quat", "size"):
            expected = np.fromstring(item[name], sep=" ")
            # Cylinder size uses only radius/half-height; the unused third
            # site component keeps MuJoCo's default and has no geometry meaning.
            width = len(expected)
            np.testing.assert_allclose(getattr(geom, name)[:width], expected, atol=1.0e-8)
            np.testing.assert_allclose(getattr(site, name)[:width], expected, atol=1.0e-8)


def test_front_guide_rollers_contact_a_200mm_step_before_the_base(tmp_path: Path) -> None:
    """The missing rolling contact must exist at the actual obstacle boundary."""
    mujoco = pytest.importorskip("mujoco")
    scene = tmp_path / "step.xml"
    scene.write_text(
        '<mujoco><include file="' + str(ROBOT_XML) + '"/>'
        '<worldbody><geom name="step" type="box" pos="1.4 0 .1" '
        'size=".2 .9 .1"/></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.935, 0.0, 0.30]
    mujoco.mj_forward(model, data)
    step = model.geom("step").id
    touched = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
        for contact in data.contact
        if step in contact.geom
        for g in contact.geom
        if g != step
    }
    assert {"left_front_guide_link_collision", "right_front_guide_link_collision"} <= touched
    assert "base_link_collision" not in touched


def test_motrix_compatibility_profile_survives_closed_loop_steps() -> None:
    """The owner-selected Motrix profile must not regress into solver panics.

    MotrixSim 0.8.2 can raise a Rust ``PanicException`` while factorizing the
    six closed-loop ``connect`` constraints in this mechanism.  Wheelbipe's
    Motrix owner explicitly disables that unstable equality path at scene
    materialization time; this short vectorized rollout keeps the check close
    to the env boundary without making Motrix a hard test dependency.
    """

    pytest.importorskip("motrixsim", reason="motrixsim is not installed")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14Flat",
        sim_backend="motrix",
        num_envs=1,
        env_cfg_override={
            "reward_config": WheelbipeRewardConfig(),
            "motrix_disable_equality": True,
        },
    )
    try:
        state = env.init_state()
        rng = np.random.default_rng(17)
        for _ in range(128):
            state = env.step(rng.uniform(-1.0, 1.0, size=(1, 6)).astype(np.float32))
            assert np.all(np.isfinite(state.reward))
            assert np.all(np.isfinite(state.terminated))
            assert np.all(np.isfinite(state.truncated))
            for value in state.obs.values():
                assert np.all(np.isfinite(value))
    finally:
        env.close()


@pytest.mark.parametrize("task_name", ["WheelbipeV14Flat", "WheelbipeV14Rough"])
@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_source_material_randomization_resets_only_collision_geoms(
    task_name: str, backend: str
) -> None:
    """Source DR must survive reset/step on both public backend contracts."""

    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    ensure_registries()
    env = registry.make(
        task_name,
        sim_backend=backend,
        num_envs=1,
        env_cfg_override={
            "reward_config": WheelbipeRewardConfig(),
            "training_semantics": "source_v14",
            "motrix_disable_equality": backend == "motrix",
            "domain_rand": {
                "randomize_body_mass": True,
                "random_com": True,
                "randomize_body_material": True,
                "use_leg_random_start": True,
                "use_predefined_leg_random_start": True,
                "source_external_force_enabled": False,
            },
        },
    )
    try:
        state = env.init_state()
        env.reset(np.asarray([0], dtype=np.int32))
        state = env.step(np.zeros((1, 6), dtype=np.float32))
        assert np.all(np.isfinite(state.reward))

        contype, conaffinity = env._backend.get_geom_contact_masks()  # noqa: SLF001
        collision = (np.asarray(contype) != 0) | (np.asarray(conaffinity) != 0)
        randomized = np.asarray(env._source_geom_friction)  # noqa: SLF001
        baseline = np.asarray(env._base_geom_friction)  # noqa: SLF001
        wheel_material = np.asarray(env._source_wheel_material)  # noqa: SLF001
        assert np.all(wheel_material[..., 1] <= wheel_material[..., 0])
        changed = np.any(np.abs(randomized - baseline[None, ...]) > 1.0e-8, axis=(0, 2))
        assert np.any(changed)
        assert not np.any(changed & ~collision)
    finally:
        env.close()


def test_source_mass_randomization_excludes_passive_guide_links() -> None:
    """The migrated EventCfgV14.add_leg_mass body pattern is not subtree-wide."""

    pytest.importorskip("mujoco", reason="mujoco is not installed")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14Flat",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "reward_config": WheelbipeRewardConfig(),
            "training_semantics": "source_v14",
            "domain_rand": {
                "randomize_body_mass": True,
                "base_mass_multiplier_range": [2.0, 2.0],
                "leg_mass_multiplier_range": [3.0, 3.0],
                "wheel_mass_multiplier_range": [4.0, 4.0],
                "random_com": False,
                "randomize_body_material": False,
                "source_external_force_enabled": False,
                "source_push_velocity_enabled": False,
            },
        },
    )
    try:
        baseline = np.asarray(env._backend.get_body_mass(), dtype=np.float64)  # noqa: SLF001
        randomized = np.asarray(env._source_body_mass, dtype=np.float64)[0]  # noqa: SLF001
        body_names = set(env._backend.get_body_names())  # noqa: SLF001
        guide_names = tuple(name for name in body_names if name.endswith("_guide_link"))
        guide_ids = np.asarray(env._backend.get_body_ids(guide_names), dtype=np.intp)  # noqa: SLF001
        leg_names = tuple(name for name in SOURCE_V14_LEG_MASS_BODY_NAMES if name in body_names)
        leg_ids = np.asarray(env._backend.get_body_ids(leg_names), dtype=np.intp)  # noqa: SLF001
        base_id = int(env._backend.get_body_id("base_link"))  # noqa: SLF001
        wheel_ids = np.asarray(  # noqa: SLF001
            env._backend.get_body_ids(("left_wheel_link", "right_wheel_link")), dtype=np.intp
        )
        np.testing.assert_allclose(randomized[guide_ids], baseline[guide_ids])
        np.testing.assert_allclose(randomized[base_id], baseline[base_id] * 2.0)
        np.testing.assert_allclose(randomized[leg_ids], baseline[leg_ids] * 3.0)
        np.testing.assert_allclose(randomized[wheel_ids], baseline[wheel_ids] * 4.0)
    finally:
        env.close()


def test_shipped_onnx_policy_hash_and_io_contract() -> None:
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")
    digest = hashlib.sha256(DEFAULT_WHEELBIPE_POLICY.read_bytes()).hexdigest()
    assert digest == POLICY_SHA256

    contract = inspect_wheelbipe_onnx(DEFAULT_WHEELBIPE_POLICY)
    assert contract.input_name == "obs"
    assert contract.output_name == "actions"
    assert contract.input_dim == 35
    assert contract.output_dim == 6

    policy = WheelbipeOnnxPolicy(DEFAULT_WHEELBIPE_POLICY)
    single = policy.predict(np.zeros(35, dtype=np.float32))
    batch = policy.predict(np.zeros((2, 35), dtype=np.float32))
    assert single.shape == (6,)
    assert batch.shape == (2, 6)
    np.testing.assert_allclose(batch[0], single, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(batch))
