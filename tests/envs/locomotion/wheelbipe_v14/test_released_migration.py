"""Regression against source data, independent of UniLab's config defaults.

The fixture records the released rough run's YAML and composed USD MassAPI
values. Neither Isaac Lab nor the external checkout is required to run it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.registry import apply_cfg_overrides, ensure_registries
from unilab.envs.locomotion.wheelbipe_v14 import (
    WheelbipeRewardConfig,
    wheelbipe_delay_profile_overrides,
)
from unilab.envs.locomotion.wheelbipe_v14.joystick import WheelbipeV14Env
from unilab.terrains import TerrainGenerator
from unilab.training.reward import extract_reward_config
from unilab.utils.rotation import np_yaw_to_quat

ROOT = Path(__file__).resolve().parents[4]
SOURCE = json.loads((Path(__file__).parent / "fixtures/released_source.json").read_text())


def test_sim2sim_registry_owner_does_not_clip_source_policy_actions() -> None:
    pytest.importorskip("mujoco")
    ensure_registries()
    overrides = wheelbipe_delay_profile_overrides("local_physics")
    overrides["reward_config"] = WheelbipeRewardConfig()
    env = registry.make(
        "WheelbipeV14Flat", sim_backend="mujoco", num_envs=1, env_cfg_override=overrides
    )
    try:
        state = env.init_state()
        action = np.asarray([[2.0, -2.0, 3.0, -3.0, 3.0, -4.0]], dtype=np.float32)
        targets = env.apply_action(action, state)
        np.testing.assert_array_equal(state.info["current_actions"], action)
        np.testing.assert_allclose(targets[:, env._native_wheel_indices], [[30.0, -40.0]])
    finally:
        env.close()


def _owner(task: str, backend: str):
    ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task}/{backend}"])
    owner = registry._envs[str(cfg.training.task_name)].env_cfg_cls()
    overrides = OmegaConf.to_container(cfg.env, resolve=True)
    overrides.update(extract_reward_config(cfg))
    apply_cfg_overrides(owner, overrides)
    owner.validate()
    return owner


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
@pytest.mark.parametrize("task", ["wheelbipe_v14_flat", "wheelbipe_v14_rough"])
def test_training_spring_matches_released_isaac_actuator(task: str, backend: str) -> None:
    owner = _owner(task, backend)
    assert owner.post_step_forward_sensor
    control = owner.control_config
    spring = SOURCE["spring"]
    for local, source in (
        ("spring_random_force", "random_force"),
        ("spring_linear_up", "linear_up"),
        ("spring_linear_down", "linear_down"),
        ("spring_linear_length", "linear_length"),
        ("spring_offset", "spring_offset"),
    ):
        assert getattr(control, local) == pytest.approx(spring[source])
    # The source flag disables additional damping, not IdealPD's 50 N s/m.
    assert spring["damping"] is False
    assert control.spring_damping == SOURCE["spring_actuator_damping"]


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_canonical_rough_reward_matches_complete_released_snapshot(backend: str) -> None:
    owner = _owner("wheelbipe_v14_rough", backend)
    assert dict(owner.reward_config.scales) == SOURCE["rewards"]
    np.testing.assert_allclose(owner.source_height_scan_xy, SOURCE["height_scan_xy"])


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_rough_playback_keeps_robot_and_heightfield_world_coordinates(backend: str) -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task=wheelbipe_v14_rough/{backend}"])
    assert cfg.training.render_spacing == 0.0
    assert cfg.play_profile.env.render_spacing == 0.0


def test_source_height_scanner_averages_yaw_aligned_step_edge() -> None:
    owner = WheelbipeV14Env.__new__(WheelbipeV14Env)
    owner._cfg = _owner("wheelbipe_v14_rough", "mujoco")
    owner._np_dtype = np.float64
    owner._source_height_scan_xy = np.asarray(SOURCE["height_scan_xy"])
    # The first pose sees six of nine rays beyond the step, the rotated
    # footprint only five. A center-point approximation would return 0.3
    # for both, producing the wrong height target pressure at this edge.
    owner._terrain_surface_sample_height = lambda xy: (xy[..., 0] >= 0.0) * 0.3
    observed, relative = owner._source_height_signals(
        np.asarray([[0.003, 0.0, 0.5], [0.003, 0.0, 0.5]]),
        np_yaw_to_quat(np.asarray([0.0, np.pi / 6])),
    )
    np.testing.assert_allclose(observed, [0.5, 0.5])
    np.testing.assert_allclose(relative, [0.3, 0.5 - 0.3 * 5 / 9])


def test_source_velocities_use_com_and_world_difference_before_rotation() -> None:
    owner = WheelbipeV14Env.__new__(WheelbipeV14Env)
    owner._cfg = SimpleNamespace(sensor=SimpleNamespace(gyro="gyro"))
    owner._source_semantics = True
    owner._np_dtype = np.float64
    # Two equal poses, different per-environment COM randomization. The base
    # faces world +Y and translates at 1 m/s along its own +X while pitching.
    quat = np_yaw_to_quat(np.full(2, np.pi / 2))
    owner._source_base_com_b = np.asarray([[0.1, 0.0, 0.2], [-0.1, 0.0, 0.3]])
    owner._source_wheel_com_b = np.asarray([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])
    owner._wheel_body_ids = np.asarray([1, 2])
    owner._backend = SimpleNamespace(
        get_base_quat=lambda: quat,
        get_base_lin_vel=lambda: np.tile([0.0, 1.0, 0.0], (2, 1)),
        get_sensor_data=lambda name: np.tile([0.0, 2.0, 0.0], (2, 1)),
        get_body_quat_w=lambda ids: np.tile(quat[:, None, :], (1, 2, 1)),
        get_body_lin_vel_w=lambda ids: np.tile([0.0, 2.0, 0.0], (2, 2, 1)),
        get_body_ang_vel_w=lambda ids: np.tile([-3.0, 0.0, 0.0], (2, 2, 1)),
        # No inertial-metadata getter or relative-frame velocity getter:
        # neither belongs in this hot path.
    )
    np.testing.assert_allclose(owner.get_local_linvel(), [[1.4, 0, -0.2], [1.6, 0, 0.2]], atol=2e-7)
    np.testing.assert_allclose(
        owner._source_wheel_linear_velocity(),
        [[[0.9, 0, 0.2], [0.9, 0, 0.2]], [[0.7, 0, -0.2], [0.7, 0, -0.2]]],
        atol=2e-7,
    )
    owner._source_semantics = False
    np.testing.assert_allclose(owner.get_local_linvel(), [[1, 0, 0], [1, 0, 0]], atol=2e-7)


def test_source_com_cache_includes_startup_randomization_and_survives_reset(monkeypatch) -> None:
    pytest.importorskip("mujoco")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14Flat",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "reward_config": WheelbipeRewardConfig(),
            "training_semantics": "source_v14",
            "domain_rand": {"random_com": True, "com_offset_x": [0.01, 0.01]},
        },
    )
    try:
        expected = env._backend.get_body_ipos()[env._base_body_ids[0]] + env._source_base_com_offset
        np.testing.assert_allclose(env._source_base_com_b, expected, atol=1e-8)

        def forbidden():
            raise AssertionError("metadata lookup outside initialization")

        monkeypatch.setattr(env._backend, "get_body_ipos", forbidden)
        env.reset(np.asarray([0, 1], dtype=np.int32))
        env.step(np.zeros((2, 6), dtype=np.float32))
        np.testing.assert_allclose(env._source_base_com_b, expected, atol=1e-8)
    finally:
        env.close()


def test_source_imu_and_body_tracking_share_current_physics_state() -> None:
    pytest.importorskip("mujoco")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14Flat",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "training_semantics": "source_v14",
            "reward_config": WheelbipeRewardConfig(),
        },
    )
    try:
        env.init_state()
        action = np.asarray([[0.4, -0.2, -0.4, 0.2, 3.0, -2.0]], dtype=np.float32)
        for _ in range(10):
            env.step(action)
            # MuJoCo free-joint angular qvel is body-local, like this robot's
            # identity-oriented IMU site. The public base angular-velocity
            # contract describes world coordinates, so inspect qvel only in
            # this backend-specific integration test.
            current_omega = env._backend._physics_state[
                :, env._backend._idx_qvel + 3 : env._backend._idx_qvel + 6
            ]
            np.testing.assert_allclose(
                env._backend.get_sensor_data(env.cfg.sensor.gyro), current_omega, atol=1e-6
            )
            np.testing.assert_allclose(
                env._backend.get_body_pos_w(env._base_body_ids)[:, 0],
                env._backend.get_base_pos(),
                atol=1e-6,
            )
    finally:
        env.close()


def test_source_height_scanner_ignores_invalid_hits_and_uses_cell_origin() -> None:
    owner = WheelbipeV14Env.__new__(WheelbipeV14Env)
    owner._cfg = _owner("wheelbipe_v14_rough", "mujoco")
    owner._np_dtype = np.float64
    owner._source_height_scan_xy = np.asarray(SOURCE["height_scan_xy"])
    owner._source_height_spawn_margin = 0.05
    owner._spawn = SimpleNamespace(origins_for=lambda ids: np.tile([0.0, 0.0, 0.25], (len(ids), 1)))
    heights = np.full((2, 9), np.inf)
    heights[0, :2] = [0.1, 0.3]
    owner._terrain_surface_sample_height = lambda xy: heights
    _, relative = owner._source_height_signals(
        np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]), np_yaw_to_quat(np.zeros(2))
    )
    np.testing.assert_allclose(relative, [0.3, 0.3])


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_canonical_rough_keeps_released_medium_stairs_and_height_commands(backend: str) -> None:
    owner = _owner("wheelbipe_v14_rough", backend)
    generator = owner.scene.terrain.generator
    for name, value in SOURCE["terrain"].items():
        assert getattr(generator, name) == pytest.approx(value)
    assert list(generator.sub_terrains) == list(SOURCE["sub_terrains"])
    for name, fields in SOURCE["sub_terrains"].items():
        for field, value in fields.items():
            assert getattr(generator.sub_terrains[name], field) == pytest.approx(value)

    generated = TerrainGenerator(generator).generate()
    names = generated.terrain_type_names
    # The missing middle stairs occupied four of fourteen source columns.
    for name in ("stair_slope_for_rm_mid", "inv_stair_slope_for_rm_mid"):
        assert np.count_nonzero(generated.terrain_type_ids[0] == names.index(name)) == 2
    assert owner.terrain_commands.enabled
    profiles = owner.terrain_commands.profiles
    np.testing.assert_allclose(profiles["stair_slope_for_rm_mid"].height_ranges, [[0.24, 0.38]])
    np.testing.assert_allclose(profiles["inv_stair_slope_for_rm_low"].height_ranges, [[0.24, 0.38]])
    assert "inv_stair_slope_for_rm_mid" not in profiles


def test_training_robot_mass_com_and_full_inertia_match_source_usd() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots/wheelbipe_v14_2/mjcf/scene_flat.xml")
    )

    def inertia_tensor(quat, diagonal):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(quat))
        rotation = rotation.reshape(3, 3)
        return rotation @ np.diag(diagonal) @ rotation.T

    for name, expected in SOURCE["bodies"].items():
        body = model.body(name)
        np.testing.assert_allclose(body.mass, expected["mass"], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(body.ipos, expected["com"], rtol=1e-6, atol=1e-9)
        # Principal values alone do not establish inertia parity: a missing
        # inertial rotation loses the passive linkage's products of inertia.
        np.testing.assert_allclose(
            inertia_tensor(body.iquat, body.inertia),
            inertia_tensor(expected["quat"], expected["inertia"]),
            rtol=1e-5,
            atol=5e-8,
        )
