"""Pinned WheelBipe V14 rough-terrain identity tests."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.wheelbipe_v14.rough import (
    WheelbipeRoughBoundaryResetConfig,
    WheelbipeRoughPlayTerrainCfg,
    WheelbipeRoughRunningTerrainCfg,
    WheelbipeRoughTerrainCfg,
    compute_wheelbipe_rough_boundary_timeout,
    wheelbipe_play_scene,
    wheelbipe_rough_boundary_half_extents,
)
from unilab.terrains import (
    HfWheelbipeCliffInvertedStairsTerrainCfg,
    HfWheelbipeGridBarsTerrainCfg,
    TerrainGenerator,
)
from unilab.utils.rotation import np_yaw_from_quat

ROTATION_NAMES = (
    "tiny_step_rot",
    "slope_for_rm_low",
    "inv_slope_for_rm_low",
    "stair_slope_for_rm_low",
    "inv_stair_slope_for_rm_low",
    "plane_for_rm_rot",
    "random_uniform_for_rm",
)
RUNNING_NAMES = (
    "low_speed_stair_for_rm",
    "tiny_step",
    "slope_for_rm_high",
    "inv_slope_for_rm_low",
    "high_speed_stair_for_rm",
    "stair_slope_for_rm_high",
    "inv_stair_slope_for_rm_low",
    "inv_stair_slope_for_rm_high",
    "plane_for_rm",
    "random_uniform_for_rm",
    "cliff_inv_stair_slope_short_for_rm",
)
PLAY_NAME = "cliff_inv_stair_slope_short_for_rm_play"
ROTATION_COLUMNS = np.asarray([0, 0, 1, 2, 3, 4, 5, 5, 5, 6], dtype=np.int32)
RUNNING_COLUMNS = np.asarray(
    [0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10],
    dtype=np.int32,
)


def test_rotation_preset_matches_pinned_active_profile() -> None:
    cfg = WheelbipeRoughTerrainCfg(seed=11)

    assert cfg.source_preset == "RM_ROTATION_TERRAINS_CFG_99"
    assert cfg.source_commit == "b8ff79f3"
    assert cfg.curriculum
    assert cfg.curriculum_column_allocation == "proportional"
    assert cfg.size == (9.0, 9.0)
    assert (cfg.num_rows, cfg.num_cols) == (10, 10)
    assert cfg.border_width == pytest.approx(5.0)
    assert tuple(cfg.sub_terrains) == ROTATION_NAMES

    tiny = cfg.sub_terrains["tiny_step_rot"]
    assert isinstance(tiny, HfWheelbipeGridBarsTerrainCfg)
    assert tiny.num_horizontal_range == (2, 4)
    assert tiny.num_vertical_range == (2, 4)
    assert tiny.bar_width_range == pytest.approx((0.05, 0.2))
    assert tiny.bar_height_range == pytest.approx((0.01, 0.03))
    assert tiny.force_even_counts

    terrain = TerrainGenerator(cfg).generate()
    assert terrain.terrain_type_names == ROTATION_NAMES
    assert np.array_equal(terrain.terrain_type_ids, np.tile(ROTATION_COLUMNS, (10, 1)))


def test_running_preset_matches_final_composed_pinned_profile() -> None:
    cfg = WheelbipeRoughRunningTerrainCfg(seed=17)

    assert cfg.source_preset == "RM_ROUGH_TERRAINS_CFG"
    assert cfg.curriculum  # V14 runtime composition overrides the raw preset's False.
    assert cfg.curriculum_column_allocation == "proportional"
    assert cfg.size == (9.0, 9.0)
    assert (cfg.num_rows, cfg.num_cols) == (10, 13)
    assert cfg.border_width == pytest.approx(10.0)
    assert tuple(cfg.sub_terrains) == RUNNING_NAMES

    low_speed = cast(Any, cfg.sub_terrains["low_speed_stair_for_rm"])
    assert low_speed.step_height_range == pytest.approx((0.15, 0.35))
    assert low_speed.step_width == pytest.approx(1.5)
    assert low_speed.platform_width == pytest.approx(3.0)
    high_speed = cast(Any, cfg.sub_terrains["high_speed_stair_for_rm"])
    assert high_speed.step_height_range == pytest.approx((0.15, 0.40))
    cliff = cfg.sub_terrains["cliff_inv_stair_slope_short_for_rm"]
    assert isinstance(cliff, HfWheelbipeCliffInvertedStairsTerrainCfg)
    assert cliff.step_height_range == pytest.approx((0.025, 0.032))
    assert cliff.height_offset_range == pytest.approx((0.3, 0.4))

    terrain = TerrainGenerator(cfg).generate()
    assert terrain.terrain_type_names == RUNNING_NAMES
    assert np.array_equal(terrain.terrain_type_ids, np.tile(RUNNING_COLUMNS, (10, 1)))


def test_play_preset_and_composed_row_counts_are_explicit() -> None:
    cfg = WheelbipeRoughPlayTerrainCfg(seed=5)

    assert cfg.source_preset == "RM_ROUGH_TERRAINS_PLAY_CFG"
    assert cfg.curriculum
    assert (cfg.num_rows, cfg.num_cols) == (1, 1)
    assert tuple(cfg.sub_terrains) == (PLAY_NAME,)
    cliff = cfg.sub_terrains[PLAY_NAME]
    assert isinstance(cliff, HfWheelbipeCliffInvertedStairsTerrainCfg)
    assert cliff.step_height_range == pytest.approx((0.03, 0.03))
    assert cliff.platform_width == pytest.approx(4.0)
    assert cliff.height_offset_range == pytest.approx((0.3, 0.4))
    assert cliff.border_width == pytest.approx(1.0)

    v1_scene = wheelbipe_play_scene(num_rows=1)
    v0_scene = wheelbipe_play_scene(num_rows=10)
    assert v1_scene.terrain is not None
    assert v0_scene.terrain is not None
    assert v1_scene.terrain.generator is not None
    assert v0_scene.terrain.generator is not None
    assert v1_scene.terrain.generator.num_rows == 1
    assert v0_scene.terrain.generator.num_rows == 10
    with pytest.raises(ValueError, match="num_rows must be positive"):
        wheelbipe_play_scene(num_rows=0)


@pytest.mark.parametrize(
    ("task_name", "preset", "rows", "cols"),
    [
        ("WheelbipeV14RoughV0", "RM_ROTATION_TERRAINS_CFG_99", 10, 10),
        ("WheelbipeV14RoughV1", "RM_ROUGH_TERRAINS_CFG", 10, 13),
        ("WheelbipeV14RoughPlayV0", "RM_ROUGH_TERRAINS_PLAY_CFG", 10, 1),
        ("WheelbipeV14RoughPlayV1", "RM_ROUGH_TERRAINS_PLAY_CFG", 1, 1),
    ],
)
def test_exact_rough_variants_select_their_composed_source_preset(
    task_name: str,
    preset: str,
    rows: int,
    cols: int,
) -> None:
    ensure_registries()
    cfg = registry._envs[task_name].env_cfg_cls()
    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.generator is not None
    generator = cast(Any, cfg.scene.terrain.generator)
    assert generator.source_preset == preset
    assert generator.curriculum
    assert (generator.num_rows, generator.num_cols) == (rows, cols)


@pytest.mark.parametrize(
    ("task_name", "use_inner", "expected_half_extents"),
    [
        ("WheelbipeV14RoughV0", False, (49.5, 49.5)),
        ("WheelbipeV14RoughV1", False, (54.5, 68.0)),
        ("WheelbipeV14RoughPlayV0", True, (44.5, 4.0)),
        ("WheelbipeV14RoughPlayV1", False, (14.0, 14.0)),
    ],
)
def test_exact_rough_variant_boundary_timeout_matches_source_on_both_sides(
    task_name: str,
    use_inner: bool,
    expected_half_extents: tuple[float, float],
) -> None:
    ensure_registries()
    cfg = cast(Any, registry._envs[task_name].env_cfg_cls())
    terrain = cfg.scene.terrain
    assert terrain is not None
    generator = terrain.generator
    assert generator is not None
    boundary = cfg.rough_terrain_boundary_reset
    assert isinstance(boundary, WheelbipeRoughBoundaryResetConfig)
    assert boundary.enabled
    assert boundary.margin == pytest.approx(0.5)
    assert boundary.use_inner_terrain_area is use_inner

    half_x, half_y = wheelbipe_rough_boundary_half_extents(generator, boundary)
    assert (half_x, half_y) == pytest.approx(expected_half_extents)
    epsilon = 0.01
    positions = np.asarray(
        [
            [half_x, 0.0, 0.0],
            [-half_x, 0.0, 0.0],
            [0.0, half_y, 0.0],
            [0.0, -half_y, 0.0],
            [half_x + epsilon, 0.0, 0.0],
            [-half_x - epsilon, 0.0, 0.0],
            [0.0, half_y + epsilon, 0.0],
            [0.0, -half_y - epsilon, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(
        compute_wheelbipe_rough_boundary_timeout(positions, generator, boundary),
        np.asarray([False, False, False, False, True, True, True, True]),
    )


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
@pytest.mark.parametrize(
    ("task_name", "names", "columns", "rows"),
    [
        ("WheelbipeV14RoughV0", ROTATION_NAMES, ROTATION_COLUMNS, 10),
        ("WheelbipeV14RoughV1", RUNNING_NAMES, RUNNING_COLUMNS, 10),
        ("WheelbipeV14RoughPlayV0", (PLAY_NAME,), np.asarray([0], dtype=np.int32), 10),
        ("WheelbipeV14RoughPlayV1", (PLAY_NAME,), np.asarray([0], dtype=np.int32), 1),
    ],
)
def test_exact_rough_terrain_metadata_survives_both_backends_reset_and_step(
    backend: str,
    task_name: str,
    names: tuple[str, ...],
    columns: np.ndarray,
    rows: int,
) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    ensure_registries()
    env = registry.make(
        task_name,
        sim_backend=backend,
        num_envs=2,
    )
    try:
        spawn = env._backend.get_terrain_spawn_data()  # noqa: SLF001
        assert spawn is not None
        assert spawn.terrain_type_names == names
        assert spawn.terrain_type_ids is not None
        assert np.array_equal(spawn.terrain_type_ids, np.tile(columns, (rows, 1)))
        assert not spawn.terrain_origins.flags.writeable
        assert not spawn.terrain_type_ids.flags.writeable

        state = env.init_state()
        reset_obs, _ = env.reset(np.asarray([0], dtype=np.int32))
        state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert reset_obs["obs"].shape == (1, 35)
        assert state.obs["obs"].shape == (2, 35)
        assert state.obs["critic"].shape == (2, 78)
        assert np.all(np.isfinite(state.obs["obs"]))
        assert np.all(np.isfinite(state.obs["critic"]))
        assert np.all(np.isfinite(state.reward))
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_running_stair_profile_masks_air_reset_and_applies_axis_command_owner(
    backend: str,
) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14RoughV1",
        sim_backend=backend,
        num_envs=8,
    )
    try:
        # Running column 5 maps to ``high_speed_stair_for_rm``. Its pinned
        # profile disables predefined air, snaps reset yaw to cardinal axes,
        # and owns the command/height ranges before the initial observation.
        env._spawn.type_cols[:] = 5  # noqa: SLF001
        env.cfg.state_machine.reset_airborne_probability = 1.0
        state = env.init_state()

        assert not np.any(env._state_machine.airborne_state)  # noqa: SLF001
        yaw = np_yaw_from_quat(env._backend.get_base_quat())  # noqa: SLF001
        cardinal = np.asarray([0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi])
        error = np.min(
            np.abs((yaw[:, None] - cardinal[None, :] + np.pi) % (2.0 * np.pi) - np.pi),
            axis=1,
        )
        assert np.all(error < 1.0e-4)
        commands = np.asarray(state.info["commands"])
        assert np.all((np.abs(commands[:, 0]) >= 1.5) & (np.abs(commands[:, 0]) <= 2.7))
        assert np.all(commands[:, 1] == 0.0)
        heights = np.asarray(state.info["height_commands"])
        assert np.all((heights >= 0.22) & (heights <= 0.34))
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_running_command_profile_tracks_current_position_across_cells(
    backend: str,
) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14RoughV1",
        sim_backend=backend,
        num_envs=2,
    )
    try:
        state = env.init_state()
        spawn = env._backend.get_terrain_spawn_data()  # noqa: SLF001
        assert spawn is not None
        env_id = np.asarray([0], dtype=np.int32)

        def move_to_column(column: int) -> None:
            qpos = np.asarray(env._backend.get_keyframe_qpos("home"))[None, :].copy()  # noqa: SLF001
            qvel = np.asarray(env._backend.get_init_qvel())[None, :].copy()  # noqa: SLF001
            qpos[0, :3] += spawn.terrain_origins[0, column]
            env._backend.set_state(env_id, qpos, qvel)  # noqa: SLF001
            env._update_commands(state.info)  # noqa: SLF001

        # Running columns 0 and 5 resolve to low-speed and high-speed stairs.
        # Their x-command ranges are disjoint apart from the 1.5 boundary, so
        # the command proves runtime lookup follows current root position and
        # not the environment's reset-assigned cell.
        move_to_column(0)
        low_speed_x = abs(float(np.asarray(state.info["commands"])[0, 0]))
        assert 0.5 <= low_speed_x <= 1.5
        assert env._terrain_command_last_type_ids[0] == 0  # noqa: SLF001

        move_to_column(5)
        high_speed_x = abs(float(np.asarray(state.info["commands"])[0, 0]))
        assert 1.5 <= high_speed_x <= 2.7
        assert env._terrain_command_last_type_ids[0] == 4  # noqa: SLF001
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_running_non_heading_yaw_override_preserves_heading_envs(backend: str) -> None:
    pytest.importorskip(backend if backend == "mujoco" else "motrixsim")
    ensure_registries()
    env = registry.make(
        "WheelbipeV14RoughV1",
        sim_backend=backend,
        num_envs=2,
    )
    try:
        env.init_state()
        sampled = {
            "commands": np.asarray([[1.0, 0.0, 0.75], [1.0, 0.0, -0.75]], dtype=np.float32),
            "is_heading_env": np.asarray([True, False]),
            "heading_commands": np.zeros((2,), dtype=np.float32),
            "is_standing_env": np.zeros((2,), dtype=bool),
            "special_mode_id": np.full((2,), -1, dtype=np.int8),
        }
        sampled, _heights = env._apply_terrain_command_profile_for_type_ids(  # noqa: SLF001
            np.asarray([0, 1], dtype=np.intp),
            sampled,
            np.asarray([0.25, 0.25], dtype=np.float32),
            # Running terrain type 0 is low_speed_stair_for_rm, whose active
            # profile owns only ang_vel_z_non_heading=(-0.1, 0.1).
            np.asarray([0, 0], dtype=np.int32),
            current_yaw=np.zeros((2,), dtype=np.float32),
            force_resample=True,
        )

        commands = np.asarray(sampled["commands"])
        assert commands[0, 2] == pytest.approx(0.75)
        assert -0.1 <= commands[1, 2] <= 0.1
    finally:
        env.close()


def test_source_custom_terrain_conversion_boundary_is_explicit_and_deterministic() -> None:
    grid = HfWheelbipeGridBarsTerrainCfg(
        size=(9.0, 9.0),
        proportion=1.0,
        bar_height_range=(0.02, 0.06),
    )
    first = grid.function(0.6, np.random.default_rng(29))
    second = grid.function(0.6, np.random.default_rng(29))

    assert grid.source_representation == "mesh_grid_bars"
    assert grid.conversion_boundary == "sampled_top_surface_no_mesh_contact_parity"
    assert np.array_equal(first.heightfield.noise, second.heightfield.noise)
    assert first.origin[2] == pytest.approx(0.044)
    assert np.max(first.heightfield.noise) > 0

    cliff = HfWheelbipeCliffInvertedStairsTerrainCfg(
        size=(9.0, 9.0),
        proportion=1.0,
        step_height_range=(0.03, 0.03),
        platform_width=4.0,
        height_offset_range=(0.3, 0.4),
        border_width=1.0,
    )
    output = cliff.function(0.5, np.random.default_rng(1))
    assert cliff.source_representation == "hf_cliff_inverted_pyramid_stairs"
    assert cliff.conversion_boundary == "heightfield_formula_preserved"
    assert output.heightfield.noise[0, 0] == 0
    assert np.min(output.heightfield.noise) == -20
    assert np.max(output.heightfield.noise) == 70
    assert np.min(output.heightfield.physical_heights_xy()) == pytest.approx(-0.1)
    assert np.max(output.heightfield.physical_heights_xy()) == pytest.approx(0.35)
    assert output.origin[2] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "cfg_type",
    [WheelbipeRoughTerrainCfg, WheelbipeRoughRunningTerrainCfg, WheelbipeRoughPlayTerrainCfg],
)
def test_source_preset_records_non_equivalent_isaac_mesh_boundary(cfg_type: type) -> None:
    cfg = cfg_type()
    assert cfg.source_border_height == pytest.approx(1.0)
    assert cfg.source_slope_threshold == pytest.approx(0.5)
    assert cfg.source_conversion_boundary == (
        "isaac_mesh_tessellation_and_border_verticalization_not_reproduced"
    )
