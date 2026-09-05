"""Procedural rough-terrain variant of the Wheelbipe V14 task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.envs.locomotion.common.terrain_spawn import TerrainCurriculumCfg
from unilab.terrains import (
    HfWheelbipeCliffInvertedStairsTerrainCfg,
    HfWheelbipeGridBarsTerrainCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
)

from .joystick import WheelbipeCommands, WheelbipeV14Env, WheelbipeV14FlatCfg
from .semantics import SOURCE_V14_RESET_CONTACT_BODY_NAMES
from .state_machine import WheelbipeStateMachineOwnerMixin, WheelbipeTerrainCommandConfig
from .task_modes import WheelbipeGimbalSpinTranslateOwnerMixin


@dataclass(kw_only=True)
class WheelbipeRoughTerrainCfg(TerrainGeneratorCfg):
    """Heightfield conversion of pinned ``RM_ROTATION_TERRAINS_CFG_99``."""

    source_preset: str = "RM_ROTATION_TERRAINS_CFG_99"
    source_commit: str = "b8ff79f3"
    source_border_height: float = 1.0
    source_slope_threshold: float = 0.5
    source_conversion_boundary: str = (
        "isaac_mesh_tessellation_and_border_verticalization_not_reproduced"
    )
    size: tuple[float, float] = (9.0, 9.0)
    num_rows: int = 10
    num_cols: int = 10
    border_width: float = 5.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    curriculum: bool = True
    curriculum_column_allocation: Literal["one_per_type", "proportional"] = "proportional"
    add_lights: bool = True
    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "tiny_step_rot": HfWheelbipeGridBarsTerrainCfg(
                proportion=0.2,
                num_horizontal_range=(2, 4),
                num_vertical_range=(2, 4),
                randomize_bar_count_difficulty=True,
                force_unequal_counts=False,
                force_even_counts=True,
                bar_width_range=(0.05, 0.2),
                randomize_bar_width_difficulty=True,
                bar_height_range=(0.01, 0.03),
            ),
            "slope_for_rm_low": hf_pyramid_slope(
                proportion=0.1,
                slope_range=(0.05, 0.15),
                platform_width=0.0,
                border_width=1.0,
            ),
            "inv_slope_for_rm_low": hf_pyramid_slope_inv(
                proportion=0.1,
                slope_range=(0.05, 0.15),
                platform_width=0.0,
                border_width=1.0,
            ),
            "stair_slope_for_rm_low": pyramid_stairs(
                proportion=0.1,
                step_height_range=(0.005, 0.015),
                step_width=0.1,
                platform_width=1.0,
                border_width=1.0,
            ),
            "inv_stair_slope_for_rm_low": pyramid_stairs_inv(
                proportion=0.1,
                step_height_range=(0.005, 0.015),
                step_width=0.1,
                platform_width=2.0,
                border_width=1.0,
            ),
            "plane_for_rm_rot": flat(proportion=0.3),
            "random_uniform_for_rm": random_rough(
                proportion=0.1,
                noise_range=(0.0, 0.03),
                noise_step=0.005,
                downsampled_scale=0.2,
                border_width=1.0,
            ),
        }
    )


@dataclass(kw_only=True)
class WheelbipeRoughRunningTerrainCfg(TerrainGeneratorCfg):
    """Heightfield conversion of final composed ``RM_ROUGH_TERRAINS_CFG``."""

    source_preset: str = "RM_ROUGH_TERRAINS_CFG"
    source_commit: str = "b8ff79f3"
    source_border_height: float = 1.0
    source_slope_threshold: float = 0.5
    source_conversion_boundary: str = (
        "isaac_mesh_tessellation_and_border_verticalization_not_reproduced"
    )
    size: tuple[float, float] = (9.0, 9.0)
    num_rows: int = 10
    num_cols: int = 13
    border_width: float = 10.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    curriculum: bool = True
    curriculum_column_allocation: Literal["one_per_type", "proportional"] = "proportional"
    add_lights: bool = True
    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "low_speed_stair_for_rm": pyramid_stairs(
                proportion=0.1,
                step_height_range=(0.15, 0.35),
                step_width=1.5,
                platform_width=3.0,
                border_width=0.0,
            ),
            "tiny_step": HfWheelbipeGridBarsTerrainCfg(
                proportion=0.2,
                num_horizontal_range=(2, 4),
                num_vertical_range=(2, 4),
                randomize_bar_count_difficulty=True,
                force_unequal_counts=False,
                force_even_counts=True,
                bar_width_range=(0.05, 0.2),
                randomize_bar_width_difficulty=True,
                bar_height_range=(0.02, 0.06),
            ),
            "slope_for_rm_high": hf_pyramid_slope(
                proportion=0.1,
                slope_range=(0.15, 0.35),
                platform_width=2.0,
                border_width=2.0,
            ),
            "inv_slope_for_rm_low": hf_pyramid_slope_inv(
                proportion=0.1,
                slope_range=(0.05, 0.15),
                platform_width=0.0,
                border_width=1.0,
            ),
            "high_speed_stair_for_rm": pyramid_stairs(
                proportion=0.1,
                step_height_range=(0.15, 0.40),
                step_width=1.5,
                platform_width=3.0,
                border_width=0.0,
            ),
            "stair_slope_for_rm_high": pyramid_stairs(
                proportion=0.1,
                step_height_range=(0.015, 0.032),
                step_width=0.1,
                platform_width=1.0,
                border_width=1.0,
            ),
            "inv_stair_slope_for_rm_low": pyramid_stairs_inv(
                proportion=0.1,
                step_height_range=(0.005, 0.015),
                step_width=0.1,
                platform_width=2.0,
                border_width=1.0,
            ),
            "inv_stair_slope_for_rm_high": pyramid_stairs_inv(
                proportion=0.2,
                step_height_range=(0.015, 0.032),
                step_width=0.1,
                platform_width=2.0,
                border_width=1.0,
            ),
            "plane_for_rm": flat(proportion=0.1),
            "random_uniform_for_rm": random_rough(
                proportion=0.1,
                noise_range=(0.0, 0.03),
                noise_step=0.005,
                downsampled_scale=0.2,
                border_width=1.0,
            ),
            "cliff_inv_stair_slope_short_for_rm": HfWheelbipeCliffInvertedStairsTerrainCfg(
                proportion=0.1,
                step_height_range=(0.025, 0.032),
                step_width=0.1,
                platform_width=3.0,
                height_offset_range=(0.3, 0.4),
                border_width=2.0,
            ),
        }
    )


@dataclass(kw_only=True)
class WheelbipeRoughPlayTerrainCfg(TerrainGeneratorCfg):
    """Final play preset based on pinned ``RM_ROUGH_TERRAINS_PLAY_CFG``.

    The raw source preset has one row. Rough-Play-v0 explicitly expands it to
    ten rows during ``__post_init__``; Rough-Play-v1 retains one row. Use the
    scene factory's ``num_rows`` argument to preserve that composed identity.
    """

    source_preset: str = "RM_ROUGH_TERRAINS_PLAY_CFG"
    source_commit: str = "b8ff79f3"
    source_border_height: float = 1.0
    source_slope_threshold: float = 0.5
    source_conversion_boundary: str = (
        "isaac_mesh_tessellation_and_border_verticalization_not_reproduced"
    )
    size: tuple[float, float] = (9.0, 9.0)
    num_rows: int = 1
    num_cols: int = 1
    border_width: float = 10.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    curriculum: bool = True
    curriculum_column_allocation: Literal["one_per_type", "proportional"] = "proportional"
    add_lights: bool = True
    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "cliff_inv_stair_slope_short_for_rm_play": HfWheelbipeCliffInvertedStairsTerrainCfg(
                proportion=0.1,
                step_height_range=(0.03, 0.03),
                step_width=0.1,
                platform_width=4.0,
                height_offset_range=(0.3, 0.4),
                border_width=1.0,
            ),
        }
    )


def wheelbipe_rotation_scene() -> SceneCfg:
    return _wheelbipe_rough_scene(WheelbipeRoughTerrainCfg())


def wheelbipe_running_scene() -> SceneCfg:
    return _wheelbipe_rough_scene(WheelbipeRoughRunningTerrainCfg())


def wheelbipe_play_scene(*, num_rows: int = 1) -> SceneCfg:
    if int(num_rows) <= 0:
        raise ValueError("WheelBipe rough play terrain num_rows must be positive")
    return _wheelbipe_rough_scene(WheelbipeRoughPlayTerrainCfg(num_rows=int(num_rows)))


def _wheelbipe_rough_scene(generator: TerrainGeneratorCfg) -> SceneCfg:
    return SceneCfg(
        model_file=str(
            ASSETS_ROOT_PATH / "robots" / "wheelbipe_v14_2" / "mjcf" / "wheelbipeV14_2.xml"
        ),
        fragment_files=[
            str(ASSETS_ROOT_PATH / "robots" / "wheelbipe_v14_2" / "locomotion_task.xml")
        ],
        terrain=TerrainSceneCfg(
            generator=generator,
            hfield_name="terrain_hfield",
            geom_name="floor",
        ),
    )


@dataclass
class WheelbipeRoughCommands(WheelbipeCommands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-1.0, 0.0, -1.0], [1.5, 0.0, 1.5]]
    )
    resampling_time: float = 5.0
    heading_command: bool = True


@dataclass
class WheelbipeRoughBoundaryResetConfig:
    """Pinned rough-grid boundary timeout configuration.

    ``use_inner_terrain_area`` matches the source task's distinction between
    the generated terrain grid itself and the full grid plus its surrounding
    border.  Boundary exits are timeouts, not terminal failures.
    """

    enabled: bool = True
    margin: float = 0.5
    use_inner_terrain_area: bool = False

    def validate(self) -> None:
        margin = float(self.margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("rough terrain boundary margin must be finite and non-negative")


def wheelbipe_rough_boundary_half_extents(
    generator: TerrainGeneratorCfg,
    config: WheelbipeRoughBoundaryResetConfig,
) -> tuple[float, float]:
    """Return the source-equivalent positive x/y boundary thresholds."""

    config.validate()
    half_x = 0.5 * float(generator.num_rows) * float(generator.size[0])
    half_y = 0.5 * float(generator.num_cols) * float(generator.size[1])
    if not config.use_inner_terrain_area:
        border_width = float(generator.border_width)
        half_x += border_width
        half_y += border_width
    margin = float(config.margin)
    return max(half_x - margin, 0.0), max(half_y - margin, 0.0)


def compute_wheelbipe_rough_boundary_timeout(
    base_pos: np.ndarray,
    generator: TerrainGeneratorCfg,
    config: WheelbipeRoughBoundaryResetConfig,
) -> np.ndarray:
    """Evaluate the pinned rough-grid boundary timeout for each environment."""

    positions = np.asarray(base_pos)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"base_pos must have shape [N, >=2], got {positions.shape}")
    if not config.enabled:
        return np.zeros(positions.shape[0], dtype=bool)
    half_x, half_y = wheelbipe_rough_boundary_half_extents(generator, config)
    return (np.abs(positions[:, 0]) > half_x) | (np.abs(positions[:, 1]) > half_y)


@registry.envcfg("WheelbipeV14Rough")
@dataclass
class WheelbipeV14RoughCfg(WheelbipeV14FlatCfg):
    # The terrain materializer supplies the floor. Starting from the pure
    # robot XML avoids attaching a second geom named ``floor``.
    scene: SceneCfg = field(default_factory=wheelbipe_rotation_scene)
    commands: WheelbipeRoughCommands = field(default_factory=WheelbipeRoughCommands)
    # Source rough critics retain absolute root-z. Only the reward reference
    # subtracts the sampled terrain height.
    use_absolute_height: bool = False
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)
    # The legacy rough owner leaves terrain-aware command profiles disabled,
    # while exact Rough-v1 supplies its pinned profile table.  Keeping the
    # field on the common rough owner lets a play-only profile opt into the
    # same command-direction contract without changing the training owner.
    terrain_commands: WheelbipeTerrainCommandConfig = field(
        default_factory=WheelbipeTerrainCommandConfig
    )
    rough_terrain_boundary_reset: WheelbipeRoughBoundaryResetConfig = field(
        default_factory=WheelbipeRoughBoundaryResetConfig
    )


@registry.env("WheelbipeV14Rough", sim_backend="mujoco")
@registry.env("WheelbipeV14Rough", sim_backend="motrix")
class WheelbipeV14RoughEnv(
    WheelbipeGimbalSpinTranslateOwnerMixin,
    WheelbipeStateMachineOwnerMixin,
    WheelbipeV14Env,
):
    _cfg: WheelbipeV14RoughCfg

    def _source_reset_contact_body_names(self) -> tuple[str, ...]:
        # The source V14 owner drops base/guide contacts for non-plane terrain
        # when gimbal is enabled, using only the two gimbal links for reset
        # termination.  Keep this decision in the rough owner; body IDs are
        # still resolved once by the shared source-state initializer.
        if self._gimbal_enabled and self._cfg.scene.terrain is not None:
            return ("gimbal_yaw_link", "gimbal_pitch_link")
        return SOURCE_V14_RESET_CONTACT_BODY_NAMES

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        terrain = self._cfg.scene.terrain
        generator = terrain.generator if terrain is not None else None
        if generator is None:
            return truncated
        base_pos = np.asarray(self._backend.get_base_pos())
        out = compute_wheelbipe_rough_boundary_timeout(
            base_pos,
            generator,
            self._cfg.rough_terrain_boundary_reset,
        )
        np.logical_or(truncated, out, out=truncated)
        return truncated


# The upstream Isaac task exposes this ``*EnvCfg`` spelling.  Preserve it as a
# non-registered alias while keeping the UniLab ``*Cfg`` convention canonical.
WheelbipeV14RoughEnvCfg = WheelbipeV14RoughCfg


__all__ = [
    "WheelbipeRoughBoundaryResetConfig",
    "WheelbipeRoughPlayTerrainCfg",
    "WheelbipeRoughTerrainCfg",
    "WheelbipeRoughRunningTerrainCfg",
    "WheelbipeV14RoughCfg",
    "WheelbipeV14RoughEnvCfg",
    "WheelbipeV14RoughEnv",
    "compute_wheelbipe_rough_boundary_timeout",
    "wheelbipe_rough_boundary_half_extents",
    "wheelbipe_rotation_scene",
    "wheelbipe_running_scene",
    "wheelbipe_play_scene",
]
