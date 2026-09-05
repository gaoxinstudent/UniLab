# Adapted from wheeled-legged_RL / agent_world terrain generators.
# Copyright (c) 2026 SCUTRobotLab
# Authors: Zhang Zhirui <2231625449@qq.com>, Cui Yu <ctty694@gmail.com>
# SPDX-License-Identifier: MIT
"""Heightfield conversions for pinned WheelBipe V14 source terrains.

The source task uses an MIT-licensed mesh grid-bars generator and a custom
heightfield cliff-stairs generator from ``wheeled-legged_RL`` commit
``b8ff79f3``.  UniLab's terrain/backend contract accepts one merged
heightfield, so the grid bars are represented by the equivalent sampled top
surface.  This preserves names, parameters and deterministic curriculum
identity, but does not claim triangle-mesh contact parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from unilab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainHeightField,
    TerrainOutput,
)

_MIN_BASE_THICKNESS = 0.01


def _terrain_output(
    noise: np.ndarray,
    *,
    cfg: SubTerrainCfg,
    horizontal_scale: float,
    vertical_scale: float,
    base_thickness_ratio: float,
    origin_z: float,
) -> TerrainOutput:
    raw_noise = np.asarray(noise)
    elevation_min = int(np.min(raw_noise))
    elevation_max = int(np.max(raw_noise))
    if elevation_min < np.iinfo(np.int16).min or elevation_max > np.iinfo(np.int16).max:
        raise ValueError("source WheelBipe terrain exceeds int16 heightfield range")
    elevation_range = max(elevation_max - elevation_min, 1)
    max_physical_height = elevation_range * vertical_scale
    heightfield = TerrainHeightField(
        noise=raw_noise.astype(np.int16, copy=True),
        size=cfg.size,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        max_physical_height=max_physical_height,
        base_thickness=max(
            max_physical_height * float(base_thickness_ratio),
            _MIN_BASE_THICKNESS,
        ),
        # Source custom terrains encode ground level at raw height zero. Keep
        # that absolute reference when UniLab normalizes by ``elevation_min``.
        z_offset=elevation_min * vertical_scale,
    )
    return TerrainOutput(
        origin=np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z]),
        heightfield=heightfield,
        flat_patches=None,
    )


def _interpolate(bounds: tuple[float, float], difficulty: float) -> float:
    low, high = sorted((float(bounds[0]), float(bounds[1])))
    return low + float(difficulty) * (high - low)


def _interpolate_count(bounds: tuple[int, int], difficulty: float) -> int:
    low, high = sorted((int(bounds[0]), int(bounds[1])))
    return int(np.rint(low + float(difficulty) * (high - low)))


def _force_even(value: int, bounds: tuple[int, int]) -> int:
    if value % 2 == 0:
        return value
    low, high = sorted((int(bounds[0]), int(bounds[1])))
    if value + 1 <= high:
        return value + 1
    if value - 1 >= low:
        return value - 1
    raise ValueError(f"cannot resolve an even bar count from {value} within {bounds}")


def _bar_centers(count: int, span: float, bar_width: float) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if count * bar_width > span:
        raise ValueError(f"{count} bars of width {bar_width} exceed terrain span {span}")
    gap = (span - count * bar_width) / (count + 1)
    first = gap + 0.5 * bar_width
    return first + np.arange(count, dtype=np.float64) * (bar_width + gap)


@dataclass(kw_only=True)
class HfWheelbipeGridBarsTerrainCfg(SubTerrainCfg):
    """Sampled-heightfield conversion of source ``MeshCustomGridBarsTerrainCfg``."""

    source_representation: ClassVar[str] = "mesh_grid_bars"
    conversion_boundary: ClassVar[str] = "sampled_top_surface_no_mesh_contact_parity"

    num_horizontal_range: tuple[int, int] = (2, 4)
    num_vertical_range: tuple[int, int] = (2, 4)
    randomize_bar_count_difficulty: bool = True
    force_unequal_counts: bool = False
    force_even_counts: bool = True
    bar_width_range: tuple[float, float] = (0.05, 0.2)
    randomize_bar_width_difficulty: bool = True
    bar_height_range: tuple[float, float] = (0.02, 0.06)
    bar_length_ratio: float = 0.95
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        horizontal_difficulty = float(difficulty)
        vertical_difficulty = float(difficulty)
        if self.randomize_bar_count_difficulty:
            horizontal_difficulty = float(rng.uniform())
            vertical_difficulty = float(rng.uniform())
        width_difficulty = (
            float(rng.uniform()) if self.randomize_bar_width_difficulty else float(difficulty)
        )
        horizontal_count = _interpolate_count(self.num_horizontal_range, horizontal_difficulty)
        vertical_count = _interpolate_count(self.num_vertical_range, vertical_difficulty)
        if self.force_even_counts:
            horizontal_count = _force_even(horizontal_count, self.num_horizontal_range)
            vertical_count = _force_even(vertical_count, self.num_vertical_range)
        if self.force_unequal_counts and horizontal_count == vertical_count:
            raise ValueError("source grid-bars force_unequal_counts is not representable here")

        bar_width = _interpolate(self.bar_width_range, width_difficulty)
        bar_height = _interpolate(self.bar_height_range, difficulty)
        width_pixels = int(round(self.size[0] / self.horizontal_scale))
        length_pixels = int(round(self.size[1] / self.horizontal_scale))
        noise = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        height_units = int(np.rint(bar_height / self.vertical_scale))

        if not 0.0 < float(self.bar_length_ratio) <= 1.0:
            raise ValueError("bar_length_ratio must be in (0, 1]")
        x_positions = (np.arange(width_pixels, dtype=np.float64) + 0.5) * self.horizontal_scale
        y_positions = (np.arange(length_pixels, dtype=np.float64) + 0.5) * self.horizontal_scale
        x_margin = 0.5 * self.size[0] * (1.0 - float(self.bar_length_ratio))
        y_margin = 0.5 * self.size[1] * (1.0 - float(self.bar_length_ratio))
        x_long = (x_positions >= x_margin) & (x_positions < self.size[0] - x_margin)
        y_long = (y_positions >= y_margin) & (y_positions < self.size[1] - y_margin)
        half_width = 0.5 * bar_width

        for center_y in _bar_centers(horizontal_count, self.size[1], bar_width):
            y_mask = np.abs(y_positions - center_y) <= half_width
            if not np.any(y_mask):
                y_mask[np.argmin(np.abs(y_positions - center_y))] = True
            noise[np.ix_(x_long, y_mask)] = height_units
        for center_x in _bar_centers(vertical_count, self.size[0], bar_width):
            x_mask = np.abs(x_positions - center_x) <= half_width
            if not np.any(x_mask):
                x_mask[np.argmin(np.abs(x_positions - center_x))] = True
            noise[np.ix_(x_mask, y_long)] = height_units

        return _terrain_output(
            noise,
            cfg=self,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            # The source mesh reports the unquantized box-top height as its
            # spawn origin; the converted top surface remains quantized.
            origin_z=float(bar_height),
        )


@dataclass(kw_only=True)
class HfWheelbipeCliffInvertedStairsTerrainCfg(SubTerrainCfg):
    """Pinned raised inverted-stairs heightfield with a flat outer cliff border."""

    source_representation: ClassVar[str] = "hf_cliff_inverted_pyramid_stairs"
    conversion_boundary: ClassVar[str] = "heightfield_formula_preserved"

    step_height_range: tuple[float, float] = (0.025, 0.032)
    step_width: float = 0.1
    platform_width: float = 3.0
    height_offset_range: tuple[float, float] = (0.3, 0.4)
    border_width: float = 2.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0

    def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
        del rng
        step_height_units = int(
            np.rint(-_interpolate(self.step_height_range, difficulty) / self.vertical_scale)
        )
        height_offset_units = int(
            np.rint(_interpolate(self.height_offset_range, difficulty) / self.vertical_scale)
        )
        width_pixels = int(round(self.size[0] / self.horizontal_scale))
        length_pixels = int(round(self.size[1] / self.horizontal_scale))
        border_pixels = int(round(self.border_width / self.horizontal_scale))
        step_pixels = max(int(round(self.step_width / self.horizontal_scale)), 1)
        platform_pixels = int(round(self.platform_width / self.horizontal_scale))
        if 2 * border_pixels >= min(width_pixels, length_pixels):
            raise ValueError("cliff terrain border leaves no interior")

        noise = np.zeros((width_pixels, length_pixels), dtype=np.int32)
        start_x = border_pixels
        start_y = border_pixels
        stop_x = width_pixels - border_pixels
        stop_y = length_pixels - border_pixels
        noise[start_x:stop_x, start_y:stop_y] = height_offset_units
        current_step_height = 0
        while (stop_x - start_x) > platform_pixels and (stop_y - start_y) > platform_pixels:
            start_x += step_pixels
            stop_x -= step_pixels
            start_y += step_pixels
            stop_y -= step_pixels
            current_step_height += step_height_units
            noise[start_x:stop_x, start_y:stop_y] = height_offset_units + current_step_height
        center_height = float(noise[width_pixels // 2, length_pixels // 2])
        return _terrain_output(
            noise,
            cfg=self,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            origin_z=center_height * self.vertical_scale,
        )


__all__ = [
    "HfWheelbipeCliffInvertedStairsTerrainCfg",
    "HfWheelbipeGridBarsTerrainCfg",
]
