"""Spawn-origin managers for locomotion envs.

``BaseSpawnManager`` is a no-op default: every env spawns at the world origin
(plus the existing per-env xy jitter from the dr_provider). Used whenever the
env has no procedural terrain — flat scenes don't need spatial separation

``TerrainSpawnManager`` overrides this for terrain scenes: it indexes
``terrain_origins[level, type_col]`` so each env spawns on a specific cell, and
optionally promotes/demotes ``level`` per-env on episode end. With
``enabled=True`` levels start at 0; with ``enabled=False`` levels are uniformly
distributed and never change — but spawn still uses cell-aware xyz so robots
land on the correct surface height.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np


class BaseSpawnManager:
    """Default no-op spawn manager: returns zeros, records nothing."""

    def origins_for(self, env_ids: np.ndarray) -> np.ndarray:
        return np.zeros((env_ids.shape[0], 3), dtype=np.float64)

    def apply_spawn(
        self,
        env_ids: np.ndarray,
        qpos_xyz: np.ndarray,
        *,
        yaw: np.ndarray | None = None,
    ) -> np.ndarray:
        del yaw
        return np.asarray(qpos_xyz, dtype=np.float64) + self.origins_for(env_ids)

    def record_episode_start(self, env_ids: np.ndarray, qpos_xyz: np.ndarray) -> None:
        del env_ids, qpos_xyz

    def update_on_done(self, done_indices: np.ndarray, current_xyz: np.ndarray) -> dict[str, float]:
        del done_indices, current_xyz
        return {}


@dataclass
class TerrainCurriculumCfg:
    enabled: bool = False
    """If True, levels start at 0 and evolve via promote/demote."""
    promote_frac: float = 0.5
    """Walked distance > promote_frac * cell_size promotes one level."""
    demote_frac: float = 0.25
    """Walked distance < demote_frac * cell_size demotes one level."""
    cycle_top_frac: float = 0.5
    """When level overflows the top row, resample uniformly in
    ``[num_rows * cycle_top_frac, num_rows - 1]``."""
    spawn_height_margin: float = 0.05
    """Extra z added on top of the sampled terrain surface height."""
    seed: int | None = None
    """Private curriculum-stream seed; ``None`` derives from run-level NumPy."""
    type_col_assignment: Literal["random", "round_robin"] = "random"
    """Initial terrain-column assignment for vectorized environments.

    ``random`` is the training-compatible default.  ``round_robin`` is useful
    for a play-only showcase because the first ``num_cols`` environments then
    cover every generated terrain column instead of drawing duplicate terrain
    types by chance.
    """
    initial_level: int | None = None
    """Optional fixed initial difficulty row (primarily for play profiles)."""

    def __post_init__(self) -> None:
        assignment = str(self.type_col_assignment).strip().lower()
        if assignment not in {"random", "round_robin"}:
            raise ValueError(
                "terrain_curriculum.type_col_assignment must be 'random' or 'round_robin', "
                f"got {self.type_col_assignment!r}"
            )
        self.type_col_assignment = assignment  # type: ignore[assignment]
        if self.initial_level is not None:
            try:
                level = int(self.initial_level)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "terrain_curriculum.initial_level must be an integer or null"
                ) from exc
            if level != self.initial_level:
                raise ValueError("terrain_curriculum.initial_level must be an integer or null")
            self.initial_level = level


class TerrainSpawnManager(BaseSpawnManager):
    def __init__(
        self,
        num_envs: int,
        terrain_origins: np.ndarray,
        cell_size: float,
        cfg: TerrainCurriculumCfg,
        sample_height: Callable[[np.ndarray], np.ndarray] | None = None,
        spawn_height_points: np.ndarray | None = None,
    ) -> None:
        if terrain_origins.ndim != 3 or terrain_origins.shape[2] != 3:
            raise ValueError(
                f"terrain_origins must have shape (num_rows, num_cols, 3); "
                f"got {terrain_origins.shape}"
            )
        num_rows, num_cols, _ = terrain_origins.shape
        if cfg.enabled and num_rows < 2:
            raise ValueError(
                f"Curriculum requires terrain_generator.num_rows >= 2; got {num_rows}."
            )

        self._terrain_origins = terrain_origins.astype(np.float64, copy=False)
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._cell_size = float(cell_size)
        if not np.isfinite(self._cell_size) or self._cell_size <= 0.0:
            raise ValueError(f"cell_size must be finite and positive, got {cell_size!r}")
        # Generated terrain origins are cell-local spawn centers. Cache the
        # two world-space axes once so runtime profile lookup never inspects a
        # generator, heightfield asset, or backend-private model.
        self._terrain_row_centers_x = np.median(self._terrain_origins[:, :, 0], axis=1)
        self._terrain_col_centers_y = np.median(self._terrain_origins[:, :, 1], axis=0)
        self._cfg = cfg
        if sample_height is not None and not callable(sample_height):
            raise TypeError("sample_height must be callable")
        self._sample_height = sample_height
        if spawn_height_points is None:
            self._spawn_height_points = np.zeros((1, 3), dtype=np.float64)
        else:
            points = np.asarray(spawn_height_points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"spawn_height_points must have shape (num_points, 3), got {points.shape}"
                )
            self._spawn_height_points = points
        # ``None`` means "derive from the run-level NumPy seed", not OS
        # entropy.  This keeps procedural spawn assignment reproducible under
        # UniLab's training seed contract and lets the Gym facade isolate the
        # complete cold-path stream while retaining a private Generator for
        # later curriculum updates.
        rng_seed = (
            int(cfg.seed)
            if cfg.seed is not None
            else int(np.random.randint(0, np.iinfo(np.uint32).max))
        )
        self._rng = np.random.default_rng(rng_seed)

        assignment = str(getattr(cfg, "type_col_assignment", "random")).strip().lower()
        if assignment == "round_robin":
            self.type_cols = (np.arange(num_envs, dtype=np.int32) % num_cols).astype(np.int32)
        elif assignment == "random":
            self.type_cols = self._rng.integers(0, num_cols, size=num_envs).astype(np.int32)
        else:
            raise ValueError(
                "terrain_curriculum.type_col_assignment must be 'random' or 'round_robin', "
                f"got {assignment!r}"
            )
        initial_level = getattr(cfg, "initial_level", None)
        if initial_level is not None:
            try:
                initial_level = int(initial_level)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "terrain_curriculum.initial_level must be an integer or null"
                ) from exc
            if initial_level < 0 or initial_level >= num_rows:
                raise ValueError(
                    "terrain_curriculum.initial_level must be within generated rows: "
                    f"got {initial_level}, num_rows={num_rows}"
                )
            self.levels = np.full(num_envs, initial_level, dtype=np.int32)
        elif cfg.enabled:
            self.levels = np.zeros(num_envs, dtype=np.int32)
        else:
            self.levels = self._rng.integers(0, num_rows, size=num_envs).astype(np.int32)

        self._episode_start_xyz = np.zeros((num_envs, 3), dtype=np.float64)
        self._has_started = np.zeros(num_envs, dtype=bool)

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def terrain_grid_shape(self) -> tuple[int, int]:
        """Return the cold-path generated terrain cell grid shape."""

        return self._num_rows, self._num_cols

    def terrain_cells_for(self, env_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return assigned ``(row, column)`` cells for selected environments."""

        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.levels.size):
            raise IndexError(f"terrain spawn env ids out of range: {ids.tolist()}")
        return self.levels[ids].copy(), self.type_cols[ids].copy()

    def terrain_cells_at_positions(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resolve current world positions to clamped generated-grid cells.

        Reset owners should use :meth:`terrain_cells_for`, because an env's
        assigned origin is authoritative before its root state is spawned.
        Runtime command/state owners use this method so crossing a row or
        column immediately switches to that cell's real terrain identity.
        """

        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in {2, 3}:
            raise ValueError(
                f"terrain positions must have shape (N, 2) or (N, 3), got {points.shape}"
            )
        if not np.all(np.isfinite(points[:, :2])):
            raise ValueError("terrain positions must contain finite x/y coordinates")

        def axis_indices(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
            if centers.size == 1:
                return np.zeros((values.size,), dtype=np.int32)
            step = float(np.median(np.diff(centers)))
            if not np.isfinite(step) or abs(step) <= np.finfo(np.float64).eps:
                step = self._cell_size
            indices = np.floor((values - float(centers[0])) / step + 0.5).astype(np.int32)
            return np.clip(indices, 0, centers.size - 1)

        rows = axis_indices(points[:, 0], self._terrain_row_centers_x)
        cols = axis_indices(points[:, 1], self._terrain_col_centers_y)
        return rows, cols

    def origins_for(self, env_ids: np.ndarray) -> np.ndarray:
        rows = self.levels[env_ids]
        cols = self.type_cols[env_ids]
        out = self._terrain_origins[rows, cols].copy()
        out[:, 2] += self._cfg.spawn_height_margin
        return out

    def apply_spawn(
        self,
        env_ids: np.ndarray,
        qpos_xyz: np.ndarray,
        *,
        yaw: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = self.levels[env_ids]
        cols = self.type_cols[env_ids]
        origins = self._terrain_origins[rows, cols]
        out = np.asarray(qpos_xyz, dtype=np.float64).copy()
        base_height = out[:, 2].copy()
        out[:, 0:2] += origins[:, 0:2]
        sample_height = self._sample_height
        if sample_height is None:
            out[:, 2] += origins[:, 2] + self._cfg.spawn_height_margin
            return out

        required_base_z = self._sample_spawn_required_base_height(
            out[:, 0:2],
            base_height=base_height,
            yaw=yaw,
            sample_height=sample_height,
        )
        out[:, 2] = required_base_z + self._cfg.spawn_height_margin
        return out

    def _sample_spawn_required_base_height(
        self,
        base_xy: np.ndarray,
        *,
        base_height: np.ndarray,
        yaw: np.ndarray | None,
        sample_height: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        points = self._spawn_height_points
        local_xy = points[:, :2]
        local_z = points[:, 2]
        base_surface = np.asarray(sample_height(base_xy), dtype=np.float64)
        required_base_z = base_surface + np.asarray(base_height, dtype=np.float64)
        if yaw is None:
            rotated_xy = np.broadcast_to(local_xy, (base_xy.shape[0],) + local_xy.shape)
        else:
            yaw_arr = np.asarray(yaw, dtype=np.float64).reshape(-1)
            if yaw_arr.shape != (base_xy.shape[0],):
                raise ValueError(f"yaw must have shape ({base_xy.shape[0]},), got {yaw_arr.shape}")
            cos_yaw = np.cos(yaw_arr)
            sin_yaw = np.sin(yaw_arr)
            rotated_xy = np.empty((base_xy.shape[0], local_xy.shape[0], 2), dtype=np.float64)
            rotated_xy[:, :, 0] = (
                cos_yaw[:, None] * local_xy[None, :, 0] - sin_yaw[:, None] * local_xy[None, :, 1]
            )
            rotated_xy[:, :, 1] = (
                sin_yaw[:, None] * local_xy[None, :, 0] + cos_yaw[:, None] * local_xy[None, :, 1]
            )

        sample_xy = base_xy[:, None, :] + rotated_xy
        sampled = np.asarray(sample_height(sample_xy.reshape(-1, 2)), dtype=np.float64).reshape(
            base_xy.shape[0], points.shape[0]
        )
        required_support_z = np.max(sampled - local_z[None, :], axis=1)
        return np.maximum(required_base_z, required_support_z)

    def record_episode_start(self, env_ids: np.ndarray, qpos_xyz: np.ndarray) -> None:
        self._episode_start_xyz[env_ids] = qpos_xyz
        self._has_started[env_ids] = True

    def update_on_done(self, done_indices: np.ndarray, current_xyz: np.ndarray) -> dict[str, float]:
        active_mask = self._has_started[done_indices]
        active = done_indices[active_mask]
        num_skipped = int((~active_mask).sum())

        if active.size == 0:
            return {
                "mean_level": float(self.levels.mean()),
                "max_level": float(self.levels.max()),
                "mean_walked": 0.0,
                "num_promoted": 0,
                "num_demoted": 0,
                "num_skipped": num_skipped,
            }

        starts = self._episode_start_xyz[active, :2]
        ends = current_xyz[active_mask, :2]
        walked = np.linalg.norm(ends - starts, axis=1)

        num_promoted = 0
        num_demoted = 0
        if self._cfg.enabled:
            promote_threshold = self._cfg.promote_frac * self._cell_size
            demote_threshold = self._cfg.demote_frac * self._cell_size
            promote_mask = walked > promote_threshold
            demote_mask = walked < demote_threshold

            promote_ids = active[promote_mask]
            demote_ids = active[demote_mask]
            num_promoted = int(promote_ids.size)
            num_demoted = int(demote_ids.size)

            self.levels[promote_ids] += 1
            self.levels[demote_ids] -= 1

            overflow_mask = self.levels[promote_ids] >= self._num_rows
            if overflow_mask.any():
                lo = int(self._num_rows * self._cfg.cycle_top_frac)
                lo = min(max(lo, 0), self._num_rows - 1)
                overflow_ids = promote_ids[overflow_mask]
                self.levels[overflow_ids] = self._rng.integers(
                    lo, self._num_rows, size=overflow_ids.size
                ).astype(np.int32)

            np.clip(self.levels, 0, self._num_rows - 1, out=self.levels)

        return {
            "mean_level": float(self.levels.mean()),
            "max_level": float(self.levels.max()),
            "mean_walked": float(walked.mean()),
            "num_promoted": num_promoted,
            "num_demoted": num_demoted,
            "num_skipped": num_skipped,
        }
