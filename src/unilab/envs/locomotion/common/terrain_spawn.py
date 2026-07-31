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

from dataclasses import dataclass

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
    performance_gating: bool = False
    """Require task-provided performance masks for level changes."""
    max_vx_error: float = 0.5
    max_tilt_rate: float = 0.05
    max_nonwheel_contact_rate: float = 0.03
    seed: int | None = None
    bootstrap_type: str | None = None
    """Optional terrain type used until the task unlocks terrain variety."""
    initial_type_cols: list[int] | None = None
    """Optional explicit terrain type assignment, one column per environment."""
    initial_levels: list[int] | None = None
    """Optional explicit terrain level assignment, one level per environment."""


class TerrainSpawnManager(BaseSpawnManager):
    def __init__(
        self,
        num_envs: int,
        terrain_origins: np.ndarray,
        cell_size: float,
        cfg: TerrainCurriculumCfg,
        terrain_surface_sampler: object | None = None,
        spawn_height_points: np.ndarray | None = None,
        type_probabilities: np.ndarray | None = None,
        initial_type_col: int | None = None,
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
        self._cfg = cfg
        self._terrain_surface_sampler = terrain_surface_sampler
        if spawn_height_points is None:
            self._spawn_height_points = np.zeros((1, 3), dtype=np.float64)
        else:
            points = np.asarray(spawn_height_points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"spawn_height_points must have shape (num_points, 3), got {points.shape}"
                )
            self._spawn_height_points = points
        self._rng = np.random.default_rng(cfg.seed)
        if type_probabilities is None:
            self._type_probabilities = None
        else:
            probabilities = np.asarray(type_probabilities, dtype=np.float64)
            if probabilities.shape != (num_cols,):
                raise ValueError(
                    f"type_probabilities must have shape ({num_cols},), got "
                    f"{probabilities.shape}"
                )
            if np.any(probabilities < 0.0) or float(probabilities.sum()) <= 0.0:
                raise ValueError("type_probabilities must be non-negative with a positive sum")
            self._type_probabilities = probabilities / probabilities.sum()

        configured_type_cols = cfg.initial_type_cols
        if initial_type_col is not None and configured_type_cols is not None:
            raise ValueError("initial_type_col conflicts with terrain initial_type_cols")
        if configured_type_cols is not None:
            type_cols = np.asarray(configured_type_cols, dtype=np.int32)
            if type_cols.shape != (num_envs,):
                raise ValueError(
                    "terrain initial_type_cols must have one entry per environment: "
                    f"{type_cols.shape} != ({num_envs},)"
                )
            if np.any((type_cols < 0) | (type_cols >= num_cols)):
                raise ValueError(
                    f"terrain initial_type_cols must be in [0, {num_cols})"
                )
            self.type_cols = type_cols.copy()
        elif initial_type_col is None:
            self.type_cols = self._sample_type_cols(num_envs)
        else:
            self._validate_type_col(initial_type_col)
            self.type_cols = np.full(num_envs, int(initial_type_col), dtype=np.int32)

        configured_levels = cfg.initial_levels
        if configured_levels is not None:
            levels = np.asarray(configured_levels, dtype=np.int32)
            if levels.shape != (num_envs,):
                raise ValueError(
                    "terrain initial_levels must have one entry per environment: "
                    f"{levels.shape} != ({num_envs},)"
                )
            if np.any((levels < 0) | (levels >= num_rows)):
                raise ValueError(f"terrain initial_levels must be in [0, {num_rows})")
            self.levels = levels.copy()
        elif cfg.enabled:
            self.levels = np.zeros(num_envs, dtype=np.int32)
        else:
            self.levels = self._rng.integers(0, num_rows, size=num_envs).astype(np.int32)

        self._episode_start_xyz = np.zeros((num_envs, 3), dtype=np.float64)
        self._has_started = np.zeros(num_envs, dtype=bool)

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def training_state_dict(self) -> dict[str, object]:
        """Return curriculum state without retaining live physics buffers."""
        return {
            "version": 1,
            "levels": self.levels.tolist(),
            "type_cols": self.type_cols.tolist(),
            "rng_state": _to_builtin(self._rng.bit_generator.state),
        }

    def load_training_state_dict(self, state: dict[str, object]) -> None:
        version = int(state.get("version", 0))
        if version != 1:
            raise ValueError(f"Unsupported terrain training state version: {version}")
        levels = np.asarray(state["levels"], dtype=self.levels.dtype)
        type_cols = np.asarray(state["type_cols"], dtype=self.type_cols.dtype)
        if levels.shape != self.levels.shape or type_cols.shape != self.type_cols.shape:
            raise ValueError(
                "Terrain training state shape mismatch: "
                f"levels={levels.shape}/{self.levels.shape}, "
                f"type_cols={type_cols.shape}/{self.type_cols.shape}"
            )
        if np.any((levels < 0) | (levels >= self._num_rows)):
            raise ValueError("Terrain training state contains an invalid level")
        if np.any((type_cols < 0) | (type_cols >= self._num_cols)):
            raise ValueError("Terrain training state contains an invalid type column")
        self.levels[:] = levels
        self.type_cols[:] = type_cols
        self._rng.bit_generator.state = state["rng_state"]  # type: ignore[assignment]
        self._has_started.fill(False)

    def _validate_type_col(self, type_col: int) -> None:
        if not 0 <= int(type_col) < self._num_cols:
            raise ValueError(
                f"terrain type column must be in [0, {self._num_cols}), got {type_col}"
            )

    def _sample_type_cols(self, count: int) -> np.ndarray:
        return np.asarray(
            self._rng.choice(
                self._num_cols,
                size=int(count),
                p=self._type_probabilities,
            ),
            dtype=np.int32,
        )

    def set_type_col(self, env_ids: np.ndarray, type_col: int) -> None:
        self._validate_type_col(type_col)
        self.type_cols[np.asarray(env_ids, dtype=np.int32)] = int(type_col)

    def resample_type_cols(self, env_ids: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.int32)
        self.type_cols[ids] = self._sample_type_cols(ids.size)

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
        if self._terrain_surface_sampler is None:
            out[:, 2] += origins[:, 2] + self._cfg.spawn_height_margin
            return out

        required_base_z = self._sample_spawn_required_base_height(
            out[:, 0:2],
            base_height=base_height,
            yaw=yaw,
        )
        out[:, 2] = required_base_z + self._cfg.spawn_height_margin
        return out

    def _sample_spawn_required_base_height(
        self,
        base_xy: np.ndarray,
        *,
        base_height: np.ndarray,
        yaw: np.ndarray | None,
    ) -> np.ndarray:
        sampler = self._terrain_surface_sampler
        sample_height = getattr(sampler, "sample_height", None)
        if not callable(sample_height):
            raise TypeError("terrain_surface_sampler must expose sample_height(xy)")

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

    def update_on_done(
        self,
        done_indices: np.ndarray,
        current_xyz: np.ndarray,
        *,
        promote_performance: np.ndarray | None = None,
        demote_performance: np.ndarray | None = None,
    ) -> dict[str, float]:
        active_mask = self._has_started[done_indices]
        active = done_indices[active_mask]
        num_skipped = int((~active_mask).sum())

        if active.size == 0:
            return {
                "mean_level": float(self.levels.mean()),
                "max_level": float(self.levels.max()),
                "min_level": float(self.levels.min()),
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
            if promote_performance is not None:
                promote = np.asarray(promote_performance, dtype=bool)
                if promote.shape != done_indices.shape:
                    raise ValueError("promote_performance must match done_indices shape")
                promote_mask &= promote[active_mask]
            if demote_performance is not None:
                demote = np.asarray(demote_performance, dtype=bool)
                if demote.shape != done_indices.shape:
                    raise ValueError("demote_performance must match done_indices shape")
                demote_mask |= demote[active_mask]
            demote_mask &= ~promote_mask

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
            "min_level": float(self.levels.min()),
            "mean_walked": float(walked.mean()),
            "num_promoted": num_promoted,
            "num_demoted": num_demoted,
            "num_skipped": num_skipped,
        }


def _to_builtin(value: object) -> object:
    """Convert NumPy RNG state into a weights-only torch-loadable payload."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
