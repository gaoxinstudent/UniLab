from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_from_euler_xyz, np_quat_mul
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.height_scan import (
    HeightScanConfig,
    init_height_scan_sensor,
    raw_height_scan_obs,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.envs.locomotion.real68.balance import (
    NONWHEEL_CONTACT_SENSORS,
    WHEEL_CONTACT_SENSORS,
    Real68BalanceCfg,
    Real68BalanceDomainRandomizationProvider,
    Real68BalanceEnv,
    Real68Commands,
    Real68DomainRandConfig,
)
from unilab.envs.locomotion.real68.balance import (
    Real68CommandCurriculumCfg as BaseReal68CommandCurriculumCfg,
)
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    wave_terrain,
)


@dataclass
class Real68RoughCommands(Real68Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-2.0, 0.0, -5.0], [3.0, 0.0, 5.0]]
    )
    resampling_time: float = 3.0
    rel_standing_envs: float = 0.0


@dataclass
class Real68RoughDomainRandConfig(Real68DomainRandConfig):
    reset_pos_xy_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    reset_height_offset_range: list[float] = field(default_factory=lambda: [0.0, 0.08])
    reset_roll_range: list[float] = field(default_factory=lambda: [-0.1, 0.1])
    reset_pitch_range: list[float] = field(default_factory=lambda: [-0.1, 0.1])
    reset_yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    reset_qvel_limit: float = 0.15


@dataclass
class Real68CommandCurriculumCfg(BaseReal68CommandCurriculumCfg):
    enabled: bool = True
    initial_vel_limit: list[list[float]] = field(
        default_factory=lambda: [[0.1, 0.0, -0.3], [0.35, 0.0, 0.3]]
    )
    final_vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-2.0, 0.0, -5.0], [3.0, 0.0, 5.0]]
    )
    terrain_unlock_vx_progress: float = 0.8


@dataclass
class RoughTerminationConfig:
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0
    fall_termination: bool = True
    min_up_proj: float = 0.2
    min_base_height: float = 0.12
    nonwheel_contact_termination: bool = True
    nonwheel_contact_threshold: float = 0.5
    nonwheel_contact_max_steps: int = 8


@dataclass(kw_only=True)
class Real68RoughTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 6
    num_cols: int = 6
    border_width: float = 1.0
    add_lights: bool = True
    horizontal_scale: float = 0.1

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "pyramid_stairs": pyramid_stairs(
                proportion=0.2,
                step_height_range=(0.025, 0.20),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.2,
                step_height_range=(0.025, 0.20),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.2,
                slope_range=(0.0, 0.3),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.2,
                slope_range=(0.0, 0.3),
                platform_width=2.0,
                border_width=0.2,
            ),
            "random_rough": random_rough(
                proportion=0.1,
                noise_range=(0.01, 0.06),
                noise_step=0.01,
                border_width=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.1,
                amplitude_range=(0.0, 0.12),
                num_waves=4,
                border_width=0.2,
            ),
        }
    )


@registry.envcfg("Real68BalanceRough")
@dataclass
class Real68BalanceRoughCfg(Real68BalanceCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "real68" / "real68.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "real68" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=Real68RoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    commands: Real68RoughCommands = field(default_factory=Real68RoughCommands)
    command_curriculum: Real68CommandCurriculumCfg = field(
        default_factory=Real68CommandCurriculumCfg
    )
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)
    domain_rand: Real68RoughDomainRandConfig = field(default_factory=Real68RoughDomainRandConfig)


class Real68BalanceRoughDomainRandomizationProvider(Real68BalanceDomainRandomizationProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))

        xy_low, xy_high = env.cfg.domain_rand.reset_pos_xy_range
        qpos[:, 0:2] += np.random.uniform(xy_low, xy_high, (num_reset, 2))
        z_low, z_high = env.cfg.domain_rand.reset_height_offset_range
        qpos[:, 2] += np.random.uniform(z_low, z_high, (num_reset,))

        roll_low, roll_high = env.cfg.domain_rand.reset_roll_range
        pitch_low, pitch_high = env.cfg.domain_rand.reset_pitch_range
        yaw_low, yaw_high = env.cfg.domain_rand.reset_yaw_range
        roll = np.random.uniform(roll_low, roll_high, (num_reset,))
        pitch = np.random.uniform(pitch_low, pitch_high, (num_reset,))
        yaw = np.random.uniform(yaw_low, yaw_high, (num_reset,))
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3], yaw=yaw)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_quat_from_euler_xyz(roll, pitch, yaw))
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])

        limit = float(env.cfg.domain_rand.reset_qvel_limit)
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-limit, limit, size=(num_reset, 6)),
            dtype=get_global_dtype(),
        )

        commands = env.sample_velocity_commands(num_reset)
        height_commands = env.sample_height_commands(num_reset)
        info_updates = {
            "commands": commands,
            "height_commands": height_commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "qacc": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "wheel_contacts": np.zeros(
                (num_reset, len(WHEEL_CONTACT_SENSORS)), dtype=get_global_dtype()
            ),
            "nonwheel_contacts": np.zeros(
                (num_reset, len(NONWHEEL_CONTACT_SENSORS)), dtype=get_global_dtype()
            ),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(
                env,
                num_reset,
                base_geom_friction=getattr(env, "_base_geom_friction", None),
                ground_geom_id=getattr(env, "_ground_geom_id", None),
            ),
        )


@registry.env("Real68BalanceRough", sim_backend="mujoco")
class Real68BalanceRoughEnv(Real68BalanceEnv):
    _cfg: Real68BalanceRoughCfg
    _height_scan_dim: int = 0

    def __init__(self, cfg: Real68BalanceRoughCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._nonwheel_contact_steps = np.zeros((num_envs,), dtype=np.int32)
        self._rough_scan_raw: np.ndarray | None = None
        self._rough_scan_base_pos: np.ndarray | None = None
        self._rough_scan_height_obs: np.ndarray | None = None
        self._rough_scan_base_height: np.ndarray | None = None
        terrain_origins = getattr(self._backend, "terrain_origins", None)
        terrain_generator = cfg.scene.terrain.generator if cfg.scene.terrain is not None else None
        if terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=getattr(self._backend, "terrain_surface_sampler", None),
            )
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    def _make_dr_provider(self) -> Real68BalanceRoughDomainRandomizationProvider:
        return Real68BalanceRoughDomainRandomizationProvider()

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        self._nonwheel_contact_steps[env_ids] = 0
        return obs, info

    def _init_reward_functions(self) -> None:
        def gated(fn):
            return lambda ctx: fn(ctx) * self._upright_scale(ctx.gravity)

        def _height_tracking(ctx: RewardContext) -> np.ndarray:
            return self._reward_height_tracking(ctx) * self._upright_scale(ctx.gravity)

        def _posture(ctx: RewardContext) -> np.ndarray:
            return self._reward_posture(ctx) * self._upright_scale(ctx.gravity)

        def _leg_symmetry(ctx: RewardContext) -> np.ndarray:
            return self._reward_leg_symmetry(ctx) * self._upright_scale(ctx.gravity)

        def _torques(ctx: RewardContext) -> np.ndarray:
            return self._reward_torques_l2(ctx) * self._upright_scale(ctx.gravity)

        def _wheel_vel(ctx: RewardContext) -> np.ndarray:
            return self._reward_wheel_vel(ctx) * self._upright_scale(ctx.gravity)

        def _nonwheel_contact(ctx: RewardContext) -> np.ndarray:
            return self._reward_nonwheel_contact(ctx) * self._upright_scale(ctx.gravity)

        def _joint_pos_penalty(ctx: RewardContext) -> np.ndarray:
            return self._reward_joint_pos_penalty(ctx) * self._upright_scale(ctx.gravity)

        def _joint_power(ctx: RewardContext) -> np.ndarray:
            return self._reward_joint_power(ctx) * self._upright_scale(ctx.gravity)

        def _alive(ctx: RewardContext) -> np.ndarray:
            return rewards.alive(ctx) * self._upright_scale(ctx.gravity)

        self._reward_fns = {
            "tracking_lin_vel": gated(rewards.tracking_lin_vel),
            "tracking_ang_vel": gated(rewards.tracking_ang_vel),
            "forward_progress": gated(rewards.forward_progress),
            "under_speed": gated(rewards.under_speed),
            "yaw_rate_when_uncommanded": gated(rewards.yaw_rate_when_uncommanded),
            "lin_vel_z": gated(rewards.lin_vel_z),
            "ang_vel_xy": gated(rewards.ang_vel_xy),
            "orientation": gated(self._reward_orientation),
            "torques": _torques,
            "wheel_vel": _wheel_vel,
            "posture": _posture,
            "leg_symmetry": _leg_symmetry,
            "height_tracking": _height_tracking,
            "joint_pos_penalty": _joint_pos_penalty,
            "joint_power": _joint_power,
            "nonwheel_contact": _nonwheel_contact,
            "alive": _alive,
            "action_rate": rewards.action_rate,
        }

    def _upright_scale(self, gravity: np.ndarray | None) -> np.ndarray:
        return rewards.upright_scale(gravity, self._num_envs)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 32, "critic": 45 + self._height_scan_dim}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._clear_height_scan_cache()
        state = super().update_state(state)
        done = state.terminated | state.truncated
        if np.any(done):
            done_indices = np.where(done)[0]
            if self._terrain_curriculum_unlocked():
                stats = self._spawn.update_on_done(
                    done_indices, self._backend.get_base_pos()[done_indices]
                )
            else:
                stats = {}
            if stats:
                log = state.info.setdefault("log", {})
                for k, v in stats.items():
                    log[f"terrain_curriculum/{k}"] = float(v)
        return state

    def _terrain_curriculum_unlocked(self) -> bool:
        cfg = self._cfg.command_curriculum
        if not cfg.enabled:
            return True
        return self._command_curriculum_vx_progress >= float(cfg.terrain_unlock_vx_progress)

    def _after_update_state(
        self,
        state: NpEnvState,
        linvel: np.ndarray,
        gyro: np.ndarray,
    ) -> None:
        self._accumulate_command_segments(state.info, linvel, gyro)
        log = state.info.get("log")
        if not isinstance(log, dict):
            return
        self._write_motion_metrics(log, state.info, linvel, gyro)
        self._update_command_curriculum()
        self._write_command_curriculum_metrics(log)
        log["command_curriculum/terrain_unlocked"] = float(self._terrain_curriculum_unlocked())
        log["command_curriculum/low_vx"] = float(self._command_curriculum_low[0])
        log["command_curriculum/high_vx"] = float(self._command_curriculum_high[0])
        log["command_curriculum/low_wz"] = float(self._command_curriculum_low[2])
        log["command_curriculum/high_wz"] = float(self._command_curriculum_high[2])
        log["terrain_curriculum/mean_level"] = float(self._spawn.levels.mean())
        log["terrain_curriculum/max_level"] = float(self._spawn.levels.max())
        log["terrain_curriculum/min_level"] = float(self._spawn.levels.min())

    def _clear_height_scan_cache(self) -> None:
        self._rough_scan_raw = None
        self._rough_scan_base_pos = None
        self._rough_scan_height_obs = None
        self._rough_scan_base_height = None

    def _ensure_height_scan_cache(self, num_obs: int) -> None:
        if (
            self._rough_scan_raw is not None
            and self._rough_scan_base_pos is not None
            and self._rough_scan_raw.shape == (num_obs, self._height_scan_dim)
            and self._rough_scan_base_pos.shape[0] == num_obs
        ):
            return
        raw_heights, base_pos = raw_height_scan_obs(self, num_obs)
        if raw_heights is None or base_pos is None:
            self._rough_scan_raw = None
            self._rough_scan_base_pos = None
            self._rough_scan_height_obs = None
            self._rough_scan_base_height = None
            return
        self._rough_scan_raw = np.asarray(raw_heights, dtype=self._np_dtype)
        self._rough_scan_base_pos = np.asarray(base_pos, dtype=self._np_dtype)
        self._rough_scan_height_obs = None
        self._rough_scan_base_height = None

    def _cached_height_scan_obs(self, num_obs: int) -> np.ndarray:
        self._ensure_height_scan_cache(num_obs)
        if self._rough_scan_raw is None or self._rough_scan_base_pos is None:
            return np.zeros((num_obs, self._height_scan_dim), dtype=self._np_dtype)
        if self._rough_scan_height_obs is None:
            heights = np.clip(
                self._rough_scan_base_pos[:, 2:3]
                - float(self._cfg.terrain_scan.vertical_offset)
                - self._rough_scan_raw,
                -1.0,
                1.0,
            )
            self._rough_scan_height_obs = np.asarray(
                heights * float(self._cfg.terrain_scan.scale), dtype=self._np_dtype
            )
        return self._rough_scan_height_obs

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        accel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        obs_dict = super()._compute_obs(
            info,
            linvel,
            gyro,
            gravity,
            accel,
            dof_pos,
            dof_vel,
        )
        critic = np.concatenate(
            [
                obs_dict["critic"],
                self._cached_height_scan_obs(gyro.shape[0]),
            ],
            axis=1,
            dtype=self._np_dtype,
        )
        return {"obs": obs_dict["obs"], "critic": critic}

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        if num_obs != self._num_envs:
            return super()._reward_base_height_values(num_obs)
        self._ensure_height_scan_cache(num_obs)
        if self._rough_scan_raw is None or self._rough_scan_base_pos is None:
            return super()._reward_base_height_values(num_obs)
        if self._rough_scan_base_height is None:
            self._rough_scan_base_height = np.asarray(
                np.mean(self._rough_scan_base_pos[:, 2:3] - self._rough_scan_raw, axis=1),
                dtype=self._np_dtype,
            )
        return self._rough_scan_base_height

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        terminated = np.zeros((self._num_envs,), dtype=bool)
        if self._cfg.termination_config.nonwheel_contact_termination:
            threshold = float(self._cfg.termination_config.nonwheel_contact_threshold)
            max_steps = max(int(self._cfg.termination_config.nonwheel_contact_max_steps), 1)
            contact_active = np.max(self._nonwheel_contacts, axis=1) > threshold
            self._nonwheel_contact_steps[contact_active] += 1
            self._nonwheel_contact_steps[~contact_active] = 0
            np.logical_or(
                terminated,
                self._nonwheel_contact_steps >= max_steps,
                out=terminated,
            )
        if not self._cfg.termination_config.fall_termination:
            return terminated
        base_height = self._reward_base_height_values(gravity.shape[0])
        np.logical_or(
            terminated,
            (gravity[:, 2] <= float(self._cfg.termination_config.min_up_proj))
            | (base_height <= float(self._cfg.termination_config.min_base_height)),
            out=terminated,
        )
        return terminated

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        if self._cfg.termination_config.terrain_out_of_bounds:
            terrain_scene = self._cfg.scene.terrain
            terrain_cfg = terrain_scene.generator if terrain_scene is not None else None
            np.logical_or(
                truncated,
                terrain_out_of_bounds(
                    self,
                    terrain_cfg,
                    float(self._cfg.termination_config.terrain_distance_buffer),
                ),
                out=truncated,
            )
        return truncated
