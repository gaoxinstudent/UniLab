from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

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
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.envs.locomotion.real68.balance import (
    NONWHEEL_CONTACT_SENSORS,
    WHEEL_CONTACT_SENSORS,
    FlatTerminationConfig,
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
    flat,
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
class RoughTerminationConfig(FlatTerminationConfig):
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
    curriculum: bool = True
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 6
    num_cols: int = 6
    border_width: float = 1.0
    add_lights: bool = True
    horizontal_scale: float = 0.1

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.25),
            "pyramid_stairs": pyramid_stairs(
                proportion=0.15,
                step_height_range=(0.01, 0.08),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.15,
                step_height_range=(0.01, 0.08),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.15,
                slope_range=(0.0, 0.18),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.15,
                slope_range=(0.0, 0.18),
                platform_width=2.0,
                border_width=0.2,
            ),
            "random_rough": random_rough(
                proportion=0.075,
                noise_range=(0.005, 0.03),
                noise_step=0.01,
                border_width=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.075,
                amplitude_range=(0.0, 0.06),
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


@registry.envcfg("Real68Balance")
@dataclass
class Real68BalanceUnifiedCfg(Real68BalanceRoughCfg):
    """Single Sim2Real training owner spanning flat and rough terrain."""


class Real68BalanceRoughDomainRandomizationProvider(Real68BalanceDomainRandomizationProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))

        xy_low, xy_high = env.cfg.domain_rand.reset_pos_xy_range
        qpos[:, 0:2] += np.random.uniform(xy_low, xy_high, (num_reset, 2))
        z_low, z_high = env.cfg.domain_rand.reset_height_offset_range
        qpos[:, 2] = float(env.cfg.recovery.initial_base_height) + np.random.uniform(
            z_low, z_high, (num_reset,)
        )

        roll_low, roll_high = env.cfg.domain_rand.reset_roll_range
        pitch_low, pitch_high = env.cfg.domain_rand.reset_pitch_range
        yaw_low, yaw_high = env.cfg.domain_rand.reset_yaw_range
        roll = np.random.uniform(roll_low, roll_high, (num_reset,))
        pitch = np.random.uniform(pitch_low, pitch_high, (num_reset,))
        yaw = np.random.uniform(yaw_low, yaw_high, (num_reset,))
        recovering = np.zeros((num_reset,), dtype=bool)
        recovery_pose_ids = np.full((num_reset,), -1, dtype=np.int32)
        recovery_cfg = env.cfg.recovery
        recovery_probability = env._recovery_reset_probability()
        if recovery_cfg.enabled and recovery_probability > 0.0:
            recovering = np.random.uniform(size=(num_reset,)) < recovery_probability
            if np.any(recovering):
                recovery_roll, recovery_pitch, recovery_height, recovery_pose_ids = (
                    env._sample_recovery_reset_poses(num_reset, recovering)
                )
                roll[recovering] = recovery_roll[recovering]
                pitch[recovering] = recovery_pitch[recovering]
                qpos[recovering, 2] = recovery_height[recovering]
        recovery_joint_pose_ids = env._sample_recovery_joint_poses(qpos, recovering)
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3], yaw=yaw)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_quat_from_euler_xyz(roll, pitch, yaw))
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])

        limit = float(env.cfg.domain_rand.reset_qvel_limit)
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-limit, limit, size=(num_reset, 6)),
            dtype=get_global_dtype(),
        )

        commands = env.sample_velocity_commands(num_reset)
        if env._last_command_clip_scale.shape[0] == num_reset:
            env._command_clip_scale[env_ids] = env._last_command_clip_scale
        height_commands = env.sample_height_commands(num_reset, commands=commands)
        effective_commands = commands.copy()
        effective_commands[recovering] = 0.0
        dr_cfg = env.cfg.domain_rand
        dr_scale = env._domain_rand_scale()
        kp_scale = np.ones((num_reset, 1), dtype=get_global_dtype())
        kd_scale = np.ones((num_reset, 1), dtype=get_global_dtype())
        motor_strength = np.ones((num_reset, env._num_action), dtype=get_global_dtype())
        action_delay_steps = np.zeros((num_reset, 1), dtype=get_global_dtype())
        if dr_cfg.randomize_control_kp:
            sampled = np.random.uniform(*dr_cfg.kp_multiplier_range, size=(num_reset, 1))
            kp_scale[:] = 1.0 + dr_scale * (sampled - 1.0)
        if dr_cfg.randomize_control_kd:
            sampled = np.random.uniform(*dr_cfg.kd_multiplier_range, size=(num_reset, 1))
            kd_scale[:] = 1.0 + dr_scale * (sampled - 1.0)
        if dr_cfg.randomize_motor_strength:
            sampled = np.random.uniform(
                *dr_cfg.motor_strength_range, size=(num_reset, env._num_action)
            )
            motor_strength[:] = 1.0 + dr_scale * (sampled - 1.0)
        if dr_cfg.randomize_action_delay:
            low, high = (int(value) for value in dr_cfg.action_delay_steps)
            sampled = np.random.randint(low, high + 1, size=(num_reset, 1))
            action_delay_steps[:] = np.where(
                np.random.uniform(size=(num_reset, 1)) < dr_scale, sampled, 0
            )

        randomization = build_common_reset_randomization(
            env,
            num_reset,
            base_geom_friction=env._base_geom_friction,
            ground_geom_id=env._ground_geom_id,
        )
        mass_delta = np.zeros((num_reset, 1), dtype=get_global_dtype())
        com_offset = np.zeros((num_reset, 3), dtype=get_global_dtype())
        ground_friction = np.full(
            (num_reset, 1), env._base_geom_friction[env._ground_geom_id, 0], dtype=get_global_dtype()
        )
        if randomization is not None:
            if randomization.base_mass_delta is not None:
                randomization.base_mass_delta *= dr_scale
                mass_delta[:, 0] = randomization.base_mass_delta
            if randomization.base_com_offset is not None:
                randomization.base_com_offset *= dr_scale
                com_offset[:] = randomization.base_com_offset
            if randomization.geom_friction is not None:
                baseline = np.broadcast_to(
                    env._base_geom_friction, randomization.geom_friction.shape
                )
                randomization.geom_friction[:] = baseline + dr_scale * (
                    randomization.geom_friction - baseline
                )
                ground_friction[:, 0] = randomization.geom_friction[:, env._ground_geom_id, 0]
        info_updates = {
            "commands": effective_commands,
            "tracking_commands": commands,
            "height_commands": height_commands,
            "recovery_active": recovering,
            "recovery_eligible": recovering,
            "recovery_completed": np.zeros((num_reset,), dtype=bool),
            "recovery_pose_ids": recovery_pose_ids,
            "recovery_joint_pose_ids": recovery_joint_pose_ids,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "qacc": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "dr_mass_delta": mass_delta,
            "dr_com_offset": com_offset,
            "dr_ground_friction": ground_friction,
            "dr_kp_scale": kp_scale,
            "dr_kd_scale": kd_scale,
            "dr_motor_strength": motor_strength,
            "dr_action_delay_steps": action_delay_steps,
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
            randomization=randomization,
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
        self._terrain_bootstrap_type_col: int | None = None
        self._terrain_type_pending_unlock = np.zeros((num_envs,), dtype=bool)
        if terrain_origins is not None and terrain_generator is not None:
            type_names = tuple(terrain_generator.sub_terrains)
            bootstrap_type = cfg.terrain_curriculum.bootstrap_type
            if bootstrap_type is not None:
                if not terrain_generator.curriculum:
                    raise ValueError(
                        "terrain_curriculum.bootstrap_type requires "
                        "scene.terrain.generator.curriculum=true"
                    )
                if bootstrap_type not in type_names:
                    raise ValueError(
                        f"terrain_curriculum.bootstrap_type={bootstrap_type!r} is not in "
                        f"scene.terrain.generator.sub_terrains={type_names}"
                    )
                self._terrain_bootstrap_type_col = type_names.index(bootstrap_type)
            terrain_locked = bool(
                self._terrain_bootstrap_type_col is not None
                and not self._terrain_curriculum_unlocked()
            )
            self._spawn = TerrainSpawnManager(
                num_envs,
                terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=getattr(self._backend, "terrain_surface_sampler", None),
                type_probabilities=np.asarray(
                    [terrain.proportion for terrain in terrain_generator.sub_terrains.values()],
                    dtype=np.float64,
                ),
                initial_type_col=(
                    self._terrain_bootstrap_type_col if terrain_locked else None
                ),
            )
            self._terrain_type_pending_unlock[:] = terrain_locked
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)
        self._critic_history = np.zeros(
            (num_envs, self._critic_one_step_dim), dtype=self._np_dtype
        )

    def _make_dr_provider(self) -> Real68BalanceRoughDomainRandomizationProvider:
        return Real68BalanceRoughDomainRandomizationProvider()

    def training_state_dict(self) -> dict[str, Any]:
        state = super().training_state_dict()
        if isinstance(self._spawn, TerrainSpawnManager):
            state["terrain_curriculum"] = {
                "version": 1,
                "spawn": self._spawn.training_state_dict(),
                "type_pending_unlock": self._terrain_type_pending_unlock.tolist(),
            }
        return state

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        super().load_training_state_dict(state)
        terrain = state.get("terrain_curriculum")
        if terrain is None:
            if isinstance(self._spawn, TerrainSpawnManager):
                raise ValueError("Real68 rough training state is missing terrain_curriculum")
            return
        if not isinstance(self._spawn, TerrainSpawnManager):
            raise ValueError("Checkpoint has terrain curriculum state but env has no terrain")
        terrain_state = cast(dict[str, Any], terrain)
        version = int(terrain_state.get("version", 0))
        if version != 1:
            raise ValueError(f"Unsupported Real68 terrain state version: {version}")
        pending = np.asarray(terrain_state["type_pending_unlock"], dtype=bool)
        if pending.shape != self._terrain_type_pending_unlock.shape:
            raise ValueError(
                "Real68 terrain pending-unlock shape mismatch: "
                f"{pending.shape} != {self._terrain_type_pending_unlock.shape}"
            )
        self._spawn.load_training_state_dict(
            cast(dict[str, object], terrain_state["spawn"])
        )
        self._terrain_type_pending_unlock[:] = pending

    def get_playback_root_xy_offsets(self) -> np.ndarray | None:
        if not isinstance(self._spawn, TerrainSpawnManager):
            return None
        env_ids = np.arange(self._num_envs, dtype=np.int32)
        return np.asarray(self._spawn.origins_for(env_ids)[:, :2], dtype=self._np_dtype)

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        self._nonwheel_contact_steps[env_ids] = 0
        return obs, info

    def _before_autoreset(self, done: np.ndarray) -> None:
        super()._before_autoreset(done)
        if not isinstance(self._spawn, TerrainSpawnManager):
            return
        env_ids = np.flatnonzero(done).astype(np.int32)
        if self._terrain_curriculum_unlocked():
            pending = self._terrain_type_pending_unlock[env_ids]
            unlock_ids = env_ids[pending]
            if unlock_ids.size:
                self._spawn.resample_type_cols(unlock_ids)
                self._terrain_type_pending_unlock[unlock_ids] = False
            return
        if self._terrain_bootstrap_type_col is not None:
            self._spawn.set_type_col(env_ids, self._terrain_bootstrap_type_col)
            self._terrain_type_pending_unlock[env_ids] = True

    def _init_reward_functions(self) -> None:
        def gated(fn):
            return lambda ctx: fn(ctx) * self._upright_scale(ctx.gravity)

        super()._init_reward_functions()
        self._reward_fns["tracking_ang_vel"] = rewards.tracking_ang_vel
        for name in (
            "yaw_rate_when_uncommanded",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "torques",
            "wheel_vel",
            "posture",
            "leg_symmetry",
            "height_tracking",
            "joint_pos_penalty",
            "joint_power",
            "nonwheel_contact",
            "alive",
        ):
            self._reward_fns[name] = gated(self._reward_fns[name])

    def _upright_scale(self, gravity: np.ndarray | None) -> np.ndarray:
        return rewards.upright_scale(gravity, self._num_envs)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": int(self._cfg.history.num_actor_history) * 28,
            "critic": self._critic_one_step_dim,
        }

    @property
    def _critic_one_step_dim(self) -> int:
        return 59 + self._height_scan_dim

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._clear_height_scan_cache()
        state = super().update_state(state)
        done = state.terminated | state.truncated
        if np.any(done):
            done_indices = np.where(done)[0]
            if self._terrain_curriculum_unlocked():
                promote_performance = None
                demote_performance = None
                terrain_cfg = self._cfg.terrain_curriculum
                if terrain_cfg.performance_gating:
                    steps = np.maximum(self._segment_steps[done_indices], 1)
                    vx_error = self._segment_vx_error_sum[done_indices] / steps
                    tilt_rate = self._segment_tilt_sum[done_indices] / steps
                    contact_rate = self._segment_nonwheel_contact_sum[done_indices] / steps
                    promote_performance = (
                        (vx_error <= float(terrain_cfg.max_vx_error))
                        & (tilt_rate <= float(terrain_cfg.max_tilt_rate))
                        & (
                            contact_rate
                            <= float(terrain_cfg.max_nonwheel_contact_rate)
                        )
                        & ~self._segment_recovery_seen[done_indices]
                    )
                    recovery_timeout = np.asarray(
                        state.info.get(
                            "termination_recovery_timeout",
                            np.zeros((self._num_envs,), dtype=bool),
                        ),
                        dtype=bool,
                    )[done_indices]
                    demote_performance = (~promote_performance) | recovery_timeout
                stats = self._spawn.update_on_done(
                    done_indices,
                    self._backend.get_base_pos()[done_indices],
                    promote_performance=promote_performance,
                    demote_performance=demote_performance,
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
        spawn = cast(TerrainSpawnManager, self._spawn)
        log["terrain_curriculum/mean_level"] = float(spawn.levels.mean())
        log["terrain_curriculum/max_level"] = float(spawn.levels.max())
        log["terrain_curriculum/min_level"] = float(spawn.levels.min())
        log["terrain_curriculum/bootstrap_type_frac"] = float(
            np.mean(self._terrain_type_pending_unlock)
        )

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

    def _augment_critic_frame(self, critic: np.ndarray, num_obs: int) -> np.ndarray:
        return np.concatenate(
            [
                critic,
                self._cached_height_scan_obs(num_obs),
            ],
            axis=1,
            dtype=self._np_dtype,
        )

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

    def _compute_termination_causes(self, gravity: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self._cfg.termination_config
        recovery_cfg = self._cfg.recovery
        protected = (
            self._recovery_active | self._last_recovery_completed
            if recovery_cfg.enabled
            else np.zeros((self._num_envs,), dtype=bool)
        )
        nonwheel_contact = np.zeros((self._num_envs,), dtype=bool)
        if self._cfg.termination_config.nonwheel_contact_termination:
            threshold = float(cfg.nonwheel_contact_threshold)
            max_steps = max(int(cfg.nonwheel_contact_max_steps), 1)
            contact_active = np.max(self._nonwheel_contacts, axis=1) > threshold
            self._nonwheel_contact_steps[contact_active] += 1
            self._nonwheel_contact_steps[~contact_active] = 0
            nonwheel_contact = self._nonwheel_contact_steps >= max_steps
        base_height = self._reward_base_height_values(gravity.shape[0])
        tilt = np.zeros((self._num_envs,), dtype=bool)
        height_low = np.zeros((self._num_envs,), dtype=bool)
        if cfg.fall_termination:
            tilt = gravity[:, 2] <= float(cfg.min_up_proj)
            height_low = base_height <= float(cfg.min_base_height)
        if recovery_cfg.enabled:
            tilt &= ~protected
            height_low &= ~protected
            nonwheel_contact &= ~protected
            timeout_steps = max(int(round(self._recovery_timeout_seconds() / self._cfg.ctrl_dt)), 1)
            recovery_timeout = self._recovery_active & (
                self._recovery_elapsed_steps >= timeout_steps
            )
        else:
            recovery_timeout = np.zeros((self._num_envs,), dtype=bool)
        terminated = tilt | height_low | nonwheel_contact | recovery_timeout
        return {
            "tilt": np.asarray(tilt, dtype=bool),
            "height": np.asarray(height_low, dtype=bool),
            "height_low": np.asarray(height_low, dtype=bool),
            "height_high": np.zeros((self._num_envs,), dtype=bool),
            "nonwheel_contact": np.asarray(nonwheel_contact, dtype=bool),
            "recovery_timeout": np.asarray(recovery_timeout, dtype=bool),
            "terminated": np.asarray(terminated, dtype=bool),
        }

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


@registry.env("Real68Balance", sim_backend="mujoco")
class Real68BalanceUnifiedEnv(Real68BalanceRoughEnv):
    _cfg: Real68BalanceUnifiedCfg
