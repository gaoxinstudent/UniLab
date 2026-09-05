"""Hydra owner-YAML assertions for Wheelbipe timing and torque contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab import cli
from unilab.base import registry
from unilab.base.registry import apply_cfg_overrides, ensure_registries
from unilab.envs.locomotion.wheelbipe_v14.semantics import SOURCE_V14_REWARD_SCALES
from unilab.training import experiment as experiment_module
from unilab.training.experiment import ExperimentTracker
from unilab.training.reward import extract_reward_config

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
@pytest.mark.parametrize("source_id", sorted(cli.UPSTREAM_WHEELBIPE_CLI_ROUTES))
def test_all_exact_cli_compositions_validate_at_the_registry_owner(
    backend: str,
    source_id: str,
) -> None:
    """Hydra's list representation must not invalidate an exact owner."""

    ensure_registries()
    source_route = cli.UPSTREAM_WHEELBIPE_CLI_ROUTES[source_id]
    route = cli.build_route(source_route.algorithm, source_id, backend)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT / "conf" / route.config_group), version_base="1.3"
    ):
        composed = compose("config", overrides=list(route.generated_overrides))

    owner_name = str(composed.training.task_name)
    owner = registry._envs[owner_name].env_cfg_cls()
    env_override = OmegaConf.to_container(composed.env, resolve=True)
    assert isinstance(env_override, dict)
    apply_cfg_overrides(owner, env_override)
    owner.validate()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
@pytest.mark.parametrize(
    "source_id",
    sorted(
        source_id
        for source_id, route in cli.UPSTREAM_WHEELBIPE_CLI_ROUTES.items()
        if route.algorithm == "ppo"
    ),
)
def test_exact_ppo_reward_injection_matches_the_named_registry_owner(
    backend: str,
    source_id: str,
) -> None:
    """Exercise the reward injection performed by the PPO backend adapter."""

    ensure_registries()
    route = cli.build_route("ppo", source_id, backend)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf" / "ppo"), version_base="1.3"):
        composed = compose("config", overrides=list(route.generated_overrides))

    owner_name = str(composed.training.task_name)
    owner = registry._envs[owner_name].env_cfg_cls()
    env_override = OmegaConf.to_container(composed.env, resolve=True)
    assert isinstance(env_override, dict)
    env_override.update(extract_reward_config(composed))
    apply_cfg_overrides(owner, env_override)
    assert owner.reward_config is not None
    assert dict(owner.reward_config.scales) == dict(type(owner)._SOURCE_REWARD_SCALES)
    owner.validate()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_exact_him_play_route_disables_training_curriculum(backend: str) -> None:
    route = cli.build_route(
        "him_ppo",
        "Robotics-Wheelbipe-V14-Flat-HIM-Play-v0",
        backend,
    )
    assert "env.him_curriculum.enabled=false" in route.generated_overrides
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf" / "custom_ppo"), version_base="1.3"):
        composed = compose("config", overrides=list(route.generated_overrides))
    assert str(composed.training.task_name) == "WheelbipeV14FlatHIMPlay"
    assert bool(composed.env.him_curriculum.enabled) is False

    ensure_registries()
    owner = registry._envs["WheelbipeV14FlatHIMPlay"].env_cfg_cls()
    override = OmegaConf.to_container(composed.env, resolve=True)
    assert isinstance(override, dict)
    apply_cfg_overrides(owner, override)
    owner.validate()


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_him_curriculum_is_serialized_in_run_config(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf" / "custom_ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task=wheelbipe_v14_flat_him/{backend}"])
    monkeypatch.setattr(
        experiment_module,
        "get_device_info_dict",
        lambda: {"platform": "test", "chip": "test", "cpu_total_cores": "1"},
    )
    log_dir = tmp_path / backend
    tracker = ExperimentTracker(
        root_dir=ROOT,
        log_dir=log_dir,
        algo_name="him_ppo",
        task_name="WheelbipeV14FlatHIM",
        sim_backend=backend,
        training_cfg=cfg.training,
        full_cfg=cfg,
        device="cpu",
    )
    tracker.start()
    payload = json.loads((log_dir / "run_config.json").read_text(encoding="utf-8"))
    curriculum = payload["config"]["env"]["him_curriculum"]
    assert curriculum["enabled"] is True
    assert curriculum["reward_key"] == "track_height_exp"
    assert curriculum["num_steps_per_env"] == 24
    assert curriculum["window_size"] == 64
    assert curriculum["stage_min_episodes"] == [500, 500]
    assert curriculum["assist_force_z_stages"] == [160.0, 80.0, 0.0]
    assert curriculum["force_interaction"] == "shared_wrench_buffer_overwrite"


@pytest.mark.parametrize(
    ("source_id", "owner"),
    [
        ("Robotics-Wheelbipe-V14-Flat-v1", "WheelbipeV14FlatV1"),
        ("Robotics-Wheelbipe-V14-Flat-v2", "WheelbipeV14FlatV2"),
        ("Robotics-Wheelbipe-V14-Flat-Play-v2", "WheelbipeV14FlatPlayV2"),
        ("Robotics-Wheelbipe-V14-Rough-v0", "WheelbipeV14RoughV0"),
        ("Robotics-Wheelbipe-V14-Rough-v1", "WheelbipeV14RoughV1"),
        ("Robotics-Wheelbipe-V14-Rough-Play-v0", "WheelbipeV14RoughPlayV0"),
        ("Robotics-Wheelbipe-V14-Rough-Play-v1", "WheelbipeV14RoughPlayV1"),
    ],
)
def test_exact_ppo_cli_composition_preserves_named_variant_contract(
    source_id: str,
    owner: str,
) -> None:
    """Canonical owner YAML must not overwrite exact registry identity."""

    ensure_registries()
    route = cli.build_route("ppo", source_id, "mujoco")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf" / "ppo"), version_base="1.3"):
        composed = compose("config", overrides=list(route.generated_overrides))
    assert str(composed.training.task_name) == owner

    direct = registry._envs[owner].env_cfg_cls()
    routed = registry._envs[owner].env_cfg_cls()
    env_override = OmegaConf.to_container(composed.env, resolve=True)
    assert isinstance(env_override, dict)
    apply_cfg_overrides(routed, env_override)

    assert routed.termination_duration_steps == direct.termination_duration_steps
    assert routed.max_episode_seconds == pytest.approx(direct.max_episode_seconds)
    assert routed.height_range == pytest.approx(direct.height_range)
    assert routed.ctrl_mode_obs_scale == pytest.approx(direct.ctrl_mode_obs_scale)
    assert routed.gimbal.control_mode == direct.gimbal.control_mode
    assert routed.gimbal.heading_target_mode == direct.gimbal.heading_target_mode
    assert routed.gimbal.fixed_heading == pytest.approx(direct.gimbal.fixed_heading)
    assert routed.gimbal.randomize_heading is direct.gimbal.randomize_heading
    assert routed.state_machine.enabled is direct.state_machine.enabled
    assert (
        routed.state_machine.airborne_command_resample.enabled
        is direct.state_machine.airborne_command_resample.enabled
    )
    assert routed.gimbal_spin_translate.enabled is direct.gimbal_spin_translate.enabled
    assert routed.gimbal_spin_translate.relative_envs == pytest.approx(
        direct.gimbal_spin_translate.relative_envs
    )
    assert np.asarray(routed.commands.vel_limit) == pytest.approx(
        np.asarray(direct.commands.vel_limit)
    )
    assert (
        routed.commands.special_mode_start_iterations
        == direct.commands.special_mode_start_iterations
    )
    assert routed.commands.special_mode_probabilities == pytest.approx(
        direct.commands.special_mode_probabilities
    )
    assert routed.commands.gimbal_mode_probability == pytest.approx(
        direct.commands.gimbal_mode_probability
    )
    assert routed.commands.zero_command_probability == pytest.approx(
        direct.commands.zero_command_probability
    )
    assert (
        routed.commands.zero_command_start_iteration == direct.commands.zero_command_start_iteration
    )
    assert routed.commands.source_curriculum_enabled is direct.commands.source_curriculum_enabled
    assert routed.domain_rand.predefined_ground_probability == pytest.approx(
        direct.domain_rand.predefined_ground_probability
    )
    assert routed.domain_rand.randomize_kp is direct.domain_rand.randomize_kp
    assert routed.domain_rand.randomize_kd is direct.domain_rand.randomize_kd
    assert routed.domain_rand.predefined_air_enabled is direct.domain_rand.predefined_air_enabled
    assert routed.domain_rand.predefined_air_probability == pytest.approx(
        direct.domain_rand.predefined_air_probability
    )
    assert (
        routed.domain_rand.predefined_air_command_limits_enabled
        is direct.domain_rand.predefined_air_command_limits_enabled
    )
    assert routed.state_machine.jump_takeoff.enabled is direct.state_machine.jump_takeoff.enabled
    assert routed.state_machine.step_up.enabled is direct.state_machine.step_up.enabled
    assert routed.state_machine.stair.enabled is direct.state_machine.stair.enabled

    routed_boundary = getattr(routed, "rough_terrain_boundary_reset", None)
    direct_boundary = getattr(direct, "rough_terrain_boundary_reset", None)
    assert (routed_boundary is None) is (direct_boundary is None)
    if direct_boundary is not None:
        assert routed_boundary.enabled is direct_boundary.enabled
        assert routed_boundary.margin == pytest.approx(direct_boundary.margin)
        assert routed_boundary.use_inner_terrain_area is direct_boundary.use_inner_terrain_area

    if owner == "WheelbipeV14RoughPlayV0":
        assert direct.domain_rand.randomize_kp is False
        assert direct.domain_rand.randomize_kd is False

    routed_terrain = routed.scene.terrain
    direct_terrain = direct.scene.terrain
    assert (routed_terrain is None) is (direct_terrain is None)
    if direct_terrain is not None:
        assert routed_terrain is not None
        assert routed_terrain.generator is not None
        assert direct_terrain.generator is not None
        for name in ("source_preset", "num_rows", "num_cols", "curriculum"):
            assert getattr(routed_terrain.generator, name) == getattr(
                direct_terrain.generator, name
            )


@pytest.mark.parametrize(
    ("config_dir", "task", "sim_dt", "ctrl_dt", "leg_limit", "wheel_limit"),
    [
        ("ppo", "wheelbipe_v14_flat/mujoco", 0.005, 0.02, 40.0, 5.0),
        ("ppo", "wheelbipe_v14_flat/motrix", 0.005, 0.02, 40.0, 5.0),
        ("ppo", "wheelbipe_v14_rough/mujoco", 0.005, 0.02, 40.0, 5.0),
        ("ppo", "wheelbipe_v14_rough/motrix", 0.005, 0.02, 40.0, 5.0),
        ("custom_ppo", "wheelbipe_v14_flat_him/mujoco", 0.005, 0.02, 40.0, 5.0),
        ("custom_ppo", "wheelbipe_v14_flat_him/motrix", 0.005, 0.02, 40.0, 5.0),
        ("custom_ppo", "wheelbipe_v14_flat_dreamwaq/mujoco", 0.005, 0.02, 40.0, 5.0),
        ("custom_ppo", "wheelbipe_v14_flat_dreamwaq/motrix", 0.005, 0.02, 40.0, 5.0),
        ("custom_ppo", "wheelbipe_v14_flat_np3o/mujoco", 0.005, 0.02, 40.0, 5.0),
        ("custom_ppo", "wheelbipe_v14_flat_np3o/motrix", 0.005, 0.02, 40.0, 5.0),
    ],
)
def test_wheelbipe_owner_yaml_declares_timing_and_torque(
    config_dir: str,
    task: str,
    sim_dt: float,
    ctrl_dt: float,
    leg_limit: float,
    wheel_limit: float,
) -> None:
    """Timing and owner torque values must not be hidden in Python defaults."""

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf" / config_dir), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task}"])

    assert float(cfg.env.sim_dt) == pytest.approx(sim_dt)
    assert float(cfg.env.ctrl_dt) == pytest.approx(ctrl_dt)
    assert float(cfg.env.control_config.leg_torque_limit) == pytest.approx(leg_limit)
    assert float(cfg.env.control_config.wheel_torque_limit) == pytest.approx(wheel_limit)

    if config_dir == "ppo":
        assert cfg.env.delay_profile == "source_v14_physics"
        assert cfg.env.delay_range_semantics == "exclusive"
        assert cfg.env.obs_delay_step_unit == "physics"
        assert bool(cfg.env.use_obs_delay) is True
        assert bool(cfg.env.use_act_delay) is True
        assert cfg.env.delay_range_semantics == "exclusive"
        assert cfg.env.obs_delay_step_unit == "physics"
        assert bool(cfg.env.use_obs_delay) is True
        assert bool(cfg.env.use_act_delay) is True
        assert bool(cfg.env.domain_rand.source_guide_material_requested) is True
        assert (
            str(cfg.env.domain_rand.source_guide_material_missing_target_status)
            == "not_applicable_missing_target"
        )
        assert int(cfg.env.domain_rand.guide_material_num_buckets) == 8
        assert bool(cfg.env.domain_rand.joint_friction.enabled) is True
        assert str(cfg.env.domain_rand.joint_friction.coulomb_backend_mode) == (
            "static_to_frictionloss_dynamic_collapsed"
        )
        expected_viscous_mode = "unsupported_omitted" if task.endswith("/motrix") else "dof_damping"
        assert str(cfg.env.domain_rand.joint_friction.viscous_backend_mode) == expected_viscous_mode
        assert list(cfg.env.domain_rand.joint_friction.front_static_range) == pytest.approx(
            [0.25, 1.0]
        )
        assert list(cfg.env.domain_rand.joint_friction.wheel_viscous_range) == pytest.approx(
            [0.0, 0.01]
        )
        assert list(cfg.env.domain_rand.joint_friction.inactive_viscous_range) == pytest.approx(
            [0.01, 0.025]
        )
        assert str(cfg.env.domain_rand.source_passive_gain_conversion_status) == (
            "unsupported_unmaterialized_actuator"
        )
        if config_dir == "ppo":
            assert cfg.env.training_semantics == "source_v14"
            assert int(cfg.algo.num_envs) == 4096
            assert float(cfg.env.control_config.spring_damping) == pytest.approx(50.0)
            assert float(cfg.env.noise_config.level) == pytest.approx(1.0)
            assert bool(cfg.env.domain_rand.randomize_body_mass) is True
            assert bool(cfg.env.domain_rand.randomize_body_material) is True
            assert int(cfg.env.domain_rand.material_num_buckets) == 64
            assert bool(cfg.env.domain_rand.material_make_consistent) is True
            assert bool(cfg.env.commands.source_curriculum_enabled) is True
            assert int(cfg.env.domain_rand.control_randomization_min_steps) == 720
            assert float(cfg.env.domain_rand.predefined_ground_command_seconds) == pytest.approx(
                1.5
            )
            assert list(cfg.env.domain_rand.predefined_ground_command_x_range) == pytest.approx(
                [-1.0, 1.0]
            )
            assert float(cfg.env.domain_rand.predefined_air_command_seconds) == pytest.approx(3.0)
            assert list(cfg.env.domain_rand.predefined_air_height_range) == pytest.approx(
                [0.18, 0.43]
            )
            assert list(cfg.env.height_range) == pytest.approx([0.20, 0.42])
            assert list(cfg.env.commands.vel_limit[0]) == pytest.approx(
                [-2.7, 0.0, -2.0 * 3.141592653589793]
            )
            expected_scales = dict(SOURCE_V14_REWARD_SCALES)
            if "rough" in task:
                expected_scales["joint_torque"] = -1.0e-5
                expected_scales["wheel_power"] = -1.0e-5
                expected_scales["stand_still_lin_vel"] = -1.0
                # The pinned rough runs strengthen both squared tracking
                # penalties relative to the flat owner's -0.1.
                expected_scales["track_lin_vel_xy_square"] = -1.0
                expected_scales["track_ang_vel_z_square"] = -1.0
                assert list(cfg.env.commands.special_mode_start_iterations) == [0, 0, 0]
                assert list(cfg.env.commands.special_mode_probabilities) == pytest.approx(
                    [0.15, 0.15, 0.30]
                )
                assert float(cfg.env.commands.gimbal_mode_probability) == pytest.approx(0.0)
                assert list(cfg.env.ctrl_mode_obs_scale) == pytest.approx(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]
                )
                assert bool(cfg.env.gimbal_spin_translate.enabled) is False
                assert bool(cfg.env.rough_terrain_boundary_reset.enabled) is True
                assert float(cfg.env.rough_terrain_boundary_reset.margin) == pytest.approx(0.5)
                assert bool(cfg.env.rough_terrain_boundary_reset.use_inner_terrain_area) is False
            else:
                assert list(cfg.env.commands.special_mode_probabilities) == pytest.approx(
                    [0.15, 0.15, 0.30]
                )
                assert float(cfg.env.commands.gimbal_mode_probability) == pytest.approx(0.0)
                assert list(cfg.env.ctrl_mode_obs_scale) == pytest.approx(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]
                )
            resolved_scales = OmegaConf.to_container(cfg.reward.scales, resolve=True)
            assert resolved_scales == expected_scales
        if config_dir == "custom_ppo":
            assert cfg.env.training_semantics == "source_v14"
            assert float(cfg.env.noise_config.level) == pytest.approx(1.0)
            assert bool(cfg.env.domain_rand.randomize_body_mass) is True
            assert bool(cfg.env.domain_rand.random_com) is True
            assert bool(cfg.env.domain_rand.randomize_body_material) is True
            assert bool(cfg.env.domain_rand.use_leg_random_start) is True
            assert bool(cfg.env.domain_rand.use_predefined_leg_random_start) is True
            assert bool(cfg.env.domain_rand.source_external_force_enabled) is True
            assert bool(cfg.env.commands.source_curriculum_enabled) is True
            assert list(cfg.env.commands.special_mode_probabilities) == pytest.approx(
                [0.15, 0.15, 0.30]
            )
            assert bool(cfg.env.ctrl_mode_obs_enabled) is False
            assert int(cfg.env.ctrl_mode_obs_dim) == 0
            resolved_scales = OmegaConf.to_container(cfg.reward.scales, resolve=True)
            assert resolved_scales == dict(SOURCE_V14_REWARD_SCALES)
            if task.endswith("/motrix"):
                assert bool(cfg.env.motrix_disable_equality) is True
            if str(cfg.algo.algorithm_name) == "him":
                curriculum = cfg.env.him_curriculum
                assert bool(curriculum.enabled) is True
                assert str(curriculum.reward_key) == "track_height_exp"
                assert int(curriculum.num_steps_per_env) == 24
                assert int(curriculum.window_size) == 64
                assert int(curriculum.min_stage_episodes) == 64
                assert bool(curriculum.normalize_by_episode_length) is True
                assert bool(curriculum.restore_defaults_after_final_threshold) is True
                assert bool(curriculum.assist_apply_on_compute) is True
                assert str(curriculum.assist_body_name) == "base_link"
                assert str(curriculum.force_interaction) == ("shared_wrench_buffer_overwrite")
                assert OmegaConf.to_container(curriculum.reward_stage_weights, resolve=True) == [
                    {"track_height_exp": 1.0, "track_height_exp_tight": 1.0},
                    {"track_height_exp": 0.8, "track_height_exp_tight": 0.6},
                ]
                assert list(curriculum.assist_force_z_stages) == pytest.approx([160.0, 80.0, 0.0])
                assert list(curriculum.thresholds) == pytest.approx([0.4, 0.4])
                assert list(curriculum.stage_min_episodes) == [500, 500]
        if config_dir == "custom_ppo" and str(cfg.algo.algorithm_name) == "np3o":
            assert float(cfg.env.np3o_tilt_limit_deg) == pytest.approx(20.0)

    # Ensure the assertion itself is based on resolved owner values rather
    # than an interpolation/default that Hydra could leave unresolved.
    assert isinstance(OmegaConf.to_container(cfg.env, resolve=True), dict)


def test_rough_owner_preserves_source_ctrl_mode_observation_scale() -> None:
    """The rough owner must keep the source V14 7D mode-tail scaling."""

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf" / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=wheelbipe_v14_rough/mujoco"])

    assert list(cfg.env.ctrl_mode_obs_scale) == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]
    )


def test_rough_owner_does_not_gate_velocity_tracking_by_height() -> None:
    """The source rough owner leaves velocity tracking ungated by height."""

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf" / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=wheelbipe_v14_rough/mujoco"])

    assert bool(cfg.env.vel_height_gate_enabled) is False
