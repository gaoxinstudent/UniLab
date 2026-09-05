"""Config system verification tests.

These tests enforce that:
1. Base Hydra configs compose without legacy config groups.
2. Every supported runtime variant resolves through exactly one task owner file.
3. Final reward/env/algo sections are present on the composed config, not mounted by Python glue.
4. Backend-specific hyperparameters preserve the intended pre-refactor behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab import cli

CONF_DIR = Path(__file__).parent.parent.parent / "conf"
_BACKENDS = ("mujoco", "mjwarp", "motrix")


def _expected_backend_from_variant(name: str) -> str | None:
    for backend in _BACKENDS:
        if name == backend or name.startswith(f"{backend}_"):
            return backend
    return None


def _compose(algo_dir: str, config_name: str = "config", overrides: list[str] | None = None):
    normalized_overrides = _normalize_overrides(algo_dir, overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / algo_dir), version_base="1.3"):
        return compose(config_name, overrides=normalized_overrides)


def _normalize_overrides(algo_dir: str, overrides: list[str] | None) -> list[str]:
    algo = "sac"
    normalized: list[str] = []
    task_selected = False

    for override in overrides or []:
        if override.startswith("algo="):
            algo = override.split("=", 1)[1]
            normalized.append(override)
            continue
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        if algo_dir == "offpolicy":
            normalized.append(f"task={algo}/g1_walk_flat/mujoco")
        elif algo_dir == "custom_ppo":
            normalized.append("task=wheelbipe_v14_flat_him/mujoco")
        else:
            normalized.append("task=go1_joystick_flat/mujoco")

    return normalized


def _assert_reward_populated(cfg, label: str):
    assert hasattr(cfg, "reward"), f"{label} missing cfg.reward"
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    assert isinstance(reward_dict, dict), f"{label} reward must resolve to mapping"
    assert "scales" in reward_dict, f"{label} reward must contain scales"
    assert len(reward_dict["scales"]) > 0, f"{label} reward.scales must be non-empty"


def _supported_task_cases() -> list[tuple[str, str, str, str, str, list[str]]]:
    cases: list[tuple[str, str, str, str, str, list[str]]] = []

    for algo_dir in ["ppo", "appo"]:
        root = CONF_DIR / algo_dir / "task"
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        algo_dir,
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )

    offpolicy_root = CONF_DIR / "offpolicy" / "task"
    for algo_root in sorted(path for path in offpolicy_root.iterdir() if path.is_dir()):
        for task_dir in sorted(path for path in algo_root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        "offpolicy",
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [
                            f"algo={algo_root.name}",
                            f"task={algo_root.name}/{task_dir.name}/{backend_file.stem}",
                        ],
                    )
                )

    # WheelBipe's history-policy owners live in their own Hydra config group;
    # compose them here as first-class task owners so support tooling cannot
    # report a file that the runtime test suite never resolves.
    custom_root = CONF_DIR / "custom_ppo" / "task"
    if custom_root.is_dir():
        for task_dir in sorted(path for path in custom_root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        "custom_ppo",
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )

    return cases


@pytest.mark.parametrize(
    "algo_dir,config_name",
    [
        ("offpolicy", "config"),
        ("appo", "config"),
        ("ppo", "config"),
        ("custom_ppo", "config"),
    ],
)
def test_algo_config_composes(algo_dir: str, config_name: str):
    cfg = _compose(algo_dir, config_name)
    assert cfg.training.task_name
    assert cfg.training.sim_backend == "mujoco"


@pytest.mark.parametrize(
    ("task", "algorithm", "history", "costs"),
    [
        ("wheelbipe_v14_flat_him", "him", 5, 0),
        ("wheelbipe_v14_flat_dreamwaq", "dreamwaq", 5, 0),
        ("wheelbipe_v14_flat_np3o", "np3o", 10, 5),
    ],
)
@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_custom_wheelbipe_owner_composes_declared_timing_and_cost_contract(
    task: str, algorithm: str, history: int, costs: int, backend: str
) -> None:
    """Custom public routes stay aligned with their owner YAML contracts."""

    cfg = _compose("custom_ppo", overrides=[f"task={task}/{backend}"])

    assert cfg.training.task_name
    assert cfg.training.sim_backend == backend
    assert cfg.algo.algorithm_name == algorithm
    assert cfg.algo.num_actor_history == history
    assert cfg.algo.num_costs == costs
    assert cfg.env.num_costs == costs
    assert cfg.env.sim_dt == pytest.approx(0.005)
    assert cfg.env.ctrl_dt == pytest.approx(0.02)
    assert cfg.env.delay_profile == "source_v14_physics"
    assert cfg.env.delay_range_semantics == "exclusive"
    assert cfg.env.use_obs_delay is True
    assert cfg.env.use_act_delay is True


@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_wheelbipe_flat_ppo_owner_defaults_to_long_run_batch(backend: str) -> None:
    """Both backend owners keep the validated 4096-env long-run default."""

    cfg = _compose("ppo", overrides=[f"task=wheelbipe_v14_flat/{backend}"])

    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 20000
    assert cfg.env.sim_dt == pytest.approx(0.005)
    assert cfg.env.delay_profile == "source_v14_physics"
    assert cfg.env.delay_range_semantics == "exclusive"
    assert cfg.env.obs_delay_step_unit == "physics"
    assert cfg.env.use_obs_delay is True
    assert cfg.env.use_act_delay is True


def test_custom_source_profiles_keep_checked_in_long_run_budgets() -> None:
    """Source runner budgets are explicit opt-in Hydra profiles."""

    cases = {
        "source_dreamwaq_long": ("dreamwaq", 4096, 10000, 500),
        "source_him_long": ("him", 4096, 5000, 500),
        "source_np3o_barlow_long": ("np3o", 4096, 3000, 500),
    }
    for profile, (algorithm, num_envs, iterations, save_interval) in cases.items():
        cfg = _compose(
            "custom_ppo",
            overrides=[
                f"profile={profile}",
                f"task=wheelbipe_v14_flat_{'np3o' if algorithm == 'np3o' else algorithm}/mujoco",
            ],
        )
        assert cfg.algo.algorithm_name == algorithm
        assert cfg.algo.num_envs == num_envs
        assert cfg.algo.num_steps_per_env == 24
        assert cfg.algo.max_iterations == iterations
        assert cfg.algo.save_interval == save_interval
        assert cfg.algo.history_reset_mode == "source_zero_current"
        assert list(cfg.algo.policy.actor_hidden_dims) == [512, 256, 128]
        assert list(cfg.algo.policy.critic_hidden_dims) == [512, 256, 128]
        assert cfg.algo.policy.init_noise_std == pytest.approx(1.0)
        assert cfg.algo.policy.activation == "elu"
        assert cfg.algo.algorithm.value_loss_coef == pytest.approx(4.0)
        assert cfg.algo.algorithm.use_clipped_value_loss is True
        assert cfg.algo.algorithm.clip_param == pytest.approx(0.2)
        assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.005)
        assert cfg.algo.algorithm.num_learning_epochs == 5
        assert cfg.algo.algorithm.num_mini_batches == 4
        assert cfg.algo.algorithm.learning_rate == pytest.approx(1.0e-4)
        assert cfg.algo.algorithm.schedule == "adaptive"
        assert cfg.algo.algorithm.gamma == pytest.approx(0.99)
        assert cfg.algo.algorithm.lam == pytest.approx(0.95)
        assert cfg.algo.algorithm.desired_kl == pytest.approx(0.01)
        assert cfg.algo.algorithm.max_grad_norm == pytest.approx(1.0)
    np3o = _compose(
        "custom_ppo",
        overrides=[
            "profile=source_np3o_barlow_long",
            "task=wheelbipe_v14_flat_np3o/mujoco",
        ],
    )
    assert np3o.algo.policy_architecture == "source_barlow"
    assert np3o.algo.policy.source_barlow.num_hist == 10
    assert np3o.algo.policy.source_barlow.num_priv_latent == 4

    dreamwaq = _compose(
        "custom_ppo",
        overrides=[
            "profile=source_dreamwaq_long",
            "task=wheelbipe_v14_flat_dreamwaq/mujoco",
        ],
    )
    assert list(dreamwaq.algo.policy.actor_hidden_dims) == [512, 256, 128]
    assert list(dreamwaq.algo.policy.critic_hidden_dims) == [512, 256, 128]
    assert list(dreamwaq.algo.policy.cenet_encoder_hidden_dims) == [256, 128, 64]
    assert list(dreamwaq.algo.policy.cenet_decoder_hidden_dims) == [64, 128, 256]


@pytest.mark.parametrize(
    ("task", "algo", "profile"),
    [
        ("Robotics-Wheelbipe-V14-Flat-HIM-v0", "him_ppo", "source_him_long"),
        (
            "Robotics-Wheelbipe-V14-Flat-DreamWaQ-v0",
            "dreamwaq",
            "source_dreamwaq_long",
        ),
        (
            "Robotics-Wheelbipe-V14-Flat-NP3OBarlow-v0",
            "np3o",
            "source_np3o_barlow_long",
        ),
    ],
)
def test_exact_custom_route_matches_canonical_source_profile(
    task: str,
    algo: str,
    profile: str,
) -> None:
    exact_route = cli.build_route(algo, task, "mujoco")
    canonical_route = cli.build_route(
        algo,
        "wheelbipe_v14_flat",
        "mujoco",
        profile=profile,
    )
    exact = _compose("custom_ppo", overrides=list(exact_route.generated_overrides))
    canonical = _compose("custom_ppo", overrides=list(canonical_route.generated_overrides))

    assert OmegaConf.to_container(exact.algo, resolve=True) == OmegaConf.to_container(
        canonical.algo,
        resolve=True,
    )
    assert exact.algo.history_reset_mode == "source_zero_current"
    if algo == "np3o":
        assert exact.algo.policy_architecture == "source_barlow"
        assert exact.algo.policy.architecture == "source_barlow"


def test_legacy_config_groups_removed():
    for path in [
        CONF_DIR / "ppo" / "reward",
        CONF_DIR / "ppo" / "backend_task_preset",
        CONF_DIR / "ppo" / "algo_preset",
        CONF_DIR / "ppo" / "sim_backend",
        CONF_DIR / "appo" / "reward",
        CONF_DIR / "appo" / "backend_task_preset",
        CONF_DIR / "appo" / "sim_backend",
        CONF_DIR / "offpolicy" / "reward",
        CONF_DIR / "offpolicy" / "backend_task_preset",
        CONF_DIR / "offpolicy" / "algo_preset",
        CONF_DIR / "offpolicy" / "sim_backend",
    ]:
        assert not path.exists(), f"legacy config group should be removed: {path}"


def test_task_files_keep_full_identity_without_hidden_backend_marker():
    for path in sorted(CONF_DIR.glob("*/task/**/*.yaml")):
        cfg = OmegaConf.load(path)
        cfg_dict_raw = OmegaConf.to_container(cfg, resolve=True) or {}
        assert isinstance(cfg_dict_raw, dict)
        assert "_selected_sim_backend" not in cfg_dict_raw, (
            f"task has hidden backend marker: {path}"
        )
        if path.stem not in _BACKENDS:
            continue
        training_raw = cfg_dict_raw.get("training", {})
        assert isinstance(training_raw, dict)
        assert "task_name" in training_raw, f"task missing task_name: {path}"
        assert "sim_backend" in training_raw, f"task missing sim_backend: {path}"


def test_motrix_task_files_do_not_declare_post_step_forward_sensor():
    for path in sorted(CONF_DIR.glob("*/task/**/*motrix*.yaml")):
        cfg = OmegaConf.load(path)

        assert OmegaConf.select(cfg, "env.post_step_forward_sensor") is None, (
            "post_step_forward_sensor is routed only to MuJoCo backends: "
            f"{path.relative_to(CONF_DIR)}"
        )


@pytest.mark.parametrize(
    "algo_dir,config_name,task,backend,task_file,overrides",
    _supported_task_cases(),
)
def test_supported_task_composes(
    algo_dir: str,
    config_name: str,
    task: str,
    backend: str,
    task_file: str,
    overrides: list[str],
):
    cfg = _compose(algo_dir, config_name, overrides=overrides)

    assert cfg.training.task_name, f"{task_file} should resolve task_name"
    assert cfg.training.sim_backend == backend, f"{task_file} should set backend"
    _assert_reward_populated(cfg, task_file)


def test_ppo_go2_arm_manip_loco_motrix_preserves_backend_overrides():
    cfg = _compose("ppo", overrides=["task=go2_arm_manip_loco/motrix"])

    assert cfg.training.task_name == "Go2ArmManipLoco"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 3000
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert cfg.env.domain_rand.randomize_dof_armature is False
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_offpolicy_g1_walk_flat_motrix_sac_preserves_backend_overrides():
    cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.2)
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_offpolicy_g1_walk_flat_mujoco_td3_uses_td3_task_owner():
    cfg = _compose("offpolicy", overrides=["algo=td3", "task=td3/g1_walk_flat/mujoco"])

    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.max_iterations == 100000
    assert cfg.algo.tau == pytest.approx(0.1)
    assert cfg.algo.actor_hidden_dim == 512
    assert cfg.algo.critic_hidden_dim == 1024
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert cfg.env.control_config.action_scale == pytest.approx(1.0)


def test_offpolicy_td3_go2_joystick_flat_motrix_composes():
    cfg = _compose(
        "offpolicy",
        overrides=["algo=td3", "task=td3/go2_joystick_flat/motrix"],
    )

    assert cfg.training.task_name == "Go2JoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.algo == "td3"
    assert cfg.algo.tau == pytest.approx(0.1)
    assert cfg.algo.algo_params.weight_decay == pytest.approx(0.1)
    assert cfg.algo.algo_params.policy_noise == pytest.approx(0.2)
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.reward.base_height_target == pytest.approx(0.3)


def test_offpolicy_td3_go1_joystick_flat_motrix_composes():
    cfg = _compose(
        "offpolicy",
        overrides=["algo=td3", "task=td3/go1_joystick_flat/motrix"],
    )

    assert cfg.training.task_name == "Go1JoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.algo == "td3"
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)


def test_offpolicy_g1_walk_flat_motrix_preserves_backend_specific_algo_value():
    mujoco_cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco"])
    motrix_cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert mujoco_cfg.algo.use_symmetry is True
    assert motrix_cfg.algo.use_symmetry is False


def test_offpolicy_g1_walk_flat_mjwarp_owner_preserves_sac_contract():
    mujoco_cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/mujoco"])
    mjwarp_cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/mjwarp"])

    assert mjwarp_cfg.training.sim_backend == "mjwarp"
    assert mjwarp_cfg.training.no_play is False
    assert mjwarp_cfg.training.play_render_mode == "record"
    assert mjwarp_cfg.algo.num_envs == mujoco_cfg.algo.num_envs
    assert mjwarp_cfg.algo.use_symmetry is mujoco_cfg.algo.use_symmetry is True
    assert mjwarp_cfg.env.control_config.action_scale == pytest.approx(
        mujoco_cfg.env.control_config.action_scale
    )
    assert mjwarp_cfg.env.mjwarp_nconmax == 128
    assert mjwarp_cfg.env.mjwarp_njmax == 256
    assert mjwarp_cfg.env.render_spacing == pytest.approx(2.0)
    assert mjwarp_cfg.env.domain_rand.randomize_kp is False
    assert mjwarp_cfg.env.domain_rand.randomize_kd is False
    assert OmegaConf.to_container(mjwarp_cfg.reward, resolve=True) == OmegaConf.to_container(
        mujoco_cfg.reward, resolve=True
    )


def test_ppo_g1_mjwarp_inherits_enabled_playback_default():
    cfg = _compose("ppo", overrides=["task=g1_walk_flat/mjwarp"])

    assert cfg.training.no_play is False
    assert cfg.training.play_render_mode == "record"


def test_ppo_g1_backend_specific_hyperparams_remain_separate():
    mujoco_cfg = _compose("ppo", overrides=["task=g1_walk_flat/mujoco"])
    motrix_cfg = _compose("ppo", overrides=["task=g1_walk_flat/motrix"])

    assert mujoco_cfg.algo.max_iterations == 2200
    assert mujoco_cfg.algo.empirical_normalization is False
    assert mujoco_cfg.algo.obs_groups.actor == ["actor"]

    assert motrix_cfg.algo.max_iterations == 2200
    assert motrix_cfg.algo.empirical_normalization is True
    assert motrix_cfg.algo.obs_groups.actor == ["policy"]
    assert OmegaConf.select(motrix_cfg, "env.motrix_max_iterations") is None
    assert motrix_cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert motrix_cfg.env.commands.vel_limit == [[0.4, 0.0, 0.0], [0.7, 0.0, 0.0]]
    assert motrix_cfg.env.gait_phase_init_mode == "offset_phase"
    assert motrix_cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert motrix_cfg.reward.scales.tracking_ang_vel == pytest.approx(0.25)
    assert motrix_cfg.reward.scales.forward_progress == pytest.approx(0.0)
    assert motrix_cfg.reward.scales.under_speed == pytest.approx(-0.2)
    assert motrix_cfg.reward.scales.penalty_feet_ori == pytest.approx(0.0)
    assert motrix_cfg.reward.scales.feet_phase == pytest.approx(1.2)
    assert motrix_cfg.reward.scales.feet_phase_contrast == pytest.approx(1.5)
    assert motrix_cfg.reward.scales.feet_phase_contact == pytest.approx(1.0)
    assert motrix_cfg.reward.scales.feet_double_stance == pytest.approx(-1.0)
    assert motrix_cfg.reward.scales.base_height == pytest.approx(-120.0)
    assert motrix_cfg.reward.scales.pose == pytest.approx(-0.05)
    assert motrix_cfg.reward.base_height_target == pytest.approx(0.765)
    assert motrix_cfg.reward.min_forward_speed_for_gait_reward == pytest.approx(0.05)
    assert motrix_cfg.reward.min_base_height == pytest.approx(0.5)
    assert motrix_cfg.reward.max_tilt_deg == pytest.approx(35.0)


@pytest.mark.parametrize(
    ("algo_dir", "overrides"),
    [
        ("ppo", ["task=g1_walk_flat/mujoco"]),
        ("ppo_him", ["task=go2_arm_manip_loco/mujoco"]),
        ("appo", ["task=g1_walk_flat/mujoco"]),
        ("offpolicy", ["algo=sac", "task=sac/g1_walk_flat/mujoco"]),
        ("offpolicy", ["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"]),
    ],
)
def test_post_step_forward_sensor_defaults_false_outside_sharpa_mujoco(
    algo_dir: str, overrides: list[str]
):
    cfg = _compose(algo_dir, overrides=overrides)

    assert cfg.env.post_step_forward_sensor is False


@pytest.mark.parametrize(
    ("algo_dir", "overrides"),
    [
        ("ppo", ["task=sharpa_inhand/mujoco"]),
        ("ppo", ["task=sharpa_inhand/mujoco_hora"]),
        ("ppo", ["task=sharpa_inhand_grasp/mujoco"]),
        ("appo", ["task=sharpa_inhand/mujoco"]),
        ("appo", ["task=sharpa_inhand/mujoco_hora"]),
        ("offpolicy", ["algo=sac", "task=sac/sharpa_inhand/mujoco_hora"]),
        ("hora_distill", ["task=sharpa_inhand/mujoco"]),
    ],
)
def test_post_step_forward_sensor_enabled_for_sharpa_mujoco(algo_dir: str, overrides: list[str]):
    cfg = _compose(algo_dir, overrides=overrides)

    assert cfg.env.post_step_forward_sensor is True


def test_mujoco_post_step_forward_sensor_can_be_overridden():
    override_cfg = _compose(
        "appo",
        overrides=["task=sharpa_inhand/mujoco_hora", "env.post_step_forward_sensor=false"],
    )

    assert override_cfg.env.post_step_forward_sensor is False


def test_appo_adaptive_lr_factors_are_overridden_only_by_dex_hand_owners():
    g1_cfg = _compose("appo", overrides=["task=g1_walk_flat/mujoco"])
    allegro_cfg = _compose("appo", overrides=["task=allegro_inhand/mujoco"])
    allegro_motrix_cfg = _compose("appo", overrides=["task=allegro_inhand/motrix"])
    sharpa_cfg = _compose("appo", overrides=["task=sharpa_inhand/mujoco"])
    sharpa_hora_cfg = _compose("appo", overrides=["task=sharpa_inhand/mujoco_hora"])

    assert g1_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert g1_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)
    assert allegro_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(2.0)
    assert allegro_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.5)
    assert allegro_motrix_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(2.0)
    assert allegro_motrix_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.5)
    assert sharpa_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert sharpa_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)
    assert sharpa_hora_cfg.algo.algorithm.adaptive_kl_factor == pytest.approx(1.2)
    assert sharpa_hora_cfg.algo.algorithm.adaptive_lr_factor == pytest.approx(1.1)


def test_ppo_go1_motrix_preserves_reward_and_algo_values():
    cfg = _compose("ppo", overrides=["task=go1_joystick_flat/motrix"])

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.env.commands.vel_limit == [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


def test_ppo_go2_motrix_preserves_backend_env_overrides():
    cfg = _compose("ppo", overrides=["task=go2_joystick_flat/motrix"])

    assert cfg.algo.num_envs == 1024
    assert cfg.algo.empirical_normalization is True
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_ppo_go2w_mujoco_uses_motor_owner_dr_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/mujoco"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.commands.vel_limit == [[0.0, 0.0, -1.0], [1.0, 0.0, 1.0]]
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False
    assert cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert cfg.env.control_config.Kp == pytest.approx(50.0)
    assert cfg.env.control_config.Kd == pytest.approx(1.5)
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(10.0)
    assert cfg.env.control_config.wheel_Kd == pytest.approx(0.5)
    assert cfg.reward.scales.tracking_ang_vel == pytest.approx(0.75)
    assert cfg.reward.scales.orientation == pytest.approx(-2.0)
    assert cfg.reward.scales.upward == pytest.approx(1.0)
    assert cfg.reward.base_height_target == pytest.approx(0.4)
    assert cfg.reward.scales.torques < 0.0


def test_ppo_go2w_motrix_uses_motor_owner_dr_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/motrix"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.env.render_offset_mode == "zero"
    assert cfg.env.commands.vel_limit == [[0.0, 0.0, -1.0], [1.0, 0.0, 1.0]]
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False
    assert cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert cfg.env.control_config.Kp == pytest.approx(50.0)
    assert cfg.env.control_config.Kd == pytest.approx(1.5)
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(10.0)
    assert cfg.env.control_config.wheel_Kd == pytest.approx(0.5)
    assert cfg.reward.scales.tracking_ang_vel == pytest.approx(0.75)
    assert cfg.reward.scales.orientation == pytest.approx(-2.0)
    assert cfg.reward.scales.upward == pytest.approx(1.0)
    assert cfg.reward.scales.torques < 0.0


def test_ppo_go2w_motrix_uses_motor_owner_scene_path():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_flat/motrix"])

    assert cfg.training.task_name == "Go2WJoystickFlat"
    assert cfg.training.sim_backend == "motrix"
    assert "model_file" not in cfg.env
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(10.0)
    assert cfg.reward.scales.torques < 0.0


def test_ppo_go2w_rough_mujoco_uses_terrain_generator():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_rough/mujoco"])

    assert cfg.training.task_name == "Go2WJoystickRough"
    assert cfg.training.sim_backend == "mujoco"
    assert str(cfg.env.scene.model_file).endswith("src/unilab/assets/robots/go2w/go2w_mujoco.xml")
    assert cfg.env.scene.terrain.hfield_name == "terrain_hfield"
    assert cfg.env.scene.terrain.geom_name == "floor"
    assert cfg.env.terrain_scan.hfield_name == "terrain_hfield"
    assert cfg.env.terrain_scan.geom_name == "floor"
    assert cfg.env.commands.resampling_time == pytest.approx(10.0)
    assert cfg.env.commands.heading_command is True
    assert cfg.env.commands.vel_limit == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    assert cfg.env.commands.heading_range == pytest.approx([-3.141592653589793, 3.141592653589793])
    assert cfg.env.control_config.clip_actions == pytest.approx(100.0)
    assert cfg.env.control_config.action_scale == pytest.approx(0.25)
    assert cfg.env.control_config.hip_action_scale == pytest.approx(0.125)
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(5.0)
    assert cfg.env.domain_rand.randomize_kp is True
    assert cfg.env.domain_rand.randomize_kd is True
    assert cfg.env.domain_rand.kp_multiplier_range == [0.5, 1.0]
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(3.0)
    assert cfg.reward.scales.hip_pos == pytest.approx(-2.0)
    assert cfg.reward.scales.joint_mirror == pytest.approx(-0.05)
    assert cfg.reward.only_positive_rewards is False
    assert cfg.algo.max_iterations == 1200


def test_ppo_go2w_rough_motrix_uses_yaw_reset_and_strong_control():
    cfg = _compose("ppo", overrides=["task=go2w_joystick_rough/motrix"])

    assert cfg.training.task_name == "Go2WJoystickRough"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.env.commands.vel_limit == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    assert cfg.env.commands.heading_range == pytest.approx([-3.141592653589793, 3.141592653589793])
    assert cfg.env.control_config.action_scale == pytest.approx(0.25)
    assert cfg.env.control_config.hip_action_scale == pytest.approx(0.125)
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(5.0)
    assert cfg.env.domain_rand.randomize_kp is True
    assert cfg.env.domain_rand.randomize_kd is True
    assert cfg.reward.scales.orientation == pytest.approx(-2.0)
    assert cfg.reward.scales.hip_pos == pytest.approx(-0.5)
    assert cfg.reward.scales.upward == pytest.approx(1.0)
    assert cfg.algo.max_iterations == 1200


def test_offpolicy_g1_walk_flat_motrix_preserves_backend_env_overrides():
    cfg = _compose("offpolicy", overrides=["algo=sac", "task=sac/g1_walk_flat/motrix"])

    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_offpolicy_flashsac_go2_joystick_mujoco_enables_full_dr_stack():
    mujoco_cfg = _compose(
        "offpolicy",
        overrides=["algo=flashsac", "task=flashsac/go2_joystick_flat/mujoco"],
    )

    assert mujoco_cfg.training.task_name == "Go2JoystickFlat"
    assert mujoco_cfg.training.sim_backend == "mujoco"

    assert mujoco_cfg.env.domain_rand.randomize_kp is True
    assert mujoco_cfg.env.domain_rand.randomize_kd is True
    assert mujoco_cfg.env.domain_rand.randomize_base_mass is True
    assert mujoco_cfg.env.domain_rand.random_com is True
    assert mujoco_cfg.env.domain_rand.randomize_gravity is True
    assert mujoco_cfg.env.domain_rand.push_robots is True
    assert mujoco_cfg.env.noise_config.level == pytest.approx(1.0)


def test_cli_override_beats_task_defaults():
    cfg = _compose(
        "ppo",
        overrides=["task=g1_walk_flat/motrix", "algo.max_iterations=1"],
    )

    assert cfg.algo.max_iterations == 1
    assert cfg.algo.empirical_normalization is True
