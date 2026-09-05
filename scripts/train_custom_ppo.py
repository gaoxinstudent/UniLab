"""Train migrated WheelBipe HIM, DreamWaQ or NP3O runtimes.

Unlike the generic PPO entrypoint this script deliberately constructs the
selected algorithm and its representation/constraint losses through
``CustomOnPolicyRunner``.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.algos.torch.custom_ppo.runner import CustomOnPolicyRunner
from unilab.base.backend import RenderClosedError, materialize_scene_visual_override
from unilab.training import (
    BackendAdapter,
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    get_latest_checkpoint,
    get_latest_run,
    get_log_root,
    parse_checkpoint_path,
    resolve_latest_checkpoint_within_runs,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.rsl_rl import RslRlVecEnvWrapper
from unilab.training.sim2sim import (
    hydrate_custom_checkpoint_config,
    policy_load_dim_guard,
    resolve_sim2sim_config,
)
from unilab.training.wheelbipe import wheelbipe_play_checkpoint_task_candidates
from unilab.utils.device import get_default_device, resolve_torch_device_alias


def _resolve_custom_device(cfg: DictConfig) -> str:
    """Resolve the custom PPO torch device, honoring an explicit override."""

    configured = OmegaConf.select(cfg, "training.device", default=None)
    if configured in (None, ""):
        return get_default_device()
    return resolve_torch_device_alias(str(configured))


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("cfg.algo must resolve to a mapping")
    return cast(dict[str, Any], value)


def _format_custom_checkpoint_error(
    cfg: DictConfig,
    *,
    load_path: Path | None,
    load_path_dir: Path | None,
    purpose: str = "play",
) -> str:
    """Explain why custom play/resume could not resolve a model checkpoint."""

    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    selected_checkpoint = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    checkpoint_hint = (
        f" algo.checkpoint={selected_checkpoint!r}"
        if selected_checkpoint not in (None, "", -1, "-1")
        else ""
    )
    if load_path_dir is not None and load_path is None and checkpoint_hint:
        reason = f"Requested checkpoint was not found under resolved_run={load_path_dir}."
    elif not task_log_root.exists():
        reason = "Task log root does not exist."
    else:
        latest_run = get_latest_run(task_log_root)
        if latest_run is None:
            reason = "No run directories were found under the task log root."
        elif get_latest_checkpoint(latest_run) is None:
            reason = f"Resolved latest run has no model_*.pt checkpoint files: {latest_run}."
        else:
            reason = "Requested run or checkpoint could not be resolved."
    return (
        f"Could not resolve a custom PPO checkpoint for {purpose} mode. "
        f"{reason} task={cfg.training.task_name} task_log_root={task_log_root} "
        f"algo.load_run={OmegaConf.select(cfg, 'algo.load_run', default='-1')!r}"
        f"{checkpoint_hint}. Set algo.load_run=<run-dir-or-checkpoint-path> and "
        "optionally algo.checkpoint=<iteration-or-filename>."
    )


def _resolve_custom_resume_path(cfg: DictConfig) -> Path | None:
    """Resolve an explicit custom-training checkpoint, or return ``None``.

    ``-1`` is the established UniLab sentinel for a fresh training run.  An
    explicit run id, run directory, or checkpoint path opts into continuation
    and is resolved through the same helper used by custom playback.
    """

    load_run = str(OmegaConf.select(cfg, "algo.load_run", default="-1"))
    if load_run in {"", "-1", "None"}:
        return None
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        raise FileNotFoundError(
            _format_custom_checkpoint_error(
                cfg,
                load_path=load_path,
                load_path_dir=load_path_dir,
                purpose="resume",
            )
        )
    return load_path


def _resolve_custom_play_path(cfg: DictConfig) -> tuple[Path | None, Path | None]:
    """Resolve custom playback, with latest-only Play-owner fallback.

    Exact custom Play IDs describe a different environment owner but train
    under the paired non-Play task root.  Only the established ``-1`` latest
    sentinel searches that paired root; an explicit run name or path remains
    scoped to the selected Play task and therefore fails closed when absent.
    """

    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is not None:
        return load_path, load_path_dir
    if str(OmegaConf.select(cfg, "algo.load_run", default="-1")) != "-1":
        return load_path, load_path_dir
    for fallback_task in wheelbipe_play_checkpoint_task_candidates(str(cfg.training.task_name)):
        fallback_root = get_log_root(ROOT_DIR, cfg) / fallback_task
        fallback_path, fallback_dir = resolve_latest_checkpoint_within_runs(
            fallback_root,
            checkpoint=OmegaConf.select(cfg, "algo.checkpoint", default=-1),
        )
        if fallback_path is not None and fallback_dir is not None:
            print(
                "Using latest non-Play checkpoint for exact Wheelbipe custom Play owner: "
                f"task={fallback_task}"
            )
            return fallback_path, fallback_dir
    return load_path, load_path_dir


def _build_custom_env_override(cfg: DictConfig) -> dict[str, Any]:
    adapter = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name=str(cfg.algo.algorithm_name),
        scene_materializer=materialize_scene_visual_override,
    )
    if bool(cfg.training.play_only):
        return cast(dict[str, Any], adapter.build_play_env_cfg_override())
    return cast(dict[str, Any], adapter.build_task_env_cfg_override())


def _custom_checkpoint_explicit_paths() -> set[str]:
    """Return architecture paths explicitly selected by the current Hydra job.

    A source run's sidecar is authoritative only when the target did not ask
    for a different architecture.  Hydra keeps the original override strings
    in ``HydraConfig``; direct unit-test callers simply get an empty set and
    therefore exercise the metadata hydration path.
    """

    if not HydraConfig.initialized():
        return set()
    try:
        overrides = list(HydraConfig.get().overrides.task)
        choices = HydraConfig.get().runtime.choices
    except Exception:
        return set()

    explicit: set[str] = set()
    for raw_override in overrides:
        key = str(raw_override).split("=", 1)[0].strip()
        key = key.lstrip("+~")
        if key.endswith("+"):
            key = key[:-1]
        if key:
            explicit.add(key)

    # Selecting a named profile is an explicit architecture choice even
    # though Hydra records only ``profile=<name>`` rather than the expanded
    # fields.  Keep source-profile/compact mismatches fail-closed instead of
    # silently replacing a user's selected profile with checkpoint values.
    try:
        profile_choice = choices.get("profile")
    except AttributeError:
        profile_choice = None
    if profile_choice not in (None, "", "null", "None"):
        explicit.update(
            {
                "algo.policy",
                "algo.policy_architecture",
                "algo.latent_dim",
                "algo.estimator",
            }
        )
    return explicit


def _hydrate_custom_checkpoint_config(cfg: DictConfig, run_dir: Path | None) -> DictConfig:
    """Adopt source policy constructor metadata before custom play/resume.

    Exact upstream routes now compose their source profile automatically.
    Canonical/legacy playback may still omit a profile, so a checkpoint
    sidecar supplies architecture fields when compact defaults differ, while
    explicit user/profile overrides remain untouched and are checked by the
    normal contract resolver.  History reset remains a separately validated
    observation-protocol contract and is never guessed from architecture.
    """

    enabled = bool(OmegaConf.select(cfg, "training.auto_load_checkpoint_config", default=True))
    return hydrate_custom_checkpoint_config(
        run_dir,
        cfg,
        enabled=enabled,
        explicit_paths=_custom_checkpoint_explicit_paths(),
    )


def _play_custom(cfg: DictConfig, device: str) -> str | None:
    """Load a custom checkpoint and run contract-checked playback.

    The custom runner owns history stacking and validates the
    algorithm/variant/cost metadata before loading the state dict.  A
    ``play_render_mode=none`` run still executes a numerical rollout, which
    keeps eval useful on headless CI and gives it a concrete success signal.
    """

    load_path, load_path_dir = _resolve_custom_play_path(cfg)
    if load_path is None or load_path_dir is None or not load_path.exists():
        raise FileNotFoundError(
            _format_custom_checkpoint_error(
                cfg,
                load_path=load_path,
                load_path_dir=load_path_dir,
            )
        )
    print(f"Loading custom PPO checkpoint: {load_path}")

    cfg = _hydrate_custom_checkpoint_config(cfg, load_path_dir)
    cfg = (
        resolve_sim2sim_config(
            load_path_dir,
            cfg,
            algo_name="custom_ppo",
            strict=bool(OmegaConf.select(cfg, "training.sim2sim_strict", default=True)),
        )
        or cfg
    )
    algo_cfg = _algo_config_dict(cfg)
    env = create_env(
        cfg,
        num_envs=int(OmegaConf.select(cfg, "training.play_env_num", default=1)),
        env_cfg_override=_build_custom_env_override(cfg),
    )
    wrapped = RslRlVecEnvWrapper(env, device=device)
    try:
        runner = CustomOnPolicyRunner(wrapped, algo_cfg, log_dir=None, device=device)
        with policy_load_dim_guard(
            env_obs_dim=getattr(wrapped, "num_obs", None),
            env_action_dim=getattr(wrapped, "num_actions", None),
            algo_name="custom_ppo",
        ):
            # Playback is an inference lifecycle, not a training resume.  In
            # particular DreamWaQ persists AdaBoot's per-environment partial
            # returns and representation/optimizer state for exact training
            # continuation.  Restoring that owner state into a play env with
            # a different ``training.play_env_num`` would either leak stale
            # episode statistics or fail on a shape mismatch (for example a
            # 4096-env training checkpoint loaded into the default 1-env
            # evaluator).  Policy weights are the only state playback needs;
            # keep iteration/optimizer restoration explicit for resume mode
            # in ``main`` below.
            runner.load(str(load_path), load_optimizer=False, load_iteration=False)
        infer = runner.get_inference_policy(device=device)
        play_steps = int(OmegaConf.select(cfg, "training.play_steps", default=200))
        if play_steps < 1:
            raise ValueError(f"training.play_steps must be positive, got {play_steps}")
        render_mode = str(OmegaConf.select(cfg, "training.play_render_mode", default="none"))

        reward_sum = 0.0
        done_count = 0
        source_architecture = runner.policy_architecture == "source_barlow"
        if render_mode.strip().lower() == "none":
            obs_td, _ = wrapped.reset()
            obs = (
                runner.build_source_observation(obs_td, reset=True)
                if source_architecture
                else obs_td["actor"]
            )
            pending_dones_numerical: torch.Tensor | None = None
            with torch.inference_mode():
                for _ in range(play_steps):
                    actions = infer(obs, pending_dones_numerical)
                    next_td, rewards, dones, _ = wrapped.step(actions)
                    reward_sum += float(rewards.mean().item())
                    done_count += int(torch.count_nonzero(dones).item())
                    obs = (
                        runner.build_source_observation(next_td, dones=dones)
                        if source_architecture
                        else next_td["actor"]
                    )
                    pending_dones_numerical = dones
            print(
                "Custom PPO playback complete: "
                f"steps={play_steps} mean_reward={reward_sum / play_steps:.6f} "
                f"done_count={done_count}"
            )
            return None

        output_video = Path(load_path_dir) / "play_video.mp4"
        pending_dones: torch.Tensor | None = None

        def initialize() -> Any:
            nonlocal pending_dones
            pending_dones = None
            obs_td = wrapped.reset()[0]
            if source_architecture:
                return runner.build_source_observation(obs_td, reset=True)
            return obs_td

        def step(obs_td: Any) -> Any:
            nonlocal pending_dones
            obs = obs_td if source_architecture else obs_td["actor"]
            with torch.inference_mode():
                actions = infer(obs, pending_dones)
                next_td, _rewards, dones, _ = wrapped.step(actions)
            pending_dones = dones
            if source_architecture:
                return runner.build_source_observation(next_td, dones=dones)
            return next_td

        playback_mode: str | None = None

        def on_plan(plan: Any) -> None:
            nonlocal playback_mode
            playback_mode = str(plan.mode)
            if plan.mode == "record":
                print(f"Rendering custom playback video to {output_video}...")
            elif plan.mode == "interactive":
                print("Starting custom interactive playback...")

        try:
            video_path = cast(
                str | None,
                env.run_playback_mode(
                    play_render_mode=render_mode,
                    play_steps=play_steps,
                    output_video=output_video,
                    initialize=initialize,
                    step=step,
                    render_spacing=float(
                        OmegaConf.select(cfg, "training.render_spacing", default=1.0)
                    ),
                    render_offset_mode=str(getattr(env.cfg, "render_offset_mode", "grid")),
                    camera_kwargs={
                        "cam_distance": float(
                            OmegaConf.select(cfg, "training.cam_distance", default=6.0)
                        ),
                        "cam_elevation": float(
                            OmegaConf.select(cfg, "training.cam_elevation", default=-20.0)
                        ),
                        "cam_azimuth": float(
                            OmegaConf.select(cfg, "training.cam_azimuth", default=90.0)
                        ),
                        "cam_lookat": OmegaConf.select(cfg, "training.cam_lookat", default=None),
                        "cam_tracking": bool(
                            OmegaConf.select(cfg, "training.cam_tracking", default=False)
                        ),
                        "cam_tracking_env_idx": int(
                            OmegaConf.select(cfg, "training.cam_tracking_env_idx", default=0)
                        ),
                        "cam_tracking_extra_envs": int(
                            OmegaConf.select(cfg, "training.cam_tracking_extra_envs", default=2)
                        ),
                    },
                    on_plan=on_plan,
                ),
            )
        except RenderClosedError:
            print("Render window closed.")
            video_path = None
        if playback_mode != "none":
            print(
                "Custom PPO playback complete: "
                f"steps={play_steps}" + (f" video={video_path}" if video_path is not None else "")
            )
        return video_path
    finally:
        env.close()


@hydra.main(version_base="1.3", config_path="../conf/custom_ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    # Apply the same algorithm-level seed contract as the standard PPO/APPO
    # entrypoints before constructing an environment or a policy.  This keeps
    # command sampling, domain randomization, and network initialization
    # reproducible for both training and play-only custom runs.
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = _resolve_custom_device(cfg)
    print(f"Using device: {device}")
    if bool(cfg.training.play_only):
        _play_custom(cfg, device)
        return
    # Resolve a continuation checkpoint before materializing the environment
    # or runner.  Custom source profiles carry constructor metadata in the
    # run sidecar; adopting it here keeps resume construction aligned with
    # playback while preserving explicit Hydra architecture overrides.
    resume_path = _resolve_custom_resume_path(cfg)
    if resume_path is not None:
        cfg = _hydrate_custom_checkpoint_config(cfg, resume_path.parent)
    env_override = cast(
        dict[str, Any],
        BackendAdapter(
            cfg,
            root_dir=ROOT_DIR,
            algo_name=str(cfg.algo.algorithm_name),
            scene_materializer=materialize_scene_visual_override,
        ).build_task_env_cfg_override(),
    )
    algo_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    max_iterations = int(algo_cfg["max_iterations"])
    if cfg.training.num_timesteps:
        max_iterations = max(
            1,
            int(
                cfg.training.num_timesteps / (algo_cfg["num_envs"] * algo_cfg["num_steps_per_env"])
            ),
        )
    # Use the shared resolver for both absolute and repository-relative
    # ``training.log_root`` values so a later ``eval --load-run`` resolves the
    # exact directory produced by this training entrypoint.
    log_root = get_log_root(ROOT_DIR, cfg)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = (
        log_root
        / str(cfg.training.task_name)
        / f"{timestamp}_{cfg.training.sim_backend}_{algo_cfg['algorithm_name']}"
    )
    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=str(log_dir),
        algo_name=str(algo_cfg["algorithm_name"]),
        task_name=str(cfg.training.task_name),
        sim_backend=str(cfg.training.sim_backend),
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=device,
        seed_info=seed_info,
    )
    tracker.start()
    env = None
    try:
        env = create_env(cfg, num_envs=int(algo_cfg["num_envs"]), env_cfg_override=env_override)
        wrapped = RslRlVecEnvWrapper(env, device=device)
        runner = CustomOnPolicyRunner(wrapped, algo_cfg, log_dir=str(log_dir), device=device)
        if resume_path is not None:
            print(f"Resuming custom PPO from {resume_path}")
            with policy_load_dim_guard(
                env_obs_dim=getattr(wrapped, "num_obs", None),
                env_action_dim=getattr(wrapped, "num_actions", None),
                algo_name="custom_ppo",
            ):
                runner.load(str(resume_path))
        runner.learn(max_iterations, init_at_random_ep_len=True)
        tracker.update_summary(
            {
                "status": "completed",
                "algorithm": runner.algorithm_name,
                "completed_iterations": runner.current_learning_iteration,
                "total_env_steps": runner.tot_timesteps,
                "final_mean_reward": float(sum(runner.rewbuffer) / len(runner.rewbuffer))
                if runner.rewbuffer
                else None,
            }
        )
        # Keep the existing one-input export for every custom owner.  The
        # explicit source Barlow profile additionally emits the upstream
        # deploy artifact (TorchScript + ONNX, ``obs``/``obs_hist`` inputs),
        # so selecting ``policy_architecture=source_barlow`` cannot leave the
        # source graph disconnected from the training/export lifecycle.
        runner.export_policy_to_onnx(str(log_dir))
        if runner.policy_architecture == "source_barlow":
            jit_path, source_onnx_path = runner.export_source_barlow_actor(str(log_dir))
            print(f"Exported source Barlow actor: torchscript={jit_path} onnx={source_onnx_path}")
    finally:
        if env is not None:
            env.close()
        tracker.finish()


if __name__ == "__main__":
    main()
