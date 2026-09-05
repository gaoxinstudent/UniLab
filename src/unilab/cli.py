"""Thin package CLI for routing to existing UniLab training entrypoints."""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

from unilab.demo import run_demo

SUPPORTED_ALGOS = ("ppo", "him_ppo", "dreamwaq", "np3o", "appo", "sac", "td3", "flashsac")
SUPPORTED_SIMS = ("mujoco", "mjwarp", "motrix")
SUPPORTED_RENDER_MODES = ("auto", "interactive", "record", "none")
OFFPOLICY_ALGOS = {"sac", "td3", "flashsac"}
CUSTOM_ONPOLICY_ALGOS = {"him_ppo", "dreamwaq", "np3o"}
# Public custom algorithms intentionally route to named owner directories.
# Keep this mapping in the CLI contract so completion and support tooling do
# not grow a second, potentially divergent variant table.
CUSTOM_ALGO_TASK_VARIANTS = {
    "him_ppo": "wheelbipe_v14_flat_him",
    "dreamwaq": "wheelbipe_v14_flat_dreamwaq",
    "np3o": "wheelbipe_v14_flat_np3o",
}
# Custom WheelBipe profiles are Hydra config groups, rather than owner-name
# suffixes.  Keep the names here so exact upstream IDs automatically compose
# their pinned long-run source settings and the canonical task can opt into
# the same settings through ``--profile`` without changing the backend owner.
# The mapping is intentionally metadata-only; Hydra remains the source of
# truth for the profile payload itself.
CUSTOM_ALGO_PROFILES = {
    "him_ppo": "source_him_long",
    "dreamwaq": "source_dreamwaq_long",
    "np3o": "source_np3o_barlow_long",
}
CUSTOM_ONPOLICY_TASK = "wheelbipe_v14_flat"


@dataclass(frozen=True)
class UpstreamWheelbipeCliRoute:
    """CLI migration metadata for one published upstream Gymnasium id.

    The Isaac project publishes task ids that encode both the algorithm and,
    for ``*-Play-*`` ids, the play-only lifecycle.  UniLab owner YAMLs use a
    stable snake-case task plus an explicit ``--algo`` flag instead.  Keeping
    this table next to the route builder makes that translation auditable and
    keeps source capability validation ahead of Hydra path resolution.  Exact
    IDs select their named bounded owner; genuinely unavailable capabilities
    still fail closed with a diagnostic.

    ``supported`` means that UniLab has an executable compatibility owner; it
    does *not* claim source Isaac asset/dynamics parity.  The owner metadata
    below makes the bounded gimbal/state-machine implementations explicit and
    keeps the exact source id from silently selecting the canonical v0 class.
    """

    canonical_task: str
    algorithm: str
    play_only: bool = False
    supported: bool = True
    unsupported_reason: str | None = None
    # Registry config class to select for an exact source id.  Canonical owner
    # YAMLs intentionally remain shared; this generated override selects the
    # named variant after Hydra composes the backend owner.
    owner_registry_name: str | None = None
    generated_overrides: tuple[str, ...] = ()


# Exact ids from ``wheeled-legged_RL``.  Keep every published id here even
# when the owner uses a bounded backend-neutral implementation; a known id
# must select that named owner rather than silently falling back to v0.
UPSTREAM_WHEELBIPE_CLI_ROUTES: dict[str, UpstreamWheelbipeCliRoute] = {
    "Robotics-Wheelbipe-V14-Flat-v0": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_flat",
        algorithm="ppo",
        owner_registry_name="WheelbipeV14FlatV0",
        generated_overrides=(
            "training.task_name=WheelbipeV14FlatV0",
            "env.vel_height_gate_enabled=false",
        ),
    ),
    "Robotics-Wheelbipe-V14-Flat-v1": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_flat",
        algorithm="ppo",
        owner_registry_name="WheelbipeV14FlatV1",
        generated_overrides=(
            "training.task_name=WheelbipeV14FlatV1",
            "env.vel_height_gate_enabled=false",
            "reward.scales.leg_joint_acc=-1.0e-7",
            "reward.scales.leg_joint_vel=-1.0e-3",
            "reward.scales.wheel_acc=-2.0e-9",
            "reward.scales.wheel_vel=-2.0e-6",
            "reward.scales.action_smoothness_leg=-5.0e-3",
            "reward.scales.action_rate=-2.0e-3",
            "reward.scales.action_smoothness_wheel=-1.0e-3",
            "+env.state_machine.enabled=true",
            "env.termination_duration_steps=10",
            "env.commands.special_mode_start_iterations=[0,0,0]",
            "env.commands.special_mode_probabilities=[0.15,0.15,0.20]",
            "env.commands.gimbal_mode_probability=0.0",
            "env.commands.zero_command_probability=0.10",
            "env.commands.zero_command_start_iteration=0",
        ),
    ),
    "Robotics-Wheelbipe-V14-Flat-v2": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_flat",
        algorithm="ppo",
        owner_registry_name="WheelbipeV14FlatV2",
        generated_overrides=(
            "training.task_name=WheelbipeV14FlatV2",
            "env.vel_height_gate_enabled=false",
            "env.gimbal.enabled=true",
            "env.gimbal.control_mode=heading_pd",
            "env.ctrl_mode_obs_scale=[1.0,1.0,1.0,1.0,1.0,1.0,1.0]",
            "env.commands.heading_command=true",
            "env.commands.special_mode_start_iterations=[0,0,0]",
            "env.commands.special_mode_probabilities=[0.10,0.10,0.20]",
            "env.commands.gimbal_mode_probability=0.20",
            "env.commands.gimbal_mode_start_iteration=0",
        ),
    ),
    "Robotics-Wheelbipe-V14-Flat-Play-v0": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_flat",
        algorithm="ppo",
        play_only=True,
        owner_registry_name="WheelbipeV14FlatPlayV0",
        generated_overrides=(
            "training.task_name=WheelbipeV14FlatPlayV0",
            "env.vel_height_gate_enabled=false",
        ),
    ),
    "Robotics-Wheelbipe-V14-Flat-Play-v2": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_flat",
        algorithm="ppo",
        play_only=True,
        owner_registry_name="WheelbipeV14FlatPlayV2",
        generated_overrides=(
            "training.task_name=WheelbipeV14FlatPlayV2",
            "env.vel_height_gate_enabled=false",
            "env.gimbal.enabled=true",
            "env.gimbal.control_mode=heading_pd",
            "+env.gimbal.heading_target_mode=fixed",
            "+env.gimbal.fixed_heading=0.0",
            "env.ctrl_mode_obs_scale=[1.0,1.0,1.0,1.0,1.0,1.0,1.0]",
            "env.commands.heading_command=true",
            "env.commands.special_mode_start_iterations=[0,0,0]",
            "env.commands.special_mode_probabilities=[0.0,0.0,0.0]",
            "env.commands.gimbal_mode_probability=1.0",
            "env.commands.gimbal_mode_start_iteration=0",
            "env.height_range=[0.20,0.30]",
        ),
    ),
    "Robotics-Wheelbipe-V14-Rough-v0": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_rough",
        algorithm="ppo",
        owner_registry_name="WheelbipeV14RoughV0",
        generated_overrides=(
            "training.task_name=WheelbipeV14RoughV0",
            "env.vel_height_gate_enabled=false",
            "env.gimbal.enabled=true",
            "env.gimbal_spin_translate.enabled=false",
            "env.commands.heading_command=true",
        ),
    ),
    "Robotics-Wheelbipe-V14-Rough-v1": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_rough",
        algorithm="ppo",
        owner_registry_name="WheelbipeV14RoughV1",
        generated_overrides=(
            "training.task_name=WheelbipeV14RoughV1",
            "env.vel_height_gate_enabled=false",
            "reward.scales.leg_joint_acc=-1.0e-7",
            "reward.scales.leg_joint_vel=-1.0e-3",
            "reward.scales.wheel_acc=-2.0e-9",
            "reward.scales.wheel_vel=-2.0e-6",
            "reward.scales.action_smoothness_leg=-5.0e-3",
            "reward.scales.action_rate=-2.0e-3",
            "reward.scales.action_smoothness_wheel=-1.0e-3",
            "reward.scales.track_lin_vel_xy=1.25",
            "+env.state_machine.enabled=true",
            # Rough-v1 derives from Flat-v1 in the pinned source, not from
            # the canonical Rough-v0/v2 gimbal-spin owner composed above.
            # Undo those canonical-only command fields explicitly so an
            # exact source id cannot inherit the wrong mutually-exclusive
            # mode bucket before the named registry config is constructed.
            "env.gimbal_spin_translate.enabled=false",
            "env.gimbal.control_mode=velocity",
            "env.gimbal.randomize_heading=false",
            "env.ctrl_mode_obs_scale=[1.0,1.0,1.0,1.0,1.0,5.0,1.0]",
            "env.termination_duration_steps=10",
            "env.commands.special_mode_start_iterations=[0,0,0]",
            "env.commands.special_mode_probabilities=[0.15,0.15,0.20]",
            "env.commands.gimbal_mode_probability=0.0",
            "env.commands.zero_command_probability=0.10",
            "env.commands.zero_command_start_iteration=0",
        ),
    ),
    "Robotics-Wheelbipe-V14-Rough-Play-v0": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_rough",
        algorithm="ppo",
        play_only=True,
        owner_registry_name="WheelbipeV14RoughPlayV0",
        generated_overrides=(
            "training.task_name=WheelbipeV14RoughPlayV0",
            "env.vel_height_gate_enabled=false",
            "env.gimbal.enabled=true",
            "env.gimbal.control_mode=heading_pd",
            "env.gimbal_spin_translate.enabled=false",
            "env.ctrl_mode_obs_scale=[1.0,1.0,1.0,1.0,1.0,1.0,1.0]",
            "env.commands.heading_command=true",
            "env.commands.vel_limit=[[2.2,0.0,-3.141592653589793],[2.2,0.0,3.141592653589793]]",
            "env.commands.rel_standing_envs=0.0",
            "env.commands.rel_heading_envs=0.5",
            "env.commands.heading_control_stiffness=1.0",
            "env.commands.source_curriculum_enabled=false",
            "env.commands.special_mode_start_iterations=[0,0,0]",
            "env.commands.special_mode_probabilities=[0.0,0.0,0.0]",
            "env.commands.gimbal_mode_probability=0.0",
            "env.commands.zero_command_probability=0.0",
            "env.domain_rand.randomize_kp=false",
            "env.domain_rand.randomize_kd=false",
            "env.domain_rand.predefined_ground_probability=0.0",
            "env.rough_terrain_boundary_reset.use_inner_terrain_area=true",
            "+env.max_episode_seconds=5.0",
        ),
    ),
    "Robotics-Wheelbipe-V14-Rough-Play-v1": UpstreamWheelbipeCliRoute(
        canonical_task="wheelbipe_v14_rough",
        algorithm="ppo",
        play_only=True,
        owner_registry_name="WheelbipeV14RoughPlayV1",
        generated_overrides=(
            "training.task_name=WheelbipeV14RoughPlayV1",
            "env.vel_height_gate_enabled=false",
            "reward.scales.leg_joint_acc=-1.0e-7",
            "reward.scales.leg_joint_vel=-1.0e-3",
            "reward.scales.wheel_acc=-2.0e-9",
            "reward.scales.wheel_vel=-2.0e-6",
            "reward.scales.action_smoothness_leg=-5.0e-3",
            "reward.scales.action_rate=-2.0e-3",
            "reward.scales.action_smoothness_wheel=-1.0e-3",
            "reward.scales.track_lin_vel_xy=1.25",
            "+env.state_machine.enabled=true",
            "env.gimbal_spin_translate.enabled=false",
            "env.gimbal.control_mode=velocity",
            "env.gimbal.randomize_heading=false",
            "env.ctrl_mode_obs_scale=[1.0,1.0,1.0,1.0,1.0,5.0,1.0]",
            "env.termination_duration_steps=10",
            "env.commands.special_mode_start_iterations=[0,0,0]",
            "env.commands.special_mode_probabilities=[0.15,0.15,0.20]",
            "env.commands.gimbal_mode_probability=0.0",
            "env.commands.zero_command_probability=0.10",
            "env.commands.zero_command_start_iteration=0",
            "env.height_range=[0.25,0.25]",
            "+env.max_episode_seconds=5.0",
        ),
    ),
    "Robotics-Wheelbipe-V14-Flat-DreamWaQ-v0": UpstreamWheelbipeCliRoute(
        canonical_task=CUSTOM_ONPOLICY_TASK,
        algorithm="dreamwaq",
        owner_registry_name="WheelbipeV14FlatDreamWaQ",
        generated_overrides=("training.task_name=WheelbipeV14FlatDreamWaQ",),
    ),
    "Robotics-Wheelbipe-V14-Flat-DreamWaQ-Play-v0": UpstreamWheelbipeCliRoute(
        canonical_task=CUSTOM_ONPOLICY_TASK,
        algorithm="dreamwaq",
        play_only=True,
        owner_registry_name="WheelbipeV14FlatDreamWaQPlay",
        generated_overrides=("training.task_name=WheelbipeV14FlatDreamWaQPlay",),
    ),
    "Robotics-Wheelbipe-V14-Flat-HIM-v0": UpstreamWheelbipeCliRoute(
        canonical_task=CUSTOM_ONPOLICY_TASK,
        algorithm="him_ppo",
        owner_registry_name="WheelbipeV14FlatHIM",
        generated_overrides=("training.task_name=WheelbipeV14FlatHIM",),
    ),
    "Robotics-Wheelbipe-V14-Flat-HIM-Play-v0": UpstreamWheelbipeCliRoute(
        canonical_task=CUSTOM_ONPOLICY_TASK,
        algorithm="him_ppo",
        play_only=True,
        owner_registry_name="WheelbipeV14FlatHIMPlay",
        generated_overrides=(
            "training.task_name=WheelbipeV14FlatHIMPlay",
            # The source Play class inherits the flat Play owner directly and
            # therefore has ``curriculum=None`` even though it shares the HIM
            # policy/history contract with the training owner.
            "env.him_curriculum.enabled=false",
        ),
    ),
    "Robotics-Wheelbipe-V14-Flat-NP3OBarlow-v0": UpstreamWheelbipeCliRoute(
        canonical_task=CUSTOM_ONPOLICY_TASK,
        algorithm="np3o",
        owner_registry_name="WheelbipeV14FlatNP3OBarlow",
        generated_overrides=("training.task_name=WheelbipeV14FlatNP3OBarlow",),
    ),
    "Robotics-Wheelbipe-V14-Flat-NP3OBarlow-Play-v0": UpstreamWheelbipeCliRoute(
        canonical_task=CUSTOM_ONPOLICY_TASK,
        algorithm="np3o",
        play_only=True,
        owner_registry_name="WheelbipeV14FlatNP3OBarlowPlay",
        generated_overrides=(
            "training.task_name=WheelbipeV14FlatNP3OBarlowPlay",
            "env.vel_height_gate_enabled=false",
            "env.np3o_tilt_limit_deg=15.0",
        ),
    ),
}
RESERVED_OVERRIDE_KEYS = {
    "algo",
    "profile",
    "task",
    "training.sim_backend",
    "training.play_only",
    "training.task_name",
}
TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Route:
    script_name: str
    config_group: str
    owner_task: str
    generated_overrides: tuple[str, ...]
    # Exact source Gym id, when the caller used one.  This is diagnostic
    # metadata only; Hydra still composes the canonical UniLab owner task.
    source_task_id: str | None = None
    # Source ``*-Play-*`` ids select the play-only lifecycle.  Keep this on
    # the route so train/eval wrappers cannot accidentally run a play profile
    # as a fresh training job.
    play_only: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_path(route: Route, root: Path) -> Path:
    return root / "scripts" / route.script_name


def _owner_yaml_path(route: Route, root: Path) -> Path:
    return root / "conf" / route.config_group / "task" / route.owner_task


def _check_private_checkout(root: Path) -> None:
    if not (root / "conf").is_dir() or not (root / "scripts").is_dir():
        raise SystemExit(
            "The current UniLab CLI expects a UniLab source checkout. "
            "Run it from the uv-managed editable environment created by this repo."
        )


def _check_reserved_overrides(overrides: Sequence[str]) -> None:
    reserved = [
        override for override in overrides if _override_key(override) in RESERVED_OVERRIDE_KEYS
    ]
    if reserved:
        joined = ", ".join(reserved)
        raise SystemExit(
            "Route-defining Hydra overrides must be provided through CLI flags, "
            f"not passthrough: {joined}"
        )


def _override_key(override: str) -> str:
    key = override.split("=", 1)[0].strip()
    return key.lstrip("+~")


def _check_task_name(task: str) -> None:
    if TASK_NAME_PATTERN.fullmatch(task) is None:
        raise SystemExit(
            "--task must be a registry or upstream task name such as `go1_joystick`; "
            "do not include slashes, dots, or path separators."
        )


def _check_profile(profile: str | None) -> None:
    if profile is None:
        return
    if TASK_NAME_PATTERN.fullmatch(profile) is None:
        raise SystemExit(
            "--profile must be a task owner/profile name such as `hora`; "
            "do not include slashes, dots, or path separators."
        )


def _check_load_run(load_run: str) -> str:
    if load_run == "-1":
        return load_run
    checkpoint_path = Path(load_run).expanduser()
    if checkpoint_path.is_absolute():
        # ``parse_checkpoint_path`` in the training owner accepts both an
        # absolute run directory and an absolute checkpoint file.  Preserve
        # that public contract here instead of forcing users through a Hydra
        # passthrough override that ``eval --help`` does not advertise.
        return str(checkpoint_path.resolve(strict=False))
    if RUN_ID_PATTERN.fullmatch(load_run) is None or load_run in {".", ".."}:
        raise SystemExit(
            "--load-run must be `-1`, a run directory name, or an absolute run/checkpoint path."
        )
    return load_run


def _check_runtime_requirements(algo: str, sim: str) -> None:
    if sim == "mujoco" and find_spec("mujoco") is None:
        raise SystemExit(
            "sim=mujoco requires the MuJoCo extra. Install it with `uv sync --extra mujoco`."
        )
    if sim == "mjwarp" and (find_spec("mujoco_warp") is None or find_spec("warp") is None):
        raise SystemExit(
            "sim=mjwarp requires the mjwarp extra. Install it with `uv sync --extra mjwarp`."
        )
    if sim == "motrix" and find_spec("motrixsim") is None:
        raise SystemExit(
            "sim=motrix requires the Motrix extra. Install it with `uv sync --extra motrix`."
        )


def _override_bool(overrides: Sequence[str], key: str) -> bool | None:
    selected: bool | None = None
    for override in overrides:
        if _override_key(override) != key or "=" not in override:
            continue
        value = override.split("=", 1)[1].strip().lower()
        if value in {"true", "1", "yes", "on"}:
            selected = True
        elif value in {"false", "0", "no", "off"}:
            selected = False
    return selected


def _override_value(overrides: Sequence[str], key: str) -> str | None:
    selected: str | None = None
    for override in overrides:
        if _override_key(override) != key or "=" not in override:
            continue
        selected = override.split("=", 1)[1].strip()
    return selected


def _needs_motrix_renderer(mode: str, sim: str, overrides: Sequence[str]) -> bool:
    if sim != "motrix":
        return False
    play_render_mode = _override_value(overrides, "training.play_render_mode")
    if play_render_mode is not None and play_render_mode.strip().lower() in {"none", "record"}:
        return False
    if mode == "eval":
        return True
    if mode == "train":
        return _override_bool(overrides, "training.no_play") is not True
    return False


def _python_executable_for_route(mode: str, sim: str, overrides: Sequence[str]) -> str:
    if platform.system() != "Darwin" or not _needs_motrix_renderer(mode, sim, overrides):
        return sys.executable

    return _mxpython_executable()


def _mxpython_executable() -> str:
    if Path(sys.executable).name == "mxpython":
        return sys.executable

    mxpython = shutil.which("mxpython")
    if mxpython is not None:
        return mxpython

    venv_mxpython = Path(sys.executable).with_name("mxpython")
    if venv_mxpython.is_file():
        return str(venv_mxpython)

    raise SystemExit(
        "macOS Motrix playback uses the native renderer and must be launched with "
        "`mxpython`. Install the Motrix extra so `mxpython` is on PATH, or use "
        "`training.no_play=true` for non-rendering training."
    )


def _mujoco_interactive_executable() -> str:
    """Select the same macOS-safe executable used by interactive demos."""

    if platform.system() != "Darwin":
        return sys.executable
    # Keep MuJoCo's macOS main-thread/app validation centralized in the
    # existing interactive-demo boundary instead of growing a divergent copy.
    from unilab.demo import _play_interactive_command_prefix

    return str(_play_interactive_command_prefix()[0])


def _resolve_upstream_cli_task(
    *, algo: str, task: str, profile: str | None
) -> tuple[str, UpstreamWheelbipeCliRoute | None]:
    """Translate a published Gym id to its UniLab owner task.

    We intentionally resolve this before checking for an owner YAML.  A
    recognized source profiles report capability problems (gimbal/FSM) instead
    of looking like a typo or an accidental path omission.
    """

    source_route = UPSTREAM_WHEELBIPE_CLI_ROUTES.get(task)
    if source_route is None:
        return task, None

    if profile is not None:
        raise SystemExit(
            f"Upstream task {task!r} already selects its owner profile; "
            "--profile cannot be combined with an exact upstream Wheelbipe id."
        )
    if algo != source_route.algorithm:
        raise SystemExit(
            f"Upstream task {task!r} requires --algo {source_route.algorithm!r}; got {algo!r}."
        )
    if not source_route.supported:
        reason = source_route.unsupported_reason or "the source capability is not available"
        raise SystemExit(
            f"Upstream task {task!r} is registered for migration but cannot be routed: "
            f"{reason}. Select a supported UniLab owner or port the required capability first."
        )
    return source_route.canonical_task, source_route


def build_route(algo: str, task: str, sim: str, profile: str | None = None) -> Route:
    task, source_route = _resolve_upstream_cli_task(algo=algo, task=task, profile=profile)
    source_task_id = next(
        (
            task_id
            for task_id, candidate in UPSTREAM_WHEELBIPE_CLI_ROUTES.items()
            if candidate is source_route
        ),
        None,
    )
    play_only = bool(source_route.play_only) if source_route is not None else False
    task_choice: str
    # Generic algorithms use profile as an owner YAML suffix (for example
    # ``sharpa_inhand/mujoco_hora``).  The migrated custom WheelBipe profiles
    # live in ``conf/custom_ppo/profile`` and therefore must leave the backend
    # owner untouched; ``build_command`` adds the Hydra group override below.
    owner = (
        sim
        if algo in CUSTOM_ONPOLICY_ALGOS
        else (f"{sim}_{profile}" if profile is not None else sim)
    )
    if algo in OFFPOLICY_ALGOS:
        task_choice = f"{algo}/{task}/{owner}"
        return Route(
            script_name="train_offpolicy.py",
            config_group="offpolicy",
            owner_task=f"{algo}/{task}/{owner}.yaml",
            generated_overrides=(f"algo={algo}", f"task={task_choice}"),
            source_task_id=source_task_id,
            play_only=play_only,
        )
    task_choice = f"{task}/{owner}"
    if algo == "ppo":
        generated = [f"task={task_choice}"]
        if source_route is not None:
            generated.extend(source_route.generated_overrides)
        return Route(
            script_name="train_rsl_rl.py",
            config_group="ppo",
            owner_task=f"{task}/{owner}.yaml",
            generated_overrides=tuple(generated),
            source_task_id=source_task_id,
            play_only=play_only,
        )
    if algo == "appo":
        generated = [f"task={task_choice}"]
        if source_route is not None:
            generated.extend(source_route.generated_overrides)
        return Route(
            script_name="train_appo.py",
            config_group="appo",
            owner_task=f"{task}/{owner}.yaml",
            generated_overrides=tuple(generated),
            source_task_id=source_task_id,
            play_only=play_only,
        )
    if algo in CUSTOM_ONPOLICY_ALGOS:
        if task != CUSTOM_ONPOLICY_TASK:
            raise SystemExit(
                f"Custom algorithm {algo!r} currently has a bounded owner only for "
                f"--task {CUSTOM_ONPOLICY_TASK}."
            )
        selected_profile = CUSTOM_ALGO_PROFILES[algo] if source_route is not None else profile
        if selected_profile is not None:
            expected_profile = CUSTOM_ALGO_PROFILES[algo]
            if selected_profile != expected_profile:
                raise SystemExit(
                    f"Custom algorithm {algo!r} supports --profile {expected_profile!r}; "
                    f"got {selected_profile!r}."
                )
        variant = CUSTOM_ALGO_TASK_VARIANTS[algo]
        generated = [f"task={variant}/{owner}", f"algo.algorithm_name={algo}"]
        if source_route is not None:
            generated.extend(source_route.generated_overrides)
        if selected_profile is not None:
            generated.append(f"profile={selected_profile}")
        if algo == "np3o":
            generated.append("env.num_costs=5")
        return Route(
            script_name="train_custom_ppo.py",
            config_group="custom_ppo",
            owner_task=f"{variant}/{owner}.yaml",
            generated_overrides=tuple(generated),
            source_task_id=source_task_id,
            play_only=play_only,
        )
    raise SystemExit(f"Unsupported algo={algo!r}; choose one of: {', '.join(SUPPORTED_ALGOS)}")


def build_command(
    *,
    mode: str,
    algo: str,
    task: str,
    sim: str,
    overrides: Sequence[str],
    profile: str | None = None,
    load_run: str | None = None,
    render_mode: str | None = None,
    root: Path | None = None,
) -> list[str]:
    selected_root = root or repo_root()
    _check_private_checkout(selected_root)
    _check_task_name(task)
    _check_profile(profile)
    _check_reserved_overrides(overrides)
    route = build_route(algo, task, sim, profile)
    if route.source_task_id is not None and algo in CUSTOM_ONPOLICY_ALGOS:
        fixed_history_overrides = [
            value for value in overrides if _override_key(value) == "algo.history_reset_mode"
        ]
        if fixed_history_overrides:
            raise SystemExit(
                "Exact upstream Wheelbipe custom IDs fix the source history reset contract; "
                "algo.history_reset_mode cannot be overridden."
            )
    _check_runtime_requirements(algo, sim)
    script = _script_path(route, selected_root)
    if not script.is_file():
        raise SystemExit(f"Entrypoint script not found: {script}")

    owner_yaml = _owner_yaml_path(route, selected_root)
    if not owner_yaml.is_file():
        raise SystemExit(
            f"No owner config exists for algo={algo}, task={task}, sim={sim}: {owner_yaml}"
        )

    if mode == "eval" and render_mode == "interactive" and algo in CUSTOM_ONPOLICY_ALGOS:
        raise SystemExit(
            "--render-mode interactive is not implemented for custom WheelBipe algorithms; "
            "use --render-mode none or the algorithm-specific validated sim2sim entrypoint."
        )

    if mode == "eval" and algo == "ppo" and sim == "mujoco" and render_mode == "interactive":
        interactive_script = selected_root / "scripts" / "play_interactive.py"
        if not interactive_script.is_file():
            raise SystemExit(f"Entrypoint script not found: {interactive_script}")
        task_override = next(
            value for value in route.generated_overrides if _override_key(value) == "task"
        )
        owner_choice = task_override.split("=", 1)[1]
        interactive_task, interactive_sim = owner_choice.rsplit("/", 1)
        generated_interactive = [
            value for value in route.generated_overrides if _override_key(value) != "task"
        ]
        if _override_value(overrides, "interactive.action_mode") is None:
            generated_interactive.append("interactive.action_mode=policy")
        if load_run is not None:
            checked_load_run = _check_load_run(load_run)
            if any(_override_key(value) == "algo.load_run" for value in overrides):
                raise SystemExit("Use either --load-run or algo.load_run=..., not both.")
            generated_interactive.append(f"algo.load_run={checked_load_run}")
        return [
            _mujoco_interactive_executable(),
            str(interactive_script),
            "--algo",
            "ppo",
            "--task",
            interactive_task,
            "--sim",
            interactive_sim,
            *generated_interactive,
            *overrides,
        ]

    generated = list(route.generated_overrides)
    if route.play_only:
        # Upstream ``*-Play-*`` ids are evaluation profiles.  UniLab's train
        # entrypoint can still execute that lifecycle explicitly, which keeps
        # one deterministic route for both wrappers while preventing a fresh
        # optimizer run from being mistaken for the source play profile.
        generated.append("training.play_only=true")
    # The dedicated custom owner is headless by default.  Make that config
    # default explicit in the routed command so the Motrix executable chooser
    # does not infer a renderer requirement merely because no passthrough
    # override was supplied (especially on macOS, where that would select
    # ``mxpython`` unnecessarily).  Explicit user overrides remain last-word
    # and therefore still opt into rendering.
    if algo in CUSTOM_ONPOLICY_ALGOS:
        if mode == "train" and _override_value(overrides, "training.no_play") is None:
            generated.append("training.no_play=true")
        if (
            mode == "eval"
            and render_mode is None
            and _override_value(overrides, "training.play_render_mode") is None
        ):
            generated.append("training.play_render_mode=none")
    if render_mode is not None:
        generated.append(f"training.play_render_mode={render_mode}")
    if mode == "eval":
        if not any(_override_key(value) == "training.play_only" for value in generated):
            generated.append("training.play_only=true")
        if load_run is not None:
            load_run = _check_load_run(load_run)
            if any(_override_key(o) == "algo.load_run" for o in overrides):
                raise SystemExit("Use either --load-run or algo.load_run=..., not both.")
            generated.append(f"algo.load_run={load_run}")

    executable = _python_executable_for_route(mode, sim, (*generated, *overrides))
    return [executable, str(script), *generated, *overrides]


def _train_eval_parser(*, mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=mode)
    parser.add_argument("--algo", required=True, choices=SUPPORTED_ALGOS)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sim", required=True, choices=SUPPORTED_SIMS)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--render-mode", choices=SUPPORTED_RENDER_MODES, default=None)
    if mode == "eval":
        parser.add_argument(
            "--load-run",
            default=None,
            help=("latest run (-1), a run directory name, or an absolute run/checkpoint path"),
        )
    return parser


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("demo_name")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def _run_train_eval(mode: str, argv: Sequence[str] | None = None) -> int:
    parser = _train_eval_parser(mode=mode)
    args, overrides = parser.parse_known_args(argv)

    command = build_command(
        mode=mode,
        algo=args.algo,
        task=args.task,
        sim=args.sim,
        profile=args.profile,
        overrides=overrides,
        load_run=getattr(args, "load_run", None),
        render_mode=args.render_mode,
    )
    return subprocess.run(command, check=False).returncode


def train_main(argv: Sequence[str] | None = None) -> int:
    return _run_train_eval("train", argv)


def eval_main(argv: Sequence[str] | None = None) -> int:
    return _run_train_eval("eval", argv)


def demo_main(argv: Sequence[str] | None = None) -> int:
    parser = _demo_parser()
    args, overrides = parser.parse_known_args(argv)
    if overrides:
        raise SystemExit(
            f"demo does not accept passthrough Hydra overrides: {', '.join(overrides)}"
        )
    return run_demo(
        demo_name=args.demo_name,
        refresh=args.refresh,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(train_main())
