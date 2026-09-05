#!/usr/bin/env -S uv run --script
"""Run a migrated WheelBipe custom actor in UniLab sim2sim.

The companion ROS2 repository intentionally supports only the normal
``35 -> 6`` graph.  This entrypoint is the UniLab-only path for the upstream
history-based algorithms: HIM-PPO, DreamWaQ and NP3O + Barlow.  The graph is
validated before an environment is created, and history management is shared
with the training/export contract in :mod:`unilab.training.wheelbipe`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# Importing the variants module is intentional: the base locomotion bootstrap
# keeps optional task modules lightweight, while these named custom owners are
# required explicitly by this entrypoint.
import unilab.envs.locomotion.wheelbipe_v14.variants  # noqa: F401
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.wheelbipe_v14 import (
    WHEELBIPE_DELAY_PROFILES,
    wheelbipe_delay_profile_overrides,
)
from unilab.training import BackendAdapter, create_env
from unilab.training.wheelbipe import (
    WheelbipeHistoryOnnxPolicy,
    WheelbipeSourceBarlowFullOnnxPolicy,
    WheelbipeSourceBarlowOnnxPolicy,
    WheelbipeSourceBarlowTorchScriptPolicy,
    is_wheelbipe_torchscript_archive,
    read_wheelbipe_custom_metadata,
    run_wheelbipe_history_policy,
    run_wheelbipe_source_barlow_full_policy,
    run_wheelbipe_source_barlow_policy,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
CUSTOM_CONFIG_DIR = ROOT_DIR / "conf" / "custom_ppo"

_ALGORITHM_TASKS: dict[str, tuple[str, int]] = {
    "him": ("WheelbipeV14FlatHIM", 5),
    "him_ppo": ("WheelbipeV14FlatHIM", 5),
    # Keep the source runner spellings accepted at the CLI boundary.  The
    # runtime/metadata layer canonicalizes these to the same owner; exposing
    # them here avoids a needless conversion step for existing source launch
    # commands.
    "ppo_him": ("WheelbipeV14FlatHIM", 5),
    "dreamwaq": ("WheelbipeV14FlatDreamWaQ", 5),
    "dream_waq": ("WheelbipeV14FlatDreamWaQ", 5),
    "ppo_dreamwaq": ("WheelbipeV14FlatDreamWaQ", 5),
    "np3o": ("WheelbipeV14FlatNP3OBarlow", 10),
}
_ALGORITHM_CANONICAL: dict[str, str] = {
    "him": "him",
    "him_ppo": "him",
    "ppo_him": "him",
    "dreamwaq": "dreamwaq",
    "dream_waq": "dreamwaq",
    "ppo_dreamwaq": "dreamwaq",
    "np3o": "np3o",
}
_ALGORITHM_OWNER_CONFIGS: dict[str, tuple[str, str]] = {
    "him": ("wheelbipe_v14_flat_him", "source_him_long"),
    "dreamwaq": ("wheelbipe_v14_flat_dreamwaq", "source_dreamwaq_long"),
    "np3o": ("wheelbipe_v14_flat_np3o", "source_np3o_barlow_long"),
}
_CUSTOM_ARTIFACTS = frozenset(
    {"compact_history_policy", "source_barlow_full", "source_barlow_actor"}
)


def _close_env(env: object) -> None:
    """Invoke the public environment lifecycle hook when one is provided."""

    close = getattr(env, "close", None)
    if callable(close):
        close()


def _compose_custom_owner_config(algorithm: str, sim: str) -> DictConfig:
    """Compose the same task/backend/profile owner used by exact training."""

    canonical = _ALGORITHM_CANONICAL[algorithm]
    task_owner, source_profile = _ALGORITHM_OWNER_CONFIGS[canonical]
    # This is a standalone entrypoint, so own Hydra's global compose lifecycle
    # explicitly.  Clearing in ``finally`` also keeps import-based tests from
    # leaking a custom config search path into later jobs.
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str(CUSTOM_CONFIG_DIR),
            version_base="1.3",
            job_name="wheelbipe_custom_sim2sim",
        ):
            return compose(
                config_name="config",
                overrides=[
                    f"task={task_owner}/{sim}",
                    f"profile={source_profile}",
                    "training.play_only=true",
                    "training.no_play=true",
                ],
            )
    finally:
        GlobalHydra.instance().clear()


def _artifact_kind(
    metadata: dict[str, object] | None,
    *,
    source_barlow_actor: bool,
) -> str:
    """Resolve one executable ABI from the export sidecar and explicit flag."""

    if metadata is None:
        return "source_barlow_actor" if source_barlow_actor else "compact_history_policy"
    raw_artifact = metadata.get("artifact")
    if raw_artifact is None:
        return "source_barlow_actor" if source_barlow_actor else "compact_history_policy"
    artifact = str(raw_artifact).strip().lower()
    if artifact not in _CUSTOM_ARTIFACTS:
        raise SystemExit(
            "Wheelbipe custom artifact metadata has unsupported artifact "
            f"{raw_artifact!r}; expected one of {', '.join(sorted(_CUSTOM_ARTIFACTS))}."
        )
    if source_barlow_actor and artifact != "source_barlow_actor":
        raise SystemExit(
            "--source-barlow-actor selected a graph whose metadata declares "
            f"artifact={artifact!r}; select barlow_twins_actor.onnx/.pt instead."
        )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=tuple(_ALGORITHM_TASKS),
        required=True,
        help="custom actor family represented by the exported graph",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "custom export from scripts/train_custom_ppo.py, or its run "
            "directory (which selects policy.onnx). Artifact metadata selects "
            "the compact 140D, source Barlow full 312D, or dual-input actor "
            "ABI. With --source-barlow-actor, a run directory or policy.onnx "
            "path selects the same-directory source actor (ONNX first, then "
            "TorchScript when present)."
        ),
    )
    parser.add_argument(
        "--source-barlow-actor",
        action="store_true",
        help=(
            "load the source-compatible dual-input Barlow actor export "
            "(obs [N,28], obs_hist [N,10,28]); normally its metadata selects "
            "this ABI automatically"
        ),
    )
    parser.add_argument(
        "--source-barlow-format",
        choices=("onnx", "torchscript"),
        default="onnx",
        help=(
            "source actor artifact to prefer when --model is a run directory "
            "or policy.onnx path (default: onnx; explicit .pt paths are always honored)"
        ),
    )
    parser.add_argument(
        "--sim",
        choices=("mujoco", "motrix"),
        default="mujoco",
        help="simulation backend",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "YAW"),
        default=None,
        help="override [linear-x, linear-y, yaw-rate] command",
    )
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--delay-profile",
        choices=WHEELBIPE_DELAY_PROFILES,
        # Compact HIM/DreamWaQ/NP3O owners are trained with the explicit
        # source-style 5 ms/4-substep timing contract by default.  Keeping the
        # CLI default aligned with the owner config prevents a misleading
        # ``local_physics`` label while the custom task still has source
        # timing/delay values enabled.  Selecting local explicitly replaces
        # those values with the stable 1 ms/control-step, delays-off owner
        # profile.  Both profiles remain timing-only and make no source
        # dynamics or asset parity claim.
        default="source_v14_physics",
        help=(
            "owner timing/delay profile (custom owners default to "
            "source_v14_physics); local_physics explicitly selects the 1 ms "
            "control-step, delays-off owner; source_v14_physics is explicit "
            "timing only and makes no source dynamics/asset parity claim"
        ),
    )
    return parser


def _resolve_model_path(
    model: Path | None,
    *,
    source_barlow_actor: bool,
    source_barlow_format: str = "onnx",
) -> Path:
    """Resolve an explicit model or source actor sibling artifact.

    A normal run directory selects its standard ``policy.onnx``; metadata on
    that artifact determines whether it is a compact history graph or the
    repaired one-input source Barlow graph.  ``--source-barlow-actor`` instead
    selects the sibling dual-input artifact.
    """

    if source_barlow_format not in {"onnx", "torchscript"}:
        raise ValueError(
            f"source_barlow_format must be 'onnx' or 'torchscript', got {source_barlow_format!r}"
        )

    if not source_barlow_actor:
        if model is None:
            raise SystemExit("--model is required unless --source-barlow-actor is selected")
        if model.is_dir():
            return model / "policy.onnx"
        return model

    suffix = ".onnx" if source_barlow_format == "onnx" else ".pt"

    def source_sibling(directory: Path) -> Path:
        """Select a source actor sibling while keeping explicit format strict."""

        preferred = directory / f"barlow_twins_actor{suffix}"
        if preferred.is_file() or source_barlow_format == "torchscript":
            return preferred
        # Existing source runs generally contain both artifacts.  Falling back
        # to a JIT sibling only when ONNX is absent makes a run directory useful
        # for freshly exported ``barlow_twins_actor.pt``-only deployments while
        # preserving the historical ONNX default.
        torchscript_sibling = directory / "barlow_twins_actor.pt"
        if torchscript_sibling.is_file():
            return torchscript_sibling
        return preferred

    if model is None:
        return source_sibling(Path.cwd())
    if model.is_dir():
        return source_sibling(model)
    candidate = model
    if candidate.name in {"policy.onnx", "model.onnx"}:
        sibling = source_sibling(candidate.parent)
        # With the legacy/default ONNX preference, retain the historical
        # behavior when no source sibling exists: ``policy.onnx`` itself is a
        # valid source graph in older exports.  An explicit TorchScript choice
        # must not silently fall back to that one-input runner graph.
        if sibling.is_file() or source_barlow_format == "torchscript":
            candidate = sibling
    return candidate


def _reject_non_source_torch_artifact(model_path: Path, *, source_barlow_actor: bool) -> None:
    """Fail closed before sending a pickle/one-input ``.pt`` to ONNX Runtime."""

    suffix = model_path.suffix.lower()
    if suffix not in {".pt", ".pth", ".ckpt"}:
        return
    if source_barlow_actor:
        raise SystemExit(
            "source Barlow --model must be a TorchScript archive or an ONNX graph; "
            f"the selected artifact is not a TorchScript archive: {model_path}"
        )
    raise SystemExit(
        "one-input custom sim2sim accepts an ONNX graph only; refusing "
        f"Torch/PyTorch artifact {model_path}. Use --source-barlow-actor for "
        "a two-input source Barlow actor."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be positive")
    if args.seed is not None:
        np.random.seed(args.seed)

    _task_name, expected_history = _ALGORITHM_TASKS[args.algorithm]
    canonical_algorithm = _ALGORITHM_CANONICAL[args.algorithm]
    if args.source_barlow_actor and _ALGORITHM_CANONICAL[args.algorithm] != "np3o":
        raise SystemExit("--source-barlow-actor is only valid with --algorithm np3o")
    model_path = _resolve_model_path(
        args.model,
        source_barlow_actor=args.source_barlow_actor,
        source_barlow_format=args.source_barlow_format,
    )
    metadata = read_wheelbipe_custom_metadata(model_path)
    artifact = _artifact_kind(
        metadata,
        source_barlow_actor=args.source_barlow_actor,
    )
    if artifact in {"source_barlow_actor", "source_barlow_full"} and canonical_algorithm != "np3o":
        raise SystemExit(f"artifact={artifact!r} is only valid with --algorithm np3o")

    # Do this before registry/env materialization.  A one-input ``policy.pt``
    # checkpoint is a pickle/state dict, not an executable deployment graph;
    # routing it through ONNX Runtime would hide the contract mismatch behind
    # an opaque parser error.
    policy: (
        WheelbipeHistoryOnnxPolicy
        | WheelbipeSourceBarlowFullOnnxPolicy
        | WheelbipeSourceBarlowOnnxPolicy
        | WheelbipeSourceBarlowTorchScriptPolicy
    )
    if artifact == "source_barlow_actor" and is_wheelbipe_torchscript_archive(model_path):
        policy = WheelbipeSourceBarlowTorchScriptPolicy(
            model_path,
            one_step_dim=28,
            expected_history_length=10,
            expected_algorithm="np3o",
        )
    else:
        _reject_non_source_torch_artifact(
            model_path,
            source_barlow_actor=artifact == "source_barlow_actor",
        )
        if artifact == "source_barlow_actor":
            policy = WheelbipeSourceBarlowOnnxPolicy(
                model_path,
                one_step_dim=28,
                expected_history_length=10,
                expected_algorithm="np3o",
            )
        elif artifact == "source_barlow_full":
            policy = WheelbipeSourceBarlowFullOnnxPolicy(
                model_path,
                expected_algorithm="np3o",
            )
        else:
            policy = WheelbipeHistoryOnnxPolicy(
                model_path,
                one_step_dim=28,
                expected_history_length=expected_history,
                expected_algorithm=canonical_algorithm,
            )
    ensure_registries()
    cfg = _compose_custom_owner_config(args.algorithm, args.sim)
    env_cfg_override = BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name=canonical_algorithm,
    ).build_play_env_cfg_override()
    # The composed backend/task/profile is the authoritative owner for reward,
    # curriculum, costs and backend compatibility.  The CLI selector is
    # intentionally limited to its explicit timing/delay contract.
    env_cfg_override.update(wheelbipe_delay_profile_overrides(args.delay_profile))
    env = create_env(
        cfg,
        num_envs=args.num_envs,
        env_cfg_override=env_cfg_override,
    )
    # Capture owner timing metadata while the backend is live.  Backends are
    # allowed to release timing/scene resources during ``close``; reporting a
    # contract after teardown made otherwise successful rollouts backend-
    # dependent.  Keep every post-create access inside the guarded lifecycle
    # so malformed adapter metadata cannot leak an environment.
    timing_contract = {}
    diagnostics: dict[str, float] | None = None
    try:
        timing_contract = dict(getattr(env, "timing_contract"))
        if artifact == "source_barlow_actor":
            diagnostics = run_wheelbipe_source_barlow_policy(
                env,
                cast(
                    WheelbipeSourceBarlowOnnxPolicy | WheelbipeSourceBarlowTorchScriptPolicy,
                    policy,
                ),
                steps=args.steps,
                command=args.command,
                height=args.height,
            )
        elif artifact == "source_barlow_full":
            diagnostics = run_wheelbipe_source_barlow_full_policy(
                env,
                cast(WheelbipeSourceBarlowFullOnnxPolicy, policy),
                steps=args.steps,
                command=args.command,
                height=args.height,
            )
        else:
            diagnostics = run_wheelbipe_history_policy(
                env,
                cast(WheelbipeHistoryOnnxPolicy, policy),
                steps=args.steps,
                command=args.command,
                height=args.height,
            )
    finally:
        _close_env(env)
    if diagnostics is None:
        raise RuntimeError("Wheelbipe custom sim2sim rollout did not return diagnostics")
    print(
        "Wheelbipe custom sim2sim complete: "
        f"algorithm={args.algorithm} artifact={artifact} "
        f"task={cfg.training.task_name} "
        f"delay_profile={timing_contract['profile']} "
        f"sim_dt={timing_contract['sim_dt']:.6f} "
        f"ctrl_dt={timing_contract['ctrl_dt']:.6f} "
        f"steps={int(diagnostics['steps'])} "
        f"mean_reward={diagnostics['mean_reward']:.6f} "
        f"done_count={int(diagnostics['done_count'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
