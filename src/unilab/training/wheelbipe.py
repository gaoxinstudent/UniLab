"""Wheelbipe V14 ONNX/TorchScript policy loading and sim2sim helpers.

The helper is intentionally small and framework-independent.  It validates the
deployment graph at the boundary, then accepts the observation emitted by the
UniLab Wheelbipe env (or a caller-built 35D vector) and returns six actions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast
from zipfile import BadZipFile, ZipFile

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.locomotion.wheelbipe_v14.base import (
    NUM_POLICY_ACTIONS,
    POLICY_OBS_CLIP,
    POLICY_OBS_DIM,
)

DEFAULT_WHEELBIPE_POLICY = (
    ASSETS_ROOT_PATH / "policies" / "wheelbipe_v14" / "V14-35-flat-and-rotation-13k.onnx"
)

# Custom exports are intentionally accompanied by a small, human-readable
# sidecar.  The graph shape remains the primary compatibility boundary, while
# this metadata lets the custom sim2sim route reject an algorithm/variant mixup
# before constructing a simulator.  A missing sidecar is accepted for legacy
# exports; a present but malformed sidecar fails closed.
CUSTOM_WHEELBIPE_METADATA_SCHEMA = "unilab.wheelbipe.custom_policy.v1"
_CUSTOM_ALGORITHM_ALIASES = {
    "him": "him",
    "him_ppo": "him",
    "ppo_him": "him",
    "dreamwaq": "dreamwaq",
    "dream_waq": "dreamwaq",
    "ppo_dreamwaq": "dreamwaq",
    "np3o": "np3o",
}
_CUSTOM_ALGORITHM_CONTRACTS = {
    "him": {"history_length": 5, "num_costs": 0},
    "dreamwaq": {"history_length": 5, "num_costs": 0},
    "np3o": {"history_length": 10, "num_costs": 5},
}
_CUSTOM_HISTORY_RESET_MODES = frozenset({"repeat", "source_zero_current"})


def _canonical_custom_history_reset_mode(value: object) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in _CUSTOM_HISTORY_RESET_MODES:
        supported = ", ".join(sorted(_CUSTOM_HISTORY_RESET_MODES))
        raise ValueError(
            f"Wheelbipe custom history_reset_mode must be one of {supported}; got {value!r}"
        )
    return mode


# Exact upstream Play aliases use a distinct registry/task name so their
# bounded environment variant remains visible.  Training, however, writes
# checkpoints under paired non-Play owners.  Keep the candidate order explicit
# at this owner boundary: PPO variants prefer the exact non-Play alias and then
# the canonical owner, while each custom Play ID has one algorithm-matched
# training root.  Callers retain latest-only/fail-closed selection semantics.
_WHEELBIPE_PLAY_CHECKPOINT_TASKS: dict[str, tuple[str, ...]] = {
    "WheelbipeV14FlatPlayV0": ("WheelbipeV14FlatV0", "WheelbipeV14Flat"),
    "WheelbipeV14FlatPlayV2": ("WheelbipeV14FlatV2", "WheelbipeV14Flat"),
    "WheelbipeV14RoughPlayV0": ("WheelbipeV14RoughV0", "WheelbipeV14Rough"),
    "WheelbipeV14RoughPlayV1": ("WheelbipeV14RoughV1", "WheelbipeV14Rough"),
    # The canonical task is kept for backwards-compatible training/eval
    # commands.  When ``--load-run=-1`` is used, prefer the source-compatible
    # rough-v1 owner before falling back to the legacy compatibility owner so
    # a checkpoint is not silently driven by a different state-machine ABI.
    "WheelbipeV14Rough": ("WheelbipeV14RoughV1", "WheelbipeV14RoughV0", "WheelbipeV14Rough"),
    "WheelbipeV14FlatDreamWaQPlay": ("WheelbipeV14FlatDreamWaQ",),
    "WheelbipeV14FlatHIMPlay": ("WheelbipeV14FlatHIM",),
    "WheelbipeV14FlatNP3OBarlowPlay": ("WheelbipeV14FlatNP3OBarlow",),
}


def wheelbipe_play_checkpoint_task_candidates(task_name: str) -> tuple[str, ...]:
    """Return non-Play task roots to try for an exact Wheelbipe Play owner.

    The result is empty for ordinary tasks.  This helper only describes
    metadata; callers decide whether a fallback is allowed (both standard and
    custom PPO lifecycles restrict it to the ``algo.load_run=-1`` latest-run
    sentinel).
    """

    return _WHEELBIPE_PLAY_CHECKPOINT_TASKS.get(str(task_name), ())


_WHEELBIPE_CANONICAL_TASK_FOR_SOURCE: dict[str, str] = {
    "WheelbipeV14FlatV0": "WheelbipeV14Flat",
    "WheelbipeV14FlatV1": "WheelbipeV14Flat",
    "WheelbipeV14FlatV2": "WheelbipeV14Flat",
    "WheelbipeV14RoughV0": "WheelbipeV14Rough",
    "WheelbipeV14RoughV1": "WheelbipeV14Rough",
}


def _wheelbipe_run_config(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir).expanduser()
    if path.is_file():
        path = path.parent
    config_path = path / "run_config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def hydrate_wheelbipe_play_config(source_run_dir: str | Path | None, target_cfg: Any) -> Any:
    """Adopt the checkpoint's exact Wheelbipe owner for canonical playback.

    The public ``wheelbipe_v14_rough`` route predates the source-v1 owner and
    therefore composes the legacy compatibility environment by default.  A
    source-v1 checkpoint has the same 35/78/6 tensor shapes, so loading it into
    that legacy environment can succeed while feeding the policy the wrong
    state-machine/control-mode semantics.  When the run sidecar identifies a
    paired exact owner, replace only the play config's task/env/reward sections
    with the recorded training view; backend selection and play-only overrides
    remain owned by the target CLI.

    Runs without a sidecar are left untouched and retain the existing
    fail-closed behavior.
    """

    if source_run_dir is None:
        return target_cfg
    payload = _wheelbipe_run_config(source_run_dir)
    if payload is None:
        return target_cfg

    # Import OmegaConf lazily so this policy/checkpoint helper remains usable
    # by deployment tools that only need ONNX/TorchScript inspection.
    from omegaconf import OmegaConf, open_dict

    source_run = payload.get("run")
    source_config = payload.get("config")
    source_task = source_run.get("task") if isinstance(source_run, dict) else None
    if source_task is None and isinstance(source_config, dict):
        source_training = source_config.get("training")
        if isinstance(source_training, dict):
            source_task = source_training.get("task_name")
    source_task = str(source_task or "").strip()
    canonical_task = _WHEELBIPE_CANONICAL_TASK_FOR_SOURCE.get(source_task)
    target_task = str(OmegaConf.select(target_cfg, "training.task_name", default="")).strip()
    if not source_task or canonical_task != target_task or source_task == target_task:
        return target_cfg

    hydrated = deepcopy(target_cfg)
    source_config = source_config if isinstance(source_config, dict) else {}
    source_env = source_config.get("env")
    source_reward = source_config.get("reward")
    with open_dict(hydrated):
        OmegaConf.update(hydrated, "training.task_name", source_task, merge=False)
        if isinstance(source_env, dict):
            # The sidecar stores the resolved CLI env section, which is a
            # partial owner override.  Replacing the canonical compatibility
            # mapping lets the registry fill omitted fields from the adopted
            # exact owner instead of retaining incompatible gimbal/FSM values.
            OmegaConf.update(hydrated, "env", deepcopy(source_env), merge=False)
        if isinstance(source_reward, dict):
            OmegaConf.update(hydrated, "reward", deepcopy(source_reward), merge=False)
    print(
        "[checkpoint] adopted Wheelbipe source owner for playback: "
        f"{target_task} -> {source_task} ({source_run_dir})"
    )
    return hydrated


def custom_wheelbipe_metadata_path(model_file: str | Path) -> Path:
    """Return the sidecar path used for a custom ``.onnx`` export.

    Appending ``.json`` (rather than replacing the suffix) keeps artifacts
    unambiguous when a caller chooses a non-standard graph filename, e.g.
    ``policy.onnx.json``.
    """

    return Path(model_file).expanduser().resolve().with_name(f"{Path(model_file).name}.json")


def _canonical_custom_algorithm(value: object) -> str:
    name = str(value).strip().lower()
    canonical = _CUSTOM_ALGORITHM_ALIASES.get(name)
    if canonical is None:
        raise ValueError(
            "Wheelbipe custom policy metadata has unsupported algorithm "
            f"{value!r}; expected one of {', '.join(sorted(_CUSTOM_ALGORITHM_CONTRACTS))}"
        )
    return canonical


def _variant_mentions_algorithm(variant_name: object, algorithm: str) -> bool:
    """Accept both owner slugs and upstream class-style variant labels."""

    normalized = str(variant_name).strip().lower().replace("_", "-")
    tokens = [token for token in normalized.split("-") if token]
    return algorithm in tokens or algorithm in "".join(tokens)


def read_wheelbipe_custom_metadata(model_file: str | Path) -> dict[str, Any] | None:
    """Read and minimally validate a custom ONNX metadata sidecar.

    ``None`` denotes a legacy export without a sidecar.  Once a sidecar is
    present it is part of the artifact contract: invalid JSON, a wrong schema,
    or a non-object payload is reported instead of silently ignored.
    """

    path = custom_wheelbipe_metadata_path(model_file)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Wheelbipe custom policy metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Wheelbipe custom policy metadata must contain a JSON object")
    schema = payload.get("schema")
    if schema != CUSTOM_WHEELBIPE_METADATA_SCHEMA:
        raise ValueError(
            "Wheelbipe custom policy metadata schema is unsupported: "
            f"expected {CUSTOM_WHEELBIPE_METADATA_SCHEMA!r}, got {schema!r}"
        )
    if "algorithm" in payload:
        _canonical_custom_algorithm(payload["algorithm"])
    return payload


def _validate_wheelbipe_custom_metadata(
    metadata: dict[str, Any],
    contract: "WheelbipeHistoryPolicyContract",
    *,
    expected_algorithm: str | None,
) -> None:
    """Validate sidecar fields against the inspected graph and CLI owner."""

    metadata_algorithm = metadata.get("algorithm")
    if metadata_algorithm is None:
        raise ValueError("Wheelbipe custom policy metadata must declare algorithm")
    canonical = _canonical_custom_algorithm(metadata_algorithm)
    if expected_algorithm is not None:
        expected = _canonical_custom_algorithm(expected_algorithm)
        if canonical != expected:
            raise ValueError(
                "Wheelbipe custom policy metadata algorithm does not match the selected route: "
                f"metadata={canonical!r}, expected={expected!r}"
            )

    expected_contract = _CUSTOM_ALGORITHM_CONTRACTS[canonical]
    if contract.history_length != int(expected_contract["history_length"]):
        raise ValueError(
            "Wheelbipe custom policy graph history does not match its metadata algorithm: "
            f"algorithm={canonical!r}, expected={expected_contract['history_length']}, "
            f"graph={contract.history_length}"
        )
    checks: tuple[tuple[str, int], ...] = (
        ("one_step_dim", contract.one_step_dim),
        ("history_length", contract.history_length),
        ("input_dim", contract.input_dim),
        ("output_dim", contract.output_dim),
        ("num_actions", contract.output_dim),
    )
    for key, expected_value in checks:
        if key not in metadata:
            continue
        try:
            actual = int(metadata[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Wheelbipe custom policy metadata {key} must be an integer") from exc
        if actual != expected_value:
            raise ValueError(
                f"Wheelbipe custom policy metadata {key} does not match the graph: "
                f"metadata={actual}, graph={expected_value}"
            )
    if "num_costs" in metadata:
        try:
            num_costs = int(metadata["num_costs"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Wheelbipe custom policy metadata num_costs must be an integer"
            ) from exc
        if num_costs != int(expected_contract["num_costs"]):
            raise ValueError(
                "Wheelbipe custom policy metadata num_costs does not match the selected algorithm: "
                f"metadata={num_costs}, expected={expected_contract['num_costs']}"
            )
    if "history_reset_mode" in metadata:
        _canonical_custom_history_reset_mode(metadata["history_reset_mode"])
    if "input_name" in metadata and str(metadata["input_name"]) != contract.input_name:
        raise ValueError(
            "Wheelbipe custom policy metadata input_name does not match the graph: "
            f"metadata={metadata['input_name']!r}, graph={contract.input_name!r}"
        )
    if "output_name" in metadata and str(metadata["output_name"]) != contract.output_name:
        raise ValueError(
            "Wheelbipe custom policy metadata output_name does not match the graph: "
            f"metadata={metadata['output_name']!r}, graph={contract.output_name!r}"
        )
    variant_name = str(metadata.get("variant_name", ""))
    if variant_name:
        if not _variant_mentions_algorithm(variant_name, canonical):
            raise ValueError(
                "Wheelbipe custom policy metadata variant does not identify its algorithm: "
                f"variant_name={variant_name!r}, algorithm={canonical!r}"
            )


@dataclass(frozen=True)
class WheelbipePolicyContract:
    input_name: str
    output_name: str
    input_dim: int = POLICY_OBS_DIM
    output_dim: int = NUM_POLICY_ACTIONS


@dataclass(frozen=True)
class WheelbipeHistoryPolicyContract:
    """I/O contract for a history-stacked custom Wheelbipe actor.

    The ROS deployment artifact is intentionally a separate, strict 35D
    contract (``WheelbipePolicyContract`` above).  HIM, DreamWaQ and NP3O
    actors are trained with the source project's compact one-step observation
    and consume a flattened history.  Keeping this contract distinct prevents
    a custom graph from being accidentally presented to the normal-only ROS
    controller.
    """

    input_name: str
    output_name: str
    one_step_dim: int
    history_length: int
    input_dim: int
    output_dim: int = NUM_POLICY_ACTIONS


@dataclass(frozen=True)
class WheelbipeSourceBarlowPolicyContract:
    """I/O contract for the upstream Barlow-Twins actor backbone.

    The source ``exporter_normal.py`` exports ``policy.actor_teacher_backbone``
    directly.  It is consequently *not* the same graph as the one-input
    312-dimensional ``on_constraint`` export emitted by
    :class:`~unilab.algos.torch.custom_ppo.runner.CustomOnPolicyRunner`.
    Keeping a separate contract makes that distinction machine-checkable:
    the actor receives the current 28-dimensional policy frame and a
    rank-three ten-frame history, and returns six actions.
    """

    input_name: str
    history_input_name: str
    output_name: str
    one_step_dim: int = 28
    history_length: int = 10
    output_dim: int = NUM_POLICY_ACTIONS

    @property
    def input_names(self) -> tuple[str, str]:
        return (self.input_name, self.history_input_name)

    @property
    def history_shape(self) -> tuple[int, int, int]:
        return (1, self.history_length, self.one_step_dim)

    @property
    def flattened_history_dim(self) -> int:
        return self.history_length * self.one_step_dim

    @property
    def source_stream_dim(self) -> int:
        """The equivalent source ``on_constraint`` width (28 + 4 + 280)."""

        return self.one_step_dim + 4 + self.flattened_history_dim


@dataclass(frozen=True)
class WheelbipeSourceBarlowFullPolicyContract:
    """I/O contract for the runner-exported one-input source Barlow graph."""

    input_name: str
    output_name: str
    one_step_dim: int = 28
    privileged_latent_dim: int = 4
    history_length: int = 10
    output_dim: int = NUM_POLICY_ACTIONS

    @property
    def history_feature_dim(self) -> int:
        return self.one_step_dim * self.history_length

    @property
    def input_dim(self) -> int:
        return self.one_step_dim + self.privileged_latent_dim + self.history_feature_dim


def _shape_dim(value: Any, index: int) -> int | None:
    try:
        dim = value[index]
    except (IndexError, TypeError):
        return None
    return int(dim) if isinstance(dim, (int, np.integer)) else None


def _strict_shape(value: Any, expected: tuple[int, int], *, label: str) -> None:
    """Require the fixed batch-and-feature shape exposed by the ROS graph."""

    try:
        shape = tuple(value)
    except TypeError as exc:
        raise ValueError(f"Wheelbipe {label} shape is not a rank-2 tensor: {value!r}") from exc
    if shape != expected:
        raise ValueError(f"Wheelbipe {label} must have strict shape {expected!r}, got {shape!r}")


def is_wheelbipe_torchscript_archive(model_file: str | Path) -> bool:
    """Return whether ``model_file`` has the TorchScript archive layout.

    ``torch.load(..., weights_only=True)`` intentionally rejects TorchScript
    archives on recent PyTorch releases.  Inspecting the zip members first lets
    callers choose the explicit ``torch.jit.load`` path without attempting an
    unsafe pickle fallback for an ordinary state-dict checkpoint.  This helper
    only identifies the container; :class:`WheelbipeTorchScriptPolicy` still
    validates the executable graph's 35D/6D contract before use.
    """

    path = Path(model_file).expanduser()
    if not path.is_file():
        return False
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
    except (BadZipFile, OSError, ValueError):
        return False
    # TorchScript's zip format stores serialized code below a ``code`` member
    # and a ``constants.pkl`` file.  There are two layouts in the wild:
    # ``code/__torch__/package/module.py`` (nested module directory) and the
    # minimal ``code/__torch__.py`` form emitted for a parameter-free traced
    # wrapper.  Inspect path components instead of assuming one archive stem;
    # otherwise a valid source actor can be rejected before its two-input ABI
    # is validated.
    has_torch_code = any(
        any(
            component == "__torch__" or component.startswith("__torch__.py")
            for component in name.replace("\\", "/").split("/")
        )
        and name.endswith((".py", ".py.debug_pkl"))
        for name in names
    )
    return has_torch_code and any(
        name.replace("\\", "/").split("/")[-1] == "constants.pkl" for name in names
    )


def _strict_source_shape(value: Any, expected: tuple[int, ...], *, label: str) -> None:
    """Require a fixed source actor tensor shape.

    Source deployment graphs are deliberately traced with a static batch of
    one.  Checking rank and every dimension here gives a useful diagnostic
    before an opaque ONNX Runtime error, and prevents accidentally loading the
    one-input 312D graph through the two-input actor contract.
    """

    try:
        shape = tuple(value)
    except TypeError as exc:
        raise ValueError(f"Wheelbipe source {label} shape is invalid: {value!r}") from exc
    if shape != expected:
        raise ValueError(
            f"Wheelbipe source {label} must have strict shape {expected!r}, got {shape!r}"
        )


def _validate_source_barlow_metadata(
    metadata: dict[str, Any],
    contract: WheelbipeSourceBarlowPolicyContract,
    *,
    expected_algorithm: str | None,
) -> None:
    """Validate the additive sidecar emitted for source Barlow artifacts."""

    metadata_algorithm = metadata.get("algorithm")
    if metadata_algorithm is None:
        raise ValueError("Wheelbipe source Barlow metadata must declare algorithm")
    canonical = _canonical_custom_algorithm(metadata_algorithm)
    if canonical != "np3o":
        raise ValueError(
            f"Wheelbipe source Barlow actor requires NP3O metadata; got algorithm={canonical!r}"
        )
    if expected_algorithm is not None:
        expected = _canonical_custom_algorithm(expected_algorithm)
        if expected != canonical:
            raise ValueError(
                "Wheelbipe source Barlow metadata algorithm does not match the selected route: "
                f"metadata={canonical!r}, expected={expected!r}"
            )

    architecture = metadata.get("architecture")
    if architecture is not None and str(architecture).strip().lower() != "source_barlow":
        raise ValueError(
            "Wheelbipe source Barlow metadata architecture must be 'source_barlow', "
            f"got {architecture!r}"
        )
    artifact = metadata.get("artifact")
    if artifact is not None and str(artifact).strip().lower() != "source_barlow_actor":
        raise ValueError(
            "Wheelbipe source Barlow metadata artifact must be 'source_barlow_actor', "
            f"got {artifact!r}"
        )

    integer_checks: tuple[tuple[str, int], ...] = (
        ("one_step_dim", contract.one_step_dim),
        ("history_length", contract.history_length),
        ("history_feature_dim", contract.flattened_history_dim),
        ("source_stream_dim", contract.source_stream_dim),
        ("output_dim", contract.output_dim),
        ("num_actions", contract.output_dim),
    )
    for key, expected_value in integer_checks:
        if key not in metadata:
            continue
        try:
            actual = int(metadata[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Wheelbipe source Barlow metadata {key} must be an integer") from exc
        if actual != expected_value:
            raise ValueError(
                f"Wheelbipe source Barlow metadata {key} does not match the graph: "
                f"metadata={actual}, graph={expected_value}"
            )

    if "num_costs" in metadata:
        try:
            num_costs = int(metadata["num_costs"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Wheelbipe source Barlow metadata num_costs must be an integer"
            ) from exc
        if num_costs != 5:
            raise ValueError(
                "Wheelbipe source Barlow actor metadata num_costs must be exactly five, "
                f"got {num_costs}"
            )
    if "history_reset_mode" in metadata:
        history_reset_mode = _canonical_custom_history_reset_mode(metadata["history_reset_mode"])
        if history_reset_mode != "source_zero_current":
            raise ValueError(
                "Wheelbipe source Barlow metadata history_reset_mode must be "
                f"'source_zero_current', got {history_reset_mode!r}"
            )

    if "input_names" in metadata:
        raw_names = metadata["input_names"]
        if not isinstance(raw_names, (list, tuple)) or len(raw_names) != 2:
            raise ValueError("Wheelbipe source Barlow metadata input_names must contain two names")
        if tuple(str(name) for name in raw_names) != contract.input_names:
            raise ValueError(
                "Wheelbipe source Barlow metadata input_names do not match the graph: "
                f"metadata={raw_names!r}, graph={contract.input_names!r}"
            )
    if "input_shapes" in metadata:
        raw_shapes = metadata["input_shapes"]
        expected_shapes = [
            [1, contract.one_step_dim],
            [1, contract.history_length, contract.one_step_dim],
        ]
        if raw_shapes != expected_shapes:
            raise ValueError(
                "Wheelbipe source Barlow metadata input_shapes do not match the graph: "
                f"metadata={raw_shapes!r}, graph={expected_shapes!r}"
            )
    if "output_shape" in metadata:
        expected_output_shape = [1, contract.output_dim]
        if metadata["output_shape"] != expected_output_shape:
            raise ValueError(
                "Wheelbipe source Barlow metadata output_shape does not match the graph: "
                f"metadata={metadata['output_shape']!r}, graph={expected_output_shape!r}"
            )
    for key, actual_name in (
        ("input_name", contract.input_name),
        ("history_input_name", contract.history_input_name),
        ("output_name", contract.output_name),
    ):
        if key in metadata and str(metadata[key]) != actual_name:
            raise ValueError(
                f"Wheelbipe source Barlow metadata {key} does not match the graph: "
                f"metadata={metadata[key]!r}, graph={actual_name!r}"
            )

    variant_name = str(metadata.get("variant_name", ""))
    if variant_name and not _variant_mentions_algorithm(variant_name, canonical):
        raise ValueError(
            "Wheelbipe source Barlow metadata variant does not identify NP3O: "
            f"variant_name={variant_name!r}"
        )


def _validate_source_barlow_full_metadata(
    metadata: dict[str, Any],
    contract: WheelbipeSourceBarlowFullPolicyContract,
    *,
    expected_algorithm: str | None,
) -> None:
    """Validate the sidecar for the runner's repaired 312D source stream."""

    metadata_algorithm = metadata.get("algorithm")
    if metadata_algorithm is None:
        raise ValueError("Wheelbipe source Barlow full metadata must declare algorithm")
    canonical = _canonical_custom_algorithm(metadata_algorithm)
    if canonical != "np3o":
        raise ValueError(
            "Wheelbipe source Barlow full policy requires NP3O metadata; "
            f"got algorithm={canonical!r}"
        )
    if expected_algorithm is not None:
        expected = _canonical_custom_algorithm(expected_algorithm)
        if expected != canonical:
            raise ValueError(
                "Wheelbipe source Barlow full metadata algorithm does not match the "
                f"selected route: metadata={canonical!r}, expected={expected!r}"
            )
    architecture = str(metadata.get("architecture", "")).strip().lower()
    if architecture != "source_barlow":
        raise ValueError(
            "Wheelbipe source Barlow full metadata architecture must be "
            f"'source_barlow', got {metadata.get('architecture')!r}"
        )
    artifact = str(metadata.get("artifact", "")).strip().lower()
    if artifact != "source_barlow_full":
        raise ValueError(
            "Wheelbipe source Barlow full metadata artifact must be "
            f"'source_barlow_full', got {metadata.get('artifact')!r}"
        )
    integer_checks = (
        ("one_step_dim", contract.one_step_dim),
        ("history_length", contract.history_length),
        ("input_dim", contract.input_dim),
        ("output_dim", contract.output_dim),
        ("num_actions", contract.output_dim),
        ("num_estimate", contract.privileged_latent_dim),
        ("num_costs", 5),
    )
    for key, expected_value in integer_checks:
        if key not in metadata:
            continue
        try:
            actual = int(metadata[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Wheelbipe source Barlow full metadata {key} must be an integer"
            ) from exc
        if actual != expected_value:
            raise ValueError(
                f"Wheelbipe source Barlow full metadata {key} does not match the graph: "
                f"metadata={actual}, graph={expected_value}"
            )
    history_reset_mode = _canonical_custom_history_reset_mode(
        metadata.get("history_reset_mode", "source_zero_current")
    )
    if history_reset_mode != "source_zero_current":
        raise ValueError(
            "Wheelbipe source Barlow full metadata history_reset_mode must be "
            f"'source_zero_current', got {history_reset_mode!r}"
        )
    if "input_name" in metadata and str(metadata["input_name"]) != contract.input_name:
        raise ValueError(
            "Wheelbipe source Barlow full metadata input_name does not match the graph: "
            f"metadata={metadata['input_name']!r}, graph={contract.input_name!r}"
        )
    if "output_name" in metadata and str(metadata["output_name"]) != contract.output_name:
        raise ValueError(
            "Wheelbipe source Barlow full metadata output_name does not match the graph: "
            f"metadata={metadata['output_name']!r}, graph={contract.output_name!r}"
        )
    variant_name = str(metadata.get("variant_name", ""))
    if variant_name and not _variant_mentions_algorithm(variant_name, canonical):
        raise ValueError(
            "Wheelbipe source Barlow full metadata variant does not identify NP3O: "
            f"variant_name={variant_name!r}"
        )


def inspect_wheelbipe_onnx(model_file: str | Path) -> WheelbipePolicyContract:
    """Inspect and validate the strict normal-mode ONNX I/O contract."""

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency is project runtime default
        raise RuntimeError("onnxruntime is required for Wheelbipe sim2sim") from exc

    path = Path(model_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Wheelbipe ONNX policy does not exist: {path}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"Wheelbipe policy must have one input and one output; got {len(inputs)} / {len(outputs)}"
        )
    input_shape = inputs[0].shape
    output_shape = outputs[0].shape
    _strict_shape(input_shape, (1, POLICY_OBS_DIM), label="policy input")
    _strict_shape(output_shape, (1, NUM_POLICY_ACTIONS), label="policy output")
    if str(inputs[0].type) != "tensor(float)" or str(outputs[0].type) != "tensor(float)":
        raise ValueError(
            "Wheelbipe policy input/output must both use tensor(float) (float32), "
            f"got {inputs[0].type!r} / {outputs[0].type!r}"
        )
    input_dim = _shape_dim(input_shape, -1)
    output_dim = _shape_dim(output_shape, -1)
    if input_dim != POLICY_OBS_DIM:
        raise ValueError(
            f"Wheelbipe policy input must be 35D, got shape {input_shape!r} ({input_dim})"
        )
    if output_dim != NUM_POLICY_ACTIONS:
        raise ValueError(
            f"Wheelbipe policy output must be 6D, got shape {output_shape!r} ({output_dim})"
        )
    return WheelbipePolicyContract(
        input_name=str(inputs[0].name),
        output_name=str(outputs[0].name),
    )


class WheelbipeOnnxPolicy:
    """CPU ONNX Runtime adapter for the shipped six-action policy."""

    def __init__(
        self,
        model_file: str | Path = DEFAULT_WHEELBIPE_POLICY,
        *,
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime is required for Wheelbipe sim2sim") from exc
        self.model_file = Path(model_file).expanduser().resolve()
        if not self.model_file.is_file():
            raise FileNotFoundError(f"Wheelbipe ONNX policy does not exist: {self.model_file}")
        selected_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_file), providers=selected_providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Wheelbipe ONNX graph must expose exactly one input and one output")
        self.contract = inspect_wheelbipe_onnx(self.model_file)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on ``(N, 35)`` or ``(35,)`` observations."""

        arr = np.asarray(observation, dtype=np.float32)
        was_vector = arr.ndim == 1
        if was_vector:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != POLICY_OBS_DIM:
            raise ValueError(f"Wheelbipe observation must have shape (N, 35), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Wheelbipe observation contains non-finite values")
        # The released graph has a static batch dimension of one.  Keep this
        # diagnostic explicit instead of returning an opaque ORT shape error.
        input_shape = self.session.get_inputs()[0].shape
        static_batch = _shape_dim(input_shape, 0)
        if static_batch is not None and static_batch != arr.shape[0]:
            if static_batch != 1 or arr.shape[0] % static_batch != 0:
                raise ValueError(
                    f"Wheelbipe ONNX graph accepts batch {static_batch}, got {arr.shape[0]}"
                )
            # A static-one deployment graph is intentionally evaluated one row
            # at a time for callers that request a vectorized rollout.
            outputs = [
                np.asarray(
                    self.session.run(
                        [self.contract.output_name],
                        {self.contract.input_name: row[None, :]},
                    )[0]
                )[0]
                for row in arr
            ]
            result = np.asarray(outputs, dtype=np.float32)
        else:
            result = np.asarray(
                self.session.run([self.contract.output_name], {self.contract.input_name: arr})[0],
                dtype=np.float32,
            )
        if result.ndim != 2 or result.shape[1] != NUM_POLICY_ACTIONS:
            raise ValueError(f"Wheelbipe ONNX output must have shape (N, 6), got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError("Wheelbipe ONNX output contains non-finite values")
        return result[0] if was_vector else result

    __call__ = predict


class WheelbipeTorchScriptPolicy:
    """TorchScript adapter for the upstream normal ``35 -> 6`` actor.

    The source training repository publishes both state-dict checkpoints
    (``model_*.pt``) and inference-only TorchScript artifacts
    (``policy.pt``).  The latter cannot be read with
    ``torch.load(weights_only=True)``; this owner-layer adapter uses the
    explicit JIT loader and applies the same strict shape/finite checks as the
    ONNX adapter.  TorchScript deserializes executable code, so callers should
    only pass trusted artifacts.

    ``predict`` is convenient for NumPy-based sim2sim callers.  ``__call__``
    preserves a Torch tensor/device so the normal PPO playback wrapper can use
    the artifact without constructing a second policy network.
    """

    def __init__(self, model_file: str | Path, *, device: Any = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a project dependency
            raise RuntimeError("torch is required for Wheelbipe TorchScript playback") from exc

        self.model_file = Path(model_file).expanduser().resolve()
        if not self.model_file.is_file():
            raise FileNotFoundError(
                f"Wheelbipe TorchScript policy does not exist: {self.model_file}"
            )
        if not is_wheelbipe_torchscript_archive(self.model_file):
            raise ValueError(
                "Wheelbipe TorchScript policy must be a serialized TorchScript archive: "
                f"{self.model_file}"
            )

        self.device = torch.device(device)
        try:
            # ``map_location`` keeps the module and playback observations on
            # one device.  Unlike ``torch.load``, this is intentionally the
            # only deserialization path for executable JIT artifacts.
            self.module = torch.jit.load(str(self.model_file), map_location=self.device).eval()
        except (RuntimeError, OSError, ValueError) as exc:
            raise ValueError(
                f"Wheelbipe TorchScript policy could not be loaded: {self.model_file}"
            ) from exc

        self.input_dtype = torch.float32
        for tensor in self.module.parameters():
            if tensor.is_floating_point():
                self.input_dtype = tensor.dtype
                break
        else:
            for tensor in self.module.buffers():
                if tensor.is_floating_point():
                    self.input_dtype = tensor.dtype
                    break

        self.contract = WheelbipePolicyContract(input_name="obs", output_name="actions")
        try:
            with torch.inference_mode():
                probe = torch.zeros(
                    (1, POLICY_OBS_DIM),
                    dtype=self.input_dtype,
                    device=self.device,
                )
                result = self.module(probe)
            self._validate_tensor_output(result, 1, label="TorchScript policy probe")
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Wheelbipe TorchScript policy must implement the strict "
                f"({POLICY_OBS_DIM},) -> ({NUM_POLICY_ACTIONS},) actor contract"
            ) from exc

    @staticmethod
    def _validate_tensor_output(result: Any, batch_size: int, *, label: str) -> Any:
        """Validate a TorchScript result and return its tensor unchanged."""

        import torch

        if not isinstance(result, torch.Tensor):
            raise ValueError(f"Wheelbipe {label} must return a tensor, got {type(result).__name__}")
        expected_shape = (int(batch_size), NUM_POLICY_ACTIONS)
        if tuple(result.shape) != expected_shape:
            raise ValueError(
                f"Wheelbipe {label} must return shape {expected_shape}, got {tuple(result.shape)}"
            )
        if not result.is_floating_point():
            raise ValueError(f"Wheelbipe {label} must return a floating tensor")
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"Wheelbipe {label} returned non-finite actions")
        return result

    def _forward_tensor(self, observations: Any) -> Any:
        """Run the module, with a row-wise fallback for static-batch-one JIT graphs."""

        import torch

        # Keep a sentinel for the branch where a vectorized call raises and
        # the row-wise fallback is selected.  The batch-one error is re-raised
        # below, so the sentinel is never returned to the validator.
        result: Any = None
        try:
            result = self.module(observations)
            if isinstance(result, torch.Tensor) and tuple(result.shape) == (
                int(observations.shape[0]),
                NUM_POLICY_ACTIONS,
            ):
                return result
        except RuntimeError:
            # A few older exports hard-code a batch of one.  Retry explicitly
            # per row for vectorized UniLab playback while allowing the
            # original error to surface for a true single-row failure.
            if int(observations.shape[0]) == 1:
                raise
        if int(observations.shape[0]) == 1:
            return result
        rows = [
            self.module(observations[index : index + 1]) for index in range(observations.shape[0])
        ]
        return torch.cat(rows, dim=0)

    def _predict_tensor(self, observations: Any) -> Any:
        """Validate and execute a rank-two Torch observation tensor."""

        import torch

        if observations.ndim != 2 or observations.shape[1] != POLICY_OBS_DIM:
            raise ValueError(
                "Wheelbipe TorchScript observation must have shape "
                f"(N, {POLICY_OBS_DIM}), got {tuple(observations.shape)}"
            )
        if observations.shape[0] < 1:
            raise ValueError("Wheelbipe TorchScript observation batch must not be empty")
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("Wheelbipe TorchScript observation contains non-finite values")
        model_input = observations.to(device=self.device, dtype=self.input_dtype)
        with torch.inference_mode():
            result = self._forward_tensor(model_input)
        return self._validate_tensor_output(
            result, int(observations.shape[0]), label="TorchScript policy"
        )

    def predict(self, observation: Any) -> np.ndarray:
        """Run inference on ``(N, 35)`` or a single 35D NumPy observation."""

        import torch

        if isinstance(observation, Mapping):
            for key in ("actor", "policy", "obs"):
                if key in observation:
                    observation = observation[key]
                    break
            else:
                raise ValueError(
                    "Wheelbipe TorchScript observation mapping must contain an "
                    "'actor', 'policy', or 'obs' tensor"
                )
        if hasattr(observation, "detach"):
            tensor = observation.detach()
            was_vector = tensor.ndim == 1
            if was_vector:
                tensor = tensor.unsqueeze(0)
            result = self._predict_tensor(tensor)
            output = result.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            arr = np.asarray(observation, dtype=np.float32)
            was_vector = arr.ndim == 1
            if was_vector:
                arr = arr[None, :]
            if arr.ndim != 2 or arr.shape[1] != POLICY_OBS_DIM:
                raise ValueError(
                    "Wheelbipe TorchScript observation must have shape "
                    f"(N, {POLICY_OBS_DIM}), got {arr.shape}"
                )
            if arr.shape[0] < 1:
                raise ValueError("Wheelbipe TorchScript observation batch must not be empty")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Wheelbipe TorchScript observation contains non-finite values")
            tensor = torch.from_numpy(arr)
            output = self._predict_tensor(tensor).detach().to(dtype=torch.float32).cpu().numpy()
        return output[0] if was_vector else output

    def __call__(self, observation: Any) -> Any:
        """Run inference while preserving tensor semantics for PPO playback."""

        import torch

        if isinstance(observation, Mapping):
            for key in ("actor", "policy", "obs"):
                if key in observation:
                    observation = observation[key]
                    break
            else:
                raise ValueError(
                    "Wheelbipe TorchScript observation mapping must contain an "
                    "'actor', 'policy', or 'obs' tensor"
                )
        if isinstance(observation, torch.Tensor):
            result = self._predict_tensor(observation)
            # Playback wrappers expect the action tensor on the same device as
            # their observations; normalize only dtype, not placement.
            return result.to(device=observation.device, dtype=torch.float32)
        result = self.predict(observation)
        return torch.as_tensor(result, dtype=torch.float32, device=self.device)


def inspect_wheelbipe_history_onnx(
    model_file: str | Path,
    *,
    one_step_dim: int = 28,
    expected_history_length: int | None = None,
    expected_algorithm: str | None = None,
) -> WheelbipeHistoryPolicyContract:
    """Inspect a compact, history-stacked custom actor export.

    Custom actors are deliberately *not* accepted by
    :func:`inspect_wheelbipe_onnx`: the ROS controller only understands the
    fixed ``[1, 35] -> [1, 6]`` normal graph.  This boundary accepts the
    flattened ``[1, 28 * history] -> [1, 6]`` graph emitted by
    :class:`~unilab.algos.torch.custom_ppo.runner.CustomOnPolicyRunner` and
    records the history length for the sim2sim adapter.
    """

    one_step_dim = int(one_step_dim)
    if one_step_dim < 1:
        raise ValueError(f"one_step_dim must be positive, got {one_step_dim}")
    if expected_history_length is not None and int(expected_history_length) < 1:
        raise ValueError(f"expected_history_length must be positive, got {expected_history_length}")

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency is project runtime default
        raise RuntimeError("onnxruntime is required for Wheelbipe custom sim2sim") from exc

    path = Path(model_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Wheelbipe custom ONNX policy does not exist: {path}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "Wheelbipe custom policy must have one input and one output; "
            f"got {len(inputs)} / {len(outputs)}"
        )

    input_shape = tuple(inputs[0].shape)
    output_shape = tuple(outputs[0].shape)
    if len(input_shape) != 2 or input_shape[0] != 1:
        raise ValueError(
            "Wheelbipe custom policy input must have strict rank-2 shape "
            f"[1, 28*history], got {input_shape!r}"
        )
    input_dim = _shape_dim(input_shape, 1)
    if input_dim is None or input_dim < one_step_dim or input_dim % one_step_dim:
        raise ValueError(
            "Wheelbipe custom policy input feature dimension must be a positive "
            f"multiple of {one_step_dim}, got {input_shape!r}"
        )
    history_length = input_dim // one_step_dim
    if expected_history_length is not None and history_length != int(expected_history_length):
        raise ValueError(
            "Wheelbipe custom policy history length does not match the selected task: "
            f"expected {int(expected_history_length)}, got {history_length}"
        )
    if output_shape != (1, NUM_POLICY_ACTIONS):
        raise ValueError(
            "Wheelbipe custom policy output must have strict shape "
            f"(1, {NUM_POLICY_ACTIONS}), got {output_shape!r}"
        )
    if str(inputs[0].type) != "tensor(float)" or str(outputs[0].type) != "tensor(float)":
        raise ValueError(
            "Wheelbipe custom policy input/output must both use tensor(float), "
            f"got {inputs[0].type!r} / {outputs[0].type!r}"
        )
    contract = WheelbipeHistoryPolicyContract(
        input_name=str(inputs[0].name),
        output_name=str(outputs[0].name),
        one_step_dim=one_step_dim,
        history_length=history_length,
        input_dim=input_dim,
    )
    metadata = read_wheelbipe_custom_metadata(path)
    if metadata is not None:
        _validate_wheelbipe_custom_metadata(
            metadata,
            contract,
            expected_algorithm=expected_algorithm,
        )
    elif expected_algorithm is not None:
        # A legacy graph without metadata remains shape-compatible.  The
        # expected algorithm is still checked against the graph's history so
        # a HIM/Dream route cannot accidentally load an NP3O export.
        canonical = _canonical_custom_algorithm(expected_algorithm)
        expected_history = _CUSTOM_ALGORITHM_CONTRACTS[canonical]["history_length"]
        if history_length != expected_history:
            raise ValueError(
                "Wheelbipe custom policy history does not match the selected algorithm: "
                f"algorithm={canonical!r}, expected={expected_history}, got={history_length}"
            )
    return contract


def inspect_wheelbipe_source_barlow_full_onnx(
    model_file: str | Path,
    *,
    expected_algorithm: str | None = "np3o",
) -> WheelbipeSourceBarlowFullPolicyContract:
    """Inspect the runner-exported one-input 312D source Barlow graph."""

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency is project runtime default
        raise RuntimeError("onnxruntime is required for source Barlow full sim2sim") from exc

    path = Path(model_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Wheelbipe source Barlow full policy does not exist: {path}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "Wheelbipe source Barlow full policy must have one input and one output; "
            f"got {len(inputs)} / {len(outputs)}"
        )
    contract = WheelbipeSourceBarlowFullPolicyContract(
        input_name=str(inputs[0].name),
        output_name=str(outputs[0].name),
    )
    _strict_source_shape(inputs[0].shape, (1, contract.input_dim), label="full policy input")
    _strict_source_shape(
        outputs[0].shape,
        (1, contract.output_dim),
        label="full policy output",
    )
    if str(inputs[0].type) != "tensor(float)" or str(outputs[0].type) != "tensor(float)":
        raise ValueError(
            "Wheelbipe source Barlow full input/output must both use tensor(float), "
            f"got {inputs[0].type!r} / {outputs[0].type!r}"
        )
    metadata = read_wheelbipe_custom_metadata(path)
    if metadata is None:
        raise ValueError(
            "Wheelbipe source Barlow full policy requires its export metadata sidecar; "
            "a bare 312D graph cannot prove the repaired stream ordering"
        )
    _validate_source_barlow_full_metadata(
        metadata,
        contract,
        expected_algorithm=expected_algorithm,
    )
    return contract


def inspect_wheelbipe_source_barlow_onnx(
    model_file: str | Path,
    *,
    one_step_dim: int = 28,
    expected_history_length: int | None = 10,
    expected_algorithm: str | None = None,
) -> WheelbipeSourceBarlowPolicyContract:
    """Inspect the source-compatible two-input Barlow actor graph.

    The checked-in upstream exporter emits a graph equivalent to
    ``actor_teacher_backbone(obs_prop, obs_hist)`` with input names ``obs``
    and ``obs_hist``.  This contract intentionally accepts the names exposed
    by a graph (so an older source export using ``nn_input0``/``nn_input1``
    remains shape-loadable), while validating the source V14 dimensions and
    any present metadata sidecar.
    """

    one_step_dim = int(one_step_dim)
    if one_step_dim < 1:
        raise ValueError(f"source Barlow one_step_dim must be positive, got {one_step_dim}")
    if expected_history_length is not None and int(expected_history_length) < 1:
        raise ValueError(
            f"source Barlow expected_history_length must be positive, got {expected_history_length}"
        )

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency is project runtime default
        raise RuntimeError("onnxruntime is required for source Barlow sim2sim") from exc

    path = Path(model_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Wheelbipe source Barlow ONNX policy does not exist: {path}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 2 or len(outputs) != 1:
        raise ValueError(
            "Wheelbipe source Barlow actor must have two inputs and one output; "
            f"got {len(inputs)} / {len(outputs)}"
        )
    if expected_history_length is None:
        history_length = _shape_dim(inputs[1].shape, 1)
        if history_length is None or history_length < 1:
            raise ValueError(
                "Wheelbipe source Barlow history input must expose a static frame dimension"
            )
    else:
        history_length = int(expected_history_length)

    _strict_source_shape(
        inputs[0].shape,
        (1, one_step_dim),
        label="actor input",
    )
    _strict_source_shape(
        inputs[1].shape,
        (1, history_length, one_step_dim),
        label="history input",
    )
    _strict_source_shape(outputs[0].shape, (1, NUM_POLICY_ACTIONS), label="actor output")
    for index, input_value in enumerate(inputs):
        if str(input_value.type) != "tensor(float)":
            raise ValueError(
                "Wheelbipe source Barlow actor inputs must use tensor(float), "
                f"input {index} has {input_value.type!r}"
            )
    if str(outputs[0].type) != "tensor(float)":
        raise ValueError(
            f"Wheelbipe source Barlow actor output must use tensor(float), got {outputs[0].type!r}"
        )
    names = [str(item.name) for item in inputs]
    if len(set(names)) != 2:
        raise ValueError(
            f"Wheelbipe source Barlow actor input names must be distinct, got {names!r}"
        )
    contract = WheelbipeSourceBarlowPolicyContract(
        input_name=names[0],
        history_input_name=names[1],
        output_name=str(outputs[0].name),
        one_step_dim=one_step_dim,
        history_length=history_length,
    )
    metadata = read_wheelbipe_custom_metadata(path)
    if metadata is not None:
        _validate_source_barlow_metadata(
            metadata,
            contract,
            expected_algorithm=expected_algorithm,
        )
    elif expected_algorithm is not None:
        canonical = _canonical_custom_algorithm(expected_algorithm)
        if canonical != "np3o":
            raise ValueError(
                f"source Barlow actor is only compatible with the NP3O algorithm, got {canonical!r}"
            )
    return contract


class WheelbipeHistoryOnnxPolicy:
    """ONNX Runtime adapter for compact HIM/DreamWaQ/NP3O actors.

    The exported custom graph has a static batch of one.  A vectorized
    sim2sim rollout may still request multiple envs; in that case this adapter
    evaluates one row at a time and preserves the same deterministic graph
    contract instead of relying on an opaque runtime shape error.
    """

    def __init__(
        self,
        model_file: str | Path,
        *,
        one_step_dim: int = 28,
        expected_history_length: int | None = None,
        expected_algorithm: str | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime is required for Wheelbipe custom sim2sim") from exc
        self.model_file = Path(model_file).expanduser().resolve()
        if not self.model_file.is_file():
            raise FileNotFoundError(
                f"Wheelbipe custom ONNX policy does not exist: {self.model_file}"
            )
        selected_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_file), providers=selected_providers)
        self.contract = inspect_wheelbipe_history_onnx(
            self.model_file,
            one_step_dim=one_step_dim,
            expected_history_length=expected_history_length,
            expected_algorithm=expected_algorithm,
        )
        self.metadata = read_wheelbipe_custom_metadata(self.model_file)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on ``(N, 28 * history)`` or one flattened vector."""

        arr = np.asarray(observation, dtype=np.float32)
        was_vector = arr.ndim == 1
        if was_vector:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.contract.input_dim:
            raise ValueError(
                "Wheelbipe custom observation must have shape "
                f"(N, {self.contract.input_dim}), got {arr.shape}"
            )
        if arr.shape[0] < 1:
            raise ValueError("Wheelbipe custom observation batch must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Wheelbipe custom observation contains non-finite values")

        # The exported graph is intentionally static-batch one.  Preserve
        # support for vectorized UniLab envs with explicit row-wise calls.
        if arr.shape[0] == 1:
            result = np.asarray(
                self.session.run(
                    [self.contract.output_name],
                    {self.contract.input_name: arr},
                )[0],
                dtype=np.float32,
            )
        else:
            result = np.asarray(
                [
                    np.asarray(
                        self.session.run(
                            [self.contract.output_name],
                            {self.contract.input_name: row[None, :]},
                        )[0]
                    )[0]
                    for row in arr
                ],
                dtype=np.float32,
            )
        if result.shape != (arr.shape[0], NUM_POLICY_ACTIONS):
            raise ValueError(
                "Wheelbipe custom ONNX output must have shape "
                f"({arr.shape[0]}, {NUM_POLICY_ACTIONS}), got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("Wheelbipe custom ONNX output contains non-finite values")
        return result[0] if was_vector else result

    __call__ = predict


class WheelbipeSourceBarlowFullOnnxPolicy:
    """ONNX adapter for the runner's one-input repaired 312D NP3O graph."""

    def __init__(
        self,
        model_file: str | Path,
        *,
        expected_algorithm: str | None = "np3o",
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime is required for source Barlow full sim2sim") from exc
        self.model_file = Path(model_file).expanduser().resolve()
        if not self.model_file.is_file():
            raise FileNotFoundError(
                f"Wheelbipe source Barlow full ONNX policy does not exist: {self.model_file}"
            )
        selected_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_file), providers=selected_providers)
        self.contract = inspect_wheelbipe_source_barlow_full_onnx(
            self.model_file,
            expected_algorithm=expected_algorithm,
        )
        metadata = read_wheelbipe_custom_metadata(self.model_file)
        if metadata is None:  # guarded by the inspector; keeps the attribute statically concrete
            raise ValueError("Wheelbipe source Barlow full metadata sidecar is required")
        self.metadata = metadata

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on one or more repaired 312D source streams."""

        arr = np.asarray(observation, dtype=np.float32)
        was_vector = arr.ndim == 1
        if was_vector:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.contract.input_dim:
            raise ValueError(
                "Wheelbipe source Barlow full observation must have shape "
                f"(N, {self.contract.input_dim}), got {arr.shape}"
            )
        if arr.shape[0] < 1:
            raise ValueError("Wheelbipe source Barlow full batch must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Wheelbipe source Barlow full observation contains non-finite values")
        if arr.shape[0] == 1:
            result = self.session.run(
                [self.contract.output_name],
                {self.contract.input_name: arr},
            )[0]
        else:
            result = np.asarray(
                [
                    np.asarray(
                        self.session.run(
                            [self.contract.output_name],
                            {self.contract.input_name: row[None, :]},
                        )[0]
                    )[0]
                    for row in arr
                ],
                dtype=np.float32,
            )
        output = _source_barlow_output(result, arr.shape[0], label="full ONNX policy")
        return output[0] if was_vector else output

    __call__ = predict


def _coerce_source_barlow_inputs(
    observation: Any,
    history: Any,
    contract: WheelbipeSourceBarlowPolicyContract,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Normalize source actor inputs while preserving the caller's batch rank."""

    # Deployment callers normally provide NumPy arrays, while the custom
    # runner/tests may hand a CPU Torch tensor directly.  Detach only at this
    # boundary; the exported artifact itself remains inference-only.
    if hasattr(observation, "detach"):
        observation = observation.detach().cpu().numpy()
    if hasattr(history, "detach"):
        history = history.detach().cpu().numpy()
    obs = np.asarray(observation, dtype=np.float32)
    was_vector = obs.ndim == 1
    if was_vector:
        obs = obs[None, :]
    if obs.ndim != 2 or obs.shape[1] != contract.one_step_dim:
        raise ValueError(
            "Wheelbipe source Barlow observation must have shape "
            f"(N, {contract.one_step_dim}) or ({contract.one_step_dim},), got {obs.shape}"
        )

    hist = np.asarray(history, dtype=np.float32)
    if hist.ndim == 2:
        if not was_vector:
            raise ValueError(
                "Wheelbipe source Barlow history must include a batch dimension "
                f"for batched observations, got {hist.shape}"
            )
        hist = hist[None, :, :]
    if hist.ndim != 3 or hist.shape[1:] != (
        contract.history_length,
        contract.one_step_dim,
    ):
        raise ValueError(
            "Wheelbipe source Barlow history must have shape "
            f"(N, {contract.history_length}, {contract.one_step_dim}), got {hist.shape}"
        )
    if hist.shape[0] != obs.shape[0]:
        raise ValueError(
            "Wheelbipe source Barlow observation/history batch sizes must match: "
            f"obs={obs.shape[0]}, history={hist.shape[0]}"
        )
    if obs.shape[0] < 1:
        raise ValueError("Wheelbipe source Barlow observation batch must not be empty")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(hist)):
        raise ValueError("Wheelbipe source Barlow inputs must contain only finite values")
    return obs, hist, was_vector


def _source_barlow_output(result: Any, batch_size: int, *, label: str) -> np.ndarray:
    """Validate and normalize a six-action source actor result."""

    # TorchScript returns a Tensor while ONNX Runtime returns ndarray.  Avoid
    # importing torch in this framework-independent training helper solely for
    # an ``isinstance`` check; both expose a NumPy conversion path.  Check the
    # source dtype before coercing to float32, otherwise an accidental integer
    # actor would pass after a lossy implicit cast.
    if hasattr(result, "is_floating_point"):
        try:
            if not bool(result.is_floating_point()):
                raise ValueError(f"Wheelbipe source Barlow {label} must return a floating tensor")
        except TypeError as exc:
            raise ValueError(
                f"Wheelbipe source Barlow {label} has an invalid tensor dtype"
            ) from exc
    elif isinstance(result, np.ndarray) and not np.issubdtype(result.dtype, np.floating):
        raise ValueError(
            f"Wheelbipe source Barlow {label} must return a floating array, got {result.dtype}"
        )
    # ``hasattr`` narrows an ``Any`` result to NumPy's static type under
    # pyright; keep the dynamic adapter operation explicitly typed because a
    # TorchScript tensor is the other supported result carrier.
    result_value: Any = result
    if hasattr(result_value, "detach"):
        result = result_value.detach().cpu().numpy()
    output = np.asarray(result, dtype=np.float32)
    if output.shape != (batch_size, NUM_POLICY_ACTIONS):
        raise ValueError(
            f"Wheelbipe source Barlow {label} must return shape "
            f"({batch_size}, {NUM_POLICY_ACTIONS}), got {output.shape}"
        )
    if not np.all(np.isfinite(output)):
        raise ValueError(f"Wheelbipe source Barlow {label} returned non-finite actions")
    return output


class WheelbipeSourceBarlowOnnxPolicy:
    """ONNX Runtime adapter for the source two-input Barlow actor.

    This adapter is intentionally separate from :class:`WheelbipeHistoryOnnxPolicy`:
    the latter consumes one flattened history tensor, while this class mirrors
    the upstream ``exporter_normal.py`` ABI ``(obs, obs_hist)`` exactly.
    """

    def __init__(
        self,
        model_file: str | Path,
        *,
        one_step_dim: int = 28,
        expected_history_length: int | None = 10,
        expected_algorithm: str | None = "np3o",
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - project runtime default
            raise RuntimeError("onnxruntime is required for source Barlow sim2sim") from exc
        self.model_file = Path(model_file).expanduser().resolve()
        if not self.model_file.is_file():
            raise FileNotFoundError(
                f"Wheelbipe source Barlow ONNX policy does not exist: {self.model_file}"
            )
        selected_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_file), providers=selected_providers)
        self.contract = inspect_wheelbipe_source_barlow_onnx(
            self.model_file,
            one_step_dim=one_step_dim,
            expected_history_length=expected_history_length,
            expected_algorithm=expected_algorithm,
        )
        self.metadata = read_wheelbipe_custom_metadata(self.model_file)

    def predict(self, observation: Any, history: Any) -> np.ndarray:
        """Run source actor inference on one or more ``(obs, obs_hist)`` pairs."""

        obs, hist, was_vector = _coerce_source_barlow_inputs(
            observation,
            history,
            self.contract,
        )
        input_shape = self.session.get_inputs()[0].shape
        static_batch = _shape_dim(input_shape, 0)
        feed = {
            self.contract.input_name: obs,
            self.contract.history_input_name: hist,
        }
        if obs.shape[0] == 1 or static_batch is None or static_batch != 1:
            result = self.session.run([self.contract.output_name], feed)[0]
            output = _source_barlow_output(result, obs.shape[0], label="ONNX actor")
        else:
            # Source exporter traces a static batch of one.  Preserve support
            # for vectorized UniLab rollouts with explicit row-wise calls,
            # matching the compact history adapter's behavior.
            rows = []
            for index in range(obs.shape[0]):
                row_result = self.session.run(
                    [self.contract.output_name],
                    {
                        self.contract.input_name: obs[index : index + 1],
                        self.contract.history_input_name: hist[index : index + 1],
                    },
                )[0]
                rows.append(_source_barlow_output(row_result, 1, label="ONNX actor")[0])
            output = np.asarray(rows, dtype=np.float32)
        return output[0] if was_vector else output

    __call__ = predict


class WheelbipeSourceBarlowTorchScriptPolicy:
    """TorchScript adapter for the source two-input Barlow actor export."""

    def __init__(
        self,
        model_file: str | Path,
        *,
        one_step_dim: int = 28,
        expected_history_length: int = 10,
        expected_algorithm: str | None = "np3o",
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a project dependency
            raise RuntimeError("torch is required for source Barlow TorchScript playback") from exc
        self.model_file = Path(model_file).expanduser().resolve()
        if not self.model_file.is_file():
            raise FileNotFoundError(
                f"Wheelbipe source Barlow TorchScript policy does not exist: {self.model_file}"
            )
        if not is_wheelbipe_torchscript_archive(self.model_file):
            raise ValueError(
                "Wheelbipe source Barlow TorchScript policy must be a serialized "
                f"TorchScript archive: {self.model_file}"
            )
        try:
            self.module = torch.jit.load(str(self.model_file), map_location="cpu").eval()
        except (RuntimeError, OSError, ValueError) as exc:
            raise ValueError(
                f"Wheelbipe source Barlow TorchScript policy could not be loaded: {self.model_file}"
            ) from exc
        # ``exporter_normal.py`` optionally emits a half-precision JIT actor.
        # Infer the graph's floating input dtype from its serialized
        # parameters/buffers so a deployment caller can continue to provide
        # ordinary NumPy float32 observations.  Float32 remains the fallback
        # for parameter-free scripted wrappers.
        self.input_dtype = torch.float32
        for tensor in self.module.parameters():
            if tensor.is_floating_point():
                self.input_dtype = tensor.dtype
                break
        else:
            for tensor in self.module.buffers():
                if tensor.is_floating_point():
                    self.input_dtype = tensor.dtype
                    break
        self.contract = WheelbipeSourceBarlowPolicyContract(
            input_name="obs",
            history_input_name="obs_hist",
            output_name="actions",
            one_step_dim=int(one_step_dim),
            history_length=int(expected_history_length),
        )
        # TorchScript does not expose ONNX-style input metadata reliably.
        # Execute one deterministic probe at construction to enforce the same
        # two-input/output shape contract before a rollout starts; this also
        # catches accidentally passing the one-input 312D runner artifact.
        try:
            with torch.inference_mode():
                probe = self.module(
                    torch.zeros(
                        1,
                        self.contract.one_step_dim,
                        dtype=self.input_dtype,
                    ),
                    torch.zeros(
                        1,
                        self.contract.history_length,
                        self.contract.one_step_dim,
                        dtype=self.input_dtype,
                    ),
                )
            _source_barlow_output(probe, 1, label="TorchScript actor probe")
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Wheelbipe source Barlow TorchScript actor does not satisfy the "
                f"two-input ({self.contract.one_step_dim}, "
                f"{self.contract.history_length}, {self.contract.one_step_dim}) "
                f"-> six-action contract: {exc}"
            ) from exc
        metadata = read_wheelbipe_custom_metadata(self.model_file)
        if metadata is not None:
            _validate_source_barlow_metadata(
                metadata,
                self.contract,
                expected_algorithm=expected_algorithm,
            )
        elif (
            expected_algorithm is not None
            and _canonical_custom_algorithm(expected_algorithm) != "np3o"
        ):
            raise ValueError(
                "source Barlow actor is only compatible with the NP3O algorithm, "
                f"got {expected_algorithm!r}"
            )

    def _predict_row(self, obs: np.ndarray, hist: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            result = self.module(
                torch.from_numpy(obs).to(dtype=self.input_dtype),
                torch.from_numpy(hist).to(dtype=self.input_dtype),
            )
        return _source_barlow_output(result, obs.shape[0], label="TorchScript actor")

    def predict(self, observation: Any, history: Any) -> np.ndarray:
        obs, hist, was_vector = _coerce_source_barlow_inputs(
            observation,
            history,
            self.contract,
        )
        if obs.shape[0] == 1:
            output = self._predict_row(obs, hist)
        else:
            output = np.asarray(
                [
                    self._predict_row(obs[index : index + 1], hist[index : index + 1])[0]
                    for index in range(obs.shape[0])
                ],
                dtype=np.float32,
            )
        return output[0] if was_vector else output

    __call__ = predict


def _source_barlow_metadata(
    contract: WheelbipeSourceBarlowPolicyContract,
    *,
    variant_name: str = "",
) -> dict[str, Any]:
    """Build the sidecar shared by the source JIT and ONNX artifacts."""

    return {
        "schema": CUSTOM_WHEELBIPE_METADATA_SCHEMA,
        "artifact": "source_barlow_actor",
        "architecture": "source_barlow",
        "algorithm": "np3o",
        "variant_name": str(variant_name),
        "one_step_dim": int(contract.one_step_dim),
        "history_length": int(contract.history_length),
        "history_reset_mode": "source_zero_current",
        "history_feature_dim": int(contract.flattened_history_dim),
        "source_stream_dim": int(contract.source_stream_dim),
        "input_shapes": [
            [1, int(contract.one_step_dim)],
            [1, int(contract.history_length), int(contract.one_step_dim)],
        ],
        "output_shape": [1, int(contract.output_dim)],
        "input_names": list(contract.input_names),
        "input_name": contract.input_name,
        "history_input_name": contract.history_input_name,
        "output_name": contract.output_name,
        "output_dim": int(contract.output_dim),
        "num_actions": int(contract.output_dim),
        "num_costs": 5,
    }


def _write_source_barlow_metadata(
    artifact_path: str | Path,
    contract: WheelbipeSourceBarlowPolicyContract,
    *,
    variant_name: str = "",
) -> None:
    metadata_path = custom_wheelbipe_metadata_path(artifact_path)
    metadata_path.write_text(
        json.dumps(
            _source_barlow_metadata(contract, variant_name=variant_name), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def _source_barlow_policy_backbone(policy: Any) -> tuple[Any, WheelbipeSourceBarlowPolicyContract]:
    """Resolve and validate a source policy's deployable teacher backbone."""

    if not bool(getattr(policy, "source_architecture", False)):
        raise ValueError(
            "source Barlow actor export requires the explicit source_barlow policy graph; "
            "compact NP3O checkpoints are not source-compatible"
        )
    try:
        backbone = policy.actor_teacher_backbone
        one_step_dim = int(policy.num_prop)
        history_length = int(policy.num_hist)
        output_dim = int(policy.num_actions)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "source Barlow actor export requires policy.actor_teacher_backbone, "
            "num_prop, num_hist and num_actions"
        ) from exc
    if one_step_dim != 28 or history_length != 10 or output_dim != NUM_POLICY_ACTIONS:
        raise ValueError(
            "source Barlow V14 actor export requires obs=(N,28), "
            f"obs_hist=(N,10,28), actions=(N,6); got ({one_step_dim},{history_length},{output_dim})"
        )
    if not callable(backbone):
        raise TypeError("policy.actor_teacher_backbone must be callable")
    contract = WheelbipeSourceBarlowPolicyContract(
        input_name="obs",
        history_input_name="obs_hist",
        output_name="actions",
        one_step_dim=one_step_dim,
        history_length=history_length,
        output_dim=output_dim,
    )
    return backbone, contract


def export_mlp_barlow_twins_actor_torchscript(
    backbone: Any,
    num_prop: int,
    num_hist: int,
    export_dir: str | Path,
    *,
    device: Any = "cpu",
    filename: str = "model.pt",
    use_fp16: bool = False,
) -> str:
    """Export a source ``MlpBarlowTwinsActor`` with its two-input ABI."""

    import copy

    import torch

    num_prop, num_hist = int(num_prop), int(num_hist)
    if num_prop != 28 or num_hist != 10:
        raise ValueError(
            "source Barlow TorchScript export requires num_prop=28 and num_hist=10, "
            f"got ({num_prop}, {num_hist})"
        )
    export_root = Path(export_dir).expanduser()
    export_root.mkdir(parents=True, exist_ok=True)
    out_path = export_root / str(filename)
    module = copy.deepcopy(backbone).to(device).eval()
    dtype = torch.float16 if use_fp16 else torch.float32
    if use_fp16:
        module = module.half()
    obs_prop = torch.randn(1, num_prop, device=device, dtype=dtype)
    obs_hist = torch.randn(1, num_hist, num_prop, device=device, dtype=dtype)

    class _BarlowTwinsActorExportWrapper(torch.nn.Module):
        def __init__(self, actor: Any) -> None:
            super().__init__()
            self.actor = actor

        def forward(self, obs: torch.Tensor, obs_hist_value: torch.Tensor) -> torch.Tensor:
            return self.actor(obs, obs_hist_value)

    wrapper = _BarlowTwinsActorExportWrapper(module).to(device).eval()
    if use_fp16:
        wrapper = wrapper.half()
    with torch.inference_mode():
        traced = cast(Any, torch.jit.trace(wrapper, (obs_prop, obs_hist)))
    traced.save(str(out_path))
    return str(out_path)


def export_mlp_barlow_twins_actor_onnx(
    backbone: Any,
    num_prop: int,
    num_hist: int,
    export_dir: str | Path,
    *,
    device: Any = "cpu",
    filename: str = "model.onnx",
    opset_version: int = 13,
    verbose: bool = False,
) -> str:
    """Export a source ``MlpBarlowTwinsActor`` to ONNX (``obs``, ``obs_hist``)."""

    import copy

    import torch

    num_prop, num_hist = int(num_prop), int(num_hist)
    if num_prop != 28 or num_hist != 10:
        raise ValueError(
            "source Barlow ONNX export requires num_prop=28 and num_hist=10, "
            f"got ({num_prop}, {num_hist})"
        )
    if int(opset_version) < 11:
        raise ValueError(f"source Barlow ONNX opset_version must be >=11, got {opset_version}")
    export_root = Path(export_dir).expanduser()
    export_root.mkdir(parents=True, exist_ok=True)
    out_path = export_root / str(filename)
    module = copy.deepcopy(backbone).to(device).eval()
    obs_prop = torch.randn(1, num_prop, device=device, dtype=torch.float32)
    obs_hist = torch.randn(1, num_hist, num_prop, device=device, dtype=torch.float32)

    class _BarlowTwinsActorExportWrapper(torch.nn.Module):
        def __init__(self, actor: Any) -> None:
            super().__init__()
            self.actor = actor

        def forward(self, obs: torch.Tensor, obs_hist_value: torch.Tensor) -> torch.Tensor:
            return self.actor(obs, obs_hist_value)

    wrapper = _BarlowTwinsActorExportWrapper(module).to(device).eval()
    with torch.inference_mode():
        # Match the source exporter warm-up call before tracing/exporting.
        _ = wrapper(obs_prop, obs_hist)
        torch.onnx.export(
            wrapper,
            (obs_prop, obs_hist),
            str(out_path),
            input_names=["obs", "obs_hist"],
            output_names=["actions"],
            export_params=True,
            opset_version=int(opset_version),
            verbose=bool(verbose),
        )
    return str(out_path)


def export_barlow_twins_actor_from_policy(
    policy: Any,
    export_dir: str | Path,
    *,
    device: Any = "cpu",
    use_fp16_jit: bool = False,
    jit_filename: str = "policy.pt",
    onnx_filename: str = "policy.onnx",
    variant_name: str = "",
) -> tuple[str, str]:
    """Export the source policy's teacher actor as TorchScript and ONNX."""

    backbone, contract = _source_barlow_policy_backbone(policy)
    jit_path = export_mlp_barlow_twins_actor_torchscript(
        backbone,
        contract.one_step_dim,
        contract.history_length,
        export_dir,
        device=device,
        filename=jit_filename,
        use_fp16=use_fp16_jit,
    )
    onnx_path = export_mlp_barlow_twins_actor_onnx(
        backbone,
        contract.one_step_dim,
        contract.history_length,
        export_dir,
        device=device,
        filename=onnx_filename,
    )
    # Sidecars are additive and written only after both artifacts are complete.
    # This lets deployment tooling distinguish the source two-input actor from
    # the runner's one-input 312D graph without changing either graph ABI.
    _write_source_barlow_metadata(jit_path, contract, variant_name=variant_name)
    _write_source_barlow_metadata(onnx_path, contract, variant_name=variant_name)
    return jit_path, onnx_path


def export_actor_critic_barlow_twins_actor(
    policy: Any,
    checkpoint_path: str | Path,
    *,
    device: Any = "cpu",
    use_fp16_jit: bool = False,
    jit_filename: str = "barlow_twins_actor.pt",
    onnx_filename: str = "barlow_twins_actor.onnx",
    variant_name: str = "",
) -> tuple[str, str]:
    """Compatibility entry point mirroring the upstream exporter helper."""

    return export_barlow_twins_actor_from_policy(
        policy,
        Path(checkpoint_path).expanduser().resolve().parent,
        device=device,
        use_fp16_jit=use_fp16_jit,
        jit_filename=jit_filename,
        onnx_filename=onnx_filename,
        variant_name=variant_name,
    )


def run_wheelbipe_policy(
    env: Any,
    policy: WheelbipeOnnxPolicy | WheelbipeTorchScriptPolicy,
    *,
    steps: int,
    command: Sequence[float] | None = None,
    height: float | None = None,
    step_callback: Callable[[Any, int], None] | None = None,
) -> dict[str, float]:
    """Run a policy and optionally publish each completed state to tooling.

    ``step_callback`` is a read-only observation boundary for visualization
    and trace owners.  It runs after command overrides are restored on the
    post-step state and receives a one-based completed-step count.
    """

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    command_arr = None if command is None else np.asarray(command, dtype=np.float32).reshape(3)
    height_value = None if height is None else float(height)
    # ``np.clip`` intentionally bounds very large *finite* deployment values
    # to the ROS policy-input range below.  Non-finite values must be rejected
    # before they are copied into ``info``: clipping ``+/-inf`` would otherwise
    # produce a finite policy input while rewards/diagnostics still observe the
    # original non-finite command.
    if command_arr is not None and not np.all(np.isfinite(command_arr)):
        raise ValueError("Wheelbipe command override must contain only finite values")
    if height_value is not None and not np.isfinite(height_value):
        raise ValueError("Wheelbipe height override must be finite")
    state = env.init_state()

    def apply_overrides(current_state: Any) -> None:
        """Keep reset-visible info and the policy vector on the same command."""

        if command_arr is not None:
            commands = current_state.info.get("commands")
            if not isinstance(commands, np.ndarray) or commands.ndim != 2 or commands.shape[1] != 3:
                raise ValueError("Wheelbipe env must expose batched info['commands'] with width 3")
            commands[:] = command_arr
            current_state.info["commands"] = commands
        if height_value is not None:
            heights = current_state.info.get("height_commands")
            if not isinstance(heights, np.ndarray) or heights.ndim != 1:
                raise ValueError("Wheelbipe env must expose batched info['height_commands']")
            heights[:] = height_value
            current_state.info["height_commands"] = heights

        # ``init_state`` and autoreset build observations before the caller can
        # modify ``info``.  The normal deployment layout puts command and
        # height at the first four slots, so update those slots explicitly.
        policy_obs = current_state.obs.get("obs")
        if policy_obs is None or policy_obs.ndim != 2 or policy_obs.shape[1] != POLICY_OBS_DIM:
            raise ValueError("Wheelbipe env must expose obs['obs'] with shape (N, 35) for sim2sim")
        clipped_command = (
            None if command_arr is None else np.clip(command_arr, -POLICY_OBS_CLIP, POLICY_OBS_CLIP)
        )
        if command_arr is not None:
            # Keep caller overrides on the same post-scale clamp boundary as
            # ``build_wheelbipe_policy_observation``.  Without this guard a
            # large CLI ``--command`` could bypass the owner contract by
            # writing directly into the already-built observation.
            assert clipped_command is not None
            policy_obs[:, :3] = clipped_command
        if height_value is not None:
            policy_obs[:, 3] = np.clip(
                height_value * 5.0,
                -POLICY_OBS_CLIP,
                POLICY_OBS_CLIP,
            )

        # Keep the privileged stream internally coherent for callers that log
        # or inspect it during a rollout.  The ONNX policy only consumes the
        # public ``obs`` stream; these offsets are part of this env's fixed
        # 78D owner contract.
        critic_obs = current_state.obs.get("critic")
        if critic_obs is not None and critic_obs.ndim == 2 and critic_obs.shape[1] == 78:
            if command_arr is not None:
                assert clipped_command is not None
                critic_obs[:, :3] = clipped_command
                critic_obs[:, 58:61] = command_arr
            if height_value is not None:
                critic_obs[:, 3] = np.clip(
                    height_value * 5.0,
                    -POLICY_OBS_CLIP,
                    POLICY_OBS_CLIP,
                )
                critic_obs[:, 54] = height_value

    apply_overrides(state)

    reward_sum = 0.0
    done_count = 0
    for completed_steps in range(1, int(steps) + 1):
        apply_overrides(state)
        actions = policy.predict(state.obs["obs"])
        if actions.ndim == 1:
            actions = actions[None, :]
        state = env.step(np.asarray(actions, dtype=np.float32))
        # NpEnv autoresets terminated rows inside ``step``.  Reapply fixed
        # deployment commands after that reset before the next inference.
        apply_overrides(state)
        reward_sum += float(np.mean(state.reward))
        done_count += int(np.count_nonzero(state.terminated | state.truncated))
        if step_callback is not None:
            step_callback(state, completed_steps)
    return {
        "steps": float(steps),
        "mean_reward": reward_sum / float(steps),
        "done_count": float(done_count),
    }


def _compact_rollout_overrides(
    command: Sequence[float] | None,
    height: float | None,
) -> tuple[np.ndarray | None, float | None]:
    command_arr = None if command is None else np.asarray(command, dtype=np.float32).reshape(3)
    height_value = None if height is None else float(height)
    if command_arr is not None and not np.all(np.isfinite(command_arr)):
        raise ValueError("Wheelbipe command override must contain only finite values")
    if height_value is not None and not np.isfinite(height_value):
        raise ValueError("Wheelbipe height override must be finite")
    return command_arr, height_value


def _apply_compact_rollout_overrides(
    current_state: Any,
    *,
    one_step_dim: int,
    command: np.ndarray | None,
    height: float | None,
) -> np.ndarray:
    """Apply deployment commands to one compact custom observation frame."""

    obs = np.asarray(current_state.obs.get("obs"), dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != one_step_dim:
        raise ValueError(
            "Wheelbipe custom env must expose compact obs with shape "
            f"(N, {one_step_dim}), got {obs.shape}"
        )
    if command is not None:
        commands = current_state.info.get("commands")
        if (
            not isinstance(commands, np.ndarray)
            or commands.ndim != 2
            or commands.shape != (obs.shape[0], 3)
        ):
            raise ValueError(
                "Wheelbipe custom env must expose batched info['commands'] with width 3"
            )
        commands[:] = command
        current_state.info["commands"] = commands
        obs[:, :3] = np.clip(command, -POLICY_OBS_CLIP, POLICY_OBS_CLIP)
    if height is not None:
        heights = current_state.info.get("height_commands")
        if (
            not isinstance(heights, np.ndarray)
            or heights.ndim != 1
            or heights.shape[0] != obs.shape[0]
        ):
            raise ValueError("Wheelbipe custom env must expose batched info['height_commands']")
        heights[:] = height
        current_state.info["height_commands"] = heights
        obs[:, 3] = np.clip(height * 5.0, -POLICY_OBS_CLIP, POLICY_OBS_CLIP)
    current_state.obs["obs"] = obs
    return obs


def run_wheelbipe_history_policy(
    env: Any,
    policy: WheelbipeHistoryOnnxPolicy,
    *,
    steps: int,
    command: Sequence[float] | None = None,
    height: float | None = None,
    history_reset_mode: str | None = None,
) -> dict[str, float]:
    """Roll out a compact history policy in a UniLab Wheelbipe owner.

    This helper is the custom-algorithm counterpart to
    :func:`run_wheelbipe_policy`.  It owns the history buffer at the sim2sim
    boundary, exactly as ``CustomOnPolicyRunner`` does during training.  It is
    intentionally independent of RSL-RL and therefore also works with a tiny
    fake env in boundary tests.

    ``WheelbipeHistoryOnnxPolicy`` validates the actor's one-step width and
    history length before this function is called.  The env must consequently
    be one of the compact (28D) task owners; passing the normal 35D owner is a
    contract error rather than an implicit truncation.
    """

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    metadata = getattr(policy, "metadata", None)
    declared_history_reset = None
    if isinstance(metadata, Mapping) and "history_reset_mode" in metadata:
        declared_history_reset = _canonical_custom_history_reset_mode(
            metadata["history_reset_mode"]
        )
    selected_history_reset = _canonical_custom_history_reset_mode(
        declared_history_reset
        if history_reset_mode is None and declared_history_reset is not None
        else ("repeat" if history_reset_mode is None else history_reset_mode)
    )
    if declared_history_reset is not None and selected_history_reset != declared_history_reset:
        raise ValueError(
            "Wheelbipe custom history reset override does not match policy metadata: "
            f"metadata={declared_history_reset!r}, override={selected_history_reset!r}"
        )
    contract = policy.contract
    command_arr, height_value = _compact_rollout_overrides(command, height)
    state = env.init_state()
    obs = _apply_compact_rollout_overrides(
        state,
        one_step_dim=contract.one_step_dim,
        command=command_arr,
        height=height_value,
    )

    def reset_history(frame: np.ndarray) -> np.ndarray:
        # ``np.tile`` concatenates complete frames and therefore matches the
        # runner's Torch ``repeat(1, history)`` semantics.  The source NP3O
        # environment instead clears its deque on reset and appends the
        # current frame, yielding zero frames plus the newest frame.  Keep
        # both modes explicit so legacy compact exports remain backward
        # compatible while exact source profiles reproduce their warm start.
        if selected_history_reset == "source_zero_current":
            result = np.zeros(
                (frame.shape[0], frame.shape[1] * contract.history_length),
                dtype=frame.dtype,
            )
            result[:, -contract.one_step_dim :] = frame
            return result
        return np.tile(frame, (1, contract.history_length))

    history = reset_history(obs)
    reward_sum = 0.0
    done_count = 0
    for _ in range(int(steps)):
        actions = np.asarray(policy.predict(history), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape != (obs.shape[0], NUM_POLICY_ACTIONS):
            raise ValueError(
                "Wheelbipe custom policy must return actions with shape "
                f"({obs.shape[0]}, {NUM_POLICY_ACTIONS}), got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Wheelbipe custom policy returned non-finite actions")
        state = env.step(actions)
        obs = _apply_compact_rollout_overrides(
            state,
            one_step_dim=contract.one_step_dim,
            command=command_arr,
            height=height_value,
        )
        history = np.concatenate((history[:, contract.one_step_dim :], obs), axis=1)
        # UniLab autoresets terminated rows inside ``step``.  Restart each
        # corresponding history at its reset frame so no previous-episode
        # latent context crosses the episode boundary.
        done = np.asarray(state.terminated | state.truncated, dtype=bool)
        if np.any(done):
            history[done] = reset_history(obs[done])
        reward_sum += float(np.mean(state.reward))
        done_count += int(np.count_nonzero(done))
    return {
        "steps": float(steps),
        "mean_reward": reward_sum / float(steps),
        "done_count": float(done_count),
    }


def run_wheelbipe_source_barlow_policy(
    env: Any,
    policy: WheelbipeSourceBarlowOnnxPolicy | WheelbipeSourceBarlowTorchScriptPolicy,
    *,
    steps: int,
    command: Sequence[float] | None = None,
    height: float | None = None,
) -> dict[str, float]:
    """Roll out a source two-input Barlow actor in a compact Wheelbipe env.

    The source actor consumes the current 28D frame and a rank-three
    ``(N,10,28)`` history.  Reuse the compact rollout's command/height
    boundary checks and history bookkeeping through a narrow adapter, while
    selecting ``source_zero`` reset semantics to match the upstream deque
    (zero-filled on reset, current frame appended as newest).
    """

    source_policy = policy

    class _FlattenedSourceAdapter:
        contract = WheelbipeHistoryPolicyContract(
            input_name="obs_history",
            output_name="actions",
            one_step_dim=int(source_policy.contract.one_step_dim),
            history_length=int(source_policy.contract.history_length),
            input_dim=int(source_policy.contract.flattened_history_dim),
            output_dim=int(source_policy.contract.output_dim),
        )

        def predict(self, flattened_history: np.ndarray) -> np.ndarray:
            history = np.asarray(flattened_history, dtype=np.float32)
            if history.ndim != 2 or history.shape[1] != self.contract.input_dim:
                raise ValueError(
                    "source Barlow rollout history adapter received unexpected shape "
                    f"{history.shape}; expected (N, {self.contract.input_dim})"
                )
            frames = history.reshape(
                history.shape[0],
                self.contract.history_length,
                self.contract.one_step_dim,
            )
            return np.asarray(
                source_policy.predict(frames[:, -1, :], frames),
                dtype=np.float32,
            )

    return run_wheelbipe_history_policy(
        env,
        _FlattenedSourceAdapter(),  # type: ignore[arg-type]
        steps=steps,
        command=command,
        height=height,
        history_reset_mode="source_zero_current",
    )


def run_wheelbipe_source_barlow_full_policy(
    env: Any,
    policy: WheelbipeSourceBarlowFullOnnxPolicy,
    *,
    steps: int,
    command: Sequence[float] | None = None,
    height: float | None = None,
) -> dict[str, float]:
    """Roll out the runner-exported one-input 312D NP3O policy.

    The stream ordering is the explicit UniLab compatibility repair used by
    ``CustomOnPolicyRunner``: current policy frame, four-value privileged
    critic tail, then ten policy-history frames oldest-to-newest.
    """

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    contract = policy.contract
    command_arr, height_value = _compact_rollout_overrides(command, height)
    state = env.init_state()

    def compact_parts(current_state: Any) -> tuple[np.ndarray, np.ndarray]:
        obs = _apply_compact_rollout_overrides(
            current_state,
            one_step_dim=contract.one_step_dim,
            command=command_arr,
            height=height_value,
        )
        critic = np.asarray(current_state.obs.get("critic"), dtype=np.float32)
        required_critic_dim = contract.one_step_dim + contract.privileged_latent_dim
        if critic.ndim != 2 or critic.shape != (obs.shape[0], required_critic_dim):
            raise ValueError(
                "Wheelbipe source Barlow full env must expose compact critic with shape "
                f"(N, {required_critic_dim}), got {critic.shape}"
            )
        latent = critic[:, contract.one_step_dim : required_critic_dim]
        if not np.all(np.isfinite(latent)):
            raise ValueError("Wheelbipe source Barlow privileged tail contains non-finite values")
        return obs, latent

    obs, latent = compact_parts(state)

    def reset_history(frame: np.ndarray) -> np.ndarray:
        history = np.zeros(
            (frame.shape[0], contract.history_feature_dim),
            dtype=frame.dtype,
        )
        history[:, -contract.one_step_dim :] = frame
        return history

    history = reset_history(obs)
    reward_sum = 0.0
    done_count = 0
    for _ in range(int(steps)):
        source_stream = np.concatenate((obs, latent, history), axis=1)
        if source_stream.shape != (obs.shape[0], contract.input_dim):
            raise ValueError(
                "Wheelbipe source Barlow full stream has unexpected shape "
                f"{source_stream.shape}; expected (N, {contract.input_dim})"
            )
        actions = np.asarray(policy.predict(source_stream), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape != (obs.shape[0], contract.output_dim):
            raise ValueError(
                "Wheelbipe source Barlow full policy must return actions with shape "
                f"({obs.shape[0]}, {contract.output_dim}), got {actions.shape}"
            )
        state = env.step(actions)
        obs, latent = compact_parts(state)
        history = np.concatenate((history[:, contract.one_step_dim :], obs), axis=1)
        done = np.asarray(state.terminated | state.truncated, dtype=bool)
        if np.any(done):
            history[done] = reset_history(obs[done])
        reward_sum += float(np.mean(state.reward))
        done_count += int(np.count_nonzero(done))
    return {
        "steps": float(steps),
        "mean_reward": reward_sum / float(steps),
        "done_count": float(done_count),
    }


__all__ = [
    "CUSTOM_WHEELBIPE_METADATA_SCHEMA",
    "DEFAULT_WHEELBIPE_POLICY",
    "WheelbipeOnnxPolicy",
    "WheelbipePolicyContract",
    "WheelbipeTorchScriptPolicy",
    "WheelbipeHistoryOnnxPolicy",
    "WheelbipeHistoryPolicyContract",
    "WheelbipeSourceBarlowOnnxPolicy",
    "WheelbipeSourceBarlowFullOnnxPolicy",
    "WheelbipeSourceBarlowFullPolicyContract",
    "WheelbipeSourceBarlowPolicyContract",
    "WheelbipeSourceBarlowTorchScriptPolicy",
    "export_actor_critic_barlow_twins_actor",
    "export_barlow_twins_actor_from_policy",
    "export_mlp_barlow_twins_actor_onnx",
    "export_mlp_barlow_twins_actor_torchscript",
    "inspect_wheelbipe_onnx",
    "is_wheelbipe_torchscript_archive",
    "inspect_wheelbipe_history_onnx",
    "inspect_wheelbipe_source_barlow_full_onnx",
    "inspect_wheelbipe_source_barlow_onnx",
    "custom_wheelbipe_metadata_path",
    "read_wheelbipe_custom_metadata",
    "wheelbipe_play_checkpoint_task_candidates",
    "run_wheelbipe_policy",
    "run_wheelbipe_history_policy",
    "run_wheelbipe_source_barlow_full_policy",
    "run_wheelbipe_source_barlow_policy",
]
