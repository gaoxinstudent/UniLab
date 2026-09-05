"""Strict checkpoint adapters for source RSL-RL MLP wrappers."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


def load_source_mlp_compatible_state_dict(
    module: nn.Module,
    state: Mapping[str, Any],
    *,
    mlp_roots: Sequence[str],
    label: str,
) -> bool:
    """Strict-load a native state dict or one complete source ``MLP.model`` graph.

    The pinned WheelBipe HIM and DreamWaQ policies wrap each MLP in a module
    named ``model``.  UniLab uses an equivalent :class:`~torch.nn.Sequential`
    directly, so source keys such as ``actor.model.0.weight`` correspond to
    ``actor.0.weight``.  Conversion is intentionally all-or-nothing: every
    parameter below every declared root must use the source spelling, and the
    converted key set must equal the target module's complete state dict.
    """

    if not isinstance(state, Mapping):
        raise ValueError(f"{label} checkpoint state_dict must be a mapping")
    source_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{label} checkpoint state_dict must map string keys to tensors")
        source_state[key] = value

    expected_state = module.state_dict()
    expected_keys = set(expected_state)
    roots = tuple(str(root).strip(".") for root in mlp_roots)
    if not roots or any(not root for root in roots) or len(set(roots)) != len(roots):
        raise ValueError(f"{label} MLP roots must be unique non-empty names")

    # Native UniLab artifacts remain a normal strict load, including shape
    # validation.  Do not pass a malformed native graph through the aliasing
    # branch merely because one unrelated key contains ``.model.``.
    if set(source_state) == expected_keys:
        _strict_load(module, source_state, label=label)
        return False

    expected_by_root: dict[str, set[str]] = {
        root: {key for key in expected_keys if key.startswith(f"{root}.")} for root in roots
    }
    if any(not keys for keys in expected_by_root.values()):
        missing_roots = sorted(root for root, keys in expected_by_root.items() if not keys)
        raise ValueError(f"{label} target has no parameters under MLP root(s) {missing_roots!r}")

    allowed_source_keys: dict[str, str] = {}
    for root, target_keys in expected_by_root.items():
        for target_key in target_keys:
            suffix = target_key[len(root) + 1 :]
            allowed_source_keys[f"{root}.model.{suffix}"] = target_key

    source_model_keys = {key for key in source_state if ".model." in key}
    unknown_model_keys = source_model_keys - set(allowed_source_keys)
    if unknown_model_keys:
        raise ValueError(
            f"{label} checkpoint contains unknown source MLP key(s): {sorted(unknown_model_keys)!r}"
        )
    if not source_model_keys:
        _strict_load(module, source_state, label=label)
        return False

    for root, target_keys in expected_by_root.items():
        required_source = {
            f"{root}.model.{target_key[len(root) + 1 :]}" for target_key in target_keys
        }
        present_source = set(source_state) & required_source
        present_native = set(source_state) & target_keys
        if present_native or present_source != required_source:
            missing = sorted(required_source - present_source)
            raise ValueError(
                f"{label} checkpoint must contain one complete source MLP graph under "
                f"{root!r}; native/source keys cannot be mixed"
                + (f"; missing={missing!r}" if missing else "")
            )

    converted = {allowed_source_keys.get(key, key): value for key, value in source_state.items()}
    converted_keys = set(converted)
    if converted_keys != expected_keys:
        missing = sorted(expected_keys - converted_keys)
        unexpected = sorted(converted_keys - expected_keys)
        raise ValueError(
            f"{label} source MLP checkpoint does not match the complete target graph; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    _strict_load(module, converted, label=label)
    return True


def dreamwaq_source_parameter_names(
    module: nn.Module, *, cenet_only: bool = False
) -> tuple[str, ...]:
    """Return the pinned source DreamWaQ Adam parameter order by canonical name."""

    named_parameters = tuple(module.named_parameters())
    available = {name for name, _parameter in named_parameters}
    ordered_roots: tuple[str, ...]
    if cenet_only:
        ordered_roots = (
            "encoder",
            "encode_mean_latent",
            "encode_logvar_latent",
            "encode_mean_vel",
            "encode_logvar_vel",
            "decoder",
        )
    else:
        noise_names = tuple(name for name in ("std", "log_std") if name in available)
        if len(noise_names) != 1:
            raise ValueError("DreamWaQ source graph must contain exactly one std/log_std parameter")
        ordered_roots = (
            *noise_names,
            "actor",
            "critic",
            "encoder",
            "encode_mean_latent",
            "encode_logvar_latent",
            "encode_mean_vel",
            "encode_logvar_vel",
            "decoder",
        )

    result: list[str] = []
    for root in ordered_roots:
        if root in {"std", "log_std"}:
            matches = [name for name, _parameter in named_parameters if name == root]
        else:
            matches = [name for name, _parameter in named_parameters if name.startswith(f"{root}.")]
        if not matches:
            raise ValueError(f"DreamWaQ source parameter root {root!r} is missing")
        result.extend(matches)

    if len(result) != len(set(result)):
        raise ValueError("DreamWaQ source parameter order contains duplicate names")
    if not cenet_only and set(result) != available:
        raise ValueError(
            "DreamWaQ source parameter schema does not cover the complete policy; "
            f"missing={sorted(available - set(result))!r}, "
            f"unexpected={sorted(set(result) - available)!r}"
        )
    return tuple(result)


def remap_source_adam_state_dict(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_optimizer_state: Mapping[str, Any],
    *,
    source_parameter_names: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Re-key one pinned source Adam state by explicit parameter name order.

    PyTorch optimizer checkpoints contain positional parameter identifiers,
    not names.  DreamWaQ registers its source modules in a different order
    from UniLab, so loading the raw optimizer payload would attach moments to
    the wrong tensors.  This adapter validates the exact one-group Adam schema
    and re-keys slots through the supplied source name order and the live
    optimizer's parameter identities.
    """

    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError(f"{label} source optimizer adapter requires torch.optim.Adam")
    if not isinstance(source_optimizer_state, Mapping):
        raise ValueError(f"{label} source optimizer state must be a mapping")
    if set(source_optimizer_state) != {"state", "param_groups"}:
        raise ValueError(f"{label} source optimizer state must contain only state and param_groups")
    source_slots = source_optimizer_state["state"]
    source_groups = source_optimizer_state["param_groups"]
    if not isinstance(source_slots, Mapping):
        raise ValueError(f"{label} source optimizer state slots must be a mapping")
    if not isinstance(source_groups, (list, tuple)) or len(source_groups) != 1:
        raise ValueError(f"{label} source optimizer must contain exactly one parameter group")

    target_snapshot = optimizer.state_dict()
    target_groups = target_snapshot.get("param_groups")
    if not isinstance(target_groups, list) or len(target_groups) != 1:
        raise ValueError(f"{label} target optimizer must contain exactly one parameter group")
    source_group = source_groups[0]
    target_group = target_groups[0]
    if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
        raise ValueError(f"{label} optimizer parameter group must be a mapping")
    if set(source_group) != set(target_group):
        raise ValueError(
            f"{label} source optimizer group schema mismatch: "
            f"source={sorted(source_group)!r}, target={sorted(target_group)!r}"
        )
    source_ids = source_group.get("params")
    target_ids = target_group.get("params")
    if not isinstance(source_ids, (list, tuple)) or not isinstance(target_ids, list):
        raise ValueError(f"{label} optimizer params field must be a sequence")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in source_ids):
        raise ValueError(f"{label} source optimizer parameter ids must be integers")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{label} source optimizer parameter ids must be unique")

    source_names = tuple(str(name) for name in source_parameter_names)
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"{label} source optimizer parameter names must be unique")
    if len(source_ids) != len(source_names):
        raise ValueError(
            f"{label} source optimizer parameter count mismatch: "
            f"ids={len(source_ids)}, names={len(source_names)}"
        )

    module_names_by_id = {id(parameter): name for name, parameter in module.named_parameters()}
    target_named_parameters: list[tuple[str, nn.Parameter]] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = module_names_by_id.get(id(parameter))
            if name is None:
                raise ValueError(f"{label} optimizer contains a parameter not owned by the policy")
            target_named_parameters.append((name, parameter))
    target_names = tuple(name for name, _parameter in target_named_parameters)
    if len(target_names) != len(set(target_names)):
        raise ValueError(f"{label} target optimizer contains duplicate parameters")
    if len(target_ids) != len(target_names) or set(target_names) != set(source_names):
        raise ValueError(
            f"{label} source/target optimizer parameter schema mismatch; "
            f"source_only={sorted(set(source_names) - set(target_names))!r}, "
            f"target_only={sorted(set(target_names) - set(source_names))!r}"
        )

    source_name_by_id = dict(zip(source_ids, source_names))
    target_id_by_name = dict(zip(target_names, target_ids))
    target_parameter_by_name = dict(target_named_parameters)
    unknown_slot_ids = set(source_slots) - set(source_ids)
    if unknown_slot_ids:
        raise ValueError(
            f"{label} source optimizer has state for unknown parameter id(s) "
            f"{sorted(unknown_slot_ids)!r}"
        )

    amsgrad = source_group.get("amsgrad")
    if not isinstance(amsgrad, bool):
        raise ValueError(f"{label} source optimizer amsgrad flag must be boolean")
    required_slot_keys = {"step", "exp_avg", "exp_avg_sq"}
    if amsgrad:
        required_slot_keys.add("max_exp_avg_sq")
    converted_slots: dict[int, Any] = {}
    for source_id, raw_slot in source_slots.items():
        if not isinstance(source_id, int) or isinstance(source_id, bool):
            raise ValueError(f"{label} source optimizer state ids must be integers")
        if not isinstance(raw_slot, Mapping) or set(raw_slot) != required_slot_keys:
            raise ValueError(
                f"{label} source Adam slot keys mismatch for parameter id {source_id}: "
                f"expected={sorted(required_slot_keys)!r}, "
                f"got={sorted(raw_slot) if isinstance(raw_slot, Mapping) else type(raw_slot).__name__!r}"
            )
        name = source_name_by_id[source_id]
        parameter = target_parameter_by_name[name]
        step = raw_slot["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not bool(torch.all(torch.isfinite(step))):
                raise ValueError(f"{label} source Adam step for {name!r} must be finite scalar")
        elif (
            isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step)
        ):
            raise ValueError(f"{label} source Adam step for {name!r} must be finite scalar")
        for slot_name in required_slot_keys - {"step"}:
            value = raw_slot[slot_name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{label} source Adam {slot_name} shape mismatch for {name!r}: "
                    f"expected={tuple(parameter.shape)}, "
                    f"got={getattr(value, 'shape', None)}"
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{label} source Adam {slot_name} for {name!r} must be finite")
        converted_slots[target_id_by_name[name]] = copy.deepcopy(dict(raw_slot))

    converted_group = copy.deepcopy(dict(source_group))
    converted_group["params"] = list(target_ids)
    return {"state": converted_slots, "param_groups": [converted_group]}


def _strict_load(module: nn.Module, state: Mapping[str, torch.Tensor], *, label: str) -> None:
    try:
        module.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"{label} checkpoint tensors do not match the selected graph: {exc}"
        ) from exc


__all__ = [
    "dreamwaq_source_parameter_names",
    "load_source_mlp_compatible_state_dict",
    "remap_source_adam_state_dict",
]
