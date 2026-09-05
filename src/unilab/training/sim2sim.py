"""Cross-backend sim2sim contract snapshot and resolution."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict


class CrossBackendIncompatibleError(RuntimeError):
    """Raised when a target play config diverges from the source training contract."""


ALLOWLIST: list[str] = [
    "training.sim_backend",
    "env.scene",
    "training.play_steps",
    "env.domain_rand",
    "env.noise_config",
    "env.commands.vel_limit",
]

WARNING_LIST: list[str] = [
    "reward.scales",
    "reward.base_height_target",
    "reward.max_tilt_deg",
    "reward.min_base_height",
    "env.control_config.simulate_action_latency",
    "env.ctrl_dt",
]

DENYLIST: list[str] = [
    "algo.obs_groups",
    "env.control_config.action_scale",
    "algo.policy.actor_hidden_dims",
    "algo.policy.critic_hidden_dims",
    "algo.empirical_normalization",
    "algo.obs_normalization",
    "env.sampling_mode",
]

SNAPSHOT_FIELDS: list[str] = DENYLIST + WARNING_LIST

ENV_STRUCTURAL_DENYLIST: list[str] = [path for path in DENYLIST if path.startswith("env.")]

# Custom WheelBipe policies have more architecture knobs than the normal PPO
# MLP.  A play config is often composed from the compact owner defaults while
# the checkpoint was trained through a source-sized profile (for example
# [512, 256, 128] instead of [256, 128, 64]).  Keep the list of fields that
# are safe to hydrate from the run sidecar in one place.  Training/reward and
# environment fields deliberately stay out of this list: they remain subject
# to the normal sim2sim deny/warning contract below.
_CUSTOM_POLICY_ARCHITECTURE_FIELDS: tuple[str, ...] = (
    "actor_hidden_dims",
    "critic_hidden_dims",
    "activation",
    "init_noise_std",
    "noise_std_type",
    "encoder_hidden_dims",
    "decoder_hidden_dims",
    "cenet_encoder_hidden_dims",
    "cenet_decoder_hidden_dims",
    "cenet_in_dim",
    "cenet_out_dim",
    "cenet_logvar_clip",
    "cenet_feature_clip",
    "logvar_clip",
    "feature_clip",
    "action_mean_clip",
    "adaboot_mode",
    "adaboot_min",
    "adaboot_max",
    "adaboot_temperature",
    "adaboot_bias",
    "architecture",
    "model_type",
)
_CUSTOM_ESTIMATOR_ARCHITECTURE_FIELDS: tuple[str, ...] = (
    "enc_hidden_dims",
    "tar_hidden_dims",
    "activation",
    "velocity_target_start",
    "target_obs_start",
    "num_prototype",
    "temperature",
    "learning_rate",
    "max_grad_norm",
)
_CUSTOM_SOURCE_Barlow_FIELDS: tuple[str, ...] = (
    "class_name",
    "num_prop",
    "num_scan",
    "num_state_est",
    "num_priv_latent",
    "num_hist",
    "scan_encoder_dims",
    "priv_encoder_dims",
    "hist_encoder",
    "fixed_std",
    "action_mean_clip",
    "teacher_act",
    "imi_flag",
    "latent_dim",
    "continue_from_last_std",
    "tanh_encoder_output",
    "num_costs",
)
_CUSTOM_ALGORITHM_ALIASES: dict[str, str] = {
    "him": "him",
    "him_ppo": "him",
    "ppo_him": "him",
    "dreamwaq": "dreamwaq",
    "dream_waq": "dreamwaq",
    "ppo_dreamwaq": "dreamwaq",
    "np3o": "np3o",
}
_CUSTOM_HISTORY_RESET_MODES = frozenset({"repeat", "source_zero_current"})


def _select(cfg: Any, path: str) -> Any:
    """Return the effective value at a dotted path (or ``None`` if absent)."""
    return OmegaConf.select(cfg, path)


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def extract_contract_snapshot(full_cfg: DictConfig) -> dict[str, Any]:
    """Extract the contract fields from a resolved training config keyed by dotted path."""
    cfg: Any = full_cfg if OmegaConf.is_config(full_cfg) else OmegaConf.create(full_cfg)
    snapshot: dict[str, Any] = {}
    for path in SNAPSHOT_FIELDS:
        value = _select(cfg, path)
        if value is None:
            continue
        snapshot[path] = _to_plain(value)
    return snapshot


def _normalize(value: Any) -> Any:
    """Canonicalize a value for order-insensitive, type-tolerant comparison."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, bool):  # must precede int: bool is a subclass of int
        return value
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)  # 0 == 0.0; YAML-int vs JSON-float parity
    return value


def _values_equal(a: Any, b: Any) -> bool:
    return bool(_normalize(a) == _normalize(b))


def _format_value(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True)


def _diff_line(path: str, source_value: Any, target_value: Any) -> str:
    return f"{path}: source={_format_value(source_value)} target={_format_value(target_value)}"


def _asymmetric_line(path: str, present_value: Any, *, source_present: bool) -> str:
    """Format a denial for an env-structural field set on exactly one side."""
    value = _format_value(present_value)
    if source_present:
        return (
            f"{path}: source={value} target=<absent> (target omits this field and "
            "falls back to the env default, which may differ; set it explicitly in the "
            "target task YAML to make the contract verifiable)"
        )
    return (
        f"{path}: source=<absent> target={value} (the trained run omitted this field "
        "and used the env default; set it explicitly so the contract can be verified)"
    )


def _read_snapshot(run_dir: Path) -> dict[str, Any] | None:
    """Read ``contract_snapshot`` from ``run_dir/run_config.json`` (``None`` if absent)."""
    path = run_dir / "run_config.json"
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    snapshot = parsed.get("contract_snapshot")
    if not isinstance(snapshot, dict):
        return None
    return snapshot


def _read_run_config(run_dir: Path) -> dict[str, Any] | None:
    """Read a complete run sidecar when it is available.

    ``contract_snapshot`` intentionally contains only the cross-backend
    fields.  Custom policy constructors also need source architecture metadata
    (CENet/estimator/Barlow dimensions), so playback may consult the additive
    ``config.algo`` payload written by :class:`ExperimentTracker`.  A missing
    or malformed sidecar is treated like an old run and leaves the caller's
    config untouched; the subsequent checkpoint loader still performs strict
    tensor validation.
    """

    path = run_dir / "run_config.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _canonical_custom_algorithm(value: object) -> str | None:
    """Return the canonical custom algorithm alias, or ``None`` for unknown values."""

    return _CUSTOM_ALGORITHM_ALIASES.get(str(value).strip().lower())


def _validate_custom_history_reset_mode(value: object, *, path: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in _CUSTOM_HISTORY_RESET_MODES:
        raise ValueError(
            f"{path} in checkpoint run_config must be one of "
            f"{', '.join(sorted(_CUSTOM_HISTORY_RESET_MODES))}"
        )
    return mode


def _validate_custom_hidden_dims(value: Any, *, path: str) -> list[int]:
    """Validate a hidden-dimension list from a JSON sidecar."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{path} in checkpoint run_config must be a non-empty list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{path} in checkpoint run_config must contain positive integers")
        try:
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{path} in checkpoint run_config must contain positive integers"
            ) from exc
        if integer <= 0 or integer != item:
            raise ValueError(f"{path} in checkpoint run_config must contain positive integers")
        result.append(integer)
    return result


def _validate_custom_integer(value: Any, *, path: str, minimum: int = 0) -> int:
    """Validate an integer contract field from a JSON sidecar."""

    if isinstance(value, bool):
        raise ValueError(f"{path} in checkpoint run_config must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path} in checkpoint run_config must be an integer") from exc
    if integer < minimum or integer != value:
        raise ValueError(f"{path} in checkpoint run_config must be >= {minimum}")
    return integer


def _custom_override_matches(path: str, explicit_paths: set[str]) -> bool:
    """Whether a user explicitly selected ``path`` or one of its parents."""

    return any(
        path == candidate or path.startswith(f"{candidate}.") for candidate in explicit_paths
    )


def hydrate_custom_checkpoint_config(
    source_run_dir: str | Path | None,
    target_cfg: DictConfig,
    *,
    enabled: bool = True,
    explicit_paths: set[str] | None = None,
) -> DictConfig:
    """Hydrate custom-policy architecture metadata from a checkpoint run.

    Source-sized custom profiles intentionally change network widths and, for
    NP3O, select the explicit ``source_barlow`` graph.  Canonical playback may
    start with compact defaults when no source profile is selected.  This
    helper copies *only* constructor and
    representation architecture fields from ``run_config.json`` before the
    sim2sim resolver and environment are materialized.  The denylist resolver
    is still called afterwards, and the custom runner performs its own strict
    state-dict checks.

    Algorithm/history/estimate/cost contracts are validated rather than
    changed: those fields alter the observation protocol and cannot be made
    safe by copying a JSON value.  A caller can opt out with ``enabled=False``
    (or by explicitly overriding an architecture path); in that case the
    normal resolver/load path fails closed on a mismatch.
    """

    if not enabled or source_run_dir is None:
        return target_cfg

    run_dir = Path(source_run_dir).expanduser()
    # Public callers may hand us the selected ``model_*.pt`` path instead of
    # the containing run directory.  The sidecar contract is always adjacent
    # to that checkpoint, so normalize the two forms at this owner boundary.
    if run_dir.is_file():
        run_dir = run_dir.parent
    payload = _read_run_config(run_dir)
    if payload is None:
        return target_cfg

    # Do not mutate Hydra's composed target in place.  Playback callers often
    # reuse the same config object for diagnostics or a subsequent backend;
    # architecture adoption is an effective, per-load view of that config.
    hydrated_cfg = deepcopy(target_cfg)

    raw_config = payload.get("config")
    source_config = raw_config if isinstance(raw_config, dict) else {}
    source_algo = source_config.get("algo")
    if not isinstance(source_algo, dict):
        source_algo = {}
    source_policy = source_algo.get("policy")
    if not isinstance(source_policy, dict):
        source_policy = {}
    source_estimator = source_algo.get("estimator")
    if not isinstance(source_estimator, dict):
        source_estimator = {}

    # Old/custom sidecars may contain only the cross-backend snapshot.  Use it
    # as a narrow fallback for MLP widths; richer constructor metadata remains
    # unavailable and therefore cannot be guessed.
    snapshot = payload.get("contract_snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    for path in ("algo.policy.actor_hidden_dims", "algo.policy.critic_hidden_dims"):
        if path.rsplit(".", 1)[-1] not in source_policy and path in snapshot:
            source_policy[path.rsplit(".", 1)[-1]] = snapshot[path]

    target_algorithm_value = OmegaConf.select(hydrated_cfg, "algo.algorithm_name", default=None)
    source_algorithm_value = source_algo.get("algorithm_name")
    if source_algorithm_value is None:
        run_meta = payload.get("run")
        if isinstance(run_meta, dict):
            source_algorithm_value = run_meta.get("algo")
    source_algorithm = _canonical_custom_algorithm(source_algorithm_value)
    target_algorithm = _canonical_custom_algorithm(target_algorithm_value)
    if source_algorithm is not None and target_algorithm is not None:
        if source_algorithm != target_algorithm:
            raise ValueError(
                "Custom checkpoint algorithm does not match the selected playback route: "
                f"checkpoint={source_algorithm!r}, target={target_algorithm!r}"
            )
    elif source_algorithm_value is not None and target_algorithm_value is not None:
        raise ValueError(
            "Custom checkpoint algorithm metadata is unsupported or ambiguous: "
            f"checkpoint={source_algorithm_value!r}, target={target_algorithm_value!r}"
        )

    # Observation/history widths are an ABI, not merely a network preference.
    # Require equality whenever both sides declare them.  Missing target keys
    # are populated only to keep minimal direct-script configs usable.
    contract_fields = (
        ("num_actor_history", 1),
        ("num_estimate", 1),
        ("num_costs", 0),
    )
    for key, minimum in contract_fields:
        source_value = source_algo.get(key)
        if source_value is None:
            continue
        source_integer = _validate_custom_integer(source_value, path=f"algo.{key}", minimum=minimum)
        target_value = OmegaConf.select(hydrated_cfg, f"algo.{key}", default=None)
        if target_value is None:
            with open_dict(hydrated_cfg):
                OmegaConf.update(
                    hydrated_cfg,
                    f"algo.{key}",
                    source_integer,
                    merge=False,
                )
            continue
        target_integer = _validate_custom_integer(
            target_value, path=f"target algo.{key}", minimum=minimum
        )
        if target_integer != source_integer:
            raise CrossBackendIncompatibleError(
                "Custom checkpoint observation contract mismatch before playback: "
                f"algo.{key}: checkpoint={source_integer} target={target_integer}"
            )

    source_history_reset = source_algo.get("history_reset_mode")
    if source_history_reset is not None:
        source_history_reset = _validate_custom_history_reset_mode(
            source_history_reset,
            path="algo.history_reset_mode",
        )
        target_history_reset = OmegaConf.select(
            hydrated_cfg, "algo.history_reset_mode", default=None
        )
        if target_history_reset is None:
            with open_dict(hydrated_cfg):
                OmegaConf.update(
                    hydrated_cfg,
                    "algo.history_reset_mode",
                    source_history_reset,
                    merge=False,
                )
        else:
            target_history_reset = _validate_custom_history_reset_mode(
                target_history_reset,
                path="target algo.history_reset_mode",
            )
            if target_history_reset != source_history_reset:
                raise CrossBackendIncompatibleError(
                    "Custom checkpoint history reset contract mismatch before playback: "
                    "algo.history_reset_mode: "
                    f"checkpoint={source_history_reset!r} target={target_history_reset!r}"
                )

    explicit = set(explicit_paths or ())
    updates: dict[str, Any] = {}

    def add_update(path: str, value: Any, *, validate_dims: bool = False) -> None:
        if value is None:
            return
        if validate_dims:
            value = _validate_custom_hidden_dims(value, path=path)
        else:
            value = deepcopy(value)
        if _custom_override_matches(path, explicit):
            # Preserve an explicit user/profile selection.  The subsequent
            # resolver and state-dict loader provide the fail-closed diagnostic
            # if that selection does not fit this checkpoint.
            return
        current = OmegaConf.select(hydrated_cfg, path, default=None)
        if current is None or not _values_equal(current, value):
            updates[path] = value

    for key in _CUSTOM_POLICY_ARCHITECTURE_FIELDS:
        add_update(
            f"algo.policy.{key}",
            source_policy.get(key),
            validate_dims=key in {"actor_hidden_dims", "critic_hidden_dims"},
        )
    for key in _CUSTOM_ESTIMATOR_ARCHITECTURE_FIELDS:
        add_update(
            f"algo.estimator.{key}", source_estimator.get(key), validate_dims="hidden_dims" in key
        )

    source_barlow = source_policy.get("source_barlow")
    # The vendored upstream NP3O YAML keeps the Barlow constructor flat
    # (``class_name``, ``scan_encoder_dims``, ...), while UniLab's owner nests
    # those fields under ``policy.source_barlow``.  Recognize that public
    # source spelling when a sidecar came from a flat config, but require a
    # source-specific marker so a compact policy merely carrying a historical
    # class-name alias is not reclassified accidentally.
    if source_barlow is None:
        class_name = str(source_policy.get("class_name", "")).split(":")[-1].split(".")[-1]
        source_class_names = {
            "ActorCriticBarlowTwins",
            "SourceBarlowTwinsActorCritic",
            "ActorCriticBarlowTwinsSource",
        }
        flat_markers = {
            "scan_encoder_dims",
            "priv_encoder_dims",
            "num_prop",
            "num_scan",
            "num_state_est",
            "num_priv_latent",
            "num_hist",
            "teacher_act",
            "imi_flag",
            "tanh_encoder_output",
            "continue_from_last_std",
        }
        if class_name in source_class_names and flat_markers.intersection(source_policy):
            source_barlow = {
                key: deepcopy(source_policy[key])
                for key in _CUSTOM_SOURCE_Barlow_FIELDS
                if key in source_policy
            }

    # ``latent_dim`` is a top-level custom policy constructor argument in the
    # UniLab owner.  Flat source dictionaries occasionally carry it under the
    # policy (or nested Barlow) map, so retain those spellings as fallbacks.
    source_latent_dim = source_algo.get("latent_dim")
    if source_latent_dim is None:
        source_latent_dim = source_policy.get("latent_dim")
    if source_latent_dim is None and isinstance(source_barlow, dict):
        source_latent_dim = source_barlow.get("latent_dim")
    add_update("algo.latent_dim", source_latent_dim)

    source_architecture = source_algo.get("policy_architecture")
    if source_architecture is None:
        source_architecture = source_policy.get("architecture", source_policy.get("model_type"))
    if source_architecture is None and source_barlow is not None:
        source_architecture = "source_barlow"
    if source_architecture is not None:
        add_update("algo.policy_architecture", source_architecture)
        add_update("algo.policy.architecture", source_architecture)

    if source_barlow is not None:
        if not isinstance(source_barlow, dict):
            raise ValueError("algo.policy.source_barlow in checkpoint run_config must be a mapping")
        # Copy the nested source owner as one validated mapping.  Constructor
        # code performs the definitive key/shape validation; retaining the
        # complete map here is necessary for source NP3O's scan/latent flags.
        current_source_barlow = OmegaConf.select(
            hydrated_cfg, "algo.policy.source_barlow", default=None
        )
        if isinstance(current_source_barlow, dict):
            merged_source_barlow = deepcopy(current_source_barlow)
            merged_source_barlow.update(deepcopy(source_barlow))
        else:
            merged_source_barlow = source_barlow
        add_update("algo.policy.source_barlow", merged_source_barlow)
    elif str(source_architecture).strip().lower().replace("-", "_") == "source_barlow":
        # A sidecar that advertises source_barlow without its nested metadata
        # is not enough to safely construct the graph.  Leave the target as-is
        # so the runner emits its explicit missing/shape diagnostic.
        print(
            f"[checkpoint] {run_dir}/run_config.json advertises source_barlow but omits "
            "policy.source_barlow metadata; preserving target constructor"
        )

    if updates:
        with open_dict(hydrated_cfg):
            for path, value in updates.items():
                OmegaConf.update(hydrated_cfg, path, value, merge=False)
        print(
            "[checkpoint] adopted custom policy architecture from "
            f"{run_dir}: {', '.join(sorted(updates))}"
        )
    return hydrated_cfg


def resolve_sim2sim_config(
    source_run_dir: str | Path | None,
    target_cfg: DictConfig,
    *,
    algo_name: str | None = None,
    strict: bool = True,
) -> DictConfig | None:
    """Validate a target play config against the source training contract.

    Returns ``None`` if ``source_run_dir`` is ``None``; otherwise returns ``target_cfg``
    unchanged (never mutated). Raises :class:`CrossBackendIncompatibleError` under
    ``strict`` when any DENYLIST field differs, including asymmetric presence for
    :data:`ENV_STRUCTURAL_DENYLIST` paths.
    """
    if source_run_dir is None:
        print("[sim2sim] no source run dir; skipping cross-backend contract check")
        return None

    run_dir = Path(source_run_dir)
    snapshot = _read_snapshot(run_dir)
    if snapshot is None:
        print(
            f"[sim2sim] {run_dir}/run_config.json has no contract_snapshot "
            "(old run); skipping cross-backend enforcement"
        )
        return target_cfg

    denials: list[str] = []
    for path, source_value in snapshot.items():
        target_value = _select(target_cfg, path)
        if target_value is None:
            if path in ENV_STRUCTURAL_DENYLIST:
                denials.append(_asymmetric_line(path, source_value, source_present=True))
            continue
        if _values_equal(source_value, target_value):
            continue
        line = _diff_line(path, source_value, target_value)
        if path in DENYLIST:
            denials.append(line)
        else:
            print(f"[sim2sim] WARNING override {line}")

    for path in ENV_STRUCTURAL_DENYLIST:
        if path in snapshot:
            continue
        if _select(target_cfg, path) is not None:
            denials.append(_asymmetric_line(path, _select(target_cfg, path), source_present=False))

    if denials:
        message = (
            "Cross-backend sim2sim contract mismatch between the trained policy and "
            f"the target play config.\nSource run: {run_dir}\n"
            "The following policy-defining fields differ and must be reconciled in "
            "the target task YAML:\n  " + "\n  ".join(denials)
        )
        if strict:
            raise CrossBackendIncompatibleError(message)
        print(f"[sim2sim] WARNING (non-strict) {message}")

    return target_cfg


def _looks_like_dim_mismatch(message: str) -> bool:
    """Return whether ``load_state_dict`` reported a parameter size mismatch."""
    return "size mismatch for " in message.lower()


@contextmanager
def policy_load_dim_guard(
    *,
    env_obs_dim: int | None = None,
    env_action_dim: int | None = None,
    algo_name: str | None = None,
) -> Iterator[None]:
    """Re-raise a tensor shape mismatch during checkpoint load as a sim2sim diagnostic.

    Non-matching errors propagate unchanged, so a valid load is never blocked.
    """
    try:
        yield
    except (RuntimeError, ValueError) as exc:
        if not _looks_like_dim_mismatch(str(exc)):
            raise
        raise CrossBackendIncompatibleError(
            "Trained policy checkpoint does not fit this play environment -- likely a "
            "cross-backend sim2sim dimension mismatch.\n"
            f"  algo: {algo_name}\n"
            f"  env policy obs dim: {env_obs_dim}\n"
            f"  env action dim: {env_action_dim}\n"
            "The checkpoint's tensor shapes do not match the env's observation/action "
            "dimensions. Check the task's obs_groups_spec and action space across "
            "backends; see resolve_sim2sim_config and run "
            "`uv run scripts/audit_sim2sim_contracts.py`.\n"
            f"Original load error:\n{exc}"
        ) from exc


class Sim2SimConfigResolver:
    """Object facade over the module-level sim2sim contract API."""

    ALLOWLIST = ALLOWLIST
    WARNING_LIST = WARNING_LIST
    DENYLIST = DENYLIST
    ENV_STRUCTURAL_DENYLIST = ENV_STRUCTURAL_DENYLIST

    @staticmethod
    def extract_snapshot(full_cfg: DictConfig) -> dict[str, Any]:
        """See :func:`extract_contract_snapshot`."""
        return extract_contract_snapshot(full_cfg)

    @staticmethod
    def resolve(
        source_run_dir: str | Path | None,
        target_cfg: DictConfig,
        *,
        algo_name: str | None = None,
        strict: bool = True,
    ) -> DictConfig | None:
        """See :func:`resolve_sim2sim_config`."""
        return resolve_sim2sim_config(
            source_run_dir, target_cfg, algo_name=algo_name, strict=strict
        )

    @staticmethod
    def load_dim_guard(
        *,
        env_obs_dim: int | None = None,
        env_action_dim: int | None = None,
        algo_name: str | None = None,
    ):
        """See :func:`policy_load_dim_guard`."""
        return policy_load_dim_guard(
            env_obs_dim=env_obs_dim, env_action_dim=env_action_dim, algo_name=algo_name
        )
