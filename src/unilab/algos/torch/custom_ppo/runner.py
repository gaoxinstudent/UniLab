"""Lifecycle runner for the migrated WheelBipe custom PPO algorithms."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Mapping
from typing import Any, Callable

import torch

from unilab.algos.torch.custom_ppo.algorithm import NP3O, DreamWaQPPO
from unilab.algos.torch.custom_ppo.models import (
    DreamWaQActorCritic,
    NP3OActorCritic,
    SourceBarlowTwinsActorCritic,
)
from unilab.algos.torch.custom_ppo.source_barlow import load_source_state_dict
from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic
from unilab.algos.torch.him_ppo.algorithm import HIMPPO
from unilab.algos.torch.him_ppo.checkpoint import (
    dreamwaq_source_parameter_names,
    load_source_mlp_compatible_state_dict,
    remap_source_adam_state_dict,
)

# The migrated source profiles are deliberately finite.  Keeping these
# values in one owner-layer table prevents a Hydra override from silently
# selecting a normal/one-step actor while the task advertises a history
# algorithm (or from disabling NP3O's constraint channels).
CUSTOM_ALGORITHM_ALIASES: dict[str, str] = {
    "him": "him",
    "him_ppo": "him",
    "ppo_him": "him",
    "dreamwaq": "dreamwaq",
    "dream_waq": "dreamwaq",
    "ppo_dreamwaq": "dreamwaq",
    "np3o": "np3o",
}

CUSTOM_ALGORITHM_CONTRACTS: dict[str, dict[str, int]] = {
    "him": {"history_length": 5, "num_estimate": 4, "num_costs": 0},
    "dreamwaq": {"history_length": 5, "num_estimate": 4, "num_costs": 0},
    "np3o": {"history_length": 10, "num_estimate": 4, "num_costs": 5},
}

CUSTOM_POLICY_ARCHITECTURES = frozenset({"compact", "source_barlow"})
CUSTOM_HISTORY_RESET_MODES = frozenset({"repeat", "source_zero_current"})


def canonical_custom_history_reset_mode(value: object) -> str:
    """Normalize the runner-owned history reset contract.

    ``source_zero_current`` mirrors the pinned WheelBipe deque lifecycle: a
    reset clears every slot and then appends the current observation as the
    newest frame.  ``repeat`` is retained as the bounded/legacy UniLab default
    so unrelated custom configurations do not change behavior implicitly.
    """

    mode = str(value).strip().lower().replace("-", "_")
    if mode not in CUSTOM_HISTORY_RESET_MODES:
        supported = ", ".join(sorted(CUSTOM_HISTORY_RESET_MODES))
        raise ValueError(
            f"Unsupported custom PPO history_reset_mode={value!r}; expected one of {supported}"
        )
    return mode


def _policy_architecture(train_cfg: Mapping[str, Any]) -> str:
    """Resolve the explicit custom policy graph selector."""

    policy_cfg = train_cfg.get("policy")
    raw = train_cfg.get("policy_architecture")
    if raw is None and isinstance(policy_cfg, Mapping):
        raw = policy_cfg.get("architecture", policy_cfg.get("model_type"))
        # The upstream source runner identifies this graph only by
        # ``class_name=ActorCriticBarlowTwins``.  Keep the historical compact
        # alias intact when no source dimensions are present, but recognize a
        # genuinely source-shaped flat config so an official YAML can be
        # passed through without a hand-written selector.
        if raw is None:
            class_name = str(policy_cfg.get("class_name", "")).split(":")[-1].split(".")[-1]
            source_markers = {
                "ActorCriticBarlowTwins",
                "SourceBarlowTwinsActorCritic",
                "ActorCriticBarlowTwinsSource",
            }
            source_fields = {
                "num_prop",
                "num_scan",
                "num_state_est",
                "num_priv_latent",
                "num_hist",
                "scan_encoder_dims",
                "priv_encoder_dims",
                "teacher_act",
                "imi_flag",
                "tanh_encoder_output",
                "continue_from_last_std",
            }
            if isinstance(policy_cfg.get("source_barlow"), Mapping) or (
                class_name in source_markers and any(field in policy_cfg for field in source_fields)
            ):
                raw = "source_barlow"
    if raw is None:
        raw = "compact"
    normalized = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "compact": "compact",
        "default": "compact",
        "source": "source_barlow",
        "source_barlow": "source_barlow",
        "barlow_source": "source_barlow",
    }
    architecture = aliases.get(normalized)
    if architecture is None:
        supported = ", ".join(sorted(CUSTOM_POLICY_ARCHITECTURES))
        raise ValueError(
            f"Unsupported custom PPO policy architecture={raw!r}; expected one of {supported}"
        )
    return architecture


def _normalize_update_metrics(update_result: Any) -> dict[str, float]:
    """Normalize custom algorithm update return values at the runner boundary.

    The compact HIM implementation intentionally preserves UniLab's historical
    four-value tuple, whereas source-style mapping storage and the DreamWaQ /
    NP3O owners return a named mapping.  Keeping this translation in one owner
    helper prevents lifecycle code from accidentally treating numeric tuple
    entries as mapping key/value pairs.
    """

    if isinstance(update_result, Mapping):
        return {str(key): float(value) for key, value in update_result.items()}
    if isinstance(update_result, tuple) and len(update_result) == 4:
        value_loss, surrogate_loss, estimation_loss, swap_loss = update_result
        return {
            "value_function": float(value_loss),
            "surrogate": float(surrogate_loss),
            "estimation": float(estimation_loss),
            "swap": float(swap_loss),
        }
    raise TypeError(
        "Custom PPO algorithm update() must return a metrics mapping or the HIM "
        f"four-tuple, got {type(update_result).__name__}"
    )


def canonical_custom_algorithm(value: object) -> str:
    """Normalize a custom algorithm name or raise a contract diagnostic."""

    name = str(value).strip().lower()
    canonical = CUSTOM_ALGORITHM_ALIASES.get(name)
    if canonical is None:
        supported = ", ".join(sorted(CUSTOM_ALGORITHM_CONTRACTS))
        raise ValueError(f"Unsupported custom PPO algorithm={value!r}; expected one of {supported}")
    return canonical


def _variant_mentions_algorithm(variant_name: object, algorithm: str) -> bool:
    """Return whether a variant label explicitly identifies ``algorithm``.

    Owner dataclasses use labels such as ``flat-him-v0`` while upstream task
    names commonly use ``WheelbipeV14FlatHIM``.  Normalize both forms at this
    metadata boundary; do not rely on a backend/runtime probe to infer the
    selected algorithm.
    """

    normalized = str(variant_name).strip().lower().replace("_", "-")
    tokens = [token for token in normalized.split("-") if token]
    return algorithm in tokens or algorithm in "".join(tokens)


def validate_custom_runner_contract(
    env: Any,
    train_cfg: dict[str, Any],
    *,
    algorithm_name: str | None = None,
    policy_architecture: str | None = None,
) -> dict[str, Any]:
    """Validate the task/algorithm/shape contract before model construction.

    ``WheelbipeVariantCfg`` carries the source variant metadata.  This helper
    intentionally validates it at the runner boundary as well: direct script
    users and Hydra overrides then receive an actionable error instead of a
    late tensor-shape or cost-critic failure.  Lightweight fake envs used by
    unit tests may omit ``cfg`` metadata; in that case shape-independent
    algorithm checks still apply and the env's dimensions remain authoritative.

    Returns a small immutable-by-convention metadata dictionary used in
    checkpoint payloads.  The checkpoint metadata is additive, so old custom
    checkpoints without it remain loadable subject to state-dict validation.
    """

    selected = canonical_custom_algorithm(
        algorithm_name
        if algorithm_name is not None
        else train_cfg.get("algorithm_name", train_cfg.get("custom_algorithm", "him"))
    )
    contract = CUSTOM_ALGORITHM_CONTRACTS[selected]
    architecture = (
        _policy_architecture(train_cfg)
        if policy_architecture is None
        else _policy_architecture({**train_cfg, "policy_architecture": policy_architecture})
    )
    history_reset_mode = canonical_custom_history_reset_mode(
        train_cfg.get("history_reset_mode", "repeat")
    )
    if architecture == "source_barlow" and selected != "np3o":
        raise ValueError("source_barlow policy architecture is only defined for the NP3O algorithm")
    if architecture == "source_barlow" and history_reset_mode != "source_zero_current":
        raise ValueError(
            "source_barlow requires algo.history_reset_mode=source_zero_current; "
            f"got {history_reset_mode!r}"
        )

    env_cfg = getattr(env, "cfg", None)
    env_algorithm = getattr(env_cfg, "custom_algorithm", None) if env_cfg is not None else None
    if env_algorithm is not None:
        env_canonical = canonical_custom_algorithm(env_algorithm)
        if env_canonical != selected:
            raise ValueError(
                "Custom PPO algorithm/task mismatch: "
                f"algorithm={selected!r}, env.cfg.custom_algorithm={env_algorithm!r}. "
                "Select the matching Wheelbipe custom task owner."
            )

        mode = str(getattr(env_cfg, "policy_observation_mode", "")).lower()
        if mode and mode not in {"compact", "source"}:
            raise ValueError(
                "Custom PPO requires a compact or explicit source policy observation owner; "
                f"env.cfg.policy_observation_mode={mode!r}"
            )

        variant_name = str(getattr(env_cfg, "variant_name", ""))
        # A metadata-bearing env must identify its algorithm in the variant
        # name too.  This catches accidental registration of a normal owner
        # under a custom task name while allowing ``*-play-*`` variants.
        if variant_name and not _variant_mentions_algorithm(variant_name, selected):
            raise ValueError(
                "Custom PPO variant metadata does not identify the selected algorithm: "
                f"variant_name={variant_name!r}, algorithm={selected!r}"
            )

        env_history = getattr(env_cfg, "num_actor_history", None)
        if env_history is not None and int(env_history) != contract["history_length"]:
            raise ValueError(
                "Custom PPO history contract mismatch: "
                f"algorithm={selected!r} expects {contract['history_length']}, "
                f"env.cfg.num_actor_history={env_history}"
            )
        env_estimate = getattr(env_cfg, "num_estimate", None)
        if env_estimate is not None and int(env_estimate) != contract["num_estimate"]:
            raise ValueError(
                "Custom PPO estimator contract mismatch: "
                f"algorithm={selected!r} expects num_estimate={contract['num_estimate']}, "
                f"env.cfg.num_estimate={env_estimate}"
            )

    history = int(train_cfg.get("num_actor_history", contract["history_length"]))
    if history != contract["history_length"]:
        raise ValueError(
            "Custom PPO history length is incompatible with the migrated source profile: "
            f"algorithm={selected!r} expects {contract['history_length']}, got {history}"
        )
    estimate = int(train_cfg.get("num_estimate", contract["num_estimate"]))
    if estimate != contract["num_estimate"]:
        raise ValueError(
            "Custom PPO estimator width is incompatible with the migrated source profile: "
            f"algorithm={selected!r} expects {contract['num_estimate']}, got {estimate}"
        )

    configured_costs = int(train_cfg.get("num_costs", contract["num_costs"]))
    if configured_costs != contract["num_costs"]:
        if selected == "np3o":
            raise ValueError(
                "NP3O requires exactly five constraint channels; "
                f"algo.num_costs={configured_costs} is invalid"
            )
        raise ValueError(
            f"{selected} does not use NP3O constraint channels; "
            f"algo.num_costs must be 0, got {configured_costs}"
        )

    env_costs = getattr(env_cfg, "num_costs", None) if env_cfg is not None else None
    if env_costs is not None and int(env_costs) != contract["num_costs"]:
        if selected == "np3o":
            raise ValueError(
                "NP3O requires exactly five environment constraint channels; "
                f"env.cfg.num_costs={env_costs} is invalid (an override to 0 is not supported)"
            )
        raise ValueError(f"{selected} expects env.cfg.num_costs=0, got {env_costs}")

    if selected == "np3o":
        algorithm_cfg = train_cfg.get("algorithm")
        if isinstance(algorithm_cfg, Mapping):
            # ``algo.algorithm.num_costs`` is not a source NP3O knob; reject
            # it explicitly instead of silently accepting an override that
            # would leave the policy/storage width unchanged.
            if "num_costs" in algorithm_cfg:
                nested_costs = int(algorithm_cfg["num_costs"])
                if nested_costs != contract["num_costs"]:
                    raise ValueError(
                        f"NP3O algorithm.num_costs must remain exactly five; got {nested_costs}"
                    )
            if "cost_limits" in algorithm_cfg:
                raw_limits = algorithm_cfg["cost_limits"]
                try:
                    limit_count = len(raw_limits)
                except TypeError as exc:
                    raise ValueError("NP3O algorithm.cost_limits must contain five values") from exc
                if limit_count != contract["num_costs"]:
                    raise ValueError(
                        "NP3O algorithm.cost_limits must contain exactly five values; "
                        f"got {limit_count}"
                    )

    actor_dim = int(getattr(env, "num_obs", 0))
    critic_dim = int(getattr(env, "num_privileged_obs", actor_dim) or actor_dim)
    action_dim = int(getattr(env, "num_actions", 0))
    if env_cfg is not None and getattr(env_cfg, "policy_observation_mode", None) == "compact":
        if actor_dim != 28:
            raise ValueError(
                "Custom Wheelbipe compact policy observation must be 28D, "
                f"got env.num_obs={actor_dim}"
            )
    if action_dim != 6:
        raise ValueError(f"Custom Wheelbipe policy action contract must be 6D, got {action_dim}")
    source_dims: dict[str, int] = {}
    source_cfg: Mapping[str, Any] = {}
    source_num_costs = contract["num_costs"]
    if architecture == "source_barlow":
        policy_cfg = train_cfg.get("policy")
        if isinstance(policy_cfg, Mapping) and isinstance(policy_cfg.get("source_barlow"), Mapping):
            source_cfg = policy_cfg["source_barlow"]
        # Accept both UniLab's nested source owner and the flat dictionary
        # emitted by the upstream ``rsl_rl_ppo_cfg.py``.  Nested values win so
        # an explicit owner override cannot be shadowed by a legacy flat key.
        merged_source_cfg = dict(source_cfg)
        if isinstance(policy_cfg, Mapping):
            for key in (
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
            ):
                if key in policy_cfg and key not in merged_source_cfg:
                    merged_source_cfg[key] = policy_cfg[key]
        source_cfg = merged_source_cfg
        # Source V14 uses policy=28, scan=0, estimated/privileged latent=4,
        # history=10.  Keep every dimension explicit in the returned contract
        # so a source checkpoint cannot be loaded into a compact stream merely
        # because both owners advertise ``num_actor_history=10``.
        source_dims = {
            "num_prop": int(source_cfg.get("num_prop", actor_dim)),
            "num_scan": int(source_cfg.get("num_scan", 0)),
            "num_state_est": int(source_cfg.get("num_state_est", estimate)),
            "num_priv_latent": int(source_cfg.get("num_priv_latent", estimate)),
            "num_hist": int(source_cfg.get("num_hist", history)),
        }
        if source_dims["num_prop"] != actor_dim:
            raise ValueError(
                "source_barlow num_prop must match the environment one-step actor width: "
                f"source={source_dims['num_prop']}, env={actor_dim}"
            )
        if source_dims["num_state_est"] != estimate:
            raise ValueError(
                "source_barlow num_state_est must match algo.num_estimate: "
                f"source={source_dims['num_state_est']}, algo={estimate}"
            )
        if source_dims["num_hist"] != history:
            raise ValueError(
                "source_barlow num_hist must match algo.num_actor_history: "
                f"source={source_dims['num_hist']}, algo={history}"
            )
        if any(value < 0 for key, value in source_dims.items() if key == "num_scan") or any(
            value <= 0 for key, value in source_dims.items() if key != "num_scan"
        ):
            raise ValueError(
                f"source_barlow dimensions must be positive (num_scan may be zero): {source_dims}"
            )
        source_input_dim = (
            source_dims["num_prop"]
            + source_dims["num_scan"]
            + source_dims["num_priv_latent"]
            + source_dims["num_hist"] * source_dims["num_prop"]
        )
        source_dims["input_dim"] = source_input_dim
        source_class_name = source_cfg.get("class_name")
        if source_class_name is not None:
            normalized_class_name = str(source_class_name).split(":")[-1].split(".")[-1]
            if normalized_class_name not in {
                "ActorCriticBarlowTwins",
                "SourceBarlowTwinsActorCritic",
                "ActorCriticBarlowTwinsSource",
            }:
                raise ValueError(
                    "source_barlow policy class_name must identify the source Barlow graph, "
                    f"got {source_class_name!r}"
                )
        source_num_costs = int(source_cfg.get("num_costs", contract["num_costs"]))
        if source_num_costs != contract["num_costs"]:
            raise ValueError(
                f"source_barlow policy num_costs must remain exactly five; got {source_num_costs}"
            )
    if selected == "np3o" and architecture == "compact" and critic_dim != 32:
        # The compact NP3O owner intentionally uses the 32D privileged frame
        # (policy-28 + body velocity-3 + observed height-1).  Upstream
        # Barlow/constraint experiments expose a separate 312D
        # ``on_constraint`` stream and require scan/teacher branches that this
        # bounded adapter does not implement.  Reject that shape here instead
        # of silently training a generic MLP against the wrong contract.
        raise ValueError(
            "Compact NP3O owner requires a 32D privileged critic; source Barlow "
            f"on_constraint/other critic width {critic_dim} is unsupported"
        )

    return {
        "algorithm": selected,
        "variant_name": str(getattr(env_cfg, "variant_name", "")) if env_cfg is not None else "",
        "actor_obs_dim": actor_dim,
        "critic_obs_dim": critic_dim,
        "history_length": history,
        "history_reset_mode": history_reset_mode,
        "num_estimate": estimate,
        "num_actions": action_dim,
        "num_costs": configured_costs,
        **(
            {
                "architecture": architecture,
                "actor_input_dim": source_dims["input_dim"],
                "critic_input_dim": source_dims["input_dim"],
                "source_num_prop": source_dims["num_prop"],
                "source_num_scan": source_dims["num_scan"],
                "source_num_state_est": source_dims["num_state_est"],
                "source_num_priv_latent": source_dims["num_priv_latent"],
                "source_num_hist": source_dims["num_hist"],
                "source_num_costs": source_num_costs,
                # The vendored source accepts these switches through
                # ``**kwargs`` but does not branch on them.  Persist their
                # values as contract metadata so a resumed run records the
                # exact source profile it was composed with.
                "source_continue_from_last_std": bool(
                    source_cfg.get("continue_from_last_std", True)
                ),
                "source_tanh_encoder_output": bool(source_cfg.get("tanh_encoder_output", False)),
            }
            if architecture == "source_barlow"
            else {}
        ),
    }


class CustomOnPolicyRunner:
    """One runner with explicit runtime selection for HIM, DreamWaQ and NP3O.

    The env remains behind UniLab's RSL-RL adapter.  Only this runtime owns
    history stacking and algorithm-specific representation/constraint losses.
    """

    def __init__(
        self, env: Any, train_cfg: dict[str, Any], log_dir: str | None = None, device: str = "cpu"
    ) -> None:
        self.env, self.cfg, self.device, self.log_dir = env, dict(train_cfg), device, log_dir
        self.algorithm_name = canonical_custom_algorithm(
            self.cfg.get("algorithm_name", self.cfg.get("custom_algorithm", "him"))
        )
        self.policy_architecture = _policy_architecture(self.cfg)

        self.num_steps_per_env = int(self.cfg.get("num_steps_per_env", 24))
        self.save_interval = int(self.cfg.get("save_interval", 100))
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        # Keep the algorithm update payload available to callers (and to
        # experiment trackers) instead of discarding source-style metrics such
        # as DreamWaQ's ``adaboot_*`` fields at the runner boundary.
        self.last_update_metrics: dict[str, float] = {}
        self._ep_returns = torch.zeros(env.num_envs, device=device)
        self._ep_lengths = torch.zeros(env.num_envs, device=device)

        actor_dim = int(env.num_obs)
        critic_dim = int(getattr(env, "num_privileged_obs", actor_dim) or actor_dim)
        self.contract = validate_custom_runner_contract(
            env,
            self.cfg,
            algorithm_name=self.algorithm_name,
            policy_architecture=self.policy_architecture,
        )
        self.history_size = int(self.contract["history_length"])
        self.history_reset_mode = str(self.contract["history_reset_mode"])
        policy_cfg = dict(self.cfg.get("policy") or {})
        # These selector/owner fields are metadata, not constructor kwargs.
        # Remove them before passing the compact policy config through so a
        # typo cannot be accepted by an open-ended ``**kwargs`` boundary.
        policy_cfg.pop("architecture", None)
        policy_cfg.pop("model_type", None)
        self.alg: Any
        if self.algorithm_name == "him":
            self.policy = HIMActorCritic(
                num_actor_obs=actor_dim * self.history_size,
                num_critic_obs=critic_dim,
                num_one_step_obs=actor_dim,
                num_actions=int(env.num_actions),
                num_estimate=int(self.contract["num_estimate"]),
                actor_hidden_dims=policy_cfg.get("actor_hidden_dims", [256, 128, 64]),
                critic_hidden_dims=policy_cfg.get("critic_hidden_dims", [256, 128, 64]),
                activation=str(policy_cfg.get("activation", "elu")),
                init_noise_std=float(policy_cfg.get("init_noise_std", 1.0)),
                estimator=dict(self.cfg.get("estimator") or {}),
            ).to(device)
            self.alg = HIMPPO(self.policy, device=device, **dict(self.cfg.get("algorithm") or {}))
            self.alg.init_storage(
                env.num_envs,
                self.num_steps_per_env,
                [actor_dim * self.history_size],
                [critic_dim],
                [env.num_actions],
            )
        elif self.algorithm_name == "dreamwaq":
            self.policy = DreamWaQActorCritic(
                actor_dim,
                critic_dim,
                int(env.num_actions),
                num_actor_history=self.history_size,
                num_estimate=int(self.contract["num_estimate"]),
                latent_dim=int(self.cfg.get("latent_dim", 16)),
                **policy_cfg,
            ).to(device)
            self.alg = DreamWaQPPO(
                self.policy, device=device, **dict(self.cfg.get("algorithm") or {})
            )
            self.alg.init_storage(
                env.num_envs,
                self.num_steps_per_env,
                actor_dim * self.history_size,
                critic_dim,
                env.num_actions,
                device,
            )
        else:
            if self.policy_architecture == "source_barlow":
                raw_source_cfg = policy_cfg.pop("source_barlow", {})
                if not isinstance(raw_source_cfg, Mapping):
                    raise ValueError("policy.source_barlow must be a mapping")
                source_cfg = dict(raw_source_cfg)
                # Official source configs keep these fields at the policy
                # level; a nested UniLab owner may carry them under
                # ``policy.source_barlow``.  Merge flat values only when the
                # nested mapping did not already provide one.
                for key in (
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
                ):
                    if key in policy_cfg and key not in source_cfg:
                        source_cfg[key] = policy_cfg[key]
                    # These are source-owner fields, not compact constructor
                    # kwargs.  Remove flat copies after merging so the strict
                    # leftover check below does not report a valid upstream
                    # config as unsupported.
                    policy_cfg.pop(key, None)
                # Dimensions were validated into ``self.contract`` above;
                # consume their metadata copies before checking constructor
                # knobs so a well-formed source profile is not reported as an
                # unknown option.
                for key in (
                    "num_prop",
                    "num_scan",
                    "num_state_est",
                    "num_priv_latent",
                    "num_hist",
                ):
                    source_cfg.pop(key, None)
                class_name = source_cfg.pop("class_name", None)
                if class_name is not None:
                    normalized_class_name = str(class_name).split(":")[-1].split(".")[-1]
                    if normalized_class_name not in {
                        "ActorCriticBarlowTwins",
                        "SourceBarlowTwinsActorCritic",
                        "ActorCriticBarlowTwinsSource",
                    }:
                        raise ValueError(
                            "source_barlow policy class_name must identify the source "
                            f"Barlow graph, got {class_name!r}"
                        )
                source_num_costs = source_cfg.pop("num_costs", self.contract["num_costs"])
                if int(source_num_costs) != int(self.contract["num_costs"]):
                    raise ValueError(
                        "source_barlow policy num_costs must remain exactly five; "
                        f"got {source_num_costs}"
                    )
                self.policy = SourceBarlowTwinsActorCritic(
                    int(self.contract["source_num_prop"]),
                    int(self.contract["source_num_scan"]),
                    int(self.contract["source_num_state_est"]),
                    int(self.contract["source_num_priv_latent"]),
                    int(self.contract["source_num_hist"]),
                    int(env.num_actions),
                    scan_encoder_dims=source_cfg.pop("scan_encoder_dims", [128, 64, 32]),
                    actor_hidden_dims=policy_cfg.pop("actor_hidden_dims", [512, 256, 128]),
                    critic_hidden_dims=policy_cfg.pop("critic_hidden_dims", [512, 256, 128]),
                    hist_encoder=bool(source_cfg.pop("hist_encoder", False)),
                    activation=str(policy_cfg.pop("activation", "elu")),
                    init_noise_std=float(policy_cfg.pop("init_noise_std", 1.0)),
                    fixed_std=bool(source_cfg.pop("fixed_std", False)),
                    action_mean_clip=source_cfg.pop("action_mean_clip", 20.0),
                    priv_encoder_dims=source_cfg.pop("priv_encoder_dims", []),
                    num_costs=int(self.contract["num_costs"]),
                    teacher_act=bool(source_cfg.pop("teacher_act", True)),
                    imi_flag=bool(source_cfg.pop("imi_flag", True)),
                    latent_dim=int(self.cfg.get("latent_dim", source_cfg.pop("latent_dim", 16))),
                    continue_from_last_std=bool(source_cfg.pop("continue_from_last_std", True)),
                    tanh_encoder_output=bool(source_cfg.pop("tanh_encoder_output", False)),
                ).to(device)
                if source_cfg:
                    raise ValueError(
                        f"Unsupported source_barlow policy config key(s): {sorted(source_cfg)!r}"
                    )
                if policy_cfg:
                    raise ValueError(
                        f"Unsupported source_barlow policy config key(s): {sorted(policy_cfg)!r}"
                    )
            else:
                self.policy = NP3OActorCritic(
                    actor_dim,
                    critic_dim,
                    int(env.num_actions),
                    num_actor_history=self.history_size,
                    latent_dim=int(self.cfg.get("latent_dim", 16)),
                    num_costs=int(self.contract["num_costs"]),
                    **policy_cfg,
                ).to(device)
            self.alg = NP3O(self.policy, device=device, **dict(self.cfg.get("algorithm") or {}))
            actor_storage_dim = int(
                self.contract.get("actor_input_dim", actor_dim * self.history_size)
            )
            critic_storage_dim = int(self.contract.get("critic_input_dim", critic_dim))
            self.alg.init_storage(
                env.num_envs,
                self.num_steps_per_env,
                actor_storage_dim,
                critic_storage_dim,
                env.num_actions,
                device,
                self.policy.num_costs,
            )

        self._history: torch.Tensor | None = None
        # Source NP3O receives a second, environment-owned history stream in
        # addition to the one-step actor observation.  UniLab's compact env
        # intentionally keeps the public ``obs``/``critic`` groups small, so
        # this adapter materializes the source stream at the runner boundary
        # from ``actor`` + the four-value privileged tail.  Keeping the ring
        # here mirrors the source environment's oldest-to-newest
        # ``policy_hist`` ordering without leaking backend state into the env.
        self._source_history: torch.Tensor | None = None

    def _history_obs(self, obs: torch.Tensor, *, reset: bool = False) -> torch.Tensor:
        if reset or self._history is None:
            self._history = self._fresh_history(obs)
        else:
            self._history = torch.cat((self._history[:, obs.shape[1] :], obs), dim=1)
        return self._history

    def _fresh_history(self, obs: torch.Tensor) -> torch.Tensor:
        """Build one reset history without changing the policy graph shape."""

        if self.history_reset_mode == "source_zero_current":
            history = torch.zeros(
                obs.shape[0],
                obs.shape[1] * self.history_size,
                dtype=obs.dtype,
                device=obs.device,
            )
            history[:, -obs.shape[1] :] = obs
            return history
        return obs.repeat(1, self.history_size)

    def _policy_device(self) -> torch.device:
        """Return the device on which the live policy parameters reside.

        ``get_inference_policy(device=...)`` is part of the public runner API
        and may move the policy after construction.  Using the constructor's
        ``self.device`` for deployment tensors would then create a subtle
        CPU/CUDA mismatch (especially for the source history adapter).  The
        module parameter is the authoritative runtime owner; the stored
        device remains a fallback for tiny test doubles with no parameters.
        """

        try:
            return next(self.policy.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device(self.device)

    def _episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        self._ep_returns += rewards
        self._ep_lengths += 1
        for idx in torch.nonzero(dones, as_tuple=False).flatten():
            self.rewbuffer.append(float(self._ep_returns[idx].item()))
            self.lenbuffer.append(float(self._ep_lengths[idx].item()))
        self._ep_returns[dones] = 0.0
        self._ep_lengths[dones] = 0.0

    def _initialize_random_episode_lengths(self) -> None:
        """Initialize the optional source-style random episode-length buffer."""

        episode_length_buf = getattr(self.env, "episode_length_buf", None)
        max_episode_length = getattr(self.env, "max_episode_length", None)
        if not isinstance(episode_length_buf, torch.Tensor) or max_episode_length is None:
            return
        try:
            high = int(max_episode_length)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Custom PPO env.max_episode_length must be an integer-compatible value"
            ) from exc
        if high > 0:
            with torch.no_grad():
                # Assign through the public VecEnv hook so UniLab's wrapper
                # can mirror the source random-start counter into NpEnv.
                self.env.episode_length_buf = torch.randint_like(episode_length_buf, high=high)

    def _source_on_constraint(
        self,
        value: Any,
        *,
        reset: bool = False,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Resolve the upstream NP3O ``on_constraint`` stream.

        A source adapter may expose the already-concatenated tensor, or the
        three public leaves (``policy``, ``priv_latent``, ``policy_hist``).
        The latter are concatenated in the exact order used by WheelBipe's
        ``env.py``.  No backend/private capability probing is involved; this
        is solely an observation payload contract.
        """

        expected = int(self.contract.get("actor_input_dim", 0))
        runtime_device = self._policy_device()
        if self._source_history is not None and self._source_history.device != runtime_device:
            # A caller may move the policy through ``get_inference_policy``
            # between episodes.  Drop stale history rather than attempting to
            # concatenate tensors owned by different devices.
            self._source_history = None

        def leaf(keys: tuple[str, ...]) -> torch.Tensor | None:
            if isinstance(value, Mapping):
                for key in keys:
                    candidate = value.get(key)
                    if isinstance(candidate, torch.Tensor):
                        return candidate
            for key in keys:
                try:
                    candidate = value[key]
                except (AttributeError, KeyError, TypeError, IndexError):
                    continue
                if isinstance(candidate, torch.Tensor):
                    return candidate
            return None

        direct = leaf(("on_constraint", "source_obs", "state"))
        if direct is None and isinstance(value, torch.Tensor):
            # A tensor is accepted only when it is already the complete source
            # stream.  A bare 28D actor frame cannot supply the privileged
            # latent segment and therefore fails closed below.
            direct = value
        if direct is None:
            policy = leaf(("policy", "actor", "obs"))
            latent = leaf(("priv_latent", "privileged_latent"))
            history = leaf(("policy_hist", "actor_history", "history"))
            if policy is not None and latent is not None and history is not None:
                # Do not validate only the concatenated width.  A malformed
                # payload such as ``27 + 4 + 281`` still sums to the source
                # 312D stream while shifting every downstream feature.  The
                # source ABI is a typed tuple: policy=(N,28), latent=(N,4),
                # history=(N,10,28) (or its explicitly flattened (N,280)
                # representation).  Validate each leaf before flattening or
                # concatenating so an adapter cannot silently train against a
                # semantically different stream.
                if policy.ndim != 2 or int(policy.shape[-1]) != int(self.contract["actor_obs_dim"]):
                    raise ValueError(
                        "source_barlow policy leaf must have shape "
                        f"(N, {self.contract['actor_obs_dim']}), got {tuple(policy.shape)}"
                    )
                if latent.ndim != 2 or int(latent.shape[-1]) != int(self.contract["num_estimate"]):
                    raise ValueError(
                        "source_barlow priv_latent leaf must have shape "
                        f"(N, {self.contract['num_estimate']}), got {tuple(latent.shape)}"
                    )
                if history.ndim == 3:
                    expected_frames = int(self.contract["history_length"])
                    expected_width = int(self.contract["actor_obs_dim"])
                    if tuple(int(dim) for dim in history.shape[1:]) != (
                        expected_frames,
                        expected_width,
                    ):
                        raise ValueError(
                            "source_barlow policy_hist leaf must have shape "
                            f"(N, {expected_frames}, {expected_width}), "
                            f"got {tuple(history.shape)}"
                        )
                    history = history.reshape(history.shape[0], -1)
                elif history.ndim == 2:
                    expected_history_width = int(self.contract["history_length"]) * int(
                        self.contract["actor_obs_dim"]
                    )
                    if int(history.shape[-1]) != expected_history_width:
                        raise ValueError(
                            "source_barlow policy_hist leaf must have width "
                            f"{expected_history_width}, got {tuple(history.shape)}"
                        )
                else:
                    raise ValueError(
                        "source_barlow policy_hist leaf must be rank-2 flattened or "
                        f"rank-3, got {tuple(history.shape)}"
                    )
                batch_sizes = (int(policy.shape[0]), int(latent.shape[0]), int(history.shape[0]))
                if len(set(batch_sizes)) != 1:
                    raise ValueError(
                        "source_barlow observation leaves must share a batch dimension, "
                        f"got policy/latent/history={batch_sizes!r}"
                    )
                direct = torch.cat((policy, latent, history), dim=-1)
            elif policy is not None:
                # The normal UniLab wrapper exposes actor + compact critic,
                # rather than source-specific leaves.  The compact critic is
                # explicitly ``[policy28, root_lin_vel_b3, obs_height1]``;
                # use only that four-value tail as source ``priv_latent``.
                critic = leaf(("critic", "privileged", "value"))
                if critic is None:
                    raise ValueError(
                        "source_barlow runner requires an on_constraint tensor, "
                        "policy + priv_latent + policy_hist leaves, or actor + critic"
                    )
                if policy.ndim != 2 or int(policy.shape[-1]) != int(self.contract["actor_obs_dim"]):
                    raise ValueError(
                        "source_barlow actor frame must have width "
                        f"{self.contract['actor_obs_dim']}, got shape={tuple(policy.shape)}"
                    )
                if critic.ndim != 2:
                    raise ValueError(
                        "source_barlow compact critic must be rank-2, "
                        f"got shape={tuple(critic.shape)}"
                    )
                latent_start = int(self.contract["actor_obs_dim"])
                latent_width = int(self.contract["num_estimate"])
                if int(critic.shape[-1]) < latent_start + latent_width:
                    raise ValueError(
                        "source_barlow actor + critic adapter requires the compact critic "
                        "tail [root_lin_vel_b3, obs_height1]; got "
                        f"critic shape={tuple(critic.shape)}"
                    )
                if int(critic.shape[0]) != int(policy.shape[0]):
                    raise ValueError(
                        "source_barlow actor and critic batch sizes must match: "
                        f"actor={tuple(policy.shape)}, critic={tuple(critic.shape)}"
                    )
                latent = critic[:, latent_start : latent_start + latent_width]
                policy = policy.to(runtime_device)
                latent = latent.to(runtime_device)
                frames = int(self.contract["history_length"])
                if reset or self._source_history is None:
                    # The source env initializes every history slot to zero,
                    # then appends the current frame before exposing
                    # ``policy_hist``.  Preserve that warm-start contract
                    # (nine zero frames + current for the V14 ten-frame
                    # stream), rather than repeating the current frame.
                    self._source_history = torch.zeros(
                        policy.shape[0],
                        frames,
                        policy.shape[1],
                        device=runtime_device,
                        dtype=policy.dtype,
                    )
                    self._source_history[:, -1, :] = policy.detach()
                else:
                    if self._source_history.shape[0] != policy.shape[0]:
                        # A play/eval caller may change the vectorized batch
                        # size.  Treat that as a fresh source episode rather
                        # than concatenating incompatible rows.
                        self._source_history = torch.zeros(
                            policy.shape[0],
                            frames,
                            policy.shape[1],
                            device=runtime_device,
                            dtype=policy.dtype,
                        )
                        self._source_history[:, -1, :] = policy.detach()
                    else:
                        self._source_history = torch.cat(
                            (self._source_history[:, 1:, :], policy.detach().unsqueeze(1)),
                            dim=1,
                        )
                if dones is not None:
                    done_mask = dones.to(device=runtime_device, dtype=torch.bool).reshape(-1)
                    if done_mask.shape[0] != policy.shape[0]:
                        raise ValueError(
                            "source_barlow done mask batch does not match actor frame: "
                            f"dones={tuple(done_mask.shape)}, actor={tuple(policy.shape)}"
                        )
                    if torch.any(done_mask):
                        self._source_history[done_mask] = 0.0
                        self._source_history[done_mask, -1, :] = policy[done_mask].detach()
                direct = torch.cat(
                    (policy, latent, self._source_history.reshape(policy.shape[0], -1)), dim=-1
                )
        if direct is None:
            raise ValueError(
                "source_barlow runner requires an on_constraint tensor or "
                "policy + priv_latent + policy_hist observation leaves"
            )
        if direct.ndim != 2 or int(direct.shape[-1]) != expected:
            raise ValueError(
                "source_barlow on_constraint width does not match its contract: "
                f"expected {expected}, got shape={tuple(direct.shape)}"
            )
        return direct.to(runtime_device)

    def build_source_observation(
        self,
        value: Any,
        *,
        reset: bool = False,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the 312D source NP3O stream from a wrapper observation.

        This public name is used by custom playback/deployment paths.  It is
        deliberately available for all runners but raises the same explicit
        contract error when a non-source architecture is selected.
        """

        if self.policy_architecture != "source_barlow":
            raise ValueError(
                "build_source_observation is only available for source_barlow custom runners"
            )
        return self._source_on_constraint(value, reset=reset, dones=dones)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True) -> None:
        # ``get_inference_policy`` switches the shared module to eval mode.
        # A caller may legitimately resume training on the same runner, so make
        # the mode transition explicit at the owner boundary just like the
        # source runners do.
        self.policy.train()
        obs_td, _ = self.env.reset()
        if self.policy_architecture == "source_barlow":
            obs = self._source_on_constraint(obs_td, reset=True)
            critic = obs
        else:
            obs = obs_td["actor"].to(self.device)
            critic = obs_td.get("critic", obs).to(self.device)
            self._history_obs(obs, reset=True)
        if init_at_random_ep_len:
            # Match the source HIMPPO runner's public initialization hook.
            # RSL-RL wrappers expose this buffer for episode accounting; fake
            # or non-RSL adapters may not, in which case there is no safe
            # state to mutate and the hook is intentionally a no-op.
            self._initialize_random_episode_lengths()
        for iteration in range(
            self.current_learning_iteration,
            self.current_learning_iteration + int(num_learning_iterations),
        ):
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    if self.policy_architecture == "source_barlow":
                        actions = self.alg.act(obs, critic)
                    else:
                        assert self._history is not None
                        actions = self.alg.act(self._history, critic)
                    next_td, rewards, dones, extras = self.env.step(actions)
                    if self.policy_architecture == "source_barlow":
                        next_obs = self._source_on_constraint(next_td, dones=dones)
                        next_critic = next_obs
                    else:
                        next_obs = next_td["actor"].to(self.device)
                        next_critic = next_td.get("critic", next_obs).to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self._episode_stats(rewards, dones)
                    if self.algorithm_name == "him":
                        self.alg.process_env_step(next_critic, rewards, dones, extras)
                    else:
                        self.alg.process_env_step(next_critic, rewards, dones, extras)
                    if self.policy_architecture != "source_barlow":
                        self._history_obs(next_obs)
                    obs = next_obs
                    # UniLab autoresets terminated rows in ``env.step``.  A
                    # history owned by the runner must restart those rows at
                    # the reset frame, otherwise latent estimators receive
                    # stale observations from the previous episode.
                    assert self._history is not None or self.policy_architecture == "source_barlow"
                    history = self._history
                    if (
                        self.policy_architecture != "source_barlow"
                        and history is not None
                        and torch.any(dones)
                    ):
                        history[dones] = self._fresh_history(next_obs[dones])
                    critic = next_critic
                self.alg.compute_returns(critic)
            if self.algorithm_name == "np3o" and hasattr(self.alg, "update_k_value"):
                # Keep the source NP3O channel-weight warm-up on the runner's
                # iteration boundary, alongside checkpoint/timestep updates.
                self.alg.update_k_value(iteration)
            # HIM-PPO keeps the historical compact four-tuple return shape,
            # while source-style mapping storage and the DreamWaQ/NP3O
            # owners expose named metrics.  Normalize at this runner boundary
            # instead of blindly calling ``dict(...)`` (which interprets a
            # four-tuple of floats as key/value pairs and fails during the
            # first real training iteration).
            self.last_update_metrics = _normalize_update_metrics(self.alg.update())
            self.current_learning_iteration = iteration + 1
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if self.log_dir and self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(
        self,
        path: str,
        infos: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> None:
        """Persist a custom checkpoint with UniLab and source-runner aliases.

        The optional ``infos``/``is_best`` arguments mirror the upstream
        ``OnPolicy*Runner.save`` API.  ``is_best`` is intentionally a logging
        hint only; the experiment tracker owns artifact promotion.  Additive
        state-dict aliases let an upstream WheelBipe checkpoint consumer read a
        native UniLab artifact, while ``load`` accepts the inverse aliases.
        """

        del is_best
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "algorithm": self.algorithm_name,
            "policy_state_dict": self.policy.state_dict(),
            "iteration": self.current_learning_iteration,
            "total_timesteps": self.tot_timesteps,
            "infos": infos,
            # Additive metadata makes algorithm/variant mismatches explicit at
            # load time while preserving compatibility with older checkpoints.
            "custom_contract": dict(self.contract),
        }
        # Upstream WheelBipe/rsl_rl runners call these fields
        # ``model_state_dict`` and ``iter``.  Keep the canonical UniLab keys
        # above so new consumers have one spelling, then expose aliases for
        # source tooling without requiring a conversion script.
        payload["model_state_dict"] = payload["policy_state_dict"]
        payload["iter"] = self.current_learning_iteration
        payload["actor_state_dict"] = payload["policy_state_dict"]
        payload["optimizer_state_dict"] = self.alg.optimizer.state_dict()
        if hasattr(self.alg, "cenet_optimizer"):
            payload["representation_optimizer_state_dict"] = self.alg.cenet_optimizer.state_dict()
            payload["vae_optimizer_state_dict"] = payload["representation_optimizer_state_dict"]
        representation_optimizer = getattr(self.alg, "representation_optimizer", None)
        if representation_optimizer is not None:
            payload["representation_optimizer_state_dict"] = representation_optimizer.state_dict()
        # HIM keeps a second Adam optimizer inside its estimator.  Persist it
        # separately from the PPO optimizer so resuming does not silently
        # reset the representation/adaptation moments.  Older checkpoints do
        # not contain this key and remain loadable below.
        estimator = getattr(self.policy, "estimator", None)
        if estimator is not None and hasattr(estimator, "optimizer"):
            payload["estimator_optimizer_state_dict"] = estimator.optimizer.state_dict()

        algorithm_state: dict[str, Any] = {}
        if hasattr(self.alg, "learning_rate"):
            algorithm_state["learning_rate"] = float(self.alg.learning_rate)
        if hasattr(self.alg, "k_value"):
            k_value = self.alg.k_value
            if isinstance(k_value, torch.Tensor):
                algorithm_state["k_value"] = k_value.detach().cpu()
        # DreamWaQ's AdaBoot coefficient is a function of a rolling episode
        # window and per-environment partial returns.  Persist that owner state
        # together with the scalar learning-rate/k-value metadata; otherwise a
        # resumed run silently starts with a different bootstrap schedule.
        algorithm_state_writer = getattr(self.alg, "algorithm_state_dict", None)
        if callable(algorithm_state_writer):
            algorithm_state["owner_state"] = algorithm_state_writer()
        if algorithm_state:
            payload["algorithm_state"] = algorithm_state
        torch.save(payload, path)

    def load(
        self,
        path: str,
        load_optimizer: bool | None = None,
        load_iteration: bool | None = None,
    ) -> dict[str, Any] | None:
        """Load a checkpoint and optionally follow source-runner flags.

        ``None`` (the default) keeps UniLab's continuation behavior and restores
        every optional state present in the artifact.  Passing explicit bools
        follows the upstream runner API: callers can load only policy weights,
        or choose optimizer and iteration restoration independently.
        """

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("custom PPO checkpoint must contain a mapping payload")
        checkpoint_algorithm = checkpoint.get("algorithm")
        if checkpoint_algorithm is not None:
            selected = canonical_custom_algorithm(checkpoint_algorithm)
            if selected != self.algorithm_name:
                raise ValueError(
                    "Custom PPO checkpoint algorithm does not match the selected runtime: "
                    f"checkpoint={selected!r}, runtime={self.algorithm_name!r}"
                )
        checkpoint_contract = checkpoint.get("custom_contract")
        if checkpoint_contract is not None and not isinstance(checkpoint_contract, dict):
            raise ValueError("Custom PPO checkpoint custom_contract must be a mapping")
        if isinstance(checkpoint_contract, dict):
            checkpoint_architecture = checkpoint_contract.get("architecture")
            if checkpoint_architecture is not None:
                selected_architecture = _policy_architecture(
                    {"policy_architecture": checkpoint_architecture}
                )
                if selected_architecture != self.policy_architecture:
                    raise ValueError(
                        "Custom PPO checkpoint policy architecture does not match the selected "
                        f"runtime: checkpoint={selected_architecture!r}, "
                        f"runtime={self.policy_architecture!r}"
                    )
            contract_algorithm = checkpoint_contract.get("algorithm")
            if contract_algorithm is not None:
                selected_contract_algorithm = canonical_custom_algorithm(contract_algorithm)
                if selected_contract_algorithm != self.algorithm_name:
                    raise ValueError(
                        "Custom PPO checkpoint contract algorithm does not match the selected "
                        f"runtime: checkpoint={selected_contract_algorithm!r}, "
                        f"runtime={self.algorithm_name!r}"
                    )
            checkpoint_variant = str(checkpoint_contract.get("variant_name", ""))
            runtime_variant = str(self.contract.get("variant_name", ""))
            if checkpoint_variant and runtime_variant:
                if not _variant_mentions_algorithm(checkpoint_variant, self.algorithm_name):
                    raise ValueError(
                        "Custom PPO checkpoint variant does not identify its algorithm: "
                        f"variant_name={checkpoint_variant!r}, algorithm={self.algorithm_name!r}"
                    )
                if not _variant_mentions_algorithm(runtime_variant, self.algorithm_name):
                    raise ValueError(
                        "Custom PPO runtime variant does not identify its algorithm: "
                        f"variant_name={runtime_variant!r}, algorithm={self.algorithm_name!r}"
                    )
            for key in (
                "actor_obs_dim",
                "critic_obs_dim",
                "history_length",
                "num_estimate",
                "num_actions",
                "num_costs",
            ):
                if key not in checkpoint_contract:
                    continue
                if int(checkpoint_contract[key]) != int(self.contract[key]):
                    raise ValueError(
                        "Custom PPO checkpoint contract mismatch for "
                        f"{key}: checkpoint={checkpoint_contract[key]!r}, "
                        f"runtime={self.contract[key]!r}"
                    )
            if "history_reset_mode" in checkpoint_contract:
                checkpoint_history_reset = canonical_custom_history_reset_mode(
                    checkpoint_contract["history_reset_mode"]
                )
                if checkpoint_history_reset != self.history_reset_mode:
                    raise ValueError(
                        "Custom PPO checkpoint contract mismatch for history_reset_mode: "
                        f"checkpoint={checkpoint_history_reset!r}, "
                        f"runtime={self.history_reset_mode!r}"
                    )
            # Source Barlow has additional dimensions and constructor
            # metadata beyond the compact contract.  Compare them when a
            # checkpoint carries the additive fields; legacy artifacts simply
            # skip this stricter audit and still undergo strict state loading.
            for key in (
                "source_num_prop",
                "source_num_scan",
                "source_num_state_est",
                "source_num_priv_latent",
                "source_num_hist",
                "source_num_costs",
                "source_continue_from_last_std",
                "source_tanh_encoder_output",
            ):
                if key not in checkpoint_contract or key not in self.contract:
                    continue
                if checkpoint_contract[key] != self.contract[key]:
                    raise ValueError(
                        "Custom PPO checkpoint source contract mismatch for "
                        f"{key}: checkpoint={checkpoint_contract[key]!r}, "
                        f"runtime={self.contract[key]!r}"
                    )
        # ``policy_state_dict`` is the native UniLab key and
        # ``actor_state_dict`` is used by older UniLab/RSL-RL adapters.  The
        # upstream WheelBipe runners save the same module under
        # ``model_state_dict``; accept that spelling at the load boundary so a
        # source checkpoint can be resumed without a conversion script.
        source_dreamwaq_graph = False
        if self.policy_architecture == "source_barlow":
            # Source DDP checkpoints may carry a uniform ``module.`` prefix;
            # the explicit source loader removes only known all-key prefixes
            # and always performs a strict graph/shape check.  It also accepts
            # a pure direct tensor mapping, which is useful for lightweight
            # source exporters that omit the runner metadata wrapper.
            load_source_state_dict(self.policy, checkpoint)
        elif self.algorithm_name == "him":
            state = checkpoint.get("policy_state_dict")
            if state is None:
                state = checkpoint.get("actor_state_dict")
            if state is None:
                state = checkpoint.get("model_state_dict")
            if state is None:
                raise KeyError(
                    "custom PPO checkpoint must contain policy_state_dict, actor_state_dict, "
                    "or model_state_dict"
                )
            load_source_mlp_compatible_state_dict(
                self.policy,
                state,
                mlp_roots=("actor", "critic"),
                label="HIM-PPO",
            )
        elif self.algorithm_name == "dreamwaq":
            state = checkpoint.get("policy_state_dict")
            if state is None:
                state = checkpoint.get("actor_state_dict")
            if state is None:
                state = checkpoint.get("model_state_dict")
            if state is None:
                raise KeyError(
                    "custom PPO checkpoint must contain policy_state_dict, actor_state_dict, "
                    "or model_state_dict"
                )
            source_dreamwaq_graph = load_source_mlp_compatible_state_dict(
                self.policy,
                state,
                mlp_roots=("actor", "critic", "encoder", "decoder"),
                label="DreamWaQ",
            )
        else:
            state = checkpoint.get("policy_state_dict")
            if state is None:
                state = checkpoint.get("actor_state_dict")
            if state is None:
                state = checkpoint.get("model_state_dict")
            if state is None:
                raise KeyError(
                    "custom PPO checkpoint must contain policy_state_dict, actor_state_dict, "
                    "or model_state_dict"
                )
            try:
                self.policy.load_state_dict(state, strict=True)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    "Custom PPO checkpoint tensors do not fit the selected task/algorithm contract: "
                    f"{exc}"
                ) from exc
        # Source runners call this field ``iter``; prefer the explicit UniLab
        # field when both are present, while treating a null canonical value as
        # absent so a valid legacy counter is not discarded.
        if load_iteration is not False:
            iteration_value = checkpoint.get("iteration")
            if iteration_value is None:
                iteration_value = checkpoint.get("iter")
            if iteration_value is not None:
                iteration = int(iteration_value)
                if iteration < 0:
                    raise ValueError(
                        f"Custom PPO checkpoint iteration must be non-negative, got {iteration}"
                    )
                self.current_learning_iteration = iteration

        # Restore optimizer moments when present.  The explicit error keeps a
        # checkpoint produced with incompatible hidden dimensions from being
        # mistaken for a valid fresh optimizer.  Legacy actor-only files skip
        # these optional states and retain the constructor defaults.
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if load_optimizer is not False and optimizer_state is not None:
            try:
                if source_dreamwaq_graph:
                    optimizer_state = remap_source_adam_state_dict(
                        self.policy,
                        self.alg.optimizer,
                        optimizer_state,
                        source_parameter_names=dreamwaq_source_parameter_names(self.policy),
                        label="DreamWaQ policy Adam",
                    )
                self.alg.optimizer.load_state_dict(optimizer_state)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Custom PPO checkpoint optimizer state does not fit the selected "
                    "task/algorithm contract"
                ) from exc

        # DreamWaQ's source runner names its CENet optimizer
        # ``vae_optimizer_state_dict``.  HIM/UniLab use the more descriptive
        # representation key; both feed the same owner optimizer here.
        representation_state = checkpoint.get("representation_optimizer_state_dict")
        if representation_state is None:
            representation_state = checkpoint.get("vae_optimizer_state_dict")
        representation_optimizer = getattr(self.alg, "representation_optimizer", None)
        if representation_optimizer is None:
            representation_optimizer = getattr(self.alg, "cenet_optimizer", None)
        if (
            load_optimizer is not False
            and representation_state is not None
            and representation_optimizer is not None
        ):
            try:
                if source_dreamwaq_graph:
                    representation_state = remap_source_adam_state_dict(
                        self.policy,
                        representation_optimizer,
                        representation_state,
                        source_parameter_names=dreamwaq_source_parameter_names(
                            self.policy, cenet_only=True
                        ),
                        label="DreamWaQ VAE Adam",
                    )
                representation_optimizer.load_state_dict(representation_state)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Custom PPO checkpoint representation optimizer state does not fit the "
                    "selected task/algorithm contract"
                ) from exc

        estimator_state = checkpoint.get("estimator_optimizer_state_dict")
        estimator = getattr(self.policy, "estimator", None)
        estimator_optimizer = getattr(estimator, "optimizer", None)
        if (
            load_optimizer is not False
            and estimator_state is not None
            and estimator_optimizer is not None
        ):
            try:
                estimator_optimizer.load_state_dict(estimator_state)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Custom PPO checkpoint estimator optimizer state does not fit the selected "
                    "task/algorithm contract"
                ) from exc

        algorithm_state = checkpoint.get("algorithm_state") if load_optimizer is not False else None
        if algorithm_state is not None and not isinstance(algorithm_state, dict):
            raise ValueError("Custom PPO checkpoint algorithm_state must be a mapping")
        if isinstance(algorithm_state, dict):
            learning_rate = algorithm_state.get("learning_rate")
            if learning_rate is not None and hasattr(self.alg, "learning_rate"):
                try:
                    learning_rate_value = float(learning_rate)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Custom PPO checkpoint learning_rate is invalid") from exc
                if (
                    not torch.isfinite(torch.tensor(learning_rate_value))
                    or learning_rate_value <= 0.0
                ):
                    raise ValueError(
                        "Custom PPO checkpoint learning_rate must be finite and positive"
                    )
                self.alg.learning_rate = learning_rate_value
                # Optimizer state carries its own learning rate; synchronize
                # the owner field too because adaptive schedules consult it.
                for param_group in self.alg.optimizer.param_groups:
                    param_group["lr"] = learning_rate_value
                if estimator is not None and estimator_optimizer is not None:
                    estimator_learning_rate = getattr(estimator, "learning_rate", None)
                    if estimator_learning_rate is not None:
                        estimator.learning_rate = learning_rate_value
                        for param_group in estimator_optimizer.param_groups:
                            param_group["lr"] = learning_rate_value

            k_value = algorithm_state.get("k_value")
            if k_value is not None and hasattr(self.alg, "k_value"):
                try:
                    restored_k = torch.as_tensor(
                        k_value, device=self.device, dtype=torch.float32
                    ).reshape(-1)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Custom PPO checkpoint k_value is invalid") from exc
                expected_k = int(getattr(self.policy, "num_costs", restored_k.numel()))
                if restored_k.numel() != expected_k:
                    raise ValueError(
                        "Custom PPO checkpoint k_value width does not fit NP3O's cost contract: "
                        f"checkpoint={restored_k.numel()}, runtime={expected_k}"
                    )
                self.alg.k_value = restored_k

            owner_state = algorithm_state.get("owner_state")
            algorithm_state_loader = getattr(self.alg, "load_algorithm_state_dict", None)
            if owner_state is not None and callable(algorithm_state_loader):
                try:
                    algorithm_state_loader(owner_state)
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    raise ValueError(
                        "Custom PPO checkpoint owner algorithm state does not fit the "
                        "selected task/algorithm contract"
                    ) from exc

        if load_iteration is not False:
            if checkpoint.get("total_timesteps") is not None:
                total_timesteps = int(checkpoint["total_timesteps"])
                if total_timesteps < 0:
                    raise ValueError(
                        "Custom PPO checkpoint total_timesteps must be non-negative, "
                        f"got {total_timesteps}"
                    )
                self.tot_timesteps = total_timesteps
            else:
                # Legacy custom checkpoints saved only the iteration.  Derive a
                # conservative counter so tracker summaries remain monotonic.
                self.tot_timesteps = (
                    self.current_learning_iteration * self.num_steps_per_env * self.env.num_envs
                )
        return checkpoint.get("infos")

    def get_inference_policy(self, device: str | None = None) -> Callable[..., torch.Tensor]:
        if device is not None:
            self.policy.to(device)
        self.policy.eval()
        if self.policy_architecture == "source_barlow":
            expected = int(self.contract["actor_input_dim"])

            def infer_source(
                obs: torch.Tensor | Mapping[str, Any], dones: torch.Tensor | None = None
            ) -> torch.Tensor:
                # Deployment callers may provide the wrapper TensorDict so
                # the adapter can maintain the source history and recover the
                # four-value privileged latent tail.  A pre-built 312D tensor
                # remains accepted for ONNX/parity callers.
                if (
                    isinstance(obs, torch.Tensor)
                    and obs.ndim == 2
                    and int(obs.shape[-1]) == expected
                ):
                    source_obs = obs.to(self._policy_device())
                else:
                    source_obs = self._source_on_constraint(obs, dones=dones)
                return self.policy.act_inference(source_obs)

            return infer_source
        one_step = int(self.env.num_obs)
        history: torch.Tensor | None = None

        def infer(obs: torch.Tensor, dones: torch.Tensor | None = None) -> torch.Tensor:
            nonlocal history
            runtime_device = self._policy_device()
            obs = obs.to(runtime_device)
            if history is None or history.shape[0] != obs.shape[0]:
                history = self._fresh_history(obs)
            else:
                history = torch.cat((history[:, one_step:], obs), dim=1)
            if dones is not None:
                done_mask = dones.to(device=runtime_device, dtype=torch.bool)
                if torch.any(done_mask):
                    history[done_mask] = self._fresh_history(obs[done_mask])
            return self.policy.act_inference(history)

        return infer

    def export_source_barlow_actor(
        self,
        path: str,
        *,
        jit_filename: str = "barlow_twins_actor.pt",
        onnx_filename: str = "barlow_twins_actor.onnx",
        device: str = "cpu",
        use_fp16_jit: bool = False,
    ) -> tuple[str, str]:
        """Export the source Barlow teacher actor with its dual-input ABI.

        ``export_policy_to_onnx`` remains the backwards-compatible UniLab
        full-policy export and consumes one 312D ``on_constraint`` tensor for
        a source runner.  The upstream ``exporter_normal.py`` instead exports
        only ``actor_teacher_backbone`` with ``obs=(N,28)`` and
        ``obs_hist=(N,10,28)``.  This explicit method emits both TorchScript
        and ONNX artifacts for that source deployment path and fails closed
        for compact policies.
        """

        if self.policy_architecture != "source_barlow":
            raise ValueError(
                "source Barlow actor export requires policy_architecture='source_barlow'; "
                "the compact policy has no source-compatible dual-input backbone"
            )
        from unilab.training.wheelbipe import export_barlow_twins_actor_from_policy

        return export_barlow_twins_actor_from_policy(
            self.policy,
            path,
            device=device,
            use_fp16_jit=use_fp16_jit,
            jit_filename=jit_filename,
            onnx_filename=onnx_filename,
            variant_name=str(self.contract.get("variant_name", "")),
        )

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx") -> str:
        os.makedirs(path, exist_ok=True)
        output = os.path.join(path, filename)
        original_device = next(self.policy.parameters()).device
        policy = self.policy.eval().cpu()

        class _Export(torch.nn.Module):
            def __init__(self, module: Any) -> None:
                super().__init__()
                self.module = module

            def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
                return self.module.act_inference(obs_history)

        export_model = _Export(policy).eval()
        dummy_width = int(
            self.contract.get("actor_input_dim", int(self.env.num_obs) * self.history_size)
        )
        dummy = torch.zeros(1, dummy_width)
        try:
            with torch.inference_mode():
                torch.onnx.export(
                    export_model,
                    (dummy,),
                    output,
                    input_names=["obs_history"],
                    output_names=["actions"],
                    opset_version=18,
                )
        finally:
            # Exporting on CPU is required by several ONNX backends, but an
            # exception must not leave a resumed CUDA runner silently on CPU.
            self.policy.to(original_device)
        # Keep deployment metadata beside the graph.  The sim2sim helper
        # treats this as an additive contract: old graph-only exports remain
        # shape-loadable, while a present sidecar lets it fail closed on an
        # algorithm/variant mismatch before creating an environment.
        env_cfg = getattr(self.env, "cfg", None)
        timing: dict[str, str | int | float | bool] = {}
        for key in (
            "sim_dt",
            "ctrl_dt",
            "delay_profile",
            "delay_range_semantics",
            "use_obs_delay",
            "use_act_delay",
            "obs_delay_step_unit",
        ):
            value = getattr(env_cfg, key, None) if env_cfg is not None else None
            if isinstance(value, (str, int, float, bool)):
                timing[key] = value
        from unilab.training.wheelbipe import (
            CUSTOM_WHEELBIPE_METADATA_SCHEMA,
            custom_wheelbipe_metadata_path,
        )

        metadata: dict[str, Any] = {
            "schema": CUSTOM_WHEELBIPE_METADATA_SCHEMA,
            "algorithm": self.algorithm_name,
            "variant_name": self.contract.get("variant_name", ""),
            "one_step_dim": int(self.contract["actor_obs_dim"]),
            "history_length": int(self.contract["history_length"]),
            "history_reset_mode": self.history_reset_mode,
            "input_dim": dummy_width,
            "output_dim": int(self.contract["num_actions"]),
            "num_actions": int(self.contract["num_actions"]),
            "num_costs": int(self.contract["num_costs"]),
            "input_name": "obs_history",
            "output_name": "actions",
            "critic_obs_dim": int(self.contract["critic_obs_dim"]),
            "num_estimate": int(self.contract["num_estimate"]),
            "architecture": str(self.contract.get("architecture", "compact")),
            # Source Barlow has two intentionally distinct deployment
            # artifacts: this runner graph consumes the complete one-input
            # ``on_constraint`` stream, while ``export_source_barlow_actor``
            # emits the upstream dual-input teacher actor.
            "artifact": (
                "source_barlow_full"
                if self.policy_architecture == "source_barlow"
                else "compact_history_policy"
            ),
            "timing": timing,
        }
        metadata_path = custom_wheelbipe_metadata_path(output)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


__all__ = [
    "CUSTOM_ALGORITHM_ALIASES",
    "CUSTOM_ALGORITHM_CONTRACTS",
    "CUSTOM_POLICY_ARCHITECTURES",
    "CUSTOM_HISTORY_RESET_MODES",
    "CustomOnPolicyRunner",
    "canonical_custom_history_reset_mode",
    "canonical_custom_algorithm",
    "validate_custom_runner_contract",
]
