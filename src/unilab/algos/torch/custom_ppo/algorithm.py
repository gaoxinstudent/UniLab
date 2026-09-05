"""Algorithm implementations for DreamWaQ and NP3O custom runtimes."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Callable, cast

import torch
from tensordict import TensorDict
from torch import nn, optim

from unilab.algos.torch.custom_ppo.models import ADABOOT_MODES
from unilab.algos.torch.custom_ppo.storage import CustomRolloutStorage
from unilab.algos.torch.him_ppo.algorithm import HIMPPO


def _extract_critic_observations(value: Any) -> torch.Tensor | None:
    """Return the critic stream from a timeout final-observation payload.

    ``RslRlVecEnvWrapper`` uses a :class:`~tensordict.TensorDict` with an
    explicit ``critic`` key.  Keeping the actor/policy fallbacks here mirrors
    the established HIM-PPO contract and makes the custom algorithms usable
    with lightweight vector-env adapters whose actor and critic streams are
    identical.  Unknown mappings are rejected by the caller rather than
    silently bootstrapping from the autoreset observation.
    """

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for key in ("critic", "policy", "actor"):
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
        return None
    if isinstance(value, TensorDict):
        for key in ("critic", "policy", "actor"):
            if key in value.keys():
                candidate = value[key]
                if isinstance(candidate, torch.Tensor):
                    return candidate
        return None
    # TensorDictBase subclasses and compatible adapters may not inherit from
    # ``TensorDict``.  Probe only their public mapping protocol; this is a
    # payload boundary, not a backend capability check.
    keys = getattr(value, "keys", None)
    if callable(keys):
        # Do not rely on the static type of ``keys()`` here.  TensorDictBase
        # adapters in the supported wrappers expose a mapping-like protocol,
        # but pyright quite correctly treats an arbitrary ``keys`` attribute
        # as ``object``.  Direct public-key lookup is both safer and more
        # permissive than an unchecked ``key in available`` operation (which
        # also fails for adapters returning a non-iterable view).
        try:
            keys()
        except (AttributeError, KeyError, TypeError):
            return None
        for key in ("critic", "policy", "actor"):
            try:
                candidate = value[key]
            except (AttributeError, KeyError, TypeError, IndexError):
                continue
            if isinstance(candidate, torch.Tensor):
                return candidate
    return None


def _batch_matrix(
    value: torch.Tensor,
    *,
    batch_size: int,
    device: str,
    label: str,
    feature_size: int | None = None,
) -> torch.Tensor:
    """Normalize a final-observation tensor to ``(batch, features)``."""

    result = value.to(device=device)
    if result.ndim == 1 and batch_size == 1:
        result = result.unsqueeze(0)
    if result.ndim != 2 or result.shape[0] != batch_size:
        raise ValueError(
            f"Custom PPO {label} must have shape ({batch_size}, features), got {tuple(result.shape)}"
        )
    if feature_size is not None and result.shape[1] != feature_size:
        raise ValueError(f"Custom PPO {label} width must be {feature_size}, got {result.shape[1]}")
    return result.detach()


def _value_vector(value: torch.Tensor, *, batch_size: int, label: str) -> torch.Tensor:
    """Normalize a scalar critic output to one value per environment."""

    result = value.detach()
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1 or result.shape[0] != batch_size:
        raise ValueError(
            f"Custom PPO {label} must have one value per environment, got {tuple(result.shape)}"
        )
    return result


def _shape_width(value: Any, *, label: str) -> int:
    """Normalize a source/RSL-RL shape declaration to one feature width.

    The upstream runners pass one-element lists (for example ``[28]``),
    while the compact owner uses plain integers.  Accepting only a single
    feature dimension keeps this adapter explicit; flattening a higher-rank
    source observation would silently change the model contract.
    """

    if isinstance(value, bool):
        raise ValueError(f"Custom PPO {label} shape must be an integer width")
    if isinstance(value, int):
        width = int(value)
    elif isinstance(value, Sequence) or isinstance(value, torch.Size):
        dims = tuple(int(dim) for dim in value)
        if len(dims) != 1:
            raise ValueError(f"Custom PPO {label} shape must be one-dimensional, got {dims}")
        width = dims[0]
    else:
        raise TypeError(f"Custom PPO {label} shape must be an integer or one-element sequence")
    if width <= 0:
        raise ValueError(f"Custom PPO {label} width must be positive, got {width}")
    return width


def _mapping_value(value: Any, keys: tuple[str, ...]) -> torch.Tensor | None:
    """Read a tensor leaf from a source observation mapping."""

    if isinstance(value, torch.Tensor):
        return value
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


def _source_actor_tensor(value: Any, policy: Any) -> torch.Tensor:
    """Resolve a source ``policy_hist``/compact actor tensor for storage."""

    actor = _mapping_value(value, ("policy_hist", "actor_history", "history", "actor", "policy"))
    if actor is None:
        raise ValueError(
            "DreamWaQ observation payload must contain a tensor under "
            "policy_hist, actor_history, history, actor, or policy"
        )
    if actor.ndim != 2:
        raise ValueError(
            f"DreamWaQ actor observations must be rank-2, got shape={tuple(actor.shape)}"
        )
    expected = int(
        getattr(policy, "cenet_in_dim", getattr(policy, "num_actor_obs", actor.shape[-1]))
    )
    one_step = int(getattr(policy, "num_one_step_obs", actor.shape[-1]))
    if int(actor.shape[-1]) == one_step and expected != one_step:
        history_size = int(getattr(policy, "history_size", expected // one_step))
        actor = actor.repeat(1, history_size)
    if int(actor.shape[-1]) != expected:
        raise ValueError(
            "DreamWaQ actor history width does not match the policy contract: "
            f"expected {expected}, got {actor.shape[-1]}"
        )
    return actor


def _source_critic_tensor(value: Any, policy: Any) -> torch.Tensor:
    """Resolve a source ``critic``/``prev_critic`` tensor for storage."""

    critic = _mapping_value(value, ("critic", "prev_critic", "policy", "actor"))
    if critic is None:
        raise ValueError(
            "DreamWaQ observation payload must contain a tensor under "
            "critic, prev_critic, policy, or actor"
        )
    if critic.ndim != 2:
        raise ValueError(
            f"DreamWaQ critic observations must be rank-2, got shape={tuple(critic.shape)}"
        )
    expected = int(getattr(policy, "num_critic_obs", critic.shape[-1]))
    if int(critic.shape[-1]) != expected:
        raise ValueError(
            "DreamWaQ critic observation width does not match the policy contract: "
            f"expected {expected}, got {critic.shape[-1]}"
        )
    return critic


def _apply_timeout_bootstrap(
    transition: dict[str, torch.Tensor],
    *,
    policy: Any,
    gamma: float,
    extras: Mapping[str, Any],
    device: str,
    cost_evaluator: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Apply final-observation time-limit bootstrapping to a custom transition.

    The wrapper marks time-limit rows as ``done`` so history/reset handling is
    deterministic, while PPO targets must still include the value of the
    *terminal* observation.  This helper mirrors the standard/HIM PPO owner:
    use the explicit final critic observation when available, fall back to
    the transition value for legacy adapters, and patch the stored next critic
    row so DreamWaQ's adaptation target sees the terminal frame rather than
    the autoreset frame.  NP3O's cost critics receive the same correction.
    """

    timeouts = extras.get("time_outs")
    if not isinstance(timeouts, torch.Tensor):
        return

    rewards = transition.get("rewards")
    next_critic = transition.get("next_critic_observations")
    values = transition.get("values")
    if not isinstance(rewards, torch.Tensor) or not isinstance(next_critic, torch.Tensor):
        raise ValueError("Custom PPO transition is missing rewards or next critic observations")
    batch_size = int(rewards.shape[0])
    timeout_bool = timeouts.to(device=device).bool().reshape(-1)
    if timeout_bool.numel() != batch_size:
        raise ValueError(
            "Custom PPO time_outs width does not match transition batch: "
            f"timeouts={timeout_bool.numel()}, batch={batch_size}"
        )
    if not bool(torch.any(timeout_bool)):
        return

    next_critic = _batch_matrix(
        next_critic,
        batch_size=batch_size,
        device=device,
        label="next critic observations",
    )
    timeout_bootstrap_payload = extras.get("time_out_bootstrap_obs")
    bootstrap_critic: torch.Tensor | None = None
    if timeout_bootstrap_payload is not None:
        bootstrap_critic = _extract_critic_observations(timeout_bootstrap_payload)
        if bootstrap_critic is None:
            raise ValueError(
                "Custom PPO time_out_bootstrap_obs must contain a critic, policy, or actor tensor"
            )
        bootstrap_critic = _batch_matrix(
            bootstrap_critic,
            batch_size=batch_size,
            device=device,
            label="timeout bootstrap critic observations",
            feature_size=int(next_critic.shape[1]),
        )

    with torch.no_grad():
        if bootstrap_critic is not None:
            bootstrap_values = policy.evaluate(bootstrap_critic)
        else:
            if not isinstance(values, torch.Tensor):
                raise ValueError(
                    "Custom PPO transition is missing critic values for timeout fallback"
                )
            bootstrap_values = values.to(device=device)
        reward_values = _value_vector(
            bootstrap_values,
            batch_size=batch_size,
            label="timeout bootstrap values",
        )

        reward_correction = (
            float(gamma) * reward_values * timeout_bool.to(dtype=reward_values.dtype)
        )
        if rewards.ndim == 1 and rewards.shape[0] == batch_size:
            transition["rewards"] = rewards.to(device=device).detach().clone() + reward_correction
        elif rewards.ndim == 2 and rewards.shape == (batch_size, 1):
            transition["rewards"] = rewards.to(
                device=device
            ).detach().clone() + reward_correction.unsqueeze(1)
        else:
            raise ValueError(
                "Custom PPO rewards must have shape (batch,) or (batch, 1), "
                f"got {tuple(rewards.shape)}"
            )

        if bootstrap_critic is not None:
            patched_next_critic = next_critic.clone()
            patched_next_critic[timeout_bool] = bootstrap_critic[timeout_bool]
            transition["next_critic_observations"] = patched_next_critic

        if cost_evaluator is not None:
            costs = transition.get("costs")
            cost_values = transition.get("cost_values")
            if not isinstance(costs, torch.Tensor) or not isinstance(cost_values, torch.Tensor):
                raise ValueError("NP3O transition is missing costs or cost values")
            if bootstrap_critic is not None:
                bootstrap_cost_values = cost_evaluator(bootstrap_critic)
            else:
                bootstrap_cost_values = cost_values.to(device=device)
            bootstrap_cost_values = bootstrap_cost_values.to(device=device).detach()
            if bootstrap_cost_values.ndim == 1:
                bootstrap_cost_values = bootstrap_cost_values.unsqueeze(1)
            if bootstrap_cost_values.ndim != 2 or bootstrap_cost_values.shape[0] != batch_size:
                raise ValueError(
                    "NP3O timeout bootstrap cost values must have shape "
                    f"({batch_size}, num_costs), got {tuple(bootstrap_cost_values.shape)}"
                )
            if costs.ndim == 1 and bootstrap_cost_values.shape[1] == 1:
                transition["costs"] = costs.to(device=device).detach().clone() + (
                    float(gamma)
                    * bootstrap_cost_values[:, 0]
                    * timeout_bool.to(dtype=bootstrap_cost_values.dtype)
                )
            elif costs.ndim == 2 and costs.shape == bootstrap_cost_values.shape:
                transition["costs"] = costs.to(device=device).detach().clone() + (
                    float(gamma)
                    * bootstrap_cost_values
                    * timeout_bool.to(dtype=bootstrap_cost_values.dtype).unsqueeze(1)
                )
            else:
                raise ValueError(
                    "NP3O costs shape does not match timeout bootstrap values: "
                    f"costs={tuple(costs.shape)}, values={tuple(bootstrap_cost_values.shape)}"
                )


class DreamWaQPPO:
    """PPO + DreamWaQ CENet reconstruction, velocity and KL objectives."""

    def __init__(
        self,
        policy: Any,
        *source_args: Any,
        device: str = "cpu",
        **cfg: Any,
    ) -> None:
        # ``PPODreamWaq`` in the source WheelBipe tree exposes the standard
        # RSL-RL hyperparameters positionally.  Keep this additive parser at
        # the owner boundary so compact UniLab keyword configs retain their
        # existing defaults, while source launch/config code can call the
        # migrated class without a Python-side rewrite.  Unsupported values
        # are retained as config metadata but never used to alter the compact
        # network contract.
        source_fields = (
            "num_learning_epochs",
            "num_mini_batches",
            "clip_param",
            "gamma",
            "lam",
            "value_loss_coef",
            "entropy_coef",
            "learning_rate",
            "vae_learning_rate",
            "num_adaptation_module_substeps",
            "kl_weight",
            "max_grad_norm",
            "use_clipped_value_loss",
            "schedule",
            "desired_kl",
            "device",
            "normalize_advantage_per_mini_batch",
            "multi_gpu_cfg",
        )
        if len(source_args) > len(source_fields):
            raise TypeError(
                "too many positional arguments for source PPODreamWaq constructor: "
                f"got {len(source_args) + 1}"
            )
        for field, value in zip(source_fields, source_args):
            if field == "device":
                if device != "cpu" and str(device) != str(value):
                    raise ValueError(
                        "PPODreamWaq device conflicts between positional and named arguments: "
                        f"{value!r} != {device!r}"
                    )
                device = str(value)
                continue
            if field in cfg and cfg[field] != value:
                raise ValueError(
                    f"PPODreamWaq {field} conflicts between positional and named arguments"
                )
            cfg.setdefault(field, value)

        self.policy = policy
        self.device = device
        self.policy.to(device)

        # AdaBoot is configured on the source policy (``adaboot_mode``) while
        # its reward-window parameters live on the source algorithm config.
        # Resolve both at this owner boundary so a non-off mode cannot be
        # accepted and then silently ignored by the PPO update.  An explicit
        # algorithm override is allowed when the policy still carries the
        # default ``off`` value; conflicting non-default declarations fail
        # closed instead of selecting one arbitrarily.
        policy_mode = str(getattr(self.policy, "adaboot_mode", "off")).strip().lower()
        mode_override = cfg.pop("adaboot_mode", None)
        if mode_override is None:
            mode = policy_mode
        else:
            mode = str(mode_override).strip().lower()
            if policy_mode not in {"off", mode}:
                raise ValueError(
                    "DreamWaQ AdaBoot mode mismatch between policy and algorithm: "
                    f"policy={policy_mode!r}, algorithm={mode!r}"
                )
        raw_use_adaboot = cfg.pop("use_adaboot", None)
        if raw_use_adaboot is not None:
            if not isinstance(raw_use_adaboot, bool):
                raise ValueError("DreamWaQ use_adaboot must be a boolean")
            if raw_use_adaboot and mode == "off":
                mode = "uncertainty"
            elif not raw_use_adaboot and mode != "off":
                raise ValueError(
                    f"DreamWaQ use_adaboot=false conflicts with non-off adaboot_mode={mode!r}"
                )
        if mode not in ADABOOT_MODES:
            supported = ", ".join(sorted(ADABOOT_MODES))
            raise ValueError(
                f"Unsupported DreamWaQ adaboot_mode={mode!r}; expected one of {supported}"
            )
        if not hasattr(self.policy, "set_adaboot_p_boot"):
            if mode != "off":
                raise ValueError("DreamWaQ non-off AdaBoot mode requires policy.set_adaboot_p_boot")
        else:
            # Keep the model and algorithm declarations synchronized.  This
            # is a public policy attribute, not backend capability probing.
            self.policy.adaboot_mode = mode
        self.adaboot_mode = mode

        self.adaboot_reward_window_size = self._positive_int_config(
            cfg.pop("adaboot_reward_window_size", 1024), "adaboot_reward_window_size"
        )
        self.adaboot_reward_cv_scale = self._finite_float_config(
            cfg.pop("adaboot_reward_cv_scale", 0.5), "adaboot_reward_cv_scale"
        )
        if self.adaboot_reward_cv_scale < 0.0:
            raise ValueError("DreamWaQ adaboot_reward_cv_scale must be non-negative")
        self.adaboot_reward_cv_offset = self._finite_float_config(
            cfg.pop("adaboot_reward_cv_offset", 0.0), "adaboot_reward_cv_offset"
        )
        self.adaboot_pboot_min = self._finite_float_config(
            cfg.pop("adaboot_pboot_min", 0.0), "adaboot_pboot_min"
        )
        self.adaboot_pboot_max = self._finite_float_config(
            cfg.pop("adaboot_pboot_max", 1.0), "adaboot_pboot_max"
        )
        if not (0.0 <= self.adaboot_pboot_min <= self.adaboot_pboot_max <= 1.0):
            raise ValueError(
                "DreamWaQ AdaBoot p_boot bounds must satisfy 0 <= adaboot_pboot_min "
                f"<= adaboot_pboot_max <= 1, got "
                f"({self.adaboot_pboot_min}, {self.adaboot_pboot_max})"
            )
        unknown_adaboot = sorted(str(key) for key in cfg if str(key).startswith("adaboot_"))
        if unknown_adaboot:
            raise ValueError(
                "Unsupported DreamWaQ AdaBoot algorithm config key(s): "
                f"{unknown_adaboot!r}; policy-only knobs belong under algo.policy"
            )

        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=float(cfg.get("learning_rate", 3e-4))
        )
        # Source ``PPODreamWaq`` treats an explicit ``vae_learning_rate=None``
        # as "reuse learning_rate".  ``dict.get`` alone cannot express that
        # distinction (and ``float(None)`` fails), so resolve the alias before
        # constructing the optimizer.
        raw_vae_learning_rate = cfg.get("vae_learning_rate")
        if raw_vae_learning_rate is None:
            raw_vae_learning_rate = cfg.get("learning_rate", 3e-4)
        self.cenet_optimizer = optim.Adam(
            self.policy.cenet_parameters(),
            lr=float(raw_vae_learning_rate),
        )
        # Upstream DreamWaQ calls this optimizer ``vae_optimizer``.  Keep a
        # reference alias (rather than a second optimizer) so source tooling
        # can inspect/load it without creating duplicate parameter state.
        self.vae_optimizer = self.cenet_optimizer
        self.learning_rate = float(cfg.get("learning_rate", 3e-4))
        self.vae_learning_rate = float(raw_vae_learning_rate)
        self.num_learning_epochs = int(cfg.get("num_learning_epochs", 1))
        self.num_mini_batches = int(cfg.get("num_mini_batches", 1))
        self.clip_param = float(cfg.get("clip_param", 0.2))
        self.gamma, self.lam = float(cfg.get("gamma", 0.99)), float(cfg.get("lam", 0.95))
        self.value_loss_coef = float(cfg.get("value_loss_coef", 1.0))
        self.use_clipped_value_loss = bool(cfg.get("use_clipped_value_loss", True))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.0))
        self.kl_weight = float(cfg.get("kl_weight", 1.0))
        self.adaptation_substeps = int(cfg.get("num_adaptation_module_substeps", 1))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.schedule = str(cfg.get("schedule", "fixed"))
        self.desired_kl = cfg.get("desired_kl", None)
        self.normalize_advantage_per_mini_batch = bool(
            cfg.get("normalize_advantage_per_mini_batch", False)
        )
        multi_gpu_cfg = cfg.get("multi_gpu_cfg")
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = (
            int(multi_gpu_cfg.get("global_rank", 0)) if isinstance(multi_gpu_cfg, Mapping) else 0
        )
        self.gpu_world_size = (
            int(multi_gpu_cfg.get("world_size", 1)) if isinstance(multi_gpu_cfg, Mapping) else 1
        )
        self.storage: CustomRolloutStorage | None = None
        # Source callers inspect ``transition`` while the compact owner uses a
        # dictionary internally.  Keep an alias to the current transition;
        # it is replaced on every ``act`` call and cleared after recording.
        self.transition: dict[str, torch.Tensor] = {}
        self._transition: dict[str, torch.Tensor] = {}
        self._source_mapping_mode = False
        self._ep_partial_returns: torch.Tensor | None = None
        self._ep_return_window: deque[float] = deque(maxlen=self.adaboot_reward_window_size)
        self._last_adaboot_stats: dict[str, float] = {
            "p_boot": 1.0,
            "r_mean": 0.0,
            "r_std": 0.0,
            "cv_r": 0.0,
            "window_count": 0.0,
        }

    @staticmethod
    def _finite_float_config(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"DreamWaQ {name} must be a finite number")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"DreamWaQ {name} must be a finite number") from exc
        if not math.isfinite(converted):
            raise ValueError(f"DreamWaQ {name} must be finite")
        return converted

    @staticmethod
    def _positive_int_config(value: Any, name: str) -> int:
        # Do not silently turn fractional values into a different window.  A
        # Hydra string such as ``"1024"`` remains accepted, matching source
        # ``int(...)`` behavior, while 2.5 is rejected as a malformed config.
        if isinstance(value, bool):
            raise ValueError(f"DreamWaQ {name} must be a positive integer")
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"DreamWaQ {name} must be a positive integer") from exc
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"DreamWaQ {name} must be a positive integer") from exc
        if not math.isfinite(numeric) or numeric != converted or converted <= 0:
            raise ValueError(f"DreamWaQ {name} must be a positive integer")
        return converted

    def init_storage(
        self,
        num_envs: int,
        num_steps: int,
        actor_dim: Any,
        critic_dim: Any | None = None,
        action_dim: Any | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize rollout storage for compact or source-style callers.

        Compact UniLab calls use ``(actor_dim, critic_dim, action_dim,
        device)``.  The upstream DreamWaQ runner uses
        ``(observations_shape: mapping, action_shape)``; source observation
        mappings are normalized to the same flattened tensor storage after
        validating their policy-history and critic widths.  This preserves the
        compact update path without pretending to implement the source
        algorithm's separate storage class.
        """

        if isinstance(actor_dim, Mapping):
            if critic_dim is None:
                raise TypeError(
                    "source DreamWaQ init_storage requires an action shape after observations_shape"
                )
            if action_dim is not None or device is not None:
                raise TypeError(
                    "source DreamWaQ init_storage accepts observations_shape and action_shape only"
                )
            observation_shapes = actor_dim
            actor_shape = observation_shapes.get("policy_hist")
            if actor_shape is None:
                actor_shape = observation_shapes.get("policy")
            if actor_shape is None:
                raise ValueError(
                    "source DreamWaQ observations_shape must contain policy_hist or policy"
                )
            actor_width = _shape_width(actor_shape, label="actor observation")
            # A source map with only ``policy`` is a valid cold-start payload;
            # the model repeats it to its configured history width.
            expected_actor = int(
                getattr(
                    self.policy, "cenet_in_dim", getattr(self.policy, "num_actor_obs", actor_width)
                )
            )
            if actor_width == int(getattr(self.policy, "num_one_step_obs", actor_width)):
                actor_width = expected_actor
            if actor_width != expected_actor:
                raise ValueError(
                    "source DreamWaQ actor observation width does not match policy: "
                    f"expected {expected_actor}, got {actor_width}"
                )
            critic_shape = observation_shapes.get("critic")
            if critic_shape is None:
                critic_shape = observation_shapes.get("prev_critic")
            if critic_shape is None:
                critic_shape = getattr(self.policy, "num_critic_obs", None)
            if critic_shape is None:
                raise ValueError(
                    "source DreamWaQ observations_shape must contain critic or prev_critic"
                )
            critic_width = _shape_width(critic_shape, label="critic observation")
            expected_critic = int(getattr(self.policy, "num_critic_obs", critic_width))
            if critic_width != expected_critic:
                raise ValueError(
                    "source DreamWaQ critic observation width does not match policy: "
                    f"expected {expected_critic}, got {critic_width}"
                )
            action_width = _shape_width(critic_dim, label="action")
            self._source_mapping_mode = True
            storage_device = self.device
        else:
            if critic_dim is None or action_dim is None or device is None:
                raise TypeError(
                    "compact DreamWaQ init_storage requires actor_dim, critic_dim, action_dim, and device"
                )
            actor_width = _shape_width(actor_dim, label="actor observation")
            critic_width = _shape_width(critic_dim, label="critic observation")
            action_width = _shape_width(action_dim, label="action")
            self._source_mapping_mode = False
            storage_device = str(device)
        self.storage = CustomRolloutStorage(
            num_envs,
            num_steps,
            actor_width,
            critic_width,
            action_width,
            storage_device,
            num_costs=0,
        )
        self._ep_partial_returns = torch.zeros(int(num_envs), device=storage_device)
        self._ep_return_window.clear()
        self._last_adaboot_stats = {
            "p_boot": 1.0,
            "r_mean": 0.0,
            "r_std": 0.0,
            "cv_r": 0.0,
            "window_count": 0.0,
        }
        # A newly initialized owner must not inherit a stale coefficient from
        # a previous run or checkpoint.  Reward-CV/hybrid will set it at the
        # first update after a rollout; uncertainty/off do not consume it.
        setter = getattr(self.policy, "set_adaboot_p_boot", None)
        if callable(setter):
            setter(None)

    def test_mode(self) -> None:
        """Switch the actor/critic to evaluation mode (source API parity)."""

        self.policy.eval()

    def train_mode(self) -> None:
        """Switch the actor/critic to training mode (source API parity)."""

        self.policy.train()

    def act(
        self,
        observations: Any,
        critic_observations: Any | None = None,
    ) -> torch.Tensor:
        """Sample an action from compact tensors or a source observation map."""

        if self.storage is None:
            raise RuntimeError("DreamWaQ act() requires init_storage() first")
        if critic_observations is None:
            critic_observations = observations
        # Resolve/validate the payload before invoking the model.  The model
        # itself accepts mappings, while storage remains tensor-only.
        actor_tensor = _source_actor_tensor(observations, self.policy)
        critic_tensor = _source_critic_tensor(critic_observations, self.policy)
        # Source DreamWaQ's AdaBoot training branch can blend the sampled
        # velocity code with the privileged critic target.  Compact UniLab
        # callers normally pass two tensors (history, critic) rather than the
        # source mapping; provide the public leaves only when that branch is
        # explicitly enabled.  Deterministic inference remains tensor-only and
        # bypasses AdaBoot in the policy owner, matching source deployment.
        policy_input: Any = observations
        if (
            not isinstance(observations, Mapping)
            and str(getattr(self.policy, "adaboot_mode", "off")).lower() != "off"
        ):
            policy_input = {
                "policy_hist": actor_tensor,
                "policy": actor_tensor[:, -int(self.policy.num_one_step_obs) :],
                "critic": critic_tensor,
            }
        action = self.policy.act(policy_input).detach()
        value = self.policy.evaluate(critic_observations).detach()
        self._transition = {
            "observations": actor_tensor.to(self.device).detach(),
            "critic_observations": critic_tensor.to(self.device).detach(),
            "actions": action,
            "values": value,
            "log_probs": self.policy.get_actions_log_prob(action).detach(),
            "mu": self.policy.action_mean.detach(),
            "sigma": self.policy.action_std.detach(),
        }
        self.transition = self._transition
        return action

    def process_env_step(
        self,
        next_critic_observations: Any = None,
        rewards: Any = None,
        dones: Any = None,
        extras: Any = None,
        *,
        next_obs_dict: Any = None,
        next_obs: Any = None,
    ) -> None:
        """Record compact or source-order transition arguments.

        Compact order is ``(next_critic, rewards, dones, extras)``.  The
        upstream DreamWaQ runner uses ``(rewards, dones, extras,
        next_obs_dict=None)``.  Mapping-vs-tensor argument positions make the
        two forms unambiguous; ambiguous payloads fail with a clear type
        diagnostic instead of being silently reordered.
        """

        if self.storage is None:
            raise RuntimeError("DreamWaQ process_env_step() requires init_storage() first")
        if next_obs is not None:
            if next_obs_dict is not None:
                raise TypeError("pass only one of next_obs and next_obs_dict")
            next_obs_dict = next_obs

        # A source positional call binds as
        # ``(next_critic_observations=rewards, rewards=dones,
        # dones=extras, extras=next_obs_dict)``.  A keyword source call
        # supplies ``next_obs_dict`` and follows the same remapping.
        source_order = isinstance(dones, Mapping) or isinstance(dones, TensorDict)
        if next_obs_dict is not None and not source_order:
            # Explicit source keyword form: rewards/dones/extras are already
            # bound to their named parameters.
            next_critic_observations = next_obs_dict
            source_order = True
        elif (
            not source_order
            and next_critic_observations is None
            and isinstance(rewards, torch.Tensor)
            and isinstance(dones, torch.Tensor)
            and isinstance(extras, Mapping)
        ):
            # Source callers may omit the optional ``next_obs_dict`` while
            # using keyword arguments.  In that form the prior critic frame
            # is the only safe next-observation fallback (the source storage
            # itself also treats the current frame as the default).
            previous_critic = self._transition.get("critic_observations")
            if previous_critic is None:
                raise ValueError(
                    "source DreamWaQ process_env_step requires next_obs_dict or a prior critic observation"
                )
            next_critic_observations = previous_critic
            source_order = True
        elif source_order:
            source_rewards = next_critic_observations
            source_dones = rewards
            source_extras = dones
            source_next = next_obs_dict if next_obs_dict is not None else extras
            rewards, dones, extras = source_rewards, source_dones, source_extras
            if source_next is None:
                previous_critic = self._transition.get("critic_observations")
                if previous_critic is None:
                    raise ValueError(
                        "source DreamWaQ process_env_step requires next_obs_dict or a prior critic observation"
                    )
                next_critic_observations = previous_critic
            else:
                next_critic_observations = _source_critic_tensor(source_next, self.policy)

        if not isinstance(rewards, torch.Tensor):
            raise TypeError("DreamWaQ rewards must be a tensor")
        if not isinstance(dones, torch.Tensor):
            raise TypeError("DreamWaQ dones must be a tensor")
        if extras is None:
            extras = {}
        if not isinstance(extras, Mapping):
            raise TypeError("DreamWaQ extras must be a mapping")
        if not isinstance(self._transition, dict) or not self._transition:
            raise RuntimeError("DreamWaQ process_env_step() requires act() first")
        next_critic_observations = _source_critic_tensor(next_critic_observations, self.policy)
        self._transition["next_critic_observations"] = (
            next_critic_observations.to(self.device).detach().clone()
        )
        self._transition["rewards"] = rewards.to(self.device).detach().clone()
        self._transition["dones"] = dones.to(self.device).detach()
        _apply_timeout_bootstrap(
            self._transition,
            policy=self.policy,
            gamma=self.gamma,
            extras=extras,
            device=self.device,
        )
        self.storage.add(self._transition)
        self._transition = {}
        self.transition = self._transition
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(dones)

    def compute_returns(self, critic_observations: Any) -> None:
        if self.storage is None:
            raise RuntimeError("DreamWaQ compute_returns() requires init_storage() first")
        self.storage.compute_returns(
            self.policy.evaluate(critic_observations).detach(), self.gamma, self.lam
        )

    def _update_episode_reward_stats(self) -> None:
        """Fold completed rollout episodes into the source AdaBoot window.

        The source runner keeps a per-environment partial return across
        rollouts and appends only rows marked done.  Mirroring that behavior at
        the algorithm/storage boundary ensures timeout-corrected rewards are
        represented consistently and avoids counting an episode twice when a
        runner calls ``update`` after a short rollout.
        """

        if self.storage is None:
            raise RuntimeError("DreamWaQ reward statistics require initialized storage")
        rewards = self.storage.rewards
        dones = self.storage.dones
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards[..., 0]
        if dones.ndim == 3 and dones.shape[-1] == 1:
            dones = dones[..., 0]
        if rewards.ndim != 2 or dones.ndim != 2 or rewards.shape != dones.shape:
            raise ValueError(
                "DreamWaQ rollout rewards/dones must have shape (steps, envs[, 1]); "
                f"got rewards={tuple(self.storage.rewards.shape)}, "
                f"dones={tuple(self.storage.dones.shape)}"
            )
        num_envs = int(rewards.shape[1])
        if self._ep_partial_returns is None or self._ep_partial_returns.numel() != num_envs:
            self._ep_partial_returns = torch.zeros(num_envs, device=self.device)
        running = self._ep_partial_returns.to(device=self.device)
        rewards = rewards.to(device=self.device)
        dones = dones.to(device=self.device).bool()
        if not bool(torch.all(torch.isfinite(rewards))):
            raise ValueError("DreamWaQ rollout rewards must be finite for AdaBoot statistics")
        for step in range(int(rewards.shape[0])):
            running = running + rewards[step]
            done_ids = torch.nonzero(dones[step], as_tuple=False).flatten()
            if done_ids.numel() > 0:
                for value in running[done_ids]:
                    self._ep_return_window.append(float(value.item()))
                running[done_ids] = 0.0
        self._ep_partial_returns = running

    def _compute_p_boot_from_episode_rewards(self) -> tuple[torch.Tensor, ...]:
        """Return ``(p_boot, mean, std, CV)`` using source AdaBoot semantics."""

        self._update_episode_reward_stats()
        assert self.storage is not None
        if len(self._ep_return_window) < 2:
            # This is the source fallback while the rolling window warms up.
            # Flattening keeps one scalar per environment even if a caller
            # supplies a singleton reward dimension.
            r_episode = self.storage.rewards
            if r_episode.ndim == 3 and r_episode.shape[-1] == 1:
                r_episode = r_episode[..., 0]
            r_episode = r_episode.to(device=self.device).sum(dim=0).reshape(-1)
        else:
            r_episode = torch.as_tensor(
                list(self._ep_return_window), device=self.device, dtype=torch.float32
            ).reshape(-1)
        if r_episode.numel() == 0 or not bool(torch.all(torch.isfinite(r_episode))):
            raise ValueError("DreamWaQ AdaBoot episode rewards must contain finite values")
        r_mean = torch.mean(r_episode)
        r_std = torch.std(r_episode, unbiased=False)
        cv_r = r_std / (torch.abs(r_mean) + 1.0e-8)
        shaped_cv = self.adaboot_reward_cv_scale * cv_r + self.adaboot_reward_cv_offset
        p_boot = torch.clamp(
            1.0 - torch.tanh(shaped_cv),
            self.adaboot_pboot_min,
            self.adaboot_pboot_max,
        )
        self._last_adaboot_stats = {
            "p_boot": float(p_boot.item()),
            "r_mean": float(r_mean.item()),
            "r_std": float(r_std.item()),
            "cv_r": float(cv_r.item()),
            "window_count": float(len(self._ep_return_window)),
        }
        return p_boot, r_mean, r_std, cv_r

    def _set_adaboot_p_boot(self, p_boot: torch.Tensor) -> None:
        """Publish the current reward-CV coefficient to the policy owner."""

        setter = getattr(self.policy, "set_adaboot_p_boot", None)
        if not callable(setter):
            if self.adaboot_mode != "off":
                raise RuntimeError(
                    "DreamWaQ non-off AdaBoot mode requires policy.set_adaboot_p_boot"
                )
            return
        if self.adaboot_mode in {"reward_cv", "hybrid"}:
            setter(p_boot.detach())
        else:
            # Clear a previous reward coefficient when switching an in-memory
            # policy to uncertainty/off before another update.
            setter(None)

    def _adaboot_alpha_metric(self, p_boot: torch.Tensor) -> float:
        getter = getattr(self.policy, "get_adaboot_p_boot", None)
        value = getter() if callable(getter) else None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0 or not bool(torch.all(torch.isfinite(value))):
                raise ValueError("DreamWaQ policy returned a non-finite AdaBoot coefficient")
            return float(value.detach().mean().item())
        if value is not None:
            try:
                # ``getattr`` intentionally accepts source policy adapters
                # whose scalar return type is not visible to the checker;
                # runtime conversion/validation below remains the contract.
                converted = float(cast(Any, value))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("DreamWaQ policy returned an invalid AdaBoot coefficient") from exc
            if not math.isfinite(converted):
                raise ValueError("DreamWaQ policy returned a non-finite AdaBoot coefficient")
            return converted
        # A policy implementation that does not expose the post-forward alpha
        # may still consume the scalar p_boot; retain the source metric
        # fallback (including the default ``off`` mode) rather than silently
        # reporting zero for a configured reward-CV statistic.
        return float(p_boot.item())

    def _velocity_target(self, critic_observations: torch.Tensor) -> torch.Tensor:
        """Return the source DreamWaQ target aligned with the current history frame."""

        start = int(self.policy.num_one_step_obs)
        end = start + int(self.policy.num_estimate)
        if critic_observations.ndim != 2 or int(critic_observations.shape[-1]) < end:
            raise ValueError(
                "DreamWaQ critic observation is too small for the current-frame velocity "
                f"target [{start}:{end}]: shape={tuple(critic_observations.shape)}"
            )
        return critic_observations[:, start:end].detach()

    def algorithm_state_dict(self) -> dict[str, Any]:
        """Return non-optimizer DreamWaQ state needed for exact resume.

        AdaBoot is driven by completed episode returns, which can span rollout
        boundaries.  Saving only the policy and optimizer therefore changes
        the next ``p_boot`` coefficient after a resume.  Keep this payload
        tensor/primitive-only so it remains safe for ``torch.load`` with
        ``weights_only=True`` and portable across CPU/GPU runs.
        """

        partial = self._ep_partial_returns
        policy_p_boot: torch.Tensor | float | None = None
        getter = getattr(self.policy, "get_adaboot_p_boot", None)
        if callable(getter):
            value = getter()
            if isinstance(value, torch.Tensor):
                policy_p_boot = value.detach().cpu()
            elif value is not None:
                policy_p_boot = float(cast(Any, value))
        return {
            "version": 1,
            "adaboot_mode": self.adaboot_mode,
            "adaboot_reward_window_size": int(self.adaboot_reward_window_size),
            "adaboot_episode_return_window": [float(value) for value in self._ep_return_window],
            "adaboot_partial_returns": None if partial is None else partial.detach().cpu(),
            "adaboot_last_stats": dict(self._last_adaboot_stats),
            "adaboot_policy_p_boot": policy_p_boot,
        }

    def load_algorithm_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`algorithm_state_dict` with fail-closed validation."""

        if not isinstance(state, Mapping):
            raise ValueError("DreamWaQ algorithm_state must be a mapping")
        saved_mode = str(state.get("adaboot_mode", self.adaboot_mode)).strip().lower()
        if saved_mode != self.adaboot_mode:
            raise ValueError(
                "DreamWaQ checkpoint AdaBoot mode does not match runtime: "
                f"checkpoint={saved_mode!r}, runtime={self.adaboot_mode!r}"
            )
        saved_window = state.get("adaboot_reward_window_size")
        if saved_window is not None and int(saved_window) != int(self.adaboot_reward_window_size):
            raise ValueError(
                "DreamWaQ checkpoint AdaBoot window does not match runtime: "
                f"checkpoint={saved_window!r}, runtime={self.adaboot_reward_window_size!r}"
            )
        raw_window = state.get("adaboot_episode_return_window", [])
        if not isinstance(raw_window, (list, tuple)):
            raise ValueError("DreamWaQ checkpoint AdaBoot return window must be a list")
        values: list[float] = []
        for raw in raw_window:
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("DreamWaQ checkpoint AdaBoot return window is invalid") from exc
            if not math.isfinite(value):
                raise ValueError("DreamWaQ checkpoint AdaBoot return window must be finite")
            values.append(value)
        self._ep_return_window.clear()
        self._ep_return_window.extend(values[-self.adaboot_reward_window_size :])

        partial = state.get("adaboot_partial_returns")
        if partial is not None:
            if not isinstance(partial, torch.Tensor):
                try:
                    partial = torch.as_tensor(partial, dtype=torch.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError("DreamWaQ checkpoint partial returns are invalid") from exc
            partial = partial.to(device=self.device, dtype=torch.float32).reshape(-1)
            if (
                self._ep_partial_returns is not None
                and partial.numel() != self._ep_partial_returns.numel()
            ):
                raise ValueError(
                    "DreamWaQ checkpoint partial-return width does not match runtime env count: "
                    f"checkpoint={partial.numel()}, runtime={self._ep_partial_returns.numel()}"
                )
            if not bool(torch.all(torch.isfinite(partial))):
                raise ValueError("DreamWaQ checkpoint partial returns must be finite")
            self._ep_partial_returns = partial.clone()
        elif self._ep_partial_returns is not None:
            self._ep_partial_returns.zero_()

        raw_stats = state.get("adaboot_last_stats", {})
        if raw_stats is not None:
            if not isinstance(raw_stats, Mapping):
                raise ValueError("DreamWaQ checkpoint AdaBoot stats must be a mapping")
            restored_stats: dict[str, float] = {}
            for key in ("p_boot", "r_mean", "r_std", "cv_r", "window_count"):
                if key not in raw_stats:
                    continue
                try:
                    value = float(raw_stats[key])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"DreamWaQ checkpoint AdaBoot stat {key!r} is invalid"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"DreamWaQ checkpoint AdaBoot stat {key!r} must be finite")
                restored_stats[key] = value
            self._last_adaboot_stats.update(restored_stats)

        p_boot = state.get("adaboot_policy_p_boot")
        setter = getattr(self.policy, "set_adaboot_p_boot", None)
        if p_boot is not None and callable(setter):
            if isinstance(p_boot, torch.Tensor):
                setter(p_boot.to(device=self.device))
            else:
                try:
                    setter(float(p_boot))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("DreamWaQ checkpoint AdaBoot p_boot is invalid") from exc

    def update(self, beta: float = 1.0) -> dict[str, float]:
        # ``beta`` is accepted for source PPODreamWaq compatibility.  The
        # compact owner keeps its configured adaptation weighting explicit and
        # does not reinterpret a late caller scalar.
        del beta
        if self.storage is None:
            raise RuntimeError("DreamWaQ update() requires init_storage() first")
        sums = {
            "value_loss": 0.0,
            "surrogate_loss": 0.0,
            "entropy": 0.0,
            "recon_loss": 0.0,
            "velocity_loss": 0.0,
            "kl_loss": 0.0,
            # Keep the source runner's AdaBoot metric names.  ``p_boot`` is
            # additive and makes it possible to distinguish the reward-CV
            # source coefficient from the post-model alpha (hybrid/uncertainty).
            "adaboot_coef": 0.0,
            "adaboot_p_boot": 0.0,
            "adaboot_r_mean": 0.0,
            "adaboot_r_std": 0.0,
            "adaboot_cv_r": 0.0,
        }
        with torch.no_grad():
            p_boot, r_mean, r_std, cv_r = self._compute_p_boot_from_episode_rewards()
        self._set_adaboot_p_boot(p_boot)
        updates = self.num_learning_epochs * self.num_mini_batches
        for batch in self.storage.batches(self.num_mini_batches, self.num_learning_epochs):
            obs = batch["observations"]
            assert isinstance(obs, torch.Tensor)
            critic = batch["critic_observations"]
            assert isinstance(critic, torch.Tensor)
            # Source PPODreamWaq publishes the coefficient immediately before
            # each policy update.  Repeating the setter per minibatch keeps the
            # contract robust to policy implementations that clear transient
            # AdaBoot state during ``act``.
            self._set_adaboot_p_boot(p_boot)
            self.policy.act(obs)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    old_mu = batch["mu"]
                    old_sigma = batch["sigma"]
                    assert isinstance(old_mu, torch.Tensor) and isinstance(old_sigma, torch.Tensor)
                    sigma = self.policy.action_std
                    mu = self.policy.action_mean
                    kl = torch.sum(
                        torch.log(sigma / old_sigma + 1.0e-5)
                        + (old_sigma.square() + (old_mu - mu).square()) / (2.0 * sigma.square())
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > float(self.desired_kl) * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif kl > 0.0 and kl < float(self.desired_kl) / 2.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            log_prob = self.policy.get_actions_log_prob(batch["actions"])  # type: ignore[arg-type]
            value = self.policy.evaluate(critic)
            ratio = torch.exp(log_prob - batch["log_probs"].squeeze(-1))  # type: ignore[union-attr]
            adv = batch["advantages"].squeeze(-1)  # type: ignore[union-attr]
            surrogate = torch.max(
                -adv * ratio, -adv * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
            ).mean()
            returns = batch["returns"]  # type: ignore[assignment]
            if self.use_clipped_value_loss:
                old_value = batch["values"]  # type: ignore[assignment]
                value_clipped = old_value + (value - old_value).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (value - returns).pow(2), (value_clipped - returns).pow(2)
                ).mean()  # type: ignore[operator]
            else:
                value_loss = (value - returns).pow(2).mean()  # type: ignore[operator]
            loss = (
                surrogate
                + self.value_loss_coef * value_loss
                - self.entropy_coef * self.policy.entropy.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            next_critic = batch["next_critic_observations"]
            assert isinstance(next_critic, torch.Tensor)
            recon = velocity = kl = torch.zeros((), device=self.device)
            for _ in range(max(self.adaptation_substeps, 1)):
                self.cenet_optimizer.zero_grad()
                (
                    _,
                    code_vel,
                    _code_latent,
                    decode,
                    mean_vel,
                    logvar_vel,
                    mean_latent,
                    logvar_latent,
                ) = self.policy.cenet_forward(obs, sample=True)
                # The pinned source supervises CENet velocity from the
                # current observation's critic tail.  Reconstruction alone
                # targets the next policy frame; taking velocity from
                # ``next_critic`` shifts that loss one control step forward.
                vel_target = self._velocity_target(critic)
                # The source DreamWaQ update reconstructs the *next* policy
                # frame emitted by the environment.  Compact Wheelbipe
                # critics carry that frame as their first 28 values, followed
                # by the privileged velocity/height target; using the current
                # history tail here would train a stale autoencoder target.
                target = next_critic[:, : self.policy.num_one_step_obs].detach()
                dones = batch.get("dones")
                valid = (
                    ~dones.reshape(-1).bool()
                    if isinstance(dones, torch.Tensor)
                    else torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
                )
                if not torch.any(valid):
                    # Match the source fallback for an all-terminal
                    # minibatch: retain a finite adaptation update rather
                    # than producing an empty reduction.
                    valid = torch.ones_like(valid, dtype=torch.bool)
                recon = (decode[valid] - target[valid]).pow(2).mean()
                velocity = (code_vel[valid] - vel_target[valid]).pow(2).mean()
                # Match source DreamWaQ's per-sample KL sum before the
                # minibatch mean; summing latent dimensions matters when
                # latent_dim is changed by an owner configuration.
                kl = (
                    -0.5
                    * (
                        1
                        + logvar_latent[valid]
                        - mean_latent[valid].pow(2)
                        - logvar_latent[valid].exp()
                    )
                    .sum(dim=-1)
                    .mean()
                )
                (recon + velocity + self.kl_weight * kl).backward()
                nn.utils.clip_grad_norm_(list(self.policy.cenet_parameters()), self.max_grad_norm)
                self.cenet_optimizer.step()
            sums["value_loss"] += float(value_loss.item())
            sums["surrogate_loss"] += float(surrogate.item())
            sums["entropy"] += float(self.policy.entropy.mean().item())
            sums["recon_loss"] += float(recon.item())
            sums["velocity_loss"] += float(velocity.item())
            sums["kl_loss"] += float(kl.item())
            sums["adaboot_coef"] += self._adaboot_alpha_metric(p_boot)
            sums["adaboot_p_boot"] += float(p_boot.item())
            sums["adaboot_r_mean"] += float(r_mean.item())
            sums["adaboot_r_std"] += float(r_std.item())
            sums["adaboot_cv_r"] += float(cv_r.item())
        self.storage.clear()
        return {key: value / max(updates, 1) for key, value in sums.items()}


class NP3O:
    """NP3O constrained PPO with cost critics and a Barlow representation loss."""

    def __init__(
        self,
        policy: Any,
        k_value: Any | None = None,
        *source_args: Any,
        device: str = "cpu",
        **cfg: Any,
    ) -> None:
        # The upstream NP3O constructor exposes the standard PPO fields
        # positionally after ``k_value``.  Parse that public order here so a
        # source config/checkpoint can call the owner directly; compact
        # keyword callers retain their existing behavior.
        source_fields = (
            "num_learning_epochs",
            "num_mini_batches",
            "clip_param",
            "gamma",
            "lam",
            "value_loss_coef",
            "cost_value_loss_coef",
            "cost_viol_loss_coef",
            "entropy_coef",
            "learning_rate",
            "max_grad_norm",
            "use_clipped_value_loss",
            "schedule",
            "desired_kl",
            "device",
            "dagger_update_freq",
            "priv_reg_coef_schedual",
            "multi_gpu_cfg",
        )
        if len(source_args) > len(source_fields):
            raise TypeError(
                "too many positional arguments for source NP3O constructor: "
                f"got {len(source_args) + 2}"
            )
        for field, value in zip(source_fields, source_args):
            if field == "device":
                if device != "cpu" and str(device) != str(value):
                    raise ValueError(
                        "source NP3O device conflicts between positional and named arguments"
                    )
                device = str(value)
                continue
            if field in cfg and cfg[field] != value:
                raise ValueError(
                    f"source NP3O {field} conflicts between positional and named arguments"
                )
            cfg.setdefault(field, value)

        self.policy, self.device = policy, device
        self.source_mode = bool(getattr(policy, "source_architecture", False))
        configured_costs = int(getattr(policy, "num_costs", 0))
        if configured_costs != 5:
            raise ValueError(
                "NP3O requires exactly five constraint channels; "
                f"policy.num_costs={configured_costs} is invalid"
            )
        if "num_costs" in cfg:
            try:
                configured_override = int(cfg["num_costs"])
            except (TypeError, ValueError) as exc:
                raise ValueError("NP3O num_costs must be exactly five") from exc
            if configured_override != 5:
                raise ValueError(
                    "NP3O requires exactly five constraint channels; "
                    f"num_costs={configured_override} is invalid"
                )
        self.policy.to(device)
        self.optimizer = optim.Adam(policy.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))
        # Source NP3O applies the optional Barlow/imitation objective through
        # the single policy optimizer.  The compact owner historically used a
        # second representation optimizer; preserve that behavior only for
        # compact mode so source moments and update ordering remain faithful.
        self.representation_optimizer = None
        if not self.source_mode:
            representation_parameters = getattr(policy, "representation_parameters", None)
            if not callable(representation_parameters):
                raise ValueError("compact NP3O policy must expose representation_parameters()")
            self.representation_optimizer = optim.Adam(
                cast(Callable[[], Any], representation_parameters)(),
                lr=float(cfg.get("representation_learning_rate", cfg.get("learning_rate", 3e-4))),
            )
        self.num_learning_epochs, self.num_mini_batches = (
            int(cfg.get("num_learning_epochs", 1)),
            int(cfg.get("num_mini_batches", 1)),
        )
        self.clip_param, self.gamma, self.lam = (
            float(cfg.get("clip_param", 0.2)),
            float(cfg.get("gamma", 0.99)),
            float(cfg.get("lam", 0.95)),
        )
        self.learning_rate = float(cfg.get("learning_rate", 3e-4))
        self.schedule = str(cfg.get("schedule", "fixed"))
        self.desired_kl = cfg.get("desired_kl", None)
        self.value_loss_coef, self.cost_value_loss_coef = (
            float(cfg.get("value_loss_coef", 1.0)),
            float(cfg.get("cost_value_loss_coef", 1.0)),
        )
        self.use_clipped_value_loss = bool(cfg.get("use_clipped_value_loss", True))
        self.cost_viol_loss_coef, self.entropy_coef = (
            float(cfg.get("cost_viol_loss_coef", 1.0)),
            float(cfg.get("entropy_coef", 0.0)),
        )
        # Source NP3O accepts ``k_value`` as the second positional argument;
        # UniLab configs historically named the same value
        # ``cost_k_initial``.  Positional source input takes precedence while
        # preserving the config aliases for existing callers.
        raw_k = (
            k_value if k_value is not None else cfg.get("cost_k_initial", cfg.get("k_value", 1.0))
        )
        self.k_value = torch.as_tensor(raw_k, device=device, dtype=torch.float32).reshape(-1)
        if self.k_value.numel() == 1:
            self.k_value = self.k_value.repeat(policy.num_costs)
        if self.k_value.numel() != policy.num_costs:
            raise ValueError(
                "cost_k_initial must contain exactly num_costs values; "
                f"got {self.k_value.numel()} for num_costs={policy.num_costs}"
            )
        self.cost_d_values = torch.as_tensor(
            cfg.get("cost_d_values", [0.0] * policy.num_costs),
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
        if self.cost_d_values.numel() == 1:
            self.cost_d_values = self.cost_d_values.repeat(policy.num_costs)
        if self.cost_d_values.numel() != policy.num_costs:
            raise ValueError(
                "cost_d_values must contain exactly num_costs values; "
                f"got {self.cost_d_values.numel()} for num_costs={policy.num_costs}"
            )
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.dagger_update_freq = int(cfg.get("dagger_update_freq", 20))
        if self.dagger_update_freq <= 0:
            raise ValueError("NP3O dagger_update_freq must be positive")
        self.priv_reg_coef_schedual = list(
            cfg.get("priv_reg_coef_schedual", cfg.get("priv_reg_coef_schedule", [0.0, 0.0, 0.0]))
        )
        self.imi_flag = bool(getattr(policy, "imi_flag", False))
        if "imi_flag" in cfg:
            if not isinstance(cfg["imi_flag"], bool):
                raise ValueError("NP3O imi_flag must be boolean")
            self.imi_flag = bool(cfg["imi_flag"])
        self.imi_weight = float(cfg.get("imi_weight", 1.0))
        if not math.isfinite(self.imi_weight) or self.imi_weight < 0.0:
            raise ValueError("NP3O imi_weight must be finite and non-negative")
        multi_gpu_cfg = cfg.get("multi_gpu_cfg")
        self.is_multi_gpu = isinstance(multi_gpu_cfg, Mapping)
        if self.is_multi_gpu:
            # Keep the narrowing explicit for static type checkers; the
            # runtime contract remains that only a mapping enables the
            # distributed settings.
            assert isinstance(multi_gpu_cfg, Mapping)
            self.gpu_global_rank = int(multi_gpu_cfg.get("global_rank", 0))
            self.gpu_world_size = int(multi_gpu_cfg.get("world_size", 1))
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        self.cost_limits = torch.as_tensor(
            cfg.get("cost_limits", [0.0] * policy.num_costs), device=device, dtype=torch.float32
        ).reshape(-1)
        if self.cost_limits.numel() == 1:
            self.cost_limits = self.cost_limits.repeat(policy.num_costs)
        if self.cost_limits.numel() != policy.num_costs:
            raise ValueError(
                f"cost_limits must contain exactly five values; got {self.cost_limits.numel()}"
            )
        if not bool(torch.all(torch.isfinite(self.cost_limits))) or bool(
            torch.any(self.cost_limits < 0.0)
        ):
            raise ValueError("cost_limits must be finite and non-negative")
        self.storage: CustomRolloutStorage | None = None
        # Keep both the compact private mapping and the source runner's
        # public ``transition`` attribute.  The upstream NP3O runner exposes
        # a mutable Transition object for lifecycle/debug integrations; an
        # alias here gives callers an explicit view of the current payload
        # without introducing a second storage protocol.
        self._transition: dict[str, torch.Tensor] = {}
        self.transition = self._transition

    def init_storage(
        self,
        num_envs: int,
        num_steps: int,
        actor_dim: Any,
        critic_dim: Any,
        action_dim: Any,
        device: Any = "cpu",
        num_costs: Any = 5,
        *,
        cost_d_values: Any = None,
    ) -> None:
        """Initialize compact or upstream ``RolloutStorageWithCost`` shapes.

        UniLab's runner passes integer widths followed by ``device`` and the
        cost count.  The source ``OnConstraintPolicyRunner`` passes one-item
        shape lists followed by ``cost_shape`` and ``cost_d_values``.  Both
        forms describe the same tensor storage; accepting the source form at
        this owner boundary makes a source runner/checkpoint reusable without
        weakening the five-channel contract.
        """

        source_shape_call = isinstance(device, (Sequence, torch.Size)) and not isinstance(
            device, (str, bytes)
        )
        if source_shape_call:
            actor_width = _shape_width(actor_dim, label="actor observation")
            critic_width = _shape_width(critic_dim, label="critic observation")
            action_width = _shape_width(action_dim, label="action")
            cost_width = _shape_width(device, label="cost")
            source_d_values = num_costs if cost_d_values is None else cost_d_values
            storage_device = self.device
            configured_num_costs = cost_width
        else:
            actor_width = _shape_width(actor_dim, label="actor observation")
            critic_width = _shape_width(critic_dim, label="critic observation")
            action_width = _shape_width(action_dim, label="action")
            storage_device = str(device)
            configured_num_costs = int(num_costs)
            source_d_values = cost_d_values

        if configured_num_costs != int(self.policy.num_costs) or configured_num_costs != 5:
            raise ValueError(
                "NP3O rollout storage requires exactly five constraint channels; "
                f"got num_costs={configured_num_costs}, policy.num_costs={self.policy.num_costs}"
            )
        if source_d_values is not None:
            try:
                d_values = torch.as_tensor(
                    source_d_values, device=self.device, dtype=torch.float32
                ).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError("NP3O cost_d_values must contain five numeric values") from exc
            if d_values.numel() != int(self.policy.num_costs):
                raise ValueError(
                    f"NP3O cost_d_values must contain exactly five values; got {d_values.numel()}"
                )
            if not bool(torch.all(torch.isfinite(d_values))):
                raise ValueError("NP3O cost_d_values must be finite")
            self.cost_d_values = d_values
        self.storage = CustomRolloutStorage(
            num_envs,
            num_steps,
            actor_width,
            critic_width,
            action_width,
            storage_device,
            num_costs=configured_num_costs,
            cost_d_values=self.cost_d_values,
        )

    def test_mode(self) -> None:
        """Switch the actor/critics to evaluation mode (source API parity)."""

        self.policy.eval()

    def train_mode(self) -> None:
        """Switch the actor/critics to training mode (source API parity)."""

        self.policy.train()

    def set_imi_flag(self, flag: bool) -> None:
        """Source NP3O compatibility hook for enabling imitation loss."""

        if not isinstance(flag, bool):
            raise ValueError("NP3O imi_flag must be boolean")
        self.imi_flag = flag

    def set_imi_weight(self, value: float) -> None:
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("NP3O imi_weight must be finite and non-negative") from exc
        if not math.isfinite(converted) or converted < 0.0:
            raise ValueError("NP3O imi_weight must be finite and non-negative")
        self.imi_weight = converted

    def act(self, observations: torch.Tensor, critic_observations: torch.Tensor) -> torch.Tensor:
        assert self.storage is not None
        action = self.policy.act(observations).detach()
        value = self.policy.evaluate(critic_observations).detach()
        cost_value = self.policy.evaluate_cost(critic_observations).detach()
        self._transition = {
            "observations": observations.detach(),
            "critic_observations": critic_observations.detach(),
            "actions": action,
            "values": value,
            "cost_values": cost_value,
            "log_probs": self.policy.get_actions_log_prob(action).detach(),
            "mu": self.policy.action_mean.detach(),
            "sigma": self.policy.action_std.detach(),
        }
        self.transition = self._transition
        return action

    def process_env_step(
        self,
        next_critic_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Mapping[str, Any] | None,
    ) -> None:
        # The upstream ``OnConstraintPolicyRunner`` invokes this method as
        # ``process_env_step(rewards, costs, dones, extras)``.  In that call
        # the first tensor is rank-1 and the second is the five-channel cost
        # matrix; detect that unambiguous public shape and route through the
        # source-order adapter below.  The compact runner's first tensor is a
        # rank-2 critic observation and therefore remains on the native path.
        if (
            isinstance(next_critic_observations, torch.Tensor)
            and isinstance(rewards, torch.Tensor)
            and next_critic_observations.ndim == 1
            and rewards.ndim == 2
            and rewards.shape[-1] == int(self.policy.num_costs)
        ):
            self.process_env_step_source(
                next_critic_observations,
                rewards,
                dones,
                extras,
            )
            return
        assert self.storage is not None
        if extras is None:
            extras = {}
        if not isinstance(extras, Mapping):
            raise TypeError("NP3O extras must be a mapping")
        costs = extras.get("costs")
        if costs is None:
            raise ValueError(
                "NP3O environment step must provide five cost channels in extras['costs']"
            )
        elif not isinstance(costs, torch.Tensor):
            costs = torch.as_tensor(costs, device=self.device, dtype=rewards.dtype)
        else:
            costs = costs.to(device=self.device)
        if costs.ndim == 1 and int(self.policy.num_costs) == 1:
            costs = costs.unsqueeze(1)
        expected_cost_shape = (int(rewards.shape[0]), int(self.policy.num_costs))
        if tuple(costs.shape) != expected_cost_shape:
            raise ValueError(
                "NP3O extras['costs'] must have shape "
                f"{expected_cost_shape}, got {tuple(costs.shape)}"
            )
        self._transition.update(
            next_critic_observations=next_critic_observations.detach().clone(),
            rewards=rewards.detach().clone(),
            dones=dones.detach(),
            costs=costs.detach().clone(),
        )
        _apply_timeout_bootstrap(
            self._transition,
            policy=self.policy,
            gamma=self.gamma,
            extras=extras,
            device=self.device,
            cost_evaluator=self.policy.evaluate_cost,
        )
        self.storage.add(self._transition)
        self._transition = {}
        self.transition = self._transition
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(dones)

    def process_env_step_source(
        self,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        extras: Mapping[str, Any] | None = None,
        next_critic_observations: torch.Tensor | None = None,
    ) -> None:
        """Record the upstream ``(rewards, costs, dones, extras)`` order."""

        if self.storage is None:
            raise RuntimeError("NP3O process_env_step_source() requires init_storage() first")
        if not isinstance(self._transition, dict) or not self._transition:
            raise RuntimeError("NP3O process_env_step_source() requires act() first")
        if next_critic_observations is None:
            next_critic_observations = self._transition.get("critic_observations")
        if not isinstance(next_critic_observations, torch.Tensor):
            raise ValueError("source NP3O requires a next critic/on_constraint observation")
        if extras is None:
            extras = {}
        if not isinstance(extras, Mapping):
            raise TypeError("source NP3O extras must be a mapping")
        if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1:
            raise ValueError(
                "source NP3O rewards must have shape (num_envs,), "
                f"got {getattr(rewards, 'shape', None)}"
            )
        if not isinstance(dones, torch.Tensor) or dones.ndim != 1:
            raise ValueError(
                "source NP3O dones must have shape (num_envs,), "
                f"got {getattr(dones, 'shape', None)}"
            )
        if not isinstance(costs, torch.Tensor):
            costs = torch.as_tensor(costs, device=self.device, dtype=rewards.dtype)
        else:
            costs = costs.to(device=self.device, dtype=rewards.dtype)
        expected_cost_shape = (int(rewards.shape[0]), int(self.policy.num_costs))
        if tuple(costs.shape) != expected_cost_shape:
            raise ValueError(
                f"source NP3O costs must have shape {expected_cost_shape}, got {tuple(costs.shape)}"
            )
        rewards = rewards.to(device=self.device)
        dones = dones.to(device=self.device)
        next_critic_observations = next_critic_observations.to(device=self.device)
        self._transition.update(
            next_critic_observations=next_critic_observations.detach().clone(),
            rewards=rewards.detach().clone(),
            dones=dones.detach(),
            costs=costs.detach().clone(),
        )
        _apply_timeout_bootstrap(
            self._transition,
            policy=self.policy,
            gamma=self.gamma,
            extras=extras,
            device=self.device,
            cost_evaluator=self.policy.evaluate_cost,
        )
        self.storage.add(self._transition)
        self._transition = {}
        self.transition = self._transition
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(dones)

    def compute_returns(self, critic_observations: torch.Tensor) -> None:
        assert self.storage is not None
        self.storage.compute_returns(
            self.policy.evaluate(critic_observations).detach(), self.gamma, self.lam
        )
        self.storage.compute_cost_returns(
            self.policy.evaluate_cost(critic_observations).detach(), self.gamma, self.lam
        )

    def compute_cost_returns(self, critic_observations: torch.Tensor) -> None:
        """Compute NP3O cost returns through the upstream public API.

        ``OnConstraintPolicyRunner`` calls ``compute_returns`` and
        ``compute_cost_returns`` as two lifecycle steps.  The UniLab runner
        computes both in ``compute_returns`` for convenience, but omitting
        this source-named method made a source-style caller fail after an
        otherwise valid rollout.  Recomputing the cost targets is
        idempotent and keeps both entrypoint contracts explicit.
        """

        if self.storage is None:
            raise RuntimeError("NP3O compute_cost_returns() requires init_storage() first")
        self.storage.compute_cost_returns(
            self.policy.evaluate_cost(critic_observations).detach(), self.gamma, self.lam
        )

    def update(self) -> dict[str, float]:
        assert self.storage is not None
        sums = {
            "value_loss": 0.0,
            "surrogate_loss": 0.0,
            "cost_value_loss": 0.0,
            "violation_loss": 0.0,
            "barlow_loss": 0.0,
            "imitation_loss": 0.0,
            "entropy": 0.0,
        }
        updates = self.num_learning_epochs * self.num_mini_batches
        for batch in self.storage.batches(self.num_mini_batches, self.num_learning_epochs):
            obs, critic = batch["observations"], batch["critic_observations"]
            assert isinstance(obs, torch.Tensor) and isinstance(critic, torch.Tensor)
            self.policy.act(obs)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    old_mu = batch["mu"]
                    old_sigma = batch["sigma"]
                    assert isinstance(old_mu, torch.Tensor) and isinstance(old_sigma, torch.Tensor)
                    sigma = self.policy.action_std
                    mu = self.policy.action_mean
                    kl = torch.sum(
                        torch.log(sigma / old_sigma + 1.0e-5)
                        + (old_sigma.square() + (old_mu - mu).square()) / (2.0 * sigma.square())
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > float(self.desired_kl) * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif kl > 0.0 and kl < float(self.desired_kl) / 2.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            log_prob = self.policy.get_actions_log_prob(batch["actions"])  # type: ignore[arg-type]
            ratio = torch.exp(log_prob - batch["log_probs"].squeeze(-1))  # type: ignore[union-attr]
            adv = batch["advantages"].squeeze(-1)  # type: ignore[union-attr]
            surrogate = torch.max(
                -adv * ratio, -adv * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
            ).mean()
            value = self.policy.evaluate(critic)
            returns = batch["returns"]
            if self.use_clipped_value_loss:
                old_value = batch["values"]
                value_clipped = old_value + (value - old_value).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (value - returns).pow(2), (value_clipped - returns).pow(2)
                ).mean()  # type: ignore[operator]
            else:
                value_loss = (value - returns).pow(2).mean()  # type: ignore[operator]
            cost_value = self.policy.evaluate_cost(critic)
            cost_returns = batch.get("cost_returns")
            cost_adv = batch.get("cost_advantages")
            cost_value_loss = (
                (
                    torch.maximum(
                        (cost_value - cost_returns).pow(2),
                        (
                            batch["cost_values"]
                            + (cost_value - batch["cost_values"]).clamp(
                                -self.clip_param, self.clip_param
                            )
                            - cost_returns
                        ).pow(2),
                    ).mean()
                    if self.use_clipped_value_loss
                    else (cost_value - cost_returns).pow(2).mean()
                )
                if isinstance(cost_returns, torch.Tensor)
                else torch.zeros((), device=self.device)
            )
            violation = torch.zeros((), device=self.device)
            cost_violation = batch.get("cost_violation")
            if isinstance(cost_adv, torch.Tensor) and isinstance(cost_violation, torch.Tensor):
                # Source NP3O combines the clipped cost surrogate with the
                # precomputed normalized constraint violation per channel,
                # then applies the channel-specific k-value schedule.
                cost_ratio = ratio.unsqueeze(-1)
                cost_surrogate = torch.maximum(
                    cost_adv * cost_ratio,
                    cost_adv * cost_ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
                ).mean(0)
                channel_violation = cost_surrogate + cost_violation.mean(0)
                violation = torch.sum(self.k_value * torch.relu(channel_violation))
            loss = (
                surrogate
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                + self.cost_viol_loss_coef * violation
                - self.entropy_coef * self.policy.entropy.mean()
            )
            imitation_loss = torch.zeros((), device=self.device)
            imitation_fn = getattr(self.policy, "imitation_learning_loss", None)
            if self.source_mode and self.imi_flag and callable(imitation_fn):
                # Source NP3O adds Barlow/velocity imitation to the same loss
                # optimized by the policy Adam; this is intentionally not a
                # second representation step.
                imitation_loss = imitation_fn(obs, self.imi_weight)
                if not isinstance(imitation_loss, torch.Tensor) or imitation_loss.ndim != 0:
                    raise ValueError(
                        "source NP3O imitation_learning_loss must return a scalar tensor"
                    )
                loss = loss + imitation_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            barlow = imitation_loss if self.source_mode else torch.zeros((), device=self.device)
            if not self.source_mode:
                assert self.representation_optimizer is not None
                self.representation_optimizer.zero_grad()
                z_s = self.policy.latent(obs)
                z_t = self.policy.latent(obs, target=True)
                c = (z_s.T @ z_t) / max(z_s.shape[0], 1)
                barlow = (torch.diagonal(c) - 1).pow(2).mean() + (
                    c - torch.diag(torch.diagonal(c))
                ).pow(2).mean()
                barlow.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.representation_parameters()), self.max_grad_norm
                )
                self.representation_optimizer.step()
            sums["value_loss"] += float(value_loss.item())
            sums["surrogate_loss"] += float(surrogate.item())
            sums["cost_value_loss"] += float(cost_value_loss.item())
            sums["violation_loss"] += float(violation.item())
            sums["barlow_loss"] += float(barlow.item())
            sums["imitation_loss"] += float(imitation_loss.item())
            sums["entropy"] += float(self.policy.entropy.mean().item())
        self.storage.clear()
        return {key: value / max(updates, 1) for key, value in sums.items()}

    def update_k_value(self, iteration: int) -> torch.Tensor:
        """Apply the source NP3O warm-up schedule and return channel weights."""

        self.k_value = torch.minimum(
            torch.ones_like(self.k_value), self.k_value * (1.0004 ** int(iteration))
        )
        return self.k_value


# Source WheelBipe spelling for the DreamWaQ optimizer/runner.
PPODreamWaq = DreamWaQPPO

__all__ = ["DreamWaQPPO", "PPODreamWaq", "HIMPPO", "NP3O"]
