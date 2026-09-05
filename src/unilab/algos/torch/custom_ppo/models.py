"""Small, self-contained Torch policies for WheelBipe custom PPO variants."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from unilab.algos.torch.custom_ppo.source_barlow import SourceBarlowTwinsActorCritic

# Keep the source DreamWaQ spelling as a closed set.  AdaBoot is optional, but
# accepting an unknown mode and only failing when a rollout happens would make
# a misspelled non-off configuration silently use the wrong representation
# path.  The algorithm owner imports this constant for the matching runtime
# validation.
ADABOOT_MODES = frozenset({"off", "reward_cv", "uncertainty", "hybrid"})


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation={name!r}")


def _mlp(in_dim: int, out_dim: int, hidden: Sequence[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    for width in hidden:
        layers.extend((nn.Linear(current, int(width)), _activation(activation)))
        current = int(width)
    layers.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*layers)


class _GaussianPolicy(nn.Module):
    """Shared distribution/value plumbing for latent PPO policies."""

    is_recurrent = False

    def __init__(
        self,
        num_actions: int,
        init_noise_std: float,
        *,
        noise_std_type: str = "scalar",
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.noise_std_type = str(noise_std_type).lower()
        if not math.isfinite(float(init_noise_std)) or float(init_noise_std) <= 0.0:
            raise ValueError("init_noise_std must be finite and positive")
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(torch.full((self.num_actions,), float(init_noise_std)))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.full((self.num_actions,), math.log(init_noise_std)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {noise_std_type!r}; expected 'scalar' or 'log'"
            )
        self.distribution: Normal | None = None

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("act() must be called before action_mean")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("act() must be called before action_std")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("act() must be called before entropy")
        return self.distribution.entropy().sum(dim=-1)

    def _set_distribution(self, mean: torch.Tensor) -> None:
        mean = torch.nan_to_num(mean, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if self.noise_std_type == "scalar":
            std_parameter = self.std
        else:
            std_parameter = torch.exp(self.log_std)
        std = torch.nan_to_num(std_parameter, nan=1.0, posinf=10.0, neginf=1e-6).clamp_min(1e-6)
        self.distribution = Normal(mean, std.expand_as(mean))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("act() must be called before get_actions_log_prob")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_std(self) -> torch.Tensor:
        """Return the current action standard deviation (source API alias)."""

        if self.noise_std_type == "scalar":
            return self.std
        return torch.exp(self.log_std)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def test(self) -> None:
        """Source-policy compatibility alias for switching to eval mode."""

        self.eval()


def _resolve_observation(value: Any, keys: tuple[str, ...], label: str) -> torch.Tensor:
    """Resolve tensor or mapping-style source observations at the model edge."""

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
    raise TypeError(f"Custom Wheelbipe {label} must be a tensor or mapping containing {keys}")


def _resolve_history_and_policy(value: Any, one_step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve source ``policy_hist`` plus its current ``policy`` frame.

    Source WheelBipe adapters carry both streams in a mapping.  Tensor-only
    UniLab callers provide the flattened history, whose newest frame is the
    actor frame.  Keeping this distinction at the model boundary avoids
    accidentally training on a stale history prefix when both keys exist.
    """

    history = _resolve_observation(
        value, ("policy_hist", "actor_history", "history", "actor", "policy"), "actor input"
    )
    current: torch.Tensor | None = None
    if isinstance(value, Mapping):
        candidate = value.get("policy")
        if isinstance(candidate, torch.Tensor):
            current = candidate
    if current is None:
        try:
            candidate = value["policy"]
        except (AttributeError, KeyError, TypeError, IndexError):
            candidate = None
        if isinstance(candidate, torch.Tensor):
            current = candidate
    if current is None:
        current = history[:, -int(one_step) :]
    return history, current


class DreamWaQActorCritic(_GaussianPolicy):
    """DreamWaQ-style CENet policy with reconstruction/KL adaptation losses.

    ``obs_history`` is a flattened ``history x one_step_obs`` tensor.  The
    deterministic inference path keeps the same interface as the upstream
    implementation while the critic receives the privileged observation.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *source_args: Any,
        num_actor_history: int = 1,
        num_estimate: int = 3,
        latent_dim: int = 16,
        cenet_in_dim: int | None = None,
        cenet_out_dim: int | None = None,
        actor_hidden_dims: Sequence[int] = (256, 128, 64),
        critic_hidden_dims: Sequence[int] = (256, 128, 64),
        encoder_hidden_dims: Sequence[int] = (128, 64),
        decoder_hidden_dims: Sequence[int] = (64, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        logvar_clip: float = 10.0,
        feature_clip: float = 50.0,
        # Upstream DreamWaQ names these fields ``cenet_*`` and passes
        # ``num_estimate, cenet_in_dim, cenet_out_dim`` positionally.  Keep
        # those spellings as additive aliases while retaining UniLab's
        # compact owner arguments above.
        cenet_encoder_hidden_dims: Sequence[int] | None = None,
        cenet_decoder_hidden_dims: Sequence[int] | None = None,
        cenet_logvar_clip: float | None = None,
        cenet_feature_clip: float | None = None,
        action_mean_clip: float | None = 20.0,
        noise_std_type: str = "scalar",
        **kwargs: Any,
    ) -> None:
        # The upstream constructor exposes ``cenet_in_dim`` and
        # ``cenet_out_dim`` as named parameters as well as positional ones.
        # Keep these aliases explicit instead of letting them fall into
        # ``**kwargs`` (where a source config could otherwise be accepted but
        # silently build a network with the wrong history/code width).
        source_input_dim: int | None = None
        source_code_dim: int | None = None
        if cenet_in_dim is not None:
            source_input_dim = int(cenet_in_dim)
            if source_input_dim <= 0:
                raise ValueError(f"cenet_in_dim must be positive, got {cenet_in_dim!r}")
        if cenet_out_dim is not None:
            source_code_dim = int(cenet_out_dim)
            if source_code_dim <= 0:
                raise ValueError(f"cenet_out_dim must be positive, got {cenet_out_dim!r}")
        if source_args:
            if len(source_args) < 3:
                raise TypeError(
                    "DreamWaQ source constructor expects num_estimate, cenet_in_dim, "
                    "and cenet_out_dim"
                )
            num_estimate = int(source_args[0])
            positional_input_dim = int(source_args[1])
            positional_code_dim = int(source_args[2])
            if source_input_dim is not None and source_input_dim != positional_input_dim:
                raise ValueError(
                    "cenet_in_dim conflicts between positional and named arguments: "
                    f"{positional_input_dim} != {source_input_dim}"
                )
            if source_code_dim is not None and source_code_dim != positional_code_dim:
                raise ValueError(
                    "cenet_out_dim conflicts between positional and named arguments: "
                    f"{positional_code_dim} != {source_code_dim}"
                )
            source_input_dim = positional_input_dim
            source_code_dim = positional_code_dim
            if source_code_dim <= num_estimate:
                raise ValueError(
                    f"cenet_out_dim ({source_code_dim}) must be larger than "
                    f"num_estimate ({num_estimate})"
                )
            latent_dim = source_code_dim - num_estimate
            # Source callers occasionally pass activation as the fourth
            # positional argument.  Honor the complete public source order so
            # a checkpoint/configuration constructor can be reused verbatim.
            source_fields = (
                "activation",
                "cenet_encoder_hidden_dims",
                "cenet_decoder_hidden_dims",
                "init_noise_std",
                "noise_std_type",
                "cenet_logvar_clip",
                "cenet_feature_clip",
                "action_mean_clip",
            )
            if len(source_args) - 3 > len(source_fields):
                raise TypeError(
                    "too many positional arguments for source DreamWaQ constructor: "
                    f"got {len(source_args) + 3}"
                )
            for field, value in zip(source_fields, source_args[3:]):
                if field == "activation":
                    activation = str(value)
                elif field == "cenet_encoder_hidden_dims":
                    cenet_encoder_hidden_dims = value
                elif field == "cenet_decoder_hidden_dims":
                    cenet_decoder_hidden_dims = value
                elif field == "init_noise_std":
                    init_noise_std = float(value)
                elif field == "noise_std_type":
                    noise_std_type = str(value)
                elif field == "cenet_logvar_clip":
                    cenet_logvar_clip = None if value is None else float(value)
                elif field == "cenet_feature_clip":
                    cenet_feature_clip = None if value is None else float(value)
                else:
                    action_mean_clip = None if value is None else float(value)
        if source_code_dim is not None:
            if source_code_dim <= num_estimate:
                raise ValueError(
                    f"cenet_out_dim ({source_code_dim}) must be larger than "
                    f"num_estimate ({num_estimate})"
                )
            latent_dim = source_code_dim - int(num_estimate)
        if cenet_encoder_hidden_dims is not None:
            encoder_hidden_dims = cenet_encoder_hidden_dims
        if cenet_decoder_hidden_dims is not None:
            decoder_hidden_dims = cenet_decoder_hidden_dims
        if cenet_logvar_clip is not None:
            logvar_clip = float(cenet_logvar_clip)
        if cenet_feature_clip is not None:
            feature_clip = float(cenet_feature_clip)
        # Preserve source AdaBoot knobs at the model boundary.  Keep the mode
        # validation here as well as in ``DreamWaQPPO``: direct source-style
        # model callers should receive the same fail-closed diagnostic, and a
        # non-off knob must never be accepted and then ignored.
        self.adaboot_mode = str(kwargs.pop("adaboot_mode", "off")).strip().lower()
        if self.adaboot_mode not in ADABOOT_MODES:
            supported = ", ".join(sorted(ADABOOT_MODES))
            raise ValueError(
                f"Unsupported DreamWaQ adaboot_mode={self.adaboot_mode!r}; "
                f"expected one of {supported}"
            )
        raw_use_adaboot = kwargs.pop("use_adaboot", False)
        if not isinstance(raw_use_adaboot, bool):
            raise ValueError("DreamWaQ use_adaboot must be a boolean")
        if raw_use_adaboot and self.adaboot_mode == "off":
            self.adaboot_mode = "uncertainty"
        raw_temperature = kwargs.pop("adaboot_temperature", 1.0)
        raw_bias = kwargs.pop("adaboot_bias", 0.0)
        raw_min = kwargs.pop("adaboot_min", 0.0)
        raw_max = kwargs.pop("adaboot_max", 1.0)
        if any(isinstance(value, bool) for value in (raw_temperature, raw_bias, raw_min, raw_max)):
            raise ValueError("DreamWaQ AdaBoot numeric policy knobs must not be booleans")
        self.adaboot_temperature = float(raw_temperature)
        self.adaboot_bias = float(raw_bias)
        self.adaboot_min = float(raw_min)
        self.adaboot_max = float(raw_max)
        if not math.isfinite(self.adaboot_temperature):
            raise ValueError("DreamWaQ adaboot_temperature must be finite")
        if not math.isfinite(self.adaboot_bias):
            raise ValueError("DreamWaQ adaboot_bias must be finite")
        if not math.isfinite(self.adaboot_min) or not math.isfinite(self.adaboot_max):
            raise ValueError("DreamWaQ adaboot_min/adaboot_max must be finite")
        if not (0.0 <= self.adaboot_min <= self.adaboot_max <= 1.0):
            raise ValueError(
                "DreamWaQ AdaBoot alpha bounds must satisfy 0 <= adaboot_min "
                f"<= adaboot_max <= 1, got ({self.adaboot_min}, {self.adaboot_max})"
            )
        # Reward-window/CV settings belong to ``DreamWaQPPO`` (the algorithm
        # owner), not the policy.  A source-style config can still flatten
        # unknown fields into this ``**kwargs`` boundary; reject any leftover
        # AdaBoot-prefixed key instead of accepting a knob that has no effect
        # on the model.  Other source compatibility kwargs remain intentionally
        # opaque because they may describe architecture metadata handled by a
        # higher-level owner.
        unknown_adaboot = sorted(str(key) for key in kwargs if str(key).startswith("adaboot_"))
        if unknown_adaboot:
            raise ValueError(
                "Unsupported DreamWaQ AdaBoot policy config key(s): "
                f"{unknown_adaboot!r}; reward-window knobs belong under algo.algorithm"
            )
        self.adaboot_p_boot: torch.Tensor | float | None = None
        self.last_adaboot_alpha: torch.Tensor | None = None
        if int(num_actor_obs) <= 0 or int(num_critic_obs) <= 0 or int(num_actions) <= 0:
            raise ValueError("DreamWaQ observation and action dimensions must be positive")
        if isinstance(num_actor_history, bool) or int(num_actor_history) <= 0:
            raise ValueError("num_actor_history must be a positive integer")
        if isinstance(num_estimate, bool) or int(num_estimate) <= 0:
            raise ValueError("num_estimate must be a positive integer")
        if len(encoder_hidden_dims) == 0 or len(decoder_hidden_dims) == 0:
            raise ValueError("DreamWaQ encoder and decoder hidden dimensions must not be empty")
        super().__init__(num_actions, init_noise_std, noise_std_type=noise_std_type)
        self.num_one_step_obs = int(num_actor_obs)
        self.history_size = int(num_actor_history)
        if source_input_dim is not None:
            if source_input_dim <= 0 or source_input_dim % self.num_one_step_obs != 0:
                raise ValueError(
                    "cenet_in_dim must be a positive multiple of num_actor_obs, got "
                    f"cenet_in_dim={source_input_dim}, num_actor_obs={num_actor_obs}"
                )
            source_history = source_input_dim // self.num_one_step_obs
            if self.history_size == 1:
                self.history_size = source_history
            elif self.history_size != source_history:
                raise ValueError(
                    "num_actor_history does not match source cenet_in_dim: "
                    f"history={self.history_size}, cenet_in_dim={source_input_dim}"
                )
        self.num_actor_obs = self.num_one_step_obs * self.history_size
        self.num_critic_obs = int(num_critic_obs)
        self.num_estimate = int(num_estimate)
        self.latent_dim = int(latent_dim)
        self.code_dim = self.num_estimate + self.latent_dim
        self.logvar_clip = None if logvar_clip is None else float(logvar_clip)
        self.feature_clip = None if feature_clip is None else float(feature_clip)
        self.cenet_logvar_clip = self.logvar_clip
        self.cenet_feature_clip = self.feature_clip
        self.action_mean_clip = None if action_mean_clip is None else float(action_mean_clip)
        self.cenet_in_dim = self.num_actor_obs if source_input_dim is None else source_input_dim
        self.num_latent = self.latent_dim
        self.last_code_vel: torch.Tensor | None = None

        encoder_layers = list(
            _mlp(
                self.cenet_in_dim,
                encoder_hidden_dims[-1],
                encoder_hidden_dims[:-1],
                activation,
            ).children()
        )
        # Source CENet constructs ``MLP(output_dim=None)`` with every listed
        # encoder width treated as a hidden layer.  Its final 64D linear is
        # therefore followed by the configured activation; the equivalent
        # direct Sequential must retain that parameter-free final layer.
        encoder_layers.append(_activation(activation))
        self.encoder = nn.Sequential(*encoder_layers)
        enc_width = int(encoder_hidden_dims[-1])
        self.encode_mean_vel = nn.Linear(enc_width, self.num_estimate)
        self.encode_logvar_vel = nn.Linear(enc_width, self.num_estimate)
        self.encode_mean_latent = nn.Linear(enc_width, self.latent_dim)
        self.encode_logvar_latent = nn.Linear(enc_width, self.latent_dim)
        self.decoder = _mlp(self.code_dim, self.num_one_step_obs, decoder_hidden_dims, activation)
        self.actor = _mlp(
            self.num_one_step_obs + self.code_dim, num_actions, actor_hidden_dims, activation
        )
        self.critic = _mlp(self.num_critic_obs, 1, critic_hidden_dims, activation)

    def _clip(self, value: torch.Tensor, limit: float | None = None) -> torch.Tensor:
        limit = self.feature_clip if limit is None else limit
        if limit is None:
            return torch.nan_to_num(value, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return torch.nan_to_num(value, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)

    def _normalize_history(self, value: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
        """Resolve and validate a source/compact history tensor.

        Source DreamWaQ adapters normally expose ``policy_hist``.  A few
        lightweight adapters (and direct inference callers) expose only the
        current ``policy`` frame; repeating that frame is the same cold-start
        convention used by the custom runner and avoids a late Linear shape
        error.  Any other width is rejected explicitly so a source graph is
        never silently trained against a different history contract.
        """

        history = _resolve_observation(
            value, ("policy_hist", "actor_history", "history", "actor", "policy"), "actor input"
        )
        if history.ndim != 2:
            raise ValueError(
                f"DreamWaQ actor history must be a rank-2 tensor, got shape={tuple(history.shape)}"
            )
        width = int(history.shape[-1])
        if width == self.num_one_step_obs and self.cenet_in_dim != self.num_one_step_obs:
            history = history.repeat(1, self.history_size)
        if int(history.shape[-1]) != self.cenet_in_dim:
            raise ValueError(
                "DreamWaQ actor history width does not match cenet_in_dim: "
                f"expected {self.cenet_in_dim}, got {history.shape[-1]}"
            )
        return self._clip(history)

    def _normalize_critic(self, value: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
        critic = _resolve_observation(
            value, ("critic", "prev_critic", "policy", "actor"), "critic input"
        )
        if critic.ndim != 2 or int(critic.shape[-1]) != self.num_critic_obs:
            raise ValueError(
                "DreamWaQ critic observation width must be "
                f"{self.num_critic_obs}, got shape={tuple(critic.shape)}"
            )
        return self._clip(critic)

    def cenet_forward(self, obs_history: torch.Tensor | Mapping[str, Any], *, sample: bool = True):
        h = self._normalize_history(obs_history)
        hidden = self.encoder(h)
        mean_vel = self._clip(self.encode_mean_vel(hidden))
        logvar_vel = self._clip(self.encode_logvar_vel(hidden), self.logvar_clip)
        mean_latent = self._clip(self.encode_mean_latent(hidden))
        logvar_latent = self._clip(self.encode_logvar_latent(hidden), self.logvar_clip)

        def sample_code(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            if not sample:
                return mean
            return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

        code_vel = sample_code(mean_vel, logvar_vel)
        code_latent = sample_code(mean_latent, logvar_latent)
        code = self._clip(torch.cat((code_vel, code_latent), dim=-1))
        decode = self.decoder(code)
        # Keep the source DreamWaQ return order, including the sampled latent
        # code as a separate element.  Callers that only need the aggregate
        # code can continue to ignore the extra field with ``code, *_``.
        return (
            code,
            code_vel,
            code_latent,
            decode,
            mean_vel,
            logvar_vel,
            mean_latent,
            logvar_latent,
        )

    def _actor_features(
        self,
        obs_history: torch.Tensor | Mapping[str, Any],
        *,
        sample: bool,
        apply_adaboot: bool = False,
        teacher_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history, current = _resolve_history_and_policy(obs_history, self.num_one_step_obs)
        if current.ndim != 2:
            raise ValueError(
                "DreamWaQ current policy observation must be rank-2, "
                f"got shape={tuple(current.shape)}"
            )
        if int(current.shape[-1]) != self.num_one_step_obs:
            if int(current.shape[-1]) == self.cenet_in_dim:
                current = current[:, -self.num_one_step_obs :]
            else:
                raise ValueError(
                    "DreamWaQ current policy width must be "
                    f"{self.num_one_step_obs}, got {current.shape[-1]}"
                )
        (
            code,
            code_vel,
            _code_latent,
            _decode,
            mean_vel,
            logvar_vel,
            _mean_latent,
            _logvar_latent,
        ) = self.cenet_forward(history, sample=sample)
        self.last_code_vel = code_vel.detach()
        if apply_adaboot and self.adaboot_mode != "off":
            # Keep the source's optional velocity blending available for
            # direct model callers.  The compact PPO owner leaves the mode
            # disabled by default, so canonical training is unchanged.
            code_vel = self._adaboot_velocity(
                code_vel,
                mean_vel,
                logvar_vel,
                gt_vel=teacher_velocity,
            )
            code = torch.cat((code_vel, code[..., self.num_estimate :]), dim=-1)
        return torch.cat((self._clip(current), code), dim=-1)

    def _teacher_velocity_from_obs(
        self, obs_history: torch.Tensor | Mapping[str, Any]
    ) -> torch.Tensor | None:
        """Resolve the source critic velocity/height target when available.

        The upstream DreamWaQ actor uses the privileged critic slice only for
        its stochastic training path.  Deterministic ``act_inference``
        intentionally bypasses AdaBoot and must not consume this target.  A
        compact tensor-only caller has no privileged stream, so returning
        ``None`` preserves the source policy's mean-velocity fallback.
        """

        critic: torch.Tensor | None = None
        if isinstance(obs_history, Mapping):
            for key in ("critic", "prev_critic"):
                candidate = obs_history.get(key)
                if isinstance(candidate, torch.Tensor):
                    critic = candidate
                    break
        if critic is None:
            return None
        if critic.ndim != 2:
            raise ValueError(
                f"DreamWaQ AdaBoot critic target must be rank-2, got shape={tuple(critic.shape)}"
            )
        # ``num_actor_obs`` is the flattened CENet input width in the
        # migrated owner (for a source-style constructor this is typically
        # 140 = 5 * 28).  The source critic, however, stores the privileged
        # velocity immediately after the *one-step* policy frame (28), not
        # after the flattened history.  Using ``num_actor_obs`` here made
        # source DreamWaQ AdaBoot silently fall back to the predicted mean
        # velocity because a 32D compact critic never reaches offset 140.
        # Keep the target offset tied to the public one-step observation
        # contract, matching the upstream ``ActorCriticDreamWaq`` helper.
        start = self.num_one_step_obs
        end = start + self.num_estimate
        if int(critic.shape[-1]) < end:
            return None
        return critic[:, start:end]

    def act(self, obs_history: torch.Tensor | Mapping[str, Any], **kwargs: Any) -> torch.Tensor:
        del kwargs
        teacher_velocity = self._teacher_velocity_from_obs(obs_history)
        self.update_distribution(
            self._actor_features(
                obs_history,
                sample=True,
                apply_adaboot=True,
                teacher_velocity=teacher_velocity,
            )
        )
        return self.distribution.sample()  # type: ignore[union-attr]

    def act_inference(self, obs_history: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
        # Match the source deployment path: ``sample=False`` uses the CENet
        # means directly and does not apply AdaBoot/privileged blending.
        return self._clip(
            self.actor(
                self._actor_features(
                    obs_history,
                    sample=False,
                    apply_adaboot=False,
                )
            ),
            self.action_mean_clip,
        )

    def update_distribution(self, observations: torch.Tensor) -> None:
        """Set the action distribution from concatenated actor features."""

        mean = self._clip(self.actor(self._clip(observations)), self.action_mean_clip)
        std = torch.nan_to_num(self.get_std(), nan=1.0, posinf=10.0, neginf=1.0e-6)
        std = std.clamp_min(1.0e-6).expand_as(mean)
        self.distribution = Normal(mean, std)

    def evaluate(self, critic_obs: torch.Tensor | Mapping[str, Any], **kwargs: Any) -> torch.Tensor:
        del kwargs
        return self.critic(self._normalize_critic(critic_obs))

    @staticmethod
    def init_weights(sequential: nn.Sequential, scales: Sequence[float]) -> None:
        """Keep the upstream ``ActorCriticDreamWaq.init_weights`` helper."""

        for index, module in enumerate(
            layer for layer in sequential if isinstance(layer, nn.Linear)
        ):
            if index < len(scales):
                # PyTorch's current type stub narrows ``gain`` to ``int``;
                # the runtime API accepts the source-compatible float scale.
                nn.init.orthogonal_(module.weight, gain=float(scales[index]))  # type: ignore[arg-type]

    def reparameterise(
        self, mean: torch.Tensor, logvar: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """Source spelling retained for callers that use the CENet helper."""

        mean = self._clip(mean)
        logvar = self._clip(logvar, self.logvar_clip)
        if not sample:
            return mean
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def set_adaboot_p_boot(self, p_boot: torch.Tensor | float | None) -> None:
        if p_boot is None:
            self.adaboot_p_boot = None
            return
        if isinstance(p_boot, torch.Tensor):
            if p_boot.numel() == 0:
                raise ValueError("DreamWaQ AdaBoot p_boot cannot be empty")
            if not bool(torch.all(torch.isfinite(p_boot))):
                raise ValueError("DreamWaQ AdaBoot p_boot must be finite")
            if bool(torch.any((p_boot < 0.0) | (p_boot > 1.0))):
                raise ValueError("DreamWaQ AdaBoot p_boot must lie in [0, 1]")
            self.adaboot_p_boot = p_boot.detach()
            return
        try:
            value = float(p_boot)
        except (TypeError, ValueError) as exc:
            raise ValueError("DreamWaQ AdaBoot p_boot must be a finite scalar/tensor") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("DreamWaQ AdaBoot p_boot must lie in [0, 1]")
        self.adaboot_p_boot = value

    def get_adaboot_p_boot(self) -> torch.Tensor | float | None:
        return (
            self.last_adaboot_alpha if self.last_adaboot_alpha is not None else self.adaboot_p_boot
        )

    def _adaboot_velocity(
        self,
        code_vel: torch.Tensor,
        mean_vel: torch.Tensor,
        logvar_vel: torch.Tensor,
        gt_vel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.adaboot_mode == "off":
            self.last_adaboot_alpha = None
            return code_vel
        alpha_unc: torch.Tensor | None = None
        alpha_boot: torch.Tensor | None = None
        if self.adaboot_mode in ("uncertainty", "hybrid"):
            uncertainty = torch.mean(logvar_vel, dim=-1, keepdim=True)
            alpha_unc = torch.sigmoid(-(self.adaboot_temperature * uncertainty + self.adaboot_bias))
        if self.adaboot_mode in ("reward_cv", "hybrid"):
            p_boot = self.adaboot_p_boot
            if p_boot is None:
                alpha_boot = torch.ones(
                    (code_vel.shape[0], 1), device=code_vel.device, dtype=code_vel.dtype
                )
            elif isinstance(p_boot, torch.Tensor):
                alpha_boot = p_boot.to(device=code_vel.device, dtype=code_vel.dtype)
                if alpha_boot.ndim == 0:
                    alpha_boot = alpha_boot.view(1, 1).expand(code_vel.shape[0], 1)
                elif alpha_boot.ndim == 1:
                    alpha_boot = alpha_boot.view(-1, 1)
                elif alpha_boot.ndim != 2:
                    raise ValueError(
                        "DreamWaQ AdaBoot p_boot tensor must be scalar, vector, or "
                        f"(batch, 1); got shape {tuple(alpha_boot.shape)}"
                    )
                if alpha_boot.shape[0] not in (1, code_vel.shape[0]):
                    raise ValueError(
                        "DreamWaQ AdaBoot p_boot batch width does not match code_vel: "
                        f"p_boot={tuple(alpha_boot.shape)}, code_vel={tuple(code_vel.shape)}"
                    )
                if alpha_boot.shape[1] != 1:
                    raise ValueError(
                        "DreamWaQ AdaBoot p_boot tensor must have one coefficient per "
                        f"sample; got shape {tuple(alpha_boot.shape)}"
                    )
                if alpha_boot.shape[0] == 1 and code_vel.shape[0] != 1:
                    alpha_boot = alpha_boot.expand(code_vel.shape[0], -1)
            else:
                alpha_boot = torch.full(
                    (code_vel.shape[0], 1),
                    float(p_boot),
                    device=code_vel.device,
                    dtype=code_vel.dtype,
                )
        if self.adaboot_mode == "uncertainty":
            alpha = alpha_unc
        elif self.adaboot_mode == "reward_cv":
            alpha = alpha_boot
        elif self.adaboot_mode == "hybrid":
            assert alpha_unc is not None and alpha_boot is not None
            alpha = alpha_unc * alpha_boot
        else:
            raise ValueError(f"Unknown adaboot_mode: {self.adaboot_mode}")
        assert alpha is not None
        alpha = alpha.clamp(min=self.adaboot_min, max=self.adaboot_max)
        teacher = mean_vel if gt_vel is None else gt_vel
        self.last_adaboot_alpha = alpha.detach()
        return alpha * code_vel + (1.0 - alpha) * teacher

    def cenet_parameters(self):
        yield from self.encoder.parameters()
        yield from self.encode_mean_vel.parameters()
        yield from self.encode_logvar_vel.parameters()
        yield from self.encode_mean_latent.parameters()
        yield from self.encode_logvar_latent.parameters()
        yield from self.decoder.parameters()


class NP3OActorCritic(_GaussianPolicy):
    """NP3O-style policy with a Barlow-Twins history encoder and cost critic."""

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *source_args: Any,
        num_actor_history: int = 1,
        latent_dim: int = 16,
        num_costs: int = 5,
        actor_hidden_dims: Sequence[int] = (256, 128, 64),
        critic_hidden_dims: Sequence[int] = (256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **source_kwargs: Any,
    ) -> None:
        if source_args or source_kwargs:
            # ``ActorCriticBarlowTwins`` in the upstream WheelBipe repository
            # has a different six-dimension constructor and additional
            # scan/privileged/teacher branches.  This compact NP3O owner is
            # intentionally not a drop-in replacement for that architecture;
            # fail closed with a diagnostic rather than accepting source
            # fields and silently training the wrong network.
            unsupported = sorted({str(key) for key in source_kwargs})
            detail = f" unsupported kwargs={unsupported!r}" if unsupported else ""
            raise NotImplementedError(
                "source ActorCriticBarlowTwins scan/teacher architecture is not "
                f"implemented by compact NP3OActorCritic; use the compact "
                f"(actor_obs, critic_obs, actions) contract.{detail}"
            )
        super().__init__(num_actions, init_noise_std)
        self.num_one_step_obs = int(num_actor_obs)
        self.history_size = int(num_actor_history)
        self.num_actor_obs = self.num_one_step_obs * self.history_size
        self.num_critic_obs = int(num_critic_obs)
        self.latent_dim = int(latent_dim)
        self.num_costs = int(num_costs)
        self.history_encoder = _mlp(self.num_actor_obs, self.latent_dim, (128, 64), activation)
        self.target_encoder = _mlp(self.num_one_step_obs, self.latent_dim, (128, 64), activation)
        self.actor = _mlp(
            self.num_one_step_obs + self.latent_dim, num_actions, actor_hidden_dims, activation
        )
        self.critic = _mlp(self.num_critic_obs, 1, critic_hidden_dims, activation)
        self.cost_critic = _mlp(self.num_critic_obs, self.num_costs, critic_hidden_dims, activation)

    def latent(self, obs_history: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        source = obs_history[:, -self.num_one_step_obs :] if target else obs_history
        encoder = self.target_encoder if target else self.history_encoder
        return torch.nn.functional.normalize(encoder(source), dim=-1)

    def _features(self, obs_history: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (obs_history[:, -self.num_one_step_obs :], self.latent(obs_history)), dim=-1
        )

    def act(self, obs_history: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        self._set_distribution(self.actor(self._features(obs_history)))
        return self.distribution.sample()  # type: ignore[union-attr]

    def act_inference(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.actor(self._features(obs_history))

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.critic(torch.nan_to_num(critic_obs, nan=0.0))

    def evaluate_cost(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.nn.functional.softplus(self.cost_critic(torch.nan_to_num(critic_obs, nan=0.0)))

    def representation_parameters(self):
        yield from self.history_encoder.parameters()
        yield from self.target_encoder.parameters()


# Source WheelBipe module names.  These aliases intentionally point to the
# same compact implementations; they do not imply that the bounded NP3O
# owner has the source model's extra scan/teacher branches.
ActorCriticDreamWaq = DreamWaQActorCritic
ActorCriticBarlowTwins = NP3OActorCritic
# Explicit source graph.  Keep the historical ``ActorCriticBarlowTwins`` alias
# pointed at the compact owner for backwards compatibility; callers that need
# the upstream 312D ``on_constraint`` architecture must opt in by name.
ActorCriticBarlowTwinsSource = SourceBarlowTwinsActorCritic
SourceActorCriticBarlowTwins = SourceBarlowTwinsActorCritic

__all__ = [
    "ADABOOT_MODES",
    "DreamWaQActorCritic",
    "NP3OActorCritic",
    "ActorCriticDreamWaq",
    "ActorCriticBarlowTwins",
    "SourceBarlowTwinsActorCritic",
    "ActorCriticBarlowTwinsSource",
    "SourceActorCriticBarlowTwins",
]
