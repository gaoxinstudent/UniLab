# SPDX-License-Identifier: CC-BY-NC-SA-4.0 AND BSD-3-Clause
#
# Adapted from the HIMLoco RSL-RL HIM actor-critic for UniLab.
# RSL-RL attribution: Copyright (c) 2021-2025, ETH Zurich and NVIDIA
# CORPORATION; BSD-3-Clause portions.
# HIMLoco attribution: Copyright (c) 2024 Junfeng Long, Zirui Wang;
# https://github.com/InternRobotics/HIMLoco; CC BY-NC-SA 4.0.
# See THIRD_PARTY_NOTICES.md for the mixed-license boundary.

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributions import Normal

from unilab.algos.torch.him_ppo.estimator import HIMEstimator, get_activation


class HIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        *args: Any,
        num_one_step_obs: int | None = None,
        num_actions: int | None = None,
        num_estimate: int = 3,
        num_actor_obs_hist: int | None = None,
        actor_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        estimator: dict | None = None,
        estimator_feature_clip: float | None = 50.0,
        action_mean_clip: float | None = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Two constructor layouts are in the wild:
        #
        # * UniLab's compact layout keeps the flattened actor width first and
        #   receives ``num_one_step_obs``/``num_actions`` next.
        # * The source WheelBipe layout receives
        #   ``num_estimate, num_actor_obs_hist, num_actions`` after the first
        #   two dimensions, with ``num_actor_obs`` meaning one-step width.
        #
        # Parse both explicitly at this boundary.  In particular, do not use
        # a value-order heuristic (the common 4/5/6 dimensions happen to be
        # increasing today but that is not a contract).
        positional = list(args)
        source_style = num_one_step_obs is None and (
            num_actor_obs_hist is not None
            or (
                len(positional) >= 3
                and all(
                    isinstance(value, Integral) and not isinstance(value, bool)
                    for value in positional[:3]
                )
            )
            or (
                num_actions is not None
                and len(positional) >= 2
                and all(
                    isinstance(value, Integral) and not isinstance(value, bool)
                    for value in positional[:2]
                )
                and num_estimate != 3
            )
        )

        if source_style:
            if positional:
                source_estimate = int(positional.pop(0))
                if num_estimate != 3 and int(num_estimate) != source_estimate:
                    raise ValueError(
                        "source num_estimate conflicts with the named num_estimate: "
                        f"{source_estimate} != {num_estimate}"
                    )
                num_estimate = source_estimate
            if positional:
                source_history = int(positional.pop(0))
                if num_actor_obs_hist is not None and int(num_actor_obs_hist) != source_history:
                    raise ValueError(
                        "source num_actor_obs_hist conflicts with the named history: "
                        f"{source_history} != {num_actor_obs_hist}"
                    )
                num_actor_obs_hist = source_history
            if num_actor_obs_hist is None:
                raise TypeError("source HIM constructor requires num_actor_obs_hist")
            if positional:
                source_actions = int(positional.pop(0))
                if num_actions is not None and int(num_actions) != source_actions:
                    raise ValueError(
                        "source num_actions conflicts with the named num_actions: "
                        f"{source_actions} != {num_actions}"
                    )
                num_actions = source_actions
            if num_actions is None:
                raise TypeError("num_actions must be provided")

            # Remaining source positional options follow its public
            # constructor order.  Unknown trailing values are rejected rather
            # than silently changing the network shape.
            source_fields = (
                "actor_hidden_dims",
                "critic_hidden_dims",
                "activation",
                "init_noise_std",
                "estimator_feature_clip",
                "action_mean_clip",
            )
            if len(positional) > len(source_fields):
                raise TypeError(
                    f"too many positional arguments for source HIM constructor: got {len(args) + 2}"
                )
            for field, value in zip(source_fields, positional):
                if field == "actor_hidden_dims":
                    actor_hidden_dims = value
                elif field == "critic_hidden_dims":
                    critic_hidden_dims = value
                elif field == "activation":
                    activation = str(value)
                elif field == "init_noise_std":
                    init_noise_std = float(value)
                elif field == "estimator_feature_clip":
                    estimator_feature_clip = None if value is None else float(value)
                else:
                    action_mean_clip = None if value is None else float(value)

            # Source ``num_actor_obs`` is the one-step width.  Materialize the
            # flattened width used internally by the compact runner below.
            num_one_step_obs = int(num_actor_obs)
            num_actor_obs = num_one_step_obs * int(num_actor_obs_hist)
        else:
            # Canonical UniLab positional compatibility (the pre-alias
            # signature was ``flat_width, critic_width, one_step, actions,
            # actor_hidden_dims, ...``).  Keyword callers take this same path.
            if positional:
                if num_one_step_obs is None:
                    num_one_step_obs = int(positional.pop(0))
                elif int(num_one_step_obs) != int(positional.pop(0)):
                    raise ValueError("conflicting positional and named num_one_step_obs")
            if positional:
                if num_actions is None:
                    num_actions = int(positional.pop(0))
                elif int(num_actions) != int(positional.pop(0)):
                    raise ValueError("conflicting positional and named num_actions")
            local_fields = (
                "actor_hidden_dims",
                "critic_hidden_dims",
                "activation",
                "init_noise_std",
                "estimator",
            )
            if len(positional) > len(local_fields):
                raise TypeError(
                    f"too many positional arguments for UniLab HIM constructor: got {len(args) + 2}"
                )
            for field, value in zip(local_fields, positional):
                if field == "actor_hidden_dims":
                    actor_hidden_dims = value
                elif field == "critic_hidden_dims":
                    critic_hidden_dims = value
                elif field == "activation":
                    activation = str(value)
                elif field == "init_noise_std":
                    init_noise_std = float(value)
                else:
                    estimator = value

        # The source WheelBipe class receives the one-step width and history
        # as ``num_actor_obs``/``num_actor_obs_hist`` and exposes ``policy_hist``
        # at its mapping boundary.  UniLab's owner normally passes the
        # already-flattened width plus ``num_one_step_obs``.  Accept both forms
        # without changing the canonical flattened representation used by the
        # local runner/exporter.
        if num_one_step_obs is None:
            num_one_step_obs = int(num_actor_obs)
            history_size = 1 if num_actor_obs_hist is None else int(num_actor_obs_hist)
            if history_size <= 0:
                raise ValueError("num_actor_obs_hist must be positive")
            num_actor_obs = int(num_one_step_obs) * history_size
        elif num_actor_obs_hist is not None:
            history_size = int(num_actor_obs_hist)
            if history_size <= 0:
                raise ValueError("num_actor_obs_hist must be positive")
            if int(num_actor_obs) != int(num_one_step_obs) * history_size:
                raise ValueError(
                    "num_actor_obs_hist does not match the flattened actor observation width: "
                    f"num_actor_obs={num_actor_obs}, num_one_step_obs={num_one_step_obs}, "
                    f"history={history_size}"
                )
        if num_actions is None:
            raise TypeError("num_actions must be provided")
        if num_one_step_obs <= 0:
            raise ValueError("num_one_step_obs must be positive")
        if num_actor_obs % num_one_step_obs != 0:
            raise ValueError(
                "num_actor_obs must be an integer multiple of num_one_step_obs "
                f"for HIM history obs, got {num_actor_obs} and {num_one_step_obs}"
            )
        if len(actor_hidden_dims) == 0 or len(critic_hidden_dims) == 0:
            raise ValueError("actor_hidden_dims and critic_hidden_dims must not be empty")

        self.history_size = int(num_actor_obs // num_one_step_obs)
        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_actor_obs_hist = int(self.history_size)
        self.estimator_feature_clip = (
            None if estimator_feature_clip is None else float(estimator_feature_clip)
        )
        self.action_mean_clip = None if action_mean_clip is None else float(action_mean_clip)
        # Source policy configs contain a few optional knobs that are not
        # needed by this compact owner.  Keep accepting them at this boundary
        # so a source config can be inspected/constructed without a Python-side
        # rewrite; unknown values remain intentionally inert.
        del kwargs
        if isinstance(num_estimate, bool) or int(num_estimate) <= 0:
            raise ValueError("num_estimate must be a positive integer")
        self.num_estimate = int(num_estimate)

        estimator_cfg = dict(estimator or {})
        if source_style:
            # HIMLoco's estimator derives these offsets from the source
            # dimensions (velocity starts immediately after the one-step
            # policy frame; the target frame starts after ``num_estimate``
            # privileged channels).  Preserve explicit caller overrides, but
            # do not silently retain UniLab's compact defaults when a source
            # constructor is used without an estimator mapping.
            estimator_cfg.setdefault("velocity_target_start", int(num_one_step_obs))
            estimator_cfg.setdefault("target_obs_start", int(num_estimate))
        configured_estimate = estimator_cfg.pop("num_estimate", None)
        if configured_estimate is not None and int(configured_estimate) != self.num_estimate:
            raise ValueError(
                "estimator.num_estimate must match the HIM policy estimate width: "
                f"{configured_estimate} != {self.num_estimate}"
            )
        self.estimator = HIMEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=self.num_one_step_obs,
            num_estimate=self.num_estimate,
            activation=activation,
            **estimator_cfg,
        )

        actor_input_dim = self.num_one_step_obs + self.num_estimate + self.estimator.num_latent
        self.actor = _build_mlp(actor_input_dim, self.num_actions, actor_hidden_dims, activation)
        self.critic = _build_mlp(self.num_critic_obs, 1, critic_hidden_dims, activation)

        self.std = nn.Parameter(float(init_noise_std) * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def test(self) -> None:
        """Source-policy compatibility alias for switching to eval mode."""

        self.eval()

    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def init_weights(sequential: nn.Sequential, scales: list[float] | tuple[float, ...]) -> None:
        """Initialize linear layers using the source helper's public name."""

        linear_layers = [module for module in sequential if isinstance(module, nn.Linear)]
        for index, module in enumerate(linear_layers):
            if index < len(scales):
                nn.init.orthogonal_(module.weight, gain=cast(Any, scales[index]))

    @staticmethod
    def _resolve_observation(value: Any, keys: tuple[str, ...], label: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Mapping):
            for key in keys:
                candidate = value.get(key)
                if isinstance(candidate, torch.Tensor):
                    return candidate
        # TensorDictBase-compatible adapters may expose only __getitem__ and
        # keys(); this is a payload boundary, not backend capability probing.
        for key in keys:
            try:
                candidate = value[key]
            except (AttributeError, KeyError, TypeError, IndexError):
                continue
            if isinstance(candidate, torch.Tensor):
                return candidate
        raise TypeError(f"HIM {label} must be a tensor or mapping containing {keys}")

    def _sanitize_tensor(self, value: torch.Tensor, clip: float | None = None) -> torch.Tensor:
        if clip is None:
            return torch.nan_to_num(value, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return torch.nan_to_num(value, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)

    def update_distribution(self, obs_history: torch.Tensor | Mapping[str, Any]) -> None:
        obs_history = self._resolve_observation(
            obs_history, ("policy_hist", "actor", "policy"), "actor input"
        )
        obs_history = self._sanitize_tensor(obs_history)
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        vel = self._sanitize_tensor(vel, self.estimator_feature_clip)
        latent = self._sanitize_tensor(latent, self.estimator_feature_clip)
        actor_input = torch.cat(
            # Histories are stored oldest -> newest by the UniLab runner.  The
            # source HIM actor conditions on the most recent one-step frame;
            # taking the prefix would silently make inference one episode
            # history stale while the estimator still consumed all frames.
            (obs_history[:, -self.num_one_step_obs :], vel, latent),
            dim=-1,
        )
        mean = self._sanitize_tensor(self.actor(actor_input), self.action_mean_clip)
        std = torch.nan_to_num(self.std, nan=1.0, posinf=10.0, neginf=1.0e-6).clamp_min(1.0e-6)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, obs_history: torch.Tensor | Mapping[str, Any], **kwargs: Any) -> torch.Tensor:
        del kwargs
        self.update_distribution(obs_history)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(
        self, obs_history: torch.Tensor | Mapping[str, Any], observations=None
    ) -> torch.Tensor:
        del observations
        obs_history = self._resolve_observation(
            obs_history, ("policy_hist", "actor", "policy"), "actor input"
        )
        obs_history = self._sanitize_tensor(obs_history)
        vel, latent = self.estimator(obs_history)
        vel = self._sanitize_tensor(vel, self.estimator_feature_clip)
        latent = self._sanitize_tensor(latent, self.estimator_feature_clip)
        actor_input = torch.cat(
            (obs_history[:, -self.num_one_step_obs :], vel, latent),
            dim=-1,
        )
        return self._sanitize_tensor(self.actor(actor_input), self.action_mean_clip)

    def test_inference(
        self, obs_history: torch.Tensor | Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic actions and estimator output (source API)."""

        obs_history = self._resolve_observation(
            obs_history, ("policy_hist", "actor", "policy"), "actor input"
        )
        obs_history = self._sanitize_tensor(obs_history)
        vel, latent = self.estimator(obs_history)
        vel = self._sanitize_tensor(vel, self.estimator_feature_clip)
        latent = self._sanitize_tensor(latent, self.estimator_feature_clip)
        actor_input = torch.cat((obs_history[:, -self.num_one_step_obs :], vel, latent), dim=-1)
        actions = self._sanitize_tensor(self.actor(actor_input), self.action_mean_clip)
        return actions, torch.cat((vel, latent), dim=-1)

    def evaluate(
        self, critic_observations: torch.Tensor | Mapping[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        del kwargs
        critic_observations = self._resolve_observation(
            critic_observations, ("critic", "policy", "actor"), "critic input"
        )
        return self.critic(self._sanitize_tensor(critic_observations))


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | tuple[int, ...],
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(last_dim, int(hidden_dim)), get_activation(activation)]
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


# Source WheelBipe imports this class as ``ActorCriticHIM``.  Keep the
# descriptive UniLab name as the canonical implementation and expose the
# source spelling as an identity alias for checkpoint/config compatibility.
ActorCriticHIM = HIMActorCritic

__all__ = ["HIMActorCritic", "ActorCriticHIM"]
