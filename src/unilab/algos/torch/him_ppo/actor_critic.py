# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from the HIMLoco RSL-RL HIM actor-critic for UniLab.

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization
from torch.distributions import Normal

from unilab.algos.torch.him_ppo.estimator import HIMEstimator, get_activation


class HIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        min_noise_std: float = 0.0,
        actor_output_gain: float | None = None,
        empirical_normalization: bool = False,
        estimator: dict | None = None,
    ) -> None:
        super().__init__()
        if num_one_step_obs <= 0:
            raise ValueError("num_one_step_obs must be positive")
        if num_actor_obs % num_one_step_obs != 0:
            raise ValueError(
                "num_actor_obs must be an integer multiple of num_one_step_obs "
                f"for HIM history obs, got {num_actor_obs} and {num_one_step_obs}"
            )
        if len(actor_hidden_dims) == 0 or len(critic_hidden_dims) == 0:
            raise ValueError("actor_hidden_dims and critic_hidden_dims must not be empty")
        if min_noise_std < 0.0:
            raise ValueError("min_noise_std must be non-negative")

        self.history_size = int(num_actor_obs // num_one_step_obs)
        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.num_one_step_obs = int(num_one_step_obs)
        self.empirical_normalization = bool(empirical_normalization)
        self.actor_obs_normalizer: nn.Module = (
            EmpiricalNormalization(self.num_actor_obs)
            if self.empirical_normalization
            else nn.Identity()
        )
        self.critic_obs_normalizer: nn.Module = (
            EmpiricalNormalization(self.num_critic_obs)
            if self.empirical_normalization
            else nn.Identity()
        )

        estimator_cfg = dict(estimator or {})
        self.estimator = HIMEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=self.num_one_step_obs,
            activation=activation,
            **estimator_cfg,
        )

        actor_input_dim = self.num_one_step_obs + 3 + self.estimator.num_latent
        self.actor = _build_mlp(actor_input_dim, self.num_actions, actor_hidden_dims, activation)
        self.critic = _build_mlp(self.num_critic_obs, 1, critic_hidden_dims, activation)
        if actor_output_gain is not None:
            output_layer = self.actor[-1]
            assert isinstance(output_layer, nn.Linear)
            nn.init.orthogonal_(output_layer.weight, gain=float(actor_output_gain))
            nn.init.zeros_(output_layer.bias)

        self.min_noise_std = float(min_noise_std)
        self.std = nn.Parameter(
            max(float(init_noise_std), self.min_noise_std) * torch.ones(self.num_actions)
        )
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

    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    def update_normalization(
        self, actor_observations: torch.Tensor, critic_observations: torch.Tensor
    ) -> None:
        if not self.empirical_normalization:
            return
        assert isinstance(self.actor_obs_normalizer, EmpiricalNormalization)
        assert isinstance(self.critic_obs_normalizer, EmpiricalNormalization)
        self.actor_obs_normalizer.update(actor_observations)
        self.critic_obs_normalizer.update(critic_observations)

    def normalize_actor_obs(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_obs_normalizer(observations)

    def normalize_critic_obs(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic_obs_normalizer(observations)

    def disable_empirical_normalization(self) -> None:
        self.empirical_normalization = False
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()

    def update_distribution(self, obs_history: torch.Tensor) -> None:
        obs_history = self.normalize_actor_obs(obs_history)
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        actor_input = torch.cat(
            (obs_history[:, -self.num_one_step_obs :], vel, latent),
            dim=-1,
        )
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std.clamp_min(self.min_noise_std))

    def clamp_action_std_(self) -> None:
        """Keep the persisted exploration parameter inside its configured contract."""
        if self.min_noise_std <= 0.0:
            return
        with torch.no_grad():
            self.std.clamp_(min=self.min_noise_std)

    def act(self, obs_history: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        self.update_distribution(obs_history)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history: torch.Tensor, observations=None) -> torch.Tensor:
        del observations
        if not isinstance(obs_history, torch.Tensor):
            obs_history = obs_history["actor"]
        obs_history = self.normalize_actor_obs(obs_history)
        vel, latent = self.estimator(obs_history)
        actor_input = torch.cat(
            (obs_history[:, -self.num_one_step_obs :], vel, latent),
            dim=-1,
        )
        return self.actor(actor_input)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.critic(self.normalize_critic_obs(critic_observations))


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
