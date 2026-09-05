# SPDX-License-Identifier: CC-BY-NC-SA-4.0 AND BSD-3-Clause
#
# Adapted from the HIMLoco RSL-RL HIMPPO algorithm for UniLab.
# RSL-RL attribution: Copyright (c) 2021-2025, ETH Zurich and NVIDIA
# CORPORATION; BSD-3-Clause portions.
# HIMLoco attribution: Copyright (c) 2024 Junfeng Long, Zirui Wang;
# https://github.com/InternRobotics/HIMLoco; CC BY-NC-SA 4.0.
# See THIRD_PARTY_NOTICES.md for the mixed-license boundary.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic
from unilab.algos.torch.him_ppo.storage import HIMRolloutStorage


class HIMPPO:
    actor_critic: HIMActorCritic

    def __init__(
        self,
        actor_critic=None,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = 0.01,
        device: str = "cpu",
        policy=None,
        **kwargs: Any,
    ) -> None:
        # The source PPOHIM constructor keeps these runner/distributed knobs
        # in its public surface.  They do not change the compact network
        # contract, but retaining their validated metadata lets source config
        # composition call this owner without silently dropping a requested
        # advantage-normalization mode.
        normalize_advantage_per_mini_batch = bool(
            kwargs.pop("normalize_advantage_per_mini_batch", False)
        )
        multi_gpu_cfg = kwargs.pop("multi_gpu_cfg", None)
        del kwargs
        if actor_critic is None:
            actor_critic = policy
        elif policy is not None and policy is not actor_critic:
            raise ValueError("HIMPPO received conflicting actor_critic and policy arguments")
        if actor_critic is None:
            raise TypeError(
                "HIMPPO requires an actor_critic (or source-compatible policy) argument"
            )
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = float(learning_rate)
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = (
            int(multi_gpu_cfg.get("global_rank", 0)) if isinstance(multi_gpu_cfg, Mapping) else 0
        )
        self.gpu_world_size = (
            int(multi_gpu_cfg.get("world_size", 1)) if isinstance(multi_gpu_cfg, Mapping) else 1
        )

        self.actor_critic = actor_critic
        # Upstream WheelBipe runners expose the model as ``alg.policy`` while
        # the original UniLab HIM adapter called it ``actor_critic``.  Keep
        # both names pointing at the same module so source-style checkpoint/
        # export integrations can reuse this algorithm without a wrapper.
        self.policy = actor_critic
        self.actor_critic.to(self.device)
        self.storage: HIMRolloutStorage | None = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.transition = HIMRolloutStorage.Transition()

        self.clip_param = float(clip_param)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.max_grad_norm = float(max_grad_norm)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ) -> None:
        self.storage = HIMRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def train_mode(self) -> None:
        self.actor_critic.train()

    def test_mode(self) -> None:
        """Switch the policy to evaluation mode (source runner API)."""

        self.actor_critic.eval()

    def act(self, obs: Any, critic_obs: Any | None = None) -> torch.Tensor:
        """Sample an action from compact tensors or a source observation map.

        UniLab's runner supplies ``(actor_tensor, critic_tensor)``.  The
        upstream WheelBipe runner supplies one mapping argument containing
        ``policy_hist`` and ``critic``.  Resolving the latter here keeps the
        owner layer explicit and avoids forcing either runner to manufacture a
        backend-specific adapter object.
        """

        actor_input: Any = obs
        critic_input: Any = critic_obs if critic_obs is not None else obs
        self.transition.actions = self.actor_critic.act(actor_input).detach()
        self.transition.values = self.actor_critic.evaluate(critic_input).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = _detach_observation(obs, self.device)
        self.transition.critic_observations = _critic_obs(critic_input).to(self.device).detach()
        return self.transition.actions

    def process_env_step(
        self,
        rewards: Any = None,
        dones: Any = None,
        extras: Any = None,
        next_critic_obs: Any = None,
        *,
        next_obs: Any = None,
    ) -> None:
        """Record one transition in either source or compact argument order.

        Source order is ``(rewards, dones, extras, next_critic_obs)``;
        historical UniLab order is ``(next_obs, rewards, dones, extras)``.
        The types at this boundary make the two forms unambiguous, and keyword
        ``next_obs=`` remains available for callers that prefer the local
        spelling.
        """

        if next_obs is not None:
            # Explicit compact keyword form.
            if next_critic_obs is not None:
                raise TypeError("pass only one of next_obs and next_critic_obs")
            next_critic_obs = next_obs
        elif _looks_like_compact_process_order(rewards, dones, extras, next_critic_obs):
            # Re-map the old positional form to the source-named locals.
            next_critic_obs, rewards, dones, extras = rewards, dones, extras, next_critic_obs

        if not isinstance(rewards, torch.Tensor):
            raise TypeError("HIM-PPO rewards must be a tensor")
        if not isinstance(dones, torch.Tensor):
            raise TypeError("HIM-PPO dones must be a tensor")
        if not isinstance(extras, Mapping):
            extras = {} if extras is None else extras
        if next_critic_obs is None:
            raise TypeError("HIM-PPO process_env_step requires next_critic_obs/next_obs")

        next_critic_tensor = _critic_obs(next_critic_obs).to(self.device).clone().detach()
        self.transition.next_critic_observations = next_critic_tensor
        self.transition.rewards = rewards.to(self.device).clone()
        self.transition.dones = dones.to(self.device)

        timeouts = extras.get("time_outs")
        timeout_bootstrap_obs = extras.get("time_out_bootstrap_obs")
        if isinstance(timeouts, torch.Tensor):
            timeout_bool = timeouts.to(self.device).bool().view(-1)
            timeout_mask = timeout_bool.float()
            if timeout_bootstrap_obs is not None and torch.count_nonzero(timeout_bool) > 0:
                bootstrap_critic_obs = _critic_obs(timeout_bootstrap_obs).to(self.device)
                bootstrap_values = self.actor_critic.evaluate(bootstrap_critic_obs).detach()
                correction = self.gamma * torch.squeeze(
                    bootstrap_values * timeout_mask.unsqueeze(1), 1
                )
                if self.transition.rewards.ndim == 2 and self.transition.rewards.shape[-1] == 1:
                    correction = correction.unsqueeze(1)
                self.transition.rewards += correction

                patched_next_critic_obs = self.transition.next_critic_observations.clone()
                patched_next_critic_obs[timeout_bool] = bootstrap_critic_obs[timeout_bool].detach()
                self.transition.next_critic_observations = patched_next_critic_obs
            else:
                transition_values = self.transition.values
                assert transition_values is not None
                correction = self.gamma * torch.squeeze(
                    transition_values * timeout_mask.unsqueeze(1), 1
                )
                if self.transition.rewards.ndim == 2 and self.transition.rewards.shape[-1] == 1:
                    correction = correction.unsqueeze(1)
                self.transition.rewards += correction

        assert self.storage is not None
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: Any) -> None:
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        assert self.storage is not None
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> tuple[float, float, float, float] | dict[str, float]:
        assert self.storage is not None
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        mean_entropy = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for batch in generator:
            # Compact storage yields ten tensors (including the current
            # critic stream).  Source-compatible mapping storage yields the
            # original nine-tuple and derives that stream from ``obs``.
            if len(batch) == 9:
                (
                    obs_batch,
                    next_critic_obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                ) = batch
                critic_obs_batch = _critic_obs(obs_batch)
            else:
                (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    next_critic_obs_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                ) = batch
            if isinstance(obs_batch, Mapping):
                obs_batch = _detach_observation(obs_batch, self.device)
            critic_obs_batch = _critic_obs(critic_obs_batch).to(self.device)
            next_critic_obs_batch = _critic_obs(next_critic_obs_batch).to(self.device)
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            estimation_loss, swap_loss = self.actor_critic.estimator.update(
                _policy_hist(obs_batch),
                next_critic_obs_batch,
                lr=self.learning_rate,
            )

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            if self.normalize_advantage_per_mini_batch:
                # Keep the source unbiased estimator for ordinary minibatches,
                # but avoid NaN when a tiny smoke run yields one sample.
                sample_count = int(advantages_batch.numel())
                advantage_std = advantages_batch.std(unbiased=sample_count > 1).clamp_min(1.0e-8)
                advantages_batch = (advantages_batch - advantages_batch.mean()) / advantage_std
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_estimation_loss += float(estimation_loss)
            mean_swap_loss += float(swap_loss)
            mean_entropy += float(entropy_batch.mean().item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        mean_entropy /= num_updates
        source_mapping = self.storage.mapping_observations
        self.storage.clear()

        if source_mapping:
            # Upstream ``PPOHIM.update`` exposes named losses to its runner;
            # retain that shape only for source-style mapping storage.  The
            # compact UniLab runner continues to receive its historical tuple.
            return {
                "value_function": mean_value_loss,
                "surrogate": mean_surrogate_loss,
                "entropy": mean_entropy,
                "estimation": mean_estimation_loss,
                "swap": mean_swap_loss,
            }

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_estimation_loss,
            mean_swap_loss,
        )


def _is_observation_mapping(value: Any) -> bool:
    """Return whether ``value`` exposes the source observation mapping API."""

    return isinstance(value, Mapping) or isinstance(value, TensorDict)


def _mapping_tensor(value: Any, keys: tuple[str, ...], label: str) -> torch.Tensor:
    """Resolve a tensor from either a dict or a TensorDict-like payload."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
    # TensorDict versions and lightweight adapters may not register as a
    # ``Mapping`` while still exposing public ``__getitem__`` access.
    for key in keys:
        try:
            candidate = value[key]
        except (AttributeError, KeyError, TypeError, IndexError):
            continue
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise KeyError(f"HIM-PPO {label} must contain one of {keys}")


def _critic_obs(obs: Any) -> torch.Tensor:
    return _mapping_tensor(obs, ("critic", "policy", "actor"), "critic observations")


def _policy_hist(obs: Any) -> torch.Tensor:
    return _mapping_tensor(obs, ("policy_hist", "actor", "policy"), "policy observations")


def _detach_observation(value: Any, device: str) -> Any:
    """Move tensor leaves of a source mapping without mutating the env payload."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device).detach()
    if isinstance(value, Mapping):
        return {
            str(key): leaf.to(device=device).detach() if isinstance(leaf, torch.Tensor) else leaf
            for key, leaf in value.items()
        }
    return value


def _looks_like_compact_process_order(
    first: Any,
    second: Any,
    third: Any,
    fourth: Any,
) -> bool:
    """Detect legacy ``(next_obs, rewards, dones, extras)`` positionals."""

    # Mapping first is the normal RslRlVecEnvWrapper path.  A tensor first is
    # also supported for lightweight tests/adapters; in that case the third
    # argument is the dones tensor and the fourth is the extras mapping.
    if _is_observation_mapping(first):
        return True
    return (
        isinstance(first, torch.Tensor)
        and isinstance(second, torch.Tensor)
        and isinstance(third, torch.Tensor)
        and isinstance(fourth, Mapping)
    )


# Upstream WheelBipe names the algorithm ``PPOHIM``.  This identity alias
# keeps source runner configs importable while retaining ``HIMPPO`` as the
# UniLab-facing spelling.
PPOHIM = HIMPPO

__all__ = ["HIMPPO", "PPOHIM"]
