# SPDX-License-Identifier: CC-BY-NC-SA-4.0 AND BSD-3-Clause
#
# Adapted from the HIMLoco RSL-RL HIM rollout storage for UniLab.
# RSL-RL attribution: Copyright (c) 2021-2025, ETH Zurich and NVIDIA
# CORPORATION; BSD-3-Clause portions.
# HIMLoco attribution: Copyright (c) 2024 Junfeng Long, Zirui Wang;
# https://github.com/InternRobotics/HIMLoco; CC BY-NC-SA 4.0.
# See THIRD_PARTY_NOTICES.md for the mixed-license boundary.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch


class HIMRolloutStorage:
    class Transition:
        def __init__(self) -> None:
            # The compact UniLab runner records a flattened tensor.  Source
            # WheelBipe runners record a mapping (``policy_hist``, ``policy``,
            # ``critic`` ...).  Keep the transition deliberately untyped at
            # this boundary so one storage owner can service both contracts.
            self.observations: torch.Tensor | Mapping[str, torch.Tensor] | None = None
            self.critic_observations: torch.Tensor | None = None
            self.next_critic_observations: torch.Tensor | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None

        def clear(self) -> None:
            self.observations = None
            self.critic_observations = None
            self.next_critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: Sequence[int] | Mapping[str, int],
        privileged_obs_shape: Sequence[int | None] | int | None,
        actions_shape: Sequence[int] | int,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.mapping_observations = isinstance(obs_shape, Mapping)
        self.observations_shape: dict[str, int] | None
        if self.mapping_observations:
            # Preserve insertion order from the environment's observation
            # specification; source checkpoints/configs rely on stable key
            # names rather than a concatenated positional width.
            source_shapes = cast(Mapping[str, int | Sequence[int]], obs_shape)
            self.observations_shape = {
                str(key): _shape_width(value) for key, value in source_shapes.items()
            }
            self.obs_shape = None
        else:
            self.observations_shape = None
            self.obs_shape = _shape_tuple(cast(Sequence[int], obs_shape))
        self.privileged_obs_shape = _shape_tuple(privileged_obs_shape)
        self.actions_shape = _shape_tuple(actions_shape)
        self.num_transitions_per_env = int(num_transitions_per_env)
        self.num_envs = int(num_envs)
        self.step = 0

        if self.mapping_observations:
            assert isinstance(self.observations_shape, dict)
            self.observations = {
                key: torch.zeros(
                    self.num_transitions_per_env,
                    self.num_envs,
                    width,
                    device=self.device,
                )
                for key, width in self.observations_shape.items()
            }
        else:
            assert self.obs_shape is not None
            self.observations = torch.zeros(
                self.num_transitions_per_env,
                self.num_envs,
                *cast(tuple[int, ...], self.obs_shape),
                device=self.device,
            )
        if self.privileged_obs_shape and self.privileged_obs_shape[0] is not None:
            if any(dim is None for dim in self.privileged_obs_shape):
                raise ValueError("privileged_obs_shape cannot contain None values")
            privileged_obs_shape = cast(tuple[int, ...], self.privileged_obs_shape)
            self.privileged_observations = torch.zeros(
                self.num_transitions_per_env,
                self.num_envs,
                *privileged_obs_shape,
                device=self.device,
            )
            self.next_privileged_observations = torch.zeros_like(self.privileged_observations)
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None

        self.rewards = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            self.num_transitions_per_env,
            self.num_envs,
            *cast(tuple[int, ...], self.actions_shape),
            device=self.device,
        )
        self.dones = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, device=self.device
        ).bool()
        self.actions_log_prob = torch.zeros_like(self.rewards)
        self.values = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)

    def add_transition(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        if transition.observations is None:
            raise ValueError("transition.observations is required")
        if transition.actions is None:
            raise ValueError("transition.actions is required")
        if transition.rewards is None:
            raise ValueError("transition.rewards is required")
        if transition.dones is None:
            raise ValueError("transition.dones is required")
        if transition.values is None:
            raise ValueError("transition.values is required")
        if transition.actions_log_prob is None:
            raise ValueError("transition.actions_log_prob is required")
        if transition.action_mean is None or transition.action_sigma is None:
            raise ValueError("transition action distribution stats are required")

        if self.mapping_observations:
            if not isinstance(transition.observations, Mapping):
                raise ValueError("mapping HIM storage requires mapping observations")
            assert isinstance(self.observations, dict)
            for key, target in self.observations.items():
                value = transition.observations.get(key)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"transition.observations is missing tensor key {key!r}")
                target[self.step].copy_(value)
        else:
            if not isinstance(transition.observations, torch.Tensor):
                raise ValueError("tensor HIM storage requires tensor observations")
            assert isinstance(self.observations, torch.Tensor)
            self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            critic_observations = transition.critic_observations
            if critic_observations is None and isinstance(transition.observations, Mapping):
                for key in ("critic", "policy", "actor"):
                    candidate = transition.observations.get(key)
                    if isinstance(candidate, torch.Tensor):
                        critic_observations = candidate
                        break
            if critic_observations is None:
                raise ValueError("transition.critic_observations is required")
            if transition.next_critic_observations is None:
                raise ValueError("transition.next_critic_observations is required")
            assert self.next_privileged_observations is not None
            self.privileged_observations[self.step].copy_(critic_observations)
            self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1).bool())
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    # The upstream WheelBipe/HIMLoco storage spells this public method in the
    # plural (``add_transitions``).  Keep the compact singular implementation
    # as the owner and expose a thin identity-compatible alias so source
    # runners can use the migrated storage without a wrapper or a silent
    # conversion of transition fields.
    def add_transitions(self, transition: Transition) -> None:
        self.add_transition(transition)

    def clear(self) -> None:
        self.step = 0

    def get_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source-style mean trajectory length and reward.

        The final rollout row closes any still-open trajectory for reporting.
        Work on a clone so querying statistics cannot mutate the termination
        mask used by a subsequent return computation.
        """

        done = self.dones.detach().clone()
        done[-1] = True
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        # ``torch.std`` defaults to the unbiased estimator, which is undefined
        # for a one-sample rollout.  Source training normally has many rows,
        # so preserve that estimator there while keeping tiny smoke/diagnostic
        # runs finite at the singleton boundary.
        sample_count = int(self.advantages.numel())
        advantage_std = self.advantages.std(unbiased=sample_count > 1).clamp_min(1.0e-8)
        self.advantages = (self.advantages - self.advantages.mean()) / advantage_std

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // int(num_mini_batches)
        if mini_batch_size <= 0:
            raise ValueError("num_mini_batches is too large for the rollout batch")
        indices = torch.randperm(
            int(num_mini_batches) * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        if self.mapping_observations:
            observations_map = cast(dict[str, torch.Tensor], self.observations)
            observations: Any = {
                key: value.flatten(0, 1) for key, value in observations_map.items()
            }
        else:
            observations = cast(torch.Tensor, self.observations).flatten(0, 1)
        if self.privileged_observations is not None:
            assert self.next_privileged_observations is not None
            critic_observations = self.privileged_observations.flatten(0, 1)
            next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
            next_critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(int(num_epochs)):
            for i in range(int(num_mini_batches)):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                if self.mapping_observations:
                    # Match source ``RolloutStorageHIM`` exactly: mapping
                    # observations first, followed by next privileged obs.
                    # The source algorithm derives its critic stream from the
                    # mapping, so no duplicate critic tensor is yielded.
                    mapping_batch = {
                        key: value[batch_idx]
                        for key, value in cast(dict[str, torch.Tensor], observations).items()
                    }
                    yield (
                        mapping_batch,
                        cast(torch.Tensor, next_critic_observations)[batch_idx],
                        actions[batch_idx],
                        values[batch_idx],
                        advantages[batch_idx],
                        returns[batch_idx],
                        old_actions_log_prob[batch_idx],
                        old_mu[batch_idx],
                        old_sigma[batch_idx],
                    )
                else:
                    observations_tensor = cast(torch.Tensor, observations)
                    critic_tensor = cast(torch.Tensor, critic_observations)
                    next_critic_tensor = cast(torch.Tensor, next_critic_observations)
                    yield (
                        observations_tensor[batch_idx],
                        critic_tensor[batch_idx],
                        actions[batch_idx],
                        next_critic_tensor[batch_idx],
                        values[batch_idx],
                        advantages[batch_idx],
                        returns[batch_idx],
                        old_actions_log_prob[batch_idx],
                        old_mu[batch_idx],
                        old_sigma[batch_idx],
                    )


def _shape_tuple(shape: Sequence[int | None] | int | None) -> tuple[int | None, ...]:
    """Normalize source integer shapes and UniLab one-element sequences."""

    if shape is None:
        return ()
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(shape)


def _shape_width(shape: int | Sequence[int]) -> int:
    """Return a flat feature width from a source observation specification."""

    if isinstance(shape, int):
        return int(shape)
    dims = tuple(int(dim) for dim in shape)
    if len(dims) != 1:
        raise ValueError(f"HIM mapping observation shapes must be one-dimensional, got {dims}")
    return dims[0]


# Preserve the exact source import spelling for storage construction and
# checkpoint tooling.  This aliases the implementation rather than creating a
# second class with divergent tensor layout semantics.
RolloutStorageHIM = HIMRolloutStorage

__all__ = ["HIMRolloutStorage", "RolloutStorageHIM"]
