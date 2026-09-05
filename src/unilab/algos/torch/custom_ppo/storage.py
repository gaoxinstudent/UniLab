"""Rollout storage shared by the migrated non-standard PPO variants."""

from __future__ import annotations

from collections.abc import Iterator

import torch


class CustomRolloutStorage:
    """Tensor-only on-policy storage with optional per-step cost channels."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        actor_dim: int,
        critic_dim: int,
        action_dim: int,
        device: str,
        num_costs: int = 0,
        cost_d_values: torch.Tensor | list[float] | None = None,
    ) -> None:
        self.num_envs, self.num_steps = int(num_envs), int(num_steps)
        self.device = device
        shape = (self.num_steps, self.num_envs)
        self.observations = torch.zeros(*shape, actor_dim, device=device)
        self.critic_observations = torch.zeros(*shape, critic_dim, device=device)
        self.next_critic_observations = torch.zeros_like(self.critic_observations)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.dones = torch.zeros(*shape, 1, dtype=torch.bool, device=device)
        self.values = torch.zeros(*shape, 1, device=device)
        self.log_probs = torch.zeros(*shape, 1, device=device)
        self.mu = torch.zeros(*shape, action_dim, device=device)
        self.sigma = torch.ones(*shape, action_dim, device=device)
        self.returns = torch.zeros_like(self.values)
        self.advantages = torch.zeros_like(self.values)
        self.num_costs = int(num_costs)
        self.costs = torch.zeros(*shape, self.num_costs, device=device) if self.num_costs else None
        self.cost_values = (
            torch.zeros(*shape, self.num_costs, device=device) if self.num_costs else None
        )
        self.cost_returns = (
            torch.zeros_like(self.cost_values) if self.cost_values is not None else None
        )
        self.cost_advantages = (
            torch.zeros_like(self.cost_values) if self.cost_values is not None else None
        )
        self.cost_violation = (
            torch.zeros_like(self.cost_values) if self.cost_values is not None else None
        )
        if self.num_costs:
            raw_d = [0.0] * self.num_costs if cost_d_values is None else cost_d_values
            self.cost_d_values = torch.as_tensor(raw_d, device=device, dtype=torch.float32).reshape(
                -1
            )
            if self.cost_d_values.numel() == 1:
                self.cost_d_values = self.cost_d_values.repeat(self.num_costs)
            if self.cost_d_values.numel() != self.num_costs:
                raise ValueError(
                    "cost_d_values must contain exactly num_costs values; "
                    f"got {self.cost_d_values.numel()} for num_costs={self.num_costs}"
                )
        else:
            self.cost_d_values = torch.empty(0, device=device)
        self.step = 0

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        if self.step >= self.num_steps:
            raise RuntimeError("custom PPO rollout storage overflow")
        required = (
            "observations",
            "critic_observations",
            "next_critic_observations",
            "actions",
            "rewards",
            "dones",
            "values",
            "log_probs",
            "mu",
            "sigma",
        )
        missing = [key for key in required if key not in transition]
        if missing:
            raise ValueError(f"transition missing required fields: {missing}")
        idx = self.step
        for key in required:
            target = getattr(self, key if key not in {"log_probs"} else "log_probs")
            value = transition[key].to(self.device)
            if key in {"rewards", "dones", "values", "log_probs"}:
                value = value.reshape(self.num_envs, 1)
            target[idx].copy_(value)
        if self.num_costs:
            costs = transition.get("costs")
            if costs is not None:
                assert self.costs is not None
                self.costs[idx].copy_(costs.to(self.device).reshape(self.num_envs, self.num_costs))
            cost_values = transition.get("cost_values")
            if cost_values is not None:
                assert self.cost_values is not None
                self.cost_values[idx].copy_(
                    cost_values.to(self.device).reshape(self.num_envs, self.num_costs)
                )
        self.step += 1

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_steps)):
            next_value = last_values if step == self.num_steps - 1 else self.values[step + 1]
            not_done = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + float(gamma) * not_done * next_value - self.values[step]
            advantage = delta + float(gamma) * float(lam) * not_done * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages.copy_(self.returns - self.values)
        # ``torch.std`` defaults to the unbiased estimator, which is undefined
        # for a one-sample smoke rollout (NaN survives ``clamp_min``).  The
        # source runner normally has a large batch, so retain its unbiased
        # statistic whenever at least two samples exist and use the population
        # statistic only for the degenerate singleton case.
        sample_count = int(self.advantages.numel())
        advantage_std = self.advantages.std(unbiased=sample_count > 1).clamp_min(1e-8)
        self.advantages.sub_(self.advantages.mean()).div_(advantage_std)

    def compute_cost_returns(
        self, last_cost_values: torch.Tensor, gamma: float, lam: float
    ) -> None:
        if self.num_costs == 0 or self.cost_values is None or self.costs is None:
            return
        assert self.cost_returns is not None and self.cost_advantages is not None
        advantage = torch.zeros_like(last_cost_values)
        for step in reversed(range(self.num_steps)):
            next_value = (
                last_cost_values if step == self.num_steps - 1 else self.cost_values[step + 1]
            )
            not_done = 1.0 - self.dones[step].float()
            delta = self.costs[step] + float(gamma) * not_done * next_value - self.cost_values[step]
            advantage = delta + float(gamma) * float(lam) * not_done * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]
        self.cost_advantages.copy_(self.cost_returns - self.cost_values)
        mean = self.cost_advantages.mean(dim=(0, 1), keepdim=True)
        cost_sample_count = self.num_envs * self.num_steps
        std = self.cost_advantages.std(
            dim=(0, 1), keepdim=True, unbiased=cost_sample_count > 1
        ).clamp_min(1e-8)
        # NP3O's constraint term uses the normalized mean advantage together
        # with the discounted distance from each configured d-value.  Keep
        # this derived tensor in storage so minibatches receive the exact
        # source objective rather than reconstructing it from a shuffled
        # subset (which would change the normalization statistics).
        assert self.cost_violation is not None
        self.cost_violation.copy_(
            ((1.0 - float(gamma)) * (self.cost_returns - self.cost_d_values) + mean) / std
        )
        self.cost_advantages.sub_(mean).div_(std)

    def batches(
        self, num_mini_batches: int, num_epochs: int
    ) -> Iterator[dict[str, torch.Tensor | None]]:
        batch_size = self.num_envs * self.num_steps
        mini = batch_size // int(num_mini_batches)
        if mini <= 0:
            raise ValueError("num_mini_batches exceeds rollout batch size")
        indices = torch.randperm(batch_size, device=self.device)[: mini * int(num_mini_batches)]
        flat = {
            name: value.flatten(0, 1)
            for name, value in {
                "observations": self.observations,
                "critic_observations": self.critic_observations,
                "next_critic_observations": self.next_critic_observations,
                "actions": self.actions,
                "values": self.values,
                "returns": self.returns,
                "advantages": self.advantages,
                "log_probs": self.log_probs,
                "mu": self.mu,
                "sigma": self.sigma,
                "dones": self.dones,
            }.items()
        }
        if (
            self.cost_values is not None
            and self.cost_returns is not None
            and self.cost_advantages is not None
        ):
            assert self.cost_violation is not None
            flat.update(
                {
                    "cost_values": self.cost_values.flatten(0, 1),
                    "cost_returns": self.cost_returns.flatten(0, 1),
                    "cost_advantages": self.cost_advantages.flatten(0, 1),
                    "cost_violation": self.cost_violation.flatten(0, 1),
                }
            )
        for _ in range(int(num_epochs)):
            for i in range(int(num_mini_batches)):
                select = indices[i * mini : (i + 1) * mini]
                yield {key: value[select] for key, value in flat.items()}  # type: ignore[union-attr]

    def clear(self) -> None:
        self.step = 0
