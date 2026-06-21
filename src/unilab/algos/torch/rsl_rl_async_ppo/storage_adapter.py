"""Adapter from async PPO rollout payloads to RSL-RL RolloutStorage."""

from __future__ import annotations

from typing import Any, cast

import torch
from tensordict import TensorDict


def fill_rollout_storage(storage: Any, rollout: dict[str, object]) -> None:
    """Replace RSL-RL storage contents with one complete async PPO rollout.

    RSL-RL does not expose a public bulk-load API for ``RolloutStorage``.  All
    internal field access required by async PPO is intentionally kept here.
    """
    observations = cast(TensorDict, rollout["observations"])
    distribution_params = cast(tuple[torch.Tensor, ...], rollout["distribution_params"])

    if observations.batch_size[0] != storage.num_transitions_per_env:
        raise ValueError(
            "rollout length does not match storage.num_transitions_per_env: "
            f"{observations.batch_size[0]} != {storage.num_transitions_per_env}"
        )
    if observations.batch_size[1] != storage.num_envs:
        raise ValueError(
            f"rollout num_envs does not match storage.num_envs: {observations.batch_size[1]} != {storage.num_envs}"
        )

    storage.clear()
    storage.observations.copy_(observations)
    storage.actions.copy_(cast(torch.Tensor, rollout["actions"]))
    storage.rewards.copy_(cast(torch.Tensor, rollout["rewards"]))
    storage.dones.copy_(cast(torch.Tensor, rollout["dones"]).bool())
    storage.values.copy_(cast(torch.Tensor, rollout["values"]))
    storage.actions_log_prob.copy_(cast(torch.Tensor, rollout["actions_log_prob"]))

    storage.distribution_params = tuple(torch.empty_like(param) for param in distribution_params)
    for index, param in enumerate(distribution_params):
        storage.distribution_params[index].copy_(param)

    storage.step = storage.num_transitions_per_env
