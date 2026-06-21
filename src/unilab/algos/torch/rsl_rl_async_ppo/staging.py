"""Learner-side staging for async RSL-RL PPO rollouts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from tensordict import TensorDict


def stage_ppo_rollout(
    raw_views: Mapping[str, np.ndarray],
    *,
    obs_shapes: Mapping[str, tuple[int, ...]],
    distribution_param_count: int,
    device: str | torch.device,
) -> dict[str, object]:
    """Copy one env-major shared rollout into learner-owned tensors."""
    device = torch.device(device)
    observations: dict[str, torch.Tensor] = {}
    last_obs: dict[str, torch.Tensor] = {}

    for group in obs_shapes:
        obs_field = f"obs/{group}"
        last_field = f"last_obs/{group}"
        if obs_field not in raw_views or last_field not in raw_views:
            raise KeyError(f"rollout is missing observation fields for group {group!r}")
        observations[group] = torch.from_numpy(raw_views[obs_field]).transpose(0, 1).to(device)
        last_obs[group] = torch.from_numpy(raw_views[last_field]).to(device)

    distribution_params = []
    for index in range(distribution_param_count):
        field = f"distribution_params/{index}"
        if field not in raw_views:
            raise KeyError(f"rollout is missing {field!r}")
        distribution_params.append(torch.from_numpy(raw_views[field]).transpose(0, 1).to(device))

    rollout = {
        "observations": TensorDict(
            observations,
            batch_size=[
                next(iter(observations.values())).shape[0],
                next(iter(observations.values())).shape[1],
            ],
            device=device,
        ),
        "last_obs": TensorDict(
            last_obs,
            batch_size=[next(iter(last_obs.values())).shape[0]],
            device=device,
        ),
        "actions": torch.from_numpy(raw_views["actions"]).transpose(0, 1).to(device),
        "rewards": torch.from_numpy(raw_views["rewards"]).transpose(0, 1).unsqueeze(-1).to(device),
        "dones": torch.from_numpy(raw_views["dones"]).transpose(0, 1).unsqueeze(-1).to(device),
        "values": torch.from_numpy(raw_views["values"]).transpose(0, 1).to(device),
        "actions_log_prob": torch.from_numpy(raw_views["actions_log_prob"]).transpose(0, 1).to(device),
        "distribution_params": tuple(distribution_params),
        "policy_version_at_collect_start": int(raw_views["policy_version_start"][0]),
        "policy_version_at_collect_end": int(raw_views["policy_version_end"][0]),
        "rollout_created_time_ns": float(raw_views["rollout_created_time_ns"][0]),
    }
    return rollout
