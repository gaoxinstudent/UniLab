"""Compatibility tests for upstream Wheelbipe vanilla-PPO checkpoints."""

from __future__ import annotations

import pytest
import torch

from unilab.training.rsl_rl import (
    is_wheelbipe_source_ppo_checkpoint,
    load_wheelbipe_source_ppo_checkpoint,
)


class _Actor(torch.nn.Module):
    def __init__(self, *, input_dim: int = 4, hidden_dim: int = 3, output_dim: int = 2) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        distribution = torch.nn.Module()
        distribution.register_parameter("std_param", torch.nn.Parameter(torch.ones(output_dim)))
        self.distribution = distribution


class _Critic(torch.nn.Module):
    def __init__(self, *, input_dim: int = 5, hidden_dim: int = 3) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, 1),
        )


def _source_payload(actor: _Actor, critic: _Critic) -> dict[str, object]:
    state: dict[str, torch.Tensor] = {
        f"actor.{key}": value.detach().clone() for key, value in actor.mlp.state_dict().items()
    }
    state.update(
        {f"critic.{key}": value.detach().clone() for key, value in critic.mlp.state_dict().items()}
    )
    state["std"] = actor.distribution.std_param.detach().clone()  # type: ignore[attr-defined]
    return {
        "model_state_dict": state,
        # The source optimizer is intentionally present but not consumed by
        # playback; its parameter IDs belong to the source runner.
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "iter": 123,
    }


def test_upstream_wheelbipe_state_maps_to_current_rsl_rl_names() -> None:
    source_actor = _Actor()
    source_critic = _Critic()
    payload = _source_payload(source_actor, source_critic)
    assert is_wheelbipe_source_ppo_checkpoint(payload)

    target_actor = _Actor()
    target_critic = _Critic()
    load_wheelbipe_source_ppo_checkpoint(target_actor, target_critic, payload)

    torch.testing.assert_close(target_actor.mlp[0].weight, source_actor.mlp[0].weight)
    torch.testing.assert_close(target_actor.mlp[2].bias, source_actor.mlp[2].bias)
    torch.testing.assert_close(
        target_actor.distribution.std_param,
        source_actor.distribution.std_param,  # type: ignore[attr-defined]
    )
    torch.testing.assert_close(target_critic.mlp[0].weight, source_critic.mlp[0].weight)


def test_upstream_wheelbipe_std_maps_to_log_parameterization() -> None:
    class LogActor(_Actor):
        def __init__(self) -> None:
            super().__init__()
            del self.distribution.std_param  # type: ignore[attr-defined]
            self.distribution.register_parameter(
                "log_std_param", torch.nn.Parameter(torch.zeros(2))
            )

    source_actor = _Actor()
    source_actor.distribution.std_param.data.copy_(torch.tensor([0.5, 2.0]))  # type: ignore[attr-defined]
    target_actor = LogActor()
    target_critic = _Critic()
    load_wheelbipe_source_ppo_checkpoint(
        target_actor,
        target_critic,
        _source_payload(source_actor, _Critic()),
    )

    torch.testing.assert_close(
        target_actor.distribution.log_std_param,
        torch.log(torch.tensor([0.5, 2.0])),  # type: ignore[attr-defined]
    )


def test_upstream_wheelbipe_shape_mismatch_is_preflighted_without_partial_load() -> None:
    source_actor = _Actor()
    source_critic = _Critic()
    payload = _source_payload(source_actor, source_critic)
    target_actor = _Actor(input_dim=7)
    target_critic = _Critic()
    actor_before = {key: value.detach().clone() for key, value in target_actor.state_dict().items()}
    critic_before = {
        key: value.detach().clone() for key, value in target_critic.state_dict().items()
    }

    with pytest.raises(ValueError, match="size mismatch for actor"):
        load_wheelbipe_source_ppo_checkpoint(target_actor, target_critic, payload)

    for key, value in target_actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[key])
    for key, value in target_critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key])


@pytest.mark.parametrize(
    "payload",
    [
        {"model_state_dict": {}},
        {"model_state_dict": {"std": torch.ones(2)}},
        {
            "model_state_dict": {
                "std": torch.ones(2),
                "actor.0.weight": torch.ones(3, 4),
                "actor.0.bias": torch.ones(3),
                "critic.0.weight": torch.ones(3, 5),
                "critic.0.bias": torch.ones(3),
                "unexpected": torch.ones(1),
            }
        },
    ],
)
def test_malformed_upstream_wheelbipe_payload_fails_closed(payload: dict[str, object]) -> None:
    assert not is_wheelbipe_source_ppo_checkpoint(payload)
    with pytest.raises(ValueError, match="Upstream Wheelbipe PPO"):
        load_wheelbipe_source_ppo_checkpoint(_Actor(), _Critic(), payload)
