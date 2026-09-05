"""Focused shape/order tests for the source NP3O observation adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unilab.algos.torch.custom_ppo.runner import CustomOnPolicyRunner


def _adapter_runner() -> CustomOnPolicyRunner:
    """Build only the runner state needed by ``_source_on_constraint``."""

    runner = object.__new__(CustomOnPolicyRunner)
    runner.device = "cpu"
    runner.contract = {
        "actor_input_dim": 312,
        "actor_obs_dim": 28,
        "num_estimate": 4,
        "history_length": 10,
    }
    runner._source_history = None
    return runner


def test_source_leaf_adapter_preserves_source_concat_order() -> None:
    runner = _adapter_runner()
    policy = torch.arange(56, dtype=torch.float32).reshape(2, 28)
    latent = torch.full((2, 4), 7.0)
    history = torch.arange(560, dtype=torch.float32).reshape(2, 280)

    result = runner._source_on_constraint(
        {"policy": policy, "priv_latent": latent, "policy_hist": history}
    )

    assert result.shape == (2, 312)
    torch.testing.assert_close(result[:, :28], policy)
    torch.testing.assert_close(result[:, 28:32], latent)
    torch.testing.assert_close(result[:, 32:], history)


@pytest.mark.parametrize(
    "payload",
    [
        # Widths that happen to sum to 312 must still be rejected: source's
        # contract is exactly 28 + 4 + (10 * 28), not an arbitrary split.
        {
            "policy": torch.zeros(2, 27),
            "priv_latent": torch.zeros(2, 4),
            "policy_hist": torch.zeros(2, 281),
        },
        {
            "policy": torch.zeros(2, 28),
            "priv_latent": torch.zeros(2, 5),
            "policy_hist": torch.zeros(2, 279),
        },
        {
            "policy": torch.zeros(2, 28),
            "priv_latent": torch.zeros(2, 4),
            "policy_hist": torch.zeros(2, 279),
        },
    ],
)
def test_source_leaf_adapter_rejects_malformed_component_widths(payload) -> None:
    runner = _adapter_runner()
    with pytest.raises(ValueError, match="source_barlow|source NP3O|width|shape"):
        runner._source_on_constraint(payload)


def test_source_compact_fallback_uses_zero_warm_start_and_newest_frame() -> None:
    runner = _adapter_runner()
    actor0 = torch.full((2, 28), 1.0)
    latent0 = torch.full((2, 4), 2.0)
    critic0 = torch.cat((actor0, latent0), dim=-1)

    first = runner._source_on_constraint({"actor": actor0, "critic": critic0}, reset=True)
    history0 = first[:, 32:].reshape(2, 10, 28)
    assert torch.count_nonzero(history0[:, :9]).item() == 0
    torch.testing.assert_close(history0[:, -1], actor0)

    actor1 = torch.full((2, 28), 3.0)
    latent1 = torch.full((2, 4), 4.0)
    second = runner._source_on_constraint(
        {"actor": actor1, "critic": torch.cat((actor1, latent1), dim=-1)}
    )
    history1 = second[:, 32:].reshape(2, 10, 28)
    assert torch.count_nonzero(history1[:, :8]).item() == 0
    torch.testing.assert_close(history1[:, -2], actor0)
    torch.testing.assert_close(history1[:, -1], actor1)


def test_source_compact_fallback_resets_only_done_rows() -> None:
    runner = _adapter_runner()
    actor = torch.ones(2, 28)
    runner._source_on_constraint(
        {"actor": actor, "critic": torch.cat((actor, torch.zeros(2, 4)), dim=-1)},
        reset=True,
    )
    next_actor = torch.full((2, 28), 5.0)
    next_latent = torch.full((2, 4), 6.0)
    result = runner._source_on_constraint(
        {"actor": next_actor, "critic": torch.cat((next_actor, next_latent), dim=-1)},
        dones=torch.tensor([True, False]),
    )
    history = result[:, 32:].reshape(2, 10, 28)
    # Row zero starts a fresh source episode; row one retains its previous
    # frame and appends the new frame.
    assert torch.count_nonzero(history[0, :-1]).item() == 0
    torch.testing.assert_close(history[0, -1], next_actor[0])
    torch.testing.assert_close(history[1, -2], actor[1])
    torch.testing.assert_close(history[1, -1], next_actor[1])
