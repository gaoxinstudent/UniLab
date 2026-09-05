from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.custom_ppo.algorithm import NP3O, PPODreamWaq
from unilab.algos.torch.custom_ppo.models import (
    ActorCriticBarlowTwins,
    ActorCriticDreamWaq,
    DreamWaQActorCritic,
    NP3OActorCritic,
    SourceActorCriticBarlowTwins,
    SourceBarlowTwinsActorCritic,
)
from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic
from unilab.algos.torch.him_ppo.algorithm import HIMPPO, PPOHIM


def test_him_accepts_source_positional_constructor_and_mapping() -> None:
    policy = HIMActorCritic(
        28,
        32,
        4,
        5,
        6,
        [8],
        [8],
        "elu",
        1.0,
        50.0,
        20.0,
        estimator={"enc_hidden_dims": [8, 4], "tar_hidden_dims": [8, 4]},
    )
    history = torch.zeros(2, 140)
    actions, estimate = policy.test_inference({"policy_hist": history})
    assert actions.shape == (2, 6)
    assert estimate.shape == (2, 8)


def test_dream_accepts_source_positional_constructor_and_eight_outputs() -> None:
    policy = DreamWaQActorCritic(
        28,
        32,
        6,
        4,
        140,
        20,
        "elu",
        [8, 4],
        [4, 8],
        1.0,
        "log",
        10.0,
        50.0,
        20.0,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
    )
    history = torch.zeros(2, 140)
    result = policy.cenet_forward({"policy_hist": history}, sample=False)
    assert len(result) == 8
    assert policy.noise_std_type == "log"
    assert policy.act_inference({"policy": torch.zeros(2, 28), "policy_hist": history}).shape == (
        2,
        6,
    )


def test_dream_accepts_source_named_cenet_dimensions() -> None:
    """Named source dimensions must shape the compact adapter, not be ignored."""

    policy = DreamWaQActorCritic(
        28,
        32,
        6,
        num_estimate=4,
        cenet_in_dim=140,
        cenet_out_dim=20,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
    )
    assert policy.cenet_in_dim == 140
    assert policy.num_actor_obs == 140
    assert policy.code_dim == 20
    outputs = policy.cenet_forward(torch.zeros(2, 140), sample=False)
    assert outputs[0].shape == (2, 20)


def test_np3o_accepts_source_positional_k_value() -> None:
    policy = NP3OActorCritic(
        28,
        32,
        6,
        num_actor_history=10,
        num_costs=5,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
    )
    algorithm = NP3O(policy, [0.1] * 5, device="cpu")
    torch.testing.assert_close(algorithm.k_value, torch.full((5,), 0.1))


def test_source_module_aliases_are_identity_aliases() -> None:
    assert ActorCriticDreamWaq is DreamWaQActorCritic
    assert ActorCriticBarlowTwins is NP3OActorCritic
    assert PPODreamWaq.__name__ == "DreamWaQPPO"
    assert PPOHIM is HIMPPO


def test_custom_ppo_package_exports_source_loader_contract() -> None:
    import unilab.algos.torch.custom_ppo as custom_ppo

    assert custom_ppo.SourceBarlowTwinsActorCritic is SourceBarlowTwinsActorCritic
    assert custom_ppo.load_source_state_dict.__name__ == "load_source_state_dict"
    assert custom_ppo.canonical_custom_algorithm("ppo_him") == "him"


def test_source_him_algorithm_policy_keyword_alias() -> None:
    policy = HIMActorCritic(
        28,
        32,
        num_estimate=4,
        num_actor_obs_hist=5,
        num_actions=6,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        estimator={"enc_hidden_dims": [8, 4], "tar_hidden_dims": [8, 4]},
    )
    algorithm = PPOHIM(policy=policy, device="cpu")
    assert algorithm.policy is policy


def test_source_barlow_graph_is_explicit_and_uses_on_constraint_contract() -> None:
    """The source graph is opt-in and keeps the upstream 312D stream."""

    policy = SourceBarlowTwinsActorCritic(
        28,
        0,
        4,
        4,
        10,
        6,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[],
        num_costs=5,
        teacher_act=True,
        imi_flag=True,
    )
    assert SourceActorCriticBarlowTwins is SourceBarlowTwinsActorCritic
    assert policy.source_architecture is True
    assert policy.num_obs == 312
    obs = torch.randn(4, 312)
    assert policy.act_inference(obs).shape == (4, 6)
    assert policy.evaluate(obs).shape == (4, 1)
    assert policy.evaluate_cost(obs).shape == (4, 5)
    loss = policy.imitation_learning_loss(obs)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_source_barlow_state_dict_roundtrip_keeps_source_module_names() -> None:
    policy = SourceBarlowTwinsActorCritic(
        28,
        0,
        4,
        4,
        10,
        6,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        priv_encoder_dims=[],
        num_costs=5,
    )
    state = policy.state_dict()
    assert "actor_teacher_backbone.mlp_encoder.model.0.weight" in state
    assert "obs_normalize._mean" in state
    # The source normalizer keeps ``count`` outside state_dict; a strict load
    # therefore exercises the checkpoint-compatible key boundary.
    assert not any(key.endswith("count") for key in state)
    restored = SourceBarlowTwinsActorCritic(
        28,
        0,
        4,
        4,
        10,
        6,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        priv_encoder_dims=[],
        num_costs=5,
    )
    restored.load_state_dict(state, strict=True)


def test_source_barlow_accepts_upstream_policy_metadata_switches_and_direct_state_mapping() -> None:
    """Source-only config switches stay visible without changing graph keys."""

    from unilab.algos.torch.custom_ppo.source_barlow import load_source_state_dict

    policy = SourceBarlowTwinsActorCritic(
        28,
        0,
        4,
        4,
        10,
        6,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        num_costs=5,
        continue_from_last_std=True,
        tanh_encoder_output=False,
    )
    assert policy.continue_from_last_std is True
    assert policy.tanh_encoder_output is False
    target = SourceBarlowTwinsActorCritic(
        28,
        0,
        4,
        4,
        10,
        6,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        num_costs=5,
    )
    load_source_state_dict(target, policy.state_dict())


def test_source_barlow_rejects_compact_history_or_cost_contract() -> None:
    with pytest.raises(ValueError, match="num_hist=10"):
        SourceBarlowTwinsActorCritic(28, 0, 4, 4, 5, 6, num_costs=5)
    with pytest.raises(ValueError, match="exactly five"):
        SourceBarlowTwinsActorCritic(28, 0, 4, 4, 10, 6, num_costs=1)


def test_np3o_accepts_source_storage_shapes_and_process_order() -> None:
    """Exercise the upstream OnConstraintPolicyRunner call contract."""

    policy = SourceBarlowTwinsActorCritic(
        28,
        0,
        4,
        4,
        10,
        6,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        priv_encoder_dims=[],
        num_costs=5,
    )
    algorithm = NP3O(
        policy,
        [1.0, 1.0, 1.0, 0.5, 0.5],
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
        imi_flag=False,
    )
    algorithm.init_storage(
        4,
        2,
        [312],
        [312],
        [6],
        [5],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    obs = torch.randn(4, 312)
    algorithm.act(obs, obs)
    algorithm.process_env_step(
        torch.randn(4),
        torch.zeros(4, 5),
        torch.zeros(4, dtype=torch.bool),
        {"costs": torch.zeros(4, 5)},
    )
    algorithm.act(obs, obs)
    algorithm.process_env_step_source(
        torch.randn(4),
        torch.zeros(4, 5),
        torch.zeros(4, dtype=torch.bool),
        {"costs": torch.zeros(4, 5)},
        next_critic_observations=obs,
    )
    algorithm.compute_returns(obs)
    metrics = algorithm.update()
    assert torch.isfinite(torch.tensor(metrics["value_loss"]))
