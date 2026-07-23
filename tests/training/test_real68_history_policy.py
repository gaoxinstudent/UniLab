from __future__ import annotations

import torch

from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic
from unilab.algos.torch.him_ppo.estimator import HIMEstimator


def test_him_velocity_only_estimator_has_no_unsupervised_latent():
    estimator = HIMEstimator(
        temporal_steps=5,
        num_one_step_obs=28,
        enc_hidden_dims=[32, 16],
        latent_dim=0,
        velocity_target_start=0,
        target_obs_start=3,
    )
    history = torch.randn(8, 140)
    critic = torch.randn(8, 59)

    velocity, latent = estimator.get_latent(history)
    estimation_loss, swap_loss = estimator.update(history, critic)

    assert velocity.shape == (8, 3)
    assert latent.shape == (8, 0)
    assert estimation_loss >= 0.0
    assert swap_loss == 0.0


def test_him_actor_uses_latest_observation_frame():
    actor_critic = HIMActorCritic(
        num_actor_obs=15,
        num_critic_obs=7,
        num_one_step_obs=3,
        num_actions=1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        estimator={"enc_hidden_dims": [4], "latent_dim": 0},
    )
    actor_critic.actor = torch.nn.Linear(6, 1, bias=False)
    with torch.no_grad():
        actor_critic.estimator.encoder[-1].weight.zero_()
        actor_critic.estimator.encoder[-1].bias.zero_()
        actor_critic.actor.weight.zero_()
        actor_critic.actor.weight[0, 0] = 1.0

    history = torch.zeros(2, 15)
    history[:, 0] = 2.0
    history[:, -3] = 7.0

    actions = actor_critic.act_inference(history)

    torch.testing.assert_close(actions, torch.full((2, 1), 7.0))


def test_him_actor_output_gain_initializes_small_unbiased_mean():
    output_gain = 0.01
    actor_critic = HIMActorCritic(
        num_actor_obs=15,
        num_critic_obs=7,
        num_one_step_obs=3,
        num_actions=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[4],
        actor_output_gain=output_gain,
        estimator={"enc_hidden_dims": [4], "latent_dim": 0},
    )

    output_layer = actor_critic.actor[-1]
    assert isinstance(output_layer, torch.nn.Linear)
    torch.testing.assert_close(output_layer.bias, torch.zeros_like(output_layer.bias))
    gram = output_layer.weight @ output_layer.weight.T
    torch.testing.assert_close(
        gram,
        torch.eye(2, dtype=gram.dtype) * output_gain**2,
        atol=1.0e-8,
        rtol=1.0e-5,
    )


def test_him_actor_enforces_configured_noise_floor():
    actor_critic = HIMActorCritic(
        num_actor_obs=15,
        num_critic_obs=7,
        num_one_step_obs=3,
        num_actions=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[4],
        init_noise_std=0.08,
        min_noise_std=0.03,
        estimator={"enc_hidden_dims": [4], "latent_dim": 0},
    )
    with torch.no_grad():
        actor_critic.std.fill_(0.001)

    actor_critic.update_distribution(torch.zeros(3, 15))
    torch.testing.assert_close(actor_critic.action_std, torch.full((3, 2), 0.03))

    actor_critic.clamp_action_std_()
    torch.testing.assert_close(actor_critic.std, torch.full((2,), 0.03))


def test_him_actor_critic_empirical_normalization_updates_both_observation_streams():
    actor_critic = HIMActorCritic(
        num_actor_obs=15,
        num_critic_obs=7,
        num_one_step_obs=3,
        num_actions=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[4],
        empirical_normalization=True,
        estimator={"enc_hidden_dims": [4], "latent_dim": 0},
    )
    actor_obs = torch.arange(60, dtype=torch.float32).reshape(4, 15)
    critic_obs = torch.arange(28, dtype=torch.float32).reshape(4, 7)

    actor_critic.update_normalization(actor_obs, critic_obs)

    torch.testing.assert_close(
        actor_critic.normalize_actor_obs(actor_obs).mean(dim=0),
        torch.zeros(15),
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actor_critic.normalize_critic_obs(critic_obs).mean(dim=0),
        torch.zeros(7),
        atol=1.0e-5,
        rtol=0.0,
    )
    assert actor_critic.actor_obs_normalizer.count.item() == 4
    assert actor_critic.critic_obs_normalizer.count.item() == 4

    actor_critic.disable_empirical_normalization()
    assert actor_critic.empirical_normalization is False
    torch.testing.assert_close(actor_critic.normalize_actor_obs(actor_obs), actor_obs)
    torch.testing.assert_close(actor_critic.normalize_critic_obs(critic_obs), critic_obs)
