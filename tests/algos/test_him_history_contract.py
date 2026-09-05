from __future__ import annotations

import torch
from torch import nn

from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic
from unilab.algos.torch.him_ppo.algorithm import HIMPPO
from unilab.algos.torch.him_ppo.estimator import HIMEstimator
from unilab.algos.torch.him_ppo.storage import HIMRolloutStorage


class _ZeroEstimator(nn.Module):
    num_latent = 1

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs_history.shape[0]
        return torch.zeros(batch, 3), torch.zeros(batch, self.num_latent)


def test_him_actor_conditions_on_newest_history_frame() -> None:
    policy = HIMActorCritic(
        num_actor_obs=4,
        num_critic_obs=1,
        num_one_step_obs=2,
        num_actions=1,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        estimator={"enc_hidden_dims": [4, 2, 1], "tar_hidden_dims": [4, 2]},
    )
    policy.estimator = _ZeroEstimator()
    captured: list[torch.Tensor] = []
    hook = policy.actor[0].register_forward_hook(lambda _m, args, _out: captured.append(args[0]))
    try:
        policy.act_inference(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    finally:
        hook.remove()
    assert captured
    # History layout is [oldest frame, newest frame]; the actor must consume
    # the newest frame while the estimator consumes the complete history.
    torch.testing.assert_close(captured[0][0, :2], torch.tensor([3.0, 4.0]))


def test_him_estimator_keeps_configured_estimate_width_in_inference() -> None:
    """The compact WheelBipe profile predicts velocity plus height (4 values)."""

    estimator = HIMEstimator(
        temporal_steps=5,
        num_one_step_obs=28,
        num_estimate=4,
        enc_hidden_dims=[8, 4],
        tar_hidden_dims=[8, 4],
    )
    estimate, latent = estimator(torch.zeros(2, 140))
    assert estimate.shape == (2, 4)
    assert latent.shape == (2, 4)

    policy = HIMActorCritic(
        num_actor_obs=140,
        num_critic_obs=32,
        num_one_step_obs=28,
        num_actions=6,
        num_estimate=4,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        estimator={"enc_hidden_dims": [8, 4], "tar_hidden_dims": [8, 4]},
    )
    actions = policy.act_inference(torch.zeros(2, 140))
    assert actions.shape == (2, 6)
    assert torch.isfinite(actions).all()


def test_him_source_constructor_alias_and_mapping_observation_contract() -> None:
    """The migrated HIM policy accepts source keyword names at its boundary."""

    policy = HIMActorCritic(
        num_actor_obs=28,
        num_critic_obs=32,
        num_estimate=4,
        num_actor_obs_hist=5,
        num_actions=6,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        estimator={"enc_hidden_dims": [8, 4], "tar_hidden_dims": [8, 4]},
    )
    history = torch.zeros(2, 140)
    critic = torch.zeros(2, 32)
    actions = policy.act_inference({"policy_hist": history})
    actions_with_estimate, estimate = policy.test_inference({"policy_hist": history})
    value = policy.evaluate({"critic": critic})

    assert policy.history_size == 5
    assert policy.num_actor_obs == 140
    assert actions.shape == (2, 6)
    torch.testing.assert_close(actions, actions_with_estimate)
    assert estimate.shape == (2, 8)
    assert value.shape == (2, 1)


def test_him_storage_singleton_return_normalization_stays_finite() -> None:
    """A one-row tiny rollout must not turn the source advantage into NaN."""

    storage = HIMRolloutStorage(
        num_envs=1,
        num_transitions_per_env=1,
        obs_shape=(2,),
        privileged_obs_shape=(3,),
        actions_shape=(1,),
        device="cpu",
    )
    transition = HIMRolloutStorage.Transition()
    transition.observations = torch.zeros(1, 2)
    transition.critic_observations = torch.zeros(1, 3)
    transition.next_critic_observations = torch.zeros(1, 3)
    transition.actions = torch.zeros(1, 1)
    transition.rewards = torch.ones(1)
    transition.dones = torch.ones(1, dtype=torch.bool)
    transition.values = torch.zeros(1, 1)
    transition.actions_log_prob = torch.zeros(1)
    transition.action_mean = torch.zeros(1, 1)
    transition.action_sigma = torch.ones(1, 1)
    storage.add_transition(transition)
    storage.compute_returns(torch.zeros(1, 1), gamma=0.99, lam=0.95)

    assert torch.isfinite(storage.advantages).all()


def test_him_minibatch_singleton_normalization_stays_finite() -> None:
    """The optional source minibatch normalization also handles one row."""

    policy = HIMActorCritic(
        num_actor_obs=2,
        num_critic_obs=3,
        num_one_step_obs=2,
        num_actions=1,
        num_estimate=1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        estimator={
            "enc_hidden_dims": [4, 2],
            "tar_hidden_dims": [4, 2],
            "velocity_target_start": 2,
            "target_obs_start": 1,
        },
    )
    algorithm = HIMPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        normalize_advantage_per_mini_batch=True,
        device="cpu",
    )
    algorithm.init_storage(1, 1, (2,), (3,), (1,))
    obs = torch.zeros(1, 2)
    critic = torch.zeros(1, 3)
    algorithm.act(obs, critic)
    algorithm.process_env_step(torch.ones(1), torch.ones(1, dtype=torch.bool), {}, critic)
    algorithm.compute_returns(critic)

    metrics = algorithm.update()
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    assert all(torch.isfinite(torch.as_tensor(value)) for value in metrics)
