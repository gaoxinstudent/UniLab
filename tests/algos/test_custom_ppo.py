from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from unilab.algos.torch.custom_ppo.algorithm import NP3O, DreamWaQPPO
from unilab.algos.torch.custom_ppo.models import DreamWaQActorCritic, NP3OActorCritic
from unilab.algos.torch.custom_ppo.storage import CustomRolloutStorage


@pytest.mark.parametrize(
    ("algorithm", "policy"),
    [
        (DreamWaQPPO, DreamWaQActorCritic),
        (NP3O, NP3OActorCritic),
    ],
)
def test_custom_runtime_completes_representation_update(algorithm, policy) -> None:
    torch.manual_seed(4)
    model = policy(35, 78, 6, num_actor_history=2)
    alg = algorithm(
        model,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
    )
    if isinstance(alg, NP3O):
        alg.init_storage(2, 2, 70, 78, 6, "cpu", 5)
    else:
        alg.init_storage(2, 2, 70, 78, 6, "cpu")
    observations = torch.randn(2, 70)
    critic_observations = torch.randn(2, 78)
    for _ in range(2):
        alg.act(observations, critic_observations)
        next_critic = torch.randn(2, 78)
        extras = {"costs": torch.rand(2, 5)}
        alg.process_env_step(next_critic, torch.randn(2), torch.zeros(2, dtype=torch.bool), extras)
        observations, critic_observations = torch.randn(2, 70), next_critic
    alg.compute_returns(critic_observations)
    losses = alg.update()
    assert losses
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_dreamwaq_cenet_forward_keeps_source_eight_value_order() -> None:
    model = DreamWaQActorCritic(35, 78, 6, num_actor_history=2, num_estimate=4)
    assert isinstance(model.encoder[-1], torch.nn.ELU)
    outputs = model.cenet_forward(torch.zeros(2, 70), sample=False)
    assert len(outputs) == 8
    (
        code,
        code_vel,
        code_latent,
        decode,
        mean_vel,
        logvar_vel,
        mean_latent,
        logvar_latent,
    ) = outputs
    assert code.shape == (2, 20)
    assert code_vel.shape == mean_vel.shape == logvar_vel.shape == (2, 4)
    assert code_latent.shape == mean_latent.shape == logvar_latent.shape == (2, 16)
    assert decode.shape == (2, 35)


def test_dreamwaq_update_uses_current_critic_velocity_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Velocity is aligned with the encoded current history, not the next frame."""

    policy = _small_dreamwaq()
    algorithm = DreamWaQPPO(
        policy,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    algorithm.init_storage(1, 1, 2, 3, 1, "cpu")
    history = torch.tensor([[0.25, -0.5]])
    current_critic = torch.tensor([[0.25, -0.5, 3.0]])
    next_critic = torch.tensor([[5.0, 6.0, 9.0]])
    algorithm.act(history, current_critic)
    algorithm.process_env_step(
        next_critic,
        torch.ones(1),
        torch.zeros(1, dtype=torch.bool),
        {},
    )
    algorithm.compute_returns(next_critic)

    seen: list[torch.Tensor] = []
    original = algorithm._velocity_target

    def capture(value: torch.Tensor) -> torch.Tensor:
        seen.append(value.detach().clone())
        return original(value)

    monkeypatch.setattr(algorithm, "_velocity_target", capture)
    algorithm.update()

    assert len(seen) == 1
    torch.testing.assert_close(seen[0], current_critic)
    torch.testing.assert_close(original(seen[0]), torch.tensor([[3.0]]))


def test_dreamwaq_source_runner_storage_and_transition_aliases() -> None:
    """Source mapping calls normalize to the compact tensor storage owner."""

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
    algorithm = DreamWaQPPO(policy, device="cpu")
    algorithm.init_storage(
        2,
        1,
        {"policy": 28, "policy_hist": 140, "critic": 32, "prev_critic": 32},
        6,
    )
    obs = {
        "policy": torch.zeros(2, 28),
        "policy_hist": torch.zeros(2, 140),
        "critic": torch.zeros(2, 32),
    }
    algorithm.act(obs)
    assert "actions" in algorithm.transition
    # Upstream order: rewards, dones, extras, next_obs_dict.
    algorithm.process_env_step(
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
        {},
        {"policy_hist": torch.ones(2, 140), "critic": torch.ones(2, 32)},
    )
    assert algorithm.storage is not None
    assert algorithm.storage.step == 1
    assert algorithm.transition == {}


def test_dreamwaq_source_keyword_process_alias_and_cold_start_policy() -> None:
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
    algorithm = DreamWaQPPO(policy, device="cpu")
    algorithm.init_storage(1, 1, {"policy": 28, "critic": 32}, 6)
    obs = {"policy": torch.zeros(1, 28), "critic": torch.zeros(1, 32)}
    algorithm.act(obs)
    algorithm.process_env_step(
        rewards=torch.ones(1),
        dones=torch.zeros(1, dtype=torch.bool),
        extras={},
        next_obs_dict=obs,
    )
    assert algorithm.storage is not None
    assert algorithm.storage.step == 1


def _small_dreamwaq(*, adaboot_mode: str = "off") -> DreamWaQActorCritic:
    return DreamWaQActorCritic(
        2,
        3,
        1,
        num_actor_history=1,
        num_estimate=1,
        latent_dim=2,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[4],
        adaboot_mode=adaboot_mode,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"adaboot_mode": "unknown"}, "Unsupported DreamWaQ adaboot_mode"),
        ({"adaboot_reward_window_size": 0}, "positive integer"),
        ({"adaboot_reward_window_size": 1.5}, "positive integer"),
        ({"adaboot_reward_cv_scale": True}, "finite number"),
        ({"adaboot_reward_cv_scale": -1.0}, "non-negative"),
        ({"adaboot_pboot_min": 0.8, "adaboot_pboot_max": 0.2}, "bounds"),
    ],
)
def test_dreamwaq_adaboot_config_is_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DreamWaQPPO(_small_dreamwaq(), device="cpu", **kwargs)


def test_dreamwaq_policy_rejects_algorithm_only_adaboot_knobs() -> None:
    with pytest.raises(ValueError, match="reward-window knobs"):
        DreamWaQActorCritic(
            2,
            3,
            1,
            num_actor_history=1,
            num_estimate=1,
            latent_dim=2,
            actor_hidden_dims=[4],
            critic_hidden_dims=[4],
            encoder_hidden_dims=[4],
            decoder_hidden_dims=[4],
            adaboot_reward_window_size=4,
        )


def test_dreamwaq_reward_cv_window_matches_source_formula() -> None:
    algorithm = DreamWaQPPO(
        _small_dreamwaq(adaboot_mode="reward_cv"),
        device="cpu",
        adaboot_reward_window_size=4,
        adaboot_reward_cv_scale=0.25,
    )
    algorithm.init_storage(2, 2, 2, 3, 1, "cpu")
    assert algorithm.storage is not None
    algorithm.storage.rewards.copy_(torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]]))
    algorithm.storage.dones.copy_(torch.tensor([[[False], [False]], [[True], [True]]]))

    p_boot, reward_mean, reward_std, cv = algorithm._compute_p_boot_from_episode_rewards()

    torch.testing.assert_close(reward_mean, torch.tensor(5.0))
    torch.testing.assert_close(reward_std, torch.tensor(2.0))
    torch.testing.assert_close(cv, torch.tensor(0.4))
    torch.testing.assert_close(p_boot, 1.0 - torch.tanh(torch.tensor(0.1)))
    assert list(algorithm._ep_return_window) == [3.0, 7.0]


def test_dreamwaq_reward_cv_updates_policy_and_metrics() -> None:
    policy = _small_dreamwaq(adaboot_mode="reward_cv")
    algorithm = DreamWaQPPO(
        policy,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
        adaboot_reward_cv_scale=0.25,
    )
    algorithm.init_storage(2, 2, 2, 3, 1, "cpu")
    observations = torch.randn(2, 2)
    critic_observations = torch.randn(2, 3)
    for reward in (1.0, 3.0):
        algorithm.act(observations, critic_observations)
        next_critic = torch.randn(2, 3)
        algorithm.process_env_step(
            next_critic,
            torch.tensor([reward, reward + 1.0]),
            torch.ones(2, dtype=torch.bool),
            {},
        )
        observations, critic_observations = torch.randn(2, 2), next_critic
    algorithm.compute_returns(critic_observations)

    metrics = algorithm.update()

    for key in (
        "adaboot_coef",
        "adaboot_p_boot",
        "adaboot_r_mean",
        "adaboot_r_std",
        "adaboot_cv_r",
    ):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))
    assert policy.adaboot_p_boot is not None
    assert metrics["adaboot_p_boot"] == pytest.approx(float(policy.adaboot_p_boot))
    assert metrics["adaboot_coef"] == pytest.approx(metrics["adaboot_p_boot"])


def test_dreamwaq_reward_cv_coefficient_is_not_inert() -> None:
    policy = _small_dreamwaq(adaboot_mode="reward_cv")
    code_vel = torch.ones(2, 1)
    mean_vel = torch.zeros(2, 1)
    logvar_vel = torch.zeros(2, 1)
    policy.set_adaboot_p_boot(0.0)
    teacher_only = policy._adaboot_velocity(code_vel, mean_vel, logvar_vel)
    policy.set_adaboot_p_boot(1.0)
    code_only = policy._adaboot_velocity(code_vel, mean_vel, logvar_vel)
    torch.testing.assert_close(teacher_only, torch.zeros_like(code_vel))
    torch.testing.assert_close(code_only, code_vel)


def test_dreamwaq_adaboot_uses_privileged_target_only_for_stochastic_act() -> None:
    """The source actor blends critic velocity during act, never inference."""

    policy = _small_dreamwaq(adaboot_mode="reward_cv")
    policy.set_adaboot_p_boot(0.0)
    history = torch.zeros(2, 2)
    critic = torch.zeros(2, 3)
    critic[:, 2] = 7.0

    assert policy._teacher_velocity_from_obs({"critic": critic}) is not None
    torch.testing.assert_close(
        policy._teacher_velocity_from_obs({"critic": critic}), critic[:, 2:3]
    )
    # ``act`` receives the mapping through the algorithm owner and therefore
    # records the source target path; the direct feature call makes the
    # resulting teacher-only code observable without depending on actor weights.
    features = policy._actor_features(
        {"policy_hist": history, "policy": history, "critic": critic},
        sample=False,
        apply_adaboot=True,
        teacher_velocity=critic[:, 2:3],
    )
    assert features.shape == (2, 5)
    assert policy.last_adaboot_alpha is not None

    policy.last_adaboot_alpha = None
    policy.act_inference({"policy_hist": history, "policy": history, "critic": critic})
    assert policy.last_adaboot_alpha is None


def test_dreamwaq_adaboot_metric_keeps_source_p_boot_fallback_when_off() -> None:
    algorithm = DreamWaQPPO(_small_dreamwaq(), device="cpu")
    p_boot = torch.tensor(0.37)
    # Source PPODreamWaq reports p_boot when the policy's optional AdaBoot
    # path is off (its post-forward alpha is None), rather than replacing the
    # statistic with a hard-coded zero.
    assert algorithm._adaboot_alpha_metric(p_boot) == pytest.approx(0.37)


def test_dreamwaq_adaboot_state_roundtrip_preserves_cross_rollout_window() -> None:
    """A resumed run must continue AdaBoot's episode-return statistics."""

    first = DreamWaQPPO(
        _small_dreamwaq(adaboot_mode="reward_cv"),
        device="cpu",
        adaboot_reward_window_size=4,
    )
    first.init_storage(2, 1, 2, 3, 1, "cpu")
    assert first._ep_partial_returns is not None
    first._ep_partial_returns.copy_(torch.tensor([1.25, -2.5]))
    first._ep_return_window.extend([3.0, 5.0])
    first._last_adaboot_stats["cv_r"] = 0.75
    first.policy.set_adaboot_p_boot(0.42)
    state = first.algorithm_state_dict()

    second = DreamWaQPPO(
        _small_dreamwaq(adaboot_mode="reward_cv"),
        device="cpu",
        adaboot_reward_window_size=4,
    )
    second.init_storage(2, 1, 2, 3, 1, "cpu")
    second.load_algorithm_state_dict(state)
    assert list(second._ep_return_window) == [3.0, 5.0]
    assert second._ep_partial_returns is not None
    torch.testing.assert_close(second._ep_partial_returns, torch.tensor([1.25, -2.5]))
    assert second._last_adaboot_stats["cv_r"] == pytest.approx(0.75)
    assert second.policy.adaboot_p_boot == pytest.approx(0.42)


def test_dreamwaq_adaboot_state_rejects_window_or_mode_mismatch() -> None:
    first = DreamWaQPPO(
        _small_dreamwaq(adaboot_mode="reward_cv"),
        device="cpu",
        adaboot_reward_window_size=4,
    )
    first.init_storage(1, 1, 2, 3, 1, "cpu")
    state = first.algorithm_state_dict()

    different_window = DreamWaQPPO(
        _small_dreamwaq(adaboot_mode="reward_cv"),
        device="cpu",
        adaboot_reward_window_size=8,
    )
    different_window.init_storage(1, 1, 2, 3, 1, "cpu")
    with pytest.raises(ValueError, match="window"):
        different_window.load_algorithm_state_dict(state)

    different_mode = DreamWaQPPO(_small_dreamwaq(adaboot_mode="off"), device="cpu")
    different_mode.init_storage(1, 1, 2, 3, 1, "cpu")
    with pytest.raises(ValueError, match="mode"):
        different_mode.load_algorithm_state_dict(state)


def test_np3o_constructor_rejects_zero_cost_policy() -> None:
    class _ZeroCostPolicy:
        num_costs = 0

    with pytest.raises(ValueError, match="exactly five"):
        NP3O(_ZeroCostPolicy(), device="cpu")


def test_np3o_constructor_rejects_zero_cost_override() -> None:
    class _FiveCostPolicy:
        num_costs = 5

    with pytest.raises(ValueError, match="exactly five"):
        NP3O(_FiveCostPolicy(), device="cpu", num_costs=0)


def _timeout_transition(*, batch_size: int = 2, critic_dim: int = 3) -> dict[str, torch.Tensor]:
    return {
        "observations": torch.zeros(batch_size, 2),
        "critic_observations": torch.zeros(batch_size, critic_dim),
        "next_critic_observations": torch.zeros(batch_size, critic_dim),
        "actions": torch.zeros(batch_size, 1),
        "rewards": torch.tensor([1.0, 2.0]),
        "dones": torch.tensor([True, False]),
        "values": torch.tensor([[7.0], [8.0]]),
        "log_probs": torch.zeros(batch_size),
        "mu": torch.zeros(batch_size, 1),
        "sigma": torch.ones(batch_size, 1),
    }


class _TimeoutPolicy:
    num_costs = 2

    @staticmethod
    def evaluate(critic: torch.Tensor) -> torch.Tensor:
        return critic[:, :1]

    @staticmethod
    def evaluate_cost(critic: torch.Tensor) -> torch.Tensor:
        return critic[:, :2]


def test_dreamwaq_bootstraps_timeout_from_final_critic_observation() -> None:
    algorithm = object.__new__(DreamWaQPPO)
    algorithm.device = "cpu"
    algorithm.gamma = 0.5
    algorithm.policy = _TimeoutPolicy()
    algorithm.storage = CustomRolloutStorage(2, 1, 2, 3, 1, "cpu")
    algorithm._transition = _timeout_transition()

    final_critic = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    algorithm.process_env_step(
        torch.zeros(2, 3),
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, False]),
        {
            "time_outs": torch.tensor([True, False]),
            "time_out_bootstrap_obs": TensorDict({"critic": final_critic}, batch_size=[2]),
        },
    )

    assert algorithm.storage is not None
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([2.5, 2.0]))
    torch.testing.assert_close(
        algorithm.storage.next_critic_observations[0],
        torch.tensor([[3.0, 4.0, 5.0], [0.0, 0.0, 0.0]]),
    )


def test_np3o_timeout_bootstrap_updates_reward_and_cost_targets() -> None:
    algorithm = object.__new__(NP3O)
    algorithm.device = "cpu"
    algorithm.gamma = 0.5
    algorithm.policy = _TimeoutPolicy()
    algorithm.storage = CustomRolloutStorage(2, 1, 2, 3, 1, "cpu", num_costs=2)
    transition = _timeout_transition()
    transition.update(
        costs=torch.ones(2, 2),
        cost_values=torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
    )
    algorithm._transition = transition

    final_critic = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    algorithm.process_env_step(
        torch.zeros(2, 3),
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, False]),
        {
            "costs": torch.ones(2, 2),
            "time_outs": torch.tensor([True, False]),
            "time_out_bootstrap_obs": TensorDict({"critic": final_critic}, batch_size=[2]),
        },
    )

    assert algorithm.storage is not None
    assert algorithm.storage.costs is not None
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([2.5, 2.0]))
    torch.testing.assert_close(algorithm.storage.costs[0], torch.tensor([[2.5, 3.0], [1.0, 1.0]]))


def test_np3o_timeout_step_requires_cost_channels() -> None:
    algorithm = object.__new__(NP3O)
    algorithm.device = "cpu"
    algorithm.gamma = 0.5
    algorithm.policy = _TimeoutPolicy()
    algorithm.storage = CustomRolloutStorage(2, 1, 2, 3, 1, "cpu", num_costs=2)
    algorithm._transition = _timeout_transition()
    with pytest.raises(ValueError, match="five cost channels"):
        algorithm.process_env_step(
            torch.zeros(2, 3),
            torch.tensor([1.0, 2.0]),
            torch.tensor([True, False]),
            {"time_outs": torch.tensor([True, False])},
        )


def test_custom_timeout_bootstrap_falls_back_to_transition_value() -> None:
    algorithm = object.__new__(DreamWaQPPO)
    algorithm.device = "cpu"
    algorithm.gamma = 0.5
    algorithm.policy = _TimeoutPolicy()
    algorithm.storage = CustomRolloutStorage(2, 1, 2, 3, 1, "cpu")
    algorithm._transition = _timeout_transition()

    algorithm.process_env_step(
        torch.zeros(2, 3),
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, False]),
        {"time_outs": torch.tensor([True, False])},
    )

    assert algorithm.storage is not None
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([4.5, 2.0]))


def test_custom_storage_singleton_advantages_remain_finite() -> None:
    """Tiny source smoke runs must not turn unbiased ``std(n=1)`` into NaN."""

    storage = CustomRolloutStorage(
        num_envs=1,
        num_steps=1,
        actor_dim=2,
        critic_dim=3,
        action_dim=1,
        device="cpu",
        num_costs=5,
    )
    transition = {
        "observations": torch.zeros(1, 2),
        "critic_observations": torch.zeros(1, 3),
        "next_critic_observations": torch.zeros(1, 3),
        "actions": torch.zeros(1, 1),
        "rewards": torch.ones(1),
        "dones": torch.ones(1, dtype=torch.bool),
        "values": torch.zeros(1),
        "log_probs": torch.zeros(1),
        "mu": torch.zeros(1, 1),
        "sigma": torch.ones(1, 1),
        "costs": torch.ones(1, 5),
        "cost_values": torch.zeros(1, 5),
    }
    storage.add(transition)
    storage.compute_returns(torch.zeros(1, 1), gamma=0.99, lam=0.95)
    storage.compute_cost_returns(torch.zeros(1, 5), gamma=0.99, lam=0.95)
    assert torch.isfinite(storage.advantages).all()
    assert storage.cost_advantages is not None
    assert torch.isfinite(storage.cost_advantages).all()
    assert storage.cost_violation is not None
    assert torch.isfinite(storage.cost_violation).all()
