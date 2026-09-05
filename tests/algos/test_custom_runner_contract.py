from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from unilab.algos.torch.custom_ppo.models import SourceBarlowTwinsActorCritic
from unilab.algos.torch.custom_ppo.runner import (
    CustomOnPolicyRunner,
    _normalize_update_metrics,
    validate_custom_runner_contract,
)
from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic
from unilab.algos.torch.him_ppo.checkpoint import (
    dreamwaq_source_parameter_names,
    remap_source_adam_state_dict,
)
from unilab.envs.locomotion.wheelbipe_v14.variants import (
    WheelbipeHIMCfg,
    WheelbipeNP3OCfg,
)
from unilab.training.wheelbipe import (
    WheelbipeSourceBarlowOnnxPolicy,
    WheelbipeSourceBarlowTorchScriptPolicy,
    custom_wheelbipe_metadata_path,
    inspect_wheelbipe_source_barlow_onnx,
)


def _env(*, algorithm: str = "him", history: int = 5, costs: int = 0, num_envs: int = 2):
    return SimpleNamespace(
        num_envs=num_envs,
        num_obs=28,
        num_privileged_obs=32,
        num_actions=6,
        cfg=SimpleNamespace(
            custom_algorithm=algorithm,
            policy_observation_mode="compact",
            variant_name=f"flat-{algorithm}-v0",
            num_actor_history=history,
            num_estimate=4,
            num_costs=costs,
        ),
    )


def _cfg(algorithm: str = "him", *, history: int = 5, estimate: int = 4, costs: int = 0) -> dict:
    return {
        "algorithm_name": algorithm,
        "num_actor_history": history,
        "num_estimate": estimate,
        "num_costs": costs,
    }


def test_custom_contract_records_compact_algorithm_metadata() -> None:
    contract = validate_custom_runner_contract(_env(), _cfg())
    assert contract == {
        "algorithm": "him",
        "variant_name": "flat-him-v0",
        "actor_obs_dim": 28,
        "critic_obs_dim": 32,
        "history_length": 5,
        "history_reset_mode": "repeat",
        "num_estimate": 4,
        "num_actions": 6,
        "num_costs": 0,
    }


def test_custom_contract_accepts_upstream_class_style_variant_label() -> None:
    env = _env()
    env.cfg.variant_name = "WheelbipeV14FlatHIM"
    contract = validate_custom_runner_contract(env, _cfg())
    assert contract["algorithm"] == "him"


def test_source_barlow_contract_requires_explicit_source_history_reset() -> None:
    env = _env(algorithm="np3o", history=10, costs=5)
    cfg = _cfg("np3o", history=10, costs=5)
    cfg.update(
        {
            "policy_architecture": "source_barlow",
            "policy": {
                "architecture": "source_barlow",
                "source_barlow": {
                    "num_prop": 28,
                    "num_scan": 0,
                    "num_state_est": 4,
                    "num_priv_latent": 4,
                    "num_hist": 10,
                    "num_costs": 5,
                },
            },
        }
    )
    with pytest.raises(ValueError, match="history_reset_mode=source_zero_current"):
        validate_custom_runner_contract(env, cfg)


@pytest.mark.parametrize(
    ("algorithm", "history", "costs", "message"),
    [
        ("him", 1, 0, "history"),
        ("dreamwaq", 5, 1, "num_costs"),
        ("np3o", 10, 0, "five"),
    ],
)
def test_custom_contract_rejects_incompatible_overrides(
    algorithm: str, history: int, costs: int, message: str
) -> None:
    env_costs = costs
    if algorithm == "np3o" and costs == 0:
        # Keep the env metadata source-compatible so this assertion exercises
        # the algorithm-side override first.
        env_costs = 5
    with pytest.raises(ValueError, match=message):
        validate_custom_runner_contract(
            _env(algorithm=algorithm, history=10 if algorithm == "np3o" else 5, costs=env_costs),
            _cfg(algorithm, history=history, costs=costs),
        )


def test_np3o_owner_config_rejects_zero_cost_override() -> None:
    cfg = WheelbipeNP3OCfg()
    cfg.num_costs = 0
    with pytest.raises(ValueError, match="exactly five"):
        cfg.validate()


def test_np3o_contract_rejects_cost_limit_width_override() -> None:
    with pytest.raises(ValueError, match="cost_limits.*five"):
        validate_custom_runner_contract(
            _env(algorithm="np3o", history=10, costs=5),
            {
                **_cfg("np3o", history=10, costs=5),
                "algorithm": {"cost_limits": [0.0]},
            },
        )


def test_him_owner_config_keeps_cost_channels_disabled() -> None:
    cfg = WheelbipeHIMCfg()
    cfg.num_costs = 1
    with pytest.raises(ValueError, match="does not use cost channels"):
        cfg.validate()


def test_custom_owner_uses_source_active_actuator_limits() -> None:
    cfg = WheelbipeHIMCfg()
    assert cfg.control_config.leg_torque_limit == 40.0
    assert cfg.control_config.wheel_torque_limit == 5.0
    assert cfg.control_config.spring_torque_limit == 1000.0


def test_checkpoint_contract_rejects_algorithm_mismatch(tmp_path) -> None:
    env = _env()
    runner = CustomOnPolicyRunner(
        env,
        {
            **_cfg(),
            "num_steps_per_env": 2,
            "policy": {"actor_hidden_dims": [8], "critic_hidden_dims": [8]},
            "estimator": {
                "enc_hidden_dims": [8, 4],
                "tar_hidden_dims": [8, 4],
                "velocity_target_start": 28,
                "target_obs_start": 4,
            },
            "algorithm": {},
        },
    )
    checkpoint_path = tmp_path / "model.pt"
    runner.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["algorithm"] = "np3o"
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="checkpoint algorithm"):
        runner.load(str(checkpoint_path))


def test_checkpoint_contract_rejects_malformed_metadata(tmp_path) -> None:
    env = _env()
    runner = CustomOnPolicyRunner(env, _runner_cfg_for_checkpoint())
    checkpoint_path = tmp_path / "malformed.pt"
    runner.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["custom_contract"] = ["not", "a", "mapping"]
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="custom_contract.*mapping"):
        runner.load(str(checkpoint_path))


def test_legacy_actor_only_checkpoint_remains_loadable(tmp_path) -> None:
    env = _env()
    cfg = _runner_cfg_for_checkpoint()
    source = CustomOnPolicyRunner(env, cfg)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"actor_state_dict": source.policy.state_dict(), "iteration": 2}, checkpoint_path)

    restored = CustomOnPolicyRunner(env, cfg)
    restored.load(str(checkpoint_path))
    assert restored.current_learning_iteration == 2
    assert restored.tot_timesteps == 2 * restored.num_steps_per_env * env.num_envs


def test_source_checkpoint_key_aliases_resume_model_and_iteration(tmp_path) -> None:
    """Wheelbipe source runners' ``model_state_dict``/``iter`` keys load."""

    env = _env()
    cfg = _runner_cfg_for_checkpoint()
    source = CustomOnPolicyRunner(env, cfg)
    checkpoint_path = tmp_path / "source-him.pt"
    torch.save(
        {
            "model_state_dict": source.policy.state_dict(),
            "iter": 4,
        },
        checkpoint_path,
    )

    restored = CustomOnPolicyRunner(env, cfg)
    restored.load(str(checkpoint_path))
    assert restored.current_learning_iteration == 4
    assert restored.tot_timesteps == 4 * restored.num_steps_per_env * env.num_envs


def _source_wrapped_mlp_state(
    state: dict[str, torch.Tensor], roots: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        converted = key
        for root in roots:
            prefix = f"{root}."
            if key.startswith(prefix):
                converted = f"{root}.model.{key[len(prefix) :]}"
                break
        result[converted] = value.clone()
    return result


@pytest.mark.parametrize(
    ("algorithm", "roots"),
    [
        ("him", ("actor", "critic")),
        ("dreamwaq", ("actor", "critic", "encoder", "decoder")),
    ],
)
def test_source_wrapped_mlp_checkpoint_strict_loads_complete_graph(
    tmp_path: Path,
    algorithm: str,
    roots: tuple[str, ...],
) -> None:
    """Pinned source MLP wrappers add one ``model`` segment to every MLP key."""

    env = _env(algorithm=algorithm)
    cfg = _runner_cfg_for_checkpoint(algorithm)
    source = CustomOnPolicyRunner(env, cfg)
    source_state = _source_wrapped_mlp_state(source.policy.state_dict(), roots)
    for root in roots:
        assert any(key.startswith(f"{root}.model.") for key in source_state)
        assert not any(
            key.startswith(f"{root}.") and not key.startswith(f"{root}.model.")
            for key in source_state
        )
    checkpoint_path = tmp_path / f"source-{algorithm}.pt"
    torch.save({"model_state_dict": source_state, "iter": 6}, checkpoint_path)

    restored = CustomOnPolicyRunner(env, cfg)
    restored.load(str(checkpoint_path))

    assert restored.current_learning_iteration == 6
    for key, expected in source.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[key], expected)


def test_source_wrapped_mlp_checkpoint_rejects_partial_native_source_mix(tmp_path: Path) -> None:
    env = _env()
    cfg = _runner_cfg_for_checkpoint()
    source = CustomOnPolicyRunner(env, cfg)
    # Actor uses the pinned source spelling while critic remains native.  A
    # permissive per-key replacement would accept this malformed hybrid.
    partial = _source_wrapped_mlp_state(source.policy.state_dict(), ("actor",))
    checkpoint_path = tmp_path / "partial-source-him.pt"
    torch.save({"model_state_dict": partial}, checkpoint_path)

    restored = CustomOnPolicyRunner(env, cfg)
    with pytest.raises(ValueError, match="complete source MLP graph|cannot be mixed"):
        restored.load(str(checkpoint_path))


def test_source_wrapped_mlp_checkpoint_rejects_unknown_model_root(tmp_path: Path) -> None:
    env = _env(algorithm="dreamwaq")
    cfg = _runner_cfg_for_checkpoint("dreamwaq")
    source = CustomOnPolicyRunner(env, cfg)
    state = _source_wrapped_mlp_state(
        source.policy.state_dict(), ("actor", "critic", "encoder", "decoder")
    )
    state["unknown.model.0.weight"] = torch.zeros(1)
    checkpoint_path = tmp_path / "unknown-source-dreamwaq.pt"
    torch.save({"model_state_dict": state}, checkpoint_path)

    restored = CustomOnPolicyRunner(env, cfg)
    with pytest.raises(ValueError, match="unknown source MLP key"):
        restored.load(str(checkpoint_path))


def test_source_wrapped_mlp_checkpoint_rejects_hidden_shape_mismatch(tmp_path: Path) -> None:
    env = _env()
    cfg = _runner_cfg_for_checkpoint()
    source = CustomOnPolicyRunner(env, cfg)
    state = _source_wrapped_mlp_state(source.policy.state_dict(), ("actor", "critic"))
    state["actor.model.0.weight"] = state["actor.model.0.weight"][:-1].clone()
    checkpoint_path = tmp_path / "wrong-source-him-shape.pt"
    torch.save({"model_state_dict": state}, checkpoint_path)

    restored = CustomOnPolicyRunner(env, cfg)
    with pytest.raises(ValueError, match="tensors do not match"):
        restored.load(str(checkpoint_path))


def _source_adam_state(
    runner: CustomOnPolicyRunner,
    optimizer: torch.optim.Adam,
    source_names: tuple[str, ...],
    *,
    base: float,
) -> tuple[dict[str, object], dict[str, float]]:
    parameters = dict(runner.policy.named_parameters())
    group = dict(optimizer.state_dict()["param_groups"][0])
    source_ids = [1000 + index for index in range(len(source_names))]
    group["params"] = source_ids
    expected: dict[str, float] = {}
    slots: dict[int, dict[str, torch.Tensor]] = {}
    for index, (source_id, name) in enumerate(zip(source_ids, source_names)):
        sentinel = base + float(index + 1)
        expected[name] = sentinel
        parameter = parameters[name]
        slots[source_id] = {
            "step": torch.tensor(float(index + 1)),
            "exp_avg": torch.full_like(parameter, sentinel),
            "exp_avg_sq": torch.full_like(parameter, sentinel + 0.25),
        }
    return {"state": slots, "param_groups": [group]}, expected


def test_source_dreamwaq_resume_reorders_main_and_vae_adam_by_explicit_names(
    tmp_path: Path,
) -> None:
    env = _env(algorithm="dreamwaq")
    cfg = _runner_cfg_for_checkpoint("dreamwaq")
    source = CustomOnPolicyRunner(env, cfg)
    source_names = dreamwaq_source_parameter_names(source.policy)
    source_vae_names = dreamwaq_source_parameter_names(source.policy, cenet_only=True)
    main_state, expected_main = _source_adam_state(
        source,
        source.alg.optimizer,
        source_names,
        base=10.0,
    )
    vae_state, expected_vae = _source_adam_state(
        source,
        source.alg.vae_optimizer,
        source_vae_names,
        base=100.0,
    )
    checkpoint_path = tmp_path / "source-dreamwaq-resume.pt"
    torch.save(
        {
            "model_state_dict": _source_wrapped_mlp_state(
                source.policy.state_dict(), ("actor", "critic", "encoder", "decoder")
            ),
            "optimizer_state_dict": main_state,
            "vae_optimizer_state_dict": vae_state,
            "iter": 7,
        },
        checkpoint_path,
    )

    restored = CustomOnPolicyRunner(env, cfg)
    restored.load(str(checkpoint_path))

    for name, parameter in restored.policy.named_parameters():
        slot = restored.alg.optimizer.state[parameter]
        torch.testing.assert_close(slot["exp_avg"], torch.full_like(parameter, expected_main[name]))
    vae_name_by_id = {id(parameter): name for name, parameter in restored.policy.named_parameters()}
    for parameter in restored.alg.vae_optimizer.param_groups[0]["params"]:
        name = vae_name_by_id[id(parameter)]
        slot = restored.alg.vae_optimizer.state[parameter]
        torch.testing.assert_close(slot["exp_avg"], torch.full_like(parameter, expected_vae[name]))

    # A real resumed update must be able to consume every restored moment;
    # shape-only construction tests would miss a slot attached to the wrong
    # equal-width parameter.
    restored.alg.optimizer.zero_grad(set_to_none=True)
    torch.stack(
        [parameter.square().mean() for parameter in restored.policy.parameters()]
    ).sum().backward()
    restored.alg.optimizer.step()
    restored.alg.vae_optimizer.zero_grad(set_to_none=True)
    torch.stack(
        [
            parameter.square().mean()
            for parameter in restored.alg.vae_optimizer.param_groups[0]["params"]
        ]
    ).sum().backward()
    restored.alg.vae_optimizer.step()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("groups", "exactly one parameter group"),
        ("slot_key", "slot keys mismatch"),
        ("slot_shape", "shape mismatch"),
    ],
)
def test_source_dreamwaq_adam_remap_rejects_malformed_schema(
    mutation: str,
    message: str,
) -> None:
    runner = CustomOnPolicyRunner(
        _env(algorithm="dreamwaq"), _runner_cfg_for_checkpoint("dreamwaq")
    )
    source_names = dreamwaq_source_parameter_names(runner.policy)
    state, _expected = _source_adam_state(
        runner,
        runner.alg.optimizer,
        source_names,
        base=1.0,
    )
    if mutation == "groups":
        state["param_groups"] = [*state["param_groups"], dict(state["param_groups"][0])]  # type: ignore[index]
    else:
        first_id = state["param_groups"][0]["params"][0]  # type: ignore[index]
        first_slot = state["state"][first_id]  # type: ignore[index]
        if mutation == "slot_key":
            first_slot["unknown_moment"] = torch.zeros(())
        else:
            first_slot["exp_avg"] = first_slot["exp_avg"].reshape(-1)[:-1]

    with pytest.raises(ValueError, match=message):
        remap_source_adam_state_dict(
            runner.policy,
            runner.alg.optimizer,
            state,
            source_parameter_names=source_names,
            label="DreamWaQ policy Adam",
        )


def test_source_runner_loads_pure_direct_tensor_checkpoint(tmp_path) -> None:
    """Source lightweight exporters may omit the RSL-RL metadata wrapper."""

    env = _SourceBarlowEnv(num_envs=1)
    cfg = _source_runner_cfg()
    source = CustomOnPolicyRunner(env, cfg)
    checkpoint_path = tmp_path / "source-direct.pt"
    torch.save(source.policy.state_dict(), checkpoint_path)

    restored = CustomOnPolicyRunner(env, cfg)
    restored.load(str(checkpoint_path))
    assert restored.policy_architecture == "source_barlow"


def test_custom_checkpoint_emits_source_aliases_and_infos(tmp_path) -> None:
    env = _env()
    runner = CustomOnPolicyRunner(env, _runner_cfg_for_checkpoint())
    runner.current_learning_iteration = 3
    checkpoint_path = tmp_path / "source-aliases.pt"

    runner.save(str(checkpoint_path), infos={"score": 1.25}, is_best=True)
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    assert "model_state_dict" in checkpoint
    assert "iter" in checkpoint
    assert "vae_optimizer_state_dict" not in checkpoint
    assert checkpoint["infos"] == {"score": 1.25}

    restored = CustomOnPolicyRunner(env, _runner_cfg_for_checkpoint())
    infos = restored.load(str(checkpoint_path), load_optimizer=False, load_iteration=False)
    assert infos == {"score": 1.25}
    assert restored.current_learning_iteration == 0
    assert restored.tot_timesteps == 0


def test_custom_checkpoint_source_flags_restore_optimizer_and_iteration_independently(
    tmp_path,
) -> None:
    env = _env()
    cfg = _runner_cfg_for_checkpoint()
    source = CustomOnPolicyRunner(env, cfg)
    parameter = next(source.policy.parameters())
    source.alg.optimizer.zero_grad()
    parameter.sum().backward()
    source.alg.optimizer.step()
    source.current_learning_iteration = 4
    source.tot_timesteps = 88
    checkpoint_path = tmp_path / "source-flags.pt"
    source.save(str(checkpoint_path))

    weights_only = CustomOnPolicyRunner(env, cfg)
    weights_only.load(str(checkpoint_path), load_optimizer=False, load_iteration=False)
    assert weights_only.current_learning_iteration == 0
    assert weights_only.tot_timesteps == 0
    assert not weights_only.alg.optimizer.state

    continuation = CustomOnPolicyRunner(env, cfg)
    continuation.load(str(checkpoint_path), load_optimizer=True, load_iteration=True)
    assert continuation.current_learning_iteration == 4
    assert continuation.tot_timesteps == 88
    assert continuation.alg.optimizer.state


def test_source_dreamwaq_vae_optimizer_alias_is_restored(tmp_path) -> None:
    """The source DreamWaQ VAE optimizer spelling maps to CENet ownership."""

    env = _env(algorithm="dreamwaq")
    cfg = _runner_cfg_for_checkpoint("dreamwaq")
    source = CustomOnPolicyRunner(env, cfg)
    checkpoint_path = tmp_path / "source-dreamwaq.pt"
    torch.save(
        {
            "model_state_dict": source.policy.state_dict(),
            "vae_optimizer_state_dict": source.alg.cenet_optimizer.state_dict(),
            "iter": 3,
        },
        checkpoint_path,
    )

    restored = CustomOnPolicyRunner(env, cfg)
    restored.load(str(checkpoint_path))
    assert restored.current_learning_iteration == 3


def _runner_cfg_for_checkpoint(algorithm: str = "him") -> dict:
    costs = 5 if algorithm == "np3o" else 0
    cfg = {
        **_cfg(algorithm, history=10 if algorithm == "np3o" else 5, costs=costs),
        "num_steps_per_env": 2,
        "policy": {"actor_hidden_dims": [8], "critic_hidden_dims": [8]},
        "estimator": {
            "enc_hidden_dims": [8, 4],
            "tar_hidden_dims": [8, 4],
            "velocity_target_start": 28,
            "target_obs_start": 4,
        },
        "algorithm": {},
    }
    if algorithm == "np3o":
        cfg["algorithm"] = {
            "cost_limits": [0.0] * 5,
            "cost_d_values": [0.0] * 5,
            "cost_k_initial": [1.0, 1.0, 1.0, 0.5, 0.5],
        }
    return cfg


def test_custom_checkpoint_restores_resume_state_and_optimizer_moments(tmp_path) -> None:
    env = _env()
    cfg = _runner_cfg_for_checkpoint()
    source = CustomOnPolicyRunner(env, cfg)

    # Create Adam moments in both owner optimizers; a state-less optimizer
    # load would otherwise pass while silently restarting adaptation.
    parameter = next(source.policy.parameters())
    source.alg.optimizer.zero_grad()
    parameter.sum().backward()
    source.alg.optimizer.step()
    him_policy = cast(HIMActorCritic, source.policy)
    estimator_parameter = next(him_policy.estimator.parameters())
    him_policy.estimator.optimizer.zero_grad()
    estimator_parameter.sum().backward()
    him_policy.estimator.optimizer.step()
    source.current_learning_iteration = 7
    source.tot_timesteps = 1234
    source.alg.learning_rate = 2.5e-4

    checkpoint_path = tmp_path / "resume.pt"
    source.save(str(checkpoint_path))

    restored = CustomOnPolicyRunner(env, cfg)
    restored.load(str(checkpoint_path))

    assert restored.current_learning_iteration == 7
    assert restored.tot_timesteps == 1234
    assert restored.alg.learning_rate == pytest.approx(2.5e-4)
    assert restored.alg.optimizer.state
    restored_him_policy = cast(HIMActorCritic, restored.policy)
    assert restored_him_policy.estimator.optimizer.state
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert "total_timesteps" in checkpoint
    assert "estimator_optimizer_state_dict" in checkpoint
    assert "algorithm_state" in checkpoint


def test_dreamwaq_playback_can_drop_env_scoped_resume_state(tmp_path) -> None:
    """Inference may use a different env count than a training checkpoint.

    DreamWaQ persists AdaBoot partial returns with one value per training
    environment.  A play-only runner must load policy weights without trying
    to restore that training-owner state, otherwise the usual 1-env evaluator
    cannot open a checkpoint produced with vectorized training.
    """

    train_env = _env(algorithm="dreamwaq", num_envs=2)
    cfg = _runner_cfg_for_checkpoint("dreamwaq")
    source = CustomOnPolicyRunner(train_env, cfg)
    checkpoint_path = tmp_path / "dreamwaq-vectorized.pt"
    source.save(str(checkpoint_path))

    play_env = _env(algorithm="dreamwaq", num_envs=1)
    playback = CustomOnPolicyRunner(play_env, cfg)
    playback.load(str(checkpoint_path), load_optimizer=False, load_iteration=False)
    assert playback.current_learning_iteration == 0
    assert playback.tot_timesteps == 0

    with pytest.raises(ValueError, match="owner algorithm state"):
        # The default load mode is intentionally strict and remains reserved
        # for same-shape training continuation.
        playback.load(str(checkpoint_path))


def test_np3o_checkpoint_restores_channel_schedule(tmp_path) -> None:
    env = _env(algorithm="np3o", history=10, costs=5)
    cfg = _runner_cfg_for_checkpoint("np3o")
    source = CustomOnPolicyRunner(env, cfg)
    source.alg.k_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    checkpoint_path = tmp_path / "np3o-resume.pt"
    source.save(str(checkpoint_path))

    restored = CustomOnPolicyRunner(env, cfg)
    restored.load(str(checkpoint_path))
    torch.testing.assert_close(
        restored.alg.k_value,
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
    )


def test_custom_runner_initializes_random_episode_length_buffer() -> None:
    runner = object.__new__(CustomOnPolicyRunner)
    runner.env = SimpleNamespace(
        episode_length_buf=torch.zeros(32),
        max_episode_length=17.0,
    )
    torch.manual_seed(5)
    runner._initialize_random_episode_lengths()
    assert torch.all(runner.env.episode_length_buf >= 0)
    assert torch.all(runner.env.episode_length_buf < 17)
    assert torch.any(runner.env.episode_length_buf != 0)


def test_custom_runner_normalizes_him_tuple_update_metrics() -> None:
    """The compact HIM owner returns a four-tuple, not key/value pairs."""

    metrics = _normalize_update_metrics((1.0, 2.0, 3.0, 4.0))
    assert metrics == {
        "value_function": 1.0,
        "surrogate": 2.0,
        "estimation": 3.0,
        "swap": 4.0,
    }


class _SourceBarlowEnv:
    """Minimal source observation adapter used to exercise the live runner path."""

    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self.num_obs = 28
        self.num_privileged_obs = 32
        self.num_actions = 6
        self.cfg = SimpleNamespace(
            custom_algorithm="np3o",
            policy_observation_mode="source",
            variant_name="flat-np3o-barlow-source-v0",
            num_actor_history=10,
            num_estimate=4,
            num_costs=5,
        )
        self._obs = self._make_obs()

    def _make_obs(self) -> dict[str, torch.Tensor]:
        return {
            "on_constraint": torch.randn(self.num_envs, 312),
        }

    def reset(self):
        self._obs = self._make_obs()
        return self._obs, {}

    def step(self, actions: torch.Tensor):
        assert actions.shape == (self.num_envs, 6)
        self._obs = self._make_obs()
        return (
            self._obs,
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {
                "costs": torch.zeros(self.num_envs, 5),
            },
        )


class _SourceCompactOnlyEnv(_SourceBarlowEnv):
    """Compact actor/critic payload; runner must materialize on_constraint."""

    def _make_obs(self) -> dict[str, torch.Tensor]:
        actor = torch.randn(self.num_envs, 28)
        privileged_tail = torch.randn(self.num_envs, 4)
        return {
            "actor": actor,
            "critic": torch.cat((actor, privileged_tail), dim=-1),
        }


def _source_runner_cfg() -> dict:
    return {
        "algorithm_name": "np3o",
        "policy_architecture": "source_barlow",
        "num_actor_history": 10,
        "history_reset_mode": "source_zero_current",
        "num_estimate": 4,
        "num_costs": 5,
        "num_steps_per_env": 2,
        "policy": {
            "architecture": "source_barlow",
            "actor_hidden_dims": [32],
            "critic_hidden_dims": [32],
            "source_barlow": {
                "num_prop": 28,
                "num_scan": 0,
                "num_state_est": 4,
                "num_priv_latent": 4,
                "num_hist": 10,
                "scan_encoder_dims": [128, 64, 32],
                "priv_encoder_dims": [],
                "teacher_act": True,
                "imi_flag": True,
            },
        },
        "algorithm": {
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-4,
            "cost_k_initial": [1.0, 1.0, 1.0, 0.5, 0.5],
            "cost_d_values": [0.0] * 5,
            "cost_limits": [0.0] * 5,
            "imi_flag": True,
        },
    }


def test_source_barlow_runner_constructs_rolls_updates_and_exports_contract(tmp_path) -> None:
    env = _SourceBarlowEnv()
    runner = CustomOnPolicyRunner(env, _source_runner_cfg(), device="cpu")
    assert isinstance(runner.policy, SourceBarlowTwinsActorCritic)
    assert runner.contract["actor_input_dim"] == 312
    runner.learn(1, init_at_random_ep_len=False)
    assert runner.last_update_metrics["imitation_loss"] >= 0.0
    infer = runner.get_inference_policy()
    assert infer(torch.zeros(1, 312)).shape == (1, 6)
    path = tmp_path / "source-model.pt"
    runner.save(str(path))
    restored = CustomOnPolicyRunner(env, _source_runner_cfg(), device="cpu")
    restored.load(str(path))
    assert restored.policy_architecture == "source_barlow"


def test_source_barlow_runner_accepts_upstream_flat_policy_dictionary() -> None:
    """The vendored source runner's flat ``policy`` config is loadable."""

    env = _SourceBarlowEnv(num_envs=1)
    cfg = _source_runner_cfg()
    cfg.pop("policy_architecture")
    policy_cfg = cfg["policy"]
    policy_cfg.pop("architecture")
    source_fields = policy_cfg.pop("source_barlow")
    policy_cfg.update(source_fields)
    policy_cfg["class_name"] = "ActorCriticBarlowTwins"
    runner = CustomOnPolicyRunner(env, cfg, device="cpu")
    assert runner.policy_architecture == "source_barlow"
    assert runner.contract["source_num_costs"] == 5
    assert runner.policy.continue_from_last_std is True


def test_source_barlow_runner_materializes_compact_on_constraint_stream() -> None:
    """A compact UniLab wrapper still gets the exact 28+4+280 source input."""

    env = _SourceCompactOnlyEnv()
    env.cfg.policy_observation_mode = "compact"
    runner = CustomOnPolicyRunner(env, _source_runner_cfg(), device="cpu")
    runner.learn(1, init_at_random_ep_len=False)
    assert runner._source_history is not None
    assert runner._source_history.shape == (env.num_envs, 10, 28)
    assert runner.last_update_metrics["imitation_loss"] >= 0.0


class _CompactHistorySequenceEnv:
    """Two-step autoreset sequence for the compact runner history owner."""

    def __init__(self) -> None:
        self.num_envs = 2
        self.num_obs = 28
        self.num_privileged_obs = 32
        self.num_actions = 6
        self.cfg = SimpleNamespace(
            custom_algorithm="him",
            policy_observation_mode="compact",
            variant_name="flat-him-v0",
            num_actor_history=5,
            num_estimate=4,
            num_costs=0,
        )
        self._step = 0

    @staticmethod
    def _payload(values: tuple[float, float]) -> dict[str, torch.Tensor]:
        actor = torch.zeros(2, 28)
        actor[:, 0] = torch.tensor(values)
        return {
            "actor": actor,
            "critic": torch.cat((actor, torch.zeros(2, 4)), dim=-1),
        }

    def reset(self):
        self._step = 0
        return self._payload((1.0, 2.0)), {}

    def step(self, actions: torch.Tensor):
        assert actions.shape == (2, 6)
        self._step += 1
        dones = torch.tensor([self._step == 1, False])
        return self._payload((10.0, 20.0)), torch.zeros(2), dones, {}


def test_source_history_train_reset_and_done_keep_oldest_to_newest_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _CompactHistorySequenceEnv()
    cfg = _runner_cfg_for_checkpoint()
    cfg["history_reset_mode"] = "source_zero_current"
    runner = CustomOnPolicyRunner(env, cfg)
    histories: list[torch.Tensor] = []

    def act(history: torch.Tensor, _critic: torch.Tensor) -> torch.Tensor:
        histories.append(history.detach().clone())
        return torch.zeros(2, 6)

    monkeypatch.setattr(runner.alg, "act", act)
    monkeypatch.setattr(runner.alg, "process_env_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.alg, "compute_returns", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.alg, "update", lambda: (0.0, 0.0, 0.0, 0.0))
    runner.learn(1, init_at_random_ep_len=False)

    first = histories[0].reshape(2, 5, 28)
    torch.testing.assert_close(first[:, :-1], torch.zeros(2, 4, 28))
    torch.testing.assert_close(first[:, -1, 0], torch.tensor([1.0, 2.0]))
    second = histories[1].reshape(2, 5, 28)
    # Row zero autoreset after step one: four zeros plus the reset frame.
    torch.testing.assert_close(second[0, :-1], torch.zeros(4, 28))
    assert second[0, -1, 0].item() == 10.0
    # Row one continues: three zeros, then the initial and next frames.
    torch.testing.assert_close(second[1, :-2], torch.zeros(3, 28))
    torch.testing.assert_close(second[1, -2:, 0], torch.tensor([2.0, 20.0]))


def test_source_history_inference_uses_same_reset_and_done_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _env(num_envs=2)
    cfg = _runner_cfg_for_checkpoint()
    cfg["history_reset_mode"] = "source_zero_current"
    runner = CustomOnPolicyRunner(env, cfg)
    histories: list[torch.Tensor] = []

    def capture(history: torch.Tensor) -> torch.Tensor:
        histories.append(history.detach().clone())
        return torch.zeros(2, 6)

    monkeypatch.setattr(runner.policy, "act_inference", capture)
    infer = runner.get_inference_policy()
    first = torch.zeros(2, 28)
    first[:, 0] = torch.tensor([1.0, 2.0])
    second = torch.zeros(2, 28)
    second[:, 0] = torch.tensor([10.0, 20.0])
    infer(first)
    infer(second, torch.tensor([True, False]))

    first_history = histories[0].reshape(2, 5, 28)
    torch.testing.assert_close(first_history[:, :-1], torch.zeros(2, 4, 28))
    second_history = histories[1].reshape(2, 5, 28)
    torch.testing.assert_close(second_history[0, :-1], torch.zeros(4, 28))
    assert second_history[0, -1, 0].item() == 10.0
    torch.testing.assert_close(second_history[1, -2:, 0], torch.tensor([2.0, 20.0]))


def test_source_barlow_runner_exports_source_dual_input_actor(tmp_path) -> None:
    """Runner export is wired to the real source teacher backbone, not a dead class."""

    pytest.importorskip("onnxruntime")
    env = _SourceBarlowEnv(num_envs=2)
    runner = CustomOnPolicyRunner(env, _source_runner_cfg(), device="cpu")
    jit_path, onnx_path = runner.export_source_barlow_actor(str(tmp_path))
    contract = inspect_wheelbipe_source_barlow_onnx(
        onnx_path,
        expected_history_length=10,
        expected_algorithm="np3o",
    )
    assert contract.input_names == ("obs", "obs_hist")
    assert contract.history_shape == (1, 10, 28)
    assert Path(jit_path).is_file()

    obs = torch.zeros(1, 28)
    hist = torch.zeros(1, 10, 28)
    # Both source deployment loaders must expose the same six-action ABI.
    assert WheelbipeSourceBarlowOnnxPolicy(onnx_path).predict(obs.numpy(), hist.numpy()).shape == (
        1,
        6,
    )
    assert WheelbipeSourceBarlowTorchScriptPolicy(jit_path).predict(
        obs.numpy(), hist.numpy()
    ).shape == (1, 6)


def test_source_state_loader_strips_uniform_ddp_prefix() -> None:
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
        priv_encoder_dims=[],
        num_costs=5,
    )
    prefixed = {f"module.policy.{key}": value.clone() for key, value in policy.state_dict().items()}
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
        priv_encoder_dims=[],
        num_costs=5,
    )
    load_source_state_dict(target, {"model_state_dict": prefixed})


def test_custom_runner_random_episode_length_hook_is_noop_without_wrapper_fields() -> None:
    runner = object.__new__(CustomOnPolicyRunner)
    runner.env = SimpleNamespace()
    runner._initialize_random_episode_lengths()


def test_custom_runner_save_accepts_bare_filename(tmp_path, monkeypatch) -> None:
    env = _env()
    runner = CustomOnPolicyRunner(env, _runner_cfg_for_checkpoint())
    monkeypatch.chdir(tmp_path)
    runner.save("model.pt")
    assert (tmp_path / "model.pt").is_file()


def test_custom_runner_export_writes_contract_sidecar(tmp_path, monkeypatch) -> None:
    env = _env()
    runner = CustomOnPolicyRunner(env, _runner_cfg_for_checkpoint())

    def fake_export(_module, _inputs, output, **_kwargs):
        # Avoid requiring an ONNX exporter implementation in this boundary
        # test; the runner's sidecar write is independent of graph tracing.
        output_path = output if isinstance(output, Path) else Path(output)
        output_path.write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    output = runner.export_policy_to_onnx(str(tmp_path))
    metadata_path = custom_wheelbipe_metadata_path(output)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["schema"] == "unilab.wheelbipe.custom_policy.v1"
    assert payload["algorithm"] == "him"
    assert payload["one_step_dim"] == 28
    assert payload["history_length"] == 5
    assert payload["history_reset_mode"] == "repeat"
    assert payload["input_dim"] == 140
    assert payload["num_costs"] == 0


def test_source_barlow_export_uses_on_constraint_width_and_architecture(
    tmp_path, monkeypatch
) -> None:
    """Source export must trace the 312D stream, not compact history width."""

    env = _SourceBarlowEnv()
    runner = CustomOnPolicyRunner(env, _source_runner_cfg())

    def fake_export(_module, _inputs, output, **_kwargs):
        output_path = output if isinstance(output, Path) else Path(output)
        output_path.write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    output = runner.export_policy_to_onnx(str(tmp_path))
    payload = json.loads(custom_wheelbipe_metadata_path(output).read_text(encoding="utf-8"))

    assert payload["architecture"] == "source_barlow"
    assert payload["artifact"] == "source_barlow_full"
    assert payload["input_dim"] == 312
    assert payload["one_step_dim"] == 28
    assert payload["output_dim"] == 6
