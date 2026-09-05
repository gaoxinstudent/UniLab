"""Unit tests for the cross-backend sim2sim contract resolver.

Pure and fast: no environment, registry, torch, or backend creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.training.sim2sim import (
    ALLOWLIST,
    DENYLIST,
    ENV_STRUCTURAL_DENYLIST,
    WARNING_LIST,
    CrossBackendIncompatibleError,
    Sim2SimConfigResolver,
    extract_contract_snapshot,
    hydrate_custom_checkpoint_config,
    policy_load_dim_guard,
    resolve_sim2sim_config,
)


def _write_sidecar(run_dir: Path, snapshot: dict[str, Any] | None) -> Path:
    payload: dict[str, Any] = {"run": {}, "config": {}}
    if snapshot is not None:
        payload["contract_snapshot"] = snapshot
    (run_dir / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def _mujoco_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": {"sim_backend": "mujoco"},
            "algo": {
                "obs_groups": {"actor": ["actor"]},
                "empirical_normalization": False,
                "policy": {
                    "actor_hidden_dims": [512, 256, 128],
                    "critic_hidden_dims": [512, 256, 128],
                },
            },
            "env": {
                "control_config": {"action_scale": 0.25, "simulate_action_latency": False},
                "ctrl_dt": 0.01,
            },
            "reward": {"scales": {"tracking_lin_vel": 2.0}, "max_tilt_deg": 25.0},
        }
    )


def test_field_lists_are_disjoint():
    deny, warn, allow = set(DENYLIST), set(WARNING_LIST), set(ALLOWLIST)
    assert deny.isdisjoint(warn)
    assert deny.isdisjoint(allow)
    assert warn.isdisjoint(allow)


def test_extract_snapshot_includes_only_present_contract_fields():
    snapshot = extract_contract_snapshot(_mujoco_cfg())
    # Present DENY/WARN fields are captured...
    assert snapshot["env.control_config.action_scale"] == 0.25
    assert snapshot["algo.obs_groups"] == {"actor": ["actor"]}
    assert snapshot["algo.empirical_normalization"] is False
    assert snapshot["reward.scales"] == {"tracking_lin_vel": 2.0}
    # ...ALLOWLIST fields are excluded...
    assert "training.sim_backend" not in snapshot
    # ...and absent fields are omitted (never stored as None).
    assert "algo.obs_normalization" not in snapshot
    assert "env.sampling_mode" not in snapshot
    assert "reward.base_height_target" not in snapshot


def test_snapshot_json_round_trips():
    snapshot = extract_contract_snapshot(_mujoco_cfg())
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_matching_contract_returns_same_cfg(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_denylist_mismatch_raises_with_field_in_message(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "action_scale" in msg
    assert "0.25" in msg
    assert "0.5" in msg


def test_denylist_nested_dict_mismatch_raises(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.algo.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, target)


def test_warning_mismatch_does_not_raise(tmp_path, capsys):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.ctrl_dt = 0.02
    assert resolve_sim2sim_config(tmp_path, target) is target
    assert "[sim2sim] WARNING override env.ctrl_dt" in capsys.readouterr().out


def test_missing_snapshot_falls_back(tmp_path, capsys):
    _write_sidecar(tmp_path, None)  # run_config.json without contract_snapshot
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target
    assert "no contract_snapshot" in capsys.readouterr().out


def test_missing_file_falls_back(tmp_path, capsys):
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target  # no run_config.json
    assert "no contract_snapshot" in capsys.readouterr().out


def test_corrupt_sidecar_falls_back(tmp_path, capsys):
    (tmp_path / "run_config.json").write_text("{ not valid json", encoding="utf-8")
    target = _mujoco_cfg()
    assert resolve_sim2sim_config(tmp_path, target) is target
    assert "no contract_snapshot" in capsys.readouterr().out


def test_none_source_returns_none(capsys):
    assert resolve_sim2sim_config(None, _mujoco_cfg()) is None
    assert "no source run dir" in capsys.readouterr().out


def test_target_missing_path_is_skipped(tmp_path):
    # Snapshot was taken from a PPO run (empirical_normalization); the off-policy
    # target only has obs_normalization, so the snapshot path is simply skipped.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": True})
    target = OmegaConf.create({"algo": {"obs_normalization": True}, "env": {}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_non_strict_downgrades_denial_to_warning(tmp_path, capsys):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    assert resolve_sim2sim_config(tmp_path, target, strict=False) is target
    assert "action_scale" in capsys.readouterr().out


def test_empirical_normalization_on_vs_off_raises(tmp_path):
    # The real case: trained with obs normalization ON, played with it OFF.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": True})
    target = OmegaConf.create({"algo": {"empirical_normalization": False}})
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, target)


def test_int_and_float_compare_equal(tmp_path):
    _write_sidecar(tmp_path, {"env.ctrl_dt": 0})
    target = OmegaConf.create({"env": {"ctrl_dt": 0.0}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_action_scale_list_form(tmp_path):
    _write_sidecar(tmp_path, {"env.control_config.action_scale": [0.5, 0.5]})
    ok = OmegaConf.create({"env": {"control_config": {"action_scale": [0.5, 0.5]}}})
    assert resolve_sim2sim_config(tmp_path, ok) is ok
    bad = OmegaConf.create({"env": {"control_config": {"action_scale": 0.25}}})
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, bad)


# --- custom checkpoint architecture hydration -------------------------------------


def _custom_target_cfg(algorithm: str) -> Any:
    """Build compact custom targets used to exercise sidecar adoption."""

    history = 10 if algorithm == "np3o" else 5
    costs = 5 if algorithm == "np3o" else 0
    policy: dict[str, Any] = {
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "activation": "elu",
        "init_noise_std": 1.0,
    }
    if algorithm == "dreamwaq":
        policy.update(
            {
                "cenet_encoder_hidden_dims": [128, 64],
                "cenet_decoder_hidden_dims": [64, 128],
                "cenet_in_dim": 140,
                "cenet_out_dim": 20,
                "cenet_logvar_clip": 10.0,
                "cenet_feature_clip": 50.0,
            }
        )
    if algorithm == "np3o":
        policy["architecture"] = "compact"
    return OmegaConf.create(
        {
            "algo": {
                "algorithm_name": algorithm,
                "num_actor_history": history,
                "num_estimate": 4,
                "num_costs": costs,
                "latent_dim": 16,
                "policy": policy,
                "estimator": {
                    "enc_hidden_dims": [128, 64, 16],
                    "tar_hidden_dims": [128, 64],
                    "velocity_target_start": 28,
                    "target_obs_start": 4,
                },
            },
            "training": {"sim_backend": "mujoco"},
        }
    )


def _write_custom_sidecar(tmp_path: Path, algorithm: str) -> Path:
    """Write representative source-sized metadata for one custom owner."""

    if algorithm == "him":
        source_algo: dict[str, Any] = {
            "algorithm_name": "him_ppo",  # source alias must canonicalize
            "num_actor_history": 5,
            "num_estimate": 4,
            "num_costs": 0,
            "latent_dim": 16,
            "policy": {
                "actor_hidden_dims": [512, 256, 128],
                "critic_hidden_dims": [512, 256, 128],
                "activation": "relu",
                "init_noise_std": 0.75,
            },
            "estimator": {
                "enc_hidden_dims": [256, 128, 32],
                "tar_hidden_dims": [256, 128],
                "velocity_target_start": 26,
                "target_obs_start": 6,
            },
        }
    elif algorithm == "dreamwaq":
        source_algo = {
            "algorithm_name": "dreamwaq",
            "num_actor_history": 5,
            "num_estimate": 4,
            "num_costs": 0,
            "latent_dim": 16,
            "policy": {
                "actor_hidden_dims": [512, 256, 128],
                "critic_hidden_dims": [512, 256, 128],
                "activation": "relu",
                "init_noise_std": 0.8,
                "cenet_encoder_hidden_dims": [256, 128, 64],
                "cenet_decoder_hidden_dims": [64, 128, 256],
                "cenet_in_dim": 140,
                "cenet_out_dim": 20,
                "cenet_logvar_clip": 7.0,
                "cenet_feature_clip": 40.0,
                "adaboot_mode": "reward_cv",
                "adaboot_min": 0.1,
                "adaboot_max": 0.9,
                "adaboot_temperature": 0.7,
                "adaboot_bias": 0.05,
            },
            "estimator": {},
        }
    else:
        source_algo = {
            "algorithm_name": "np3o",
            "num_actor_history": 10,
            "num_estimate": 4,
            "num_costs": 5,
            "latent_dim": 16,
            "policy_architecture": "source_barlow",
            "policy": {
                "actor_hidden_dims": [512, 256, 128],
                "critic_hidden_dims": [512, 256, 128],
                "architecture": "source_barlow",
                "source_barlow": {
                    "class_name": "ActorCriticBarlowTwins",
                    "num_prop": 28,
                    "num_scan": 0,
                    "num_state_est": 4,
                    "num_priv_latent": 4,
                    "num_hist": 10,
                    "scan_encoder_dims": [128, 64, 32],
                    "priv_encoder_dims": [],
                    "hist_encoder": False,
                    "fixed_std": False,
                    "teacher_act": True,
                    "imi_flag": True,
                    "continue_from_last_std": True,
                    "tanh_encoder_output": False,
                    "num_costs": 5,
                },
            },
            "estimator": {},
        }
    source_algo["history_reset_mode"] = "source_zero_current"
    (tmp_path / "run_config.json").write_text(
        json.dumps({"config": {"algo": source_algo}}), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.parametrize("algorithm", ["him", "dreamwaq", "np3o"])
def test_custom_checkpoint_hydrates_all_source_architecture_families(
    tmp_path: Path, algorithm: str, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _custom_target_cfg(algorithm)
    original = OmegaConf.to_container(target, resolve=True)
    hydrated = hydrate_custom_checkpoint_config(
        _write_custom_sidecar(tmp_path, algorithm),
        target,
    )

    # Hydration returns an effective copy; the composed compact target remains
    # reusable and source metadata is adopted before runner construction.
    assert hydrated is not target
    assert OmegaConf.to_container(target, resolve=True) == original
    assert hydrated.algo.policy.actor_hidden_dims == [512, 256, 128]
    assert hydrated.algo.policy.critic_hidden_dims == [512, 256, 128]
    assert hydrated.algo.history_reset_mode == "source_zero_current"
    assert "adopted custom policy architecture" in capsys.readouterr().out

    if algorithm == "him":
        assert hydrated.algo.estimator.enc_hidden_dims == [256, 128, 32]
        assert hydrated.algo.estimator.tar_hidden_dims == [256, 128]
        assert hydrated.algo.estimator.target_obs_start == 6
    elif algorithm == "dreamwaq":
        assert hydrated.algo.policy.cenet_encoder_hidden_dims == [256, 128, 64]
        assert hydrated.algo.policy.cenet_decoder_hidden_dims == [64, 128, 256]
        assert hydrated.algo.policy.cenet_logvar_clip == 7.0
        assert hydrated.algo.policy.adaboot_mode == "reward_cv"
    else:
        assert hydrated.algo.policy_architecture == "source_barlow"
        assert hydrated.algo.policy.architecture == "source_barlow"
        assert hydrated.algo.policy.source_barlow.num_hist == 10
        assert hydrated.algo.policy.source_barlow.scan_encoder_dims == [128, 64, 32]


def test_custom_checkpoint_hydration_preserves_explicit_architecture_override(tmp_path: Path):
    target = _custom_target_cfg("him")
    hydrated = hydrate_custom_checkpoint_config(
        _write_custom_sidecar(tmp_path, "him"),
        target,
        explicit_paths={"algo.policy.actor_hidden_dims"},
    )
    assert hydrated.algo.policy.actor_hidden_dims == [256, 128, 64]
    # Other non-explicit constructor fields still follow the checkpoint.
    assert hydrated.algo.policy.critic_hidden_dims == [512, 256, 128]


def test_custom_checkpoint_hydration_rejects_observation_contract_mismatch(tmp_path: Path):
    target = _custom_target_cfg("np3o")
    source_dir = _write_custom_sidecar(tmp_path, "np3o")
    payload = json.loads((source_dir / "run_config.json").read_text(encoding="utf-8"))
    payload["config"]["algo"]["num_actor_history"] = 5
    (source_dir / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CrossBackendIncompatibleError, match="num_actor_history"):
        hydrate_custom_checkpoint_config(source_dir, target)


def test_custom_checkpoint_hydration_rejects_history_reset_contract_mismatch(
    tmp_path: Path,
) -> None:
    target = _custom_target_cfg("him")
    target.algo.history_reset_mode = "repeat"
    with pytest.raises(CrossBackendIncompatibleError, match="history reset contract"):
        hydrate_custom_checkpoint_config(_write_custom_sidecar(tmp_path, "him"), target)


def test_custom_checkpoint_hydration_can_be_disabled(tmp_path: Path):
    target = _custom_target_cfg("him")
    original = OmegaConf.to_container(target, resolve=True)
    result = hydrate_custom_checkpoint_config(
        _write_custom_sidecar(tmp_path, "him"), target, enabled=False
    )
    assert result is target
    assert OmegaConf.to_container(target, resolve=True) == original


# --- env-structural asymmetric-presence fail-closed --------------------------------


def test_env_structural_denylist_is_the_env_subset():
    assert ENV_STRUCTURAL_DENYLIST == ["env.control_config.action_scale", "env.sampling_mode"]
    assert set(ENV_STRUCTURAL_DENYLIST) <= set(DENYLIST)


def test_env_field_present_in_source_absent_in_target_raises(tmp_path):
    # Forward asymmetry: source (mujoco) sets action_scale, target (motrix) omits it
    # and would fall back to a differing env default. Fail closed instead of skipping.
    _write_sidecar(tmp_path, {"env.control_config.action_scale": [0.5, 0.5, 0.5]})
    target = OmegaConf.create({"env": {"control_config": {}}})  # no action_scale set
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "action_scale" in msg
    assert "target=<absent>" in msg


def test_env_field_present_in_target_absent_in_source_raises(tmp_path):
    # Reverse asymmetry: the trained run omitted sampling_mode (used the env default),
    # the target sets it explicitly. Still unverifiable -> fail closed.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": False})
    target = OmegaConf.create(
        {"algo": {"empirical_normalization": False}, "env": {"sampling_mode": "adaptive"}}
    )
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        resolve_sim2sim_config(tmp_path, target)
    msg = str(excinfo.value)
    assert "sampling_mode" in msg
    assert "source=<absent>" in msg


def test_env_field_symmetric_absence_does_not_raise(tmp_path):
    # Neither side sets the env structural field -> both use the same env default -> ok.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": False})
    target = OmegaConf.create({"algo": {"empirical_normalization": False}, "env": {}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_algo_field_absent_in_target_still_skipped_not_fail_closed(tmp_path):
    # Regression: algo-specific fields keep the cross-algo skip and must NOT fail closed
    # the way env structural fields now do.
    _write_sidecar(tmp_path, {"algo.empirical_normalization": True})
    target = OmegaConf.create({"algo": {"obs_normalization": True}, "env": {}})
    assert resolve_sim2sim_config(tmp_path, target) is target


def test_env_field_fail_closed_even_if_default_might_match(tmp_path):
    # Fail-closed semantics: the guard cannot resolve the omitted side's runtime
    # default, so it raises even when that default could happen to equal the explicit
    # value. The fix is a one-line explicit declaration in the target YAML.
    _write_sidecar(tmp_path, {"env.control_config.action_scale": 0.25})
    target = OmegaConf.create({"env": {"control_config": {}}})
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, target)


# --- play-time runtime dimension guard ---------------------------------------------


def test_dim_guard_passes_through_on_success():
    # No error inside the block -> the guard is a no-op.
    ran = False
    with policy_load_dim_guard(env_obs_dim=10, env_action_dim=3, algo_name="ppo"):
        ran = True
    assert ran


def test_dim_guard_translates_torch_size_mismatch():
    # torch raises RuntimeError("size mismatch ...") on a shape-incompatible load.
    err = "Error(s) in loading state_dict for ActorCritic:\n\tsize mismatch for actor.0.weight"
    with pytest.raises(CrossBackendIncompatibleError) as excinfo:
        with policy_load_dim_guard(env_obs_dim=42, env_action_dim=12, algo_name="ppo"):
            raise RuntimeError(err)
    msg = str(excinfo.value)
    assert "42" in msg and "12" in msg  # env dims surfaced
    assert "audit_sim2sim_contracts" in msg  # actionable pointer
    assert isinstance(excinfo.value.__cause__, RuntimeError)  # original chained


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA out of memory"),
        RuntimeError("Expected all tensors to be on the same device, but found cuda:0 and cpu"),
        RuntimeError("expected scalar type Float but found Half"),
        RuntimeError("shape '[12, 4]' is invalid for input of size 36"),
        RuntimeError("no kernel image is available for execution on the device"),
        RuntimeError("copying a param failed: expected all tensors to use the same dtype"),
        ValueError("Expected input batch_size (32) to match target batch_size (16)"),
    ],
)
def test_dim_guard_reraises_unrelated_errors_unchanged(error):
    with pytest.raises(type(error)) as excinfo:
        with policy_load_dim_guard(env_obs_dim=10, env_action_dim=3):
            raise error
    assert excinfo.value is error


def test_dim_guard_does_not_swallow_keyerror():
    # A missing checkpoint key is not a dim mismatch; it must surface unchanged.
    with pytest.raises(KeyError):
        with policy_load_dim_guard(env_obs_dim=10, env_action_dim=3):
            raise KeyError("actor")


# --- Sim2SimConfigResolver class facade + user-level bypass ------------------------


def test_resolver_class_exposes_field_lists():
    assert Sim2SimConfigResolver.DENYLIST is DENYLIST
    assert Sim2SimConfigResolver.WARNING_LIST is WARNING_LIST
    assert Sim2SimConfigResolver.ALLOWLIST is ALLOWLIST
    assert Sim2SimConfigResolver.ENV_STRUCTURAL_DENYLIST is ENV_STRUCTURAL_DENYLIST


def test_resolver_class_resolve_delegates_and_raises(tmp_path):
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    with pytest.raises(CrossBackendIncompatibleError):
        Sim2SimConfigResolver.resolve(tmp_path, target)


def test_resolver_class_strict_false_is_user_bypass(tmp_path, capsys):
    # training.sim2sim_strict=false maps to strict=False: a DENYLIST denial becomes a
    # warning and play proceeds with the target cfg (the load-time dim guard still bites).
    _write_sidecar(tmp_path, extract_contract_snapshot(_mujoco_cfg()))
    target = _mujoco_cfg()
    target.env.control_config.action_scale = 0.5
    assert Sim2SimConfigResolver.resolve(tmp_path, target, strict=False) is target
    assert "action_scale" in capsys.readouterr().out


def test_resolver_class_extract_and_dim_guard_delegate():
    snap = Sim2SimConfigResolver.extract_snapshot(_mujoco_cfg())
    assert snap["env.control_config.action_scale"] == 0.25
    with pytest.raises(CrossBackendIncompatibleError):
        with Sim2SimConfigResolver.load_dim_guard(env_obs_dim=5, env_action_dim=2):
            raise RuntimeError("size mismatch for actor.0.weight")


def _compose_task(task: str) -> Any:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf" / "ppo")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        return compose("config", overrides=[f"task={task}"])


def test_g1_walk_flat_mujoco_inherits_base_contract():
    # The MuJoCo owner carries the full contract in its standalone owner config.
    mujoco = _compose_task("g1_walk_flat/mujoco")
    assert OmegaConf.select(mujoco, "env.control_config.action_scale") == 0.25
    assert OmegaConf.select(mujoco, "algo.empirical_normalization") is False
    assert OmegaConf.select(mujoco, "algo.obs_groups.actor") == ["actor"]


def test_g1_walk_flat_cross_backend_play_is_guarded(tmp_path):
    # Motrix intentionally overrides contract fields, so MuJoCo->Motrix is guarded.
    snapshot = extract_contract_snapshot(_compose_task("g1_walk_flat/mujoco"))
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": snapshot}), encoding="utf-8"
    )
    motrix = _compose_task("g1_walk_flat/motrix")
    with pytest.raises(CrossBackendIncompatibleError):
        resolve_sim2sim_config(tmp_path, motrix)
