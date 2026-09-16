"""CLI boundary tests for the Wheelbipe sim2sim entrypoints."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive branch
        raise RuntimeError(f"could not load script {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_custom_cli_defaults_to_custom_owner_timing_profile() -> None:
    module = _load_script("sim2sim_wheelbipe_custom_cli_test", "sim2sim_wheelbipe_custom.py")
    args = module._parser().parse_args(["--algorithm", "him_ppo", "--model", "policy.onnx"])
    assert args.delay_profile == "source_v14_physics"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("ppo_him", "him"), ("dream_waq", "dreamwaq"), ("ppo_dreamwaq", "dreamwaq")],
)
def test_custom_cli_accepts_source_algorithm_aliases(alias: str, canonical: str) -> None:
    """Legacy source runner names resolve to the same compact owner contract."""

    module = _load_script(f"sim2sim_wheelbipe_alias_{alias}", "sim2sim_wheelbipe_custom.py")
    args = module._parser().parse_args(["--algorithm", alias, "--model", "policy.onnx"])
    assert args.algorithm == alias
    assert module._ALGORITHM_CANONICAL[args.algorithm] == canonical


def test_custom_cli_allows_explicit_local_profile() -> None:
    module = _load_script("sim2sim_wheelbipe_custom_local_cli_test", "sim2sim_wheelbipe_custom.py")
    args = module._parser().parse_args(
        [
            "--algorithm",
            "him_ppo",
            "--model",
            "policy.onnx",
            "--delay-profile",
            "local_physics",
        ]
    )
    assert args.delay_profile == "local_physics"


def test_custom_cli_source_actor_resolves_same_directory_sibling(tmp_path: Path) -> None:
    module = _load_script("sim2sim_wheelbipe_source_cli_test", "sim2sim_wheelbipe_custom.py")
    source_actor = tmp_path / "barlow_twins_actor.onnx"
    source_actor.write_bytes(b"onnx")
    args = module._parser().parse_args(
        [
            "--algorithm",
            "np3o",
            "--source-barlow-actor",
            "--model",
            str(tmp_path / "policy.onnx"),
        ]
    )
    assert module._resolve_model_path(args.model, source_barlow_actor=True) == source_actor


def test_custom_cli_source_actor_resolves_torchscript_sibling(tmp_path: Path) -> None:
    module = _load_script(
        "sim2sim_wheelbipe_source_cli_torchscript_sibling_test",
        "sim2sim_wheelbipe_custom.py",
    )
    source_actor = tmp_path / "barlow_twins_actor.pt"
    source_actor.write_bytes(b"torchscript")

    # A source run that only contains the upstream JIT actor is usable without
    # an extra flag; ONNX remains the preferred default when both artifacts
    # exist, while the format flag makes the preference explicit.
    assert module._resolve_model_path(tmp_path, source_barlow_actor=True) == source_actor
    assert (
        module._resolve_model_path(
            tmp_path,
            source_barlow_actor=True,
            source_barlow_format="torchscript",
        )
        == source_actor
    )


def test_custom_cli_source_actor_format_flag_can_select_pt_when_both_exist(tmp_path: Path) -> None:
    module = _load_script(
        "sim2sim_wheelbipe_source_cli_format_test",
        "sim2sim_wheelbipe_custom.py",
    )
    onnx_actor = tmp_path / "barlow_twins_actor.onnx"
    jit_actor = tmp_path / "barlow_twins_actor.pt"
    onnx_actor.write_bytes(b"onnx")
    jit_actor.write_bytes(b"torchscript")
    args = module._parser().parse_args(
        [
            "--algorithm",
            "np3o",
            "--source-barlow-actor",
            "--source-barlow-format",
            "torchscript",
            "--model",
            str(tmp_path),
        ]
    )
    assert (
        module._resolve_model_path(
            args.model,
            source_barlow_actor=True,
            source_barlow_format=args.source_barlow_format,
        )
        == jit_actor
    )


def test_custom_cli_source_actor_can_use_run_directory_or_default() -> None:
    module = _load_script(
        "sim2sim_wheelbipe_source_cli_default_test", "sim2sim_wheelbipe_custom.py"
    )
    args = module._parser().parse_args(["--algorithm", "np3o", "--source-barlow-actor"])
    assert module._resolve_model_path(args.model, source_barlow_actor=True).name == (
        "barlow_twins_actor.onnx"
    )


def test_custom_cli_rejects_compact_pt_before_registry_or_onnx_loading(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script(
        "sim2sim_wheelbipe_compact_pt_reject_test",
        "sim2sim_wheelbipe_custom.py",
    )
    compact_checkpoint = tmp_path / "policy.pt"
    torch.save({"model_state_dict": {}}, compact_checkpoint)
    monkeypatch.setattr(module, "ensure_registries", lambda: pytest.fail("registry not needed"))
    monkeypatch.setattr(module, "create_env", pytest.fail)

    with pytest.raises(SystemExit, match="accepts an ONNX graph only"):
        module.main(
            [
                "--algorithm",
                "him_ppo",
                "--model",
                str(compact_checkpoint),
                "--steps",
                "1",
            ]
        )


def test_custom_cli_routes_source_barlow_torchscript_to_dual_input_adapter(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_script(
        "sim2sim_wheelbipe_source_cli_torchscript_route_test",
        "sim2sim_wheelbipe_custom.py",
    )

    class _SourceActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(28 + 10 * 28, 6)

        def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
            return self.linear(torch.cat((obs, obs_hist.reshape(obs.shape[0], -1)), dim=-1))

    source_actor = _SourceActor().eval()
    model_path = tmp_path / "barlow_twins_actor.pt"
    traced = cast(
        torch.jit.ScriptModule,
        torch.jit.trace(
            source_actor,
            (torch.zeros(1, 28), torch.zeros(1, 10, 28)),
        ),
    )
    traced.save(str(model_path))

    captured: dict[str, object] = {}

    class _FakeEnv:
        timing_contract = {"profile": "source_v14_physics", "sim_dt": 0.005, "ctrl_dt": 0.02}

        def close(self) -> None:
            captured["closed"] = True

    def fake_create_env(*args, **kwargs):
        captured["create_args"] = args
        captured["create_kwargs"] = kwargs
        return _FakeEnv()

    def fake_rollout(env, policy, *, steps, command=None, height=None):
        del env, command, height
        captured["policy"] = policy
        captured["steps"] = steps
        return {"steps": float(steps), "mean_reward": 0.0, "done_count": 0.0}

    monkeypatch.setattr(module, "ensure_registries", lambda: None)
    monkeypatch.setattr(module, "create_env", fake_create_env)
    monkeypatch.setattr(module, "run_wheelbipe_source_barlow_policy", fake_rollout)

    assert (
        module.main(
            [
                "--algorithm",
                "np3o",
                "--source-barlow-actor",
                "--model",
                str(model_path),
                "--steps",
                "2",
            ]
        )
        == 0
    )
    from unilab.training.wheelbipe import WheelbipeSourceBarlowTorchScriptPolicy

    policy = captured["policy"]
    assert isinstance(policy, WheelbipeSourceBarlowTorchScriptPolicy)
    assert policy.predict(
        np.zeros((1, 28), dtype=np.float32),
        np.zeros((1, 10, 28), dtype=np.float32),
    ).shape == (1, 6)
    assert captured["steps"] == 2
    assert captured["closed"] is True
    create_kwargs = captured["create_kwargs"]
    assert isinstance(create_kwargs, dict)
    env_cfg_override = create_kwargs["env_cfg_override"]
    assert env_cfg_override["num_costs"] == 5
    assert "Wheelbipe custom sim2sim complete" in capsys.readouterr().out


class _CustomArtifactGraph(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 6, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.linear(observation)


class _DualArtifactGraph(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(28 + 10 * 28, 6, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((obs, obs_hist.reshape(obs.shape[0], -1)), dim=-1))


def _export_custom_artifact(path: Path, *, algorithm: str, artifact: str) -> None:
    if artifact == "source_barlow_actor":
        graph = _DualArtifactGraph().eval()
        args = (torch.zeros(1, 28), torch.zeros(1, 10, 28))
        input_names = ["obs", "obs_hist"]
    else:
        input_dim = 312 if artifact == "source_barlow_full" else 140
        graph = _CustomArtifactGraph(input_dim).eval()
        args = (torch.zeros(1, input_dim),)
        input_names = ["obs_history"]
    with torch.inference_mode():
        torch.onnx.export(
            graph,
            args,
            str(path),
            input_names=input_names,
            output_names=["actions"],
            opset_version=18,
        )

    history_length = 10 if algorithm == "np3o" else 5
    metadata: dict[str, object] = {
        "schema": "unilab.wheelbipe.custom_policy.v1",
        "algorithm": algorithm,
        "architecture": "source_barlow" if algorithm == "np3o" else "compact",
        "artifact": artifact,
        "variant_name": ("flat-np3o-barlow-v0" if algorithm == "np3o" else f"flat-{algorithm}-v0"),
        "one_step_dim": 28,
        "history_length": history_length,
        "history_reset_mode": "source_zero_current",
        "output_dim": 6,
        "num_actions": 6,
        "num_estimate": 4,
        "num_costs": 5 if algorithm == "np3o" else 0,
        "output_name": "actions",
    }
    if artifact == "source_barlow_actor":
        metadata.update(
            {
                "history_feature_dim": 280,
                "source_stream_dim": 312,
                "input_names": ["obs", "obs_hist"],
                "input_shapes": [[1, 28], [1, 10, 28]],
                "output_shape": [1, 6],
                "input_name": "obs",
                "history_input_name": "obs_hist",
            }
        )
    else:
        metadata.update(
            {
                "input_dim": 312 if artifact == "source_barlow_full" else 140,
                "input_name": "obs_history",
            }
        )
    path.with_name(f"{path.name}.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize("sim", ["mujoco", "motrix"])
@pytest.mark.parametrize(
    ("algorithm", "artifact", "expected_task"),
    [
        ("him", "compact_history_policy", "WheelbipeV14FlatHIM"),
        ("dreamwaq", "compact_history_policy", "WheelbipeV14FlatDreamWaQ"),
        ("np3o", "source_barlow_full", "WheelbipeV14FlatNP3OBarlow"),
        ("np3o", "source_barlow_actor", "WheelbipeV14FlatNP3OBarlow"),
    ],
)
def test_custom_cli_runs_export_artifact_with_composed_owner_on_both_backends(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    sim: str,
    algorithm: str,
    artifact: str,
    expected_task: str,
) -> None:
    """Exercise the real ONNX loader, owner validation and one simulator step."""

    pytest.importorskip("onnxruntime")
    pytest.importorskip("mujoco" if sim == "mujoco" else "motrixsim")
    module = _load_script(
        f"sim2sim_wheelbipe_custom_{algorithm}_{artifact}_{sim}",
        "sim2sim_wheelbipe_custom.py",
    )
    model_path = tmp_path / f"{artifact}.onnx"
    _export_custom_artifact(model_path, algorithm=algorithm, artifact=artifact)

    assert (
        module.main(
            [
                "--algorithm",
                algorithm,
                "--model",
                str(model_path),
                "--sim",
                sim,
                "--steps",
                "1",
                "--num-envs",
                "1",
                "--seed",
                "7",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert f"artifact={artifact}" in output
    assert f"task={expected_task}" in output


def test_custom_cli_rejects_source_actor_for_non_np3o(monkeypatch) -> None:
    module = _load_script("sim2sim_wheelbipe_source_cli_reject_test", "sim2sim_wheelbipe_custom.py")
    monkeypatch.setattr(module, "ensure_registries", lambda: None)
    with pytest.raises(SystemExit, match="only valid with --algorithm np3o"):
        module.main(
            [
                "--algorithm",
                "him",
                "--source-barlow-actor",
                "--model",
                "policy.onnx",
                "--steps",
                "1",
            ]
        )


def test_normal_cli_rejects_non_positive_steps_before_model_loading(monkeypatch) -> None:
    module = _load_script("sim2sim_wheelbipe_cli_test", "sim2sim_wheelbipe.py")
    monkeypatch.setattr(sys, "argv", ["sim2sim_wheelbipe.py", "--steps", "0"])
    with pytest.raises(SystemExit, match="--steps must be positive"):
        module.main()


def test_ros2_cli_exposes_cold_path_deployment_config() -> None:
    module = _load_script("sim2sim_wheelbipe_ros2_cli_test", "sim2sim_wheelbipe_ros2.py")
    args = module._parser().parse_args(["--config", "custom.yaml", "--steps", "1"])
    assert args.config == Path("custom.yaml")
    assert args.steps == 1


def test_ros2_cli_print_contract_does_not_require_model_or_simulator(capsys) -> None:
    module = _load_script("sim2sim_wheelbipe_ros2_contract_cli_test", "sim2sim_wheelbipe_ros2.py")
    assert module.main(["--print-contract"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["native_ros2"] is False
    assert payload["ros_graph"] is False
    assert payload["serial_io"] is False
    assert payload["realtime_guarantee"] is False
    assert payload["api"]["policy_input_shape"] == [1, 35]
    assert payload["protocol"]["state_packet_size"] == 143
