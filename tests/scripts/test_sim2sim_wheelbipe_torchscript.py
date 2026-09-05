"""Normal sim2sim routing tests for upstream ``policy.pt`` artifacts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from unilab.training.wheelbipe import WheelbipeTorchScriptPolicy

ROOT = Path(__file__).resolve().parents[2]
_SOURCE_POLICY_CANDIDATES = sorted(
    (ROOT / "third-party" / "wheeled-legged_RL" / "pretrained").glob("**/policy.pt")
)
# Keep the test useful when the vendored source refreshes its date-stamped run
# names; the sentinel path makes the skip branch deterministic in a checkout
# that intentionally omits third-party artifacts.
SOURCE_POLICY = (
    _SOURCE_POLICY_CANDIDATES[0]
    if _SOURCE_POLICY_CANDIDATES
    else ROOT / "third-party" / "wheeled-legged_RL" / "pretrained" / "policy.pt"
)


def _load_script(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive branch
        raise RuntimeError(f"could not load script {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_normal_sim2sim_routes_actual_source_policy_pt_on_cpu(monkeypatch, capsys) -> None:
    """Use the vendored upstream archive and verify the helper's CPU boundary."""

    if not SOURCE_POLICY.is_file():
        pytest.skip("vendored upstream policy.pt is unavailable")

    module = _load_script(
        "sim2sim_wheelbipe_actual_torchscript_test",
        "sim2sim_wheelbipe.py",
    )
    captured: dict[str, object] = {}

    class FakeEnv:
        timing_contract = {"profile": "local_physics", "sim_dt": 0.001, "ctrl_dt": 0.02}

        def close(self) -> None:
            captured["closed"] = True

    def fake_make(*args, **kwargs):
        captured["make_args"] = args
        captured["make_kwargs"] = kwargs
        return FakeEnv()

    def fake_rollout(env, policy, *, steps, command=None, height=None):
        del env, command, height
        captured["policy"] = policy
        captured["steps"] = steps
        return {"steps": float(steps), "mean_reward": 0.0, "done_count": 0.0}

    monkeypatch.setattr(module, "ensure_registries", lambda: None)
    monkeypatch.setattr(module, "registry", SimpleNamespace(make=fake_make))
    monkeypatch.setattr(module, "run_wheelbipe_policy", fake_rollout)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sim2sim_wheelbipe.py", "--model", str(SOURCE_POLICY), "--steps", "2"],
    )

    assert module.main() == 0
    policy = captured["policy"]
    assert isinstance(policy, WheelbipeTorchScriptPolicy)
    assert policy.device.type == "cpu"
    assert captured["steps"] == 2
    assert captured["closed"] is True
    assert "Wheelbipe sim2sim complete" in capsys.readouterr().out
