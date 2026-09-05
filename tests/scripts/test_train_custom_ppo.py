from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_custom_ppo_resume_test", ROOT / "scripts" / "train_custom_ppo.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive branch
        raise RuntimeError("could not load train_custom_ppo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(
    tmp_path: Path,
    load_run: str,
    *,
    task_name: str = "WheelbipeV14FlatHIM",
) -> object:
    return OmegaConf.create(
        {
            "training": {"task_name": task_name, "log_root": str(tmp_path)},
            "algo": {
                "algo_log_name": "custom_ppo",
                "load_run": load_run,
                "checkpoint": -1,
            },
        }
    )


def test_custom_resume_resolves_explicit_run_checkpoint(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "WheelbipeV14FlatHIM" / "run-1"
    run_dir.mkdir(parents=True)
    model = run_dir / "model_7.pt"
    model.write_bytes(b"checkpoint")

    assert module._resolve_custom_resume_path(_cfg(tmp_path, "run-1")) == model


def test_custom_resume_skips_default_fresh_run(tmp_path: Path) -> None:
    module = _load_module()
    assert module._resolve_custom_resume_path(_cfg(tmp_path, "-1")) is None


def test_custom_resume_missing_checkpoint_fails_closed(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(FileNotFoundError, match="resume mode"):
        module._resolve_custom_resume_path(_cfg(tmp_path, "missing-run"))


@pytest.mark.parametrize(
    ("play_task", "training_task"),
    [
        ("WheelbipeV14FlatHIMPlay", "WheelbipeV14FlatHIM"),
        ("WheelbipeV14FlatDreamWaQPlay", "WheelbipeV14FlatDreamWaQ"),
        ("WheelbipeV14FlatNP3OBarlowPlay", "WheelbipeV14FlatNP3OBarlow"),
    ],
)
def test_custom_play_latest_falls_back_to_paired_training_root(
    tmp_path: Path,
    play_task: str,
    training_task: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    run_dir = tmp_path / training_task / "run-1"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_7.pt"
    checkpoint.write_bytes(b"checkpoint")

    resolved, resolved_dir = module._resolve_custom_play_path(
        _cfg(tmp_path, "-1", task_name=play_task)
    )

    assert resolved == checkpoint
    assert resolved_dir == run_dir
    assert "Using latest non-Play checkpoint" in capsys.readouterr().out


def test_custom_play_explicit_run_does_not_use_paired_training_root(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "WheelbipeV14FlatHIM" / "explicit-run"
    run_dir.mkdir(parents=True)
    (run_dir / "model_7.pt").write_bytes(b"checkpoint")

    resolved, resolved_dir = module._resolve_custom_play_path(
        _cfg(
            tmp_path,
            "explicit-run",
            task_name="WheelbipeV14FlatHIMPlay",
        )
    )

    assert resolved is None
    assert resolved_dir is None


def test_custom_device_override_takes_precedence_over_auto_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "get_default_device", lambda: "cuda")
    cfg = OmegaConf.create({"training": {"device": "cpu"}})

    assert module._resolve_custom_device(cfg) == "cpu"


def test_custom_device_defaults_to_available_runtime_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "get_default_device", lambda: "mps")
    cfg = OmegaConf.create({"training": {"device": None}})

    assert module._resolve_custom_device(cfg) == "mps"
