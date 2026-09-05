from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic


def test_him_iteration_progress_is_one_equivalent_multiline_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unilab.algos.torch.him_ppo.runner import HIMOnPolicyRunner

    runner = cast(Any, HIMOnPolicyRunner.__new__(HIMOnPolicyRunner))
    runner.logger = SimpleNamespace(
        rewbuffer=deque([1.0, 3.0]),
        lenbuffer=deque([10.0, 14.0]),
    )

    with caplog.at_level(logging.INFO, logger="unilab.algos.torch.him_ppo.runner"):
        runner._log_iter(
            it=1,
            tot=3,
            value_loss=0.25,
            surrogate_loss=0.5,
            estimation_loss=0.75,
            swap_loss=1.0,
            elapsed=2.0,
            infos={"log": {"reward/feet": 1.25}},
        )

    records = [
        record for record in caplog.records if record.name == "unilab.algos.torch.him_ppo.runner"
    ]
    assert len(records) == 1
    lines = records[0].getMessage().splitlines()
    assert lines[0] == "-" * 80
    assert lines[-1] == "-" * 80
    assert f"{'Iteration':>40}: 1/3" in lines
    assert f"{'Mean episode reward':>40}: 2.0000" in lines
    assert f"{'Mean episode length':>40}: 12.0" in lines
    assert f"{'reward/feet':>40}: 1.2500" in lines
    assert f"{'Time elapsed':>40}: 00:00:02" in lines
    assert f"{'ETA':>40}: 00:00:04" in lines


def _tiny_him_export_runner() -> Any:
    """Build only the runner fields needed by the exporter boundary."""

    from unilab.algos.torch.him_ppo.runner import HIMOnPolicyRunner

    runner = cast(Any, HIMOnPolicyRunner.__new__(HIMOnPolicyRunner))
    runner.actor_critic = HIMActorCritic(
        num_actor_obs=2,
        num_critic_obs=3,
        num_one_step_obs=2,
        num_actions=1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        estimator={"enc_hidden_dims": [4, 2], "tar_hidden_dims": [4, 2]},
    )
    return runner


def test_him_runner_loads_complete_source_mlp_wrapper_graph(tmp_path) -> None:
    """The dedicated HIM runner accepts pinned ``actor.model`` source keys."""

    runner = _tiny_him_export_runner()
    runner.device = "cpu"
    source_state: dict[str, torch.Tensor] = {}
    for key, value in runner.actor_critic.state_dict().items():
        if key.startswith("actor."):
            key = f"actor.model.{key[len('actor.') :]}"
        elif key.startswith("critic."):
            key = f"critic.model.{key[len('critic.') :]}"
        source_state[key] = value.clone()
    checkpoint = tmp_path / "source-him.pt"
    torch.save({"model_state_dict": source_state}, checkpoint)

    restored = _tiny_him_export_runner()
    restored.device = "cpu"
    restored.load(str(checkpoint), load_optimizer=False, load_iteration=False)

    for key, expected in runner.actor_critic.state_dict().items():
        torch.testing.assert_close(restored.actor_critic.state_dict()[key], expected)


@pytest.mark.parametrize("export_kind", ["onnx", "jit"])
def test_him_export_restores_policy_device_when_export_fails(
    export_kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed CPU export must still execute the runner's device restore."""

    runner = _tiny_him_export_runner()
    original_to = runner.actor_critic.to
    restore_calls: list[torch.device] = []

    def tracked_to(device: str | torch.device, *args: Any, **kwargs: Any) -> Any:
        restore_calls.append(torch.device(device))
        return original_to(device, *args, **kwargs)

    monkeypatch.setattr(runner.actor_critic, "to", tracked_to)
    if export_kind == "onnx":

        def fail_export(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("synthetic ONNX exporter failure")

        monkeypatch.setattr(torch.onnx, "export", fail_export)
        export = runner.export_policy_to_onnx
    else:

        def fail_trace(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("synthetic TorchScript exporter failure")

        monkeypatch.setattr(torch.jit, "trace", fail_trace)
        export = runner.export_policy_to_jit

    with pytest.raises(RuntimeError, match="synthetic"):
        export(str(tmp_path))

    assert restore_calls == [torch.device("cpu")]
    assert {parameter.device for parameter in runner.actor_critic.parameters()} == {
        torch.device("cpu")
    }
