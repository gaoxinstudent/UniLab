"""Boundary tests for compact history-policy sim2sim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from unilab.algos.torch.custom_ppo.source_barlow import SourceEmpiricalNormalization
from unilab.base.np_env import NpEnvState
from unilab.training.wheelbipe import (
    WheelbipeHistoryOnnxPolicy,
    WheelbipeHistoryPolicyContract,
    WheelbipeSourceBarlowFullOnnxPolicy,
    WheelbipeSourceBarlowFullPolicyContract,
    WheelbipeSourceBarlowOnnxPolicy,
    WheelbipeSourceBarlowPolicyContract,
    WheelbipeSourceBarlowTorchScriptPolicy,
    custom_wheelbipe_metadata_path,
    export_barlow_twins_actor_from_policy,
    export_mlp_barlow_twins_actor_torchscript,
    inspect_wheelbipe_history_onnx,
    inspect_wheelbipe_source_barlow_full_onnx,
    inspect_wheelbipe_source_barlow_onnx,
    run_wheelbipe_history_policy,
    run_wheelbipe_source_barlow_full_policy,
    run_wheelbipe_source_barlow_policy,
)


class _HistoryRecordingPolicy:
    def __init__(
        self,
        history_length: int = 2,
        *,
        history_reset_mode: str | None = None,
    ) -> None:
        self.contract = WheelbipeHistoryPolicyContract(
            input_name="obs_history",
            output_name="actions",
            one_step_dim=28,
            history_length=history_length,
            input_dim=28 * history_length,
        )
        self.metadata = (
            None if history_reset_mode is None else {"history_reset_mode": history_reset_mode}
        )
        self.observations: list[np.ndarray] = []

    def predict(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(np.asarray(observation).copy())
        return np.zeros((observation.shape[0], 6), dtype=np.float32)


class _SourceRecordingPolicy:
    def __init__(self, history_length: int = 10) -> None:
        self.contract = WheelbipeSourceBarlowPolicyContract(
            input_name="obs",
            history_input_name="obs_hist",
            output_name="actions",
            one_step_dim=28,
            history_length=history_length,
        )
        self.observations: list[tuple[np.ndarray, np.ndarray]] = []

    def predict(self, observation: np.ndarray, history: np.ndarray) -> np.ndarray:
        self.observations.append((np.asarray(observation).copy(), np.asarray(history).copy()))
        return np.zeros((observation.shape[0], 6), dtype=np.float32)


class _SourceFullRecordingPolicy:
    def __init__(self) -> None:
        self.contract = WheelbipeSourceBarlowFullPolicyContract(
            input_name="obs_history",
            output_name="actions",
        )
        self.observations: list[np.ndarray] = []

    def predict(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(np.asarray(observation).copy())
        return np.zeros((observation.shape[0], 6), dtype=np.float32)


def test_source_empirical_normalization_until_and_inverse() -> None:
    """Retain the source helper's learning cap and inverse API."""

    normalizer = SourceEmpiricalNormalization(2, until=2)
    first = torch.tensor([[1.0, 3.0], [3.0, 7.0]])
    normalized = normalizer(first)
    assert normalizer.count == 2
    # The source ``until`` cap freezes statistics once the accumulated batch
    # count reaches the limit; a subsequent batch must not update them.
    mean_before = normalizer.mean
    std_before = normalizer.std
    normalizer(torch.tensor([[101.0, 203.0]]))
    assert normalizer.count == 2
    torch.testing.assert_close(normalizer.mean, mean_before)
    torch.testing.assert_close(normalizer.std, std_before)
    torch.testing.assert_close(normalizer.inverse(normalized), first)

    frozen = SourceEmpiricalNormalization(2, until=0)
    frozen(first)
    assert frozen.count == 0


class _CompactResettingEnv:
    def __init__(self) -> None:
        self.calls = 0

    def _state(self) -> NpEnvState:
        obs = np.zeros((1, 28), dtype=np.float32)
        obs[0, 10] = float(self.calls)
        return NpEnvState(
            obs={"obs": obs, "critic": np.zeros((1, 78), dtype=np.float32)},
            reward=np.asarray([0.25], dtype=np.float32),
            terminated=np.zeros((1,), dtype=bool),
            truncated=np.zeros((1,), dtype=bool),
            info={
                "commands": np.asarray([[9.0, 9.0, 9.0]], dtype=np.float32),
                "height_commands": np.asarray([9.0], dtype=np.float32),
            },
        )

    def init_state(self) -> NpEnvState:
        return self._state()

    def step(self, actions: np.ndarray) -> NpEnvState:
        assert actions.shape == (1, 6)
        self.calls += 1
        state = self._state()
        # Exercise the autoreset-visible state on the second call.
        if self.calls == 2:
            state.terminated[:] = True
        return state


class _SourceFullResettingEnv(_CompactResettingEnv):
    def _state(self) -> NpEnvState:
        state = super()._state()
        latent = np.asarray(
            [[10.0 + self.calls, 20.0 + self.calls, 30.0 + self.calls, 40.0 + self.calls]],
            dtype=np.float32,
        )
        state.obs["critic"] = np.concatenate((state.obs["obs"], latent), axis=1)
        return state


def test_history_rollout_shifts_frames_and_reapplies_overrides() -> None:
    env = _CompactResettingEnv()
    policy = _HistoryRecordingPolicy(history_length=2)

    diagnostics = run_wheelbipe_history_policy(
        env,
        policy,  # type: ignore[arg-type]
        steps=3,
        command=(0.3, 0.0, -0.4),
        height=0.27,
    )

    assert diagnostics == {"steps": 3.0, "mean_reward": 0.25, "done_count": 1.0}
    assert len(policy.observations) == 3
    for history in policy.observations:
        assert history.shape == (1, 56)
        np.testing.assert_allclose(history[0, :3], [0.3, 0.0, -0.4])
        np.testing.assert_allclose(history[0, 28:31], [0.3, 0.0, -0.4])
        np.testing.assert_allclose(history[0, 3], 1.35)
        np.testing.assert_allclose(history[0, 31], 1.35)
    # The newest frame changes after each env step and is retained at the end
    # of the flattened history.
    assert policy.observations[0][0, 28 + 10] == 0.0
    assert policy.observations[1][0, 28 + 10] == 1.0
    assert policy.observations[2][0, 28 + 10] == 2.0
    # The autoreset on step two restarts the complete history; no frame from
    # the prior episode may leak into the third inference.
    np.testing.assert_allclose(policy.observations[2][0, :28], policy.observations[2][0, 28:])


def test_history_rollout_rejects_normal_35d_owner() -> None:
    env = _CompactResettingEnv()
    policy = _HistoryRecordingPolicy()
    state = env.init_state()
    state.obs["obs"] = np.zeros((1, 35), dtype=np.float32)
    env.init_state = lambda: state  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="compact obs"):
        run_wheelbipe_history_policy(env, policy, steps=1)  # type: ignore[arg-type]


def test_history_rollout_uses_source_reset_declared_by_export_metadata() -> None:
    env = _CompactResettingEnv()
    policy = _HistoryRecordingPolicy(
        history_length=2,
        history_reset_mode="source_zero_current",
    )

    run_wheelbipe_history_policy(
        env,
        policy,  # type: ignore[arg-type]
        steps=3,
        command=(0.3, 0.0, -0.4),
    )

    first = policy.observations[0].reshape(1, 2, 28)
    np.testing.assert_allclose(first[:, 0], 0.0)
    np.testing.assert_allclose(first[0, 1, :3], [0.3, 0.0, -0.4])
    # Step two terminates, so inference three receives a fresh source deque:
    # one zero frame followed by the autoreset-visible current frame.
    third = policy.observations[2].reshape(1, 2, 28)
    np.testing.assert_allclose(third[:, 0], 0.0)
    np.testing.assert_allclose(third[0, 1, :3], [0.3, 0.0, -0.4])
    assert third[0, 1, 10] == 2.0


def test_history_rollout_rejects_reset_override_against_metadata() -> None:
    policy = _HistoryRecordingPolicy(history_reset_mode="source_zero_current")
    with pytest.raises(ValueError, match="does not match policy metadata"):
        run_wheelbipe_history_policy(
            _CompactResettingEnv(),
            policy,  # type: ignore[arg-type]
            steps=1,
            history_reset_mode="repeat",
        )


def test_source_barlow_rollout_uses_zero_warm_start_and_done_reset() -> None:
    env = _CompactResettingEnv()
    policy = _SourceRecordingPolicy()

    diagnostics = run_wheelbipe_source_barlow_policy(env, policy, steps=3)  # type: ignore[arg-type]

    assert diagnostics == {"steps": 3.0, "mean_reward": 0.25, "done_count": 1.0}
    assert len(policy.observations) == 3
    first_obs, first_history = policy.observations[0]
    assert first_obs.shape == (1, 28)
    assert first_history.shape == (1, 10, 28)
    # Source env reset clears all deque slots and appends the current frame as
    # newest; unlike compact history playback, it does not repeat frame 0.
    np.testing.assert_allclose(first_history[0, :-1], 0.0)
    np.testing.assert_allclose(first_history[0, -1], first_obs[0])
    # The autoreset on step two starts a fresh zero-filled history before the
    # third inference; no frame from the previous episode leaks through.
    third_obs, third_history = policy.observations[2]
    np.testing.assert_allclose(third_history[0, :-1], 0.0)
    np.testing.assert_allclose(third_history[0, -1], third_obs[0])
    assert third_history[0, -1, 10] == 2.0


def test_source_barlow_full_rollout_orders_stream_and_resets_history() -> None:
    env = _SourceFullResettingEnv()
    policy = _SourceFullRecordingPolicy()

    diagnostics = run_wheelbipe_source_barlow_full_policy(
        env,
        policy,  # type: ignore[arg-type]
        steps=3,
    )

    assert diagnostics == {"steps": 3.0, "mean_reward": 0.25, "done_count": 1.0}
    assert len(policy.observations) == 3
    first = policy.observations[0]
    assert first.shape == (1, 312)
    np.testing.assert_allclose(first[:, 28:32], [[10.0, 20.0, 30.0, 40.0]])
    first_history = first[:, 32:].reshape(1, 10, 28)
    np.testing.assert_allclose(first_history[:, :-1], 0.0)
    np.testing.assert_allclose(first_history[:, -1], first[:, :28])

    # The second environment step terminates, so inference three must use the
    # autoreset-visible frame/latent and a fresh zero-plus-current history.
    third = policy.observations[2]
    np.testing.assert_allclose(third[:, 28:32], [[12.0, 22.0, 32.0, 42.0]])
    third_history = third[:, 32:].reshape(1, 10, 28)
    np.testing.assert_allclose(third_history[:, :-1], 0.0)
    np.testing.assert_allclose(third_history[:, -1], third[:, :28])
    assert third[0, 10] == 2.0


@pytest.mark.parametrize(
    ("command", "height"),
    [((np.nan, 0.0, 0.0), None), ((np.inf, 0.0, 0.0), None), ((0.0, 0.0, 0.0), np.nan)],
)
def test_history_rollout_rejects_nonfinite_command_overrides(
    command: tuple[float, float, float], height: float | None
) -> None:
    env = _CompactResettingEnv()
    policy = _HistoryRecordingPolicy()

    with pytest.raises(ValueError, match="finite"):
        run_wheelbipe_history_policy(
            env,
            policy,  # type: ignore[arg-type]
            steps=1,
            command=command,
            height=height,
        )

    assert env.calls == 0
    assert policy.observations == []


def _export_history_graph(path: Path, history_length: int = 5) -> None:
    model = torch.nn.Linear(28 * history_length, 6, bias=False).eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (torch.zeros(1, 28 * history_length),),
            str(path),
            input_names=["obs_history"],
            output_names=["actions"],
            opset_version=18,
        )


def _export_source_barlow_full_graph(path: Path) -> None:
    model = torch.nn.Linear(312, 6, bias=False).eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (torch.zeros(1, 312),),
            str(path),
            input_names=["obs_history"],
            output_names=["actions"],
            opset_version=18,
        )


def _write_source_barlow_full_metadata(path: Path) -> None:
    custom_wheelbipe_metadata_path(path).write_text(
        json.dumps(
            {
                "schema": "unilab.wheelbipe.custom_policy.v1",
                "algorithm": "np3o",
                "architecture": "source_barlow",
                "artifact": "source_barlow_full",
                "variant_name": "flat-np3o-barlow-v0",
                "one_step_dim": 28,
                "history_length": 10,
                "history_reset_mode": "source_zero_current",
                "input_dim": 312,
                "output_dim": 6,
                "num_actions": 6,
                "num_estimate": 4,
                "num_costs": 5,
                "input_name": "obs_history",
                "output_name": "actions",
            }
        ),
        encoding="utf-8",
    )


def test_source_barlow_full_export_contract_and_loader(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    model_path = tmp_path / "policy.onnx"
    _export_source_barlow_full_graph(model_path)
    _write_source_barlow_full_metadata(model_path)

    contract = inspect_wheelbipe_source_barlow_full_onnx(model_path)
    assert contract.input_dim == 312
    assert contract.history_feature_dim == 280
    policy = WheelbipeSourceBarlowFullOnnxPolicy(model_path)
    assert policy.predict(np.zeros((2, 312), dtype=np.float32)).shape == (2, 6)
    with pytest.raises(ValueError, match="multiple of 28"):
        WheelbipeHistoryOnnxPolicy(
            model_path,
            expected_history_length=10,
            expected_algorithm="np3o",
        )


def test_custom_onnx_sidecar_validates_algorithm_contract(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    model_path = tmp_path / "policy.onnx"
    _export_history_graph(model_path)
    custom_wheelbipe_metadata_path(model_path).write_text(
        json.dumps(
            {
                "schema": "unilab.wheelbipe.custom_policy.v1",
                "algorithm": "him",
                "variant_name": "flat-him-v0",
                "one_step_dim": 28,
                "history_length": 5,
                "input_dim": 140,
                "output_dim": 6,
                "num_actions": 6,
                "num_costs": 0,
                "input_name": "obs_history",
                "output_name": "actions",
            }
        ),
        encoding="utf-8",
    )

    contract = inspect_wheelbipe_history_onnx(
        model_path,
        expected_history_length=5,
        expected_algorithm="him_ppo",
    )
    assert contract.input_dim == 140
    policy = WheelbipeHistoryOnnxPolicy(
        model_path,
        expected_history_length=5,
        expected_algorithm="him_ppo",
    )
    assert policy.metadata is not None
    assert policy.predict(np.zeros((1, 140), dtype=np.float32)).shape == (1, 6)


def test_custom_onnx_sidecar_rejects_algorithm_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    model_path = tmp_path / "policy.onnx"
    _export_history_graph(model_path)
    custom_wheelbipe_metadata_path(model_path).write_text(
        json.dumps(
            {
                "schema": "unilab.wheelbipe.custom_policy.v1",
                "algorithm": "np3o",
                "history_length": 10,
                "num_costs": 5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="history_length|algorithm"):
        inspect_wheelbipe_history_onnx(
            model_path,
            expected_history_length=5,
            expected_algorithm="him",
        )


class _SourceActorExportGraph(torch.nn.Module):
    """Small two-input graph used to exercise the source export ABI."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(28 + 10 * 28, 6)

    def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((obs, obs_hist.reshape(obs.shape[0], -1)), dim=-1))


class _SourceActorExportPolicy:
    source_architecture = True
    num_prop = 28
    num_hist = 10
    num_actions = 6

    def __init__(self) -> None:
        self.actor_teacher_backbone = _SourceActorExportGraph().eval()


def test_source_barlow_dual_input_export_and_loaders(tmp_path: Path) -> None:
    """The source actor ABI remains distinct from the flattened 312D graph."""

    pytest.importorskip("onnxruntime")
    policy = _SourceActorExportPolicy()
    jit_path, onnx_path = export_barlow_twins_actor_from_policy(
        policy,
        tmp_path,
        jit_filename="barlow_actor.pt",
        onnx_filename="barlow_actor.onnx",
        variant_name="flat-np3o-barlow-source-v0",
    )

    assert Path(jit_path).is_file()
    assert Path(onnx_path).is_file()
    metadata = json.loads(custom_wheelbipe_metadata_path(onnx_path).read_text(encoding="utf-8"))
    assert metadata["artifact"] == "source_barlow_actor"
    assert metadata["architecture"] == "source_barlow"
    assert metadata["input_names"] == ["obs", "obs_hist"]
    assert metadata["input_shapes"] == [[1, 28], [1, 10, 28]]
    assert metadata["output_shape"] == [1, 6]
    assert metadata["source_stream_dim"] == 312
    assert metadata["history_reset_mode"] == "source_zero_current"

    contract = inspect_wheelbipe_source_barlow_onnx(
        onnx_path,
        expected_history_length=10,
        expected_algorithm="np3o",
    )
    assert isinstance(contract, WheelbipeSourceBarlowPolicyContract)
    assert contract.input_names == ("obs", "obs_hist")
    assert contract.history_shape == (1, 10, 28)
    assert contract.source_stream_dim == 312

    obs = np.arange(28, dtype=np.float32)[None, :]
    hist = np.arange(10 * 28, dtype=np.float32).reshape(1, 10, 28)
    expected = (
        policy.actor_teacher_backbone(torch.from_numpy(obs), torch.from_numpy(hist))
        .detach()
        .numpy()
    )
    onnx_policy = WheelbipeSourceBarlowOnnxPolicy(onnx_path)
    # ONNX Runtime and eager Torch can differ by a few ulps around BatchNorm
    # and fused linear kernels; retain a tight export-level tolerance without
    # making the test flaky across CPU instruction sets.
    np.testing.assert_allclose(onnx_policy.predict(obs, hist), expected, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        onnx_policy.predict(obs[0], hist[0]), expected[0], rtol=5e-5, atol=5e-5
    )

    jit_policy = WheelbipeSourceBarlowTorchScriptPolicy(jit_path)
    np.testing.assert_allclose(jit_policy.predict(obs, hist), expected, rtol=1e-5, atol=1e-5)


def test_source_barlow_dual_input_contract_rejects_flattened_graph(tmp_path: Path) -> None:
    """The one-input 312D runner export must not load through the source ABI."""

    pytest.importorskip("onnxruntime")
    model_path = tmp_path / "flattened.onnx"
    model = torch.nn.Linear(312, 6, bias=False).eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (torch.zeros(1, 312),),
            str(model_path),
            input_names=["obs_history"],
            output_names=["actions"],
            opset_version=18,
        )
    with pytest.raises(ValueError, match="two inputs"):
        inspect_wheelbipe_source_barlow_onnx(model_path)


def test_source_barlow_torchscript_loader_accepts_fp16_export(tmp_path: Path) -> None:
    """The upstream optional half-precision JIT artifact remains loadable."""

    class _HalfActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(28 + 10 * 28, 6)

        def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
            return self.linear(torch.cat((obs, obs_hist.reshape(obs.shape[0], -1)), dim=-1))

    jit_path = export_mlp_barlow_twins_actor_torchscript(
        _HalfActor(),
        28,
        10,
        tmp_path,
        use_fp16=True,
    )
    policy = WheelbipeSourceBarlowTorchScriptPolicy(jit_path)
    assert policy.input_dtype == torch.float16
    output = policy.predict(
        np.zeros((1, 28), dtype=np.float32),
        np.zeros((1, 10, 28), dtype=np.float32),
    )
    assert output.shape == (1, 6)
    assert np.all(np.isfinite(output))


def test_source_barlow_torchscript_loader_rejects_integer_output(tmp_path: Path) -> None:
    """The source actor ABI requires floating-point actions, not castable ints."""

    class _IntegerActor(torch.nn.Module):
        def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
            del obs_hist
            return torch.zeros((obs.shape[0], 6), dtype=torch.int64)

    path = tmp_path / "integer_actor.pt"
    traced = cast(
        torch.jit.ScriptModule,
        torch.jit.trace(
            _IntegerActor(),
            (torch.zeros(1, 28), torch.zeros(1, 10, 28)),
        ),
    )
    traced.save(str(path))
    with pytest.raises(ValueError, match="floating"):
        WheelbipeSourceBarlowTorchScriptPolicy(path)


def test_source_barlow_torchscript_loader_rejects_one_input_runner_graph(
    tmp_path: Path,
) -> None:
    """A compact/runner JIT graph must not be mistaken for the source actor."""

    model_path = tmp_path / "policy.pt"
    runner_graph = torch.nn.Linear(312, 6).eval()
    traced = cast(
        torch.jit.ScriptModule,
        torch.jit.trace(runner_graph, torch.zeros(1, 312)),
    )
    traced.save(str(model_path))

    with pytest.raises(ValueError, match="two-input"):
        WheelbipeSourceBarlowTorchScriptPolicy(model_path)
