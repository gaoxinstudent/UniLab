"""Boundary tests for upstream Wheelbipe TorchScript policy playback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from unilab.training.wheelbipe import (
    WheelbipeTorchScriptPolicy,
    is_wheelbipe_torchscript_archive,
)


def _save_traced_policy(path: Path, *, output_dim: int = 6) -> None:
    torch.manual_seed(7)
    module = torch.nn.Sequential(torch.nn.Linear(35, output_dim), torch.nn.Tanh()).eval()
    traced = torch.jit.trace(module, torch.zeros(1, 35))
    traced.save(str(path))


def test_torchscript_policy_accepts_numpy_and_tensordict_inputs(tmp_path: Path) -> None:
    model_file = tmp_path / "policy.pt"
    _save_traced_policy(model_file)

    assert is_wheelbipe_torchscript_archive(model_file)
    policy = WheelbipeTorchScriptPolicy(model_file)
    observations = np.zeros((2, 35), dtype=np.float32)

    actions = policy.predict(observations)
    assert actions.shape == (2, 6)
    assert np.isfinite(actions).all()

    # RslRlVecEnvWrapper exposes actor observations in a TensorDict.  The
    # owner-layer adapter must select the actor stream instead of coercing the
    # entire TensorDict into a misleading (N, 16) object-array shape.
    actor_observations = TensorDict(
        {
            "actor": torch.zeros((2, 35)),
            "critic": torch.zeros((2, 78)),
        },
        batch_size=[2],
    )
    tensor_actions = policy(actor_observations)
    assert isinstance(tensor_actions, torch.Tensor)
    assert tensor_actions.shape == (2, 6)
    assert tensor_actions.device.type == "cpu"
    assert torch.isfinite(tensor_actions).all()


@pytest.mark.parametrize("output_dim", [5, 7])
def test_torchscript_policy_rejects_wrong_action_contract(tmp_path: Path, output_dim: int) -> None:
    model_file = tmp_path / f"policy-{output_dim}.pt"
    _save_traced_policy(model_file, output_dim=output_dim)

    with pytest.raises(ValueError, match=r"strict \(35,\) -> \(6,\) actor contract"):
        WheelbipeTorchScriptPolicy(model_file)


def test_torchscript_archive_detector_rejects_non_archive(tmp_path: Path) -> None:
    model_file = tmp_path / "weights.pt"
    torch.save({"actor_state_dict": {}}, model_file)

    assert not is_wheelbipe_torchscript_archive(model_file)


def test_torchscript_archive_detector_accepts_minimal_code_layout(tmp_path: Path) -> None:
    """Parameter-free traces use ``code/__torch__.py`` without a module dir."""

    class _IdentityActor(torch.nn.Module):
        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            return torch.zeros((observation.shape[0], 6), dtype=observation.dtype)

    model_file = tmp_path / "minimal-policy.pt"
    torch.jit.trace(_IdentityActor(), torch.zeros(1, 35)).save(str(model_file))
    assert is_wheelbipe_torchscript_archive(model_file)
    policy = WheelbipeTorchScriptPolicy(model_file)
    assert policy.predict(np.zeros((1, 35), dtype=np.float32)).shape == (1, 6)
