from __future__ import annotations

import numpy as np
import pytest

from unilab.base.backend import MOTRIX_AVAILABLE, create_backend
from unilab.envs.locomotion.wheelbipe_v14.joystick import WheelbipeV14FlatCfg


@pytest.mark.parametrize(
    "backend_type",
    [
        "mujoco",
        pytest.param(
            "motrix",
            marks=pytest.mark.skipif(not MOTRIX_AVAILABLE, reason="motrixsim is unavailable"),
        ),
    ],
)
def test_wheelbipe_body_contact_force_contract_is_vectorized(backend_type: str) -> None:
    cfg = WheelbipeV14FlatCfg()
    backend = create_backend(
        backend_type,
        cfg.scene,
        2,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
    )
    try:
        backend.materialize()
        body_ids = backend.get_body_ids(["left_wheel_link", "right_wheel_link", "base_link"])
        force_norm = backend.get_body_contact_force_norm(body_ids)
        assert force_norm.shape == (2, 3)
        assert np.issubdtype(force_norm.dtype, np.floating)
        assert np.all(np.isfinite(force_norm))
        assert np.all(force_norm >= 0.0)
    finally:
        backend.cleanup_scene_assets()


def test_wheelbipe_contact_force_contract_rejects_unsensored_body() -> None:
    cfg = WheelbipeV14FlatCfg()
    backend = create_backend(
        "mujoco",
        cfg.scene,
        1,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
    )
    try:
        backend.materialize()
        # The migrated source asset intentionally exposes scalar touch sensors
        # for the front suspension bodies.  Use a spring link, which remains
        # unsensored, to exercise the backend's missing-sensor contract.
        body_ids = backend.get_body_ids(["left_spring1_link"])
        with pytest.raises(NotImplementedError, match="missing scalar touch sensors"):
            backend.get_body_contact_force_norm(body_ids)
    finally:
        backend.cleanup_scene_assets()
