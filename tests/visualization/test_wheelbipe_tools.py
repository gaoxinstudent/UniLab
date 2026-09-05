"""Focused contracts for migrated WheelBipe play/trace tooling."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.visualization.wheelbipe_keyboard import WheelbipeKeyboardController
from unilab.visualization.wheelbipe_trace import (
    WHEELBIPE_TRACE_FIELDS,
    WheelbipeRealtimeBuffer,
    WheelbipeRealtimeSample,
    WheelbipeTraceRecorder,
    build_wheelbipe_trace_html,
    capture_wheelbipe_playback_telemetry,
    export_wheelbipe_trace_html,
    normalize_wheelbipe_trace_row,
)


def _controller() -> WheelbipeKeyboardController:
    return WheelbipeKeyboardController.from_vel_limit(
        [[-0.6, 0.0, -0.8], [1.0, 0.0, 0.8]],
        default_height=0.22,
        height_range=(0.20, 0.42),
    )


def test_keyboard_preserves_source_press_release_height_and_jump_contract() -> None:
    controller = _controller()
    info = {
        "commands": np.zeros((1, 3), dtype=np.float32),
        "height_commands": np.asarray([0.22], dtype=np.float32),
    }

    assert controller.key_press("w") is True
    assert controller.key_press("a") is True
    assert controller.key_press("z") is True
    controller.advance(0.5)
    assert controller.key_release("W") is True
    assert controller.key_release("A") is True
    assert controller.key_release("Z") is True
    assert controller.key_press("Q") is True
    controller.apply(info)

    assert info["commands"].tolist() == [[0.0, 0.0, 0.0]]
    assert info["height_commands"][0] == pytest.approx(0.27)
    assert info["jump_takeoff_request"].tolist() == [True]

    # Q is transferred once.  The frontend never triggers a jump method or
    # bypasses the state machine; an already-present flag is left untouched.
    info["jump_takeoff_request"][:] = False
    controller.apply(info)
    assert info["jump_takeoff_request"].tolist() == [False]


def test_keyboard_viewer_mode_latches_velocity_but_nudges_height_safely() -> None:
    controller = _controller()
    controller.viewer_key("W")
    controller.viewer_key("A")
    controller.viewer_key("Z")

    assert controller.command.tolist() == pytest.approx([0.8, 0.0, 0.8])
    assert controller.current_height == pytest.approx(0.32)

    controller.viewer_key("L")
    assert controller.command.tolist() == [0.0, 0.0, 0.0]
    assert controller.current_height == pytest.approx(0.22)


def _trace_row(*, time: float = 0.02, terrain: str = "flat") -> dict[str, object]:
    return {
        "sim_time_s": time,
        "episode_time_s": time,
        "env_id": 0,
        "terrain": terrain,
        "cmd_x": 0.8,
        "cmd_y": 0.0,
        "cmd_yaw": 0.1,
        "vel_x_b": 0.7,
        "vel_y_b": 0.0,
        "yaw_rate_b": 0.09,
        "height_cmd": 0.22,
        "height_obs": 0.21,
        "height_relative": 0.21,
        "height_reward_ref": 0.22,
        "airborne": 0,
        "reward_total": 1.25,
        "reward_tracking": 1.0,
    }


def test_trace_html_is_self_contained_interactive_and_script_safe() -> None:
    hostile = "flat</script><img src=x onerror=alert(1)>"
    html = build_wheelbipe_trace_html([_trace_row(terrain=hostile)])

    assert '<canvas id="plot">' in html
    assert "Reset zoom" in html
    assert 'addEventListener("wheel"' in html
    assert "reward heatmap" in html
    assert "</script><img" not in html
    assert "\\u003c/script\\u003e" in html


@pytest.mark.parametrize(
    ("field", "value"),
    [("sim_time_s", float("nan")), ("airborne", 2), ("env_id", -1)],
)
def test_trace_row_rejects_malformed_numeric_contract(field: str, value: object) -> None:
    row = _trace_row()
    row[field] = value
    with pytest.raises(ValueError, match=field):
        normalize_wheelbipe_trace_row(row)


def test_trace_recorder_samples_stream_and_offline_export(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace.csv"
    html_path = tmp_path / "live.html"
    with WheelbipeTraceRecorder(
        csv_path,
        html_path=html_path,
        reward_scales={"tracking": 1.0},
        sample_dt=0.05,
    ) as recorder:
        assert recorder.append(_trace_row(time=0.02)) is True
        assert recorder.append(_trace_row(time=0.04)) is False
        assert recorder.append(_trace_row(time=0.08)) is True

    with csv_path.open(newline="", encoding="utf-8") as stream:
        parsed = list(csv.DictReader(stream))
    assert len(parsed) == 2
    assert list(parsed[0]) == [*WHEELBIPE_TRACE_FIELDS, "reward_tracking"]
    assert html_path.is_file()

    exported = export_wheelbipe_trace_html(csv_path, html_path=tmp_path / "offline.html")
    assert exported.name == "offline.html"
    assert "WheelBipe trace" in exported.read_text(encoding="utf-8")


def test_realtime_buffer_is_bounded_and_publishes_copied_snapshot() -> None:
    buffer = WheelbipeRealtimeBuffer(max_points=2, num_leg_joints=4)
    received: list[int] = []
    buffer.subscribe(lambda sample: received.append(sample.step))
    for step in range(3):
        buffer.append(
            WheelbipeRealtimeSample(
                step=step,
                target_height=0.22,
                actual_height=0.21,
                jump_phase=0.0,
                wheel_power=(1.0, 2.0),
                leg_torques=(3.0, 4.0, 5.0, 6.0),
                spring_forces=(7.0, 8.0),
            )
        )

    snapshot = buffer.snapshot()
    assert received == [0, 1, 2]
    assert snapshot["step"].tolist() == [1, 2]
    assert snapshot["leg_torques"].shape == (2, 4)
    snapshot["step"][:] = 99
    assert buffer.snapshot()["step"].tolist() == [1, 2]


def test_capture_uses_public_owner_state_without_mislabeling_log_means() -> None:
    policy_obs = np.zeros((1, 35), dtype=np.float32)
    policy_obs[0, 6] = 0.25
    state = SimpleNamespace(
        obs={"obs": policy_obs},
        reward=np.asarray([1.5], dtype=np.float32),
        info={
            "commands": np.asarray([[0.8, 0.0, -0.2]], dtype=np.float32),
            "height_commands": np.asarray([0.24], dtype=np.float32),
            "observed_height": np.asarray([0.23], dtype=np.float32),
            "torques": np.arange(1, 9, dtype=np.float32)[None, :],
            "steps": np.asarray([3], dtype=np.uint32),
            "state_machine_airborne": np.asarray([True]),
            "state_machine_jump_phase": np.asarray([2], dtype=np.int8),
            "log": {"reward/tracking": 123.0},
        },
    )

    class FakeEnv:
        cfg = SimpleNamespace(
            ctrl_dt=0.02,
            reward_config=SimpleNamespace(base_height_target=0.22),
        )

        def __init__(self) -> None:
            self.state = state

        def get_local_linvel(self) -> np.ndarray:
            return np.asarray([[0.7, 0.1, 0.0]], dtype=np.float32)

        def get_full_dof_vel(self) -> np.ndarray:
            return np.arange(2, 10, dtype=np.float32)[None, :]

    telemetry = capture_wheelbipe_playback_telemetry(FakeEnv(), sim_time_s=0.06)

    assert telemetry.trace_row["yaw_rate_b"] == pytest.approx(0.5)
    assert telemetry.trace_row["episode_time_s"] == pytest.approx(0.06)
    assert telemetry.trace_row["reward_total"] == pytest.approx(1.5)
    assert "reward_tracking" not in telemetry.trace_row
    assert telemetry.realtime_sample.wheel_power == pytest.approx((12.0, 56.0))
