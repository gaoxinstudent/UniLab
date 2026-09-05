#!/usr/bin/env -S uv run --script
"""Run a released Wheelbipe V14 ONNX or TorchScript policy in a UniLab owner.

This is the reproducible sim2sim boundary: the same 35D normal-mode vector
used by the ROS2 controller is produced by the UniLab env, checked against the
ONNX graph, and fed back as six policy actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.wheelbipe_v14 import (
    WHEELBIPE_DELAY_PROFILES,
    WheelbipeRewardConfig,
    wheelbipe_delay_profile_overrides,
)
from unilab.training.wheelbipe import (
    DEFAULT_WHEELBIPE_POLICY,
    WheelbipeOnnxPolicy,
    WheelbipeTorchScriptPolicy,
    is_wheelbipe_torchscript_archive,
    run_wheelbipe_policy,
)
from unilab.visualization.wheelbipe_trace import (
    WheelbipeRealtimeBuffer,
    WheelbipeRealtimePlotter,
    WheelbipeTraceRecorder,
    capture_wheelbipe_playback_telemetry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_WHEELBIPE_POLICY,
        help=(
            "35D->6D ONNX policy or trusted TorchScript policy.pt "
            "(defaults to the vendored ONNX deployment artifact)"
        ),
    )
    parser.add_argument(
        "--sim",
        choices=("mujoco", "motrix"),
        default="mujoco",
        help="simulation backend (the Motrix owner uses its explicit constraint profile)",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "YAW"),
        default=None,
        help="override [linear-x, linear-y, yaw-rate] command",
    )
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--delay-profile",
        choices=WHEELBIPE_DELAY_PROFILES,
        default="local_physics",
        help=(
            "owner timing/delay profile (local_physics selects the 1 ms, "
            "control-step, delays-off owner); source_v14_physics is explicit "
            "timing only and makes no source dynamics/asset parity claim"
        ),
    )
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=None,
        help="write the source-compatible velocity/reward trace CSV",
    )
    parser.add_argument(
        "--trace-html",
        type=Path,
        default=None,
        help="interactive HTML path (requires --trace-csv; defaults beside the CSV)",
    )
    parser.add_argument("--trace-env-id", type=int, default=0)
    parser.add_argument("--trace-sample-dt", type=float, default=0.0)
    parser.add_argument("--trace-max-rows", type=int, default=20_000)
    parser.add_argument(
        "--realtime-plot",
        action="store_true",
        help="open the optional Matplotlib WheelBipe telemetry panels",
    )
    parser.add_argument("--realtime-max-points", type=int, default=200)
    parser.add_argument("--realtime-update-interval", type=int, default=5)
    return parser


def _close_env(env: object) -> None:
    """Invoke the public environment lifecycle hook when one is provided."""

    close = getattr(env, "close", None)
    if callable(close):
        close()


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be positive")
    if args.trace_html is not None and args.trace_csv is None:
        raise SystemExit("--trace-html requires --trace-csv")
    if args.trace_env_id < 0 or args.trace_env_id >= args.num_envs:
        raise SystemExit("--trace-env-id must select one configured environment")
    if not np.isfinite(args.trace_sample_dt) or args.trace_sample_dt < 0.0:
        raise SystemExit("--trace-sample-dt must be finite and non-negative")
    if args.trace_max_rows < 1:
        raise SystemExit("--trace-max-rows must be positive")
    if args.realtime_max_points < 1 or args.realtime_update_interval < 1:
        raise SystemExit("realtime plot sizes/intervals must be positive")
    if args.seed is not None:
        np.random.seed(args.seed)

    ensure_registries()
    policy: WheelbipeOnnxPolicy | WheelbipeTorchScriptPolicy
    if is_wheelbipe_torchscript_archive(args.model):
        # Sim2sim deliberately remains CPU inference, matching the ROS
        # deployment contract.  The owner adapter validates the same strict
        # 35D->6D graph before materializing the simulator.
        policy = WheelbipeTorchScriptPolicy(args.model, device="cpu")
    else:
        policy = WheelbipeOnnxPolicy(args.model)
    reward_cfg = WheelbipeRewardConfig()
    profile_overrides = wheelbipe_delay_profile_overrides(args.delay_profile)
    profile_overrides.update(
        {
            "reward_config": reward_cfg,
            # Keep the solver compatibility choice explicit at the owner
            # boundary.  It is ignored by the MuJoCo backend.
            "motrix_disable_equality": args.sim == "motrix",
        }
    )
    env = registry.make(
        "WheelbipeV14Flat",
        sim_backend=args.sim,
        num_envs=args.num_envs,
        env_cfg_override=profile_overrides,
    )
    # Read timing metadata before backend teardown; ``close`` may invalidate
    # backend-owned scene/timing state on some implementations.  Keep every
    # post-create access inside the guarded lifecycle so even malformed timing
    # metadata cannot leak an environment.
    timing_contract = {}
    diagnostics: dict[str, float] | None = None
    recorder: WheelbipeTraceRecorder | None = None
    realtime_buffer: WheelbipeRealtimeBuffer | None = None
    realtime_plotter: WheelbipeRealtimePlotter | None = None
    try:
        timing_contract = dict(getattr(env, "timing_contract"))
        if args.trace_csv is not None:
            recorder = WheelbipeTraceRecorder(
                args.trace_csv,
                html_path=args.trace_html,
                reward_scales=reward_cfg.scales,
                sample_dt=args.trace_sample_dt,
                max_rows=args.trace_max_rows,
            )
        if args.realtime_plot:
            realtime_buffer = WheelbipeRealtimeBuffer(
                max_points=args.realtime_max_points,
                num_leg_joints=4,
            )
            realtime_plotter = WheelbipeRealtimePlotter(
                realtime_buffer,
                update_interval=args.realtime_update_interval,
            )

        def publish_step(_state: object, completed_steps: int) -> None:
            if recorder is None and realtime_buffer is None:
                return
            telemetry = capture_wheelbipe_playback_telemetry(
                env,
                sim_time_s=completed_steps * float(timing_contract["ctrl_dt"]),
                env_id=args.trace_env_id,
            )
            if recorder is not None:
                recorder.append(telemetry.trace_row)
            if realtime_buffer is not None:
                realtime_buffer.append(telemetry.realtime_sample)

        if recorder is not None or realtime_buffer is not None:
            # Preserve the pre-tooling call shape when no consumer is active.
            # Embedders commonly replace ``run_wheelbipe_policy`` with a
            # minimal rollout callable, and a meaningless ``None`` keyword
            # should not force those call sites to adopt the optional tracing
            # extension.
            diagnostics = run_wheelbipe_policy(
                env,
                policy,
                steps=args.steps,
                command=args.command,
                height=args.height,
                step_callback=publish_step,
            )
        else:
            diagnostics = run_wheelbipe_policy(
                env,
                policy,
                steps=args.steps,
                command=args.command,
                height=args.height,
            )
    finally:
        try:
            if recorder is not None:
                recorder.close()
            if realtime_plotter is not None:
                realtime_plotter.close()
        finally:
            _close_env(env)
    if diagnostics is None:
        raise RuntimeError("Wheelbipe sim2sim rollout did not return diagnostics")
    print(
        "Wheelbipe sim2sim complete: "
        f"delay_profile={timing_contract['profile']} "
        f"sim_dt={timing_contract['sim_dt']:.6f} "
        f"ctrl_dt={timing_contract['ctrl_dt']:.6f} "
        f"steps={int(diagnostics['steps'])} "
        f"mean_reward={diagnostics['mean_reward']:.6f} "
        f"done_count={int(diagnostics['done_count'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
