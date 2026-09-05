#!/usr/bin/env -S uv run --script
"""Inspect/package native ROS2 sources or run the no-ROS UniLab adapter.

The upstream ROS2 executable needs ROS2 Humble, ros2_control and a C++ ONNX
runtime plugin.  The default rollout path is the supported no-ROS fallback: it
feeds the same normal 35D/6D policy boundary through
``unilab.training.wheelbipe_ros2.WheelbipeRos2Controller`` and advances the
existing UniLab owner.  It intentionally creates no ROS graph and never opens
the RealBridge serial device.  Cold-path flags can verify or materialize the
pinned native colcon sources, but never build, source, launch, or operate them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.wheelbipe_v14 import (
    WheelbipeRewardConfig,
    wheelbipe_delay_profile_overrides,
)
from unilab.training.wheelbipe import DEFAULT_WHEELBIPE_POLICY, WheelbipeOnnxPolicy
from unilab.training.wheelbipe_ros2 import (
    WheelbipeRos2Controller,
    WheelbipeRos2NativeBundleError,
    load_wheelbipe_ros2_config,
    materialize_wheelbipe_ros2_native_workspace,
    require_wheelbipe_ros2_native_runtime,
    verify_wheelbipe_ros2_native_bundle,
    wheelbipe_robot_state_from_policy_observation,
    wheelbipe_ros2_contract_snapshot,
    wheelbipe_ros2_native_runtime_probe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_WHEELBIPE_POLICY,
        help="normal 35D->6D ONNX graph (defaults to the vendored artifact)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "deployment YAML for the source controller parameters "
            "(defaults to conf/deployment/wheelbipe_v14_ros2.yaml)"
        ),
    )
    parser.add_argument("--sim", choices=("mujoco", "motrix"), default="mujoco")
    parser.add_argument("--steps", type=int, default=200, help="UniLab control steps")
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "YAW"),
        default=None,
        help="override [linear-x, linear-y, yaw-rate]; ROS2 controller consumes VX and YAW",
    )
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--delay-profile",
        choices=("local_physics", "source_v14_physics"),
        default="local_physics",
        help="UniLab owner timing profile; source_v14_physics is timing-only",
    )
    cold_path = parser.add_mutually_exclusive_group()
    cold_path.add_argument(
        "--print-contract",
        action="store_true",
        help=(
            "print the source API/protocol snapshot and exit without loading a model "
            "or materializing a simulator"
        ),
    )
    cold_path.add_argument(
        "--verify-native-bundle",
        action="store_true",
        help=(
            "verify package manifests, plugin XML/CMake registration, launch/config syntax, "
            "source hashes, and asset overlays; does not import or run ROS"
        ),
    )
    cold_path.add_argument(
        "--probe-native-runtime",
        action="store_true",
        help=(
            "report optional ROS Python modules/executables and bundle status without "
            "importing ROS or creating a graph"
        ),
    )
    cold_path.add_argument(
        "--require-native-runtime",
        action="store_true",
        help=(
            "fail unless the static bundle and minimum external ROS/colcon tools exist; "
            "still does not build or launch"
        ),
    )
    cold_path.add_argument(
        "--materialize-native-workspace",
        type=Path,
        metavar="NEW_WORKSPACE",
        help=(
            "copy the verified colcon sources plus pinned mesh/policy assets into a new "
            "destination; existing paths are rejected and nothing is built or launched"
        ),
    )
    return parser


def _close_env(env: object) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _apply_env_overrides(state: Any, command: np.ndarray | None, height: float | None) -> None:
    """Keep env observations and controller topic values coherent after reset."""

    policy_obs = np.asarray(state.obs["obs"])
    if policy_obs.ndim != 2 or policy_obs.shape[1] != 35:
        raise ValueError(
            f"ROS2 adapter requires env obs['obs'] shape (N, 35), got {policy_obs.shape}"
        )
    if command is not None:
        commands = np.asarray(state.info.get("commands"), dtype=np.float64)
        if commands.ndim != 2 or commands.shape[1] != 3:
            raise ValueError("Wheelbipe owner must expose batched info['commands'] with width 3")
        commands[...] = command
        state.info["commands"] = commands
        policy_obs[:, :3] = np.clip(command, -100.0, 100.0)
    if height is not None:
        heights = np.asarray(state.info.get("height_commands"), dtype=np.float64)
        if heights.ndim != 1:
            raise ValueError("Wheelbipe owner must expose batched info['height_commands']")
        heights[...] = height
        state.info["height_commands"] = heights
        policy_obs[:, 3] = np.clip(height * 5.0, -100.0, 100.0)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_contract:
        # Keep this path entirely on the cold metadata boundary: no ONNX
        # session, registry materialization, simulator, DDS graph or serial
        # descriptor is touched.  A caller may still provide --config to audit
        # an alternate source-compatible parameter file.
        config = load_wheelbipe_ros2_config(args.config)
        print(json.dumps(wheelbipe_ros2_contract_snapshot(config), sort_keys=True))
        return 0
    if args.verify_native_bundle:
        try:
            result = verify_wheelbipe_ros2_native_bundle()
        except WheelbipeRos2NativeBundleError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.probe_native_runtime:
        print(json.dumps(wheelbipe_ros2_native_runtime_probe(), sort_keys=True))
        return 0
    if args.require_native_runtime:
        try:
            result = require_wheelbipe_ros2_native_runtime()
        except WheelbipeRos2NativeBundleError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.materialize_native_workspace is not None:
        try:
            result = materialize_wheelbipe_ros2_native_workspace(args.materialize_native_workspace)
        except (OSError, WheelbipeRos2NativeBundleError) as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.seed is not None:
        np.random.seed(args.seed)
    command = None if args.command is None else np.asarray(args.command, dtype=np.float64)
    if command is not None and not np.all(np.isfinite(command)):
        raise SystemExit("--command must contain only finite values")
    if args.height is not None and not np.isfinite(float(args.height)):
        raise SystemExit("--height must be finite")

    ensure_registries()
    # Loading/validating the graph before materializing a simulator preserves
    # the source controller's strict one-input/one-output boundary.
    policy = WheelbipeOnnxPolicy(args.model)
    # Parse the owner deployment profile once on the cold path.  The CLI is an
    # unattended rollout, so it intentionally requests RL after the source
    # INIT hold while preserving every other value from the checked-in YAML.
    deployment_config = load_wheelbipe_ros2_config(args.config)
    controller = WheelbipeRos2Controller(
        policy,
        config=replace(deployment_config, auto_enter_rl=True),
    )
    controller.on_init()
    controller.on_configure()
    controller.on_activate()

    profile_overrides = wheelbipe_delay_profile_overrides(args.delay_profile)
    profile_overrides.update(
        {
            "reward_config": WheelbipeRewardConfig(),
            "motrix_disable_equality": args.sim == "motrix",
        }
    )
    env = registry.make(
        "WheelbipeV14Flat",
        sim_backend=args.sim,
        num_envs=1,
        env_cfg_override=profile_overrides,
    )
    state = None
    owner_timing: dict[str, Any] = {}
    total_reward = 0.0
    done_count = 0
    inference_count = 0
    sim_time = 0.0
    # The owner normally advances one 20 ms control interval per ``step``.
    # Resolve the number of adapter ticks from the selected deployment config
    # instead of silently hard-coding the source defaults; a non-integral ratio
    # cannot represent the source fixed-rate loop and fails before rollout.
    controller_updates_per_env_step: int | None = None
    physics_dt: float | None = None
    try:
        state = env.init_state()
        owner_timing = dict(getattr(env, "timing_contract", {}))
        control_dt = float(owner_timing.get("ctrl_dt", 0.02))
        update_rate = int(deployment_config.update_rate_hz)
        ratio = control_dt * float(update_rate)
        rounded_ratio = int(round(ratio))
        if rounded_ratio < 1 or abs(ratio - rounded_ratio) > 1.0e-9:
            raise RuntimeError(
                "ROS2 adapter update_rate_hz must produce an integral number of ticks "
                f"per owner control step; got update_rate_hz={update_rate}, ctrl_dt={control_dt}"
            )
        controller_updates_per_env_step = rounded_ratio
        physics_dt = 1.0 / float(update_rate)
        for _step in range(args.steps):
            _apply_env_overrides(state, command, args.height)
            action = np.zeros((1, 6), dtype=np.float32)
            for _ in range(controller_updates_per_env_step):
                observation = np.asarray(state.obs["obs"][0], dtype=np.float32)
                robot_state = wheelbipe_robot_state_from_policy_observation(
                    observation,
                    timestamp=sim_time,
                    period=physics_dt,
                )
                if command is not None:
                    controller.set_motion_command(
                        float(command[0]), float(command[2]), timestamp=sim_time
                    )
                if args.height is not None:
                    controller.set_height_command(float(args.height), timestamp=sim_time)
                output = controller.update(
                    robot_state,
                    timestamp=sim_time,
                    period=physics_dt,
                )
                if output.action is not None:
                    action[0] = np.asarray(output.action, dtype=np.float32)
                inference_count = int(controller.inference_count)
                sim_time += physics_dt
            state = env.step(action)
            total_reward += float(np.asarray(state.reward).reshape(-1)[0])
            done = bool(np.asarray(state.terminated).reshape(-1)[0]) or bool(
                np.asarray(state.truncated).reshape(-1)[0]
            )
            if done:
                done_count += 1
                controller.reset_runtime(sim_time)
    finally:
        _close_env(env)
        controller.on_deactivate()
    mean_reward = total_reward / float(args.steps)
    print(
        "Wheelbipe ROS2 Python adapter complete: "
        f"adapter=python native_ros2=false ros_graph=false sample_hold=true sim={args.sim} steps={args.steps} "
        f"controller_rate_hz={deployment_config.update_rate_hz} "
        f"inference_rate_hz={deployment_config.inference_frequency_hz} "
        f"inference_count={inference_count} mean_reward={mean_reward:.6f} "
        f"done_count={done_count} timing_profile={owner_timing.get('profile', args.delay_profile)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
