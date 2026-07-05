from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .runtime import GamepadCommander, KeyboardCommander, Real68Sim2Sim


def _default_bundle_dir() -> Path:
    root = Path(__file__).resolve().parent
    bundles = root / "bundles"
    if not bundles.exists():
        raise FileNotFoundError(
            f"No bundles directory found at {bundles}. Run prepare_bundle.py first."
        )
    candidates = sorted(path for path in bundles.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No bundle directories found under {bundles}. Run prepare_bundle.py first."
        )
    return candidates[-1]


def _headless_run(sim: Real68Sim2Sim, *, steps: int) -> None:
    start_forward = float(sim.data.qpos[sim.forward_axis])
    linvel_samples: list[float] = []
    for _ in range(steps):
        sim.step()
        sim.maybe_print_status()
        linvel = sim.sensors.read(sim.data, sim.cfg["sensor_names"]["local_linvel"])
        linvel_samples.append(float(linvel[sim.forward_axis] * sim.forward_sign))
    distance = float((sim.data.qpos[sim.forward_axis] - start_forward) * sim.forward_sign)
    mean_vx = float(np.mean(linvel_samples)) if linvel_samples else 0.0
    print(
        f"[sim2sim] headless summary: steps={steps} distance_forward={distance:.3f} mean_vx={mean_vx:.3f}"
    )


def _build_commander(
    sim: Real68Sim2Sim,
    *,
    input_device: str,
    joystick_index: int,
    deadzone: float,
    vx_scale: float | None,
    wz_scale: float | None,
):
    if input_device == "keyboard":
        commander = KeyboardCommander()
        commander.command[:] = sim.command
        return commander

    command_limits = np.asarray(sim.cfg["command_limits"], dtype=np.float64)
    default_vx_scale = float(
        max(abs(command_limits[0, 0]), abs(command_limits[1, 0]), abs(sim.command[0]), 0.8)
    )
    default_wz_scale = float(
        max(abs(command_limits[0, 2]), abs(command_limits[1, 2]), abs(sim.command[2]), 0.4)
    )
    return GamepadCommander(
        joystick_index=joystick_index,
        deadzone=deadzone,
        vx_scale=default_vx_scale if vx_scale is None else float(vx_scale),
        wz_scale=default_wz_scale if wz_scale is None else float(wz_scale),
    )


def _interactive_run(
    sim: Real68Sim2Sim,
    *,
    input_device: str,
    joystick_index: int,
    deadzone: float,
    vx_scale: float | None,
    wz_scale: float | None,
) -> None:
    import mujoco.viewer

    commander = _build_commander(
        sim,
        input_device=input_device,
        joystick_index=joystick_index,
        deadzone=deadzone,
        vx_scale=vx_scale,
        wz_scale=wz_scale,
    )

    def _on_key(keycode: int) -> None:
        commander.handle(keycode)

    def _apply_control_requests() -> None:
        if commander.reset_requested:
            sim.reset()
            commander.reset_requested = False
        if commander.next_terrain_requested:
            sim.reset(advance_terrain=True)
            commander.next_terrain_requested = False

    print("[sim2sim] Opening MuJoCo viewer.")
    if input_device == "keyboard":
        print(
            "[sim2sim] Controls: W/S forward, A/D yaw, Space zero, R reset, T next terrain, "
            "P pause, N single-step, F follow-camera, 1/2/3 presets."
        )
    else:
        print(
            "[sim2sim] PS2 controls: left stick Y choose forward/back direction, "
            "right stick X choose yaw direction, D-pad up/down adjust |vx|, "
            "D-pad left/right adjust |wz|, CROSS zero, START reset, SELECT pause, CIRCLE next terrain, "
            "TRIANGLE follow-camera, SQUARE single-step."
        )
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=_on_key) as viewer:
            viewer.cam.distance = float(sim.cfg["terrain"]["follow_camera_distance"])
            viewer.cam.elevation = -20.0
            viewer.cam.azimuth = 90.0
            last = time.perf_counter()
            while viewer.is_running():
                now = time.perf_counter()
                dt = now - last
                last = now
                commander.poll()
                _apply_control_requests()
                if commander.follow_camera:
                    sim.set_viewer_camera(viewer)
                if commander.paused and not commander.single_step:
                    viewer.sync()
                    time.sleep(0.01)
                    continue
                sim.update_command(commander.command)
                sim.step()
                commander.single_step = False
                viewer.sync()
                sim.maybe_print_status()
                sleep_time = max(float(sim.model.opt.timestep) - dt, 0.0)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        commander.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Real68 MuJoCo sim2sim player.")
    parser.add_argument("--bundle-dir", type=Path, default=None, help="Prepared bundle directory.")
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "WZ"),
        help="Initial velocity command override.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without launching the viewer.")
    parser.add_argument("--steps", type=int, default=4000, help="Headless simulation steps.")
    parser.add_argument("--random-yaw", action="store_true", help="Use randomized yaw on reset.")
    parser.add_argument(
        "--auto-reset", action="store_true", help="Automatically reset on fall/contact failure."
    )
    parser.add_argument(
        "--terrain-cell", type=int, default=None, help="Initial terrain cell index."
    )
    parser.add_argument(
        "--input-device",
        choices=("keyboard", "ps2"),
        default="keyboard",
        help="Interactive control source.",
    )
    parser.add_argument(
        "--joystick-index", type=int, default=0, help="pygame joystick index for PS2 mode."
    )
    parser.add_argument(
        "--deadzone", type=float, default=0.12, help="Joystick deadzone for PS2 mode."
    )
    parser.add_argument(
        "--vx-scale", type=float, default=None, help="Max forward speed mapped from stick."
    )
    parser.add_argument(
        "--wz-scale", type=float, default=None, help="Max yaw rate mapped from stick."
    )
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve() if args.bundle_dir is not None else _default_bundle_dir()
    command = np.asarray(args.command, dtype=np.float64) if args.command is not None else None
    sim = Real68Sim2Sim(
        bundle_dir,
        command_override=command,
        random_yaw=bool(args.random_yaw),
        auto_reset=bool(args.auto_reset),
    )
    if args.terrain_cell is not None:
        sim.set_terrain_cell(int(args.terrain_cell))
        sim.reset()
    if args.headless:
        _headless_run(sim, steps=int(args.steps))
        return
    _interactive_run(
        sim,
        input_device=str(args.input_device),
        joystick_index=int(args.joystick_index),
        deadzone=float(args.deadzone),
        vx_scale=args.vx_scale,
        wz_scale=args.wz_scale,
    )


if __name__ == "__main__":
    main()
