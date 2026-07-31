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
    print(f"[sim2sim] final status: {sim.status_line()}")
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
    ps2_vx_axis: int,
    ps2_wz_axis: int,
    key_timeout: float,
):
    if input_device == "keyboard":
        return KeyboardCommander(
            vx_max=float(max(abs(sim.command_limits[0, 0]), abs(sim.command_limits[1, 0]))),
            wz_max=float(max(abs(sim.command_limits[0, 2]), abs(sim.command_limits[1, 2]))),
            key_timeout=key_timeout,
        )

    default_vx_scale = float(max(abs(sim.command[0]), 0.8))
    default_wz_scale = float(max(abs(sim.command[2]), 0.4))
    return GamepadCommander(
        joystick_index=joystick_index,
        deadzone=deadzone,
        vx_scale=default_vx_scale if vx_scale is None else float(vx_scale),
        wz_scale=default_wz_scale if wz_scale is None else float(wz_scale),
        vx_max=float(max(abs(sim.command_limits[0, 0]), abs(sim.command_limits[1, 0]))),
        wz_max=float(max(abs(sim.command_limits[0, 2]), abs(sim.command_limits[1, 2]))),
        vx_axis=ps2_vx_axis,
        wz_axis=ps2_wz_axis,
    )


def _interactive_run(
    sim: Real68Sim2Sim,
    *,
    input_device: str,
    joystick_index: int,
    deadzone: float,
    vx_scale: float | None,
    wz_scale: float | None,
    ps2_vx_axis: int,
    ps2_wz_axis: int,
    key_timeout: float,
    render_fps: float,
) -> None:
    import mujoco.viewer

    commander = _build_commander(
        sim,
        input_device=input_device,
        joystick_index=joystick_index,
        deadzone=deadzone,
        vx_scale=vx_scale,
        wz_scale=wz_scale,
        ps2_vx_axis=ps2_vx_axis,
        ps2_wz_axis=ps2_wz_axis,
        key_timeout=key_timeout,
    )

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
            "[sim2sim] Terminal controls: W/S forward, A/D yaw; Space zero; "
            "R reset, T next terrain, P pause, N single-step, F camera, Q quit."
        )
        print(
            "[sim2sim] Speed: +/- both axes, [/ ] forward step, ,/. yaw step; "
            "1/2/3 timed presets. Commands expire when key repeat stops."
        )
    else:
        print(
            "[sim2sim] PS2 controls: left stick Y choose forward/back direction, "
            "right stick X choose yaw direction, D-pad up/down adjust |vx|, "
            "D-pad left/right adjust |wz|, CROSS zero, START reset, SELECT pause, CIRCLE next terrain, "
            "TRIANGLE follow-camera, SQUARE single-step. Commands are clipped to the bundle limits."
        )
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.distance = float(sim.cfg["terrain"]["follow_camera_distance"])
            viewer.cam.elevation = -20.0
            viewer.cam.azimuth = 90.0
            render_period = 1.0 / max(float(render_fps), 1.0)
            physics_steps_per_frame = max(
                1, int(round(render_period / float(sim.model.opt.timestep)))
            )
            print(
                f"[sim2sim] render={float(render_fps):.1f} Hz, "
                f"physics_steps_per_frame={physics_steps_per_frame}"
            )
            while viewer.is_running():
                frame_start = time.perf_counter()
                commander.poll()
                _apply_control_requests()
                if commander.quit_requested:
                    break
                if commander.follow_camera:
                    sim.set_viewer_camera(viewer)
                if commander.paused and not commander.single_step:
                    viewer.sync()
                    time.sleep(0.01)
                    continue
                sim.update_command(commander.command)
                steps = 1 if commander.single_step else physics_steps_per_frame
                for _ in range(steps):
                    sim.step()
                commander.single_step = False
                viewer.sync()
                sim.maybe_print_status()
                sleep_time = max(render_period - (time.perf_counter() - frame_start), 0.0)
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
    parser.add_argument(
        "--render-fps", type=float, default=60.0, help="Interactive viewer refresh rate."
    )
    parser.add_argument(
        "--key-timeout",
        type=float,
        default=0.35,
        help="Seconds before a terminal movement command expires without key repeat.",
    )
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
        "--vx-scale",
        type=float,
        default=None,
        help="Initial forward-speed magnitude and D-pad step-size reference for PS2 mode.",
    )
    parser.add_argument(
        "--wz-scale",
        type=float,
        default=None,
        help="Initial yaw-rate magnitude and D-pad step-size reference for PS2 mode.",
    )
    parser.add_argument("--ps2-vx-axis", type=int, default=1, help="PS2 forward-stick axis index.")
    parser.add_argument("--ps2-wz-axis", type=int, default=2, help="PS2 yaw-stick axis index.")
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
        ps2_vx_axis=int(args.ps2_vx_axis),
        ps2_wz_axis=int(args.ps2_wz_axis),
        key_timeout=float(args.key_timeout),
        render_fps=float(args.render_fps),
    )


if __name__ == "__main__":
    main()
