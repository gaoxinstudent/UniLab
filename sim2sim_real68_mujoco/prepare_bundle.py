from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.xml import materialize_mujoco_hfield_attached_scene
from unilab.envs.locomotion.real68.balance import Real68Sensor
from unilab.envs.locomotion.real68.base import (
    ACTIVE_JOINT_POS_SENSORS,
    ACTIVE_JOINT_VEL_SENSORS,
    CALF_INDICES,
    DEFAULT_ACTIVE_ANGLES,
    HIP_INDICES,
    HOME_BASE_HEIGHT,
    NONWHEEL_CONTACT_SENSORS,
    POSTURE_INDICES,
    WHEEL_CONTACT_SENSORS,
    WHEEL_INDICES,
)
from unilab.envs.locomotion.real68.base import ControlConfig as Real68ControlDefaults
from unilab.envs.locomotion.real68.rough import Real68RoughTerrainCfg
from unilab.training.run import get_latest_run

_DEFAULT_LOG_ROOT = Path("logs/rsl_rl_ppo/Real68BalanceRough")


def _latest_run_dir() -> Path:
    run_dir = get_latest_run(_DEFAULT_LOG_ROOT)
    if run_dir is None:
        raise FileNotFoundError(f"No run directories found under {_DEFAULT_LOG_ROOT}")
    return run_dir.resolve()


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _terrain_cfg_from_run(run_cfg: dict[str, Any]) -> Real68RoughTerrainCfg:
    cfg = Real68RoughTerrainCfg()
    generator_cfg = (
        run_cfg.get("config", {})
        .get("env", {})
        .get("scene", {})
        .get("terrain", {})
        .get("generator", {})
    )
    if "seed" in generator_cfg:
        cfg.seed = generator_cfg["seed"]
    if "curriculum" in generator_cfg:
        cfg.curriculum = bool(generator_cfg["curriculum"])
    if "size" in generator_cfg:
        cfg.size = tuple(float(x) for x in generator_cfg["size"])
    if "num_rows" in generator_cfg:
        cfg.num_rows = int(generator_cfg["num_rows"])
    if "num_cols" in generator_cfg:
        cfg.num_cols = int(generator_cfg["num_cols"])
    if "border_width" in generator_cfg:
        cfg.border_width = float(generator_cfg["border_width"])
    return cfg


def _sim2sim_config(run_dir: Path, output_dir: Path, run_cfg: dict[str, Any]) -> dict[str, Any]:
    env_cfg = run_cfg["config"]["env"]
    reward_cfg = run_cfg["config"]["reward"]
    control_cfg = env_cfg["control_config"]
    control_defaults = Real68ControlDefaults()
    domain_rand = env_cfg["domain_rand"]
    termination = env_cfg["termination_config"]
    commands = env_cfg["commands"]
    sensor_cfg = Real68Sensor()
    return {
        "scene_file": "scene.xml",
        "policy_file": "policy.onnx",
        "run_config_file": "run_config.json",
        "terrain_origins_file": "terrain_origins.npy",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "sim_dt": 0.001,
        "ctrl_dt": 0.02,
        "steps_per_control": int(round(0.02 / 0.001)),
        "home_base_height": HOME_BASE_HEIGHT,
        "base_height_target": float(reward_cfg["base_height_target"]),
        "default_active_angles": DEFAULT_ACTIVE_ANGLES.tolist(),
        "active_joint_pos_sensors": list(ACTIVE_JOINT_POS_SENSORS),
        "active_joint_vel_sensors": list(ACTIVE_JOINT_VEL_SENSORS),
        "wheel_contact_sensors": list(WHEEL_CONTACT_SENSORS),
        "nonwheel_contact_sensors": list(NONWHEEL_CONTACT_SENSORS),
        "sensor_names": {
            "local_linvel": sensor_cfg.local_linvel,
            "gyro": sensor_cfg.gyro,
            "gravity": sensor_cfg.gravity,
            "accel": sensor_cfg.accel,
            "quat": sensor_cfg.quat,
        },
        "indices": {
            "hip": HIP_INDICES.tolist(),
            "wheel": WHEEL_INDICES.tolist(),
            "calf": CALF_INDICES.tolist(),
            "posture": POSTURE_INDICES.tolist(),
        },
        "control_config": {
            "clip_actions": float(control_cfg.get("clip_actions", control_defaults.clip_actions)),
            "hip_velocity_scale": float(
                control_cfg.get("hip_velocity_scale", control_defaults.hip_velocity_scale)
            ),
            "wheel_velocity_scale": float(
                control_cfg.get("wheel_velocity_scale", control_defaults.wheel_velocity_scale)
            ),
            "calf_action_scale": float(
                control_cfg.get("calf_action_scale", control_defaults.calf_action_scale)
            ),
            "hip_kd": float(control_cfg.get("hip_kd", control_defaults.hip_kd)),
            "wheel_kd": float(control_cfg.get("wheel_kd", control_defaults.wheel_kd)),
            "calf_kp": float(control_cfg.get("calf_kp", control_defaults.calf_kp)),
            "calf_kd": float(control_cfg.get("calf_kd", control_defaults.calf_kd)),
        },
        "command_limits": commands["vel_limit"],
        "reset_config": {
            "xy_jitter": float(abs(domain_rand.get("reset_pos_xy_range", [-0.5, 0.5])[1])),
            "z_offset_range": domain_rand.get("reset_height_offset_range", [0.0, 0.08]),
            "roll_range": domain_rand.get("reset_roll_range", [-0.1, 0.1]),
            "pitch_range": domain_rand.get("reset_pitch_range", [-0.1, 0.1]),
            "yaw_range": domain_rand.get("reset_yaw_range", [-3.141592653589793, 3.141592653589793]),
            "reset_qvel_limit": float(domain_rand.get("reset_qvel_limit", 0.15)),
            "spawn_height_margin": 0.05,
        },
        "termination_config": {
            "min_up_proj": float(termination["min_up_proj"]),
            "min_base_height": float(termination["min_base_height"]),
            "nonwheel_contact_threshold": float(termination.get("nonwheel_contact_threshold", 0.5)),
            "nonwheel_contact_max_steps": int(termination.get("nonwheel_contact_max_steps", 8)),
        },
        "terrain": {
            "geom_name": "floor",
            "hfield_name": "terrain_hfield",
            "cell_size": 8.0,
            "follow_camera_distance": 3.0,
        },
    }


def _copy_artifacts(run_dir: Path, output_dir: Path) -> None:
    for name in ("policy.onnx", "run_config.json"):
        source = run_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Missing required artifact: {source}")
        shutil.copy2(source, output_dir / name)


def prepare_bundle(run_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = _load_run_config(run_dir)
    terrain_cfg = _terrain_cfg_from_run(run_cfg)
    robot_model = ASSETS_ROOT_PATH / "robots" / "real68" / "real68.xml"
    fragment = ASSETS_ROOT_PATH / "robots" / "real68" / "locomotion_task.xml"
    _, terrain_origins = materialize_mujoco_hfield_attached_scene(
        model_file=str(robot_model),
        terrain_cfg=terrain_cfg,
        output_dir=output_dir,
        fragment_files=[str(fragment)],
        hfield_name="terrain_hfield",
        geom_name="floor",
        return_surface_sampler=False,
    )
    np.save(output_dir / "terrain_origins.npy", terrain_origins)
    _copy_artifacts(run_dir, output_dir)
    sim2sim_cfg = _sim2sim_config(run_dir, output_dir, run_cfg)
    (output_dir / "sim2sim_config.json").write_text(
        json.dumps(sim2sim_cfg, indent=2),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a standalone Real68 sim2sim bundle.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Training run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Bundle output directory. Defaults to sim2sim_real68_mujoco/bundles/<run-name>.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir is not None else _latest_run_dir()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (Path(__file__).resolve().parent / "bundles" / run_dir.name).resolve()
    )
    bundle_dir = prepare_bundle(run_dir, output_dir)
    print(f"Prepared sim2sim bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
