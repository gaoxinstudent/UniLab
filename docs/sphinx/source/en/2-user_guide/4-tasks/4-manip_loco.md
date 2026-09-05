# Manip-Loco

`go2_arm_manip_loco` combines Go2 locomotion with the Airbot arm. The registered
env is `Go2ArmManipLoco`.

## Owner Configs

- PPO owner: `conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`
- HIM-PPO owner: `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`
- Scene entry: `src/unilab/assets/robots/go2_arm/scene_flat.xml`

## PPO

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

## HIM-PPO

The Go2 arm HIM-PPO owner is `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`.
`src/unilab/cli.py` does not expose this legacy Go2 owner as a top-level
`uv run train --algo ...` route. The separate flat-only WheelBipe custom
`him_ppo` route is documented in {doc}`5-wheelbipe_v14`.

The current committed owner path is MuJoCo. Keep backend selection in
`--task go2_arm_manip_loco --sim mujoco`, and do not override
`training.sim_backend` alone.
