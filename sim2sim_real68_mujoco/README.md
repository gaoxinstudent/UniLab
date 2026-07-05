# Real68 MuJoCo Sim2Sim

This directory is a standalone Real68 rough-terrain sim2sim player. Runtime
depends on `mujoco`, `numpy`, and `onnxruntime`. PS2 gamepad control also
requires `pygame`. It does not import UniLab env or training code when running
the policy.

## Prepare a bundle

This step copies the policy artifacts from a training run and materializes a
standalone rough-terrain MuJoCo scene:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m sim2sim_real68_mujoco.prepare_bundle \
  --run-dir logs/rsl_rl_ppo/Real68BalanceRough/2026-06-28_00-35-49_mujoco
```

Default output:

```text
sim2sim_real68_mujoco/bundles/2026-06-28_00-35-49_mujoco/
```

## Interactive run

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m sim2sim_real68_mujoco.main
```

Controls:

- `W/S`: increase/decrease forward velocity command
- `A/D`: increase/decrease yaw-rate command
- `Space`: zero the command
- `R`: reset
- `T`: switch to the next terrain cell and reset
- `P`: pause/resume
- `N`: single-step when paused
- `F`: toggle follow-camera
- `1/2/3`: forward command presets

## Interactive run with PS2 gamepad

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m sim2sim_real68_mujoco.main \
  --input-device ps2
```

Controls:

- left stick `Y`: forward/backward command
- right stick `X`: yaw-rate command
- `CROSS`: zero current command
- `START`: reset
- `SELECT`: pause/resume
- `CIRCLE`: switch to the next terrain cell and reset
- `TRIANGLE`: toggle follow-camera
- `SQUARE`: single-step when paused

PS2 control is absolute, not incremental: stick deflection maps directly to
`vx` / `wz`, and releasing the stick returns the command to zero.

Default behavior is manual-reset only. The runtime will not automatically jump
between terrain cells unless you press `R` / `T` or pass `--auto-reset`.

## Headless validation

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m sim2sim_real68_mujoco.main \
  --headless \
  --steps 4000 \
  --terrain-cell 2 \
  --command 0.5 0.0 0.0
```

This prints periodic status and a final distance / mean-velocity summary.
