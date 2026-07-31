# Real68 MuJoCo Sim2Sim

This directory is a standalone MuJoCo player for the unified `Real68Balance`
Sim2Real policy. Runtime depends on `mujoco`, `numpy`,
and `onnxruntime`. PS2 gamepad control also requires `pygame`. It does not
import UniLab env or training code when running the policy.

The player reconstructs the training actor observation history and direct
mixed position/velocity controller, including action latency, command limits,
and the differential-drive command feasibility limit. It is not a controller
in its own right: an exported policy that did not learn to balance, stand, or
track a command will behave the same way here.

Only use a bundle generated from the exact training run being evaluated. The
player accepts the `real68_balance_v2` contract with five 28-value actor frames
(`140` ONNX inputs) and six direct mixed-control actions. Legacy 29/32-value
MLP bundles remain readable when they do not declare the v2 schema. The player
checks the scene timestep and rejects other observation/action contracts.

## Prepare a bundle

This step copies the policy artifacts from a training run and materializes the
matching standalone MuJoCo scene:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -m sim2sim_real68_mujoco.prepare_bundle \
  --run-dir logs/rsl_rl_ppo/Real68Balance/<run>
```

Default output:

```text
sim2sim_real68_mujoco/bundles/2026-06-28_00-35-49_mujoco/
```

## Interactive run

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -m sim2sim_real68_mujoco.main
```

Controls:

- Keyboard input is read from the terminal, so MuJoCo viewer hotkeys are not
  intercepted.
- `W/S`: forward/backward command while the terminal key repeats
- `A/D`: left/right yaw command while the terminal key repeats
- `Space`: zero the command immediately
- `+/-`: increase/decrease both speed steps
- `[/]`: decrease/increase forward speed step
- `,/.`: decrease/increase yaw speed step
- `1/2/3`: timed forward command presets
- `R`: reset, `T`: next terrain, `P`: pause, `N`: single-step, `F`: camera, `Q`: quit

Terminals do not expose physical key-up events. Movement commands therefore
expire after `--key-timeout` seconds without a repeated key press (default
`0.35 s`), which makes releasing a key return to zero while still supporting
normal terminal key repeat for held keys.

The viewer batches physics steps before each render. The default refresh is
60 Hz; change it with `--render-fps` if needed.

## Interactive run with PS2 gamepad

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -m sim2sim_real68_mujoco.main \
  --input-device ps2
```

Controls:

- left stick `Y`: choose forward/backward direction only
- right stick `X`: choose yaw direction only
- D-pad `up/down`: increase/decrease `|vx|`
- D-pad `left/right`: increase/decrease `|wz|`
- `CROSS`: zero current command
- `START`: reset
- `SELECT`: pause/resume
- `CIRCLE`: switch to the next terrain cell and reset
- `TRIANGLE`: toggle follow-camera
- `SQUARE`: single-step when paused

PS2 control uses direction + magnitude split:

- stick deflection only selects the sign of `vx` / `wz`
- D-pad adjusts the command magnitudes
- releasing the stick returns that axis command to zero
- initial magnitudes are `|vx|=0.8` and `|wz|=0.4` by default; override them
  with `--vx-scale` and `--wz-scale`
- default stick axes are `vx=1` and `wz=2`; use `--ps2-vx-axis` or
  `--ps2-wz-axis` when the connection log reports a different controller mapping

Default behavior is manual-reset only. The runtime will not automatically jump
between terrain cells unless you press `R` / `T` or pass `--auto-reset`.

## Headless validation

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -m sim2sim_real68_mujoco.main \
  --headless \
  --steps 4000 \
  --terrain-cell 2 \
  --command 0.5 0.0 0.0
```

This prints periodic status and a final distance / mean-velocity summary.

## Motion-control acceptance

Validate an exported policy on flat terrain before testing terrain cells or
large combined commands. Run each command in a fresh process so reset state
does not carry across checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -m sim2sim_real68_mujoco.main \
  --headless --steps 4000 --command 0.0 0.0 0.0
UV_CACHE_DIR=/tmp/uv-cache uv run -m sim2sim_real68_mujoco.main \
  --headless --steps 4000 --command 0.2 0.0 0.0
UV_CACHE_DIR=/tmp/uv-cache uv run -m sim2sim_real68_mujoco.main \
  --headless --steps 4000 --command -0.2 0.0 0.0
UV_CACHE_DIR=/tmp/uv-cache uv run -m sim2sim_real68_mujoco.main \
  --headless --steps 4000 --command 0.0 0.0 0.2
```

Inspect the final `mean_vx`, distance, and `failed` status. A zero command
must not be judged by distance alone: use the status stream to check that the
robot remains upright without non-wheel contact. For nonzero forward commands,
the sign of `mean_vx` must match the command before increasing the magnitude.

Keyboard and gamepad commands are clipped to the run's `env.commands.vel_limit`
or, for a completed command curriculum, its final velocity envelope; they are
also clipped to the wheel-speed feasibility diamond used during training. The status
line displays the effective command after clipping. Do not use an arbitrary
high command to diagnose policy quality; it is outside the policy's trained
distribution even when the player accepts it.

The bundled historical artifact is a legacy run and is not the production
lineage. Production training uses the unified `Real68Balance` task, where the
same actor is trained with flat and rough terrain. Terrain scans are privileged
critic observations only, so the deployed actor contract remains unchanged.
Use the low-speed checks above to separate a policy-quality limitation from a
sim2sim contract failure.

All resets start upright. Runtime falls still activate the recovery state machine.
