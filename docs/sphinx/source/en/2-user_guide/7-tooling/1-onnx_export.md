# ONNX Export

ONNX export is normally tied to playback in the training scripts. The PPO and
HIM-PPO scripts set `EXPORT_POLICY=True` when run as scripts, then export during
`training.play_only=true` playback. APPO and off-policy playback paths also
export `policy.onnx` and verify it with ONNX Runtime in their script code. The
WheelBipe custom history runner is the explicit exception: it exports its
history-stacked graph (and a `policy.onnx.json` contract sidecar) at the end of
a headless training run, because its dedicated sim-to-sim helper owns history
initialization. See {doc}`../../2-user_guide/4-tasks/5-wheelbipe_v14` for the
custom graph dimensions and algorithm checks.

## Examples

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

Use the same `--algo`, `--task`, and `--sim` values that produced the
checkpoint. For deployment context, see
{doc}`../../3-deployment/1-sim_to_real/5-onnx_runtime`.
