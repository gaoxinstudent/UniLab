# UniLab Go1 命令速查

根据 UniLab 在线文档与仓库内文档整理：

- 在线入口：<https://unilabsim.github.io/UniLab-doc/zh_CN/1-getting_started/0-index.html>
- 本地文档：
  `docs/sphinx/source/zh_CN/1-getting_started/3-evaluation_and_playback.md`
  `docs/sphinx/source/zh_CN/2-user_guide/1-training/1-cli_reference.md`
  `docs/sphinx/source/zh_CN/2-user_guide/1-training/5-resume_and_checkpoints.md`
  `docs/sphinx/source/zh_CN/3-deployment/1-sim_to_real/5-onnx_runtime.md`

本文已在 2026-06-25 用仓库内 `go1_joystick_rough` 的 MuJoCo PPO 路径做过 smoke。

## 任务与前提

- 当前仓库里可直接用的 Go1 owner 是 `go1_joystick_rough/mujoco`。
- owner YAML 在 `conf/ppo/task/go1_joystick_rough/mujoco.yaml`。
- 统一入口优先用 `uv run train` 和 `uv run eval`。
- MuJoCo 无头录视频在这台机器上建议强制 EGL：

```bash
env -u DISPLAY MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
```

## 训练

常规训练：

```bash
uv run train --algo ppo --task go1_joystick_rough --sim mujoco
```

快速 smoke：

```bash
uv run train --algo ppo --task go1_joystick_rough --sim mujoco \
  training.log_root=logs/agent_smoke/rsl_rl_ppo \
  algo.num_envs=8 \
  algo.num_steps_per_env=4 \
  algo.max_iterations=1 \
  training.no_play=true
```

产物位置：

```text
logs/rsl_rl_ppo/Go1JoystickRough/<timestamp>_mujoco/
```

或当你手动指定时：

```text
logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough/<timestamp>_mujoco/
```

## 评估与视频导出

常规评估：

```bash
uv run eval --algo ppo --task go1_joystick_rough --sim mujoco --load-run -1
```

在当前这台无头 Linux 机器上，更稳的录像命令：

```bash
env -u DISPLAY MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
uv run eval --algo ppo --task go1_joystick_rough --sim mujoco \
  --load-run -1 \
  --render-mode record
```

更短的 smoke 版本：

```bash
env -u DISPLAY MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
uv run eval --algo ppo --task go1_joystick_rough --sim mujoco \
  --load-run -1 \
  --render-mode record \
  training.log_root=logs/agent_smoke/rsl_rl_ppo \
  training.play_steps=20
```

说明：

- MuJoCo 的 `auto` 最终也会走录像导出。
- `--render-mode none` 会跳过 playback，因此也不会生成视频或触发 PPO 的导出路径。
- 当前 PPO MuJoCo 回放产物写在 run 目录根下，文件名是 `play_video.mp4`。

## 导出 ONNX

PPO 不需要单独的导出脚本，直接复用 `eval` 路径：

```bash
env -u DISPLAY MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
uv run eval --algo ppo --task go1_joystick_rough --sim mujoco \
  --load-run -1 \
  --render-mode record
```

成功后会在 run 目录里看到：

```text
policy.onnx
policy.pt
play_video.mp4
```

## 可视化

训练曲线：

```bash
uv run tensorboard --logdir logs/rsl_rl_ppo/Go1JoystickRough
```

如果你用了单独日志根：

```bash
uv run tensorboard --logdir logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough
```

无界面检查事件文件是否可读：

```bash
uv run tensorboard --inspect --logdir logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough
```

交互式 MuJoCo viewer 入口：

```bash
uv run scripts/play_interactive.py --algo ppo --task go1_joystick_rough --sim mujoco \
  algo.load_run=<run_id> \
  interactive.action_mode=policy
```

说明：

- `play_interactive.py` 的 CLI 入口已核对。
- 本次 smoke 没有长时间占用 GUI 窗口，只验证了参数入口和常规 `eval` 录像链路。

## 续训

PPO 当前推荐使用显式 `run_id`：

```bash
uv run train --algo ppo --task go1_joystick_rough --sim mujoco \
  algo.load_run=<run_id> \
  training.no_play=true
```
例如
```bash
uv run train --algo ppo --task real68_balance_flat --sim mujoco \
    algo.load_run=2026-06-25_20-17-33_mujoco \
    algo.num_envs=4096 \
    algo.max_iterations=200 \
    training.no_play=true
```

快速 smoke：

```bash
uv run train --algo ppo --task go1_joystick_rough --sim mujoco \
  training.log_root=logs/agent_smoke/rsl_rl_ppo \
  algo.load_run=2026-06-25_20-38-54_mujoco \
  algo.num_envs=8 \
  algo.num_steps_per_env=4 \
  algo.max_iterations=1 \
  training.no_play=true
```

查看可用 `run_id`：

```bash
find logs/rsl_rl_ppo/Go1JoystickRough -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
```

或当你使用独立日志根时：

```bash
find logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
```

重要说明：

- 仓库文档 `docs/sphinx/source/zh_CN/2-user_guide/1-training/5-resume_and_checkpoints.md`
  写了 `algo.load_run=-1` 可用于续训。
- 但当前 `scripts/train_rsl_rl.py` 的训练分支只会在 `algo.load_run != "-1"` 时调用
  `runner.load(...)`。
- 所以对 PPO 而言，当前稳定做法是显式传 `run_id`，不要依赖 `algo.load_run=-1`。

## 本次实际验证

已跑通：

```bash
uv run train --algo ppo --task go1_joystick_rough --sim mujoco \
  training.log_root=logs/agent_smoke/rsl_rl_ppo \
  algo.num_envs=8 \
  algo.num_steps_per_env=4 \
  algo.max_iterations=1 \
  training.no_play=true
```

```bash
env -u DISPLAY MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
uv run eval --algo ppo --task go1_joystick_rough --sim mujoco \
  --load-run -1 \
  --render-mode record \
  training.log_root=logs/agent_smoke/rsl_rl_ppo \
  training.play_steps=20
```

```bash
uv run tensorboard --inspect --logdir logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough
```

```bash
uv run train --algo ppo --task go1_joystick_rough --sim mujoco \
  training.log_root=logs/agent_smoke/rsl_rl_ppo \
  algo.load_run=2026-06-25_20-38-54_mujoco \
  algo.num_envs=8 \
  algo.num_steps_per_env=4 \
  algo.max_iterations=1 \
  training.no_play=true
```

验证得到的关键产物：

- `logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough/2026-06-25_20-38-54_mujoco/model_0.pt`
- `logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough/2026-06-25_20-38-54_mujoco/policy.onnx`
- `logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough/2026-06-25_20-38-54_mujoco/policy.pt`
- `logs/agent_smoke/rsl_rl_ppo/Go1JoystickRough/2026-06-25_20-38-54_mujoco/play_video.mp4`

