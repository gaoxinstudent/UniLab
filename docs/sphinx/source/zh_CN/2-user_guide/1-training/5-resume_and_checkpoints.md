# 续训与检查点

检查点选择由算法层面的字段控制。请使用 `algo.load_run`，而不是
`training.load_run`。

## 续训

使用 run id，或用 `-1` 表示相关日志目录中最新的一次运行：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.load_run=-1 \
  training.no_play=true

uv run train --algo sac --task g1_walk_flat --sim mujoco \
  algo.load_run=2026-03-16_01-35-12_mujoco \
  training.no_play=true
```

## 回放检查点

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` 会将 `--load-run` 映射到底层的检查点选择器，并设置回放模式：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

`--load-run` 接受 `-1`、run 目录名，或 run 目录/checkpoint 的绝对路径。带路径分隔符的
相对路径仍会被拒绝，避免其含义受到 Hydra 工作目录影响。仍可直接传
`algo.load_run=...`，但两种写法不能同时使用。

## 随机种子

训练种子的解析在 `src/unilab/training/seed.py` 中实现。算法配置目前携带
`algo.seed`，并且在实验跟踪启用时，该辅助逻辑会记录种子元数据。
