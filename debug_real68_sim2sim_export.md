# Real68 新模型导出到 sim2sim_real68_mujoco

本文说明如何把 `logs/rsl_rl_ppo/Real68BalanceFlat` 下新训练完成的 PPO 模型导出，并用于 `sim2sim_real68_mujoco`。

## 1. 确认训练 run

先找到你要导出的 run 目录。当前目录结构通常是：

```text
logs/rsl_rl_ppo/Real68BalanceFlat/<run_name>/
```

例如：

```text
logs/rsl_rl_ppo/Real68BalanceFlat/2026-07-05_18-44-41_mujoco
```

run 目录里至少应有：

```text
model_*.pt
run_config.json
```

如果之前已经做过一次 `eval`，目录里还会有：

```text
policy.onnx
policy.pt
play_video.mp4
```

## 2. 导出 policy.onnx 和 policy.pt

UniLab 这里不是单独写一个导出脚本给 Real68 PPO，而是通过 `eval` 触发导出。

在仓库根目录执行：

```bash
uv run eval --algo ppo --task real68_balance_veltrack_flat --sim mujoco --load-run 2026-07-05_18-44-41_mujoco
```

说明：

- `--load-run` 填你要导出的 run 目录名，不要填完整绝对路径。
- 这条命令会进入 `scripts/train_rsl_rl.py` 的 play/eval 流程，并在对应 run 目录下导出：

```text
policy.onnx
policy.pt
```

导出完成后，检查：

```text
logs/rsl_rl_ppo/Real68BalanceFlat/2026-07-05_18-44-41_mujoco/policy.onnx
logs/rsl_rl_ppo/Real68BalanceFlat/2026-07-05_18-44-41_mujoco/policy.pt
logs/rsl_rl_ppo/Real68BalanceFlat/2026-07-05_18-44-41_mujoco/run_config.json
```

如果只想确认最新 run，可以先看：

```bash
find logs/rsl_rl_ppo/Real68BalanceFlat -maxdepth 2 -type f \( -name 'model_*.pt' -o -name 'policy.onnx' \) | tail -n 40
```

## 3. 打包到 sim2sim_real68_mujoco

`sim2sim_real68_mujoco` 运行时不直接读取训练目录，而是读取 bundle。

执行：

```bash
uv run python -m sim2sim_real68_mujoco.prepare_bundle \
  --run-dir logs/rsl_rl_ppo/Real68BalanceFlat/2026-07-05_18-44-41_mujoco
```

默认会生成：

```text
sim2sim_real68_mujoco/bundles/2026-07-05_18-44-41_mujoco/
```

这个 bundle 里会包含：

```text
scene.xml
terrain_origins.npy
policy.onnx
run_config.json
sim2sim_config.json
```

如果你想指定输出目录：

```bash
uv run python -m sim2sim_real68_mujoco.prepare_bundle \
  --run-dir logs/rsl_rl_ppo/Real68BalanceFlat/2026-07-05_18-44-41_mujoco \
  --output-dir sim2sim_real68_mujoco/bundles/my_real68_bundle
```

## 4. 在 sim2sim_real68_mujoco 里使用新模型

### 4.1 直接使用指定 bundle 启动

```bash
uv run python -m sim2sim_real68_mujoco.main \
  --bundle-dir sim2sim_real68_mujoco/bundles/2026-07-05_18-44-41_mujoco
```

### 4.2 使用手柄控制

```bash
uv run python -m sim2sim_real68_mujoco.main \
  --bundle-dir sim2sim_real68_mujoco/bundles/2026-07-05_18-44-41_mujoco \
  --input-device ps2
```

### 4.3 无头验证

```bash
uv run python -m sim2sim_real68_mujoco.main \
  --bundle-dir sim2sim_real68_mujoco/bundles/2026-07-05_18-44-41_mujoco \
  --headless \
  --steps 4000 \
  --command 0.5 0.0 0.0
```

如果要测试转向：

```bash
uv run python -m sim2sim_real68_mujoco.main \
  --bundle-dir sim2sim_real68_mujoco/bundles/2026-07-05_18-44-41_mujoco \
  --headless \
  --steps 4000 \
  --command 0.4 0.0 0.8
```

## 5. 常见问题

### 5.1 `prepare_bundle` 报缺少 `policy.onnx`

说明你还没有先跑 `eval` 导出 ONNX。

先执行：

```bash
uv run eval --algo ppo --task real68_balance_veltrack_flat --sim mujoco --load-run <run_name>
```

再执行 `prepare_bundle`。

### 5.2 `sim2sim` 跑出来效果不对

优先检查三件事：

1. `bundle-dir` 是否对应最新 run。
2. `run_config.json` 是否来自当前训练配置。
3. 训练时的 `cmd_yaw` 是否已经真正纳入课程分布，而不是仍然只训练直行。

### 5.3 想切换成最新 bundle

如果不传 `--bundle-dir`，`sim2sim_real68_mujoco.main` 会默认取：

```text
sim2sim_real68_mujoco/bundles/
```

下按名字排序后的最后一个目录。

为了避免误用旧模型，建议显式传 `--bundle-dir`。

## 6. 推荐流程

每次新训练结束后，按这个顺序执行：

```bash
uv run eval --algo ppo --task real68_balance_veltrack_flat --sim mujoco --load-run <run_name>
uv run python -m sim2sim_real68_mujoco.prepare_bundle --run-dir logs/rsl_rl_ppo/Real68BalanceFlat/<run_name>
uv run python -m sim2sim_real68_mujoco.main --bundle-dir sim2sim_real68_mujoco/bundles/<run_name> --input-device ps2
```

这样可以保证：

- run 目录里有最新 `policy.onnx`
- `sim2sim_real68_mujoco` 用的是对应 bundle
- 不会混用旧 checkpoint 和新 runtime
