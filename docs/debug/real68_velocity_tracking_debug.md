# Real68 速度跟踪 debug 过程记录

## 当前结论

之前两条核心判断是错的：

1. 不是 `termination` 太严。
2. 也不是 `forward_progress` clip 才是主因。

真正的问题是 `command curriculum` 的评估口径有偏差：

- 统计用的是“当前还活着的 env”的 batch 均值。
- env 自动 reset 以后，失败样本会被立刻替换掉，坏样本不再参与后续均值。
- 小命令样本更容易存活，也更容易把 `mean_abs_vx / mean_abs_cmd_x` 拉高。
- 结果是课程会被“假进步”驱动，日志看起来在解锁更大命令，但真实 hardest command segment 的速度跟踪并没有达标。

这正好解释了此前现象：

- `command_curriculum/progress` 会涨。
- `speed_ratio` 看起来能过线。
- 但真正拿到较大 `vx` 命令时，机器人还是几乎不走，甚至主要靠 yaw 动作或短命存活样本支撑日志。

## 错误结论为什么会出现

### `max_tilt_cos` 的理解之前反了

终止条件是看 `gravity[:, 2] <= max_tilt_cos`。

- `max_tilt_cos` 越大，要求越接近直立，终止越严格。
- `max_tilt_cos` 越小，允许倾斜越多，终止越宽松。

所以把 `0.45 -> 0.55` 解释成“放宽 termination”本身就是错误的。

### batch 均值被 survivorship bias 污染

之前记录的这些量：

- `metrics/mean_abs_vx`
- `metrics/mean_abs_cmd_x`
- `metrics/vx_error`
- `command_curriculum/speed_ratio`

本质上都偏向“当前还没死掉的 env”。这会系统性高估真实训练水平。

### command distribution 也在掩盖 hardest case

课程早期大量采样的是小 `vx` 命令。即使策略只会追很小的速度，也能把总体均值做得很好看，但这不代表它已经具备扩展到更大速度区间的能力。

## 现在的修正

### 在 autoreset 前补 env hook

文件：[src/unilab/base/np_env.py](/home/gx/Desktop/UniLab/src/unilab/base/np_env.py)

- 新增 `_before_autoreset(done)`
- 在 `_reset_done_envs()` 前调用

作用：

- 让 env 能在样本被 reset 之前，把这一段 command segment 的统计先记下来。
- 失败样本不会再被 silent 丢掉。

### Real68 command curriculum 改成按 command segment 统计

文件：[src/unilab/envs/locomotion/real68/balance.py](/home/gx/Desktop/UniLab/src/unilab/envs/locomotion/real68/balance.py)

现在的做法：

- 每个 env 维护当前 command segment 的累计统计。
- 在两种时刻 finalize：
  - command resample 前
  - autoreset 前
- 统计项包括：
  - `mean_abs_vx`
  - `vx_error`
  - `wz_error`

### 课程推进不再看全局均值，而是看 hardest sufficiently-sampled bucket

同文件：[src/unilab/envs/locomotion/real68/balance.py](/home/gx/Desktop/UniLab/src/unilab/envs/locomotion/real68/balance.py)

现在按命令幅值分桶：

- `vx` 分桶
- `yaw` 分桶

推进规则变成：

- 只看“样本数足够”的最难桶
- 该桶同时满足：
  - `speed_ratio` 达标
  - `vx_error` / `wz_error` 达标
- 才允许课程继续推进

这避免了“小命令样本把整体均值冲高，从而误解锁大命令”的问题。

### 增加能直接反映课程真实状态的日志

新增日志：

- `command_curriculum/eval_vx_error`
- `command_curriculum/eval_wz_error`
- `command_curriculum/eval_bucket_high_vx`
- `command_curriculum/eval_bucket_high_wz`
- `command_curriculum/eval_count_vx`
- `command_curriculum/eval_count_wz`
- `command_curriculum/segments_recorded`

rough 任务也复用了同一套统计逻辑，避免 flat / rough 两边行为再分叉。

## 已回退的错误方向改动

以下方向已经回退，不再作为当前方案的一部分：

- `forward_progress_unclipped`
- wheel actuator `ctrlrange ±12 -> ±40`
- 基于错误解释去放宽/修改 veltrack flat 的 termination 参数

当前 `real68_balance_veltrack_flat` 已恢复到：

- `wheel_velocity_scale: 16.0`
- `min_base_height: 0.18`
- `max_tilt_cos: 0.45`

## 修正后的短训练结论

在修正课程统计之后，短训练表现变成：

- `command_curriculum/progress` 不再虚假增长
- `command_curriculum/speed_ratio` 维持在大约 `0.13 ~ 0.14`
- `command_curriculum/eval_vx_error` 维持在大约 `0.69 ~ 0.70`
- `command_curriculum/eval_count_vx` 会持续上升，说明 hardest bucket 已经有足够样本

这说明：

- 之前的“课程在进步”是错觉
- 现在日志终于开始诚实反映真实训练状态
- 真正剩下的问题已经收敛成一句话：

**Real68 目前的策略和奖励/控制组合，确实还不会做高质量速度跟踪；之前只是课程统计把问题遮住了。**

## 后续新增定位

在继续排查后，又确认了另一个关键事实：

- 当前 velocity task 的起步阶段并不是“单一速度跟踪”。
- 它同时在做两件事：
  - 从 `0.1 ~ 0.8 m/s` 的宽前向命令带里均匀采样
  - 同时随机采样高度目标

这会让 PPO 在最开始就面对一组互相差异很大的动态平衡问题，容易退回到“低速安全折中”。

后续短烟测显示：

- 固定单一前向命令时，`speed_ratio` 会明显高于宽区间混合命令
- 固定高度目标时，任务也更接近真正的“速度跟踪”
- 把初始课程收窄到 `0.35 ~ 0.45 m/s`，表现明显好于原来的 `0.1 ~ 0.8 m/s`

因此，当前保留的 owner 配置方向是：

- velocity task 固定高度目标
- 初始命令课程先从窄前向速度带开始
- 学会后再自动扩展到更大的前向速度和反向速度

## 后续该看什么

后续如果继续调 Real68 velocity tracking，应该盯下面这些量，而不是只看 batch 均值：

- `command_curriculum/eval_vx_error`
- `command_curriculum/eval_count_vx`
- `command_curriculum/eval_bucket_high_vx`
- `command_curriculum/speed_ratio`
- `metrics/mean_abs_vx`
- `metrics/mean_abs_cmd_x`

如果 `eval_count_vx` 已经足够大，而 hardest bucket 的 `speed_ratio` 仍长期过低，就说明问题已经从“统计口径”转成了真正的控制/奖励/可达性问题，需要单独继续排。
