# HIM-PPO

HIM-PPO 有自己的配置组和脚本。入口是 `scripts/train_him_ppo.py`，基础配置是
`conf/ppo_him/config.yaml`，已提交的 task owner 是
`conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`。

## 当前入口

旧的 Go2 机械臂 HIM-PPO 路径由 `scripts/train_him_ppo.py` 实现，仍是脚本级入口。
迁移后的 WheelBipe 紧凑 HIM owner 是独立的 flat-task 路由，可通过
`uv run train --algo him_ppo --task wheelbipe_v14_flat --sim <backend>` 使用，
由 `scripts/train_custom_ppo.py` 实现。其历史与 checkpoint contract 见
{doc}`../4-tasks/5-wheelbipe_v14`。

## Owner 细节

Go2 机械臂 owner 从基础配置中填充所需的历史维度：

- `algo.num_one_step_obs=76`
- `algo.num_actor_history=5`
- `algo.num_critic_history=1`
- `training.task_name=Go2ArmManipLoco`

一旦有可用的检查点，回放将使用相同的 HIM-PPO 实现入口。请将面向用户的 PPO 示例保
持在受支持的顶层 CLI 形式上；仅在调试该专用技术栈时才使用 HIM-PPO 脚本路径。

两条路径都不是默认 PPO 路径：旧路径用于 Go2 机械臂 manip-loco owner，custom
路径仅用于 WheelBipe flat owner。
