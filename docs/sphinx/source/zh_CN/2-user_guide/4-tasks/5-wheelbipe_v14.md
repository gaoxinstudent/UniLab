# WheelBipe V14

UniLab 已纳入从 SCUTRobotLab 公共训练仓库和 ROS 2 部署仓库迁移的
WheelBipe V14 normal 模式运动控制 owner。面向策略的观测、动作和执行器映射
位于环境 owner 层；MuJoCo 或 Motrix 通过统一的 task/backend CLI 参数选择。

## 可用 owner

| 任务 | owner YAML 中的后端 | 环境注册名 |
| --- | --- | --- |
| `wheelbipe_v14_flat` | `mujoco`、`motrix` | `WheelbipeV14Flat` |
| `wheelbipe_v14_rough` | `mujoco`、`motrix` | `WheelbipeV14Rough` |

使用 `--sim` 选择后端；不要把 `training.sim_backend` 当作独立的后端开关。

### 上游 exact task id

上游 README 发布了下面 15 个 Gymnasium id。UniLab 在环境 registry 中保留这些完整字符串，
训练 CLI 接受全部 15 个 id。使用 exact id 时，应把完整字符串直接传给 `--task`（例如
`--task Robotics-Wheelbipe-V14-Flat-v0`）；下表“CLI 路由”列展示翻译后选用的 canonical
owner。每个路由都是带固定 reward/config graph 的可执行 source-contract owner；这不表示
MuJoCo 或 Motrix 会复现 Isaac/PhysX 动力学或数值轨迹。`*-Play-*` 路由会设置
`training.play_only=true`（通常应通过
`uv run eval` 使用）。

| 上游 id | UniLab CLI 路由 | 结果 |
| --- | --- | --- |
| `Robotics-Wheelbipe-V14-Flat-v0` | `--algo ppo --task wheelbipe_v14_flat` | 兼容 owner |
| `Robotics-Wheelbipe-V14-Flat-v1` | `--algo ppo --task wheelbipe_v14_flat` | state-machine owner |
| `Robotics-Wheelbipe-V14-Flat-v2` | `--algo ppo --task wheelbipe_v14_flat` | gimbal owner |
| `Robotics-Wheelbipe-V14-Flat-Play-v0` | `--algo ppo --task wheelbipe_v14_flat` | 兼容 owner，仅 play |
| `Robotics-Wheelbipe-V14-Flat-Play-v2` | `--algo ppo --task wheelbipe_v14_flat` | gimbal owner，仅 play |
| `Robotics-Wheelbipe-V14-Rough-v0` | `--algo ppo --task wheelbipe_v14_rough` | rough/gimbal owner |
| `Robotics-Wheelbipe-V14-Rough-v1` | `--algo ppo --task wheelbipe_v14_rough` | rough/state-machine owner |
| `Robotics-Wheelbipe-V14-Rough-Play-v0` | `--algo ppo --task wheelbipe_v14_rough` | rough/gimbal owner，仅 play |
| `Robotics-Wheelbipe-V14-Rough-Play-v1` | `--algo ppo --task wheelbipe_v14_rough` | rough/state-machine owner，仅 play |
| `Robotics-Wheelbipe-V14-Flat-DreamWaQ-v0` | `--algo dreamwaq --task wheelbipe_v14_flat` | 兼容 owner |
| `Robotics-Wheelbipe-V14-Flat-DreamWaQ-Play-v0` | `--algo dreamwaq --task wheelbipe_v14_flat` | 兼容 owner，仅 play |
| `Robotics-Wheelbipe-V14-Flat-HIM-v0` | `--algo him_ppo --task wheelbipe_v14_flat` | 兼容 owner |
| `Robotics-Wheelbipe-V14-Flat-HIM-Play-v0` | `--algo him_ppo --task wheelbipe_v14_flat` | 兼容 owner，仅 play |
| `Robotics-Wheelbipe-V14-Flat-NP3OBarlow-v0` | `--algo np3o --task wheelbipe_v14_flat` | 兼容 owner |
| `Robotics-Wheelbipe-V14-Flat-NP3OBarlow-Play-v0` | `--algo np3o --task wheelbipe_v14_flat` | 兼容 owner，仅 play |

只有上表 15 个 id 会被识别为上游 exact alias。未知字符串会走普通 task/config 校验路径，
在没有 owner 时失败；CLI 不会静默回退到 `wheelbipe_v14_flat` 或 `wheelbipe_v14_rough`。
v1/v2 命名 owner 会在初始化阶段物化 state-machine 传感器或两个
gimbal 执行器，同时保持六维 action contract。公开的状态转移、sensor contract 与
gimbal actuator contract 已在环境 owner 层实现并在两个后端测试；backend 动力学仍由
各 simulator 决定，不宣称与 source Isaac runtime 逐位等价。

基于历史的策略 owner 也通过同一个 CLI 暴露，但目前仅限 flat 任务。下面列出内部
owner 名称，便于把 checkpoint 与正确的 runner 对齐：

| 算法 | 公共 task | owner 环境 | 历史帧数 | cost 通道 |
| --- | --- | --- | ---: | ---: |
| `him_ppo` | `wheelbipe_v14_flat` | `WheelbipeV14FlatHIM` | 5 | 0 |
| `dreamwaq` | `wheelbipe_v14_flat` | `WheelbipeV14FlatDreamWaQ` | 5 | 0 |
| `np3o` | `wheelbipe_v14_flat` | `WheelbipeV14FlatNP3OBarlow` | 10 | 5 |

这些 custom 路由已支持 `mujoco` 和 `motrix`，没有 `mjwarp` owner；它们使用专用的
`scripts/train_custom_ppo.py` runner，而不是普通 PPO runner。
它们的 owner YAML 当前选择显式的 `source_v14_physics` timing/delay profile
（5 ms 物理步、每个 20 ms 控制步四个子步），并开启历史策略的 delay buffer。
timing/delay 是迁移 owner contract 的一部分；观测、reward、DR、command、termination
与状态机语义也有对应 owner 与测试。profile 名称本身只选择 timing/delay，不代表上游
动力学或资产等价。
DreamWaQ AdaBoot 现在作为显式可选项提供。在 DreamWaQ owner
上设置 `algo.policy.adaboot_mode=reward_cv`（或 `hybrid`）即可使用已完成
episode return 的滚动窗口和上游风格的
`p_boot = 1 - tanh(scale * CV + offset)` 系数。owner 会校验窗口/边界并在
update 中输出 `adaboot_*` metrics，默认值仍为 `off`。这里只实现 reward-CV
集成，不声称完整 DreamWaQ 或 AdaBoot 论文 parity；uncertainty/hybrid 表示
路径以及所有数值仍受上面的 compact owner 维度约束。

## 公开 source 证据矩阵

| 公开 source feature | UniLab owner/config | 自动化或实跑证据 | 明确边界 |
| --- | --- | --- | --- |
| 15 个公开 task id；flat/rough、状态机和 gimbal 变体 | registry alias 与 `conf/ppo/task/wheelbipe_v14_{flat,rough}/` | registry/config 测试及 `tests/envs/locomotion/wheelbipe_v14/test_{contract,gimbal_state_machine,state_machine_stack,owner_yaml_contract}.py` | Isaac Sim/PhysX 不是 UniLab runtime；不宣称 backend 动力学逐位等价。 |
| normal PPO ABI：35 维 actor、78 维 critic、6 维 action | `WheelbipeV14Flat`/`WheelbipeV14Rough` 与双后端 owner YAML | `test_training_semantics.py`、source checkpoint 测试及下方 artifact 实跑表 | strict-load 与有限步 rollout 只证明 ABI/可执行性，不证明收敛或 reward 曲线等价。 |
| source reward、termination、command curriculum、delay、DR、contact 状态机与 torque mapping | 环境 owner 与 task YAML（普通 PPO 默认使用 `source_v14_physics`；`local_physics` 仅作显式诊断 profile） | MuJoCo/Motrix WheelBipe 环境 contract/timing/training-semantics 测试 | Motrix 通过声明的兼容 profile 关闭 6 个闭环 equality；不宣称跨 simulator 物理等价。 |
| 上游 `model_state_dict`、TorchScript `policy.pt` 与 ONNX policy | 标准 PPO eval adapter 与 `scripts/sim2sim_wheelbipe.py` | malformed/shape 严格测试及下方实际 `model_8000.pt`/`policy.pt`/ONNX 执行 | TorchScript 必须可信；ONNX 走专用 helper；上游 checkpoint 不含 UniLab `contract_snapshot`。 |
| HIM-PPO、DreamWaQ、NP3O + Barlow 历史策略 | `conf/custom_ppo/task/wheelbipe_v14_flat_*` 与 custom runner | source-key checkpoint/optimizer 回归、pinned DreamWaQ 类数值对齐、source-Barlow 双输入 artifact 测试，以及明确的 NP3O 312D 兼容修复 | 仅 flat MuJoCo/Motrix；上游 live env 的 351D 流与 312D model contract 自相矛盾；公开 checkout 未包含 custom checkpoint 可供实际加载；无 `mjwarp`；AdaBoot `reward_cv` 不等于完整论文实现。 |
| 键盘命令与 velocity/reward trace | `src/unilab/visualization/wheelbipe_{keyboard,trace}.py` | `tests/visualization/test_wheelbipe_tools.py` 与 interactive CLI 路由测试 | viewer key-down 命令会锁存；跳跃是否接受仍由状态机决定；trace 是诊断而非 benchmark。 |
| 无 ROS 的 ROS controller/state/wire contract | `WheelbipeRos2Controller`、deployment YAML 与 Python packet/gate helper | `tests/training/test_wheelbipe_ros2.py` 及 MuJoCo/Motrix 数值 helper 实跑 | 进程内 cadence emulation 不等于 DDS graph、realtime controller、serial transport 或硬件安全证明。 |
| native ROS 2/controller_manager/pluginlib/serial/teleop source | `deployment/ros2/wheelbipe_v14_native/manifest.yaml` 与打包的 colcon source | exact-digest bundle verify、workspace materialize、runtime probe、reconnect/stale/teleop fail-closed 测试，以及 2026-09-05 RoboStack Humble 环境下外部 workspace 的 build + headless launch（INIT→IDLE→PREPARE→RL、joint_states 演化） | 未打开串口、未连接机器人硬件；不宣称 realtime 或硬件安全。 |

## 训练

默认 PPO owner 已写入迁移后的策略维度和控制参数，并默认采用原 V14 训练的
`source_v14_physics` profile（5 ms MuJoCo physics、20 ms policy control、physics-step
观测/动作延迟）。冻结 sim2sim/ROS 是部署适配器，不能反向改写训练 owner；需要隔离
部署时序影响时才显式选择 `local_physics`（1 ms、无注入延迟）进行诊断。
flat 任务的 MuJoCo、Motrix owner 都以 4096 个并行环境和 20000 个 iteration 作为长训练默认值；资源受限的机器可用
`algo.num_envs=<数量>` 显式缩小 batch。无界面训练时显式关闭回放：

```bash
uv run train --algo ppo --task wheelbipe_v14_flat --sim mujoco training.no_play=true
uv run train --algo ppo --task wheelbipe_v14_flat --sim motrix training.no_play=true
uv run train --algo ppo --task wheelbipe_v14_rough --sim mujoco training.no_play=true
```

### Rough 训练契约与 warm-start

canonical `wheelbipe_v14_rough` 的训练契约与发布的上游
`rough_rotation_stair` 2026-07-23 run 对齐：actor/critic 均为
`[256, 128, 64]`、`value_loss_coef=2.0`、seed 66，观测/动作/延迟/DR 与命令
special-mode 配置保持该 run 的 pinned 语义。该 run 本身从 flat `model_8000.pt`
warm-start。复现 flat→rough 训练时，将 `algo.load_run` 指向
`pretrained/26_infantry/flat_and_rotation/2026-07-19_09-14-50/model_8000.pt`。
下面命令复现源 flat→rough 初始化路径。若改为继续训练已发布 rough `model_1000.pt`，应单独验收：

训练模型使用源 USD 的质量、质心和完整惯量：base 为 15.96301746 kg，
质心 X 为 +0.00582119 m，主动腿关节 armature 为 0.015795。
被动连杆的惯量包含主轴旋转，不能只复制惯量对角项。训练气弹簧使用
400–600 N / 70 mm、offset 0.06076 m、预载随机量 ±50 N，
IdealPD 阻尼 50 N·s/m。源配置的 `spring_settings.damping=false` 仅关闭额外阻尼。
ROS2 仓库的 650–450 N / 75 mm、500 N·s/m 是独立部署模型，不能覆盖训练 owner。

canonical rough 地形对应发布模型 `2026-07-23_16-23-21/params/env.yaml`：
10 行、14 列，包含上行/下行中等台阶（单级 0.025–0.04 m、宽 0.1 m），
并使用该快照的地形高度命令。较早的 `2026-07-23_10-19-59` 和当前上游
Rough-v0 配置只有 10 列，不能把同一天的不同训练配置混为一个基线。
exact upstream id 仍保留其固定的源码预设。源数值回归见
`tests/envs/locomotion/wheelbipe_v14/test_released_migration.py`。
高度奖励使用随航向旋转的 3×3 地面采样均值（20 mm 范围、10 mm 间距），
不再只取机身正下方一点；critic 仍保留世界高度。地面采样通过后端提供的
heightfield surface contract 完成，不能将它宣称为 PhysX raycaster 的数值等价实现。
MuJoCo owner 显式开启 `post_step_forward_sensor=true`，使 IMU、body tracking
和 qpos/qvel 都对应同一物理步末，再施加源观测延迟；否则会额外引入一个物理步的传感器延迟。
MuJoCo 适配层只刷新运动学传感器，并保留实际积分步的力/加速度传感器，
防止运行时的零控制 `forward` 把负载下接触力重算成无控制力的结果。
源奖励和 critic 使用机身质心速度（包含每个环境的 COM 随机偏移）。轮子 critic
速度先将世界坐标下的轮子/机身质心速度相减，再旋转到机身坐标系并令 Y 分量为零；
不能直接使用会扣除旋转参考系运输项的 backend 相对速度。惯性偏移仅在初始化时读取并缓存。

```bash
uv run train --algo ppo --task wheelbipe_v14_rough --sim mujoco \
  algo.num_envs=4096 algo.max_iterations=2000 training.no_play=true \
  algo.load_run=/path/to/wheeled-legged_RL/pretrained/26_infantry/flat_and_rotation/\
2026-07-19_09-14-50/model_8000.pt
```

warm-start 只恢复 actor/critic 权重（含 std），optimizer 与 iteration 计数保持
新建。canonical rough 的 `track_lin_vel_xy_square` 和 `track_ang_vel_z_square`
均为 -0.1，与 16:23:21 的完整奖励表一致；较早的 10:19:59 run 和 exact source
Rough-v0/v1 使用 -1.0。策略导出的 35D→6D 接口可以由 ROS2 控制器直接加载，但接口一致
不等于越障通过。
registry 直接构造的 sim2sim owner 同样保留原始动作，不再截断到 ±1；旧默认值
会把轮速目标限制到 ±10 rad/s（约 0.6 m/s），与训练路径不一致。物理参数修复
不会改变 35D 输入维度，旧 checkpoint 虽能加载，仍须重新评估。

过去报告的 forwardness ≈ 0.94 只表示朝前运动，机器人在障碍前打滑也可能得到
高分。越障验证必须记录障碍几何、速度/高度命令、两轮是否越过出口、是否绕行、
跌倒和重置次数；不能使用训练平均 reward 或 forwardness 代替通过率。
2026-09-16 的早期 140 mm 台阶探测仅复用了 ROS2 场景几何及简化 PD，
没有包含原桥接器的二阶电机响应、力矩变化率限制和高速降额，不能用其通过率
判断原生 ROS2 的越障性能。当前验收先恢复源 rough；ROS2 适配与节点时序另行验证。

### 加载上游 vanilla-PPO checkpoint

上游训练仓库的 vanilla-PPO 权重保存在 `model_*.pt` 的
`model_state_dict` 中；UniLab 原生 RSL-RL checkpoint 使用的是
`actor_state_dict`。WheelBipe PPO eval 路由会在创建环境前识别上游 schema，随后
逐层校验目标 actor/critic 的键和值 shape，再加载用于推理。为了避免把训练期状态
误当作部署状态，optimizer state 和上游 iteration 计数不会恢复；若要继续训练，
应先把上游 checkpoint 转换成 UniLab 原生 run。

通过公共 `--load-run` flag 传入 checkpoint 或 run 目录的绝对路径。仍可直接使用
`algo.load_run=...` Hydra override，但不能同时传入两种形式：

```bash
uv run eval --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode none \
  --load-run /path/to/wheeled-legged_RL/pretrained/.../model_8000.pt \
  training.play_steps=20
uv run eval --algo ppo --task wheelbipe_v14_flat --sim motrix \
  --render-mode none \
  --load-run /path/to/wheeled-legged_RL/pretrained/.../model_8000.pt \
  training.play_steps=20
```

常见的上游网络结构 `[256, 128, 64]` 已是两个 WheelBipe owner YAML 的默认值。
如果 source run 使用 `[128, 64, 32]`，必须显式选择一致的维度：

```bash
uv run eval --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode none --load-run /path/to/model_8000.pt \
  training.play_steps=20 \
  algo.policy.actor_hidden_dims=[128,64,32] \
  algo.policy.critic_hidden_dims=[128,64,32]
```

网络结构、观测、动作或 distribution shape 不一致时会 fail-closed，并给出诊断，
不会部分加载策略。PPO + MuJoCo 使用 `--render-mode interactive` 时，公共 `eval`
会路由到专用交互 viewer；同一严格 adapter 可处理原生 `actor_state_dict`、上游普通
PPO `model_state_dict` 与可信的 35D→6D TorchScript artifact。custom 算法不会被
隐式伪装成 PPO viewer 支持。

可用 Hydra override 开启迁移后的 WheelBipe 键盘 contract：

```bash
uv run eval --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode interactive --load-run /path/to/model_8000.pt \
  interactive.keyboard=true
```

MuJoCo viewer 只提供 key-down 事件：`W/S` 设置并锁存前进速度，`A/D` 设置并锁存
yaw rate，`Z/X` 安全地单次调节高度，`L` 或 Enter 复位。`Q` 只写入一次性的
`info["jump_takeoff_request"]`；是否接受跳跃仍完全由 WheelBipe 状态机决定。

上游发布包还包含只用于推理的 TorchScript `policy.pt` archive。标准 PPO
`eval` 路由可以直接接收该文件（使用
`--load-run /path/to/policy.pt`），并严格校验 `float` 35 维 actor 输入与 6 维
动作输出，然后在不构造第二个网络的情况下执行数值回放。TorchScript archive 不含
optimizer 或 resume 状态，而且其中的序列化代码可执行，因此只能传入可信 artifact。
直接使用 `policy.pt` 时不需要 hidden-dimension override；目标 owner 仍必须提供正常的
35 维 observation/6 维 action contract。

### Rough 地形播放

rough PPO owner 提供了仅用于播放的展示配置：保留训练时的
`RM_ROUGH_TERRAINS_CFG` running-terrain 场景和地形命令 profile，让前几个并行
环境按生成地形列轮询分配，并固定在中等难度、沿地形正方向前进。只在播放时关闭
空中重置、周期推力、外力、特殊命令 bucket 和随机初始 yaw，避免录制一开始就在
空中或原地跳跃；step-up/airborne 状态机仍保持开启，并继续由地形/接触传感器驱动。

使用 `--task wheelbipe_v14_rough` 播放时，如果 checkpoint 的 sidecar 是
`WheelbipeV14RoughV1` 等 exact owner 写出的，PPO 会在创建环境前采用 sidecar 中的
owner，避免同样的网络维度掩盖了 legacy compatibility owner 与 V1 状态机契约不一致。
复制 checkpoint 时建议同时保留旁边的 `run_config.json`。

例如，跟踪第一个生成地形列录制 10 秒（将
`training.cam_tracking_env_idx` 改为 `0`、`4`、`8` 或 `12` 可选择其它列）：

```bash
MUJOCO_GL=egl uv run eval --algo ppo --task Robotics-Wheelbipe-V14-Rough-v1 \
  --sim mujoco --render-mode record training.play_steps=500 \
  training.play_env_num=16 training.cam_tracking=true \
  training.cam_tracking_extra_envs=0 training.cam_tracking_env_idx=0 \
  --load-run /path/to/model_5000.pt
```

与源发布包一致的断崖越障展示使用 exact `Robotics-Wheelbipe-V14-Rough-Play-v0`
路由：10 行 `cliff_inv_stair_slope_short_for_rm_play` 地形、每 5 s 重置、地形
profile 固定 2.5 m/s 前向命令。该地形的 bowl 中心低于地面、四周是 0.03 m 台阶
和 +0.3–0.4 m 断崖；历史 body-frame forwardness ≈ 0.94 仅衡量运动方向，
不证明完成越障。以下命令用于可视化检查：

```bash
MUJOCO_GL=egl uv run eval --algo ppo --task Robotics-Wheelbipe-V14-Rough-Play-v0 \
  --sim mujoco --render-mode record training.play_steps=1500 \
  training.play_env_num=16 training.cam_tracking=true \
  training.cam_tracking_extra_envs=0 training.cam_tracking_env_idx=0 \
  --load-run /path/to/rough_checkpoint.pt
```

训练/回放导出的 `policy.onnx`（单输入 `obs` `float32[1,35]`、单输出 `actions`
`float32[1,6]`，索引 28–34 恒为 `[1,0,0,0,0,0,0]`）与
`wheelbipe_ros2_sim2sim` 的 normal-only 策略合同一致：把该 ONNX 放入该仓库
`src/controllers/template_ros2_controller/policy/parallel/`，并用
`WHEELBIPE_RL_MODEL_PATH` 指向它即可由同一 ROS 2 controller 推理，无需改图或
预处理参数（command/gyro/gravity/joint 的 scale-clamp 参数为部署合同，跨模型共享）。

### Artifact 实跑证据记录

下表中的 artifact 路由已在两个后端执行。这里仅记录 loader/execution 证据；source
观测、动作和主动 DR contract 修正后，旧 owner 语义下的两步 mean reward 已删除，
因为继续展示这些数值会形成陈旧证据，而不是 benchmark。训练/config provenance
不同的各行不能用来比较 reward 或收敛效果。

| Artifact 与路由 | MuJoCo | Motrix | 额外检查 |
| --- | --- | --- | --- |
| 固定上游 `model_8000.pt`，PPO `eval` | 严格加载并完成有限步 rollout | 严格加载并完成有限步 rollout | 接受 `model_state_dict` schema，不会部分加载 |
| 同目录上游 `policy.pt`，PPO `eval` | 可信 TorchScript rollout | 可信 TorchScript rollout | 检查输入上与 `model_8000.pt` actor 输出完全相同 |
| 同目录上游 `policy.onnx`，专用 helper | ONNX rollout | ONNX rollout | 已检查 graph shape 和 actor 数值对齐 |
| UniLab `wheelbipe_v14_flat_long_4096/model_2700.pt`，PPO `eval` | 原生 checkpoint rollout/export | 原生 checkpoint rollout/export | 该模型在最终 source-contract 修正前训练；仅作为带 provenance 标签的 loader smoke 保留，不是迁移后性能结果 |
| UniLab rough `rr256fixsq` run `model_1999.pt`（4096 envs × 2000 iters，warm-start 源 `model_2500.pt`），PPO `eval` | 原生 checkpoint rollout、断崖播放录制与 `policy.onnx` 导出 | —（同一 checkpoint 的可加载性由 sim2sim 契约 audit 覆盖） | 断崖播放 body-frame forwardness ≈ 0.94、平均前进速度 ≈ 1.0 m/s，该方向指标不能证明跨越障碍（尚无原生 ROS2 越障验收）；导出的 ONNX 为 `obs[1,35] → actions[1,6]`，与 `wheelbipe_ros2_sim2sim` 合同一致 |

公共 flag 的绝对 checkpoint 路径和绝对 run 目录两种形式均已验证；exact upstream
`*-Play-v0` alias 走同一路由。例如：

```bash
uv run eval --algo ppo --task Robotics-Wheelbipe-V14-Flat-Play-v0 \
  --sim motrix --render-mode none \
  --load-run /home/gx/UniLab/third-party/wheeled-legged_RL/pretrained/26_infantry/flat_and_rotation/2026-07-19_09-14-50/model_8000.pt \
  training.play_steps=2
uv run eval --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode none \
  --load-run /home/gx/UniLab/logs/wheelbipe_v14_flat_long_4096 \
  training.play_steps=2
```

`record` 使用 exact-checksum artifact copy 在两个后端生成了有效 MP4 container。MuJoCo
`interactive` 在严格加载后到达 viewer，并因本机无 display 给出明确失败；Motrix
interactive 的 CLI route 已由命令/路由测试覆盖，但本次验证没有实际打开 Motrix 窗口。

### Custom 历史策略训练与评估

用公共 CLI 选择算法进行短时无界面 smoke。即使关闭 playback，custom runner 也会在
最终 checkpoint 旁导出带历史输入的 `policy.onnx`：

```bash
uv run train --algo him_ppo --task wheelbipe_v14_flat --sim mujoco \
  algo.num_envs=4 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=true training.log_root=/tmp/wheelbipe-runs
```

canonical `wheelbipe_v14_flat` 路由在长时间运行时可选择对应的 source profile。
六个 exact custom train/Play ID 会自动组合其匹配 profile，并继续拒绝额外
`--profile`；因此 exact ID 与 canonical + 下列 profile 得到相同的 algorithm 配置。
profile 作为 Hydra group 组合，`mujoco`/`motrix` owner 不变；没有 CUDA 的主机或
确定性 smoke 可以显式指定 CPU：

```bash
uv run train --algo him_ppo --task wheelbipe_v14_flat --sim mujoco \
  --profile source_him_long training.device=cpu training.no_play=true
uv run train --algo dreamwaq --task wheelbipe_v14_flat --sim mujoco \
  --profile source_dreamwaq_long training.device=cpu training.no_play=true
uv run train --algo np3o --task wheelbipe_v14_flat --sim mujoco \
  --profile source_np3o_barlow_long training.device=cpu training.no_play=true
```

Custom eval 会加载 `model_*.pt`，先校验 checkpoint 的算法、历史长度、观测/动作维度，
以及 NP3O 的 5 个 cost 元数据，然后执行数值 rollout。如果 run 目录存在 UniLab
`run_config.json`，custom owner 会在构造 evaluator 前自动采用其中仅影响架构的元数据
（HIM estimator 宽度、DreamWaQ CENet 宽度或 NP3O source-Barlow 字段）；显式的
`--profile`/架构 override 仍会严格校验，不匹配时 fail closed。设置
`training.auto_load_checkpoint_config=false` 可要求 owner 显式匹配。需要 backend renderer
时可选择 `--render-mode record` 或 `interactive`：

```bash
uv run eval --algo him_ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode none training.play_steps=20 \
  algo.load_run=/tmp/wheelbipe-runs/WheelbipeV14FlatHIM/<run-directory>
```

要继续 custom 训练，可在 `uv run train` 命令中把同一个 run 目录或 checkpoint 路径传给
`algo.load_run`。custom runner 会在收集新 rollout 前恢复 policy、optimizer/adaptation
状态、iteration 计数器以及 NP3O 通道 schedule；默认的 `algo.load_run=-1` 表示新建运行。

#### Source-enabled custom reward curriculum

只有 exact HIM training owner 启用 pinned `CurriculumCfgV14`；HIM Play owner 会关闭
该 curriculum。每个 completed-reset batch 都会统计 weighted `track_height_exp` episode
sum 除以 `max_episode_length_s` 后的 batch mean。为对齐 source manager 的计数方式，
`num_steps_per_env=24` 会把配置的 64-sample window 放大为 `64 × 24` 次 compute call，
每阶段 500-episode minimum 放大为 `500 × 24` 次 call。两个 threshold transition 是：

| 阶段 | Height reward 权重（`exp` / `tight`） | base world-frame +Z assist | 晋级条件 |
| --- | ---: | ---: | --- |
| 0 | `1.0 / 1.0` | 160 N | 满足阶段最小计数后，completed window mean 至少为 `0.4` |
| 1 | `0.8 / 0.6` | 80 N | 满足阶段最小计数后，completed window mean 至少为 `0.4` |
| 恢复默认值 | `0.0 / 1.0` | 0 N | curriculum 终态 |

assist 与 interval random wrench 不会相加。pinned source 把二者写入同一个
external-wrench buffer：构造、reset 或全局 stage advance 会写入当前 160/80/0 N assist，
其中只有 +Z force，torque 清零；5--10 s random-wrench event 到期时会覆盖对应环境的
latch；该环境下次 reset 或下次全局 stage advance 再写回 assist。UniLab 保留这项逐环境
写入顺序。MuJoCo/Motrix 的
external-force API 只作用一个本地 step，因此每步通过公共
`SimBackend.apply_body_force`/`apply_body_torque` contract 重发 latched value 只是明确的
后端转换，不是额外相加，也不会让 env 访问 backend 私有状态。

这里仍有一项明确的坐标系转换边界：source 每次写 assist 时，会依当时的
body quaternion 把 world +Z 转为 body-local 向量，并把该本地向量持久留在
Isaac wrench buffer。UniLab 则锁存 sampled world-frame +Z，通过公共 upcoming-step
world-frame backend contract 重发。事件数值和覆盖顺序已保留；两次写入之间
由 body 旋转造成的力方向演化不宣称物理等价。

只有 exact NP3O training owner 启用 pinned linear height-to-velocity reward gate；
NP3O Play 会关闭它。令绝对高度误差为 `e`：`e <= 0.05` m 时 gate 为 `1`，
`e >= 0.10` m 时为 `0`，中间按 `(0.10 - e) / 0.05` 线性下降。gate 会在 owner
reward weight 生效前乘到 linear-velocity 与 yaw-rate tracking term（包括 tight/square
变体），避免 policy 通过牺牲 commanded height 获取完整 velocity-tracking reward。
exact owner validation 会拒绝修改 train/Play 开关差异或两个 threshold。

HIM 和 DreamWaQ 加载边界也接受 pinned source runner 产生的完整
`model_state_dict`（其 MLP key 含 `.model.` 容器层），但不接受本地/source key
混用、缺失或额外的 graph。DreamWaQ 的 main Adam 与 `vae_optimizer_state_dict`
会按上游明确的参数顺序重排，不根据 tensor shape 猜测对应关系。回归测试对
pinned DreamWaQ 类的 CENet、action 和 value 输出得到 `0.0` 最大绝对误差。
公开训练 checkout 里的 110 个 `model_*.pt` 都是 normal PPO graph，因此这些
custom 结论是 source-shaped 严格回归与 pinned 类数值对齐，而不是已发布 custom
artifact 的实际加载证据。UniLab checkpoint 保留了上游字段别名，但不会反向生成
`.model.` key graph 或 source DreamWaQ Adam 顺序，所以不声称 UniLab → source runner
双向 resume。

source profile 的 history reset 也属于配置与 artifact contract：reset 先清零全部 deque
槽位，再把当前帧放在最后，所以首个 actor 输入是 `[0, ..., 0, current]`；done 后、eval
与 custom sim2sim 使用相同的 oldest-to-newest 顺序。bounded/旧图没有该 sidecar 字段时
仍保留原有 repeat-current 行为，不会被隐式改写。

紧凑 custom actor 的单帧观测为 28 维、critic 为 32 维；runner 为 HIM/DreamWaQ 堆叠 5 帧，
紧凑 NP3O 堆叠 10 帧。因此这些紧凑导出输入分别为 `[1,140]` 或 `[1,280]`，动作是
`[1,6]`，不能直接接入 normal 35 维 ROS 部署图。

pinned NP3O checkout 在 active graph 上有一个不能被 checkpoint 证据消解的维度矛盾：
live env 拼出的 `on_constraint` 是 351D（policy 28 + 实际 `priv_latent` 43 + history 280），
而 model/runner constructor 声明并归一化 312D（28 + 4 + 280）。公开 checkout 没有 custom
checkpoint，因此不能用已发布 artifact 判断上游最终采用了哪一侧。UniLab 显式选择
可执行 model contract 的 312D，并只取 compact critic 的 4D privileged tail；这是有记录的
兼容修复，不是上游 live-env graph 或 custom checkpoint 的 exact 复现。

显式的 `source_barlow` NP3O 路径因此消费这个修复后的 contract：runner 的
`policy.onnx` 使用单输入 `[1,312]`；source-
compatible teacher artifact `barlow_twins_actor.pt`/`.onnx` 则消费两个输入 `obs` `[1,28]`
与 `obs_hist` `[1,10,28]`，输出 `[1,6]` 动作。每个 custom 导出还会在图旁写出 sidecar，
记录规范化 algorithm/variant、维度、history reset 和 cost 通道元数据。custom sim2sim helper
在 sidecar 存在时会校验这些字段；没有 sidecar 的旧图仍只做 shape 校验并可加载。
上游 HIM/DreamWaQ 另外导出 mapping-ABI TorchScript/ONNX；UniLab 当前自动导出的是
flattened-history ONNX，不声称这两类导出 ABI 完全对等。NP3O/Barlow 的双输入
TorchScript 和 ONNX 是上述已覆盖的独立 contract。

长时间实验前可以先做短的 contract smoke：

```bash
uv run train --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  algo.num_envs=4 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=true
```

要从 UniLab checkpoint 生成部署 artifact，需要开启回放并选择无界面的录制模式。
PPO 的标准 play lifecycle 会在 checkpoint 旁写出 `policy.onnx`，同时仍走统一的
runner 和 environment contract：

```bash
uv run train --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  algo.num_envs=4 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=false training.play_render_mode=record \
  training.play_steps=2 training.play_env_num=1 \
  training.log_root=/tmp/wheelbipe-runs
```

已有 run 可以只走 play-only lifecycle 导出，不再训练新的 iteration：

```bash
uv run eval --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode record training.play_steps=1 training.play_env_num=1 \
  algo.load_run=/tmp/wheelbipe-runs/WheelbipeV14Flat/<run-directory>
```

生成的 `<run-directory>/policy.onnx` 就是下面 sim2sim helper 的输入模型。source delay
contract 可显式选择：5 ms 物理步、每个 20 ms 控制步四个 substep、physics-step
observation/action buffer，以及高端排除的 lag 范围。普通 PPO owner 默认就是该
`source_v14_physics` profile；`local_physics` 仅表示 UniLab 无延迟的 1 ms physics /
20 ms control timing 和闭区间范围语义。timing profile 不代表上游动力学、资产、ROS
循环或 sim-to-real 等价性：

```bash
uv run --no-sync scripts/sim2sim_wheelbipe.py \
  --delay-profile source_v14_physics --steps 200
```

`obs_delay_cfg`/`act_delay_cfg` 的范围和历史长度写在两个后端 owner YAML 中；观测支持
`control` 或 `physics` 步单位，动作在 backend 声明的逐物理步回调中延迟。两种 latency
不能与 `control_config.simulate_action_latency` 叠加，reset 时会为每个环境清空历史并采样
新的整数 lag。

owner 控制器在初始化时还会把配置的 torque limit 与 backend 已 materialize 的执行器范围
取交集。exact V14 owner 对每个主动腿部通道请求 40 N m、每个轮通道请求 5 N m；仓库内置
MJCF 也独立施加相同的轮部 ±5 N m 范围。实际交集可从环境的 `torque_contract` 诊断快照读取。

PPO 超参数、奖励权重、域随机化和地形设置以 owner YAML 为准：
`conf/ppo/task/wheelbipe_v14_flat/` 与
`conf/ppo/task/wheelbipe_v14_rough/`。

### Source 域随机化 contract

exact source owner 保留 pinned `EventCfgV14` 的调度语义，而不是把它缩成一个泛化的
“domain randomization”开关。分离的 `env.domain_randomization_contract` 快照会记录
后端最终采用的转换和缺口。当前主动 contract 如下：

| 调度 | Pinned source event | UniLab 物化方式与边界 |
| --- | --- | --- |
| startup mass 与 COM | base mass × `[0.9, 1.3]`；leg/gimbal body mass × `[0.9, 1.1]`；wheel mass × `[0.9, 1.1]`；base COM 的 x 为 `[-0.04, 0.04]` m、y/z 为 `[-0.02, 0.02]` m | 通过公共 backend reset/materialization contract 施加逐环境 mass multiplier 与 base-COM offset。source inertia event 已关闭，不属于这里的迁移声明。 |
| startup body material | base static/dynamic `[0.01, 0.1]`、restitution `[0.02, 0.2]`、64 buckets；wheel static `[0.5, 1.2]`、dynamic `[0.4, 1.0]`、restitution `[0.02, 0.2]`、64 buckets；guide static/dynamic `[0.1, 0.7]`、restitution `[0.01, 0.1]`、8 buckets；consistent sample 会把 dynamic clamp 到不大于 static | MuJoCo/Motrix 只有一个 sliding-friction 通道，因此以 sampled dynamic friction 驱动该通道，不虚构 PhysX static/dynamic/restitution contact law。本地 asset 包含 16 个被动导向轮及其圆柱碰撞体，guide material 映射到对应碰撞体；缺少 target 的下游精简 asset 仍会 warning 并记录 `not_applicable_missing_target`。 |
| startup joint friction，add/uniform | front/rear 主动关节 static/dynamic `[0.25, 1.0]`、viscous `[0.05, 0.2]`；wheel static `[0.05, 0.25]`、viscous `[0, 0.01]`；inactive linkage static `[0.05, 0.1]`、viscous `[0.01, 0.025]`；gimbal static/viscous `[0.002, 0.01]` | source 未单列 dynamic range 时，使用 static range 生成诊断用 dynamic sample，并 clamp 到 static。sampled static 映射为 additive DoF `frictionloss`；两个本地后端都只有一个 Coulomb 系数，所以独立 dynamic sample 仅作诊断。MuJoCo 将 viscous friction 映射到 DoF damping；Motrix 明确记录 viscous `unsupported_omitted` 并 warning，但 Coulomb 随机化仍生效。 |
| 满足间隔的 reset actuator gain 与 effort | 全局 stiffness/damping × `[0.75, 1.25]`，reset 最小间隔 720 steps；spring2 stiffness × `0.01`、damping × `[0.5, 1.5]`；leg effort × `[0.8, 1.1]`；wheel effort × `[0.9, 1.1]` | 已物化主动 leg、wheel、spring2 与 gimbal group。spring2 的 source stiffness baseline 为 0，因此 × `0.01` 后仍为 0；固定 50 N s/m damping 先乘全局系数，再乘 spring 系数。source passive-linkage `IdealPD` damping baseline `0.01` 也应进入全局 event，但本地 asset 没有对应 passive actuator：会 warning 并记录 `unsupported_unmaterialized_actuator`；这些关节的 startup joint friction 仍生效。`Rough-Play-v0` 只关闭全局 Kp/Kd event，保留继承的 spring/effort event。 |
| episode reset | root roll/pitch 为 `[-0.15, 0.15]` rad，yaw 为 `[-3.14, 3.14]` rad；左右弹簧各自采样 `[-50, 50]` N preload | pose 与 preload 都通过 owner reset contract 重采样；preload 会叠加到下述 400--600 N 线性弹簧 profile。 |
| interval disturbance | 每 5--10 s 为 root velocity 的 x/y 加 `[-0.25, 0.25]` m/s，z 为 0；base world-frame force XYZ 各 `[-10, 10]` N、torque XYZ 各 `[-1, 1]` N m | MuJoCo 与 Motrix 都通过 typed backend contract 实现 root velocity、body force 与 body torque；不支持的后端会失败，不会静默丢弃 torque。在 HIM owner 中 random wrench 会覆盖而不是叠加 vertical-assist latch，详见上文。 |
| V2 gimbal startup | heading controller 的 Kp 为 `[20, 40]`、Kd 为 `[0.05, 0.1]`，uniform | 这是 env-owned V2 heading controller 的独立 startup event，不是 reset-time articulation gain event。 |

这些转换保留的是公开 task/config identity，不是 Isaac Sim/PhysX 的接触物理、经过
solver 积分后的分布、数值轨迹或收敛效果。尤其是单 Coulomb material/joint-friction
映射、Motrix viscous omission 与未物化的 passive `IdealPD` gain，
都是明确的非 parity 边界。

### 源 USD 的实例碰撞体

源 `wheelbipeV14_2_1` USD 的碰撞形状位于实例内部。只遍历普通 prim 会漏掉
16 个导向轮圆柱、4 个前连杆 box 和云台 yaw 圆柱；只保留 body/joint/inertia
会使导向轮无法接触台阶。MJCF 现包含源文件的全部 33 个 box/cylinder 碰撞体，
对应 touch site 使用同一位置、姿态和尺寸。几何在静态资源中物化，不在训练热路径
读取 USD。独立源快照与 SHA 位于测试 fixture `source_collision_primitives.json`；
导向轮与 200 mm 台阶的接触测试验证其在机身 box 接触之前参与碰撞。

### 大台阶训练与七月 rough 的区别

`wheelbipe_v14_rough` 对应七月发布的 rotation/stair 任务。要复现服务器
`2026-09-01_17-03-38_rtx4090x4_from_flat3500` 的训练，应使用已有精确任务入口
`Robotics-Wheelbipe-V14-Rough-v1`。它使用 13 列 running 地形，包含
150–350 mm 低速台阶、150–400 mm 高速台阶和源 Airborne/StepUp 状态机。
不能用七月小台阶测试的通过率代替 200 mm 垂直台阶验收。

```bash
uv run train --algo ppo --task Robotics-Wheelbipe-V14-Rough-v1 --sim mujoco \
  algo.seed=44 algo.num_envs=512 algo.max_iterations=20000 \
  training.device=cpu training.no_play=true \
  algo.load_run=/absolute/path/to/source_flat3500/model_3500.pt
```

以上是单进程复现入口；源 run 的四 GPU 执行规模和训练收敛不能由短程 warm start
代替。ROS2 的 35D/6D ONNX 可以加载这类模型，但原部署配置的弹簧、惯量、材料与
训练不同，正常模式也不提供源训练的地形扫描。部署时需独立记录模型、物理配置、
速度和高度命令，并验证两轮确实上台阶、离开平台且机器人保持直立。

### Rough 地形 contract

rough exact owner 不共用一个泛化地形：`Rough-v0` 使用 source
Rotation99 的 7 个 family/10 列分配，`Rough-v1` 使用 running 的 11 个
family/13 列课程；两个 Play owner 都只使用
`cliff_inv_stair_slope_short_for_rm_play`，分别为 10 行和 1 行。列数由
source 的 proportional allocation 规则产生，地形类型名/id 作为 backend
materialization metadata 传回 env。reset 使用分配的 env origin 选择地形命令模板；
rollout 中则依当前 root XY 重新映射 cell，因此机器人跨列后命令约束会立即
跟随新地形。超出 grid 的坐标会在查表时 clamp，不会访问越界 metadata。

rough timeout 使用 source 的整体 grid 边界：半宽/半长由行列数与 cell size
计算，普通 owner 加入 border，再减去 0.5 m margin；`Rough-Play-v0`
显式使用 inner terrain area，不把 border 计入有效区。这是整体边界，不是
每个 cell 边界，与上述跨 cell 命令切换相容。上游 grid-bars 是 box mesh；
UniLab 将其顶面转换为双后端通用 heightfield，保留高度轮廓但不声称三角剖分
或接触物理相同。

source critic 的 height 通道始终是 world-frame root z；Flat/Rough-v1 先将其
clip 到 `[0.05, 0.45]`。只有 reward 路径在 rough/airborne owner 下再减去
terrain ground estimate，不会把 terrain-relative height 误塞进 78 维 critic。
`leg_joint_acc`/`wheel_acc` 也不再由 env 用 control-step 差分猜测；两个
backend 通过公共 `get_dof_acc()` contract 在 WheelBipe 的每个 5 ms physics substep
更新有限差分值，reset 时清零。这是 MuJoCo/Motrix 对 Isaac generalized
acceleration 的明确转换，不是数值物理等价声明。

Motrix owner YAML 显式启用了冷路径兼容 profile，关闭该机构的 6 个 MJCF
闭环 equality constraint；否则当前 MotrixSim solver 可能在接触求解时失败。
MuJoCo 仍保留 canonical constraint，因此 Motrix rollout 可作为后端 smoke 和
训练目标的证据，但不代表逐位物理等价。

## 策略 contract

仓库内置部署图是固定的 normal-mode 策略：一个 `obs` 输入
（`float32[1,35]`）和一个 `actions` 输出（`float32[1,6]`）。actor 观测顺序为：

```text
command[3], height*5[1], gyro*0.5[3], projected_gravity[3],
leg_position[4], wheel_position[2], leg_velocity*0.1[4],
wheel_velocity*0.1[2], previous_action[6], normal_mode[7]
```

normal 模式的轮位置槽位保留为零，最后的模式向量固定为
`[1, 0, 0, 0, 0, 0, 0]`。两个后端 owner YAML 都显式固定 critic group 为 78 维。

相同的 35-field layout 不代表预处理相同。exact source-training owner 先逐分量 clip raw
值，再施加 scale：command clip 到 `[-100, 100]`；height command 先 clip 到 `[0, 1]`
再 ×5；gyro 先 clip 到 `[-100, 100]` 再 ×0.5；gravity 和 leg position clip 到
`[-100, 100]`；leg/wheel velocity 先 clip 到 `[-200, 200]` 再 ×0.1；previous action
clip 到 `[-100, 100]`。pinned clip dictionary 没有 control-mode 项；七维 tail 只施加
逐槽位 scale（normal/state owner 的零基 slot 5，即第六个字段，为 ×5；V2 全部为 1），
然后 assembled source actor tensor 会把 non-finite 值清为 0。source actor 不再执行
post-scale global clip。generic/ROS deployment builder 则有意保留独立发布的
`scaleClamp` contract：先 scale，再把每个结果字段 global clip 到 `[-100, 100]`。

15 个 pinned exact owner 均保持 source `debug_value_diagnosis=false`，因此 raw
actor/critic 的 non-finite 或最大绝对值诊断不会启用 observation-outlier
termination gate。这不会关闭立即物理数值安全：base linear velocity、base
angular velocity 或 active-joint velocity 出现 non-finite 时仍会立即终止；
active-joint velocity 超过 500 rad/s、base angular velocity 超过 200 rad/s 或
base linear velocity 超过 100 m/s 也同样立即生效。上述 actor-side
`nan_to_num` 只是观测清洗，不会修复底层物理状态，也不会取代这些安全检查。

策略动作顺序是
`left_front1`、`left_rear1`、`right_front1`、`right_rear1`、
`left_wheel`、`right_wheel`。MuJoCo 机器人描述有 8 个原生执行器槽位；适配层把
6 个公开动作映射到 4 个腿部和 2 个轮部槽位，并由 owner-level 控制器驱动 2 个
弹簧槽位。映射在环境构造阶段解析，step 热路径不会探测后端私有能力。

pinned source runner 文件设置 `clip_actions: null`。exact owner 将其编码为
`clip_actions=inf`：finite raw policy action 不会被预裁剪，并原样进入 observation
history 以及 action-rate/smoothness reward。只有解码后的物理 target 会限幅：四个
leg position target 在 ×0.5 并叠加 default pose 后 clip 到 `[-3.14, 3.14]` rad；
两个 wheel velocity target 在 ×10 后 clip 到 `[-100, 100]` rad/s。因此 runner action
不设界并不等于 actuator command 不设界，也不会被 generic/ROS safety wrapper 替换。

弹簧槽位使用 V14 Isaac 训练 profile：压缩行程 0.07 m 内，弹簧力从 400 N
线性变化到 600 N；每个 episode 采样 [-50, 50] N 的预载随机量，并保留固定
50 N s/m actuator damping。source 域随机化先施加全局 `[0.75, 1.25]` gain
multiplier，再施加 spring-only `[0.5, 1.5]` damping multiplier；source 中关闭的
只是额外的伸展/压缩随机阻尼项，并非这项固定 actuator damping。ROS 气弹簧硬件
模型仍属于独立的 adapter/dynamics contract。

## 使用训练或发布策略进行 sim2sim

可以直接用仓库内的 ONNX artifact 驱动 UniLab MuJoCo owner：

```bash
uv run --no-sync scripts/sim2sim_wheelbipe.py --steps 200
uv run --no-sync scripts/sim2sim_wheelbipe.py --steps 200 --command 0.3 0.0 0.0
uv run --no-sync scripts/sim2sim_wheelbipe.py \
  --model /tmp/wheelbipe-runs/WheelbipeV14Flat/<run-directory>/policy.onnx \
  --sim mujoco --steps 200
uv run --no-sync scripts/sim2sim_wheelbipe.py \
  --model /path/to/wheeled-legged_RL/pretrained/.../policy.pt \
  --sim mujoco --steps 200
```

该 helper 接受 `--sim mujoco` 或 `--sim motrix`。Motrix 使用 owner 显式声明的
constraint 兼容 profile；该 profile 不会作用于 MuJoCo scene。

仅当策略已针对 5 ms/4-substep timing profile 训练或验证时，才应使用
`--delay-profile source_v14_physics`。命令会明确打印所选 timing profile 的作用范围；
更完整的 owner-contract 证据见上方矩阵。

脚本在 step 环境前会校验图的 I/O 和观测维度。使用 `--model` 可以测试其他兼容的
`obs`/`actions` 图；输入或输出维度不匹配时会明确拒绝。
normal helper 同时接受单输入 ONNX artifact 或可信的上游 TorchScript `policy.pt`，
并检查 35 维 observation 与 6 维 action 的有限性。TorchScript 推理固定在 CPU；如果
导出图把 batch 固定为 1，而 rollout 请求多个环境，helper 会逐行执行。不要把 custom
历史图传给该 helper。

normal sim2sim helper 可按 source-compatible schema 流式写入速度/reward CSV，周期性
刷新独立的交互 HTML，并可选显示四组 Matplotlib 实时面板：

```bash
uv run scripts/sim2sim_wheelbipe.py --steps 1000 \
  --trace-csv /tmp/wheelbipe-trace.csv \
  --trace-html /tmp/wheelbipe-trace.html --realtime-plot
uv run scripts/export_wheelbipe_trace_html.py /tmp/wheelbipe-trace.csv \
  -o /tmp/wheelbipe-trace-offline.html
```

实时数据 API `WheelbipeRealtimeBuffer` 与显示后端解耦；只有选择
`--realtime-plot` 才会导入 Matplotlib。当前 owner 会记录所选环境的 total reward；
只有 owner 显式提供逐环境 `info["reward_terms"]` 时才生成分项列，绝不会把低频/全局
log mean 冒充成逐环境 reward。

对于 custom 历史策略导出，请使用配套 helper；它会在 episode 边界初始化并重置历史：

```bash
uv run --no-sync scripts/sim2sim_wheelbipe_custom.py \
  --algorithm him_ppo \
  --model /tmp/wheelbipe-runs/WheelbipeV14FlatHIM/<run-directory>/policy.onnx \
  --sim mujoco --steps 200
```

该 helper 接受上游和规范化拼写 `him`、`him_ppo`、`ppo_him`、`dreamwaq`、`dream_waq`、
`ppo_dreamwaq`、`np3o`；NP3O 会在所选环境 owner 中保留 5 个 constraint 通道。helper
会 compose 与 exact 训练相同的 source-profile task/backend owner，因此 reward、curriculum、
cost 与 Motrix 兼容字段不会退回 bounded 默认值；创建环境前，export sidecar 会选择 compact
flattened-history、修复后的 312D source-Barlow 或双输入 actor ABI。不要把 custom 图传给
`scripts/sim2sim_wheelbipe.py`，后者的 contract 是 normal 35 维输入。

显式的 `source_barlow` NP3O profile 还会生成
`barlow_twins_actor.pt` 和 `barlow_twins_actor.onnx`。这两个 artifact 来自
`actor_teacher_backbone` 的 source-compatible teacher graph，输入为两个：
`obs` `float32[1,28]` 与 `obs_hist` `float32[1,10,28]`，输出为
`actions` `float32[1,6]`。它们与 runner 的单输入 312 维 `policy.onnx` 是不同
loader contract，不能互相加载。标准 NP3O run 目录（或 `policy.onnx`）会直接执行文档所述的
修复流 `[policy28, privileged-tail4, history280]`；这一兼容选择不构成 checkpoint-exact 声明。
需要双输入 ABI 时，在 custom sim2sim 显式选择 source actor；可传 run 目录，或传同目录
的完整 `policy.onnx`（helper 会解析旁边的 source actor）。默认优先 ONNX；如果两个 artifact
同时存在，可用 `--source-barlow-format torchscript` 显式选择 `.pt`，也可直接传
`barlow_twins_actor.pt`：

```bash
uv run --no-sync scripts/sim2sim_wheelbipe_custom.py \
  --algorithm np3o --source-barlow-actor \
  --model /tmp/wheelbipe-runs/WheelbipeV14FlatNP3OBarlow/<run-directory> \
  --sim mujoco --steps 200

uv run --no-sync scripts/sim2sim_wheelbipe_custom.py \
  --algorithm np3o --source-barlow-actor \
  --source-barlow-format torchscript \
  --model /tmp/wheelbipe-runs/WheelbipeV14FlatNP3OBarlow/<run-directory> \
  --sim mujoco --steps 200
```

source actor rollout 在 reset 时使用上游的十帧零填充 history，并在 autoreset 行上重新开始
history。环境创建前会校验 metadata sidecar；这只记录 ABI 和算法兼容性，不代表上游动力学或
收敛 parity。TorchScript 会反序列化可执行代码，只应加载可信 archive；单输入 custom `.pt`
checkpoint/graph 会被拒绝，不会误接 source adapter，应改用其 ONNX 导出。

## ROS 2 部署路径

上游部署仓库包含 C++ `template_ros2_controller` 和
`template_real_ros2_ctrl::RealBridge`。UniLab 保留两条彼此独立的路径：无需安装 ROS
即可执行的进程内 adapter，以及可物化到外部 colcon workspace 的 exact-digest native
ROS 2 source bundle。

### 进程内 adapter（无需 ROS）

可执行的 owner-layer adapter 是
`unilab.training.wheelbipe_ros2.WheelbipeRos2Controller`，它可以和 UniLab
环境在同一进程中运行：

```bash
uv run --no-sync scripts/sim2sim_wheelbipe_ros2.py --sim mujoco --steps 200
uv run --no-sync scripts/sim2sim_wheelbipe_ros2.py --sim motrix --steps 200 \
  --command 0.3 0.0 0.0 --height 0.27
# 可用 --config 指定 source-compatible 参数文件（在创建 simulator 前只解析一次）。
uv run --no-sync scripts/sim2sim_wheelbipe_ros2.py \
  --config conf/deployment/wheelbipe_v14_ros2.yaml --steps 200
# 在不加载 ONNX 或 simulator 的情况下查看 API/protocol 边界。
uv run --no-sync scripts/sim2sim_wheelbipe_ros2.py --print-contract
```

适配器保留了可审计的 source 边界：

| 上游 contract | UniLab adapter |
| --- | --- |
| ros2-control 顺序的 8 个关节 | `WHEELBIPE_ROS2_JOINT_NAMES`（4 腿、2 轮、2 弹簧） |
| `motion_command` `geometry_msgs/msg/Twist` | `set_motion_command(linear_x, angular_z)`；`linear.y` 保留为 0 |
| `height_command` `std_msgs/msg/Float64` | `set_height_command(height)` |
| `state_command` / `current_state` `std_msgs/msg/Int32` | `set_state_command(0..3)` 与 `WheelbipeRos2ControlOutput.state` |
| `joint_commands` `sensor_msgs/msg/JointState` 与 `joint_final_torque` `Float64MultiArray` | `WheelbipeRos2JointCommand` 的 position/velocity/effort/Kp/Kd/final-torque 数组 |
| 500 Hz ros2_control update、50 Hz ONNX inference | 每个 20 ms UniLab env step 内执行 10 次 500 Hz adapter update（复用 held env sample；仅为 cadence emulation） |

normal 图仍是严格的 `float32[1,35]`（`obs`）到 `float32[1,6]`
（`actions`）。Python owner 对齐 source 的 INIT（保持 10 ms）、IDLE、PREPARE、RL
转换，命令限幅与 0.5 s timeout，可选 moving-average/low-pass action filter，
hardware-PD 输出模式、有限值检查和 safe-stop。source revision 与默认参数记录在
`conf/deployment/wheelbipe_v14_ros2.yaml`。
当 `use_dt7=true` 时，上游 real hardware description 还会提供 `dt7` sensor，包含
`cmd_state`、`cmd_vel_x`、`cmd_omega_z`、`cmd_height` 四个字段；无 ROS owner 通过
`WheelbipeRos2Dt7Command` 接收这些值，但不会打开硬件传输。没有 policy 的 RL 请求或
超出范围的 DT7 state byte 会沿用 source callback 的 fail-closed 行为并被忽略。
CLI 的 `--config` 选择这个 cold-path profile；该 rollout 入口会在 10 ms INIT
保持后自动请求 RL。

这是进程内 simulation/deployment API，不是 ROS graph：不会发布 DDS topic、加载上游
C++ plugin、打开 `/dev/wheelbipe_h7`，也不提供实机安全认证。纯 Python RealBridge
helpers `encode_wheelbipe_real_command_packet` 与 `decode_wheelbipe_real_state_packet`
保留已审计的 MIT wire layout（command 158 字节、state 143 字节、header `A8 E6`、
trailer `C3 F7`、反射 CRC16 多项式 `0x8005`），但只负责编解码，不负责串口传输或
realtime 保证。纯 `WheelbipeRealBridgeGate` 在不打开设备的前提下建模固定的 1,000 ms
reconnect throttle、100 ms stale-state inhibition、lifecycle activation 与 safe-stop output。
state packet 必须通过 marker、finite payload、四元数范数和 CRC
检查才会被接受。

### 打包的 native ROS 2 source bundle

`deployment/ros2/wheelbipe_v14_native/` 打包了 `wheelbipe_ros2_sim2sim` 固定 revision
`daa34f54d56cab91b3989d8152a7ce7b61092994` 的 MIT source：7 个 colcon package、72 个
source 文件、3 个 pluginlib descriptor、4 个 launch 文件、4 个 config、2 条 udev rule，
以及仓库内 policy 与 21 个 mesh 的 overlay。bundle 包含 `controller_manager` launch、
controller/RealBridge/MuJoCo-system plugin、serial reconnect/stale-state source 与
keyboard/Xbox teleop source。以下命令无需 import ROS 即可 verify 或 materialize：

```bash
uv run scripts/sim2sim_wheelbipe_ros2.py --verify-native-bundle
uv run scripts/sim2sim_wheelbipe_ros2.py --probe-native-runtime
uv run scripts/sim2sim_wheelbipe_ros2.py \
  --materialize-native-workspace /tmp/wheelbipe-v14-colcon
uv run scripts/sim2sim_wheelbipe_ros2.py --require-native-runtime
```

已验证 source-tree digest 为
`67f0628533f6f8cc849159426fe6769636c304ab4596d34cbfeed4a2f4e53b27`。
materialize 会复制 22 个 asset-overlay 文件，绝不会 build、source 或 launch workspace。
2026-08-31 验证主机的 probe 报告缺少 `rclpy`、`launch`、`launch_ros`、
`ament_index_python`、`ros2`、`colcon`、`xacro` 与 `rosdep`；因此
`--require-native-runtime` 以 return code 1 fail closed。ROS 2 Humble、Linux x86-64、
ONNX Runtime C++ 1.20.0、MuJoCo 3.5.0 及 manifest 中其余依赖属于外部 optional
deployment boundary。

2026-09-05 在同一台开发机上使用 RoboStack conda 环境
`wheelbipe_humble`（ROS 2 Humble）完成了外部 `wheelbipe_ros2_sim2sim`
workspace 的实际 build 与 headless launch：MuJoCo 场景加载、`joint_state_broadcaster`
与 `template_ros2_controller` 配置激活、状态机 INIT → IDLE → PREPARE → RL 完整流转，
并通过 `ros2 topic echo /wheelbipe_V14/joint_states` 观察到 8 个关节在策略驱动下连续
演化。所用的 policy 就是本任务导出的 rough `policy.onnx`（该仓库
`policy/parallel/V14-rough-unilab-1999.onnx`，`[1,35] → [1,6]`）。期间发现
`install/` 中一份旧 `wheelbipe_V14.yaml` 的 `joints` 顺序（front 交叉排列）与
controller 的关节合同不一致，重新 `colcon build` 后与 src 同步即恢复。

静态 verify 与纯 Python reconnect/stale/teleop 测试不能证明 workspace 可在 ROS 环境编译
运行，但上段记录的是真实 ROS 2 headless launch 证据。仍未打开
`/dev/wheelbipe_h7`、未连接机器人硬件，因此不宣称 realtime、硬件安全或
sim-to-real 认证。

## 来源与许可

源仓库及固定 revision 记录在根目录的 `THIRD_PARTY_NOTICES.md`。重新分发的网格和
策略许可文件与资产放在一起：

- `src/unilab/assets/robots/wheelbipe_v14_2/meshes/ASSET_LICENSE.md`
- `src/unilab/assets/robots/wheelbipe_v14_2/mjcf/ASSET_LICENSE.md`
- `src/unilab/assets/policies/wheelbipe_v14/ASSET_LICENSE.md`

策略文件为
`src/unilab/assets/policies/wheelbipe_v14/V14-35-flat-and-rotation-13k.onnx`，
SHA-256 为：

```text
a1244761f7ede02f8c80d076d4315a25f014df43df3f7f0d20c2ca5bcd518719
```

UniLab 原有环境/配置代码继续使用仓库 Apache-2.0 许可；复制的 SCUTRobotLab 资产继续
遵循各自目录中的 MIT 声明。`src/unilab/algos/torch/him_ppo/` 下的历史算法文件保留
HIMLoco 的 CC BY-NC-SA 4.0 署名（其中 BSD 部分已单独标注）；DreamWaQ/NP3O 适配边界
没有 blanket Apache 或商业使用授权。重新分发算法代码前请查阅根目录
`THIRD_PARTY_NOTICES.md`。

## 当前边界

本次迁移覆盖 normal 35D/6D 策略路径、命名的 state-machine/gimbal 变体以及上面的
flat 历史策略路由，使用 UniLab 的 MuJoCo、Motrix owner。Motrix 的无闭环 profile 是明确的
solver 兼容边界，不宣称跨后端物理或 sim-to-real 等价性。无 ROS Python adapter 可执行，
固定 native C++ source 已打包且可 materialize，但两者都不是 ROS graph 或真机已验证证据。
上游历史 TensorBoard/event log 不会导入 UniLab run history；Isaac Sim/PhysX runtime 不会
迁移成 UniLab backend；也不包含未公开的 STM32 H7 下位机 firmware。
内置 normal 图是 flat-and-rotation 示例；粗糙地形部署应使用单独训练并完成 contract 检查的策略，
不应静默复用该图。不宣称收敛、source learning curve、逐位物理、realtime 或硬件安全等价。
