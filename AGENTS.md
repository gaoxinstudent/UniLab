# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always use `uv run`, not python**.

UniLab 是一个 **高性能、模块化、contract 驱动** 的 RL infrastructure 仓库。

## Common Commands

```bash
make format         # ruff format + ruff check --fix
make type           # mypy src/unilab + pyright
make check          # format + type (PR 提交前必须通过)
make test           # 非 slow 测试
make test-cov       # 非 slow 测试 + 覆盖率报告
make test-slow      # slow 集成测试和训练冒烟测试
make test-all       # make check && make test-cov
make clean          # 清理构建产物和缓存

# 平台特定同步
make setup          # Linux CUDA / macOS: uv sync --extra mujoco --extra motrix
make sync-rocm      # Linux AMD ROCm
make sync-xpu       # Linux Intel Arc / iGPU

# 运行单个测试
uv run pytest tests/path/to/test_file.py::test_name -v
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  CLI:  conf/<algo>/config.yaml  ← Hydra defaults + task owner   │
│        scripts/train_*.py       ← 组装流程，无长期业务逻辑        │
├──────────────────────────────────────────────────────────────────┤
│  Training:  src/unilab/training/  (run.py, experiment.py,        │
│             sim2sim.py, reward.py, rsl_rl.py, backend_adapter)   │
├──────────────────────────────────────────────────────────────────┤
│  Algorithms:  src/unilab/algos/torch/  (ppo, appo, offpolicy,    │
│               him_ppo, hora, fast_sac, fast_td3, flash_sac)      │
│               src/unilab/algos/mlx/ppo/  (macOS-only MLX PPO)    │
├──────────────────────────────────────────────────────────────────┤
│  IPC:  src/unilab/ipc/  (async_runner.py, replay_pipelines/)     │
│        SharedMemory → collector subprocess ↔ learner process     │
├──────────────────────────────────────────────────────────────────┤
│  Environments:  src/unilab/envs/                                 │
│    locomotion/  ← go1, go2, go2w, g1, go2_arm, real68           │
│    manipulation/ ← allegro_inhand, sharpa_inhand, stewart        │
│    motion_tracking/ ← g1 (tracking, box, flip, wall_flip), x2   │
├──────────────────────────────────────────────────────────────────┤
│  Base Contracts:  src/unilab/base/                               │
│    base.py  ← ABEnv (env contract), EnvCfg                       │
│    np_env.py ← NpEnv (numpy env with NpEnvState)                 │
│    registry.py ← env 注册/发现 (@envcfg, @env, make)             │
│    backend/base.py ← SimBackend (backend contract)               │
│    scene.py ← SceneCfg                                           │
├──────────────────────────────────────────────────────────────────┤
│  Backends:  src/unilab/base/backend/                             │
│    mujoco/  ← MuJoCoUni backend                                  │
│    motrix/  ← MotrixSim backend                                  │
├──────────────────────────────────────────────────────────────────┤
│  Domain Rand:  src/unilab/dr/                                    │
│  Terrains:     src/unilab/terrains/                              │
│  Visualization: src/unilab/visualization/                        │
│  Tools:        src/unilab/tools/ (completion, pull_assets, etc.) │
└──────────────────────────────────────────────────────────────────┘
```

### 核心分层合同

1. **`ABEnv`** (`base/base.py`) — 最抽象的环境接口，不与 numpy 耦合。
2. **`NpEnv`** (`base/np_env.py`) — numpy 向量化环境，`step()` 返回 `NpEnvState`（`obs: dict[str, ndarray]`，`reward/terminated/truncated: ndarray`，`info: dict`）。`reset()` 签名 `(obs_dict, info_dict)`。
3. **`SimBackend`** (`base/backend/base.py`) — 统一的后端抽象接口。env 层**只能**调用 `SimBackend` 中已声明的方法；禁止直接调用后端子类私有方法。
4. **`scripts/`** — 只组装流程，不承载长期业务规则。业务规则留在 owner 模块。

### 注册系统

env 通过 decorator 注册，支持按 backend 分派：

```python
# 1. 注册 config
@envcfg("g1_walk_flat")
@dataclass
class G1WalkFlatCfg(G1BaseCfg): ...

# 2. 注册 env 实现（per backend）
@env("g1_walk_flat", "mujoco")
class G1WalkFlatMujocoEnv(G1BaseEnv): ...

@env("g1_walk_flat", "motrix")
class G1WalkFlatMotrixEnv(G1BaseEnv): ...
```

注册发现链：`envs/locomotion/__init__.py` → `__unilab_registry_modules__` → import 各 robot 包的 `__init__.py` → 触发 `@envcfg` / `@env` decorator。`ensure_registries()` 在训练入口调用。

### Config 层次结构

采用 Hydra 三层 owner YAML 模式，通过 `task=<task>/<backend>` 选择：

```
conf/<algo>/config.yaml          # 算法默认配置
  └── task: <task>/<backend>.yaml # task owner（由 CLI --task/--sim 选择）
        defaults:                 # 继承 base.yaml 共享合同字段
          - /task/<task>/base     # ← Sim2Sim DENYLIST 字段必须在此共享
          - _self_
```

base.yaml 是 DENYLIST 字段的唯一来源，保证跨后端一致。mujoco.yaml / motrix.yaml 只放后端特有调参（reward scales、noise、curriculum 等）。

### 算法矩阵

| 算法 | 入口 | Config Group | 类型 |
|------|------|-------------|------|
| PPO | `train_rsl_rl.py` | `ppo` | On-policy |
| MLX PPO | `train_mlx_ppo.py` | `ppo` (macOS-only) | On-policy |
| APPO | `train_appo.py` | `appo` | Async on-policy |
| SAC | `train_offpolicy.py` | `offpolicy/algo/sac` | Off-policy |
| TD3 | `train_offpolicy.py` | `offpolicy/algo/td3` | Off-policy |
| FlashSAC | `train_offpolicy.py` | `offpolicy/algo/flashsac` | Off-policy |
| HORA | `train_hora_distill.py` | `hora_distill` | Distillation |
| HIM-PPO | `train_him_ppo.py` | `ppo_him` | On-policy |

统一 CLI：`uv run train --algo <algo> --task <task> --sim <mujoco|motrix>`，`uv run eval` 用于播放，`uv run demo` 用于预训练演示。

### 异步架构 (AsyncRunner)

`src/unilab/ipc/async_runner.py` — APPO/SAC/TD3 的基础类：
- **SharedMemory**: CPU collector 子进程通过 `SharedReplayBuffer` 与 GPU learner 主进程通信
- **Collector**: 子进程运行 env step，产出 transitions
- **Learner**: 主进程从 shared buffer 采样训练
- **错误传播**: collector 子进程崩溃通过 error pipe 传递到主进程

## Core Principles

1. **Contract first**: 不为了一次通过绕过 env / backend / runner contract。
2. **Fix at owner layer**: `scripts/` 只组装流程，不承载长期业务规则。
3. **Config first**: task / reward / backend 优先通过 Hydra + registry 表达。
4. **Backend isolation**: MuJoCo / Motrix 差异留在 backend 适配层和配置层。
5. **Evidence only**: support claim 只写仓库里已有的注册、配置、测试或 benchmark 事实。
6. **Validate near risk**: 在最接近风险的边界补验证，不只跑顶层命令。
7. **Cold-path asset access only**: asset/XML/model metadata 只允许在 init / materialization / cache 等低频路径处理；热路径不能解析 asset，也不能靠 `getattr` / `hasattr` 探测 backend 私有能力。

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。算法超参数直接走 YAML compose，不经 Python 层解释。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。env 层只能调用 `SimBackend`（`base.py`）中已声明的方法；若某方法只在 MuJoCo 或 Motrix 中存在，必须先将其加入 `SimBackend` 抽象接口（可抛 `NotImplementedError`），禁止直接在 env 里调用 backend 子类的私有方法（即"功能泄漏/feature leakage"）。新增 backend 专有能力时，需同步更新 `SimBackend`。 |
| Asset / Metadata | `ASSETS_ROOT_PATH`、`model_file`、XML / asset 元数据只允许在 init / materialization / cache 等低频路径访问；`step/reset/domain randomization` 等热路径不得解析 asset 或基于 asset 元数据做运行时分支。 |
| Asset / XML structure | `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml` 等 fragment），**禁止放进 robot.xml**。robot.xml 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关；keyframe 是 task 起始姿态，属于场景或 task 资源。motrix 后端需要 keyframe 时通过 `scene.fragment_files` 引用 fragment XML。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |
| Sim2Sim 契约 | 跨后端 play 时，影响策略 I/O / 网络结构的字段必须跨后端一致；不一致即 `CrossBackendIncompatibleError`。详见下方 Sim2Sim 章节。 |

## Sim2Sim 跨后端配置契约

`src/unilab/training/sim2sim.py` 按 dotted path 维护三类字段：

- **DENYLIST**（差异即 `CrossBackendIncompatibleError`）：`algo.obs_groups`、`env.control_config.action_scale`、`algo.policy.actor_hidden_dims` / `critic_hidden_dims`、`algo.empirical_normalization` / `algo.obs_normalization`、`env.sampling_mode`。`env.*` 子集对**任一方向**的不对称出现也 fail-closed；`algo` 专属字段目标缺省时按设计跳过（跨算法合法）。
- **WARNING_LIST**：`reward.*`、`env.control_config.simulate_action_latency`、`env.ctrl_dt`。
- **ALLOWLIST**（自由覆盖）：`training.sim_backend`、`env.scene`、`training.play_steps`、`env.domain_rand`、`env.noise_config`、`env.commands.vel_limit`。

训练时 `ExperimentTracker.start()` 把上述字段写入 `run_config.json` 的 `contract_snapshot`（不改 checkpoint 格式，旧 run 无 snapshot 时 fallback + warning）；五个 play 入口在建 env 前调用 `resolve_sim2sim_config` 校验，并用 `policy_load_dim_guard` 包裹 checkpoint 加载以把维度不匹配的隐晦报错重抛为显式诊断。设 `training.sim2sim_strict=false` 可把 DENYLIST 差异降级为 warning（默认 `true`）。DENYLIST 字段应通过 task 的 `base.yaml` 共享（范例：`conf/ppo/task/g1_walk_flat/{base,mujoco,motrix}.yaml`）；跨后端契约审计见 `scripts/audit_sim2sim_contracts.py`。

## Key File Pointers

- PPO: `scripts/train_rsl_rl.py`
- MLX PPO: `scripts/train_mlx_ppo.py`
- APPO: `scripts/train_appo.py`
- SAC / TD3: `scripts/train_offpolicy.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- registry: `src/unilab/base/registry.py`
- training run helpers: `src/unilab/training/run.py`
- config schema: `src/unilab/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`
- sim2sim 跨后端契约: `src/unilab/training/sim2sim.py`
- visualization helpers: `src/unilab/visualization/`
- env shared numeric helpers: `src/unilab/envs/common/rotation.py`, `src/unilab/envs/common/math.py`
- MLX rotation helpers: `src/unilab/algos/mlx/common/rotation.py`

## Test Structure

```
tests/
  base/         ← backend contract tests
  envs/         ← per-task env tests
  algos/        ← algorithm unit tests
  ipc/          ← IPC/async runner tests
  training/     ← training pipeline tests
  integration/  ← end-to-end integration tests
  scripts/      ← script-level tests
  config/       ← config correctness tests
  dr/           ← domain randomization tests
  benchmark/    ← performance benchmarks
  terrains/     ← terrain generation tests
```

## GitHub CLI (gh) 速查

### Issue 查看
```bash
gh issue view <number>
gh api repos/<owner>/<repo>/issues/<number> --jq '.body'
```

### PR 创建与管理
```bash
gh pr create --title "标题" --body "内容" --base main
gh pr list
gh pr view
```

### PR Gate

创建或更新 PR 前必须满足：

1. 最终提交已经完成，且 `git status --short --branch` 确认工作树干净。
2. 最终提交已经通过 `make test-all`。
3. 如果用户明确说明已经跑过 `make test-all`，不要重复跑；但必须在 PR body 的 Validation 里记录 `make test-all` 已完成。
4. 如果 `make test-all` 未通过且用户没有明确 override，不要创建或更新 PR。

### CI 工作流查看
```bash
gh run list
gh run list --workflow=<workflow-name>
gh run view <run-id>
gh run list --status=failure
```

### 常用组合
```bash
gh api repos/unilabsim/UniLab/issues/174 --jq '.title, .body'
git push -u origin fix/issue-174-mlx-ppo-config-alignment
gh pr create --title "fix: xxx" --body "Fixes #174" --base main
```

## Context

- 架构标准与验证详情：[docs/sphinx/source/zh_CN/4-developer_guide/0-index.md](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- 协作流程与 PR 规范：[docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md](docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md)
- 开发者入口（环境、命令、提交规范）：[CONTRIBUTING.md](CONTRIBUTING.md)
- 文档本地构建与发布到 UniLab-doc：[docs/sphinx/README.md#本地发布到-unilab-doc](docs/sphinx/README.md#本地发布到-unilab-doc)
