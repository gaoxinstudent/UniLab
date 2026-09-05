# WheelBipe V14

UniLab includes the normal-mode WheelBipe V14 locomotion owner migrated from
SCUTRobotLab's public training and ROS 2 deployment repositories. The owner
keeps policy-facing contracts in the environment layer and selects MuJoCo or
Motrix through the usual task/backend CLI flags.

## Available owners

| Task | Backends in the owner YAMLs | Environment registry name |
| --- | --- | --- |
| `wheelbipe_v14_flat` | `mujoco`, `motrix` | `WheelbipeV14Flat` |
| `wheelbipe_v14_rough` | `mujoco`, `motrix` | `WheelbipeV14Rough` |

Choose the backend with `--sim`; do not override `training.sim_backend` as a
separate switch.

### Exact upstream task ids

The source README publishes the following 15 Gymnasium ids.  UniLab keeps the
exact strings in the environment registry and accepts all 15 in the training
CLI.  To use an exact id, pass that literal string to `--task` (for example,
`--task Robotics-Wheelbipe-V14-Flat-v0`); the route column below shows the
canonical owner selected after translation. Every route is an executable
source-contract owner with a pinned reward/config graph; this does not claim
that MuJoCo or Motrix reproduces Isaac/PhysX dynamics or numerical trajectories.
`*-Play-*` routes select
`training.play_only=true` (and should normally be used with `uv run eval`).

| Upstream id | UniLab CLI route | Result |
| --- | --- | --- |
| `Robotics-Wheelbipe-V14-Flat-v0` | `--algo ppo --task wheelbipe_v14_flat` | compatibility owner |
| `Robotics-Wheelbipe-V14-Flat-v1` | `--algo ppo --task wheelbipe_v14_flat` | state-machine owner |
| `Robotics-Wheelbipe-V14-Flat-v2` | `--algo ppo --task wheelbipe_v14_flat` | gimbal owner |
| `Robotics-Wheelbipe-V14-Flat-Play-v0` | `--algo ppo --task wheelbipe_v14_flat` | compatibility owner, play-only |
| `Robotics-Wheelbipe-V14-Flat-Play-v2` | `--algo ppo --task wheelbipe_v14_flat` | gimbal owner, play-only |
| `Robotics-Wheelbipe-V14-Rough-v0` | `--algo ppo --task wheelbipe_v14_rough` | rough/gimbal owner |
| `Robotics-Wheelbipe-V14-Rough-v1` | `--algo ppo --task wheelbipe_v14_rough` | rough/state-machine owner |
| `Robotics-Wheelbipe-V14-Rough-Play-v0` | `--algo ppo --task wheelbipe_v14_rough` | rough/gimbal owner, play-only |
| `Robotics-Wheelbipe-V14-Rough-Play-v1` | `--algo ppo --task wheelbipe_v14_rough` | rough/state-machine owner, play-only |
| `Robotics-Wheelbipe-V14-Flat-DreamWaQ-v0` | `--algo dreamwaq --task wheelbipe_v14_flat` | compatibility owner |
| `Robotics-Wheelbipe-V14-Flat-DreamWaQ-Play-v0` | `--algo dreamwaq --task wheelbipe_v14_flat` | compatibility owner, play-only |
| `Robotics-Wheelbipe-V14-Flat-HIM-v0` | `--algo him_ppo --task wheelbipe_v14_flat` | compatibility owner |
| `Robotics-Wheelbipe-V14-Flat-HIM-Play-v0` | `--algo him_ppo --task wheelbipe_v14_flat` | compatibility owner, play-only |
| `Robotics-Wheelbipe-V14-Flat-NP3OBarlow-v0` | `--algo np3o --task wheelbipe_v14_flat` | compatibility owner |
| `Robotics-Wheelbipe-V14-Flat-NP3OBarlow-Play-v0` | `--algo np3o --task wheelbipe_v14_flat` | compatibility owner, play-only |

Only the 15 ids listed above are recognized as exact upstream aliases.  An
unknown string follows the ordinary task/config validation path and fails if
no owner exists; the CLI never silently falls back to `wheelbipe_v14_flat` or
`wheelbipe_v14_rough`.  The
named v1/v2 owners materialize their state-machine sensors or two gimbal
actuators during initialization, while keeping the six-action policy contract.
Their public state transitions, sensor contract, and gimbal actuator contract
are implemented at the environment owner layer and exercised on both backends.
Backend dynamics remain simulator-specific and are not claimed to be bitwise
equivalent to the source Isaac runtime.

The migrated history-policy owners are exposed through the same CLI, but only
for the flat task. Their internal owner names are listed here so a checkpoint
can be matched to the correct runner:

| Algorithm | Public task | Owner environment | History | Cost channels |
| --- | --- | --- | ---: | ---: |
| `him_ppo` | `wheelbipe_v14_flat` | `WheelbipeV14FlatHIM` | 5 | 0 |
| `dreamwaq` | `wheelbipe_v14_flat` | `WheelbipeV14FlatDreamWaQ` | 5 | 0 |
| `np3o` | `wheelbipe_v14_flat` | `WheelbipeV14FlatNP3OBarlow` | 10 | 5 |

These custom routes are implemented for `mujoco` and `motrix`; no `mjwarp` owner is
declared. They use the dedicated `scripts/train_custom_ppo.py` runner rather
than the normal PPO runner.
Their owner YAMLs select the explicit `source_v14_physics` timing/delay profile
(5 ms physics steps, four substeps per 20 ms control step) and enable the
history-policy delay buffers. Timing/delay is one part of the migrated owner
contract alongside observation, reward, domain-randomization, command,
termination, and state-machine semantics. The profile name itself selects only
timing/delay and does not claim source dynamics or asset parity.
DreamWaQ AdaBoot is available as an explicit opt-in on the DreamWaQ
owner. Set `algo.policy.adaboot_mode=reward_cv` (or `hybrid`) to use the
rolling completed-episode return window and source-style
`p_boot = 1 - tanh(scale * CV + offset)` coefficient. The owner validates the
window/bounds, publishes `adaboot_*` update metrics, and defaults to `off`.
This is a reward-CV integration only; it does not claim complete DreamWaQ or
AdaBoot paper parity. The corresponding uncertainty/hybrid representation path
and all values are still subject to the compact owner dimensions above.

## Public-source evidence matrix

| Public source feature | UniLab owner/config | Automated or executed evidence | Explicit boundary |
| --- | --- | --- | --- |
| 15 published task ids; flat/rough, state-machine, and gimbal variants | registry aliases plus `conf/ppo/task/wheelbipe_v14_{flat,rough}/` | registry/config tests and `tests/envs/locomotion/wheelbipe_v14/test_{contract,gimbal_state_machine,state_machine_stack,owner_yaml_contract}.py` | Isaac Sim/PhysX is not a UniLab runtime; backend dynamics are not bitwise-equivalent claims. |
| Normal PPO ABI: 35D actor, 78D critic, 6D action | `WheelbipeV14Flat`/`WheelbipeV14Rough` and both backend owner YAMLs | `test_training_semantics.py`, source-checkpoint tests, and the executed artifact table below | Strict-load and finite rollouts establish ABI/execution, not convergence or reward-curve parity. |
| Source reward, termination, command curriculum, delay, DR, contact-conditioned state machine, and torque mapping | environment owner plus task YAMLs (`source_v14_physics` for normal PPO; `local_physics` only as an explicit diagnostic profile) | WheelBipe environment contract/timing/training-semantics suites on MuJoCo and Motrix | Motrix disables six closed-loop equalities through its declared compatibility profile; no cross-simulator physics parity claim. |
| Upstream `model_state_dict`, TorchScript `policy.pt`, and ONNX policy | standard PPO eval adapter and `scripts/sim2sim_wheelbipe.py` | strict malformed/shape tests, actual `model_8000.pt`/`policy.pt`/ONNX execution below | TorchScript must be trusted; ONNX uses the dedicated helper; source checkpoints do not contain UniLab `contract_snapshot`. |
| HIM-PPO, DreamWaQ, NP3O + Barlow history policies | `conf/custom_ppo/task/wheelbipe_v14_flat_*` and custom runner | source-key checkpoint/optimizer regressions, numerical alignment with the pinned DreamWaQ class, source-Barlow two-input artifact tests, and a documented NP3O 312D compatibility repair | Flat MuJoCo/Motrix only; the upstream live env's 351D stream contradicts its 312D model contract; the public checkout has no custom checkpoint for an actual artifact load; no `mjwarp`; AdaBoot `reward_cv` is not full paper parity. |
| Keyboard commands and velocity/reward traces | `src/unilab/visualization/wheelbipe_{keyboard,trace}.py` | `tests/visualization/test_wheelbipe_tools.py` and interactive CLI routing tests | Viewer key-down semantics are latched; jump acceptance remains state-machine-owned; traces are diagnostics, not benchmarks. |
| ROS controller/state/wire contract without ROS | `WheelbipeRos2Controller`, deployment YAML, and Python packet/gate helpers | `tests/training/test_wheelbipe_ros2.py`, plus numerical MuJoCo/Motrix helper runs | In-process cadence emulation is not a DDS graph, realtime controller, serial transport, or hardware-safety proof. |
| Native ROS 2/controller_manager/pluginlib/serial/teleop source | `deployment/ros2/wheelbipe_v14_native/manifest.yaml` and packaged colcon sources | exact-digest bundle verification, workspace materialization, runtime probe, reconnect/stale/teleop fail-closed tests, plus a 2026-09-05 build and headless launch of the external workspace under RoboStack Humble (INIT→IDLE→PREPARE→RL with evolving joint_states) | No serial port or robot hardware was exercised; no realtime or hardware-safety claim. |

## Train

The default PPO owner follows the migrated policy dimensions and control
parameters and uses the source V14 training `source_v14_physics` profile (5 ms
MuJoCo physics, 20 ms policy control, and physics-step observation/action
delays). The frozen sim2sim/ROS loop is a deployment adapter and does not
redefine the training owner; select `local_physics` explicitly only to isolate
deployment-timing effects (1 ms, delay-free). Both flat-task owners
default to 4,096 parallel environments and 20,000 iterations for long runs;
resource-constrained hosts can reduce the batch explicitly with
`algo.num_envs=<count>`. For a headless run, disable playback explicitly:

```bash
uv run train --algo ppo --task wheelbipe_v14_flat --sim mujoco training.no_play=true
uv run train --algo ppo --task wheelbipe_v14_flat --sim motrix training.no_play=true
uv run train --algo ppo --task wheelbipe_v14_rough --sim mujoco training.no_play=true
```

A short contract smoke run is useful before a long experiment:

```bash
uv run train --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  algo.num_envs=4 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=true
```

### Rough training contract and warm start

The canonical `wheelbipe_v14_rough` training contract aligns with the released
upstream `rough_rotation_stair` 2026-07-23 run: `[256, 128, 64]` actor and
critic, `value_loss_coef=2.0`, seed 66, and the pinned observation/action/
delay/DR and special-mode command semantics of that run. That source run
itself warm-starts from the flat `model_8000.pt`, so UniLab long training
should likewise continue from the same-family source weights:

```bash
uv run train --algo ppo --task wheelbipe_v14_rough --sim mujoco \
  algo.num_envs=10240 algo.max_iterations=2000 training.no_play=true \
  algo.load_run=/path/to/wheeled-legged_RL/pretrained/26_infantry/rough_rotation_stair/\
2026-07-23_10-19-59/model_2500.pt
```

Robot-body and gas-spring physics are aligned to the `wheelbipe_ros2_sim2sim`
MuJoCo deployment asset (UniLab training physics = ROS 2 sim2sim deployment
physics; the sim2sim repository itself stays untouched): base mass 17.963 kg,
leg joints `frictionloss=1.5`/`armature=0.035`, wheel joints
`frictionloss=0.023`; the gas spring is linear 650 → 450 N over
q ∈ [-0.005, 0.07] with 500 N/(m/s) damping (mapped to `spring_offset=0.07`,
`spring_linear_up=650`, `spring_linear_down=450`, `spring_linear_length=0.075`,
`spring_damping=500`, no per-episode preload randomization). The model is
verified to climb a 140 mm single step using the same step geometry as the
ROS 2 deployment scene.

The warm start restores actor/critic weights (including std) only; the
optimizer and iteration counters start fresh. Terrain-crossing quality is
measured on the `cliff_inv_stair_slope_short_for_rm_play` play terrain by
body-frame forwardness (the signed forward-velocity share of the total speed):
the source weight reaches forwardness ≈ 0.94 with ≈ 0.9–1.1 m/s mean forward
speed in UniLab MuJoCo playback, consistent with the source environment's
`rough_v1_play_velocity_trace.csv`. The under-trained compact 128-64-32
family (`rough_rotation_stair` 2026-07-30, 1,000 iterations) only reaches
forwardness ≈ 0.63 and is therefore not the canonical training contract;
pass an explicit dims override to load compact-family artifacts (see below).

The rough owner's reward weights match the source rough run term by term,
including two strengthened entries the flat owner does not share:
`track_lin_vel_xy_square` and `track_ang_vel_z_square` are both `-1.0`
(flat uses `-0.1`). These squared-error penalties carry most of the pressure
for precise velocity/yaw-rate tracking on rough terrain; inheriting the flat
`-0.1` lets fine-tuning drift toward spin/orientation terms and drops crossing
forwardness from ≈ 0.93 to ≈ 0.7. The pinned reward graphs of the exact
Rough-v0/v1 owners also fix both entries at `-1.0`.

### Loading an upstream vanilla-PPO checkpoint

The source training repository stores its vanilla-PPO weights in a
`model_*.pt` payload under `model_state_dict`, whereas UniLab's native RSL-RL
checkpoints use `actor_state_dict`. The WheelBipe PPO eval route recognizes the
source schema before materializing the environment, then validates every target
actor/critic key and tensor shape before loading weights for inference.
Optimizer state and the source iteration counter are deliberately not resumed;
continue training from a source checkpoint only after converting it into a
UniLab-native run.

Pass either an absolute checkpoint path or run directory through the public
`--load-run` flag. The direct `algo.load_run=...` Hydra override remains
available, but do not pass both forms:

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

The common upstream architecture `[256, 128, 64]` is the default in both
WheelBipe owner YAMLs. Source runs using `[128, 64, 32]` must select matching
dimensions explicitly:

```bash
uv run eval --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode none --load-run /path/to/model_8000.pt \
  training.play_steps=20 \
  algo.policy.actor_hidden_dims=[128,64,32] \
  algo.policy.critic_hidden_dims=[128,64,32]
```

An architecture, observation, action, or distribution-shape mismatch fails
closed with a diagnostic instead of partially loading a policy. For PPO on
MuJoCo, `--render-mode interactive` routes the public `eval` command to the
dedicated interactive viewer; the same strict adapter handles native
`actor_state_dict`, upstream vanilla-PPO `model_state_dict`, and trusted
35D-to-6D TorchScript artifacts. Other custom algorithms are not implicitly
routed through this PPO viewer.

Enable the migrated WheelBipe keyboard contract with a Hydra override:

```bash
uv run eval --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode interactive --load-run /path/to/model_8000.pt \
  interactive.keyboard=true
```

The MuJoCo viewer provides key-down events only: `W/S` set a latched forward
speed, `A/D` set a latched yaw rate, `Z/X` safely nudge height, and `L` or Enter
resets the command. `Q` queues `info["jump_takeoff_request"]`; the WheelBipe
state machine remains the sole owner that accepts or rejects the jump.

The source release also contains inference-only TorchScript `policy.pt` archives.
The standard PPO `eval` route accepts one of these files directly (use
`--load-run /path/to/policy.pt`); it validates the strict `float` 35D actor
input and 6D action output, then runs numerical playback without constructing a
second network. A TorchScript archive has no optimizer or resume state, and its
serialized code is executable, so only trusted artifacts should be supplied.
When a direct `policy.pt` is used, hidden-dimension overrides are unnecessary;
the target owner still must expose the normal 35-observation/6-action contract.

### Rough-terrain playback

The rough PPO owner has a play-only display profile. It keeps the training
`RM_ROUGH_TERRAINS_CFG` running-terrain scene and terrain-aware command
profiles, assigns the first vectorized environments round-robin across the
generated terrain columns, pins a mid-level difficulty, and sends them along
the positive terrain axis. Airborne reset sampling, interval pushes, external
forces, special command buckets, and random reset yaw are disabled only for
playback so a recording shows the robot approaching the actual steps/slopes
rather than starting in mid-air. The step-up/airborne state machine remains
enabled and is still driven by terrain/contact sensors.

When `--task wheelbipe_v14_rough` is used with a checkpoint whose sidecar was
written by an exact owner such as `WheelbipeV14RoughV1`, PPO playback adopts
that recorded owner before constructing the environment. This prevents a
same-shaped checkpoint from silently running under the legacy compatibility
owner. Keeping `run_config.json` next to copied checkpoints is therefore
recommended.

For example, record 10 seconds while tracking the first generated terrain column
(set `training.cam_tracking_env_idx` to `0`, `4`, `8`, or `12` for other columns):

```bash
MUJOCO_GL=egl uv run eval --algo ppo --task Robotics-Wheelbipe-V14-Rough-v1 \
  --sim mujoco --render-mode record training.play_steps=500 \
  training.play_env_num=16 training.cam_tracking=true \
  training.cam_tracking_extra_envs=0 training.cam_tracking_env_idx=0 \
  --load-run /path/to/model_5000.pt
```

The cliff-crossing showcase matching the source release uses the exact
`Robotics-Wheelbipe-V14-Rough-Play-v0` route: ten rows of
`cliff_inv_stair_slope_short_for_rm_play`, a 5 s episode reset, and a terrain
profile pinning a constant 2.5 m/s forward command. The bowl center of this
terrain is below ground with 0.03 m steps and a +0.3–0.4 m cliff rim; crossing
quality is measured as body-frame forwardness (the signed forward-velocity
share of total speed), and the canonical 256-128-64 source weights reach
forwardness ≈ 0.94:

```bash
MUJOCO_GL=egl uv run eval --algo ppo --task Robotics-Wheelbipe-V14-Rough-Play-v0 \
  --sim mujoco --render-mode record training.play_steps=1500 \
  training.play_env_num=16 training.cam_tracking=true \
  training.cam_tracking_extra_envs=0 training.cam_tracking_env_idx=0 \
  --load-run /path/to/rough_checkpoint.pt
```

The `policy.onnx` exported by training/playback (single `obs`
`float32[1,35]` input, single `actions` `float32[1,6]` output, indices 28–34
always `[1,0,0,0,0,0,0]`) matches the `wheelbipe_ros2_sim2sim` normal-only
policy contract: place the ONNX under that repository's
`src/controllers/template_ros2_controller/policy/parallel/` and point
`WHEELBIPE_RL_MODEL_PATH` at it. The same ROS 2 controller then runs inference
without graph or preprocessing changes; the command/gyro/gravity/joint
scale-clamp parameters are a deployment contract shared across models.

### Recorded artifact execution evidence

The artifact routes below were exercised on both backends. This table records
loader/execution evidence only: earlier two-step mean rewards were removed
after the source observation, action, and active-DR contracts were corrected,
because values from the previous owner semantics would be stale rather than a
benchmark. Rows with different training/config provenance must not be compared
as reward or convergence evidence.

| Artifact and route | MuJoCo | Motrix | Additional check |
| --- | --- | --- | --- |
| pinned upstream `model_8000.pt` through PPO `eval` | strict load and finite rollout | strict load and finite rollout | `model_state_dict` schema accepted without partial loading |
| sibling upstream `policy.pt` through PPO `eval` | trusted TorchScript rollout | trusted TorchScript rollout | actor output equals `model_8000.pt` exactly for the checked input |
| sibling upstream `policy.onnx` through the dedicated helper | ONNX rollout | ONNX rollout | graph shape and numerical actor comparison checked |
| UniLab `wheelbipe_v14_flat_long_4096/model_2700.pt` through PPO `eval` | native-checkpoint rollout/export | native-checkpoint rollout/export | trained before the final source-contract corrections; retained as a provenance-labeled loader smoke, not a migrated-performance result |
| UniLab rough `rr256fixsq` run `model_1999.pt` (4,096 envs × 2,000 iterations, warm-started from source `model_2500.pt`) through PPO `eval` | native-checkpoint rollout, cliff-play recording, and `policy.onnx` export | — (loadability of the same checkpoint covered by the sim2sim contract audit) | cliff-play body-frame forwardness ≈ 0.94 with ≈ 1.0 m/s mean forward speed, consistent with the source environment record; the exported ONNX is `obs[1,35] → actions[1,6]`, matching the `wheelbipe_ros2_sim2sim` contract |

The checkpoint and containing-directory forms of the public flag were both
validated with absolute paths; the exact upstream `*-Play-v0` alias used the
same route. For example:

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

`record` produced valid MP4 containers on both backends from exact-checksum
artifact copies. MuJoCo `interactive` reached the viewer after strict loading
and then failed clearly on this headless host because no display was present;
the Motrix interactive CLI route is covered by command/routing tests, but no
Motrix window was opened in this validation.

### Custom history-policy training and evaluation

Run a short headless custom training smoke with the algorithm selected by the
public CLI. The custom runner exports a history-aware `policy.onnx` beside the
final checkpoint even when playback is disabled:

```bash
uv run train --algo him_ppo --task wheelbipe_v14_flat --sim mujoco \
  algo.num_envs=4 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=true training.log_root=/tmp/wheelbipe-runs
```

For a source-sized canonical `wheelbipe_v14_flat` run, select the matching long
profile. The six exact custom train/Play IDs compose their matching profile
automatically and continue to reject an additional `--profile`; an exact ID
therefore produces the same algorithm config as canonical + the profile below.
The profile is composed as a Hydra group (the `mujoco`/`motrix` owner remains
the same), and a CPU override is useful for a deterministic smoke or a host
without CUDA:

```bash
uv run train --algo him_ppo --task wheelbipe_v14_flat --sim mujoco \
  --profile source_him_long training.device=cpu training.no_play=true
uv run train --algo dreamwaq --task wheelbipe_v14_flat --sim mujoco \
  --profile source_dreamwaq_long training.device=cpu training.no_play=true
uv run train --algo np3o --task wheelbipe_v14_flat --sim mujoco \
  --profile source_np3o_barlow_long training.device=cpu training.no_play=true
```

Custom evaluation loads `model_*.pt`, validates the checkpoint's algorithm,
history, observation, action, and (for NP3O) five-cost metadata, then performs
a numerical rollout. When the selected run contains UniLab's `run_config.json`,
the custom owner automatically adopts its architecture-only metadata (HIM
estimator widths, DreamWaQ CENet widths, or NP3O source-Barlow fields) before
constructing the evaluator; explicit `--profile`/architecture overrides remain
strict and fail closed on a mismatch. Set
`training.auto_load_checkpoint_config=false` to require an explicit matching
owner. Select `--render-mode record` or `interactive` when a backend renderer
is desired:

```bash
uv run eval --algo him_ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode none training.play_steps=20 \
  algo.load_run=/tmp/wheelbipe-runs/WheelbipeV14FlatHIM/<run-directory>
```

To continue custom training, pass the same explicit run directory or checkpoint
path as `algo.load_run` on a `uv run train` command. The custom runner restores
the policy, optimizer/adaptation state, iteration counter, and NP3O channel
schedule before collecting new rollouts; the default `algo.load_run=-1` starts
a fresh run.

#### Source-enabled custom reward curricula

Only the exact HIM training owner enables the pinned `CurriculumCfgV14`; its
Play owner disables the curriculum. At each completed-reset batch, the owner
takes the batch mean of the weighted `track_height_exp` episode sum divided by
`max_episode_length_s`. Matching the source manager's accounting,
`num_steps_per_env=24` expands the configured 64-sample window to `64 × 24`
compute calls and each 500-episode stage minimum to `500 × 24` calls. The two
threshold transitions are:

| Stage | Height reward weights (`exp` / `tight`) | World-frame base +Z assist | Advance condition |
| --- | ---: | ---: | --- |
| 0 | `1.0 / 1.0` | 160 N | completed window mean at least `0.4`, after the stage minimum |
| 1 | `0.8 / 0.6` | 80 N | completed window mean at least `0.4`, after the stage minimum |
| restored defaults | `0.0 / 1.0` | 0 N | terminal curriculum state |

The assist and interval random wrench are not added. The pinned source writes
both into the same external-wrench buffer: construction, reset, or a global
stage advance writes the current 160/80/0 N +Z assist and zero torque; a due
5--10 s random-wrench event overwrites that environment's latch; its next
reset or the next global stage advance writes the assist again. UniLab
preserves that per-environment write order. Re-submitting the latched value
each local step through the public
`SimBackend.apply_body_force`/`apply_body_torque` contract is only the explicit
conversion required by the MuJoCo/Motrix one-step external-force APIs, not an
additive force or an environment call to backend-private state.

There is one explicit frame-conversion boundary. At each assist write, the
source converts world +Z through the body's current quaternion and persists
that body-local vector in the Isaac wrench buffer. UniLab instead latches the
sampled world-frame +Z value and re-submits it through the public upcoming-step
world-frame backend contract. Event values and overwrite order are preserved;
the force-direction evolution caused by body rotation between two writes is
not claimed to be physics-equivalent.

Only the exact NP3O training owner enables the pinned linear height-to-velocity
reward gate; NP3O Play disables it. For absolute height error `e`, the gate is
`1` when `e <= 0.05` m, `0` when `e >= 0.10` m, and
`(0.10 - e) / 0.05` in between. It multiplies the linear-velocity and yaw-rate
tracking terms (including their tight/square variants) before owner reward
weights are applied, so a policy cannot earn full velocity-tracking reward by
sacrificing commanded height. Exact owner validation rejects changes to this
training/Play split or its two thresholds.

The HIM and DreamWaQ load boundary also accepts the complete
`model_state_dict` emitted by the pinned source runner, whose MLP keys include
a `.model.` container segment. Mixed native/source keys and incomplete or
extra graphs fail closed. DreamWaQ's main Adam and
`vae_optimizer_state_dict` are remapped through the source's explicit
parameter order rather than inferred from tensor shapes. Regression tests
obtain a maximum absolute error of `0.0` for CENet, action, and value outputs
against the pinned DreamWaQ class. All 110 `model_*.pt` files in the public
training checkout are normal-PPO graphs, so this custom evidence is a strict
source-shaped regression and pinned-class numerical comparison, not an actual
load of a released custom artifact. UniLab checkpoints preserve source field
aliases but do not reverse-convert `.model.` keys or DreamWaQ Adam ordering;
bidirectional UniLab-to-source-runner resume is therefore not claimed.

The source profile's history reset is also a config and artifact contract: a
reset clears every deque slot and then puts the current frame last, so the
first actor input is `[0, ..., 0, current]`. Done rows, eval, and custom
sim2sim preserve the same oldest-to-newest order. Bounded/legacy graphs that
lack this sidecar field keep the prior repeat-current behavior.

The compact custom actor has a 28D one-step observation and a 32D critic
observation; the runner stacks five frames for HIM/DreamWaQ and ten for the
compact NP3O path. Those compact exports therefore use `[1,140]` or `[1,280]`
(with `[1,6]` actions), and are not compatible with the normal 35D ROS
deployment graph.

The pinned NP3O checkout has an active-graph dimension contradiction that no
released checkpoint can resolve: the live env builds a 351D `on_constraint`
(policy 28 + the actual 43D `priv_latent` + history 280), while the model/runner
constructor declares and normalizes 312D (28 + 4 + 280). The public checkout
contains no custom checkpoint, so a released artifact cannot identify which
side was ultimately intended. UniLab explicitly selects the executable 312D
model contract and takes only the compact critic's 4D privileged tail. This is
a documented compatibility repair, not an exact reproduction of the upstream
live-env graph or a custom checkpoint.

The explicit `source_barlow` NP3O path consequently consumes that repaired
contract: its runner `policy.onnx` uses one `[1,312]` input, while the
source-compatible teacher artifacts
`barlow_twins_actor.pt` and `.onnx` use two inputs, `obs` `[1,28]` and
`obs_hist` `[1,10,28]`, and emit `[1,6]` actions. Each custom export writes a
sidecar beside the graph with canonical algorithm, variant, dimensions,
history reset, and cost-channel metadata. The custom sim2sim helper validates that
sidecar when present (legacy graph-only exports remain shape-checked and
loadable).
The source HIM/DreamWaQ runners additionally export mapping-ABI TorchScript and
ONNX graphs; UniLab currently exports flattened-history ONNX for those two
owners and does not claim export-ABI equivalence. NP3O/Barlow's two-input
TorchScript and ONNX artifacts are the separate contract described above.

To create a deployment artifact from a UniLab checkpoint, leave playback
enabled and select the headless recording mode.  The PPO play lifecycle then
writes `policy.onnx` beside the checkpoint (and still uses the normal runner
and environment contracts):

```bash
uv run train --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  algo.num_envs=4 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=false training.play_render_mode=record \
  training.play_steps=2 training.play_env_num=1 \
  training.log_root=/tmp/wheelbipe-runs
```

For an existing run, export without another training iteration by using the
same play-only lifecycle:

```bash
uv run eval --algo ppo --task wheelbipe_v14_flat --sim mujoco \
  --render-mode record training.play_steps=1 training.play_env_num=1 \
  algo.load_run=/tmp/wheelbipe-runs/WheelbipeV14Flat/<run-directory>
```

The resulting `<run-directory>/policy.onnx` is the model consumed by the
sim2sim helper below. The source delay contract is the canonical default (and
can also be selected explicitly):
5 ms physics steps, four substeps per 20 ms control step, physics-step
observation/action buffers, and high-exclusive lag ranges. The explicit
`local_physics` profile selects UniLab's delay-free 1 ms physics / 20 ms control
timing with inclusive range semantics. A timing profile does not claim source
dynamics, asset, ROS-loop, or sim-to-real parity:

```bash
uv run --no-sync scripts/sim2sim_wheelbipe.py \
  --delay-profile source_v14_physics --steps 200
```

The `obs_delay_cfg`/`act_delay_cfg` ranges and history lengths are recorded in
both backend owner YAMLs.  Observations support `control` or `physics` step
units; actions are delayed from the backend's declared per-physics-step
callback.  The two latency contracts cannot be combined with
`control_config.simulate_action_latency`; reset clears each selected
environment's history and samples a new integer lag.

The owner controller also intersects its configured torque limits with the
materialized backend actuator ranges at initialization. The exact V14 owner
requests 40 N m for each active leg channel and 5 N m for each wheel; the
vendored MJCF independently enforces the same ±5 N m wheel bound. Effective
bounds are exposed in the environment's `torque_contract` diagnostic snapshot.

The owner YAMLs are the source of truth for PPO hyperparameters, reward scales,
domain randomization, and terrain settings:
`conf/ppo/task/wheelbipe_v14_flat/` and
`conf/ppo/task/wheelbipe_v14_rough/`.

### Source domain-randomization contract

The exact source owners preserve the pinned `EventCfgV14` schedule instead of
reducing it to one generic "domain randomization" switch. The detached
`env.domain_randomization_contract` snapshot records the resolved backend
conversions and omissions. The active contract is:

| Schedule | Pinned source event | UniLab materialization and boundary |
| --- | --- | --- |
| Startup mass and COM | base mass × `[0.9, 1.3]`; leg and gimbal body masses × `[0.9, 1.1]`; wheel masses × `[0.9, 1.1]`; base COM x-offset in `[-0.04, 0.04]` m and y/z offsets in `[-0.02, 0.02]` m | Per-environment mass multipliers and base-COM offsets are applied through the public backend reset/materialization contract. Source inertia events are disabled and are not claimed here. |
| Startup body material | base static/dynamic `[0.01, 0.1]`, restitution `[0.02, 0.2]`, 64 buckets; wheels static `[0.5, 1.2]`, dynamic `[0.4, 1.0]`, restitution `[0.02, 0.2]`, 64 buckets; guide static/dynamic `[0.1, 0.7]`, restitution `[0.01, 0.1]`, 8 buckets; consistent samples clamp dynamic friction to static friction | MuJoCo/Motrix expose one sliding-friction channel, so sampled dynamic friction drives that channel; a PhysX static/dynamic/restitution contact law is not fabricated. The public local asset has no `*_guide_link` target on either backend, so guide material is recorded as `not_applicable_missing_target` with a warning. |
| Startup joint friction, additive uniform | active front/rear static and dynamic `[0.25, 1.0]`, viscous `[0.05, 0.2]`; wheel static `[0.05, 0.25]`, viscous `[0, 0.01]`; inactive linkage static `[0.05, 0.1]`, viscous `[0.01, 0.025]`; gimbal static/viscous `[0.002, 0.01]` | Where the source omits a dynamic range, the static range supplies the diagnostic dynamic sample, clamped to static. The sampled static term maps to additive DoF `frictionloss`; the independently sampled dynamic term is diagnostic because both local backends have one Coulomb coefficient. MuJoCo maps viscous friction to DoF damping. Motrix explicitly records `unsupported_omitted` for viscous friction and warns; its Coulomb randomization remains active. |
| Eligible reset actuator gains and effort | global stiffness/damping × `[0.75, 1.25]` with a 720-step minimum reset interval; spring2 stiffness × `0.01` and damping × `[0.5, 1.5]`; leg effort × `[0.8, 1.1]`; wheel effort × `[0.9, 1.1]` | Active leg, wheel, spring2, and gimbal groups are materialized. Spring2's source stiffness baseline is zero, so × `0.01` remains zero; its fixed 50 N s/m damping receives the global multiplier and then the spring multiplier. The source passive-linkage `IdealPD` damping baseline `0.01` also belongs to the global event, but the local asset has no corresponding passive actuator: this is warned and recorded as `unsupported_unmaterialized_actuator`; startup joint friction for those joints remains active. `Rough-Play-v0` disables only the global Kp/Kd event and keeps its inherited spring/effort events. |
| Episode reset | root roll/pitch in `[-0.15, 0.15]` rad and yaw in `[-3.14, 3.14]` rad; independent spring preload in `[-50, 50]` N per side | Pose randomization and preload are resampled through the owner reset contract. The preload is added to the 400--600 N linear spring profile below. |
| Interval disturbance | every 5--10 s, root velocity delta x/y in `[-0.25, 0.25]` m/s with z zero; base world-frame force XYZ in `[-10, 10]` N and torque XYZ in `[-1, 1]` N m | Both MuJoCo and Motrix implement the typed root-velocity, body-force, and body-torque backend contracts; unsupported backends fail instead of dropping torque. On HIM, the random wrench overwrites rather than adds to the vertical-assist latch, as detailed above. |
| V2 gimbal startup | heading-controller Kp in `[20, 40]` and Kd in `[0.05, 0.1]`, uniform | This is a separate startup event for the environment-owned V2 heading controller, not the reset-time articulation-gain event. |

These conversions preserve the published task/config identity, not Isaac
Sim/PhysX contact physics, distributions after solver integration, numerical
trajectories, or convergence. In particular, the single-Coulomb material and
joint-friction mappings, absent guide target, Motrix viscous omission, and
unmaterialized passive `IdealPD` gain are explicit non-parity boundaries.

### Rough-terrain contract

The exact rough owners do not share one generic terrain. `Rough-v0` uses the
source Rotation99 allocation with seven families over ten columns, while
`Rough-v1` uses the running curriculum with eleven families over thirteen
columns. Both Play owners use only
`cliff_inv_stair_slope_short_for_rm_play`, with ten and one rows respectively.
Column counts follow the source proportional-allocation rule, and terrain type
names/ids return from backend materialization as environment metadata. Reset
selects the terrain-command profile from the assigned environment origin;
during rollout it remaps the current root XY to a cell, so crossing a column
immediately selects the new terrain's command constraints. Out-of-grid
coordinates are clamped for metadata lookup.

Rough timeout uses the source whole-grid boundary: half extents come from row
and column counts plus cell size, normal owners include the border, and the
result is reduced by a 0.5 m margin. `Rough-Play-v0` explicitly uses the inner
terrain area and excludes that border. This is not a per-cell boundary, so it
is compatible with the runtime cell crossing above. The source grid-bars are
box meshes; UniLab converts their top surface to the backend-neutral
heightfield contract. The height profile is preserved, but identical triangle
tessellation or contact physics is not claimed.

The source critic height channel always contains world-frame root z;
Flat/Rough-v1 first clips it to `[0.05, 0.45]`. Only the reward path then
subtracts the terrain ground estimate for rough/airborne owners. A
terrain-relative value is never substituted into the 78D critic.
`leg_joint_acc`/`wheel_acc` likewise no longer use an environment-owned
control-step difference: both backends expose `get_dof_acc()` and update its
finite difference at each 5 ms WheelBipe physics substep, clearing it on
reset. This is an explicit MuJoCo/Motrix conversion of Isaac generalized
acceleration, not a numerical-physics equivalence claim.

The Motrix owner YAMLs explicitly enable a cold-path compatibility profile that
disables the six MJCF closed-loop equality constraints.  The current MotrixSim
solver can otherwise fail during this mechanism's contact solve.  MuJoCo keeps
the canonical constraints; therefore a Motrix rollout is a backend smoke and
training target, not evidence of bitwise physics equivalence.

## Policy contract

The bundled deployment graph is a fixed normal-mode policy with one `obs`
input (`float32[1,35]`) and one `actions` output (`float32[1,6]`). The actor
observation is assembled in this order:

```text
command[3], height*5[1], gyro*0.5[3], projected_gravity[3],
leg_position[4], wheel_position[2], leg_velocity*0.1[4],
wheel_velocity*0.1[2], previous_action[6], normal_mode[7]
```

The wheel-position slots are reserved zeros in normal mode and the final mode
vector is `[1, 0, 0, 0, 0, 0, 0]`. PPO's critic group is explicitly fixed at
78 dimensions in both backend owner YAMLs.

The identical 35-field layout does not imply identical preprocessing. Exact
source-training owners clip each raw component before scaling: command to
`[-100, 100]`; height command to `[0, 1]` before ×5; gyro to `[-100, 100]`
before ×0.5; gravity and leg position to `[-100, 100]`; leg and wheel velocity
to `[-200, 200]` before ×0.1; and previous action to `[-100, 100]`. The pinned
clip dictionary has no control-mode entry: its seven-slot tail receives only
the per-slot scale (normal/state owners use ×5 on zero-based slot 5, the sixth
field; V2 uses all ones), then the assembled source actor tensor sanitizes
non-finite values to zero. There is no source post-scale global actor clip. The
generic/ROS deployment builder deliberately keeps the separately published
`scaleClamp` contract: scale first, then globally clamp each resulting field
to `[-100, 100]`.

All 15 pinned exact owners keep the source `debug_value_diagnosis=false`.
Consequently, the raw actor/critic non-finite or maximum-absolute-value
diagnostic does not enable an observation-outlier termination gate. This does
not disable immediate physical numerical safety: non-finite base linear,
base angular, or active-joint velocity still terminates immediately, as does
active-joint velocity above 500 rad/s, base angular velocity above 200 rad/s,
or base linear velocity above 100 m/s. The actor-side `nan_to_num` above is
observation sanitation only; it does not repair the underlying physical state
or replace those safety checks.

Policy action order is
`left_front1`, `left_rear1`, `right_front1`, `right_rear1`, `left_wheel`,
`right_wheel`. The MuJoCo description has eight native actuator slots; the
adapter maps those six public actions to the four leg and two wheel slots and
drives the two spring slots from the owner-level controller. The mapping is
resolved during environment construction, not by probing backend internals in
the step path.

Pinned source runner files set `clip_actions: null`. Exact owners encode that
as `clip_actions=inf`: finite raw policy actions are not pre-clamped and remain
the values stored in observation history and consumed by action-rate and
smoothness rewards. Only decoded physical targets are bounded: the four leg
position targets are clamped to `[-3.14, 3.14]` rad after ×0.5 scaling and
default-pose offset, and the two wheel velocity targets are clamped to
`[-100, 100]` rad/s after ×10 scaling. Thus the unbounded runner action is not
an unbounded actuator command, and it is not replaced by the generic/ROS
safety wrapper.

The spring slots use the V14 Isaac training profile: force is linear in compressed
travel from 400 N to 600 N over 0.07 m, with an episode-level random preload in
[-50, 50] N and fixed 50 N s/m actuator damping. Source domain randomization
first applies the global `[0.75, 1.25]` gain multiplier and then the spring-only
`[0.5, 1.5]` damping multiplier. The source flag that disables an additional
randomized stretch/contract damping term does not disable this fixed actuator
damping. The ROS gas-spring hardware model remains a separate adapter/dynamics
contract.

## Sim-to-sim with a trained or released policy

The checked-in ONNX artifact can be exercised against the UniLab MuJoCo owner:

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

The helper accepts `--sim mujoco` or `--sim motrix`.  Motrix uses the explicit
owner constraint-compatibility profile; it is not applied to the MuJoCo scene.

Use `--delay-profile source_v14_physics` only when the policy was trained or
validated for its 5 ms/4-substep timing profile. The command reports the
selected timing-profile scope explicitly; the broader owner-contract evidence
is listed above.

The script validates the graph I/O and observation dimensions before stepping
the environment. Use `--model` to test another compatible `obs`/`actions`
graph; a graph with a different input or output dimension is rejected.
The normal helper accepts either the one-input ONNX artifact or a trusted
upstream TorchScript `policy.pt`; both are checked for finite 35D observations
and finite six-action outputs. TorchScript inference is kept on CPU and static
batch-one exports are evaluated row by row when a vectorized rollout requests
multiple environments. Do not pass a custom history graph to this helper.

The normal sim2sim helper can stream the source-compatible velocity/reward
schema to CSV, periodically refresh a standalone interactive HTML file, and
optionally show the four live Matplotlib panels:

```bash
uv run scripts/sim2sim_wheelbipe.py --steps 1000 \
  --trace-csv /tmp/wheelbipe-trace.csv \
  --trace-html /tmp/wheelbipe-trace.html --realtime-plot
uv run scripts/export_wheelbipe_trace_html.py /tmp/wheelbipe-trace.csv \
  -o /tmp/wheelbipe-trace-offline.html
```

The live data API is display-neutral (`WheelbipeRealtimeBuffer`); Matplotlib is
imported only when `--realtime-plot` is selected. The current owner records the
selected environment's total reward. Per-term columns are emitted only when an
owner exposes per-environment `info["reward_terms"]`; cadenced/global log means
are deliberately not mislabeled as per-environment rewards.

For a custom history-policy export, use the companion helper so history is
initialized and reset at episode boundaries:

```bash
uv run --no-sync scripts/sim2sim_wheelbipe_custom.py \
  --algorithm him_ppo \
  --model /tmp/wheelbipe-runs/WheelbipeV14FlatHIM/<run-directory>/policy.onnx \
  --sim mujoco --steps 200
```

The helper accepts the source and canonical spellings `him`, `him_ppo`,
`ppo_him`, `dreamwaq`, `dream_waq`, `ppo_dreamwaq`, and `np3o`; NP3O keeps its
five constraint channels in the selected environment owner. It composes the
same source-profiled task/backend owner as exact training, so reward,
curriculum, cost, and Motrix compatibility fields do not fall back to bounded
defaults. The export sidecar selects the compact flattened-history, repaired
312D source-Barlow, or dual-input actor ABI before environment construction.
Do not pass a custom graph to `scripts/sim2sim_wheelbipe.py`, whose contract is
the normal 35D input.

The explicit `source_barlow` NP3O profile also writes
`barlow_twins_actor.pt` and `barlow_twins_actor.onnx`. These are the
source-compatible teacher artifacts from `actor_teacher_backbone`, with two
inputs (`obs` `float32[1,28]` and `obs_hist` `float32[1,10,28]`) and one
`float32[1,6]` `actions` output. They are distinct from the runner's
one-input 312D `policy.onnx`; neither artifact is accepted by the other's
loader contract. The standard NP3O run directory (or `policy.onnx`) executes
the documented repaired stream `[policy28, privileged-tail4, history280]`
directly; this compatibility choice is not a checkpoint-exact claim. Use the
source actor explicitly in custom sim2sim when that ABI is desired (the run
directory or its full `policy.onnx` may be supplied and the sibling is then
resolved). ONNX remains the default sibling; pass
`--source-barlow-format torchscript` to select the `.pt` sibling when both
artifacts are present, or pass `barlow_twins_actor.pt` directly:

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

The source actor rollout uses the upstream zero-filled ten-frame history at
reset and restarts that history for autoreset rows. Its metadata sidecar is
validated before the environment is materialized; this records ABI and
algorithm compatibility only, not source dynamics or convergence parity. The
TorchScript route loads executable code and is therefore restricted to trusted
archives. One-input custom `.pt` checkpoints/graphs are rejected instead of
being sent to the source adapter; use their ONNX export with the normal custom
route.

## ROS 2 deployment paths

The source deployment repository contains a C++ `template_ros2_controller` and
the `template_real_ros2_ctrl::RealBridge`. UniLab keeps two separate paths: an
executable in-process adapter that needs no ROS installation, and an
exact-digest native ROS 2 source bundle that can be materialized into an
external colcon workspace.

### In-process adapter (no ROS required)

The executable owner-layer adapter is
`unilab.training.wheelbipe_ros2.WheelbipeRos2Controller`. It can be driven in
the same process as the UniLab env:

```bash
uv run --no-sync scripts/sim2sim_wheelbipe_ros2.py --sim mujoco --steps 200
uv run --no-sync scripts/sim2sim_wheelbipe_ros2.py --sim motrix --steps 200 \
  --command 0.3 0.0 0.0 --height 0.27
# Use an alternate source-compatible parameter file (parsed once before the
# simulator is materialized).
uv run --no-sync scripts/sim2sim_wheelbipe_ros2.py \
  --config conf/deployment/wheelbipe_v14_ros2.yaml --steps 200
# Inspect the API/protocol boundary without loading ONNX or a simulator.
uv run --no-sync scripts/sim2sim_wheelbipe_ros2.py --print-contract
```

The adapter keeps the source boundary explicit:

| Source contract | UniLab adapter |
| --- | --- |
| eight joints in ros2-control order | `WHEELBIPE_ROS2_JOINT_NAMES` (four legs, two wheels, two springs) |
| `motion_command` `geometry_msgs/msg/Twist` | `set_motion_command(linear_x, angular_z)`; `linear.y` is reserved/zero |
| `height_command` `std_msgs/msg/Float64` | `set_height_command(height)` |
| `state_command` / `current_state` `std_msgs/msg/Int32` | `set_state_command(0..3)` and `WheelbipeRos2ControlOutput.state` |
| `joint_commands` `sensor_msgs/msg/JointState` plus `joint_final_torque` `Float64MultiArray` | `WheelbipeRos2JointCommand` with position/velocity/effort/Kp/Kd/final-torque arrays |
| 500 Hz ros2_control update, 50 Hz ONNX inference | ten 500 Hz adapter updates per 20 ms UniLab env step (held env sample; cadence emulation only) |

The normal graph remains strict `float32[1,35]` (`obs`) to `float32[1,6]`
(`actions`). The Python owner mirrors the source INIT (10 ms hold), IDLE,
PREPARE and RL transitions, command clamping/0.5 s timeout, optional moving
average/low-pass action filter, hardware-PD output modes, finite checks and
safe-stop output. Its source revision and defaults are recorded in
`conf/deployment/wheelbipe_v14_ros2.yaml`.
When `use_dt7=true`, the source real hardware description contributes the
additional `dt7` sensor with `cmd_state`, `cmd_vel_x`, `cmd_omega_z`, and
`cmd_height`; the no-ROS owner accepts those four values through
`WheelbipeRos2Dt7Command` and does not open the hardware transport. An RL state
request without a policy, or an out-of-range DT7 state byte, is ignored using
the source callback's fail-closed behavior.
The CLI's `--config` option selects that cold-path profile; the rollout entry
point requests RL automatically after the 10 ms INIT hold.

This is an in-process simulation/deployment API, not a ROS graph. It does not
publish DDS topics, load the upstream C++ plugin, open `/dev/wheelbipe_h7`, or
provide hardware safety certification. The pure Python RealBridge helpers
`encode_wheelbipe_real_command_packet` and
`decode_wheelbipe_real_state_packet` preserve the audited MIT wire layout
(158-byte command, 143-byte state, header `A8 E6`, trailer `C3 F7`, reflected
CRC16 polynomial `0x8005`). They are parse/encode utilities only; serial
transport and real-time guarantees remain outside this owner. The pure
`WheelbipeRealBridgeGate` models the pinned 1,000 ms reconnect throttle, 100 ms
stale-state inhibition, lifecycle activation, and safe-stop output without
opening a device. A
state packet must pass marker, finite-payload, quaternion-norm and CRC checks
before it is accepted.

### Packaged native ROS 2 source bundle

`deployment/ros2/wheelbipe_v14_native/` packages the pinned MIT source from
`wheelbipe_ros2_sim2sim` revision
`daa34f54d56cab91b3989d8152a7ce7b61092994`: seven colcon packages and 72
source files, three pluginlib descriptors, four launch files, four config
files, two udev rules, and overlays for the checked-in policy plus 21 meshes.
The bundle includes the `controller_manager` launch, the controller,
RealBridge and MuJoCo-system plugins, serial reconnect/stale-state source, and
keyboard/Xbox teleoperation source. Verify or materialize it without importing
ROS:

```bash
uv run scripts/sim2sim_wheelbipe_ros2.py --verify-native-bundle
uv run scripts/sim2sim_wheelbipe_ros2.py --probe-native-runtime
uv run scripts/sim2sim_wheelbipe_ros2.py \
  --materialize-native-workspace /tmp/wheelbipe-v14-colcon
uv run scripts/sim2sim_wheelbipe_ros2.py --require-native-runtime
```

The verified source-tree digest is
`67f0628533f6f8cc849159426fe6769636c304ab4596d34cbfeed4a2f4e53b27`.
Materialization copies 22 asset-overlay files and never builds, sources, or
launches the workspace. On the 2026-08-31 validation host, the probe reported
missing `rclpy`, `launch`, `launch_ros`, `ament_index_python`, `ros2`, `colcon`,
`xacro`, and `rosdep`; `--require-native-runtime` therefore failed closed with
return code 1. ROS 2 Humble, Linux x86-64, ONNX Runtime C++ 1.20.0, MuJoCo
3.5.0, and the remaining manifest dependencies are an external optional
deployment boundary.

Static verification and pure-Python reconnect/stale/teleop tests do not prove
that the workspace compiles or runs in a ROS installation, but on 2026-09-05
the same development host built and headless-launched the external
`wheelbipe_ros2_sim2sim` workspace under the RoboStack conda environment
`wheelbipe_humble` (ROS 2 Humble): the MuJoCo scene loaded,
`joint_state_broadcaster` and `template_ros2_controller` configured and
activated, the FSM completed INIT → IDLE → PREPARE → RL, and
`ros2 topic echo /wheelbipe_V14/joint_states` showed all eight joints evolving
under the policy. The loaded policy is the rough `policy.onnx` exported by this
task (`policy/parallel/V14-rough-unilab-1999.onnx` in that repository,
`[1,35] → [1,6]`). A stale `install/` copy of `wheelbipe_V14.yaml` listed the
joints in a front-interleaved order that violated the controller's joint
contract; a `colcon` rebuild re-synced it with `src/` and the launch succeeded.
`/dev/wheelbipe_h7` was still not opened and no robot hardware was connected,
so no realtime, hardware-safety, or sim-to-real certification is claimed.

## Provenance and licenses

The source repositories and pinned revisions are recorded in the repository
root's `THIRD_PARTY_NOTICES.md`. The redistributed mesh and policy notices are
kept beside their assets:

- `src/unilab/assets/robots/wheelbipe_v14_2/meshes/ASSET_LICENSE.md`
- `src/unilab/assets/robots/wheelbipe_v14_2/mjcf/ASSET_LICENSE.md`
- `src/unilab/assets/policies/wheelbipe_v14/ASSET_LICENSE.md`

The policy file is
`src/unilab/assets/policies/wheelbipe_v14/V14-35-flat-and-rotation-13k.onnx`.
Its SHA-256 is:

```text
a1244761f7ede02f8c80d076d4315a25f014df43df3f7f0d20c2ca5bcd518719
```

The original UniLab environment/configuration code remains under the repository
Apache-2.0 license; SCUTRobotLab's copied assets remain under their MIT notice.
The history-algorithm files under `src/unilab/algos/torch/him_ppo/` retain
HIMLoco's CC BY-NC-SA 4.0 attribution (with separately identified BSD
portions), and the DreamWaQ/NP3O adaptation boundary has no blanket
Apache/commercial-use grant. See the root `THIRD_PARTY_NOTICES.md` before
redistributing algorithm code.

## Current boundary

This migration covers the normal 35D/6D policy path, the named state-machine
and gimbal variants, and the flat history-policy routes above, with
UniLab's MuJoCo and Motrix owners. The Motrix
constraint-free profile is an explicit solver
compatibility boundary, so it does not claim cross-backend physics or
sim-to-real equivalence. The no-ROS Python adapter is executable, and the
pinned native C++ sources are packaged and materializable, but neither is
evidence that a ROS graph or real robot has been validated. Historical source
TensorBoard/event logs are not imported into UniLab run history; the Isaac
Sim/PhysX runtime is not migrated as a backend; and no unpublished STM32 H7
lower-board firmware is included. The
bundled normal graph is the flat-and-rotation example; a rough-terrain
deployment should be produced from a separately trained and checked policy
rather than silently reusing it. No convergence, source learning-curve,
bitwise-physics, realtime, or hardware-safety equivalence is claimed.
