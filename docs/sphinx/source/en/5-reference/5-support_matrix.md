# Support Matrix

This matrix is generated conceptually from registry entries, owner YAMLs, and
tests. The generator implementation is `src/unilab/utils/support_matrix.py`; the
write target for the generated block is currently the Chinese reference page
`docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md`.

The generated reference there is the canonical backend × task table (including
`mjwarp` and the custom WheelBipe rows): {doc}`/zh_CN/5-reference/5-support_matrix`.

## Backend Selection Rules

- The default backend is `mujoco`.
- Switch to Motrix with `--sim motrix` on the unified CLI.
- `--algo`, `--task`, and `--sim` jointly select the owner YAML.
- Do not treat `training.sim_backend` as a standalone backend switch.

## Playback Differences

- `mujoco`: `--render-mode auto` exports `play_video.mp4`.
- `motrix`: `--render-mode auto` opens an interactive renderer window; it does
  not record a video and is not bound by `play_steps`.
- `mjwarp`: only explicit, finite-step `record` is supported; playback reuses
  the MuJoCo offline renderer and does not support `auto`, `interactive`, or a
  native renderer.
- `--render-mode record`: all three backends record a video only.
- `--render-mode none`: no renderer playback. Entrypoints that explicitly
  implement headless numerical evaluation (for example WheelBipe PPO and the
  custom history-policy routes) may still run a finite rollout.

## Evidence Grades

| Grade | Repository Evidence |
| --- | --- |
| `Registered` | The env/backend pair appears after `registry.ensure_registries()`. |
| `Configured` | A matching owner YAML exists under `conf/ppo/task`, `conf/appo/task`, `conf/offpolicy/task`, or the dedicated `conf/custom_ppo/task` group. |
| `Tested` | Checked-in tests cover the entrypoint/task-owner/backend combination (including the repository's config/registry coverage), or an explicit maintainer validation record exists; this grade does not imply a default recommendation or full dynamics parity. |
| `Benchmarked` | A checked-in benchmark manifest exists for the combination. |
| `Recommended` | Explicit recommendation metadata exists in the repo. |

The current generator reports no checked-in benchmark manifest and no separate
recommendation metadata, so rows do not auto-promote to `Benchmarked` or
`Recommended`.

The generic generator leaves the three custom WheelBipe rows at `Configured`
because it does not grade their dedicated runner and artifact suites. This is
not an “unported” status: checkpoint/resume/export, history-aware evaluation,
source-Barlow artifacts, and MuJoCo/Motrix sim-to-sim routes have focused tests
and are indexed in the evidence matrix in
{doc}`../2-user_guide/4-tasks/5-wheelbipe_v14`. They are flat-only, use
`scripts/train_custom_ppo.py`, and have no `mjwarp` owner. The task-owner slugs
shown below are internal config-group names; the public CLI uses
`--task wheelbipe_v14_flat` together with the selected `--algo`.

## Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix |
| --- | --- | --- | --- |
| PPO (torch) | `go1_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2_joystick_rough` | Tested | Tested |
| PPO (torch) | `g1_walk_flat` | Tested | Tested |
| PPO (torch) | `g1_motion_tracking` | Tested | Tested |
| PPO (torch) | `g1_flip_tracking` | Tested | Tested |
| PPO (torch) | `g1_wall_flip_tracking` | Tested | Tested |
| PPO (torch) | `allegro_inhand` | Tested | Tested |
| PPO (torch) | `sharpa_inhand` | Tested | Tested |
| PPO (torch) | `sharpa_inhand_grasp` | Tested | Tested |
| PPO (torch) | `allegro_inhand_grasp` | Tested | Tested |
| PPO (torch) | `g1_box_tracking` | Tested | Tested |
| PPO (torch) | `g1_climb_tracking` | Tested | Tested |
| PPO (torch) | `g1_motion_tracking_deploy` | Tested | Registered |
| PPO (torch) | `go1_joystick_rough` | Tested | Tested |
| PPO (torch) | `go2_arm_manip_loco` | Tested | - |
| PPO (torch) | `go2_footstand` | Tested | - |
| PPO (torch) | `go2w_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2w_joystick_rough` | Tested | Tested |
| APPO (torch) | `go1_joystick_flat` | Tested | Registered |
| APPO (torch) | `go2_joystick_flat` | Tested | Registered |
| APPO (torch) | `g1_walk_flat` | Tested | Registered |
| APPO (torch) | `g1_motion_tracking` | Tested | Tested |
| APPO (torch) | `g1_flip_tracking` | Tested | Tested |
| APPO (torch) | `g1_wall_flip_tracking` | Tested | Tested |
| APPO (torch) | `allegro_inhand` | Tested | Tested |
| APPO (torch) | `sharpa_inhand` | Tested | Registered |
| APPO (torch) | `g1_climb_tracking` | Tested | Tested |
| SAC (torch) | `g1_walk_flat` | Tested | Tested |
| SAC (torch) | `g1_walk_rough` | Tested | Tested |
| SAC (torch) | `g1_motion_tracking` | Tested | Tested |
| SAC (torch) | `g1_wbt_obs` | Tested | Registered |
| TD3 (torch) | `go1_joystick_flat` | Registered | Tested |
| TD3 (torch) | `go2_joystick_flat` | Registered | Tested |
| TD3 (torch) | `g1_walk_flat` | Tested | Registered |
| FlashSAC (torch) | `go2_joystick_flat` | Tested | Registered |
| FlashSAC (torch) | `g1_walk_flat` | Tested | Registered |
| HIM-PPO (custom) | `wheelbipe_v14_flat_him` | Configured | Configured |
| DreamWaQ (custom) | `wheelbipe_v14_flat_dreamwaq` | Configured | Configured |
| NP3O + Barlow (custom) | `wheelbipe_v14_flat_np3o` | Configured | Configured |

## Source Index

- Registry bootstrap: `src/unilab/envs/**` registrations via
  `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`,
  `conf/offpolicy/task/**`, and the dedicated custom owners under
  `conf/custom_ppo/task/**`.
- Generic compose coverage:
  `tests/config/test_config_system.py::test_supported_task_composes`.
