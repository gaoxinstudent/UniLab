# HIM-PPO

HIM-PPO has its own config group and script. The entrypoint is
`scripts/train_him_ppo.py`, the base config is `conf/ppo_him/config.yaml`, and
the committed task owner is `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`.

## Current Entrypoint

The legacy Go2 arm HIM-PPO path is implemented by `scripts/train_him_ppo.py`
and remains a script-level route. WheelBipe's migrated compact HIM owner is a
separate flat-task route exposed as
`uv run train --algo him_ppo --task wheelbipe_v14_flat --sim <backend>` and
implemented by `scripts/train_custom_ppo.py`. See {doc}`../4-tasks/5-wheelbipe_v14`
for its history and checkpoint contract.

## Owner Details

The Go2 arm owner fills the required history dimensions from the base config:

- `algo.num_one_step_obs=76`
- `algo.num_actor_history=5`
- `algo.num_critic_history=1`
- `training.task_name=Go2ArmManipLoco`

Playback uses the same HIM-PPO implementation entrypoint once a checkpoint is
available. Keep user-facing PPO examples on the supported top-level CLI shape;
use the HIM-PPO script path only when debugging that specialized stack.

Neither path is the default PPO route. Use the legacy path for the Go2 arm
manip-loco owner and the custom route only for the WheelBipe flat owner.
