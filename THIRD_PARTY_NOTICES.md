# Third-party notices

UniLab's original source code is licensed under the Apache License 2.0 (see
[`LICENSE`](LICENSE)). This file records the provenance and separate license
terms for the WheelBipe artifacts added by the migration. A third-party notice
does not change the license of either the original UniLab code or an upstream
artifact.

## SCUTRobotLab source repositories

The migration was audited against these public repositories and their pinned
heads:

| Repository | Pinned commit | License |
| --- | --- | --- |
| [`scutrobotlab/wheeled-legged_RL`](https://github.com/scutrobotlab/wheeled-legged_RL) | [`b8ff79f3df855faf9dc92f4a282bd80c42649466`](https://github.com/scutrobotlab/wheeled-legged_RL/tree/b8ff79f3df855faf9dc92f4a282bd80c42649466) | MIT |
| [`scutrobotlab/wheelbipe_ros2_sim2sim`](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim) | [`daa34f54d56cab91b3989d8152a7ce7b61092994`](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/tree/daa34f54d56cab91b3989d8152a7ce7b61092994) | MIT |

The source checkouts are kept in the local, git-ignored `third-party/` directory
for auditability; they are not imported at runtime and are not part of the
UniLab distribution.

The accompanying RoboMaster technical report,
[*Wheel-legged infantry reinforcement-learning motion-control training and
deployment framework*](https://bbs.robomaster.com/article/1942489?source=1),
was used to define the public migration boundary: V14 flat/rough spin tasks,
PPO/DreamWaQ/HIMLoco/NP3O examples, closed-chain and gas-spring dynamics,
observation/action delay, and the ROS 2 MuJoCo/real-hardware upper-computer
path. The report itself is published under CC BY-NC-SA 4.0. No report text or
media is redistributed here; repository code/assets retain the licenses stated
below.

## Redistributed WheelBipe artifacts

- The 21 STL files under
  `src/unilab/assets/robots/wheelbipe_v14_2/meshes/` are SCUTRobotLab assets
  released under MIT. The complete notice is beside the files in
  `src/unilab/assets/robots/wheelbipe_v14_2/meshes/ASSET_LICENSE.md`.
- The robot description and flat scene under
  `src/unilab/assets/robots/wheelbipe_v14_2/mjcf/` are derived from the
  SCUTRobotLab ROS 2 description and remain MIT-licensed. The local notice and
  list of UniLab adaptations are in
  `src/unilab/assets/robots/wheelbipe_v14_2/mjcf/ASSET_LICENSE.md`.
- The ONNX deployment artifact
  `src/unilab/assets/policies/wheelbipe_v14/V14-35-flat-and-rotation-13k.onnx`
  is released by SCUTRobotLab under MIT. Its full notice, source paths and
  digest are in `src/unilab/assets/policies/wheelbipe_v14/ASSET_LICENSE.md`.
  The training source path is
  `pretrained/26_infantry/flat_and_rotation/2026-07-19_09-14-50/exported/2026-07-19_09-14-50_13k.onnx`
  at the pinned `wheeled-legged_RL` revision.
- UniLab-only environment adapters, owner configs, and the task-level keyframe
  fragment (`locomotion_task.xml`) are integration work in this repository and
  remain covered by the repository's Apache-2.0 terms. The robot description is
  kept separate from that keyframe fragment to preserve UniLab's asset
  contract. **This statement does not cover the algorithm adaptations listed
  below.**
- `src/unilab/terrains/source_wheelbipe.py` adapts the MIT-licensed grid-bars
  and cliff inverted-stairs formulas from
  `source/agent_world/agent_world/terrains/height_field.py` at the pinned
  `wheeled-legged_RL` revision. The source grid bars are box meshes; UniLab
  converts their top surface to its backend-neutral heightfield contract and
  therefore does not claim identical triangle tessellation or contact physics.
- `src/unilab/visualization/wheelbipe_keyboard.py` and
  `src/unilab/visualization/wheelbipe_trace.py` adapt the pinned SCUTRobotLab
  keyboard, realtime plotter, and velocity-trace exporter. Their file headers
  preserve the upstream MIT notice and author attribution; the thin
  `scripts/export_wheelbipe_trace_html.py` entrypoint carries the same notice.
- `deployment/ros2/wheelbipe_v14_native/` redistributes the pinned MIT-licensed
  native ROS 2 source bundle from `wheelbipe_ros2_sim2sim` revision
  `daa34f54d56cab91b3989d8152a7ce7b61092994`. Its manifest records source
  paths, per-file digests, the plugin/launch/config inventory, optional native
  dependencies, udev rules, and policy/mesh overlays. Bundle verification or
  workspace materialization does not build, source, or launch ROS and does not
  open a serial or input device.

The ONNX file is intentionally pinned by content, not only by filename:

```text
SHA-256  a1244761f7ede02f8c80d076d4315a25f014df43df3f7f0d20c2ca5bcd518719
I/O      input "obs" float32[1,35] -> output "actions" float32[1,6]
```

Other runtime dependencies (MuJoCo, MotrixSim, PyTorch, ONNX Runtime, and
Hydra) retain their upstream licenses and are installed through UniLab's
dependency configuration; this notice does not relicense them.

## Algorithm adaptations and license boundaries

The migration includes a small history-policy runtime in
`src/unilab/algos/torch/him_ppo/`. Its `actor_critic.py`, `algorithm.py`,
`estimator.py`, `runner.py`, and `storage.py` files identify themselves as
adaptations of HIMLoco in their file headers. HIMLoco is credited to Junfeng
Long and Zirui Wang and is released under **CC BY-NC-SA 4.0**. Those portions
must retain the attribution, non-commercial restriction, and share-alike
terms; they are not Apache-2.0 code and this repository's Apache-2.0 license
does not relicense them. The BSD-3-Clause SPDX marker in the adapted files
describes the RSL-RL portions only and must be read together with this notice.

The `src/unilab/algos/torch/custom_ppo/` package contains UniLab
integration code for the upstream DreamWaQ and NP3O profiles and imports the
HIM adaptation above. DreamWaQ's source repository and ddt_rl_isaacgym (the
NP3O reference) do not provide a top-level license in the audited revisions.
Consequently, no blanket Apache-2.0 or commercial-use grant is claimed for
code that is copied or substantially adapted from those projects. Before
redistributing a derivative of those algorithm portions, preserve the source
attribution and obtain a license clarification from the respective authors.
The custom package's original owner/configuration glue remains Apache-2.0 only
to the extent that it is separable from those third-party algorithm portions.

## Items outside the migration

Historical upstream TensorBoard/event logs and training-run histories are not
redistributed or imported into UniLab's run history. The local git-ignored
source checkout may contain public pretrained artifacts used for audit and
execution checks, but it is not part of the distribution.

Isaac Sim/Isaac Lab/PhysX is not vendored or represented as a UniLab runtime
backend by this migration. The public task contracts are implemented through
the MuJoCo and Motrix owners, without a claim of bitwise physics, reward-curve,
or convergence equivalence. No unpublished STM32 H7 lower-controller firmware
or source is included. The packaged RealBridge source, serial packet contract,
udev rules, and reconnect/stale-state tests do not replace that firmware and
do not establish realtime behavior, hardware safety, or a validated real-robot
deployment.
