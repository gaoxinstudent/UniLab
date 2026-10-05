# 10.4 closed-chain MJCF

Generated from `/home/gx/Downloads/10.4/urdf/10.4.urdf` on 2026-10-05, following the `real68` asset layout.

- `10_4.xml`: robot-only MJCF with CAD meshes, six main leg/wheel actuators, extra guide rollers, gas-spring bodies, and two passive force actuators. The four active leg motors (`rear1`/`front1`, left and right) now expose ±80 N·m control/force ranges; the original ±20 N·m range was insufficient against the added gas-spring load.
- `scene_flat.xml`: MuJoCo flat-floor scene with contacts, IMU/joint sensors, and a `home` keyframe.
- `scene_fixed_base.xml`: observation-only scene; the chassis is welded to the world at `z=0.8 m` so the closed-chain motion can be inspected without the robot falling. It also adds four zero-centered delta-angle controls (`interactive_*_delta`) for dragging active leg joints in MuJoCo’s **Control** panel. Their force saturation is ±120 N·m so the position controls can overcome the gas-spring and loop reaction torque; pause before editing and leave the six inherited motor controls at zero.
- `locomotion_task.xml`: task-level contacts/sensors/keyframe fragment, following the `real68` layout; use it through a scene/task include rather than loading it as a standalone MuJoCo model.

The two gas springs are represented by their revolute + prismatic joints plus equal force actuators on the prismatic coordinates. The supplied force plot was reduced to the same left/right average curve for both legs: 175 N at the lower end of the exported 80 mm slide range and 231.5 N at the upper end. This is an endpoint-based linear approximation, not a replacement for the complete manufacturer curve. The prismatic damping remains 0.15 N·s/m; it is intentionally modest so it does not mask closed-chain behavior.

The four-bar loops are closed with MuJoCo `equality/connect` constraints using the same topology as `real68`: rear1/rear2 are the hip and wheel-carrier links, front1 is the active calf link, and front2/front3/front4 are the passive loop links.

The exported `scene_flat.xml` was compiled and stepped for 200 MuJoCo steps on 2026-10-05; the model has `nq=39`, `nv=38`, `nu=8`, and four loop-closure constraints. `scene_fixed_base.xml` has `nu=12` (six torque motors, two gas-spring force actuators, and four inspection controls), adds one world-to-chassis weld constraint, and was stepped successfully for 500 steps.

## Interactive closed-chain inspection

Launch the fixed-base scene with the MuJoCo 3.9 simulator:

```bash
/home/gx/Downloads/mujoco-3.9.0-linux-x86_64/mujoco-3.9.0/bin/simulate \
  src/unilab/assets/robots/10_4/scene_fixed_base.xml
```

In the simulator, pause the rollout and adjust `interactive_left_rear1_delta`, `interactive_left_front1_delta`, `interactive_right_rear1_delta`, or `interactive_right_front1_delta` in the **Control** panel. These sliders are angle offsets in radians from the CAD home pose, with a ±2π range; `ctrl=0` is the home pose and avoids an artificial startup pull toward absolute angle zero. To extend/retract one leg, change its `rear1` and `front1` sliders together. Positive coordinated offsets are the same sign on the left; because the right-side hinge axes are mirrored, use the opposite sign on the right for the same physical direction. The passive four-bar links follow through the equality constraints. The original `*_actuator` entries are direct torque motors with ±80 N·m ranges; keep their controls at zero while using the delta controls, or use the motor pair directly with the same left/right sign convention.
