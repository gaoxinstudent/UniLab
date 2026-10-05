#!/usr/bin/env python3
"""
测试924机器人在MuJoCo中使用Isaac Sim训练的ONNX模型
"""
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

# 加载MuJoCo场景
scene_path = "src/unilab/assets/robots/wheelbipe_924/scene_flat.xml"
print(f"加载MuJoCo场景: {scene_path}")
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

# 重置到home keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
print(f"✓ 初始化到home keyframe")

# 加载ONNX策略
onnx_path = "/home/gx/Desktop/924_onnx/policy.onnx"
print(f"加载ONNX模型: {onnx_path}")
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"  输入: {input_name}, 输出: {output_name}")

# 获取关节索引（按Isaac Sim的执行器顺序）
# legs_act: left_rear1, right_rear1, left_front1, right_front1 (4个)
# wheel: left_wheel, right_wheel (2个)
actuated_joints = [
    'left_rear1_joint',
    'right_rear1_joint',
    'left_front1_joint',
    'right_front1_joint',
    'left_wheel_joint',
    'right_wheel_joint'
]
joint_indices = []
for jname in actuated_joints:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    joint_indices.append(jid)
joint_indices = np.array(joint_indices)

print(f"✓ 关节映射完成: {len(joint_indices)}个关节")
print(f"  腿关节(4): {actuated_joints[:4]}")
print(f"  轮子(2): {actuated_joints[4:]}")

# 观测维度
obs_dim = 35

# action缓冲区
last_action = np.zeros(6, dtype=np.float32)

# 命令（前进速度）
cmd_vx = 0.5
cmd_vy = 0.0
cmd_wz = 0.0

print("\n开始仿真...")
print("命令: vx=0.5 m/s")

def get_observation():
    """构建观测向量（35维）"""
    # Base姿态 (4) - 注意MuJoCo是wxyz，Isaac Sim可能不同
    base_quat = data.qpos[3:7]  # w,x,y,z

    # Base角速度 (3)
    base_gyro = data.qvel[3:6]

    # 关节位置 (6)
    joint_pos = data.qpos[joint_indices + 7]  # +7跳过freejoint

    # 关节速度 (6)
    joint_vel = data.qvel[joint_indices + 6]  # +6跳过freejoint

    # 命令 (3)
    commands = np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32)

    # 上次动作 (6)
    actions = last_action

    # 基座线速度投影 (3) - 在base frame中
    base_lin_vel = data.qvel[:3]

    # 组装观测 (35维)
    obs = np.concatenate([
        base_quat,        # 4
        base_gyro,        # 3
        joint_pos,        # 6
        joint_vel,        # 6
        commands,         # 3
        actions,          # 6
        base_lin_vel,     # 3
        np.zeros(4),      # padding到35维
    ]).astype(np.float32)

    return obs[:obs_dim]

# 调试标志
debug_print = True

# 启动可视化
with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0
    max_steps = 5000  # 100秒

    while viewer.is_running() and step_count < max_steps:
        # 获取当前观测
        current_obs = get_observation()

        # ONNX推理（直接使用当前观测）
        action = session.run([output_name], {input_name: current_obs[None, :]})[0][0]
        last_action = action.copy()

        # 调试输出
        if debug_print and step_count < 5:
            print(f"\n步数 {step_count}:")
            print(f"  obs前10维: {current_obs[:10]}")
            print(f"  action原始: {action}")
            print(f"  joint_pos: {data.qpos[joint_indices + 7]}")

        # 应用动作到关节控制器
        # 根据配置，动作应该是对4个腿关节 + 2个轮子的控制
        # action[0:2] -> rear1 (left, right)
        # action[2:4] -> front1 (left, right)
        # action[4:6] -> wheel (left, right)
        # 但实际上应该映射到执行器顺序
        data.ctrl[:6] = action

        # 步进仿真
        mujoco.mj_step(model, data)
        step_count += 1

        # 更新可视化（50Hz）
        if step_count % 20 == 0:
            viewer.sync()

            # 打印状态
            if step_count % 1000 == 0:
                base_pos = data.qpos[:3]
                base_vel = data.qvel[:3]
                print(f"步数: {step_count}, 位置: [{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f}], "
                      f"速度: [{base_vel[0]:.2f}, {base_vel[1]:.2f}, {base_vel[2]:.2f}]")

print(f"\n仿真结束，总步数: {step_count}")
