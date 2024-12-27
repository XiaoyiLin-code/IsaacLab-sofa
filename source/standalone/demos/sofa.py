"""
This script demonstrates how to simulate bipedal robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/bipeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
import math
import torch

from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

SOFA_HAND = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/lightcone/workspace/SOFA/hhd1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, 0.0, 1),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=10,
            friction=0.5,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np

class FingerMLP(nn.Module):
    def __init__(self, input_dim=16, output_dim=16, start=0):
        super(FingerMLP, self).__init__()
        upper=[0.05]*9+[0.3]*7+[0.05]*9
        self.lower_joint_limits = torch.tensor([0] * 25).to("cuda:0")
        self.upper_joint_limits = torch.tensor(upper).to("cuda:0")

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.tanh1 = nn.Tanh()

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.tanh3 = nn.Tanh()

        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.tanh4 = nn.Tanh()

        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.tanh5 = nn.Tanh()

        self.fc6 = nn.Linear(128, output_dim)
        self.tanh6 = nn.Tanh()

        # Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.tanh1(self.bn1(self.fc1(x)))
        residual = x
        x = self.tanh2(self.bn2(self.fc2(x)))
        x = self.tanh3(self.bn3(self.fc3(x)))
        x = x + residual  # Residual connection

        x = self.tanh4(self.bn4(self.fc4(x)))
        x = self.tanh5(self.bn5(self.fc5(x)))
        x = self.tanh6(self.fc6(x))

        x = 0.5 * (x + 1.0)  # Scale tanh output from [-1, 1] to [0, 1]
        x = x * (self.upper_joint_limits - self.lower_joint_limits) + self.lower_joint_limits
        return x


def main():
    """Main function."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mlps = {key: FingerMLP(1, 25).to(device) for key in ['ff','mf','th','rf','lf']}
    try:
        for key, mlp in mlps.items():
            mlp.load_state_dict(torch.load("/home/lightcone/workspace/SOFA/mlp/train/models/mlp_model_81.pth"))
        print("Model loaded successfully.")
    except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)





    # Load Kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=0.1))
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robot
    h1 = Articulation(SOFA_HAND.replace(prim_path="/World/H1"))
    robots = [h1]

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # 定义目标的关节顺序
    target_joint_order = [f"ff_joint{i}"for i in range(1,26)]+[f"mf_joint{i}"for i in range(1,26)]+[f"th_joint{i}"for i in range(1,26)]+[f"rf_joint{i}"for i in range(1,26)]+[f"lf_joint{i}"for i in range(1,26)]
    target_joint_order = {name: i for i, name in enumerate(target_joint_order)}
    while simulation_app.is_running():
        sim.step()
        sim_time += sim_dt
        count += 1
        for robot in robots:
            joint_poses = []  # 用于存储每个 MLP 的输出关节位置
            
            # 获取机器人当前的关节数据和关节名称
            current_joint_positions = robot.data.joint_pos  # 实时关节状态
            current_joint_names = robot.data.joint_names  # 机器人当前的关节名称列表
            print("Current joint positions:", current_joint_names)
            # 遍历每个 MLP
            for i, (key, mlp) in enumerate(mlps.items()):
                mlp.eval()  # 切换到评估模式
                if(count%100<=50):
                # 假设每个机器人需要不同的输入特征，这里可以设置实际的输入逻辑
                    input_tensor = np.array([50], dtype=np.float32)  # 固定输入值为 50（或其他需要的值）
                elif(count%100>50):
                    input_tensor = np.array([0], dtype=np.float32)
                # 使用 MLP 模型预测关节角度
                joint_pose = mlp(torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device))
                joint_poses.append(joint_pose)

            # 将所有 MLP 的输出关节位置拼接在一起
            joint_pos = torch.cat(joint_poses, dim=1).squeeze(0)  # 沿着正确的维度拼接并移除 batch 维度

            
            # 根据 target_joint_order 调整 joint_pos 的顺序
            # 建立一个映射表，用于将预测值重新排序
            current_joint_order_to_index = {name: i for i, name in enumerate(current_joint_names)}
            sorted_indices = [target_joint_order[name] for name in current_joint_names if name in target_joint_order]
            print("Sorted indices:", sorted_indices)
            # 根据排序索引调整关节位置
            joint_pos_np = joint_pos.cpu().detach()
            print("Original joint positions:", joint_pos_np)
            sorted_joint_pos_np = joint_pos_np[sorted_indices]  # 按照目标顺序排列的关节位置
            print("Sorted joint positions:", sorted_joint_pos_np)
            # 写入机器人

            robot.set_joint_position_target(sorted_joint_pos_np)  # 设置目标关节位置
            robot.write_data_to_sim()  # 写入仿真数据
            
            # 更新机器人仿真状态
            robot.update(sim_dt)
            
            # 获取最新的机器人关节状态
            updated_joint_positions = robot.data.joint_pos  # 实时获取关节数据
            print("Updated joint positions after control:", updated_joint_positions)
            
            # 对比 input_tensor 和机器人最新关节状态
            comparison = {
                "input_tensor": input_tensor.tolist(),
                "predicted_joint_positions": sorted_joint_pos_np.tolist(),
                "actual_joint_positions": updated_joint_positions.tolist(),
            }
            print("Comparison at timestep:", comparison)

        # 备注：
        # 1. `robot.get_joint_positions()` 假设存在此方法用于读取机器人实时的关节状态数据。
        # 2. 如果需要记录所有时间步的数据，可以将 `comparison` 保存到一个列表或文件中。
        # 3. 仿真时间步结束后，你可以对比不同时间步的输入和输出之间的关系进行分析。


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
