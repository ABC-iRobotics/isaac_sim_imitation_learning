# Isaac Sim Imitation Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![repo size](https://img.shields.io/github/repo-size/ABC-iRobotics/isaac_sim_imitation_learning)
![GitHub Repo stars](https://img.shields.io/github/stars/ABC-iRobotics/isaac_sim_imitation_learning)
![GitHub forks](https://img.shields.io/github/forks/ABC-iRobotics/isaac_sim_imitation_learning)

## Introduction
Imitation learning framework for the **TM5-900** collaborative robot arm equipped with the **OnRobot RG6** gripper. The package interfaces **Isaac Sim 4.5** with *ROS 2 Humble* and can be used as an expert demonstration generator.

The resulting demonstrations are saved in the [LeRobot dataset format](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format).

## Features

- [Ubuntu 22.04 PC](https://ubuntu.com/certified/laptops?q=&limit=20&category=Laptop&vendor=Dell&vendor=HP&vendor=Lenovo&release=22.04+LTS)
- [ROS 2 Humble (Python3)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [Isaac Sim 4.5](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format)
- SceneHandler node using [Omniverse](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/Modules.html) libraries
- AnalyticSolver node using ground truth information to solve the randomized task
- TrajectoryRecorder node records robot state - action pairs

## Prerequisites

- [MoveIt 2](https://moveit.picknik.ai/main/index.html)
- [TMFlow 2.2](https://www.tm-robot.com/en/tmflow)
- [tm_drive](https://github.com/TechmanRobotInc/tmr_ros2)
- [Isaac Sim 4.5](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [onrobot-ros2](https://github.com/ABC-iRobotics/onrobot-ros2)
- [tm5-900_rg6_moveit_config](https://github.com/ABC-iRobotics/tm5-900_rg6_moveit_config)

## Setup guide

Using the [previous](#prerequisites) section's links, install every prerequisites.

Navigate into your ROS 2 workspace and copy this repo to your source folder
```
cd src && git clone https://github.com/ABC-iRobotics/isaac_sim_imitation_learning.git && cd ..
```

Build the package (and installing python prerequisites)
```
colcon build --packages-select isaac_sim_msgs isaac_sim_scene_handler analytic_solver
```

## Usage

After installation use the TMFlow to make a simulated robot. Set up the [Ethernet Slave](https://github.com/TechmanRobotInc/TM_Export) settings.

The expert demonstration generation demo can be initialized with the included launch file.

```
ros2 launch isaac_sim_imitation_learning bringup_robot_scene.launch.py robot_ip:=<robot_ip>
```
where *robot_ip* is the simulated robot's ip address.

Using a separate terminal the demonstration generation can be called via a ROS 2 service.

```
ros2 service call /TrajectoryRecorder/GetTrajectory isaac_sim_msgs/srv/Demonstration "{amount:<num_of_demos>, path: '<save_path>'}"
```

where *num_of_demos* is the number of requested successfull demonstrations and *save_path* is the absolute path ("~" can be used) where the completed demonstrations are saved.

## Troubleshooting

In case of any issues, check the official resources:
- [OnRobot RG6](https://onrobot.com/en/products/rg6-finger-gripper)
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [TMFlow](https://www.tm-robot.com/en/tmflow)

## Author

[Makány András](https://github.com/andras-makany)  - Graduate student at Obuda University

## License

This software is released under the MIT License, see [LICENSE](./LICENSE).