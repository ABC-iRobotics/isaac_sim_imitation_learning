import os
import sys
from collections import OrderedDict
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from PIL import Image as CVImage
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger

from isaac_sim_msgs.srv import CollisionRequest, TubeParameter

try:
    sys.path.append(get_package_share_directory("pymoveit2"))  # Install
except PackageNotFoundError:
    # Symbolic link
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pymoveit2.gripper_interface import GripperInterface
from pymoveit2.moveit2 import MoveIt2


class IL_Gym_Env(gym.Env):
    def __init__(self, node: Node):
        if node is None:
            print("Cannot make this environment without a ROS 2 node.")
            rclpy.Node.get_logger().info("This environment cannot be made without a ROS 2 node.")
            return None

        self._node = node
        self.logger = self._node.get_logger()
        self.fault = False
        self._reached_target = False
        self._success_buffer = 0

        self.bridge = CvBridge()

        config_path = os.path.join(get_package_share_directory("imitation_learning"), "config")

        with open(config_path + "/config.yaml") as f:
            self.config = yaml.safe_load(f)

        self._robot_joints: list[str] = [field["name"] for field in self.config["action"]["joints"]]

        self._gripper_joints: list[list[str]] = [
            [field["name"] for field in manipulator["joints"]]
            for manipulator in self.config["action"]["manipulators"]
        ]

        self._cameras: dict[str:str] = {
            field["name"]: field["topic"] for field in self.config["observation"]["cameras"]
        }

        # ~~~~~~~~~~~ Command interface ~~~~~~~~~~~ #
        self._reentrant_group = ReentrantCallbackGroup()

        self._robot_interface: MoveIt2 = MoveIt2(
            self._node,
            joint_names=[field["name"] for field in self.config["action"]["joints"]],
            base_link_name=self.config["base_link"],
            end_effector_name=self.config["ee_link"],
            group_name=self.config["group_name"],
            callback_group=self._reentrant_group,
        )
        self._robot_interface

        self._robot_interface.pipeline_id = "ompl"
        self._robot_interface.planner_id = "APSConfigDefault"

        self._gripper_interfaces: list[GripperInterface] = [
            GripperInterface(
                self._node,
                gripper_joint_names=[joint["name"] for joint in manipulator["joints"]],
                open_gripper_joint_positions=[joint["open"] for joint in manipulator["joints"]],
                closed_gripper_joint_positions=[joint["close"] for joint in manipulator["joints"]],
                max_effort=manipulator["max_effort"],
                gripper_group_name=manipulator["name"],
                callback_group=self._reentrant_group,
                gripper_command_action_name=manipulator["action_name"],
            )
            for manipulator in self.config["action"]["manipulators"]
        ]

        self.jointStateSub = self._node.create_subscription(
            JointState,
            "/joint_states",
            self.statesCallback,
            10,
            callback_group=self._reentrant_group,
        )
        self.jointCommandSub = self._node.create_subscription(
            JointState,
            "/isaac_joint_commands",
            self.commandsCallback,
            10,
            callback_group=self._reentrant_group,
        )
        self.cameraSubs = [
            self._node.create_subscription(
                Image,
                topic,
                lambda msg: self.cameraCallback(msg, idx),
                qos_profile=10,
                callback_group=self._reentrant_group,
            )
            for idx, (_, topic) in enumerate(self._cameras.items())
        ]

        self.resetSceneClient: Client = self._node.create_client(
            Trigger, "/IsaacSim/NewScene", callback_group=self._reentrant_group
        )
        self.collisionRequest = self._node.create_client(
            CollisionRequest,
            "/IsaacSim/RequestCollisionCheck",
            callback_group=self._reentrant_group,
        )
        self.tubeParameterRequest = self._node.create_client(
            TubeParameter, "/IsaacSim/RequestTubeParameter", callback_group=self._reentrant_group
        )

        while (
            not self.resetSceneClient.wait_for_service(timeout_sec=5.0)
            and not self.collisionRequest.wait_for_service(timeout_sec=5.0)
            and not self.tubeParameterRequest.wait_for_service(timeout_sec=5.0)
        ):
            self.logger.info("Waiting for service")
            sleep(1)
        self.logger.info("Service available")

        self._success_checker = self._node.create_timer(
            2.0, self._check_success, self._reentrant_group, self._node.get_clock()
        )

        # ~~~~~~~~~~~~~~~ Gym spaces ~~~~~~~~~~~~~~ #
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            dict(
                {
                    "state": gym.spaces.Dict(
                        dict(
                            {
                                joint["name"]: gym.spaces.Box(
                                    joint["lower"], joint["upper"], dtype=np.float32
                                )
                                for joint in self.config["observation"]["joints"]
                            },
                            **{
                                manipulator["name"]: gym.spaces.Box(
                                    manipulator["lower"], manipulator["upper"], dtype=np.float32
                                )
                                for manipulator in self.config["observation"]["manipulators"]
                            },
                        )
                    ),
                    "images": gym.spaces.Dict(
                        dict(
                            {
                                camera["name"]: gym.spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(camera["height"], camera["width"], camera["channels"]),
                                    dtype=np.uint8,
                                )
                                for camera in self.config["observation"]["cameras"]
                            }
                        )
                    ),
                }
            )
        )

        self.action_space = gym.spaces.Dict(
            {
                joint["name"]: gym.spaces.Box(joint["lower"], joint["upper"], dtype=np.float32)
                for joint in self.config["action"]["joints"]
            },
            **{
                joint["name"]: gym.spaces.Box(joint["lower"], joint["upper"], dtype=np.float32)
                for manipulator in self.config["action"]["manipulators"]
                for joint in manipulator["joints"]
            },
            **{
                camera["name"]: gym.spaces.Discrete(2)
                for camera in self.config["action"]["cameras"]
            },
        )

        # ~~~~~~~~~~~~~~~ Gym memory ~~~~~~~~~~~~~~ #
        self._robot_action: list[float] = [0.0 for _ in self._robot_joints]
        self._gripper_action: list[list[float]] = [
            [0.0 for _ in manipulator] for manipulator in self._gripper_joints
        ]

        self.action_tolerance = np.deg2rad(5.0)

        self._robot_state: list[float] = [0.0 for _ in self._robot_joints]
        self._gripper_state: list[list[float]] = [
            [0.0 for _ in manipulator] for manipulator in self._gripper_joints
        ]

        self._cameraImage: list[CVImage.Image] = [CVImage.Image() for _, _ in self._cameras.items()]

        self._robot_default_position: list[float] = [
            joint["default"] for joint in self.config["action"]["joints"]
        ]
        self._gripper_default_position: list[list[float]] = [
            [joint["default"] for joint in manipulator["joints"]]
            for manipulator in self.config["action"]["manipulators"]
        ]

    def _get_obs(self):
        obs = OrderedDict(
            {
                "state": dict(
                    {
                        name: np.array([self._robot_state[idx]], dtype=np.float32)
                        for idx, name in enumerate(self._robot_joints)
                    },
                    **{
                        joint: np.array([self._gripper_state[idx][jdx]], dtype=np.float32)
                        for idx, manipulator in enumerate(self._gripper_joints)
                        for jdx, joint in enumerate(manipulator)
                    },
                ),
                "images": dict(
                    {
                        name: np.array(self._cameraImage[idx], dtype=np.uint8)
                        for idx, (name, _) in enumerate(self._cameras.items())
                    }
                ),
            }
        )
        for name, _ in self._cameras.items():
            if obs["images"][name] is None:
                obs["images"][name] = np.zeros(self.observation_space["images"][name].shape)
        return obs

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset()

        executor = self._node.executor

        self._success_checker.cancel()

        msg: Trigger = Trigger.Request()
        self.callService(self.resetSceneClient, msg)

        self._robot_interface.move_to_configuration(self._robot_default_position)

        for idx, _ in enumerate(self._gripper_joints):
            self.moveRobotGripper(idx, self._gripper_default_position[idx][0])

        self._success_checker.reset()

        executor.add_node(self._node)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: gym.spaces.Dict):
        # ~~~~~~~~~~~~ Robot arm action ~~~~~~~~~~~ #
        robot_commands: list[float]
        for idx, joint in enumerate(action["joints"]):
            robot_commands[idx] = joint
        if any(
            robot_commands[i] - self._robot_action[i] > self.action_tolerance
            for i in range(len(robot_commands))
        ):
            _robot_movement = Thread(target=self.moveRobotArmToConfiguration, args=[robot_commands])
            _robot_movement.start()

        # ~~~~~~~~~~~~~ Gripper action ~~~~~~~~~~~~ #
        gripper_command: list[list[float]]
        for idx, manipulator in enumerate(self._gripper_joints):
            for jdx, joint in enumerate(manipulator["joints"]):
                gripper_command[idx][jdx] = joint
        for i in range(len(gripper_command)):
            for j in range(len(gripper_command[i])):
                if gripper_command[i][j] - self._gripper_action[i][j] > self.action_tolerance:
                    _gripper_movement = Thread(
                        target=self.moveRobotGripper, args=[i, gripper_command[i][j]]
                    )
                    _gripper_movement.start()

        obs = self._get_obs()
        info = self._get_info()

        # Sparse reward and negative counterpart to encourage efficiency
        reward = 1.0 if self._reached_target else -0.01

        if self._reached_target:
            self._success_checker += 1
        else:
            self._success_checker = 0

        # Check if tube reached goal
        terminated = True if self._success_checker >= 5 else False

        # Some movement planning failed
        truncated = self.fault
        # Clear fault
        if self.fault:
            self.fault = False

        self.logger.info("Step action completed")
        return obs, reward, terminated, truncated, info

    def _check_success(self):
        request: CollisionRequest.Request = CollisionRequest.Request()
        request.prim1 = "/World/Racks/Rack_Goal"
        request.prim2 = "/World/Tubes/Tube_Target"
        inRack: bool = self.callService(self.collisionRequest, request, silent=True).collision

        request: CollisionRequest.Request = CollisionRequest.Request()
        request.prim1 = "/World/Tubes/Tube_Target"
        request.prim2 = "/World/Room/table_low_327"
        notOnFloor: bool = not self.callService(
            self.collisionRequest, request, silent=True
        ).collision

        request: TubeParameter.Request = TubeParameter.Request()
        request.path = "/World/Tubes/Tube_Target"
        pose_res: TubeParameter.Response = self.callService(
            service=self.tubeParameterRequest, request=request, silent=True
        )

        tube_orientation: Rotation = Rotation.from_quat(
            [
                pose_res.pose.orientation.x,
                pose_res.pose.orientation.y,
                pose_res.pose.orientation.z,
                pose_res.pose.orientation.w,
            ]
        )
        tube_orientation = Rotation.from_euler("xyz", [np.pi / 2, 0.0, 0.0]) * tube_orientation
        tube_orientation = tube_orientation.as_euler("xyz")
        notBinted: bool = not (
            (abs(tube_orientation[0]) > np.deg2rad(10))
            or (abs(tube_orientation[1]) > np.deg2rad(10))
        )

        self._reached_target = inRack and notOnFloor and notBinted
        self.logger.debug("Target reached: " + (str)(self._reached_target))

    # ======================================================== #
    # ==================== State callbacks =================== #
    # ======================================================== #
    def statesCallback(self, msg: JointState):
        for idx, name in enumerate(msg.name):
            if name in self._robot_joints:
                self._robot_state[self._robot_joints.index(name)] = msg.position[idx]

            for jdx, manipulator in enumerate(self._gripper_joints):
                if name in manipulator:
                    self._gripper_state[jdx][manipulator.index(name)] = msg.position[idx]

    def commandsCallback(self, msg: JointState):
        for idx, name in enumerate(msg.name):
            if name in self._robot_joints:
                self._robot_action[self._robot_joints.index(name)] = msg.position[idx]

            if name in self._gripper_joints:
                self._gripper_action[self._gripper_joints.index(name)] = msg.position[idx]

    def cameraCallback(self, msg: Image, idx: int):
        self._cameraImage[idx] = cv2.cvtColor(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), cv2.COLOR_BGR2RGB
        )
        # CVImage.fromarray(cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), cv2.COLOR_BGR2RGB))

    # ======================================================== #
    # =============== Robot movement functions =============== #
    # ======================================================== #
    def moveRobotArm(
        self,
        pose: Pose | PoseStamped | None = None,
        position: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        velocity: float = 0.0,
        acceleration: float = 0.0,
        cartesian: bool = False,
    ):
        if not velocity == 0.0:
            self._robot_interface.max_velocity = velocity
        if not acceleration == 0.0:
            self._robot_interface.max_acceleration = acceleration
        try:
            if isinstance(pose, Pose) or isinstance(pose, PoseStamped):
                self._robot_interface.move_to_pose(
                    pose=pose,
                    cartesian=cartesian,
                    cartesian_max_step=0.000025,
                    cartesian_fraction_threshold=0.000001,
                )
            elif isinstance(position, tuple[float, float, float]) and isinstance(
                orientation, tuple[float, float, float, float]
            ):
                self._robot_interface.move_to_pose(
                    position=position,
                    quat_xyzw=orientation,
                    cartesian=cartesian,
                    cartesian_max_step=0.000025,
                    cartesian_fraction_threshold=0.000001,
                )
        except Exception as e:
            self.fault = True
            self.logger.info("Error: " + str(e))
            raise Exception()
        self.logger.info("Movement completed!")

    def moveRobotArmToConfiguration(self, joint_angles: list[float]):
        try:
            self._robot_interface.move_to_configuration(joint_angles, self._robot_joints)
        except Exception as e:
            self.fault = True
            self.logger.info("Error: " + str(e))
        if self.fault:
            raise Exception()
        self.logger.info("Movement completed!")

    def callService(
        self, service: Client, request, message: str | None = None, silent: bool = False
    ):
        if isinstance(message, str):
            self.logger.info(message)
        if not silent:
            self.logger.info("Calling " + str(service.srv_name) + " service.")
        executor = self._node.executor
        future = service.call_async(request)
        rclpy.spin_until_future_complete(self._node, future)
        response = future.result()
        self._node.executor = executor
        executor.add_node(self._node)
        executor.wake()
        if not silent:
            self.logger.info("Called" + str(service.srv_name) + " successfully.")
        return response

    def moveRobotGripper(self, manipulator: int, joint_angle: float, effort: float = None):
        try:
            executor = self._node.executor
            self._gripper_interfaces[manipulator].move_to_position(joint_angle)
            executor.add_node(self._node)
            self.logger.info("Waiting for execution.")
            sleep(3)
            self._gripper_interfaces[manipulator].wait_until_executed()
            self.logger.info("Waiting ended")
            # sleep(1)
        except Exception as e:
            self.fault = True
            self.logger.info("Error: " + str(e))
        if self.fault:
            raise Exception()
        self.logger.info("Movement completed!")
