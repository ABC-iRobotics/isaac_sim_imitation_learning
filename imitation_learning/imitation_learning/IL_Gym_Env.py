import os
import sys
from functools import wraps
from pathlib import Path
from time import sleep
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from PIL import Image as CVImage
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.executors import Executor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import SetBool, Trigger

from isaac_sim_msgs.srv import CollisionRequest, Prompt, TubeParameter

try:
    sys.path.append(get_package_share_directory("pymoveit2"))  # Install
except PackageNotFoundError:
    # Symbolic link
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pymoveit2.gripper_interface import GripperInterface
from pymoveit2.moveit2 import MoveIt2


def executor_safe(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        if not isinstance(args[0], Node):
            raise RuntimeError("Function is not tied to a ROS 2 Node.")
        executor = args[0].executor
        result = f(*args, **kwargs)
        args[0].executor = executor
        executor.add_node(args[0])
        executor.wake()
        return result

    return decorator


class IL_Gym_Env(gym.Env, Node):
    def __init__(self, executor: Executor):
        super().__init__("Gym_Env")
        self.logger = self.get_logger()
        self.fault = False
        self._reached_target = False
        self._success_buffer = 0
        self.task = ""

        executor.add_node(self)

        self.bridge = CvBridge()

        config_path = os.path.join(get_package_share_directory("imitation_learning"), "config")

        with open(config_path + "/config.yaml") as f:
            self.config = yaml.safe_load(f)

        self.name: str = self.config["robot"]

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
            self,
            joint_names=self._robot_joints,
            base_link_name=self.config["base_link"],
            end_effector_name=self.config["ee_link"],
            group_name=self.config["group_name"],
            callback_group=self._reentrant_group,
        )

        self._robot_interface.max_velocity = 0.5
        self._robot_interface.max_acceleration = 0.5
        self._robot_interface.pipeline_id = "ompl"
        self._robot_interface.planner_id = "RRTConnectkConfigDefault"
        # self._robot_interface.pipeline_id = "isaac_ros_cumotion"
        # self._robot_interface.planner_id = "cuMotion"

        self._gripper_interfaces: list[GripperInterface] = [
            GripperInterface(
                self,
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

        self.jointStateSub = self.create_subscription(
            JointState,
            "/joint_states",
            self.statesCallback,
            10,
            callback_group=self._reentrant_group,
        )

        self.jointCommandPub = self.create_publisher(
            JointState, "/isaac_joint_commands", 10, callback_group=self._reentrant_group
        )

        self.cameraSubs = [
            self.create_subscription(
                Image,
                topic,
                lambda msg, id=idx: self.cameraCallback(msg, id),
                qos_profile=10,
                callback_group=self._reentrant_group,
            )
            for idx, (_, topic) in enumerate(self._cameras.items())
        ]

        self.resetSceneClient: Client = self.create_client(
            Trigger, "/IsaacSim/NewScene", callback_group=self._reentrant_group
        )
        self.collisionRequest = self.create_client(
            CollisionRequest,
            "/IsaacSim/RequestCollisionCheck",
            callback_group=self._reentrant_group,
        )
        self.tubeParameterRequest = self.create_client(
            TubeParameter, "/IsaacSim/RequestTubeParameter", callback_group=self._reentrant_group
        )

        self.instructionServer = self.create_service(
            Prompt,
            "/Instructor/Prompt",
            self.instruction_callback,
            callback_group=self._reentrant_group,
        )

        # self.instructorClient: Client = self.create_client(
        #     Prompt, "/Instructor/Prompt", callback_group=self._reentrant_group
        # )

        self.closeGripper = self.create_client(
            SetBool, "/IsaacSim/CloseGripper", callback_group=self._reentrant_group
        )

        while (
            not self.resetSceneClient.wait_for_service(timeout_sec=30.0)
            and not self.collisionRequest.wait_for_service(timeout_sec=30.0)
            and not self.tubeParameterRequest.wait_for_service(timeout_sec=30.0)
            # and not self.instructorClient.wait_for_service(timeout_sec=300.0)
            and not self.closeGripper.wait_for_service(timeout_sec=30.0)
        ):
            self.logger.info("Waiting for service")
            sleep(1)
        self.logger.info("Service available")

        self._check_success_timer = self.create_timer(
            2.0, self._check_success, self._reentrant_group, self.get_clock()
        )

        # ~~~~~~~~~~~~~~~ Gym memory ~~~~~~~~~~~~~~ #
        self._robot_action: list[float] = [0.0 for _ in self._robot_joints]
        self._gripper_action: list[list[float]] = [
            [0.0 for _ in manipulator] for manipulator in self._gripper_joints
        ]

        self.action_tolerance = np.deg2rad(1)

        self._robot_state: list[float] = [0.0 for _ in self._robot_joints]
        self._gripper_state: list[list[float]] = [
            [0.0 for _ in manipulator] for manipulator in self._gripper_joints
        ]

        self._cameraImage: list[CVImage.Image] = [
            CVImage.fromarray(
                np.zeros(
                    (
                        field["height"],
                        field["width"],
                        field["channels"],
                    ),
                    dtype=np.uint8,
                ),
            )
            for field in self.config["observation"]["cameras"]
        ]
        self._rawCameraImage: list[Image] = [Image() for _, _ in self._cameras.items()]

        self._robot_default_position: list[float] = [
            joint["default"] for joint in self.config["action"]["joints"]
        ]
        self._gripper_default_position: list[list[float]] = [
            [joint["default"] for joint in manipulator["joints"]]
            for manipulator in self.config["action"]["manipulators"]
        ]
        # ~~~~~~~~~~~~~~~ Gym spaces ~~~~~~~~~~~~~~ #

        self.observation_features = dict(
            {joint["name"]: float for joint in self.config["observation"]["joints"]},
            **{
                manipulator["name"]: float
                for manipulator in self.config["observation"]["manipulators"]
            },
            **{
                camera["name"]: (camera["channels"], camera["height"], camera["width"])
                for camera in self.config["observation"]["cameras"]
            },
        )

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                f"{joint['name']}": gym.spaces.Box(joint["lower"], joint["upper"], dtype=np.float32)
                for joint in self.config["observation"]["joints"]
            },
            **{
                f"{manipulator['name']}": gym.spaces.Box(
                    manipulator["lower"], manipulator["upper"], dtype=np.float32
                )
                for manipulator in self.config["observation"]["manipulators"]
            },
            # {
            #     f"{OBS_STATE}": gym.spaces.Box(
            #         low=obs_state_lower,
            #         high=obs_state_upper,
            #         shape=(obs_state_length,),
            #         dtype=np.float32,
            #     )
            # },
            **{
                f"{camera['name']}": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(camera["channels"], camera["height"], camera["width"]),
                    dtype=np.uint8,
                )
                for camera in self.config["observation"]["cameras"]
            },
        )

        self.action_features = dict(
            {(joint["name"]): float for joint in self.config["action"]["joints"]},
            **{
                (joint["name"]): float
                for manipulator in self.config["action"]["manipulators"]
                for joint in manipulator["joints"]
            },
        )

        self.action_space = gym.spaces.Dict(
            {
                (joint["name"]): gym.spaces.Box(joint["lower"], joint["upper"], dtype=np.float32)
                for joint in self.config["action"]["joints"]
            },
            **{
                (joint["name"]): gym.spaces.Box(joint["lower"], joint["upper"], dtype=np.float32)
                for manipulator in self.config["action"]["manipulators"]
                for joint in manipulator["joints"]
            },
            # {f"{ACTION}": gym.spaces.Box(
            #     low=np.array(action_lower, dtype=np.float32),
            #     high=np.array(action_upper, dtype=np.float32),
            #     shape=(action_length,),
            #     dtype=np.float32,
            # )}
        )

    def resize_keep_max_area(self, img, target_width=1280, target_height=720):
        h, w = img.shape[:2]

        # Skálázási faktorok
        scale_w = target_width / w
        scale_h = target_height / h

        # A nagyobb skálát választjuk, hogy ne maradjon üres rész
        scale = max(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Átméretezés
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Középre vágás
        x_start = (new_w - target_width) // 2
        y_start = (new_h - target_height) // 2

        cropped = resized[y_start : y_start + target_height, x_start : x_start + target_width]

        return cropped

    def get_obs(self) -> dict[str, Any]:
        return self._get_obs()

    def _get_obs(self):
        obs = dict(
            {f"{joint}": self._robot_state[idx] for idx, joint in enumerate(self._robot_joints)},
            **{
                f"{joint}": self._gripper_state[idx][jdx]
                for idx, manipulator in enumerate(self._gripper_joints)
                for jdx, joint in enumerate(manipulator)
            },
            **{
                f"{camera}": (
                    np.array(self._cameraImage[idx])
                    if self._cameraImage[idx].size != 0
                    else np.zeros(
                        (
                            self.config["observation"]["cameras"][idx]["channels"],
                            self.config["observation"]["cameras"][idx]["height"],
                            self.config["observation"]["cameras"][idx]["width"],
                        ),
                        dtype=np.uint8,
                    )
                )
                for idx, camera in enumerate(self._cameras.keys())
            },
        )
        return obs

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset()

        self._check_success_timer.cancel()

        msg: Trigger = Trigger.Request()
        self.callService(self.resetSceneClient, msg)

        # self._robot_interface.move_to_configuration(self._robot_default_position)

        for idx, _ in enumerate(self._gripper_joints):
            self.moveRobotGripper(idx, self._gripper_default_position[idx][0])

        self._check_success_timer.reset()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        movement_started = False
        # self.logger.info((str)(action))
        # self.logger.info((str)(self._robot_joints))
        robot_message: JointState = JointState()
        robot_message.header.frame_id = "base"
        robot_message.header.stamp = self.get_clock().now().to_msg()
        # ~~~~~~~~~~~~ Robot arm action ~~~~~~~~~~~ #
        robot_commands: list[float] = [0.0 for _ in range(6)]
        for key, joint in action.items():
            if key in self._robot_joints:
                robot_commands[self._robot_joints.index(key)] = joint
        # if any( abs(robot_commands[i] - self._robot_action[i]) >
        # self.action_tolerance for i in range(len(robot_commands)) ):
        self._robot_action = robot_commands
        # _robot_movement = Thread(target=self.moveRobotArmToConfiguration, args=[robot_commands])
        # _robot_movement.start()
        # movement_started = True
        # self.logger.info((str)(robot_commands))

        # self.moveRobotArmToConfiguration(robot_commands)

        robot_message.name.extend(self._robot_joints)
        robot_message.position.extend(self._robot_action)

        # ~~~~~~~~~~~~~ Gripper action ~~~~~~~~~~~~ #
        close = None
        gripper_command: list[list[float]] = [[0]]
        for idx, manipulator in enumerate(self._gripper_joints):
            for jdx, joint in enumerate(manipulator):
                gripper_command[idx][jdx] = action[joint]
        for i in range(len(gripper_command)):
            for j in range(len(gripper_command[i])):
                if gripper_command[i][j] - self._gripper_action[i][j] > self.action_tolerance:
                    close = True
                elif self._gripper_action[i][j] - gripper_command[i][j] > self.action_tolerance:
                    close = False
        if close is not None and not close:
            self.callService(self.closeGripper, SetBool.Request(data=close))
            # _gripper_movement = Thread(target=self.moveRobotGripper, args=[i, gripper_command[i][j]])
            # _gripper_movement.start()
            # movement_started = True

            # self.moveRobotGripper(i, gripper_command[i][0])
        self._gripper_action = gripper_command

        if close is not None and close:
            self.callService(self.closeGripper, SetBool.Request(data=close))
            # _gripper_movement = Thread(target=self.moveRobotGripper, args=[i, gripper_command[i][j]])
            # _gripper_movement.start()
            # movement_started = True

            # self.moveRobotGripper(i, gripper_command[i][0])

        robot_message.name.extend(self._gripper_joints[0])
        robot_message.position.extend(self._gripper_action[0])

        self.jointCommandPub.publish(robot_message)

        # self.logger.info(f"Published to joints {robot_message.name} with positions {robot_message.position}")

        sleep(0.1)

        # if movement_started: sleep(2)
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

        # self.logger.info("Step action completed")
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
            # Robot arm states
            if name in self._robot_joints:
                self._robot_state[self._robot_joints.index(name)] = msg.position[idx]

            # Gripper states
            for jdx, manipulator in enumerate(self._gripper_joints):
                if name in manipulator:
                    self._gripper_state[jdx][manipulator.index(name)] = msg.position[idx]

    def commandsCallback(self, msg: JointState):
        for idx, name in enumerate(msg.name):
            # Robot arm commands
            if name in self._robot_joints:
                self._robot_action[self._robot_joints.index(name)] = msg.position[idx]

            # Gripper commands
            if name in self._gripper_joints:
                self._gripper_action[self._gripper_joints.index(name)] = msg.position[idx]

    def cameraCallback(self, msg: Image, idx: int):
        # Image acquisition and conversion
        camera = self.config["observation"]["cameras"][idx]
        self._rawCameraImage[idx] = msg
        image = cv2.cvtColor(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), cv2.COLOR_BGR2RGB
        )
        # Resize image to expected size
        self._cameraImage[idx] = self.resize_keep_max_area(image, camera["width"], camera["height"])

    def instruction_callback(
        self, request: Prompt.Request, response: Prompt.Response
    ) -> Prompt.Response:

        self.task: str = f"{request.prompt}"

        response.response = "OK"

        return response

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
            # self.fault = True
            self.logger.info("Error: " + str(e))
            raise Exception()
        self.logger.info("Movement completed!")

    def moveRobotArmToConfiguration(self, joint_angles: list[float]):
        try:
            # self._robot_interface.pipeline_id = "isaac_ros_cumotion"
            # self._robot_interface.planner_id = "cuMotion"
            self._robot_interface.pipeline_id = "ompl"
            self._robot_interface.planner_id = "RRTConnectkConfigDefault"

            self._robot_interface.move_to_configuration(
                joint_angles, self._robot_joints, start_joint_state=self._robot_state
            )
            sleep(2)
            self._robot_interface.wait_until_executed()
            self._robot_interface.reset_new_joint_state_checker()
            while not self._robot_interface.new_joint_state_available:
                executor_safe(self._robot_interface._node.executor.spin_once)(timeout_sec=0.1)
        except Exception as e:
            # self.fault = True
            self.logger.info("Error: " + str(e))
        self.logger.info("Movement completed!")

    def callService(
        self, service: Client, request, message: str | None = None, silent: bool = False
    ):
        if isinstance(message, str):
            self.logger.info(message)
        if not silent:
            self.logger.info("Calling " + str(service.srv_name) + " service.")
        response = service.call(request)
        if not silent:
            self.logger.info("Called" + str(service.srv_name) + " successfully.")
        return response

    def moveRobotGripper(self, manipulator: int, joint_angle: float, effort: float = None):
        try:
            action = JointState()
            action.header.stamp = self.get_clock().now().to_msg()
            action.header.frame_id = "base"
            action.name = ["finger_joint"]
            action.position = [joint_angle]
            action.effort = [effort]
            with self.gripperActionLock:
                self.demonstration_info["gripper_action"] = action
            self._gripper_interface.move_to_position(action.position[0])
            self.logger.info("Waiting for execution.")
            sleep(2)
            self._gripper_interface.wait_until_executed()
            self.logger.info("Waiting ended")
            while not self._gripper_interface.new_joint_state_available:
                executor_safe(self._gripper_interface.executor.spin_once)(timeout_sec=0.1)
        except Exception as e:
            # self.fault = True
            self.logger.info("Error: " + str(e))
        self.logger.info("Movement completed!")

    # ======================================================== #
    # ==================== Getting subtask =================== #
    # ======================================================== #

    def query_subtask(self, subtask_list: list[str]) -> str:
        query_prompt: Prompt.Request = Prompt.Request()
        prompt_text = "Given the following list of subtasks: "
        prompt_text += "\nDo nothing."
        for idx, subtask in enumerate(subtask_list):
            prompt_text += f"\n{idx+2}. {subtask}"
        prompt_text += "\nThe task is to move one test tube to another rack. What is the most appropriate subtask to perform next based on the current observation?"
        query_prompt.prompt = prompt_text

        # Use the first camera image for subtask querying
        query_prompt.image = self._rawCameraImage[list(self._cameras.keys()).index("base")]

        response: Prompt.Response = self.callService(
            self.instructorClient, query_prompt, silent=True
        )
        return response.response
