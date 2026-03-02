# ======================================================== #
# ===================== ROS 2 Imports ==================== #
# ======================================================== #
import sys
import threading

# ======================================================== #
# ==================== Python Imports ==================== #
# ======================================================== #
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
from pathlib import Path
from time import sleep

import numpy as np
import rclpy
import rclpy.action
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Transform
from onrobot_rg_msgs.srv import GripperPose
from rclpy.action import CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState

# ======================================================== #
# ==================== Message Imports =================== #
# ======================================================== #
from std_msgs.msg import Header
from std_srvs.srv import SetBool, Trigger

from isaac_sim_msgs.action import Demonstration
from isaac_sim_msgs.srv import CollisionRequest, PoseRequest, PrimAttribute, TubeParameter

# ======================================================== #
# ====================== Own Imports ===================== #
# ======================================================== #
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


class AnalyticSolver(Node):
    def __init__(self, nodename: str):
        super().__init__(nodename)
        # ~~~~~~~~~~~~ Robot variables ~~~~~~~~~~~~ #
        self._jointState: JointState = JointState()
        self._lastJointState: JointState = JointState()
        self.jointNames = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.gripper_joint = ["finger_joint"]
        self.camera_names = ["base", "eih"]
        self.subtasks = [
            "Move to home position.",
            "Orienting above the purple tube.",
            "Grasp the purple tube and pull it out.",
            "Move to goal rack and inserting the tube.",
        ]
        self.ready1 = [np.pi / 2, 0.0, np.pi / 2, 0.0, np.pi / 2, 0.0]

        # ~~~~~~~~~~~~~ Miscellaneous ~~~~~~~~~~~~~ #
        self.mutually_exclusive_group = MutuallyExclusiveCallbackGroup()
        self.reentrant_group = ReentrantCallbackGroup()

        self._robot_interface = Node("robot_interface")
        self._gripper_interface = Node("gripper_interface")

        # ~~~~~~~~~~~ MoveIt parameters ~~~~~~~~~~~ #
        self.robot_interface = MoveIt2(
            self._robot_interface,
            joint_names=self.jointNames,
            base_link_name="base",
            end_effector_name="flange",
            group_name="tmr_arm",
            callback_group=self.reentrant_group,
        )

        self.gripper_interface = GripperInterface(
            self._gripper_interface,
            gripper_joint_names=["finger_joint"],
            open_gripper_joint_positions=[36.0],
            closed_gripper_joint_positions=[0.0],
            max_effort=120.0,
            gripper_group_name="rg6",
            callback_group=self.reentrant_group,
        )

        self.robot_interface.pipeline_id = "isaac_ros_cumotion"
        self.robot_interface.planner_id = "cuMotion"

        self.robot_interface.max_velocity = 1.0
        self.robot_interface.max_acceleration = 1.0
        self.robot_interface.cartesian_avoid_collisions = False
        self.robot_interface.cartesian_jump_threshold = 0.0

        # ~~~~~~~~ Demonstration variables ~~~~~~~~ #
        self.fault = False
        self.solving = False

        self.demonstration_info = {
            "header": Header(),
            "arm_state": JointState(),
            "arm_action": JointState(),
            "gripper_state": JointState(),
            "gripper_action": JointState(),
            "eih": Image(),
            "base": Image(),
            "message": "",
        }

        self.armStateLock = threading.Lock()
        self.armActionLock = threading.Lock()
        self.gripperStateLock = threading.Lock()
        self.gripperActionLock = threading.Lock()
        self.cameraLock = threading.Lock()

        # ~~~~~~~~~~~~ Isaac Sim paths ~~~~~~~~~~~~ #
        self.target_tube = "/World/Tubes/Tube_Target"
        self.start_rack = "/World/Racks/Rack_Start"
        self.target_rack = "/World/Racks/Rack_Goal"

        self.rate = self.create_rate(100)

        self._active_goal = None
        self._active_goal_lock = threading.Lock()

        # Persistent 20 Hz feedback timer
        self.feedback_timer = self.create_timer(
            1.0 / 20.0, self._feedback_tick, callback_group=self.reentrant_group
        )

        # ~~~~~~~~~~~~~~ Main topics ~~~~~~~~~~~~~~ #
        self.jointStateSub = self.create_subscription(
            JointState,
            "/joint_states",
            self.refreshRobotValues,
            10,
            callback_group=self.reentrant_group,
        )
        self.jointCommandSub = self.create_subscription(
            JointState,
            "/isaac_joint_commands",
            self.jointCommandCallback,
            10,
            callback_group=self.reentrant_group,
        )

        self.eih_cameraSub = self.create_subscription(
            Image,
            "/eih_camera",
            lambda msg: self.cameraCallback(msg, "eih"),
            10,
            callback_group=self.reentrant_group,
        )

        self.base_cameraSub = self.create_subscription(
            Image,
            "/base_camera",
            lambda msg: self.cameraCallback(msg, "base"),
            10,
            callback_group=self.reentrant_group,
        )

        # ~~~~~~~~~~~ Isaac Sim services ~~~~~~~~~~ #
        self.poseRequest = self.create_client(
            PoseRequest, "/IsaacSim/RequestPose", callback_group=self.mutually_exclusive_group
        )
        self.tubeGraspPoseRequest = self.create_client(
            PoseRequest,
            "/IsaacSim/RequestTubeGraspPose",
            callback_group=self.mutually_exclusive_group,
        )
        self.tubeParameterRequest = self.create_client(
            TubeParameter,
            "/IsaacSim/RequestTubeParameter",
            callback_group=self.mutually_exclusive_group,
        )
        self.closeGripper = self.create_client(
            SetBool, "/IsaacSim/CloseGripper", callback_group=self.reentrant_group
        )
        self.goalPoseRequest = self.create_client(
            PoseRequest, "/IsaacSim/RequestTubeGoalPose", callback_group=self.reentrant_group
        )
        self.collisionRequest = self.create_client(
            CollisionRequest, "/IsaacSim/RequestCollisionCheck", callback_group=self.reentrant_group
        )
        self.setAttribute = self.create_client(
            PrimAttribute, "/IsaacSim/SetAttribute", callback_group=self.reentrant_group
        )

        # ~~~~~~~~~~~~~~ RG6 services ~~~~~~~~~~~~~ #
        self.gripperPoseRequest = self.create_client(
            GripperPose, "/onrobot/pose", callback_group=self.mutually_exclusive_group
        )

        # ~~~~~~~ Wait until services start ~~~~~~~ #
        while not (
            self.poseRequest.wait_for_service(timeout_sec=5.0)
            and self.tubeGraspPoseRequest.wait_for_service(timeout_sec=5.0)
            and self.gripperPoseRequest.wait_for_service(timeout_sec=5.0)
            and self.tubeParameterRequest.wait_for_service(timeout_sec=5.0)
            and self.setAttribute.wait_for_service(timeout_sec=5.0)
        ):
            if not rclpy.ok():
                self.get_logger().error("Interrupted while waiting for the servers.")
                return
            else:
                self.get_logger().info("Servers not available, waiting again...")

        # ~~~~~~ Self services and actions to solve scene ~~~~~ #
        self.solve_scene = self.create_service(
            Trigger,
            "/AnalyticSolver/SolveScene",
            self.solveSceneCallback,
            callback_group=self.reentrant_group,
        )
        self.self_solve_scene = self.create_client(
            Trigger, "/AnalyticSolver/SolveScene", callback_group=self.reentrant_group
        )

        self.create_demonstration = rclpy.action.ActionServer(
            self,
            Demonstration,
            "/AnalyticSolver/GetDemonstration",
            callback_group=self.reentrant_group,
            goal_callback=self.get_demonstration_goal_callback,
            cancel_callback=self.get_demonstration_cancel_callback,
            execute_callback=self.get_demonstration_execute_callback,
        )

    def addInterfaces(self):
        self.executor.add_node(self._robot_interface)
        self.executor.add_node(self._gripper_interface)
        self.executor.wake()

        # ~~~~~~ Set robot to start position ~~~~~~ #
        time.sleep(2)
        self.moveRobotArmToConfiguration(self.ready1)

        self.get_logger().info("AnalyticSolver ready.")

    # ======================================================== #
    # ==================== State callbacks =================== #
    # ======================================================== #
    def refreshRobotValues(self, msg: JointState):
        if any(joint_name == "joint_1" for joint_name in msg.name):
            self._lastJointState = self._jointState
            self._jointState = msg

        # ~~~~~~~~~~~~~ Driver faulted ~~~~~~~~~~~~ #
        if (
            len([i for i in self._jointState.name if "joint_" in i]) > 0
            and all(
                pos == 0.0
                for pos, joint in zip(self._jointState.position, self._jointState.name)
                if "joint_" in joint
            )
            and self.solving
        ):
            self.fault = True

        if len([i for i in self._jointState.name if "joint_" in i]) > 0:
            self.demonstration_info["arm_state"] = msg
        try:
            if len([i for i in msg.name if "finger_joint" in i]) > 0:
                for idx, name in enumerate(msg.name):
                    if "finger_joint" in name:
                        state = JointState()
                        state.header = msg.header
                        state.name = [name]
                        state.position = [msg.position[idx]]
                        state.velocity = [msg.velocity[idx]]
                        state.effort = [msg.effort[idx]]
                        with self.gripperStateLock:
                            self.demonstration_info["gripper_state"] = state
        except BaseException:
            pass

    def jointCommandCallback(self, msg: JointState):
        if len([i for i in self._jointState.name if "joint_" in i]) > 0:
            with self.armActionLock:
                self.demonstration_info["arm_action"] = msg

    def cameraCallback(self, msg: Image, camera_name: str = None):
        if camera_name is None:
            return
        with self.cameraLock:
            self.demonstration_info[camera_name] = msg

    # ======================================================== #
    # ================ Demonstration callback ================ #
    # ======================================================== #
    def solveSceneCallback(
        self,
        request: Trigger.Request = Trigger.Request(),
        response: Trigger.Response = Trigger.Response(),
    ):

        self.get_logger().info("Starting solving scene.")
        self.solving = False
        self.fault = False

        self.moveRobotGripper(0.0, 200.0)
        self.solving = True

        try:
            # ======================================================== #
            # ============= Subtask - Go to home position ============ #
            # ======================================================== #
            self.demonstration_info["message"] = self.subtasks[0]

            goal: PoseStamped = PoseStamped()
            goal.header.frame_id = "base"
            _g_p = goal.position
            _g_o = goal.orientation

            # ~~~~~~~~~ Go home configuration ~~~~~~~~~ #
            self.moveRobotArmToConfiguration(self.ready1)

            # ~~~~~~~~~~~~~ Get grasp pose ~~~~~~~~~~~~ #
            request: PoseRequest.Request = PoseRequest.Request()
            request.path = self.target_tube
            self.grasp_pose: Transform = self.callService(
                service=self.tubeGraspPoseRequest, request=request
            ).pose

            # ~~~~~~~~~~ Get tube parameters ~~~~~~~~~~ #
            request: TubeParameter.Request = TubeParameter.Request()
            request.path = self.target_tube
            self.tubeDimensions = self.callService(
                service=self.tubeParameterRequest, request=request
            )
            self.tubeDimensions = [
                self.tubeDimensions.dimensions.x,
                self.tubeDimensions.dimensions.y,
                self.tubeDimensions.dimensions.z,
            ]
            self.tubeWidth = min(self.tubeDimensions)
            self.tubeHeight = max(self.tubeDimensions)
            self.get_logger().debug(
                "Height: " + str(self.tubeHeight) + " Width: " + str(self.tubeWidth)
            )

            # ~~~~~~~ Get start rack parameters ~~~~~~~ #
            request: PoseRequest.Request = PoseRequest.Request()
            request.path = self.start_rack
            self.rackPose: Transform = self.callService(
                service=self.poseRequest, request=request
            ).pose

            # ======================================================== #
            # ========= Subtask - Orient and move above tube ========= #
            # ======================================================== #
            self.demonstration_info["message"] = self.subtasks[1]

            #  Set gripper to open and calculate grasp parameters  #
            request: GripperPose.Request = GripperPose.Request()
            request.known.x = self.tubeWidth + 0.005
            self.gripper_goal: Pose2D = self.callService(
                service=self.gripperPoseRequest, request=request
            ).pose
            self.get_logger().debug(
                "Gripper width: "
                + str(self.gripper_goal.x)
                + " Gripper height: "
                + str(self.gripper_goal.y)
            )
            self.moveRobotGripper(np.deg2rad(-35), 200.0)

            # ~~~~~~~~~~~~ Move above tube ~~~~~~~~~~~~ #
            request: TubeParameter.Request = TubeParameter.Request()
            request.path = self.target_tube
            self.tubeDimensions: TubeParameter.Response = self.callService(
                service=self.tubeParameterRequest, request=request
            )
            self.get_logger().debug(str(self.tubeDimensions))
            tube_orientation: Rotation = Rotation.from_quat(
                [
                    self.tubeDimensions.pose.orientation.x,
                    self.tubeDimensions.pose.orientation.y,
                    self.tubeDimensions.pose.orientation.z,
                    self.tubeDimensions.pose.orientation.w,
                ]
            )
            tube_orientation = Rotation.from_euler("xyz", [np.pi / 2, 0.0, 0.0]) * tube_orientation

            rack_orientation: Rotation = Rotation.from_quat(
                [
                    self.rackPose.rotation.x,
                    self.rackPose.rotation.y,
                    self.rackPose.rotation.z,
                    self.rackPose.rotation.w,
                ]
            )

            # ~~~~~~~~~~~~~~ Get TCP pose ~~~~~~~~~~~~~ #
            request: PoseRequest.Request = PoseRequest.Request()
            request.path = "/World/tm5_900/flange_link"
            ee_pose: Transform = self.callService(service=self.poseRequest, request=request).pose
            ee_rotation: Rotation = Rotation.from_quat(
                [ee_pose.rotation.x, ee_pose.rotation.y, ee_pose.rotation.z, ee_pose.rotation.w]
            )

            minimal_rotation = rack_orientation.as_euler("xyz")[2]
            if abs(rack_orientation.as_euler("xyz")[2] - ee_rotation.as_euler("xyz")[2]) > abs(
                np.pi - abs(rack_orientation.as_euler("xyz")[2] - ee_rotation.as_euler("xyz")[2])
            ):
                minimal_rotation = minimal_rotation - np.pi

            orientation: Rotation = Rotation.from_euler(
                "xyz",
                [
                    (np.pi - tube_orientation.as_euler("xyz")[0]),
                    -tube_orientation.as_euler("xyz")[1],
                    minimal_rotation,
                ],
            )

            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y + 0.2])

            (_g_o.x, _g_o.y, _g_o.z, _g_o.w) = orientation.as_quat()
            (_g_p.x, _g_p.y, _g_p.z) = (
                self.grasp_pose.translation.x - gripper_translation[0],
                self.grasp_pose.translation.y - gripper_translation[1],
                self.grasp_pose.translation.z - gripper_translation[2],
            )
            goal.header.stamp = self.get_clock().now().to_msg()

            self.moveRobotArm(goal, cartesian=False, velocity=0.5, acceleration=0.5)

            # ======================================================== #
            # =========== Subtask - Grasp tube and pull out ========== #
            # ======================================================== #
            self.demonstration_info["message"] = self.subtasks[2]

            # ~~~~~~~~~~~~~ Approach tube ~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y])

            (_g_p.x, _g_p.y, _g_p.z) = (
                self.grasp_pose.translation.x - gripper_translation[0],
                self.grasp_pose.translation.y - gripper_translation[1],
                self.grasp_pose.translation.z - gripper_translation[2],
            )
            goal.header.stamp = self.get_clock().now().to_msg()

            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.5, acceleration=0.5)

            # ~ Close gripper and use surface gripper ~ #
            self.moveRobotGripper(self.gripper_goal.theta, 200.0)
            sleep(1)
            self.get_logger().debug("Activating Surface Gripper")
            request: SetBool.Request = SetBool.Request()
            request.data = True
            self.callService(service=self.closeGripper, request=request)
            self.get_logger().debug("Surface Gripper activated")

            # ~~~~~~~~~~~~~ Pull out tube ~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y + 0.2])
            (_g_p.x, _g_p.y, _g_p.z) = (
                self.grasp_pose.translation.x - gripper_translation[0],
                self.grasp_pose.translation.y - gripper_translation[1],
                self.grasp_pose.translation.z - gripper_translation[2],
            )
            goal.header.stamp = self.get_clock().now().to_msg()

            self.moveRobotArm(goal, velocity=0.2, acceleration=0.2)

            # ======================================================== #
            # ====== Subtask - Move to goal rack and insert tube ===== #
            # ======================================================== #
            self.demonstration_info["message"] = self.subtasks[3]

            # ~~~~~~~~~~~ Approach goal rack ~~~~~~~~~~ #
            request: PoseRequest.Request = PoseRequest.Request()
            request.path = self.target_rack
            rack_pose: Transform = self.callService(
                service=self.goalPoseRequest, request=request
            ).pose
            self.get_logger().debug(str(rack_pose))
            rack_orientation: Rotation = Rotation.from_quat(
                [
                    rack_pose.rotation.x,
                    rack_pose.rotation.y,
                    rack_pose.rotation.z,
                    rack_pose.rotation.w,
                ]
            )

            request: TubeParameter.Request = TubeParameter.Request()
            request.path = self.target_tube
            self.tubePose: TubeParameter.Response = self.callService(
                service=self.tubeParameterRequest, request=request
            )
            self.get_logger().debug(str(self.tubePose))
            tube_translation: Pose = Pose()
            tube_translation.position.x = self.tubePose.pose.position.x
            tube_translation.position.y = self.tubePose.pose.position.y
            tube_translation.position.z = self.tubePose.pose.position.z
            tube_orientation: Rotation = Rotation.from_quat(
                [
                    self.tubePose.pose.orientation.x,
                    self.tubePose.pose.orientation.y,
                    self.tubePose.pose.orientation.z,
                    self.tubePose.pose.orientation.w,
                ]
            )
            tube_orientation = Rotation.from_euler("xyz", [np.pi / 2, 0.0, 0.0]) * tube_orientation

            tube_orientation_euler = tube_orientation.as_euler("xyz")
            if (abs(tube_orientation_euler[0]) > np.deg2rad(45)) or (
                abs(tube_orientation_euler[1]) > np.deg2rad(45)
            ):
                self.get_logger().debug("Tube is not upright.")
                response.message = "Tube is not upright."
                response.success = False
                return response

            # ~~~~~~~~~~~~~~ Get TCP pose ~~~~~~~~~~~~~ #
            request: PoseRequest.Request = PoseRequest.Request()
            request.path = "/World/tm5_900/flange_link"
            ee_pose: Transform = self.callService(service=self.poseRequest, request=request).pose
            ee_rotation: Rotation = Rotation.from_quat(
                [ee_pose.rotation.x, ee_pose.rotation.y, ee_pose.rotation.z, ee_pose.rotation.w]
            )

            minimal_rotation = rack_orientation.as_euler("xyz")[2]
            if abs(rack_orientation.as_euler("xyz")[2] - ee_rotation.as_euler("xyz")[2]) > abs(
                np.pi - abs(rack_orientation.as_euler("xyz")[2] - ee_rotation.as_euler("xyz")[2])
            ):
                minimal_rotation = minimal_rotation - np.pi

            orientation: Rotation = Rotation.from_euler(
                "xyz",
                [
                    (np.pi - tube_orientation.as_euler("xyz")[0]),
                    -tube_orientation.as_euler("xyz")[1],
                    minimal_rotation,
                ],
            )

            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y + 0.2])

            (_g_o.x, _g_o.y, _g_o.z, _g_o.w) = orientation.as_quat()
            (_g_p.x, _g_p.y, _g_p.z) = (
                self.grasp_pose.translation.x - gripper_translation[0],
                self.grasp_pose.translation.y - gripper_translation[1],
                self.grasp_pose.translation.z - gripper_translation[2],
            )

            goal.header.stamp = self.get_clock().now().to_msg()

            self.get_logger().debug("Rotation: " + str(orientation.as_euler("xyz", degrees=True)))

            self.moveRobotArm(goal, cartesian=False, velocity=0.5, acceleration=0.5)

            # ~~~~~~~~~~~~~~ Put in tube ~~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y])
            (_g_p.x, _g_p.y, _g_p.z) = (
                rack_pose.translation.x - gripper_translation[0],
                rack_pose.translation.y - gripper_translation[1],
                rack_pose.translation.z - gripper_translation[2],
            )
            goal.header.stamp = self.get_clock().now().to_msg()

            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.1, acceleration=0.1)

            # ~ Open gripper and use surface gripper ~ #

            self.get_logger().debug("Activating Surface Gripper")
            request: SetBool.Request = SetBool.Request()
            request.data = False
            self.callService(service=self.closeGripper, request=request)
            self.get_logger().debug("Surface Gripper activated")

            self.moveRobotGripper(0.0, 200.0)

            # ======================================================== #
            # ============= Subtask - Go to home position ============ #
            # ======================================================== #
            self.demonstration_info["message"] = self.subtasks[0]

            # ~~~~~~~~~~~~~~ Get away from tube ~~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y + 0.2])
            (_g_p.x, _g_p.y, _g_p.z) = (
                rack_pose.translation.x - gripper_translation[0],
                rack_pose.translation.y - gripper_translation[1],
                rack_pose.translation.z - gripper_translation[2],
            )
            goal.header.stamp = self.get_clock().now().to_msg()

            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.5, acceleration=0.5)

            # ~~~~~~~~~ Go home configuration ~~~~~~~~~ #
            self.moveRobotArmToConfiguration(self.ready1)

            self.rate.sleep()

            request: CollisionRequest.Request = CollisionRequest.Request()
            request.prim1 = "/World/Racks/Rack_Start"
            request.prim2 = "/World/Tubes/Tube_Target"
            inStart: bool = self.callService(self.collisionRequest, request).collision

            request: CollisionRequest.Request = CollisionRequest.Request()
            request.prim1 = "/World/Racks/Rack_Goal"
            request.prim2 = "/World/Tubes/Tube_Target"
            inRack: bool = self.callService(self.collisionRequest, request).collision

            request: CollisionRequest.Request = CollisionRequest.Request()
            request.prim1 = "/World/Tubes/Tube_Target"
            request.prim2 = "/World/Tubes"
            nonColliding: bool = not self.callService(self.collisionRequest, request).collision

            request: CollisionRequest.Request = CollisionRequest.Request()
            request.prim1 = "/World/Tubes/Tube_Target"
            request.prim2 = "/World/Room/table_low_327"
            notOnFloor: bool = not self.callService(self.collisionRequest, request).collision

            request: TubeParameter.Request = TubeParameter.Request()
            request.path = self.target_tube
            pose_res: TubeParameter.Response = self.callService(
                service=self.tubeParameterRequest, request=request
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
            self.get_logger().debug("Tube orientation: " + str(tube_orientation))
            notBinted: bool = not (
                (abs(tube_orientation[0]) > np.deg2rad(45))
                or (abs(tube_orientation[1]) > np.deg2rad(45))
            )

            response.success = inRack and notOnFloor and notBinted and not inStart
            response.message = ""

        except Exception as e:
            request: SetBool.Request = SetBool.Request()
            request.data = False
            self.callService(service=self.closeGripper, request=request)

            self.get_logger().error(f"Service call failed: {e}")
            response.success = False
            response.message = str(e)

        self.get_logger().info("Demonstration ended.")
        self.solving = False
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
            self.robot_interface.max_velocity = velocity
        if not acceleration == 0.0:
            self.robot_interface.max_acceleration = acceleration
        try:
            self.robot_interface.pipeline_id = "isaac_ros_cumotion"
            self.robot_interface.planner_id = "cuMotion"
            if isinstance(pose, Pose) or isinstance(pose, PoseStamped):
                success = self.robot_interface.move_to_pose(
                    pose=pose,
                    cartesian=cartesian,
                    cartesian_max_step=0.000025,
                    cartesian_fraction_threshold=0.000001,
                    start_joint_state=self._jointState,
                )
                if not success:
                    self.robot_interface.pipeline_id = "ompl"
                    self.robot_interface.planner_id = "RRTConnectkConfigDefault"
                    success = self.robot_interface.move_to_pose(
                        pose=pose,
                        cartesian=cartesian,
                        cartesian_max_step=0.000025,
                        cartesian_fraction_threshold=0.000001,
                        start_joint_state=self._jointState,
                    )

            elif isinstance(position, tuple[float, float, float]) and isinstance(
                orientation, tuple[float, float, float, float]
            ):
                success = self.robot_interface.move_to_pose(
                    position=position,
                    quat_xyzw=orientation,
                    cartesian=cartesian,
                    cartesian_max_step=0.000025,
                    cartesian_fraction_threshold=0.000001,
                    start_joint_state=self._jointState,
                )
                if not success:
                    self.robot_interface.pipeline_id = "ompl"
                    self.robot_interface.planner_id = "RRTConnectkConfigDefault"
                    success = self.robot_interface.move_to_pose(
                        position=position,
                        quat_xyzw=orientation,
                        cartesian=cartesian,
                        cartesian_max_step=0.000025,
                        cartesian_fraction_threshold=0.000001,
                        start_joint_state=self._jointState,
                    )
            sleep(2)
            self.robot_interface.wait_until_executed()
            while not self.robot_interface.new_joint_state_available:
                executor_safe(self._robot_interface.executor.spin_once)(timeout_sec=0.1)
        except Exception as e:
            self.fault = True
            self.get_logger().info("Error: " + str(e))
        self.get_logger().info("Movement completed!")
        return success

    def moveRobotArmToConfiguration(self, joint_angles: list[float]):
        try:
            self.robot_interface.pipeline_id = "isaac_ros_cumotion"
            self.robot_interface.planner_id = "cuMotion"

            success = self.robot_interface.move_to_configuration(
                joint_angles, self.jointNames, start_joint_state=self._jointState
            )
            if not success:
                self.robot_interface.pipeline_id = "ompl"
                self.robot_interface.planner_id = "RRTConnectkConfigDefault"
                success = self.robot_interface.move_to_configuration(
                    joint_angles, self.jointNames, start_joint_state=self._jointState
                )
            sleep(2)
            self.robot_interface.wait_until_executed()
            self.robot_interface.reset_new_joint_state_checker()
            while not self.robot_interface.new_joint_state_available:
                executor_safe(self._robot_interface.executor.spin_once)(timeout_sec=0.1)
        except Exception as e:
            self.fault = True
            self.get_logger().info("Error: " + str(e))
        self.get_logger().info("Movement completed!")
        return success

    def moveRobotGripper(self, joint_angle: float, effort: float):
        try:
            action = JointState()
            action.header.stamp = self.get_clock().now().to_msg()
            action.header.frame_id = "base"
            action.name = ["finger_joint"]
            action.position = [joint_angle]
            action.effort = [effort]
            with self.gripperActionLock:
                self.demonstration_info["gripper_action"] = action
            self.gripper_interface.move_to_position(action.position[0])
            self._logger.info("Waiting for execution.")
            sleep(2)
            self.gripper_interface.wait_until_executed()
            self._logger.info("Waiting ended")
            while not self.gripper_interface.new_joint_state_available:
                executor_safe(self._gripper_interface.executor.spin_once)(timeout_sec=0.1)
        except Exception as e:
            self.fault = True
            self.get_logger().info("Error: " + str(e))
        self.get_logger().info("Movement completed!")

    # ======================================================== #
    # ====== /AnalyticSolver/GetDemonstration functions ====== #
    # ======================================================== #
    def get_demonstration_goal_callback(self, goal_request):
        self.get_logger().info("Received demonstration goal request")
        return GoalResponse.ACCEPT

    def get_demonstration_cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def get_demonstration_execute_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info("Getting demonstration")

        executor = self.executor
        result = Demonstration.Result()

        with self._active_goal_lock:
            self._active_goal = goal_handle

        with self.cameraLock:
            for camera in self.camera_names:
                self.demonstration_info[camera] = Image()

        # solver = threading.Thread(target=self.solveSceneCallback, daemon=True)
        # solver.start()
        with ThreadPoolExecutor(max_workers=1) as e:
            future = e.submit(self.solveSceneCallback)

        try:
            # Poll for cancel while solver runs
            while True:
                # if not solver.is_alive():
                #     break
                try:
                    trig_resp: Trigger.Response = future.result(timeout=1.0)
                    break  # finished
                except TimeoutError as e:
                    pass  # still running

                if goal_handle.is_cancel_requested:
                    self.get_logger().warn("Cancel requested; stopping solver")
                    goal_handle.canceled()
                    raise Exception("Goal cancelled")

            # Build final result snapshot under locks
            result = Demonstration.Result()
            with self.armActionLock:
                result.arm_action = self.demonstration_info["arm_action"]
            with self.armStateLock:
                result.arm_state = self.demonstration_info["arm_state"]
            with self.gripperActionLock:
                result.gripper_action = self.demonstration_info["gripper_action"]
            with self.gripperStateLock:
                result.gripper_state = self.demonstration_info["gripper_state"]
            with self.cameraLock:
                result.camera = [self.demonstration_info[camera] for camera in self.camera_names]

            result.header.frame_id = "base"
            result.header.stamp = self.get_clock().now().to_msg()
            result.success = getattr(trig_resp, "success", False)
            result.message = getattr(
                trig_resp, "message", "Place the purple tube into the green rack"
            )

            goal_handle.succeed()
            self.get_logger().info("Successfully executed goal")

        except Exception as e:
            self.get_logger().error(f"Execute error: {e!r}")
            goal_handle.abort()
            result.success = False
            result.message = str(e)
        finally:
            # clear active goal so timer stops publishing
            with self._active_goal_lock:
                self._active_goal = None
            # sleep(0.1)
            # executor.add_node(self)
            # executor.wake()
            return result

    def _feedback_tick(self):
        # Quick, non-blocking; do nothing if no active goal
        with self._active_goal_lock:
            gh = self._active_goal
        if gh is None:
            # self._logger.info("No active goal; skipping feedback tick")
            return

        if (
            len(self.demonstration_info["arm_state"].name) == 0
            or len(self.demonstration_info["arm_action"].name) == 0
            or len(self.demonstration_info["gripper_state"].name) == 0
            or len(self.demonstration_info["gripper_action"].name) == 0
            or any(len(self.demonstration_info[camera].data) == 0 for camera in self.camera_names)
        ):
            return  # try again next tick

        fb = Demonstration.Feedback()
        with self.armActionLock:
            fb.arm_action = self.demonstration_info["arm_action"]
        with self.armStateLock:
            fb.arm_state = self.demonstration_info["arm_state"]
        with self.gripperActionLock:
            fb.gripper_action = self.demonstration_info["gripper_action"]
        with self.gripperStateLock:
            fb.gripper_state = self.demonstration_info["gripper_state"]
        with self.cameraLock:
            fb.camera = [self.demonstration_info[camera] for camera in self.camera_names]

        fb.header.frame_id = "base"
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.message = self.demonstration_info.get("message", "")
        gh.publish_feedback(fb)

    # ======================================================== #
    # ===================== Call services ==================== #
    # ======================================================== #
    @executor_safe
    def callService(self, service: Client, request, message: str | None = None):
        if isinstance(message, str):
            self.get_logger().info(message)
        self.get_logger().info("Calling " + str(service.srv_name) + " service.")
        # future = service.call_async(request)
        # rclpy.spin_until_future_complete(self, future)
        # response = future.result()
        response = service.call(request)
        self.get_logger().info("Called" + str(service.srv_name) + " successfully.")
        return response


def main():
    try:
        rclpy.init(args=None)

        executor = MultiThreadedExecutor(num_threads=4)
        solver = AnalyticSolver("AnalyticSolver")
        executor.add_node(solver)
        solver.addInterfaces()
        executor.spin()

        solver.destroy_node()

        rclpy.shutdown()

    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
