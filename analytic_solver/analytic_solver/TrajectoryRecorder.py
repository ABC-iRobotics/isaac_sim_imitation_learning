# ======================================================== #
# ==================== Python Imports ==================== #
# ======================================================== #
import os
import re
import sys
import threading
from concurrent.futures import Future
from functools import wraps

import cv2

# ======================================================== #
# ===================== ROS 2 Imports ==================== #
# ======================================================== #
import rclpy
import rclpy.action
from cv_bridge import CvBridge
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

# ======================================================== #
# ==================== ROS 2 Messages ==================== #
# ======================================================== #
from std_srvs.srv import Trigger

from isaac_sim_msgs.action import Demonstration
from isaac_sim_msgs.srv import Demonstration as DemoSrv

# ======================================================== #
# ==================== LeRobot Imports =================== #
# ======================================================== #
sys.path.append("../../modules/lerobot")


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


class TrajectoryRecorder(Node):
    def __init__(self, nodename: str):
        super().__init__(nodename)

        self.bridge = CvBridge()

        self.success = False

        self.reentrantGroup = ReentrantCallbackGroup()
        self.mutuallyExclusiveGroup = MutuallyExclusiveCallbackGroup()

        # ~~~~~~~~~~~~~~ ROS Clients ~~~~~~~~~~~~~~ #
        self.newSceneRequest = self.create_client(
            Trigger, "/IsaacSim/NewScene", callback_group=self.reentrantGroup
        )
        self.solveSceneRequest = self.create_client(
            Trigger, "/AnalyticSolver/SolveScene", callback_group=self.reentrantGroup
        )

        # ~~~~~~~~~~~~~~ ROS Servers ~~~~~~~~~~~~~~ #
        self.getTrajectoryService = self.create_service(
            DemoSrv,
            "/TrajectoryRecorder/GetTrajectory",
            self.handle_get_demonstrations,
            callback_group=self.reentrantGroup,
        )
        self.generateTrajectory: rclpy.action.ActionClient = rclpy.action.ActionClient(
            self,
            Demonstration,
            "/AnalyticSolver/GetDemonstration",
            callback_group=self.reentrantGroup,
        )

        self.rate = self.create_rate(50)

        while (
            not self.newSceneRequest.wait_for_service(timeout_sec=5.0)
            and not self.solveSceneRequest.wait_for_service(timeout_sec=5.0)
            and not self.generateTrajectory.wait_for_server(timeout_sec=5.0)
        ):
            if not rclpy.ok():
                self.get_logger().error("Interrupted while waiting for the server.")
                return
            else:
                self.get_logger().info("Server not available, waiting again...")

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

    def get_demonstration_feedback_callback(self, feedback_msg: Demonstration):
        self.get_logger().debug("Got feedback.")
        try:
            # ~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~ #
            observation = {
                "joint_1": feedback_msg.feedback.arm_state.position[0],
                "joint_2": feedback_msg.feedback.arm_state.position[1],
                "joint_3": feedback_msg.feedback.arm_state.position[2],
                "joint_4": feedback_msg.feedback.arm_state.position[3],
                "joint_5": feedback_msg.feedback.arm_state.position[4],
                "joint_6": feedback_msg.feedback.arm_state.position[5],
                "finger_joint": feedback_msg.feedback.gripper_state.position[0],
                "base": self.resize_keep_max_area(
                    cv2.cvtColor(
                        self.bridge.imgmsg_to_cv2(
                            feedback_msg.feedback.camera[0], desired_encoding="bgr8"
                        ),
                        cv2.COLOR_BGR2RGB,
                    )
                ),
                "eih": self.resize_keep_max_area(
                    cv2.cvtColor(
                        self.bridge.imgmsg_to_cv2(
                            feedback_msg.feedback.camera[1], desired_encoding="bgr8"
                        ),
                        cv2.COLOR_BGR2RGB,
                    )
                ),
            }
            observation_frame = build_dataset_frame(
                self.dataset.features, observation, prefix="observation"
            )

            # ~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~ #
            action = {
                "joint_1": feedback_msg.feedback.arm_action.position[0],
                "joint_2": feedback_msg.feedback.arm_action.position[1],
                "joint_3": feedback_msg.feedback.arm_action.position[2],
                "joint_4": feedback_msg.feedback.arm_action.position[3],
                "joint_5": feedback_msg.feedback.arm_action.position[4],
                "joint_6": feedback_msg.feedback.arm_action.position[5],
                "finger_joint": feedback_msg.feedback.gripper_action.position[0],
            }
            action_frame = build_dataset_frame(self.dataset.features, action, prefix="action")

            frame = {**observation_frame, **action_frame, "task": feedback_msg.feedback.message}
            # TODO: Feedback task name in ROS 2 action
            self.dataset.add_frame(frame)

            self.start = False
        except Exception as e:
            self.get_logger().info("Error: " + str(e))

    def getTrajectoryServiceCallback(self, request: DemoSrv.Request, response):
        self.get_logger().info("Received getTrajectory request")

        response = DemoSrv.Response()

        # ~~~~~~~~~~~~~~~~~ Inputs ~~~~~~~~~~~~~~~~ #
        amount = request.amount
        path = request.path

        # ~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~ #
        obs_features = {
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
            "joint_6": float,
            "finger_joint": float,
            "base": (720, 1280, 3),
            "eih": (720, 1280, 3),  # (1216, 1936, 3),
        }

        # ~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~ #
        action_features = {
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
            "joint_6": float,
            "finger_joint": float,
        }

        obs_features = hw_to_dataset_features(obs_features, "observation", use_video=True)
        action_features = hw_to_dataset_features(action_features, "action", use_video=True)
        dataset_features = {**action_features, **obs_features}

        # ~~~~~~~~~~~~~ Create dataset ~~~~~~~~~~~~ #
        repo_id = "dataset_" + str(self.find_max_dataset_number(path) + 1)
        self.dataset: LeRobotDataset
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=20,
            features=dataset_features,
            root=os.path.join(os.path.expanduser(path), repo_id),
            robot_type="tm5-900",
            use_videos=True,
            image_writer_threads=10,
            image_writer_processes=10,
            tolerance_s=0.25,
        )

        # ~~~~~~~~~~ Start demonstrations ~~~~~~~~~ #
        for _ in range(amount):
            self.success = False
            while not self.success:
                try:
                    # Empty buffer in case of unsuccessful previous episode
                    self.dataset.clear_episode_buffer()
                    self.start = True
                    self.done_execution = False
                    self.callService(
                        service=self.newSceneRequest, request=Trigger.Request()
                    )  # Generate new scene

                    # executor = self.executor
                    self.send_goal_future: Future[
                        ClientGoalHandle
                    ] = self.generateTrajectory.send_goal_async(
                        Demonstration.Goal(),
                        feedback_callback=self.get_demonstration_feedback_callback,
                    )  # Call demonstration generation action

                    while not self.send_goal_future.done():
                        executor_safe(rclpy.spin_once)(self)

                    self.goal_handle: ClientGoalHandle = self.send_goal_future.result()
                    if self.goal_handle is None:
                        self.get_logger().info("Goal was aborted")
                        continue
                    if not self.goal_handle.accepted:
                        self.get_logger().info("Goal rejected by server")
                        return
                    else:
                        self.get_logger().info("Goal accepted by server, waiting for result")

                    # self.result_future : Future[Demonstration.Result] = self.goal_handle.get_result_async()
                    # while not self.result_future.done():
                    #     executor_safe(rclpy.spin_once)(self)
                    # self.success = self.result_future.result().result.success
                    self.success = self.goal_handle.get_result().result.success
                except Exception as e:
                    self.get_logger().info("Exception at demo generation: " + str(e))

            self.get_logger().info("Got successful demonstration")
            self.dataset.save_episode()

        response.success = True
        response.message = ""
        return response

    @executor_safe
    def callService(self, service: Client, request, message: str | None = None):
        if isinstance(message, str):
            self.get_logger().info(message)
        self.get_logger().info("Calling " + str(service.srv_name) + " service.")
        response = service.call(request)
        self.get_logger().info("Called" + str(service.srv_name) + " successfully.")
        return response

    def find_max_dataset_number(self, path):
        max_number = -1
        pattern = re.compile(r"^dataset_(\d+)$")
        directory = os.path.expanduser(path)
        for name in os.listdir(directory):
            match = pattern.match(name)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)

        return max_number

    def handle_get_demonstrations(self, request, response):
        """Service callback to start the demonstration sequence."""
        self.get_logger().info("Received getTrajectory request")

        response = DemoSrv.Response()

        # ~~~~~~~~~~~~~~~~~ Inputs ~~~~~~~~~~~~~~~~ #
        amount = request.amount
        path = request.path

        # ~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~ #
        obs_features = {
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
            "joint_6": float,
            "finger_joint": float,
            "base": (720, 1280, 3),
            "eih": (720, 1280, 3),  # (1216, 1936, 3),
        }

        # ~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~ #
        action_features = {
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
            "joint_6": float,
            "finger_joint": float,
        }

        obs_features = hw_to_dataset_features(obs_features, "observation", use_video=True)
        action_features = hw_to_dataset_features(action_features, "action", use_video=True)
        dataset_features = {**action_features, **obs_features}

        # ~~~~~~~~~~~~~ Create dataset ~~~~~~~~~~~~ #
        repo_id = "dataset_" + str(self.find_max_dataset_number(path) + 1)
        self.dataset: LeRobotDataset
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=20,
            features=dataset_features,
            root=os.path.join(os.path.expanduser(path), repo_id),
            robot_type="tm5-900",
            use_videos=True,
            image_writer_threads=10,
            image_writer_processes=10,
            tolerance_s=0.25,
        )

        # ~~~~~~~~~~ Start demonstrations ~~~~~~~~~ #

        thread = threading.Thread(
            target=self._run_demonstration_sequence, kwargs={"amount": amount}, daemon=True
        )
        thread.start()
        thread.join()

        response.success = True
        response.message = ""
        return response

    def _run_demonstration_sequence(self, amount: int):
        """Run a sequence of action calls in a separate thread, one after the other."""
        # Wait until the action server is available (to avoid sending goals to
        # a non-existent server)
        if not self.solveSceneRequest.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "Action server '/AnalyticSolver/GetDemonstration' not available, aborting sequence."
            )
            return
        if not self.newSceneRequest.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "Service server '/IsaacSim/NewScene' not available, aborting sequence."
            )
            return

        for _ in range(amount):
            self.success = False
            while not self.success:
                try:
                    # Empty buffer in case of unsuccessful previous episode
                    self.dataset.clear_episode_buffer()
                    self.start = True
                    self.done_execution = False
                    self.callService(
                        service=self.newSceneRequest, request=Trigger.Request()
                    )  # Generate new scene

                    # Prepare the goal message (using Fibonacci goal for
                    # example)
                    goal_msg = Demonstration.Goal()

                    # Create a fresh Event to wait for this goal's result
                    done_event = threading.Event()

                    # Send the goal asynchronously with a feedback callback
                    send_future: Future[ClientGoalHandle] = self.generateTrajectory.send_goal_async(
                        goal_msg, feedback_callback=self.get_demonstration_feedback_callback
                    )
                    # When the goal is done (accepted/rejected), call
                    # _goal_response_callback
                    send_future.add_done_callback(
                        lambda fut, ev=done_event: self._goal_response_callback(fut, ev)
                    )

                    # Wait for the action to complete (the event will be set in
                    # the result callback)
                    done_event.wait()  # This blocks the thread until the goal is done, without busy-looping

                except Exception as e:
                    self.get_logger().info("Exception at demo generation: " + str(e))

            self.get_logger().info("Got successful demonstration")
            self.dataset.save_episode()

        self.dataset.finalize()
        self.get_logger().info("All demonstration action calls completed.")

    def _goal_response_callback(self, future, done_event):
        """Done callback for send_goal_async to handle goal acceptance and result request."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Action goal was rejected by the server.")
            done_event.set()  # No result will come; unblock the waiting thread
            return

        self.get_logger().info("Action goal accepted; waiting for result...")
        # Request the result asynchronously
        result_future: Future[Demonstration.Result] = goal_handle.get_result_async()
        # When the result is ready, call _result_callback (pass along the same
        # event to signal completion)
        result_future.add_done_callback(
            lambda res_fut, ev=done_event: self._result_callback(res_fut, ev)
        )

    def _result_callback(self, result_future, done_event):
        """Done callback for get_result_async to handle the action result."""
        self.success = result_future.result().result.success
        self.get_logger().info(f"Action result received; success: {self.success}")
        done_event.set()


def main():

    # Init ROS2
    try:
        rclpy.init(args=None)

        executor = MultiThreadedExecutor()
        recorder = TrajectoryRecorder("TrajectoryRecorder")
        executor.add_node(recorder)
        executor.spin()

        recorder.destroy_node()

        rclpy.shutdown()

    except (KeyboardInterrupt, ExternalShutdownException):
        if isinstance(recorder, TrajectoryRecorder) and isinstance(
            recorder.dataset, LeRobotDataset
        ):
            recorder.dataset.clear_episode_buffer()


if __name__ == "__main__":
    main()
