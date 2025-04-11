# ======================================================== #
# ===================== ROS 2 Imports ==================== #
# ======================================================== #
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup


# ======================================================== #
# ==================== Message Imports =================== #
# ======================================================== #
from std_srvs.srv import Trigger
from geometry_msgs.msg import Transform, Pose, PoseStamped, Pose2D, Vector3
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand
from moveit_msgs.msg import RobotTrajectory
from isaac_sim_msgs.srv import PoseRequest, TubeParameter
from onrobot_rg_msgs.srv import GripperPose


# ======================================================== #
# =================== MoveIt 2 Imports =================== #
# ======================================================== #
from moveit.planning import MoveItPy
from moveit.planning import PlanningComponent
from moveit.planning import TrajectoryExecutionManager
from moveit.core.robot_state import RobotState
from moveit.planning import PlanningSceneMonitor
from moveit.planning import PlanRequestParameters


# ======================================================== #
# ==================== Python Imports ==================== #
# ======================================================== #
from time import sleep
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation


# ======================================================== #
# ====================== Own Imports ===================== #
# ======================================================== #
from analytical_solver.moveit_interface.moveit2 import MoveIt2



callback_group = ReentrantCallbackGroup()

class AnalyticalSolver(Node):

    # * Target prim's path to solve the task
    target = '/World/Tubes/Tube_Target'

    def __init__(self, nodename : str, robot : MoveItPy, robot_interface : MoveIt2):
        super().__init__(nodename)

        self.targetPose : Transform
        self.jointState : JointState = JointState()
        self.robotState : RobotState
        self.stopped = False
        self.lastJointState : JointState = JointState()

        self.mutually_exclusive_group = MutuallyExclusiveCallbackGroup()
        self.reentrant_group = ReentrantCallbackGroup()
        

        self.robot_interface = robot_interface
        self.robot_interface.planner_id = ('ompl.planner_configs.RRTConnectkConfigDefault')
        self.robot_interface.max_velocity = 0.5
        self.robot_interface.max_acceleration = 0.2
        self.robot_interface.cartesian_avoid_collisions = False
        self.robot_interface.cartesian_jump_threshold = 0.0

        # Wait for IsaacSim node to set up
        self.robot_velocity = self.create_subscription(JointState, '/joint_states', self.refreshRobotValues, 10, callback_group=self.reentrant_group)
        self.poseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestPose', callback_group=self.mutually_exclusive_group)
        self.tubeGraspPoseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestTubeGraspPose', callback_group=self.mutually_exclusive_group)
        self.gripperPoseRequest = self.create_client(GripperPose, '/onrobot/pose', callback_group=self.mutually_exclusive_group)
        self.tubeParameterRequest = self.create_client(TubeParameter, '/IsaacSim/RequestTubeParameter', callback_group=self.mutually_exclusive_group)
        self.gripperController = rclpy.action.ActionClient(self, GripperCommand, '/onrobot_controller', callback_group=self.reentrant_group)
        
        while not (self.poseRequest.wait_for_service(timeout_sec=5.0) and 
            self.tubeGraspPoseRequest.wait_for_service(timeout_sec=5.0) and 
            self.gripperPoseRequest.wait_for_service(timeout_sec=5.0) and
            self.tubeParameterRequest.wait_for_service(timeout_sec=5.0)
            ):
            if not rclpy.ok():
                self.get_logger().error('Interruped while waiting for the servers.')
                return
            else:
                self.get_logger().info('Servers not available, waiting again...')

        self.move_to_tube = self.create_service(Trigger, '/AnalyticSolver/MoveToTube', self.moveToTubeCallback, callback_group=self.reentrant_group)


        self.robot = robot
        self.robot_arm : PlanningComponent = robot.get_planning_component('tmr_arm')
        self.robot_gripper : PlanningComponent = robot.get_planning_component('rg6')
        self.get_logger().info('MoveItPy instance created')
        self.execution_manager : TrajectoryExecutionManager = self.robot.get_trajectory_execution_manager()
        self.planningscene : PlanningSceneMonitor = self.robot.get_planning_scene_monitor()

        self.robot_arm.set_start_state_to_current_state()

        self.robot_arm.set_goal_state(configuration_name='ready1')

        self.moveRobotArmJointSpace()

        # sleep(2)

        # goal : PoseStamped = PoseStamped()
        # goal.header.stamp = self.get_clock().now().to_msg()
        # goal.header.frame_id = 'base'
        # goal.pose = self.robotState.get_pose('flange')
        # goal.pose.position.x = goal.pose.position.x + 0.3

        # self.moveRobotArmCartesianSpace(goal)



    def refreshRobotValues(self, msg : JointState):
            if any(joint_name == 'joint_1' for joint_name in msg.name):
                self.lastJointState = self.jointState
                self.jointState = msg
                #self.get_logger().info('[' + ', '.join([str(self.jointState.name[idx]) + ': ' + str(velocity) for idx, velocity in enumerate(self.jointState.velocity) if 'joint_' in self.jointState.name[idx]]) +']')
                self.stopped = True if all(abs(pos - last_pos) < 1e-3 and 
                                    abs(vel - last_vel) < 1e-3 for pos, last_pos, vel, last_vel, joint in 
                                    zip(self.jointState.position, self.lastJointState.position, self.jointState.velocity, self.lastJointState.velocity, self.jointState.name) 
                                    if 'joint_' in joint) else False
                
                

    def targetPose(self):
        self.get_logger().info('Calling RequestPose')
        while not self.poseRequest.wait_for_service(timeout_sec=30.0):
            if not rclpy.ok():
                self.get_logger().error('Interruped while waiting for the server.')
                return
            else:
                self.get_logger().info('Server not available, waiting again...')

        self.request = PoseRequest.Request()
        self.request.path = self.target
        self.future = self.poseRequest.call_async(self.request)
        self.future.add_done_callback(self.targetPoseCallback)

    def targetPoseCallback(self, future):
        response = future.result()
        self.targetPose = response.pose
        self.get_logger().info('Received response')

    def moveToTubeCallback(self, request : Trigger.Request, response : Trigger.Response):
        self.get_logger().info('Move to tube request recived')

        request : PoseRequest.Request = PoseRequest.Request()
        request.path = '/World/Tubes/Tube_Target'
        future  = self.tubeGraspPoseRequest.call_async(request)
        self.get_logger().info('Start spin')
        rclpy.spin_once(self)
        while not future.done():
            rclpy.spin_once(self)
            sleep(0.1)
        while isinstance(future.result(), type(None)):
            rclpy.spin_once(self)
            sleep(0.1)

        self.get_logger().info('Done pose request')
        try:
            self.targetPose : PoseRequest.Response = future.result()
            self.get_logger().info(str(self.targetPose))
            self.targetPose : Transform = self.targetPose.pose
            self.get_logger().info('Got pose!')
            
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = '/World/Racks/Rack_Start'
            
            future  = self.poseRequest.call_async(request)
            while not future.done():
                rclpy.spin_once(self)
                sleep(0.1)
            while isinstance(future.result(), type(None)):
                rclpy.spin_once(self)
                sleep(0.1)
                
            self.rackPose : Transform = future.result().pose
    
    
            request : TubeParameter.Request = TubeParameter.Request()
            request.path = '/World/Tubes/Tube_Target'
            future  = self.tubeParameterRequest.call_async(request)
            while not future.done():
                rclpy.spin_once(self)
                sleep(0.1)
            while isinstance(future.result(), type(None)):
                rclpy.spin_once(self)
                sleep(0.1)
                
            self.tubeDimensions = [future.result().dimensions.x, future.result().dimensions.y, future.result().dimensions.z]
            self.tubeWidth = min(self.tubeDimensions)
            self.tubeHeight = max(self.tubeDimensions)
            self.get_logger().info('Height: ' + str(self.tubeHeight) + ' Width: ' + str(self.tubeWidth))
            request : GripperPose.Request = GripperPose.Request()
            request.known.x = self.tubeWidth - 0.004
            
            future  = self.gripperPoseRequest.call_async(request)
            while not future.done():
                rclpy.spin_once(self)
                sleep(0.1)
            while isinstance(future.result(), type(None)):
                rclpy.spin_once(self)
                sleep(0.1)
                
            self.gripper_goal : Pose2D = future.result().pose
            self.get_logger().info('Gripper width: ' + str(self.gripper_goal.x) + ' Gripper height: ' + str(self.gripper_goal.y))
            
            self.moveRobotGripper(np.deg2rad(-35), 200.0)
            
            self.robot_arm.set_start_state_to_current_state()
    
            goal : PoseStamped = PoseStamped()
            goal.header.frame_id = "base"
            goal.header.stamp = self.get_clock().now().to_msg()
    
    
            # ! TODO: wtf is going on here?
            tube_orientation = Rotation.from_quat([self.targetPose.rotation.x, self.targetPose.rotation.y, self.targetPose.rotation.z, self.targetPose.rotation.w]).as_euler('xyz')
            tube_orientation = Rotation.from_euler('xyz', [tube_orientation[0], tube_orientation[1], 0.0])
    
            orientation : Rotation = Rotation.from_quat([self.rackPose.rotation.x, self.rackPose.rotation.y, self.rackPose.rotation.z, self.rackPose.rotation.w])
            orientation : Rotation = Rotation.from_euler('zxy', [-orientation.as_euler('zxy')[0], np.pi/2+tube_orientation.as_euler('zxy')[1], tube_orientation.as_euler('zxy')[2]])
            tube_orientation.apply(orientation.as_matrix())

            goal.pose.orientation.x = orientation.as_quat()[0]
            goal.pose.orientation.y = orientation.as_quat()[1]
            goal.pose.orientation.z = orientation.as_quat()[2]
            goal.pose.orientation.w = orientation.as_quat()[3]
            goal.pose.position.x = self.targetPose.translation.x
            goal.pose.position.y = self.targetPose.translation.y
            goal.pose.position.z = self.targetPose.translation.z + self.gripper_goal.y + 0.2
    
            self.get_logger().info('X: ' + str(goal.pose.position.x) + ' Y: ' + str(goal.pose.position.y) + ' Z: ' + str(goal.pose.position.z))
    
            self.robot_arm.set_goal_state(pose_stamped_msg=goal, pose_link="flange")
    
            self.moveRobotArmJointSpace()
            self.get_logger().info('Done joint space movement')
            rclpy.spin_once(self)
            
            self.robot_arm.set_start_state_to_current_state()
            goal.pose.position.z -= 0.2
            goal.header.stamp = self.get_clock().now().to_msg()
    
            self.moveRobotArmCartesianSpace(goal)
            
            sleep(1)

            self.moveRobotGripper(self.gripper_goal.theta, 200.0)
            
            
            
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        

        response.success = True
        response.message = ''

        return response

    def moveRobotArmJointSpace(
        self,
        single_plan_parameters=None,
        multi_plan_parameters=None,
        ):
        """A helper function to plan and execute a motion."""
        # plan to goal
        self.get_logger().info("Planning trajectory")
        if multi_plan_parameters is not None:
                plan_result = self.robot_arm.plan(
                        multi_plan_parameters=multi_plan_parameters
                )
        elif single_plan_parameters is not None:
                plan_result = self.robot_arm.plan(
                        single_plan_parameters=single_plan_parameters
                )
        else:
                plan_result = self.robot_arm.plan()

        # execute the plan
        if plan_result:
                self.get_logger().info("Executing plan")
                robot_trajectory : RobotTrajectory = plan_result.trajectory
                self.get_logger().info(str(robot_trajectory))
                self.robot.execute(robot_trajectory, controllers=[])
                self.execution_manager.wait_for_trajectory_completion()
                sleep(1)
                rclpy.spin_once(self)
                while not self.stopped:
                    rclpy.spin_once(self)
                self.stopped = False
                sleep(1)
                self.get_logger().info('Movement completed!')
        else:
                self.get_logger().error("Planning failed")

        self.robot_arm.set_start_state_to_current_state()
        self.robotState = self.robot_arm.get_start_state()
        self.planningscene.clear_octomap()

    def moveRobotArmCartesianSpace(self, pose : Pose | PoseStamped):
        self.get_logger().info('Start planing and moving')
        self.robot_interface.move_to_pose(
                pose=pose,
                cartesian=True,
                cartesian_max_step=0.000025,
                cartesian_fraction_threshold=0.000001,
            )
        self.execution_manager.wait_for_trajectory_completion()
        self.get_logger().info('Execution done')
        sleep(1)
        rclpy.spin_once(self)
        while not self.stopped:
            rclpy.spin_once(self)
        self.stopped = False
        sleep(1)
        self.get_logger().info('Movement completed!')

        self.robot_arm.set_start_state_to_current_state()
        self.robotState = self.robot_arm.get_start_state()
        self.planningscene.clear_octomap()

    def moveRobotGripper(self, joint_angle : float, effort : float):
        self._done_gripper_execute = False
        goal : GripperCommand.Goal = GripperCommand.Goal()
        goal.command.position = joint_angle
        goal.command.max_effort = effort
        self.send_goal_future = self.gripperController.send_goal_async(
            goal, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)
        while not self._done_gripper_execute:
            sleep(0.1)
            rclpy.spin_once(self)
        self._done_gripper_execute = False

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by server')
            return
        else:
            self.get_logger().info('Goal accepted by server, waiting for result')
    
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback')
    
    def result_callback(self, future):
        result = future.result().result
        self._done_gripper_execute = True
        self.get_logger().info('Server successfully executed goal')

def main():
    try:
        rclpy.init(args=None)

        executor = MultiThreadedExecutor()
        node = Node('move_group_interface')

        robot = MoveItPy(node_name='MoveItPy')

        robot_interface = MoveIt2(
            node,
            joint_names=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'],
            base_link_name='base',
            end_effector_name='flange',
            group_name='tmr_arm',
            callback_group=callback_group
        )

        solver = AnalyticalSolver('AnalyticalSolver', robot, robot_interface)

        executor.add_node(node)
        #executor.add_node(robot_interface)
        executor.add_node(solver)
        executor.spin()

        solver.destroy_node()
        robot.shutdown()

        rclpy.shutdown()

    except(KeyboardInterrupt, ExternalShutdownException):
        pass

if __name__ == '__main__':
    main()
