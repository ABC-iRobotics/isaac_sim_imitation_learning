# ======================================================== #
# ===================== ROS 2 Imports ==================== #
# ======================================================== #
import rclpy
import rclpy.action
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.client import Client


# ======================================================== #
# ==================== Message Imports =================== #
# ======================================================== #
from std_msgs.msg import Bool
from geometry_msgs.msg import Transform, Pose, PoseStamped, Pose2D, Vector3
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory

from std_srvs.srv import Trigger
from isaac_sim_msgs.srv import PoseRequest, TubeParameter
from onrobot_rg_msgs.srv import GripperPose

from control_msgs.action import GripperCommand

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
from typing import Optional


# ======================================================== #
# ====================== Own Imports ===================== #
# ======================================================== #
from analytical_solver.moveit_interface.moveit2 import MoveIt2



callback_group = ReentrantCallbackGroup()

class AnalyticalSolver(Node):

    # * Target prim's path to solve the task
    target = '/World/Tubes/Tube_Target'

    def __init__(self, nodename : str):
        super().__init__(nodename)

        self.targetPose : Transform
        self.jointState : JointState = JointState()
        self.robotState : RobotState
        self.gripperState : bool = False
        self.stopped = False
        self.lastJointState : JointState = JointState()

        self.mutually_exclusive_group = MutuallyExclusiveCallbackGroup()
        self.reentrant_group = ReentrantCallbackGroup()
        

        self.robot_interface = MoveIt2(
            Node('move_group_interface'),
            joint_names=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'],
            base_link_name='base',
            end_effector_name='flange',
            group_name='tmr_arm',
            callback_group=ReentrantCallbackGroup()
        )
        
        #self.robot_interface = robot_interface
        self.robot_interface.planner_id = ('STRIDEkConfigDefault')
        self.robot_interface.max_velocity = 0.5
        self.robot_interface.max_acceleration = 0.2
        self.robot_interface.cartesian_avoid_collisions = False
        self.robot_interface.cartesian_jump_threshold = 0.0
        
        self.ready1 = [np.pi/2, 0.0, np.pi/2, 0.0, np.pi/2, 0.0]

        # Wait for IsaacSim node to set up
        self.robot_velocity = self.create_subscription(JointState, '/joint_states', self.refreshRobotValues, 10, callback_group=self.reentrant_group)
        self.poseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestPose', callback_group=self.mutually_exclusive_group)
        self.tubeGraspPoseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestTubeGraspPose', callback_group=self.mutually_exclusive_group)
        self.gripperPoseRequest = self.create_client(GripperPose, '/onrobot/pose', callback_group=self.mutually_exclusive_group)
        self.tubeParameterRequest = self.create_client(TubeParameter, '/IsaacSim/RequestTubeParameter', callback_group=self.mutually_exclusive_group)
        self.gripperController = rclpy.action.ActionClient(self, GripperCommand, '/onrobot_controller', callback_group=self.reentrant_group)
        self.closeGripper = self.create_client(Trigger, '/IsaacSim/CloseGripper', callback_group=self.reentrant_group)
        self.goalPoseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestTubeGoalPose', callback_group=self.reentrant_group)
        
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

        self.robot_interface.move_to_configuration(self.ready1, ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])

    def callService(self, service : Client, request, message : str | None = None):
        if isinstance(message, str):
            self.get_logger().info(message)
        self.get_logger().info('Calling ' + str(service.srv_name) + ' service.')
        future  = service.call_async(request)
        rclpy.spin_once(self)
        while not future.done():
            rclpy.spin_once(self)
            sleep(0.01)
        while isinstance(future.result(), type(None)):
            rclpy.spin_once(self)
            sleep(0.01)
        rclpy.spin_once(self)
        sleep(0.01)
        rclpy.spin_once(self)
        response = future.result()
        self.get_logger().info('Called' + str(service.srv_name) + ' successfully.')
        return response
        

    def gripperStateCallback(self, msg : Bool):
        self.gripperState = msg.data

    def refreshRobotValues(self, msg : JointState):
            if any(joint_name == 'joint_1' for joint_name in msg.name):
                self.lastJointState = self.jointState
                self.jointState = msg
                #self.get_logger().info('[' + ', '.join([str(self.jointState.name[idx]) + ': ' + str(velocity) for idx, velocity in enumerate(self.jointState.velocity) if 'joint_' in self.jointState.name[idx]]) +']')
                self.stopped = True if all(abs(pos - last_pos) < 1e-3 and 
                                    abs(vel - last_vel) < 1e-3 for pos, last_pos, vel, last_vel, joint in 
                                    zip(self.jointState.position, self.lastJointState.position, self.jointState.velocity, self.lastJointState.velocity, self.jointState.name) 
                                    if 'joint_' in joint) else False

    def moveToTubeCallback(self, request : Trigger.Request, response : Trigger.Response):
        self.get_logger().info('Move to tube request recived')
        try:
            # ~~~~~~~~~~~~~ Get grasp pose ~~~~~~~~~~~~ #
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = '/World/Tubes/Tube_Target'
            self.grasp_pose : Transform = self.callService(service=self.tubeGraspPoseRequest, request=request).pose
            
            # ~~~~~~~~~~ Get tube parameters ~~~~~~~~~~ #
            request : TubeParameter.Request = TubeParameter.Request()
            request.path = '/World/Tubes/Tube_Target'
            self.tubeDimensions = self.callService(service=self.tubeParameterRequest, request=request)
            self.tubeDimensions = [self.tubeDimensions.dimensions.x, self.tubeDimensions.dimensions.y, self.tubeDimensions.dimensions.z]
            self.tubeWidth = min(self.tubeDimensions)
            self.tubeHeight = max(self.tubeDimensions)
            self.get_logger().debug('Height: ' + str(self.tubeHeight) + ' Width: ' + str(self.tubeWidth))
            
            # ~~~~~~~ Get start rack parameters ~~~~~~~ #
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = '/World/Racks/Rack_Start'
            self.rackPose : Transform = self.callService(service=self.poseRequest, request=request).pose
                
            #  Set gripper to open and calculate grasp parameters  #
            request : GripperPose.Request = GripperPose.Request()
            request.known.x = self.tubeWidth
            self.gripper_goal : Pose2D = self.callService(service=self.gripperPoseRequest, request=request).pose
            self.get_logger().debug('Gripper width: ' + str(self.gripper_goal.x) + ' Gripper height: ' + str(self.gripper_goal.y))
            
            self.moveRobotGripper(np.deg2rad(-35), 200.0)
        
            # ~~~~~~~~~~~~ Move above tube ~~~~~~~~~~~~ #
            goal : PoseStamped = PoseStamped()
            goal.header.frame_id = "base"
            goal.header.stamp = self.get_clock().now().to_msg()
            
            request : TubeParameter.Request = TubeParameter.Request()
            request.path = '/World/Tubes/Tube_Target'
            self.tubeDimensions : TubeParameter.Response = self.callService(service=self.tubeParameterRequest, request=request)
            self.get_logger().debug(str(self.tubeDimensions))
            tube_orientation : Rotation = Rotation.from_quat([self.tubeDimensions.pose.orientation.x, self.tubeDimensions.pose.orientation.y, self.tubeDimensions.pose.orientation.z, self.tubeDimensions.pose.orientation.w])
            tube_orientation = Rotation.from_euler('xyz', [np.pi/2, 0.0, 0.0]) * tube_orientation
            
            rack_orientation : Rotation = Rotation.from_quat([self.rackPose.rotation.x, self.rackPose.rotation.y, self.rackPose.rotation.z, self.rackPose.rotation.w])

            orientation : Rotation = \
                Rotation.from_euler('xyz', [-(np.pi-tube_orientation.as_euler('xyz')[0]), tube_orientation.as_euler('xyz')[1], rack_orientation.as_euler('xyz')[2]])
            
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y+0.2])
            
            goal.pose.orientation.x = orientation.as_quat()[0]
            goal.pose.orientation.y = orientation.as_quat()[1]
            goal.pose.orientation.z = orientation.as_quat()[2]
            goal.pose.orientation.w = orientation.as_quat()[3]
            
            goal.pose.position.x = self.grasp_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = self.grasp_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = self.grasp_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
            
            self.moveRobotArm(goal, cartesian=False, velocity=0.5, acceleration=0.5)
            
            sleep(0.1)
            rclpy.spin_once(self)
        
        
            # ~~~~~~~~~~~~~ Approach tube ~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y]) 
            goal.pose.position.x = self.grasp_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = self.grasp_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = self.grasp_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
        
            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.5, acceleration=0.5)
            
            sleep(1)
        
            # ~ Close gripper and use surface gripper ~ #
            self.moveRobotGripper(self.gripper_goal.theta, 200.0)
            
            self.get_logger().debug('Activating Surface Gripper')
            request : Trigger.Request = Trigger.Request()
            self.callService(service=self.closeGripper, request=request)
            self.get_logger().debug('Surface Gripper activated')
            
            sleep(1)
            
            # ~~~~~~~~~~~~~ Pull out tube ~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y + 0.2]) 
            goal.pose.position.x = self.grasp_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = self.grasp_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = self.grasp_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
            
            self.moveRobotArm(goal, velocity=0.05, acceleration=0.05)
            
            sleep(2)
            rclpy.spin_once(self)
            
            # ~~~~~~~~~~~ Approach goal rack ~~~~~~~~~~ #
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = '/World/Racks/Rack_Goal'
            rack_pose : Transform = self.callService(service=self.goalPoseRequest, request=request).pose
            self.get_logger().debug(str(rack_pose))
            rack_orientation : Rotation = Rotation.from_quat([rack_pose.rotation.x, rack_pose.rotation.y, rack_pose.rotation.z, rack_pose.rotation.w])
            
            request : TubeParameter.Request = TubeParameter.Request()
            request.path = '/World/Tubes/Tube_Target'
            self.tubeDimensions : TubeParameter.Response = self.callService(service=self.tubeParameterRequest, request=request)
            self.get_logger().debug(str(self.tubeDimensions))
            tube_orientation : Rotation = Rotation.from_quat([self.tubeDimensions.pose.orientation.x, self.tubeDimensions.pose.orientation.y, self.tubeDimensions.pose.orientation.z, self.tubeDimensions.pose.orientation.w])
            tube_orientation = Rotation.from_euler('xyz', [np.pi/2, 0.0, 0.0]) * tube_orientation
              
            self.get_logger().debug('Tube orientation: ' + str(tube_orientation.as_euler('xyz', degrees=True)))
            self.get_logger().debug('Rack orientation: ' + str(rack_orientation.as_euler('xyz', degrees=True)))
            
            orientation : Rotation = \
                Rotation.from_euler('xyz', [-(np.pi-tube_orientation.as_euler('xyz')[0]), tube_orientation.as_euler('xyz')[1], rack_orientation.as_euler('xyz')[2]])
            
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y+0.2])
            
            goal.pose.orientation.x = orientation.as_quat()[0]
            goal.pose.orientation.y = orientation.as_quat()[1]
            goal.pose.orientation.z = orientation.as_quat()[2]
            goal.pose.orientation.w = orientation.as_quat()[3]
            
            goal.pose.position.x = rack_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = rack_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = rack_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
            
            self.get_logger().debug('Rotation: ' + str(orientation.as_euler('xyz', degrees=True)))
            self.get_logger().debug('Target position: [' + str(rack_pose.translation.x - gripper_translation[0]) + ', ' +\
                str(rack_pose.translation.y - gripper_translation[1]) + ', ' +\
                str(rack_pose.translation.z - gripper_translation[2]) + ']')
            self.moveRobotArm(goal, cartesian=False, velocity=0.5, acceleration=0.5)
            
            # ~~~~~~~~~~~~~~ Put in tube ~~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y]) 
            goal.pose.position.x = rack_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = rack_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = rack_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
        
            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.05, acceleration=0.05)
            
            sleep(1)
            
            # ~ Open gripper and use surface gripper ~ #
            self.get_logger().debug('Activating Surface Gripper')
            request : Trigger.Request = Trigger.Request()
            self.callService(service=self.closeGripper, request=request)
            self.get_logger().debug('Surface Gripper activated')
            
            self.moveRobotGripper(0.0, 200.0)
            
            sleep(1)
            
            # ~~~~~~~~~~~~~~ Get away from tube ~~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y + 0.2]) 
            goal.pose.position.x = rack_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = rack_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = rack_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
        
            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.5, acceleration=0.5)
            
            sleep(1)
            
            # ~~~~~~~~~ Go home configuration ~~~~~~~~~ #
            self.robot_interface.move_to_configuration(self.ready1, ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
            
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        

        response.success = True
        response.message = ''

        return response

    def moveRobotArm(self,  pose : Pose | PoseStamped | None = None, 
                            position : tuple[float, float, float] | None = None, orientation : tuple[float, float, float, float] | None = None,
                            velocity : float = 0.0, acceleration : float = 0.0,
                            cartesian : bool = False):
        self.get_logger().info('Start planing and moving')
        if not velocity == 0.0:
            self.robot_interface.max_velocity = velocity
        if not acceleration == 0.0:
            self.robot_interface.max_acceleration = acceleration
        if isinstance(pose, Pose) or isinstance(pose, PoseStamped):
            self.robot_interface.move_to_pose(
                pose=pose,
                cartesian=cartesian,
                cartesian_max_step=0.000025,
                cartesian_fraction_threshold=0.000001,
            )
        elif isinstance(position, tuple[float, float, float]) and isinstance(orientation, tuple[float, float, float, float]):
            self.robot_interface.move_to_pose(
                position=position,
                quat_xyzw=orientation,
                cartesian=cartesian,
                cartesian_max_step=0.000025,
                cartesian_fraction_threshold=0.000001,
            )
        self.robot_interface.wait_until_executed()
        self.get_logger().info('Execution done')
        sleep(0.1)
        rclpy.spin_once(self)
        while not self.stopped:
            rclpy.spin_once(self)
        self.stopped = False
        self.get_logger().info('Movement completed!')

    def moveRobotGripper(self, joint_angle : float, effort : float):
        
        goal : GripperCommand.Goal = GripperCommand.Goal()
        goal.command.position = joint_angle
        goal.command.max_effort = effort
        self.send_goal_future = self.gripperController.send_goal_async(
            goal, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)
        sleep(1)
        self._done_gripper_execute = False
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

    def changeGripperState(self, state : bool):
        if not self.gripperState == state:
            
            self.gripperState = not self.gripperState
           
def main():
    try:
        rclpy.init(args=None)

        executor = MultiThreadedExecutor()
        solver = AnalyticalSolver('AnalyticalSolver')
        executor.add_node(solver)
        executor.add_node(solver.robot_interface._node)
        executor.spin()

        solver.destroy_node()

        rclpy.shutdown()

    except(KeyboardInterrupt, ExternalShutdownException):
        pass

if __name__ == '__main__':
    main()
