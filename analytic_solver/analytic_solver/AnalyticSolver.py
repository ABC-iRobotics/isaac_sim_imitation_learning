# ======================================================== #
# ===================== ROS 2 Imports ==================== #
# ======================================================== #
import rclpy
import rclpy.action
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.client import Client
from rclpy.action import GoalResponse, CancelResponse


# ======================================================== #
# ==================== Message Imports =================== #
# ======================================================== #
from std_msgs.msg import Header
from geometry_msgs.msg import Transform, Pose, PoseStamped, Pose2D
from sensor_msgs.msg import JointState, Image

from std_srvs.srv import Trigger, SetBool
from isaac_sim_msgs.srv import PoseRequest, TubeParameter, CollisionRequest
from onrobot_rg_msgs.srv import GripperPose

from control_msgs.action import GripperCommand
from isaac_sim_msgs.action import Demonstration


# ======================================================== #
# ==================== Python Imports ==================== #
# ======================================================== #
import time
from time import sleep
import numpy as np
from scipy.spatial.transform import Rotation


# ======================================================== #
# ====================== Own Imports ===================== #
# ======================================================== #
from analytic_solver.moveit_interface.moveit2 import MoveIt2



class AnalyticSolver(Node):

    def __init__(self, nodename : str):
        super().__init__(nodename)

        # ~~~~~~~~~~~~ Robot variables ~~~~~~~~~~~~ #
        self._jointState : JointState = JointState()
        self._lastJointState : JointState = JointState()
        self._arm_stopped = False
        self._gripper_stopped = False
        self.jointNames = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.ready1 = [np.pi/2, 0.0, np.pi/2, 0.0, np.pi/2, 0.0]
        self.cameraPose = [np.deg2rad(77), np.deg2rad(-90), np.deg2rad(60), np.deg2rad(65), np.deg2rad(100), np.deg2rad(0)]
        self.cameraImage = Image()
        
        # ~~~~~~~~~~~ MoveIt parameters ~~~~~~~~~~~ #
        self.robot_interface = MoveIt2(
            Node('move_group_interface'),
            joint_names=self.jointNames,
            base_link_name='base',
            end_effector_name='flange',
            group_name='tmr_arm',
            callback_group=ReentrantCallbackGroup()
        )
        
        self.robot_interface.planner_id = 'geometric::STRIDE'
        self.robot_interface.max_velocity = 0.5
        self.robot_interface.max_acceleration = 0.2
        self.robot_interface.cartesian_avoid_collisions = False
        self.robot_interface.cartesian_jump_threshold = 0.0
        
        # ~~~~~~~~ Demonstration variables ~~~~~~~~ #
        self.fault = False
        self.solving = False
        
        self.demonstration_info = {
            "header" : Header(),
            "arm_state" : JointState(),
            "arm_action" : JointState(),
            "gripper_state" : JointState(),
            "gripper_action" : JointState(),
            "camera" : Image()
        }

        # ~~~~~~~~~~~~ Isaac Sim paths ~~~~~~~~~~~~ #
        self.target_tube = '/World/Tubes/Tube_Target'
        self.start_rack = '/World/Racks/Rack_Start'
        self.target_rack = '/World/Racks/Rack_Goal'
        

        # ~~~~~~~~~~~~~ Miscellaneous ~~~~~~~~~~~~~ #
        self.mutually_exclusive_group = MutuallyExclusiveCallbackGroup()
        self.reentrant_group = ReentrantCallbackGroup()
        
        self.rate = self.create_rate(100)

        
        # ~~~~~~~~~~~~~~ Main topics ~~~~~~~~~~~~~~ #
        self.jointStateSub = self.create_subscription(JointState, '/joint_states', self.refreshRobotValues, 10, callback_group=self.reentrant_group)
        self.jointCommandSub = self.create_subscription(JointState, '/isaac_joint_commands', self.jointCommandCallback, 10, callback_group=self.reentrant_group)
        self.cameraSub = self.create_subscription(Image, '/rgb', self.cameraCallback, 10, callback_group=self.reentrant_group)

        # ~~~~~~~~~~~ Isaac Sim services ~~~~~~~~~~ #
        self.poseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestPose', callback_group=self.mutually_exclusive_group)
        self.tubeGraspPoseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestTubeGraspPose', callback_group=self.mutually_exclusive_group)
        self.tubeParameterRequest = self.create_client(TubeParameter, '/IsaacSim/RequestTubeParameter', callback_group=self.mutually_exclusive_group)
        self.closeGripper = self.create_client(SetBool, '/IsaacSim/CloseGripper', callback_group=self.reentrant_group)
        self.goalPoseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestTubeGoalPose', callback_group=self.reentrant_group)
        self.collisionRequest = self.create_client(CollisionRequest, '/IsaacSim/RequestCollisionCheck', callback_group=self.reentrant_group)
        
        # ~~~~~~~~~~~~~~ RG6 services ~~~~~~~~~~~~~ #
        self.gripperPoseRequest = self.create_client(GripperPose, '/onrobot/pose', callback_group=self.mutually_exclusive_group)
        self.gripperController = rclpy.action.ActionClient(self, GripperCommand, '/rg6_controller', callback_group=self.reentrant_group)
        
        
        # ~~~~~~~ Wait until services start ~~~~~~~ #
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

        # ~~~~~~ Self services and actions to solve scene ~~~~~ #
        self.solve_scene = self.create_service(Trigger, '/AnalyticSolver/SolveScene', self.solveSceneCallback, callback_group=self.reentrant_group)
        self.self_solve_scene = self.create_client(Trigger, '/AnalyticSolver/SolveScene', callback_group=self.reentrant_group)
        
        self.create_demonstration = rclpy.action.ActionServer(
            self, Demonstration, '/AnalyticSolver/GetDemonstration',
            callback_group=self.reentrant_group,
            goal_callback=self.get_demonstration_goal_callback,
            cancel_callback=self.get_demonstration_cancel_callback,
            execute_callback=self.get_demonstration_execute_callback)


        # ~~~~~~ Set robot to start position ~~~~~~ #
        time.sleep(2)
        self.moveRobotArmToConfiguration(self.ready1)
        
        self.get_logger().info('AnalyticSolver ready.')

# ======================================================== #
# ==================== State callbacks =================== #
# ======================================================== #
    def refreshRobotValues(self, msg : JointState):
        if any(joint_name == 'joint_1' for joint_name in msg.name):
            self._lastJointState = self._jointState
            self._jointState = msg

        # ~~~~~~~~~~~~~ Driver faulted ~~~~~~~~~~~~ #
        if len([i for i in self._jointState.name if 'joint_' in i]) > 0 and all(pos == 0.0 for pos, joint in 
        zip(self._jointState.position, self._jointState.name) if 'joint_' in joint) and self.solving:
            self.fault = True
            
        self._arm_stopped = True if len([i for i in self._jointState.name if 'joint_' in i]) > 0 and all(abs(pos - last_pos) < 5e-2 and 
                            abs(vel) < 5e-2 for pos, last_pos, vel, joint in 
                            zip(self._jointState.position, self._lastJointState.position, self._jointState.velocity, self._jointState.name) 
                            if 'joint_' in joint) else False
        
        if len([i for i in self._jointState.name if 'joint_' in i]) > 0:
            self.demonstration_info["arm_state"] = msg
                            
                            
        self._gripper_stopped = True if len([i for i in self._jointState.name if 'finger_joint' in i]) > 0 and all(abs(pos - last_pos) < 5e-2 and 
                            abs(vel) < 5e-2 for pos, last_pos, vel, joint in 
                            zip(self._jointState.position, self._lastJointState.position, self._jointState.velocity, self._jointState.name) 
                            if 'finger_joint' in joint) else False
        
        if len([i for i in self._jointState.name if 'finger_joint' in i]) > 0:
            for idx, name in enumerate(self._jointState.name):
                if 'finger_joint' in name:
                    state = JointState()
                    state.header = self._jointState.header
                    state.name = [name]
                    state.position = [self._jointState.position[idx]]
                    state.velocity = [self._jointState.velocity[idx]]
                    state.effort = [self._jointState.effort[idx]]
                    self.demonstration_info["gripper_state"] = state
                            
    def jointCommandCallback(self, msg : JointState):
        if len([i for i in self._jointState.name if 'joint_' in i]) > 0:
            self.demonstration_info["arm_action"] = msg
            
    def cameraCallback(self, msg : Image):
        self.cameraImage = msg
        
# ======================================================== #
# ================ Demonstration callback ================ #
# ======================================================== #
    def solveSceneCallback(self, request : Trigger.Request, response : Trigger.Response):
        
        self.solving = False
        self.fault = False
        
        self.demonstration_info["camera"] = self.cameraImage
        # self.moveRobotGripper(np.deg2rad(-20.0), 200.0)
        self.moveRobotGripper(0.0, 200.0)
        self.solving = True
        
        try:
            goal : PoseStamped = PoseStamped()
            goal.header.frame_id = "base"
            
            # ~~~~~~~~~ Go home configuration ~~~~~~~~~ #
            self.moveRobotArmToConfiguration(self.ready1)
            
            # ~~~~~~~~~~ Create before image ~~~~~~~~~~ #
            self.moveRobotArmToConfiguration(self.cameraPose)
            sleep(2)
            
            self.demonstration_info["camera"] = self.cameraImage
            
            self.moveRobotArmToConfiguration(self.ready1)
            
            # ~~~~~~~~~~~~~ Get grasp pose ~~~~~~~~~~~~ #
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = self.target_tube
            self.grasp_pose : Transform = self.callService(service=self.tubeGraspPoseRequest, request=request).pose
            
            # ~~~~~~~~~~ Get tube parameters ~~~~~~~~~~ #
            request : TubeParameter.Request = TubeParameter.Request()
            request.path = self.target_tube
            self.tubeDimensions = self.callService(service=self.tubeParameterRequest, request=request)
            self.tubeDimensions = [self.tubeDimensions.dimensions.x, self.tubeDimensions.dimensions.y, self.tubeDimensions.dimensions.z]
            self.tubeWidth = min(self.tubeDimensions)
            self.tubeHeight = max(self.tubeDimensions)
            self.get_logger().debug('Height: ' + str(self.tubeHeight) + ' Width: ' + str(self.tubeWidth))
            
            # ~~~~~~~ Get start rack parameters ~~~~~~~ #
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = self.start_rack
            self.rackPose : Transform = self.callService(service=self.poseRequest, request=request).pose
                
            #  Set gripper to open and calculate grasp parameters  #
            request : GripperPose.Request = GripperPose.Request()
            request.known.x = self.tubeWidth + 0.008
            self.gripper_goal : Pose2D = self.callService(service=self.gripperPoseRequest, request=request).pose
            self.get_logger().debug('Gripper width: ' + str(self.gripper_goal.x) + ' Gripper height: ' + str(self.gripper_goal.y))
            self.moveRobotGripper(np.deg2rad(-35), 200.0)

            # ~~~~~~~~~~~~ Move above tube ~~~~~~~~~~~~ #
            request : TubeParameter.Request = TubeParameter.Request()
            request.path = self.target_tube
            self.tubeDimensions : TubeParameter.Response = self.callService(service=self.tubeParameterRequest, request=request)
            self.get_logger().debug(str(self.tubeDimensions))
            tube_orientation : Rotation = Rotation.from_quat([self.tubeDimensions.pose.orientation.x, self.tubeDimensions.pose.orientation.y, self.tubeDimensions.pose.orientation.z, self.tubeDimensions.pose.orientation.w])
            tube_orientation = Rotation.from_euler('xyz', [np.pi/2, 0.0, 0.0]) * tube_orientation
            
            rack_orientation : Rotation = Rotation.from_quat([self.rackPose.rotation.x, self.rackPose.rotation.y, self.rackPose.rotation.z, self.rackPose.rotation.w])
            
            # ~~~~~~~~~~~~~~ Get TCP pose ~~~~~~~~~~~~~ #
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = '/World/tm5_900/flange_link'
            ee_pose : Transform = self.callService(service=self.poseRequest, request=request).pose
            ee_rotation : Rotation = Rotation.from_quat([ee_pose.rotation.x, ee_pose.rotation.y, ee_pose.rotation.z, ee_pose.rotation.w])
            
            
            minimal_rotation = rack_orientation.as_euler('xyz')[2] 
            if abs(rack_orientation.as_euler('xyz')[2] - ee_rotation.as_euler('xyz')[2]) > abs(np.pi - abs(rack_orientation.as_euler('xyz')[2] - ee_rotation.as_euler('xyz')[2])):
                minimal_rotation = minimal_rotation - np.pi
            
            orientation : Rotation = \
                Rotation.from_euler('xyz', [(np.pi-tube_orientation.as_euler('xyz')[0]), -tube_orientation.as_euler('xyz')[1], minimal_rotation])
            
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y+0.2])
            
            goal.pose.orientation.x = orientation.as_quat()[0]
            goal.pose.orientation.y = orientation.as_quat()[1]
            goal.pose.orientation.z = orientation.as_quat()[2]
            goal.pose.orientation.w = orientation.as_quat()[3]
            
            goal.pose.position.x = self.grasp_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = self.grasp_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = self.grasp_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
            
            self.moveRobotArm(goal, cartesian=False, velocity=0.2, acceleration=0.2)
            
            self.demonstration_info["camera"] = self.cameraImage

            # ~~~~~~~~~~~~~ Approach tube ~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y+0.01]) 
            goal.pose.position.x = self.grasp_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = self.grasp_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = self.grasp_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
        
            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.2, acceleration=0.2)

            # ~ Close gripper and use surface gripper ~ #
            self.moveRobotGripper(self.gripper_goal.theta, 200.0)
            
            self.get_logger().debug('Activating Surface Gripper')
            request : SetBool.Request = SetBool.Request()
            request.data = True
            self.callService(service=self.closeGripper, request=request)
            self.get_logger().debug('Surface Gripper activated')

            # ~~~~~~~~~~~~~ Pull out tube ~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y + 0.2]) 
            goal.pose.position.x = self.grasp_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = self.grasp_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = self.grasp_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
            
            self.moveRobotArm(goal, velocity=0.05, acceleration=0.05)
            
            self.demonstration_info["camera"] = self.cameraImage

            # ~~~~~~~~~~~ Approach goal rack ~~~~~~~~~~ #
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = self.target_rack
            rack_pose : Transform = self.callService(service=self.goalPoseRequest, request=request).pose
            self.get_logger().debug(str(rack_pose))
            rack_orientation : Rotation = Rotation.from_quat([rack_pose.rotation.x, rack_pose.rotation.y, rack_pose.rotation.z, rack_pose.rotation.w])
            
            request : TubeParameter.Request = TubeParameter.Request()
            request.path = self.target_tube
            self.tubePose : TubeParameter.Response = self.callService(service=self.tubeParameterRequest, request=request)
            self.get_logger().debug(str(self.tubePose))
            tube_translation : Pose = Pose()
            tube_translation.position.x = self.tubePose.pose.position.x
            tube_translation.position.y = self.tubePose.pose.position.y
            tube_translation.position.z = self.tubePose.pose.position.z
            tube_orientation : Rotation = Rotation.from_quat([self.tubePose.pose.orientation.x, self.tubePose.pose.orientation.y, self.tubePose.pose.orientation.z, self.tubePose.pose.orientation.w])
            tube_orientation = Rotation.from_euler('xyz', [np.pi/2, 0.0, 0.0]) * tube_orientation
            
            # ~~~~~~~~~~~~~~ Get TCP pose ~~~~~~~~~~~~~ #
            request : PoseRequest.Request = PoseRequest.Request()
            request.path = '/World/tm5_900/flange_link'
            ee_pose : Transform = self.callService(service=self.poseRequest, request=request).pose
            ee_rotation : Rotation = Rotation.from_quat([ee_pose.rotation.x, ee_pose.rotation.y, ee_pose.rotation.z, ee_pose.rotation.w])
            
            minimal_rotation = rack_orientation.as_euler('xyz')[2] 
            if abs(rack_orientation.as_euler('xyz')[2] - ee_rotation.as_euler('xyz')[2]) > abs(np.pi - abs(rack_orientation.as_euler('xyz')[2] - ee_rotation.as_euler('xyz')[2])):
                minimal_rotation = minimal_rotation - np.pi
            
            orientation : Rotation = \
                Rotation.from_euler('xyz', [(np.pi-tube_orientation.as_euler('xyz')[0]), -tube_orientation.as_euler('xyz')[1], minimal_rotation])
            
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y])
            
            goal.pose.orientation.x = orientation.as_quat()[0]
            goal.pose.orientation.y = orientation.as_quat()[1]
            goal.pose.orientation.z = orientation.as_quat()[2]
            goal.pose.orientation.w = orientation.as_quat()[3]
            
            goal.pose.position.x = rack_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = rack_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = rack_pose.translation.z - gripper_translation[2] + 0.2
            goal.header.stamp = self.get_clock().now().to_msg()
            
            self.get_logger().debug('Rotation: ' + str(orientation.as_euler('xyz', degrees=True)))
            self.get_logger().debug('Target position: [' + str(rack_pose.translation.x - gripper_translation[0]) + ', ' +\
                str(rack_pose.translation.y - gripper_translation[1]) + ', ' +\
                str(rack_pose.translation.z - gripper_translation[2]) + ']')
            self.moveRobotArm(goal, cartesian=False, velocity=0.2, acceleration=0.2)
            
            self.demonstration_info["camera"] = self.cameraImage
            
            # ~~~~~~~~~~~~~~ Put in tube ~~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y]) 
            goal.pose.position.x = rack_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = rack_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = rack_pose.translation.z - gripper_translation[2]
            goal.header.stamp = self.get_clock().now().to_msg()
        
            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.05, acceleration=0.05)

            # ~ Open gripper and use surface gripper ~ #
            self.get_logger().debug('Activating Surface Gripper')
            request : SetBool.Request = SetBool.Request()
            request.data = False
            self.callService(service=self.closeGripper, request=request)
            self.get_logger().debug('Surface Gripper activated')
            
            self.moveRobotGripper(0.0, 200.0)

            # ~~~~~~~~~~~~~~ Get away from tube ~~~~~~~~~~~~~~ #
            gripper_translation = orientation.apply([0.0, 0.0, self.gripper_goal.y]) 
            goal.pose.position.x = rack_pose.translation.x - gripper_translation[0]
            goal.pose.position.y = rack_pose.translation.y - gripper_translation[1]
            goal.pose.position.z = rack_pose.translation.z - gripper_translation[2] + 0.2
            goal.header.stamp = self.get_clock().now().to_msg()
        
            self.moveRobotArm(pose=goal, cartesian=True, velocity=0.2, acceleration=0.2)
            
            self.demonstration_info["camera"] = self.cameraImage

            # ~~~~~~~~~~~ Create after image ~~~~~~~~~~ #
            self.moveRobotArmToConfiguration(self.cameraPose)
            sleep(1)
            
            self.demonstration_info["camera"] = self.cameraImage
            
            # ~~~~~~~~~ Go home configuration ~~~~~~~~~ #
            self.moveRobotArmToConfiguration(self.ready1)
            
            self.rate.sleep()
            
            request : CollisionRequest.Request = CollisionRequest.Request()
            request.prim1 = "/World/Racks/Rack_Goal"
            request.prim2 = "/World/Tubes/Tube_Target"
            inRack : bool = self.callService(self.collisionRequest, request).collision
            
            request : CollisionRequest.Request = CollisionRequest.Request()
            request.prim1 = "/World/Tubes/Tube_Target"
            request.prim2 = "/World/Tubes"
            nonColliding : bool = not self.callService(self.collisionRequest, request).collision
            
            request : CollisionRequest.Request = CollisionRequest.Request()
            request.prim1 = "/World/Tubes/Tube_Target"
            request.prim2 = "/World/Room/table_low_327"
            notOnFloor : bool = not self.callService(self.collisionRequest, request).collision
            
            request : TubeParameter.Request = TubeParameter.Request()
            request.path = self.target_tube
            pose_res : TubeParameter.Response = self.callService(service=self.tubeParameterRequest, request=request)
            tube_orientation : Rotation = Rotation.from_quat([pose_res.pose.orientation.x, pose_res.pose.orientation.y, pose_res.pose.orientation.z, pose_res.pose.orientation.w])
            tube_orientation = Rotation.from_euler('xyz', [np.pi/2, 0.0, 0.0]) * tube_orientation
            tube_orientation = tube_orientation.as_euler('xyz')
            self.get_logger().debug('Tube orientation: ' + str(tube_orientation))
            notBinted : bool = not ((abs(tube_orientation[0]) > np.pi/4) or (abs(tube_orientation[1]) > np.pi/4))
            
            response.success = inRack and notOnFloor and notBinted
            response.message = ''
        
        except Exception as e:
            request : SetBool.Request = SetBool.Request()
            request.data = False
            self.callService(service=self.closeGripper, request=request)
            self.get_logger().error(f"Service call failed: {e}")
            response.success = False
            response.message = str(e)

        self.get_logger().info('Demonstration ended.')
        self.solving = False
        return response

# ======================================================== #
# =============== Robot movement functions =============== #
# ======================================================== #
    def moveRobotArm(self,  pose : Pose | PoseStamped | None = None, 
                            position : tuple[float, float, float] | None = None, orientation : tuple[float, float, float, float] | None = None,
                            velocity : float = 0.0, acceleration : float = 0.0,
                            cartesian : bool = False):
        if not velocity == 0.0:
            self.robot_interface.max_velocity = velocity
        if not acceleration == 0.0:
            self.robot_interface.max_acceleration = acceleration
        try:
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
            while not self._arm_stopped:
                self.rate.sleep()
            self._arm_stopped = False
        except Exception as e:
            self.fault = True
            self.get_logger().info('Error: ' + str(e))
        if self.fault: 
                raise Exception()
        self.get_logger().info('Movement completed!')
        
    def moveRobotArmToConfiguration(self, joint_angles : list[float]):
        try:
            self.robot_interface.move_to_configuration(joint_angles, self.jointNames)
            self.robot_interface.wait_until_executed()
        except Exception as e:
            self.fault = True
            self.get_logger().info('Error: ' + str(e))
        if self.fault: 
                raise Exception()
        self.get_logger().info('Movement completed!')
        
    def moveRobotGripper(self, joint_angle : float, effort : float):
        
        goal : GripperCommand.Goal = GripperCommand.Goal()
        goal.command.position = joint_angle
        goal.command.max_effort = effort
        action = JointState()
        action.header.stamp = self.get_clock().now().to_msg()
        action.header.frame_id = 'base'
        action.name = ['finger_joint']
        action.position = [goal.command.position]
        action.effort = [goal.command.max_effort]
        self.demonstration_info["gripper_action"] = action
        self._gripper_stopped = False
        self.gripperController.send_goal(goal)
        start = self.get_clock().now().nanoseconds
        while not (self._gripper_stopped or (self.get_clock().now().nanoseconds - start)/1e9 > 10):
            self.rate.sleep()
        self.get_logger().info('Done gripper movement')
        if self.fault: 
                raise Exception()
                      
# ======================================================== #
# ====== /AnalyticSolver/GetDemonstration functions ====== #
# ======================================================== #
    def get_demonstration_goal_callback(self, goal_request):
        self.get_logger().info('Received demonstration goal request')
        return GoalResponse.ACCEPT
    
    def get_demonstration_cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
    
    def get_demonstration_execute_callback(self, goal_handle):
        self.get_logger().info('Getting demonstration')
        feedback = Demonstration.Feedback()
        result = Demonstration.Result()

        self.demonstration_info["camera"] = Image()
        msg = Trigger.Request()
        future = self.self_solve_scene.call_async(msg)
        while (
            not self.solving and
                (
                    len(self.demonstration_info["arm_state"].name) <= 0 or
                    len(self.demonstration_info["arm_action"].name) <= 0 or
                    len(self.demonstration_info["gripper_state"].name) <= 0 or
                    len(self.demonstration_info["gripper_action"].name) <= 0 or
                    len(self.demonstration_info["camera"].data) <= 0
                )
            ):
            self.rate.sleep()
        start_time = time.time_ns()
        while not future.done():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.end_demonstration = True
                self.get_logger().error('Goal Canceled')
                return
            
            feedback.arm_action = self.demonstration_info["arm_action"]
            feedback.arm_state = self.demonstration_info["arm_state"]
            
            feedback.gripper_action = self.demonstration_info["gripper_action"]
            feedback.gripper_state = self.demonstration_info["gripper_state"]
            
            feedback.camera = self.demonstration_info["camera"]
            
            feedback.header.frame_id = "base"
            feedback.header.stamp = self.get_clock().now().to_msg()
            
            goal_handle.publish_feedback(feedback)

            time_to_wait = 0.05 - (float(time.time_ns() - start_time))/1e9
            if time_to_wait > 1e-3:
                sleep(0.05 - (float(time.time_ns() - start_time))/1e9)
            start_time = time.time_ns()
        
        result.arm_action = self.demonstration_info["arm_action"]
        result.arm_state = self.demonstration_info["arm_state"]
        
        result.gripper_action = self.demonstration_info["gripper_action"]
        result.gripper_state = self.demonstration_info["gripper_state"]
        
        result.camera = self.demonstration_info["camera"]
        
        result.header.frame_id = "base"
        result.header.stamp = self.get_clock().now().to_msg()
        result.success = future.result().success
        result.message = ""

        goal_handle.succeed()
        self.get_logger().info('Successfully executed goal')
        return result
                
# ======================================================== #
# ===================== Call services ==================== #
# ======================================================== #
    def callService(self, service : Client, request, message : str | None = None):
        if isinstance(message, str):
            self.get_logger().info(message)
        self.get_logger().info('Calling ' + str(service.srv_name) + ' service.')
        future  = service.call_async(request)
        self.rate.sleep()
        while not future.done():
            self.rate.sleep()
        while isinstance(future.result(), type(None)):
            self.rate.sleep()
        self.rate.sleep()
        response = future.result()
        self.get_logger().info('Called' + str(service.srv_name) + ' successfully.')
        return response

def main():
    try:
        rclpy.init(args=None)

        executor = MultiThreadedExecutor()
        solver = AnalyticSolver('AnalyticSolver')
        executor.add_node(solver)
        executor.add_node(solver.robot_interface._node)
        executor.spin()

        solver.destroy_node()

        rclpy.shutdown()

    except(KeyboardInterrupt, ExternalShutdownException):
        pass

if __name__ == '__main__':
    main()
