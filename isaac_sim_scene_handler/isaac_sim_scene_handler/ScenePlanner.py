#! ${USER}/isaacsim/setup_python_env.sh

# ======================================================== #
# ==================== Python Imports ==================== #
# ======================================================== #
import sys
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
import random

# ======================================================== #
# ===================== ROS 2 Imports ==================== #
# ======================================================== #
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory


# ======================================================== #
# ==================== Message Imports =================== #
# ======================================================== #
from geometry_msgs.msg import Transform
from sensor_msgs.msg import JointState, Image

from std_srvs.srv import Trigger
from isaac_sim_msgs.srv import PoseRequest, TubeParameter, CollisionRequest, SetJoint

# ======================================================== #
# =================== Isaac Sim imports ================== #
# ======================================================== #
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RaytracedLighting", "headless": False, "anti_aliasing": 1}

# Simple example showing how to start and stop the helper
simulation_app = SimulationApp(CONFIG)

import carb
from carb import Float3, Float4

import omni
import omni.usd

from isaacsim.core.api import World
from isaacsim.core.utils import extensions, stage
from omni.isaac.dynamic_control import _dynamic_control
from isaacsim.storage.native import get_assets_root_path, is_file
from isaacsim.core.utils.stage import is_stage_loading, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.core.prims import XFormPrim

# enable ROS2 bridge extension
extensions.enable_extension("isaacsim.ros2.bridge")
extensions.enable_extension("isaacsim.util.clash_detection")

from isaacsim.util.clash_detection import ClashDetector

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

class IsaacSim(Node):

    # ~~~~ Define assets and areas, vectors ~~~ # 
    robot_path = '/World/tm5_900'
    
    rack_list : list = [
        '/World/Racks/Rack_Start',
        '/World/Racks/Rack_Goal',
    ]
    
    rack_usage : dict = {name : 6 * [False] for name in rack_list}
    
    rack_storage_position = np.array([0.0, -0.3, 0.01])
    rack_storage_position_relative = np.array([0.0, -0.1, 0.0])
    rack_storage_rotation = Rotation.from_euler('xyz',[np.pi/2, 0.0, np.pi/2])
    
    tube_list : list = [
        '/World/Tubes/Tube_Target',
        '/World/Tubes/Tube_Fix_01',
        '/World/Tubes/Tube_Fix_02',
        '/World/Tubes/Tube_Fix_03',
        '/World/Tubes/Tube_Fix_04',
        '/World/Tubes/Tube_Fix_05',
        '/World/Tubes/Tube_Fix_06',
        '/World/Tubes/Tube_Fix_07',
        '/World/Tubes/Tube_Fix_08',
        '/World/Tubes/Tube_Fix_09',
        '/World/Tubes/Tube_Fix_10',
    ]
    
    tube_usage : dict = {name : False for name in tube_list}
    
    tube_storage_position = np.array([0.02, -0.4, 0.01])
    tube_storage_position_relative = np.array([0.04, 0.0, 0.0])
    tube_storage_rotation = Rotation.from_euler('xyz',[0, 0, 0])
    
    workspace = np.array([[-0.35, 0.4, 0.01], [0.35, 0.55, 0.01]])
    rack_to_hole_translation = np.array([-0.07, 0.06, 0.01])
    hole_to_hole_translation = np.array([0.0, 0.03, 0.0])

    def __init__(self, nodename : str) -> None:
        
        super().__init__(nodename)
        
        self.reentran_group = ReentrantCallbackGroup()
        self.mutual_group = MutuallyExclusiveCallbackGroup()
        
        self.joint_command = self.create_subscription(JointState, '/isaac_joint_commands', self.NoneCallback, 10, callback_group=self.reentran_group)
        self.joint_states = self.create_publisher(JointState, '/joint_states', 10, callback_group=self.reentran_group)
        self.rgb = self.create_publisher(Image, '/rgb', 10, callback_group=self.reentran_group)
        
        self.launchIsaacSim()
        
        self.stage = get_current_stage()
        
        # Initialize clash detection engine
        self.robot = Robot(prim_path='/World/tm5_900', name="tm5_900")
        self.robot.initialize()
        simulation_app.update()
        
        # ~~~~~~~~~ Create clash detector ~~~~~~~~~ #
        self.clash_detector = ClashDetector(self.stage, logging=False, clash_data_layer=False)
        self.clash_detector.set_scope('')
        self.cache = bounds_utils.create_bbox_cache()
        
        # ~~~~~~~ Remaining tube parameters ~~~~~~~ #
        self.tube_height = 0.01*max(self.getAbsoluteBoundingBox('/World/Tubes/Tube_Target')[2])
        self.tube_diameter = 0.01*min(self.getAbsoluteBoundingBox('/World/Tubes/Tube_Target')[2])
        
        # ~~~~~~~~~ Create ROS 2 services ~~~~~~~~~ #
        self.resetScene = self.create_service(Trigger, '/IsaacSim/ResetScene', self.resetSceneCallback, callback_group=self.mutual_group)
        self.newScene = self.create_service(Trigger, '/IsaacSim/NewScene', self.newSceneCallback, callback_group=self.mutual_group)
        self.poseRequestServer = self.create_service(PoseRequest, '/IsaacSim/RequestPose', self.poseRequestCallback, callback_group=self.reentran_group)
        self.tubeGraspPoseRequestServer = self.create_service(PoseRequest, '/IsaacSim/RequestTubeGraspPose', self.tubeGraspPoseRequestCallback, callback_group=self.reentran_group)
        self.tubeParameterRequestServer = self.create_service(TubeParameter, '/IsaacSim/RequestTubeParameter', self.tubeParameterRequestCallback, callback_group=self.reentran_group)
        self.tubeGoalPoseRequestServer = self.create_service(PoseRequest, '/IsaacSim/RequestTubeGoalPose', self.tubeGoalPoseRequestCallback, callback_group=self.reentran_group)
        self.collisionCheckRequestServer = self.create_service(CollisionRequest, '/IsaacSim/RequestCollisionCheck', self.collisionCheckRequestCallback, callback_group=self.reentran_group)
        self.setJointRequestServer = self.create_service(SetJoint, '/IsaacSim/SetJoint', self.setJointCallback, callback_group=self.reentran_group)
    
    
# ======================================================== #
# ============== Isaac Sim running functions ============= #
# ======================================================== #
    def launchIsaacSim(self):

        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0)
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        
        
        # ~~~~~~~~~~ Find and load stage ~~~~~~~~~~ #
        package_name = 'tm5-900_rg6_moveit_config'
        USD_NAME = "tm5-900_real.usd"
        usd_path = get_package_share_directory(package_name) + '/config/tm5-900/' + USD_NAME
        try:
            result = is_file(usd_path)
        except:
            result = False
        
        if result:
            omni.usd.get_context().open_stage(usd_path)
        else:
            carb.log_error(
                f"the usd path {usd_path} could not be opened, please make sure that {usd_path} is a valid usd file in {assets_root_path}"
            )
            simulation_app.close()
            sys.exit()
        # Wait two frames so that stage starts loading
        simulation_app.update()
        simulation_app.update()
        
        print("Loading stage...")
        
        while is_stage_loading():
            simulation_app.update()
        print("Loading Complete")
        
        simulation_app.update()
        while stage.is_stage_loading():
            simulation_app.update()
        
        self.world.play()
        self.stage = omni.usd.get_context().get_stage()
        
    def runApp(self):
        while simulation_app.is_running():
            # Run with a fixed step size
            reset_needed = False
            simulation_app.update()
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.world.is_playing():
                if reset_needed:
                    self.world.reset()
                    reset_needed = False
        
        self.timeline.stop()
        self.world.stop()
        simulation_app.close()  # Cleanup application


# ======================================================== #
# =================== Object parameters ================== #
# ======================================================== #
    def setObjectKinematic(self, prim_path, value: bool):
        kinematic = self.stage.GetPrimAtPath(prim_path).GetAttribute('physics:kinematicEnabled')
        kinematic.Set(value)

    def poseRequest(self, prim_path):
        prim = self.dc.get_rigid_body(prim_path)
        prim_pose = self.dc.get_rigid_body_pose(prim)
        response = PoseRequest.Response()
        
        response.pose.translation.x = prim_pose.p[0]
        response.pose.translation.y = prim_pose.p[1]
        response.pose.translation.z = prim_pose.p[2]
        
        response.pose.rotation.x = prim_pose.r[0]
        response.pose.rotation.y = prim_pose.r[1]
        response.pose.rotation.z = prim_pose.r[2]
        response.pose.rotation.w = prim_pose.r[3]
        
        return response

    def getWorldBoundingBox(self, path : str):    
        self.cache = bounds_utils.create_bbox_cache()
        return bounds_utils.compute_aabb(self.cache, path)
    
    def getAbsoluteBoundingBox(self, path : str):
        self.cache = bounds_utils.create_bbox_cache()
        centroid, axes, half_extent = bounds_utils.compute_obb(self.cache, prim_path=path)
        return (centroid, axes, half_extent)
    

# ======================================================== #
# ====================== Collisions ====================== #
# ======================================================== #
    def collisionWithRacks(self, prim_path) -> bool:
        self.clash_detector.set_scope('/World/Racks')
        return self.clash_detector.is_prim_clashing(get_prim_at_path(prim_path))
        
    def collisionWithTubes(self, prim_path) -> bool:
        self.clash_detector.set_scope('/World/Tubes')
        return self.clash_detector.is_prim_clashing(get_prim_at_path(prim_path))


# ======================================================== #
# ================== Object manipulation ================= #
# ======================================================== #
    def placeObject(self, prim_path, transform : _dynamic_control.Transform):
        self.setObjectKinematic(prim_path, True)
        prim = self.dc.get_rigid_body(prim_path)
        self.dc.set_rigid_body_pose(prim, transform)
        simulation_app.update()
        self.setObjectKinematic(prim_path, False)
        self.dc.set_rigid_body_angular_velocity(prim, Float3([0.0, 0.0, 0.0]))
        self.dc.set_rigid_body_linear_velocity(prim, Float3([0.0, 0.0, 0.0]))

    def resetRacks(self):
        for idx, rack_path in enumerate(self.rack_list):
                transform = _dynamic_control.Transform( 
                    (self.rack_storage_position + idx*self.rack_storage_position_relative).tolist(), 
                    self.rack_storage_rotation.as_quat().tolist()
                )
    
                self.placeObject(rack_path, transform)
                
        self.rack_usage = {name : 6 * [False] for name in self.rack_list}
       
    def resetTubes(self):
        for idx, tube_path in enumerate(self.tube_list):
            transform = _dynamic_control.Transform( 
                Float3(self.tube_storage_position + idx*self.tube_storage_position_relative), 
                Float4(self.tube_storage_rotation.as_quat())
            )

            self.placeObject(tube_path, transform)
        
        self.tube_usage = {name : False for name in self.tube_list}
                            
                            
# ======================================================== #
# ==================== Randomize Scene =================== #
# ======================================================== #
    def placeInWorkspace(self, rack_path, angle):
        transform = _dynamic_control.Transform( 
            Float3(
                random.uniform(self.workspace[0][0], self.workspace[1][0]),
                random.uniform(self.workspace[0][1], self.workspace[1][1]),
                random.uniform(self.workspace[0][2], self.workspace[1][2])
            ), 
            Float4(Rotation.from_euler('xyz',[np.pi/2, 0.0, angle]).as_quat())
        )
        
        self.placeObject(rack_path, transform)
        
    def placeTubeInRack(self, rack_path, tube_path, slot_num):
        
        rack = self.dc.get_rigid_body(rack_path)

        rotation = Rotation.from_quat(self.dc.get_rigid_body_pose(rack).r)
        translation = self.dc.get_rigid_body_pose(rack).p
        translation = np.array([translation[0], translation[1], translation[2]]) + \
        Rotation.from_euler('xyz', [0.0, 0.0, rotation.as_euler('xyz')[2]]).apply(
                    self.rack_to_hole_translation + slot_num * self.hole_to_hole_translation
            )
        
        transform = _dynamic_control.Transform(
            Float3(translation.tolist()),
            Float4(Rotation.from_euler('xyz', [np.pi/2, 0.0, rotation.as_euler('xyz')[2]]).as_quat()) # Assets orientation is layed. Pi/2 rotation in X makes it upright.
        )

        self.rack_usage[rack_path][slot_num] = True
        self.tube_usage[tube_path] = True

        self.placeObject(tube_path, transform)
        # Wait a few frames to make simulation stable
        for _ in range(5):
            simulation_app.update() 
                
    def randomizeRackContent(self, rack_path, tube_list : list, min_placed_tubes):
        
        # Empty slots
        choices = [i for i in range(6)]
        
        for i in range(random.randint(min_placed_tubes, len(tube_list))):
            slot = random.choice(choices)
            self.placeTubeInRack(rack_path, tube_list[i], slot)
            choices.remove(slot)
    
    def realisticPose(self, prim_path) -> bool:
        robot_position, _ = XFormPrim(self.robot_path).get_world_pose()
        pose = self.poseRequest(prim_path).pose
        # Tube is within 1.0m range of robot
        if(
            abs(pose.translation.x - robot_position[0]) < 1.0 and
            abs(pose.translation.x - robot_position[1]) < 1.0 and
            abs(pose.translation.x - robot_position[2]) < 1.0
        ):
            return True
            
        return False

    def getFirstEmptySlotTransformation(self, path : str):
        # First empty slot
        slot_num = -1
        for slot in self.rack_usage[path]:
            slot_num += 1
            if not slot:
                break
        
        rack = self.dc.get_rigid_body(path)

        translation = self.dc.get_rigid_body_pose(rack).p
        rotation = Rotation.from_quat(self.dc.get_rigid_body_pose(rack).r)
        
        translation = np.array([translation[0], translation[1], translation[2]]) + \
        Rotation.from_euler('xyz', [0.0, 0.0, rotation.as_euler('xyz')[2]]).apply(
                    self.rack_to_hole_translation + (slot_num-1) * self.hole_to_hole_translation
            )
            
        transformation : Transform = Transform()
        transformation.translation.x = translation[0]
        transformation.translation.y = translation[1]
        transformation.translation.z = translation[2]
        transformation.rotation.x = rotation.as_quat()[0]
        transformation.rotation.y = rotation.as_quat()[1]
        transformation.rotation.z = rotation.as_quat()[2]
        transformation.rotation.w = rotation.as_quat()[3]
        
        return transformation
    
  
# ======================================================== #
# ==================== ROS 2 Callbacks =================== #
# ======================================================== #     
    
    # * /IsaacSim/RequestTubeGoalPose
    def tubeGoalPoseRequestCallback(self, request : PoseRequest.Request, response : PoseRequest.Response):
        self.get_logger().info('Get Tube goal pose request')
        goal_rack_pose : Transform = self.getFirstEmptySlotTransformation('/World/Racks/Rack_Goal')
        goal_rack_rotation = Rotation.from_quat([goal_rack_pose.rotation.x, goal_rack_pose.rotation.y, goal_rack_pose.rotation.z, goal_rack_pose.rotation.w])
        goal_rack_rotation = Rotation.from_euler('xyz', [0.0, 0.0, goal_rack_rotation.as_euler('xyz')[2]])
        
        axes : ndarray
        _, axes, half_extent = self.getAbsoluteBoundingBox(request.path)
        
        scale_factor = np.absolute(np.linalg.eigvals(axes))
        end_vec = goal_rack_rotation.apply([scale_factor[0]*min(half_extent)/2, scale_factor[1]*min(half_extent)/2, scale_factor[2]*max(half_extent)])
        
        response.pose.translation.x = goal_rack_pose.translation.x + end_vec[0] 
        response.pose.translation.y = goal_rack_pose.translation.y + end_vec[1]
        response.pose.translation.z = goal_rack_pose.translation.z + end_vec[2]
        
        response.pose.rotation.x = goal_rack_rotation.as_quat()[0]
        response.pose.rotation.y = goal_rack_rotation.as_quat()[1]
        response.pose.rotation.z = goal_rack_rotation.as_quat()[2]
        response.pose.rotation.w = goal_rack_rotation.as_quat()[3]
        
        return response
    
    # * /IsaacSim/RequestCollisionCheck
    def collisionCheckRequestCallback(self, request : CollisionRequest.Request, response : CollisionRequest.Response):
        self.clash_detector.set_scope(request.prim1)
        response.collision = self.clash_detector.is_prim_clashing(get_prim_at_path(request.prim2))
        return response
        
    # * /IsaacSim/SetJoint
    def setJointCallback(self, request : SetJoint.Request, response : SetJoint.Response):
        try:  
            action = ArticulationAction(joint_positions=np.array([float(request.value)]), joint_indices=np.array([self.robot.get_dof_index(dof_name=request.joint)]))
            self.robot.apply_action(action)

            response.success = True
            response.message = ''
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response
     
    # * /IsaacSim/RequestPose
    def poseRequestCallback(self, request, response):
        self.get_logger().info('Pose request')
        response = self.poseRequest(request.path)
        return response

    # * /IsaacSim/RequestTubeGraspPose
    def tubeGraspPoseRequestCallback(self, request : PoseRequest.Request, response : PoseRequest.Response):
        simulation_app.update()
        
        self.cache = bounds_utils.create_bbox_cache()
        
        axes : ndarray
        centroid, axes, half_extent = self.getAbsoluteBoundingBox(request.path)
        
        scale_factor = np.absolute(np.linalg.eigvals(axes))
        rotation = Rotation.from_matrix((axes.T/scale_factor).T).inv()
        quat = rotation.as_quat()
        end_vec = rotation.apply(np.multiply(scale_factor, [extent if extent == max(half_extent) else 0.0 for extent in half_extent]))
        
        self.get_logger().debug('Axes: ' + str(axes))
        self.get_logger().debug('Rotation: ' + str(rotation.as_matrix()))
        self.get_logger().debug('End vector: ' + str(end_vec))
        self.get_logger().debug('Centroid: ' + str(centroid))
        
        response : PoseRequest.Response = PoseRequest.Response()
        response.pose.translation.x = centroid[0] + end_vec[0] 
        response.pose.translation.y = centroid[1] + end_vec[1]
        response.pose.translation.z = centroid[2] + end_vec[2]
        
        response.pose.rotation.x = quat[0]
        response.pose.rotation.y = quat[1]
        response.pose.rotation.z = quat[2]
        response.pose.rotation.w = quat[3]
        
        return response
       
    # * /IsaacSim/RequestTubeParameter
    def tubeParameterRequestCallback(self, request : TubeParameter.Request, response : TubeParameter.Response):
        axes : ndarray
        centroid, axes, half_extent = self.getAbsoluteBoundingBox(request.path)
        scale_factor = np.absolute(np.linalg.eigvals(axes))
        quat = Rotation.from_matrix((axes.T / scale_factor).T).as_quat()
        
        self.get_logger().debug('Half extent: ' + str(half_extent))
        self.get_logger().debug('Scaling factor: ' + str(scale_factor))
        
        response.dimensions.x = 2*scale_factor[0]*half_extent[0]
        response.dimensions.y = 2*scale_factor[1]*half_extent[1]
        response.dimensions.z = 2*scale_factor[2]*half_extent[2]
        
        response.pose.position.x = centroid[0]
        response.pose.position.y = centroid[1]
        response.pose.position.z = centroid[2]
        response.pose.orientation.x = quat[0]
        response.pose.orientation.y = quat[1]
        response.pose.orientation.z = quat[2]
        response.pose.orientation.w = quat[3]
        return response
     
    # * /IsaacSim/NewScene 
    def newSceneCallback(self, request = Trigger.Request(), response = Trigger.Response()):
        self.get_logger().info('New scene request')
        
        self.clash_detector.set_scope('/World/Racks')
        while True:
            validStage = False
            while not validStage:
                self.resetSceneCallback()
                validStage = True
                for idx, rack_path in enumerate(self.rack_list):
                    self.placeInWorkspace(rack_path, random.uniform(0.0, 2*np.pi))
                    simulation_app.update()
                    if self.clash_detector.is_prim_clashing(get_prim_at_path(rack_path), query_name=f"rack_{idx}_query"):
                        self.get_logger().info('Collision detected!')
                        validStage = False
                        
            self.clash_detector.set_scope('')
            self.randomizeRackContent(self.rack_list[0], self.tube_list[:6], 1)
            self.randomizeRackContent(self.rack_list[1], self.tube_list[6:11], 0)
            
            if all( not self.tube_usage[tube_path] or 
                    (self.realisticPose(tube_path) and self.collisionWithRacks(tube_path) and not self.collisionWithTubes(tube_path)) 
                    for tube_path in self.tube_list):
                break
        
        self.clash_detector.set_scope('')
        
        self.get_logger().info('\nRacks:\n' + str(self.rack_usage) + '\nTubes:\n' + str(self.tube_usage))
        
        response.success = True
        response.message = 'New scene created successfully.'
        
        return response

    # * /IsaacSim/ResetScene
    def resetSceneCallback(self, request = Trigger.Request(), response = Trigger.Response()):
        self.get_logger().info('Reset scene request')
        self.resetTubes()
        self.resetRacks()
        self.get_logger().info('Successfully reseted scene')
        response.success = True
        response.message = ""
        return response

    def NoneCallback(self, request):
        pass
   

def main():
    #Init ROS2
    try:
        rclpy.init(args=None)

        node = IsaacSim('SceneHandler')
        node.executor = MultiThreadedExecutor()
        node.runApp()
        
        #Exiting
        node.destroy_node()
        rclpy.shutdown()
            
    except(KeyboardInterrupt, ExternalShutdownException):
        pass
        
if __name__ == '__main__':
    main()