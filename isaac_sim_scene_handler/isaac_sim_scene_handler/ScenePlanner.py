#! ${USER}/isaacsim/setup_python_env.sh

import os
import sys
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import random
from time import sleep
from std_srvs.srv import Trigger
from isaac_sim_msgs.srv import PoseRequest, TubeParameter
from sensor_msgs.msg import JointState

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
from isaacsim import SimulationApp

import ament_index_python.packages as ament_packages

CONFIG = {"renderer": "RaytracedLighting", "headless": False}
#CONFIG = {"headless": False}

USD_NAME = "tm5-900_dev.usd"

# Simple example showing how to start and stop the helper
simulation_app = SimulationApp(CONFIG)

import carb
from carb import Float3, Float4

import omni

from omni.isaac.dynamic_control import _dynamic_control
from isaacsim.core.api import World
from isaacsim.core.utils import extensions, stage
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.dynamic_control import _dynamic_control
from isaacsim.storage.native import get_assets_root_path, is_file
from isaacsim.core.utils.stage import is_stage_loading, get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.bounds as bounds_utils

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

    rack_list : list = [
        '/World/Racks/Rack_Start',
        '/World/Racks/Rack_Goal',
    ]
    
    rack_storage_position = np.array([-1.0, 0.0, 0.0])
    rack_storage_position_relative = np.array([-0.1, 0.0, 0.0])
    rack_storage_rotation = Rotation.from_euler('xyz',[np.pi/2, 0.0, 0.0])
    
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
    
    tube_storage_position = np.array([-1.0, 0.0, 0.022])
    tube_storage_position_relative = np.array([0.0, -0.04, 0.0])
    tube_storage_rotation = Rotation.from_euler('xyz',[np.pi/2, np.pi/2, 0])
    
    workspace = np.array([[0.4, -0.2, 0.00], [0.6, 0.2, 0.00]])
    rack_to_hole_translation = np.array([-0.0675, 0.065, 0.0])
    hole_to_hole_translation = np.array([0.0, 0.03, 0.0])
    
    
    

    def __init__(self, nodename : str) -> None:
        
        super().__init__(nodename)
        
        self.joint_command = self.create_subscription(JointState, '/joint_command', self.NoneCallback, 10)
        self.joint_states = self.create_publisher(JointState, '/joint_states', 10)
        
        self.launchIsaacSim()

        self.tube_in_use : dict = {}
        
        self.stage = get_current_stage()
        
        # Initialize clash detection engine
        self.clash_detector = ClashDetector(self.stage, logging=False, clash_data_layer=False)
        
        self.resetAssets()
        simulation_app.update()
        
        self.clash_detector.set_scope('/World/Racks')
        for idx, rack_path in enumerate(self.rack_list):
            validStage = False
            while not validStage:
                self.placeInWorkspace(rack_path, random.uniform(0.0, 2*np.pi))
                simulation_app.update()
                if not self.clash_detector.is_prim_clashing(get_prim_at_path(rack_path), query_name=f"rack_{idx}_query"):
                    validStage = True
                else:
                    self.get_logger().info('Collision detected!')
        self.clash_detector.set_scope('')
        self.cache = bounds_utils.create_bbox_cache()
        
        self.tube_height = 0.01*max(self.getAbsoluteBoundingBox('/World/Tubes/Tube_Target')[2])
        self.tube_diameter = 0.01*min(self.getAbsoluteBoundingBox('/World/Tubes/Tube_Target')[2])
        
        self.randomizeRackContent(self.rack_list[0], self.tube_list[:6], 1)
        self.randomizeRackContent(self.rack_list[1], self.tube_list[6:11], 0)
        
        self.resetScene = self.create_service(Trigger, '/IsaacSim/ResetScene', self.resetSceneCallback)
        self.newScene = self.create_service(Trigger, '/IsaacSim/NewScene', self.newSceneCallback)
        self.poseRequest = self.create_service(PoseRequest, '/IsaacSim/RequestPose', self.poseRequestCallback)
        self.tubeGraspPoseRequest = self.create_service(PoseRequest, '/IsaacSim/RequestTubeGraspPose', self.tubeGraspPoseRequestCallback)
        self.tubeParameterRequest = self.create_service(TubeParameter, '/IsaacSim/RequestTubeParameter', self.tubeParameterRequestCallback)
    
    def launchIsaacSim(self):
        
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0)
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        
        
        package_name = 'tm5-900_rg6_moveit_config'
        usd_path = '/home/user/ros2_ws/src/tm5-900_rg6_moveit_config/config/tm5-900/tm5-900_dev.usd'
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
        #stage.add_reference_to_stage(os.path.join(ament_packages.get_package_share_directory(package_name) , 'config', 'tm5-900' , USD_NAME), '/World')
        #stage.add_reference_to_stage('/home/user/ros2_ws/src/tm5-900_rg6_moveit_config/config/tm5-900/tm5-900_dev.usd', '/World')
        #prims.create_prim("/DistantLight", "DistantLight")
        
        simulation_app.update()
        while stage.is_stage_loading():
            simulation_app.update()
        
        self.world.play()
        
    def runApp(self):
        while simulation_app.is_running():
            # Run with a fixed step size
            reset_needed = False
            #self.world.step(render=True)
            simulation_app.update()
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.world.is_playing():
                if reset_needed:
                    self.ros_world.reset()
                    reset_needed = False
        
        self.timeline.stop()
        self.world.stop()
        simulation_app.close()  # Cleanup application

    def resetAssets(self):
            for idx, tube_path in enumerate(self.tube_list):
                tube = self.dc.get_rigid_body(tube_path)
                transform = _dynamic_control.Transform( 
                    Float3(self.tube_storage_position + idx*self.tube_storage_position_relative), 
                    Float4(self.tube_storage_rotation.as_quat())
                )
                self.tube_in_use[tube_path] = False
    
                self.dc.set_rigid_body_pose(tube, transform)
                
            for idx, rack_path in enumerate(self.rack_list):
                rack = self.dc.get_rigid_body(rack_path)
                transform = _dynamic_control.Transform( 
                    (self.rack_storage_position + idx*self.rack_storage_position_relative).tolist(), 
                    self.rack_storage_rotation.as_quat().tolist()
                )
    
                self.dc.set_rigid_body_pose(rack, transform)
                
    def resetSceneCallback(self, request, response):
        self.get_logger().info('Reset scene request')
        self.resetAssets()
        
        response.success = True
        response.message = ""
        return response

    def placeInWorkspace(self, rack_path, angle):

        rack = self.dc.get_rigid_body(rack_path)
        
        transform = _dynamic_control.Transform( 
            Float3(
                random.uniform(self.workspace[0][0], self.workspace[1][0]),
                random.uniform(self.workspace[0][1], self.workspace[1][1]),
                random.uniform(self.workspace[0][2], self.workspace[1][2])
            ), 
            Float4(Rotation.from_euler('xyz',[np.pi/2, 0.0, angle]).as_quat())
        )
        
        self.dc.set_rigid_body_pose(rack, transform)
        
    def placeTubeInRack(self, rack_path, tube_path, slot_num):
    
        tube = self.dc.get_rigid_body(tube_path)
        rack = self.dc.get_rigid_body(rack_path)

        rotation = Rotation.from_quat(self.dc.get_rigid_body_pose(rack).r)
        translation = self.dc.get_rigid_body_pose(rack).p
        self.get_logger().debug('Rack pose: ' + str(translation))
        self.get_logger().debug('Rack rotation: ' + str(rotation.as_euler('xyz')))
        abs_translation = translation
        translation = np.array([translation[0], translation[1], translation[2]]) + \
        Rotation.from_euler('xyz', [0.0, 0.0, rotation.as_euler('xyz')[2]]).apply(
                    self.rack_to_hole_translation + slot_num * self.hole_to_hole_translation
            )
            
        
        transform = _dynamic_control.Transform(
            Float3(translation.tolist()),
            Float4(Rotation.from_euler('xyz', [np.pi/2, 0.0, rotation.as_euler('xyz')[2]]).as_quat())
        )
        
        self.get_logger().debug('Tube abs translation: ' + str(translation))
        self.get_logger().debug('Tube rel translation: ' + str(np.array(transform.p) - np.array(abs_translation)) + ' ' + str(transform.r))
        self.get_logger().debug('Tube transformation: ' + str(transform))
        self.tube_in_use[tube_path] = True

        self.dc.set_rigid_body_pose(tube, transform)
        self.dc.set_rigid_body_angular_velocity(tube, Float3([0.0, 0.0, 0.0]))
        self.dc.set_rigid_body_linear_velocity(tube, Float3([0.0, 0.0, 0.0]))
            
    def randomizeRackContent(self, rack_path, tube_list : list, min_placed_tubes):
        rack = self.dc.get_rigid_body(rack_path)
        
        choices = [i for i in range(6)]
        
        for i in range(random.randint(min_placed_tubes, len(tube_list))):
            slot = random.choice(choices)
            self.get_logger().debug('Rack: ' + rack_path)
            self.get_logger().debug('Tube: ' + tube_list[i])
            self.get_logger().debug('Slot: ' + str(slot))
            self.placeTubeInRack(rack_path, tube_list[i], slot)
            choices.remove(slot)
    
    def newSceneCallback(self, request, response):
        self.get_logger().info('New scene request')
        
        self.resetAssets()
        simulation_app.update()
        
        self.clash_detector.set_scope('/World/Racks')
        for idx, rack_path in enumerate(self.rack_list):
            validStage = False
            while not validStage:
                self.placeInWorkspace(rack_path, random.uniform(0.0, 2*np.pi))
                simulation_app.update()
                if not self.clash_detector.is_prim_clashing(get_prim_at_path(rack_path), query_name=f"rack_{idx}_query"):
                    validStage = True
                else:
                    self.get_logger().info('Collision detected!')
        
        self.randomizeRackContent(self.rack_list[0], self.tube_list[:6], 1)
        self.randomizeRackContent(self.rack_list[1], self.tube_list[6:11], 0)
        
        response.success = True
        response.message = ''
        
        return response

    def poseRequestCallback(self, request, response):
        self.get_logger().info('Pose request')
        prim = self.dc.get_rigid_body(request.path)
        prim_pose = self.dc.get_rigid_body_pose(prim)
        
        response.pose.translation.x = prim_pose.p[0]
        response.pose.translation.y = prim_pose.p[1]
        response.pose.translation.z = prim_pose.p[2]
        self.get_logger().info('['+str(prim_pose.p[0])+','+str(prim_pose.p[1])+','+str(prim_pose.p[2])+']')
        
        response.pose.rotation.x = prim_pose.r[0]
        response.pose.rotation.y = prim_pose.r[1]
        response.pose.rotation.z = prim_pose.r[2]
        response.pose.rotation.w = prim_pose.r[3]
        self.get_logger().info('['+str(prim_pose.r[0])+','+str(prim_pose.r[1])+','+str(prim_pose.r[2])+','+str(prim_pose.r[3])+']')
        
        return response

    def tubeGraspPoseRequestCallback(self, request : PoseRequest.Request, response : PoseRequest.Response):
        simulation_app.update()
        self.get_logger().info('Tube pose request')
        prim = self.dc.get_rigid_body(request.path)
        prim_pose = self.dc.get_rigid_body_pose(prim)
        
        self.cache = bounds_utils.create_bbox_cache()
        
        axes : ndarray
        centroid, axes, half_extent = self.getAbsoluteBoundingBox(request.path)
        
        scale_factor = np.absolute(np.linalg.eigvals(axes))
        rotation = Rotation.from_matrix((axes.T/scale_factor).T).inv()
        end_vec = rotation.apply(np.multiply(scale_factor, [extent if extent == max(half_extent) else 0.0 for extent in half_extent]))
        quat = rotation.as_quat()
        
        self.get_logger().info('\nAxes:\n' + str(axes) + '\nRotation:\n' + str(rotation.as_matrix()))
        self.get_logger().info('End vector: ' + str(end_vec))
        self.get_logger().info('Centroid: ' + str(centroid))
        
        response : PoseRequest.Response = PoseRequest.Response()
        response.pose.translation.x = centroid[0] + end_vec[0] 
        response.pose.translation.y = centroid[1] + end_vec[1]
        response.pose.translation.z = centroid[2] + end_vec[2]
        self.get_logger().info('['+str(response.pose.translation.x)+','+str(response.pose.translation.y )+','+str(response.pose.translation.z)+']')
        
        response.pose.rotation.x = quat[0] #prim_pose.r[0]
        response.pose.rotation.y = quat[1] #prim_pose.r[1]
        response.pose.rotation.z = quat[2] #prim_pose.r[2]
        response.pose.rotation.w = quat[3] #prim_pose.r[3]
        self.get_logger().info('['+str(prim_pose.r[0])+','+str(prim_pose.r[1])+','+str(prim_pose.r[2])+','+str(prim_pose.r[3])+']')
        
        
        # z_rotation = Rotation.from_euler('xyz', [0.0, 0.0, Rotation.from_quat(prim_pose.r).as_euler('xyz')[2]])
        # grasp_translation = np.array(prim_pose.p) + \
        #     z_rotation.apply(np.array([self.tube_diameter/2, -self.tube_diameter/2, self.tube_height]))
        
        # response : PoseRequest.Response = PoseRequest.Response()
        # response.pose.translation.x = grasp_translation[0]
        # response.pose.translation.y = grasp_translation[1]
        # response.pose.translation.z = grasp_translation[2] + 0.02
        # self.get_logger().info('['+str(response.pose.translation.x)+','+str(response.pose.translation.y )+','+str(response.pose.translation.z)+']')
        
        # response.pose.rotation.x = prim_pose.r[0]
        # response.pose.rotation.y = prim_pose.r[1]
        # response.pose.rotation.z = prim_pose.r[2]
        # response.pose.rotation.w = prim_pose.r[3]
        # self.get_logger().info('['+str(prim_pose.r[0])+','+str(prim_pose.r[1])+','+str(prim_pose.r[2])+','+str(prim_pose.r[3])+']')
        
        return response
        
    def tubeParameterRequestCallback(self, request : TubeParameter.Request, response : TubeParameter.Response):
        axes : ndarray
        centroid, axes, half_extent = self.getAbsoluteBoundingBox(request.path)
        scale_factor = np.absolute(np.linalg.eigvals(axes))
        quat = Rotation.from_matrix((axes.T / scale_factor).T).as_quat()
        self.get_logger().info('Half extent: ' + str(half_extent) + ' Scaling factor: ' + str(scale_factor))
        
        response.dimensions.x = 2*scale_factor[0]*half_extent[0]
        response.dimensions.y = 2*scale_factor[1]*half_extent[1]
        response.dimensions.z = 2*scale_factor[2]*half_extent[2]
        
        response.axis.x = quat[0]
        response.axis.y = quat[1]
        response.axis.z = quat[2]
        response.axis.w = quat[3]
        return response
        
    def NoneCallback(self, msg):
        pass
        
    def getWorldBoundingBox(self, path : str):    
        return bounds_utils.compute_aabb(self.cache, path)
        
    def getAbsoluteBoundingBox(self, path : str):
        centroid, axes, half_extent = bounds_utils.compute_obb(self.cache, prim_path=path)
        return (centroid, axes, half_extent)
    
def main():
    #Init ROS2
    try:
        rclpy.init(args=None)

        node = IsaacSim('IsaacSim')
        node.runApp()
        
        #Exiting
        node.destroy_node()
        rclpy.shutdown()
            
    except(KeyboardInterrupt, ExternalShutdownException):
        pass