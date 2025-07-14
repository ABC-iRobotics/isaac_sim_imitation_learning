# ======================================================== #
# ==================== Python Imports ==================== #
# ======================================================== #
import time, os, re, cv2
from cv_bridge import CvBridge
from PIL import Image

# ======================================================== #
# ===================== ROS 2 Imports ==================== #
# ======================================================== #
import rclpy
import rclpy.action
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.client import Client

# ======================================================== #
# ==================== ROS 2 Messages ==================== #
# ======================================================== #
from std_srvs.srv import Trigger
from isaac_sim_msgs.srv import Demonstration as DemoSrv

from isaac_sim_msgs.action import Demonstration

# ======================================================== #
# ==================== LeRobot Imports =================== #
# ======================================================== #
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame

class TrajectoryRecorder(Node):
    
    def __init__(self, nodename : str):
        super().__init__(nodename)
        
        self.bridge = CvBridge()
        
        self.success = False
        
        self.reentrantGroup = ReentrantCallbackGroup()
        self.mutuallyExclusiveGroup = MutuallyExclusiveCallbackGroup()
        
        # ~~~~~~~~~~~~~~ ROS Clients ~~~~~~~~~~~~~~ #
        self.newSceneRequest = self.create_client(Trigger, '/IsaacSim/NewScene', callback_group=self.reentrantGroup)
        self.solveSceneRequest = self.create_client(Trigger, '/AnalyticSolver/SolveScene', callback_group=self.reentrantGroup)
        
        # ~~~~~~~~~~~~~~ ROS Servers ~~~~~~~~~~~~~~ #
        self.getTrajectoryService = self.create_service(DemoSrv, '/TrajectoryRecorder/GetTrajectory', self.getTrajectoryServiceCallback, callback_group=self.reentrantGroup)
        self.generateTrajectory : rclpy.action.ActionClient = rclpy.action.ActionClient(self, Demonstration, '/AnalyticSolver/GetDemonstration', callback_group=self.reentrantGroup)
        
        while   not self.newSceneRequest.wait_for_service(timeout_sec=5.0) and \
                not self.solveSceneRequest.wait_for_service(timeout_sec=5.0) and \
                not self.generateTrajectory.wait_for_server(timeout_sec=5.0):
            if not rclpy.ok():
                self.get_logger().error('Interruped while waiting for the server.')
                return
            else:
                self.get_logger().info('Server not available, waiting again...')
    
    def get_demonstration_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by server')
            return
        else:
            self.get_logger().info('Goal accepted by server, waiting for result')
    
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_demonstration_result_callback)
    
    def get_demonstration_feedback_callback(self, feedback_msg : Demonstration):
        try:
            # ~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~ #
            observation = {
                'joint_1': feedback_msg.feedback.arm_state.position[0],
                'joint_2': feedback_msg.feedback.arm_state.position[1],
                'joint_3': feedback_msg.feedback.arm_state.position[2],
                'joint_4': feedback_msg.feedback.arm_state.position[3],
                'joint_5': feedback_msg.feedback.arm_state.position[4],
                'joint_6': feedback_msg.feedback.arm_state.position[5],
                'finger_joint': feedback_msg.feedback.gripper_state.position[0],
                'cam': Image.fromarray(cv2.cvtColor(self.bridge.imgmsg_to_cv2(feedback_msg.feedback.camera, desired_encoding='bgr8'), cv2.COLOR_BGR2RGB))
            }
            observation_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")
            
            # ~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~ #
            action = {
                'joint_1': feedback_msg.feedback.arm_action.position[0],
                'joint_2': feedback_msg.feedback.arm_action.position[1],
                'joint_3': feedback_msg.feedback.arm_action.position[2],
                'joint_4': feedback_msg.feedback.arm_action.position[3],
                'joint_5': feedback_msg.feedback.arm_action.position[4],
                'joint_6': feedback_msg.feedback.arm_action.position[5],
                'finger_joint': feedback_msg.feedback.gripper_action.position[0],
                'use_camera': 1.0 if len(feedback_msg.feedback.camera.data) > 0 else 0.0
            }
            action_frame = build_dataset_frame(self.dataset.features, action, prefix="action")
            
            frame = {**observation_frame, **action_frame}
            self.dataset.add_frame(frame, task='') # TODO: Feedback task name in ROS 2 action

            self.start = False
        except Exception as e:
            self.get_logger().info('Error: ' + str(e))
            
    def get_demonstration_result_callback(self, future):
        result : Demonstration.Result = future.result().result
        
        self.success = result.success
        self.done_execution = True
        self.get_logger().info('Server successfully executed goal')
    
    def getTrajectoryServiceCallback(self, request : DemoSrv.Request, response):
        self.get_logger().info('Received getTrajectory request')
        
        response = DemoSrv.Response()
        
        # ~~~~~~~~~~~~~~~~~ Inputs ~~~~~~~~~~~~~~~~ #
        amount = request.amount
        path = request.path
        
        # ~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~ #
        obs_features={
            'joint_1': float,
            'joint_2': float,
            'joint_3': float,
            'joint_4': float,
            'joint_5': float,
            'joint_6': float,
            'finger_joint': float,
            'cam': (1216, 1936, 3)             
        }
        
        # ~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~ #
        action_features={
            'joint_1': float,
            'joint_2': float,
            'joint_3': float,
            'joint_4': float,
            'joint_5': float,
            'joint_6': float,
            'finger_joint': float,
            'use_camera': float
        }
        
        obs_features = hw_to_dataset_features(obs_features, 'observation', use_video=True)
        action_features = hw_to_dataset_features(action_features, 'action', use_video=True)
        dataset_features = {**action_features, **obs_features}
        
        # ~~~~~~~~~~~~~ Create dataset ~~~~~~~~~~~~ #
        repo_id = 'dataset_' + str(self.find_max_dataset_number(path)+1)
        self.dataset : LeRobotDataset
        self.dataset = LeRobotDataset.create(
                repo_id=repo_id, 
                fps=20,
                features=dataset_features,
                root=os.path.join(os.path.expanduser(path), repo_id),
                robot_type='tm5-900',
                use_videos=True,
                image_writer_threads=10,
                image_writer_processes=10,
                tolerance_s=0.1
                )
                
        # ~~~~~~~~~~ Start demonstrations ~~~~~~~~~ #
        for _ in range(amount):
            self.success = False
            while not self.success:
                try:
                    self.dataset.clear_episode_buffer() # Empty buffer in case of unsuccessful previous episode
                    self.start = True
                    self.done_execution = False
                    self.callService(service=self.newSceneRequest, request=Trigger.Request()) # Generate new scene
                    goal = Demonstration.Goal()
                    self.send_goal_future = self.generateTrajectory.send_goal_async(
                       goal, feedback_callback=self.get_demonstration_feedback_callback) # Call demonstration generation action
                    self.send_goal_future.add_done_callback(self.get_demonstration_goal_response_callback)
                    while not self.done_execution:
                        time.sleep(1/50.0) # Wait until episode ends
                        self.get_logger().debug('DEmonstration ongoing...')
                except Exception as e:
                    self.get_logger().info('Exception at demo generation: ' + str(e))

            self.get_logger().info('Got successful demonstration')    
            self.dataset.save_episode()
            
        response.success = True
        response.message = ''
        return response

    def callService(self, service : Client, request, message : str | None = None):
        if isinstance(message, str):
            self.get_logger().info(message)
        self.get_logger().info('Calling ' + str(service.srv_name) + ' service.')
        response = service.call(request)
        self.get_logger().info('Called' + str(service.srv_name) + ' successfully.')
        return response
    
    def find_max_dataset_number(self, path):
        max_number = -1
        pattern = re.compile(r'^dataset_(\d+)$')
        directory = os.path.expanduser(path)
        for name in os.listdir(directory):
            match = pattern.match(name)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)
    
        return max_number
    
      
def main():
    
     #Init ROS2
    try:
        rclpy.init(args=None)

        executor = MultiThreadedExecutor()
        recorder = TrajectoryRecorder('TrajectoryRecorder')
        executor.add_node(recorder)
        executor.spin()

        recorder.destroy_node()

        rclpy.shutdown()

    except(KeyboardInterrupt, ExternalShutdownException):
        pass

if __name__ == '__main__':
    main()