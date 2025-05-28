import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.client import Client

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool

from std_srvs.srv import Trigger

import json, time
from time import sleep

class TrajectoryRecorder(Node):
    
    def __init__(self, nodename : str):
        super().__init__(nodename)
        
        self.reentrantGroup = ReentrantCallbackGroup()
        self.mutuallyExclusiveGroup = MutuallyExclusiveCallbackGroup()
        
        self.rate = self.create_rate(100)
        
        self.robotJoints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.gripperJoints = ['finger_joint']
        
        self.jointState = JointState()
        self.gripperState = JointState()
        self.jointCommand = JointState()
        self.cameraImage = Image()
        self.grasp = Bool()
        
        self.jointStateSub = self.create_subscription(JointState, '/joint_states', self.jointStateCallback, 10, callback_group=self.reentrantGroup)
        self.graspSub = self.create_subscription(Bool, '/grasp', self.jointCommandCallback, 10, callback_group=self.reentrantGroup)
        self.jointCommandSub = self.create_subscription(JointState, '/joint_command', self.jointCommandCallback, 10, callback_group=self.reentrantGroup)
        self.cameraSub = self.create_subscription(Image, '/rgb', self.cameraCallback, 10, callback_group=self.reentrantGroup)
        
        self.newSceneRequest = self.create_client(Trigger, '/IsaacSim/NewScene', callback_group=self.reentrantGroup)
        self.solveSceneRequest = self.create_client(Trigger, '/AnalyticSolver/SolveScene', callback_group=self.reentrantGroup)
        
        self.getTrajectoryService = self.create_service(Trigger, '/TrajectoryRecorder/GetTrajectory', self.getTrajectoryServiceCallback)
        
        while   not self.newSceneRequest.wait_for_service(timeout_sec=5.0) and \
                not self.solveSceneRequest.wait_for_service(timeout_sec=5.0):
            if not rclpy.ok():
                self.get_logger().error('Interruped while waiting for the server.')
                return
            else:
                self.get_logger().info('Server not available, waiting again...')

    def jointStateCallback(self, msg : JointState):
        if (set(msg.name).issubset(self.robotJoints)):
            self.jointState = msg
        elif(set(msg.name).issubset(self.gripperJoints)):
            self.gripperState = msg
            
    def jointCommandCallback(self, msg : JointState):
        if (set(msg.name).issubset(self.robotJoints)):
            self.jointcommand = msg
        elif(set(msg.name).issubset(self.gripperJoints)):
            self.jointcommand = msg

    def cameraCallback(self, msg : Image):
        self.cameraImage = msg
        
    def gripperCallback(self, msg : Bool):
        self.grasp = msg
        
    def getTrajectoryServiceCallback(self, request, response):
        self.get_logger().info('Received getTrajectory request')
        success = False
        request : Trigger.Request = Trigger.Request()
        
        # while not success:
        while True:
            self.callService(service=self.newSceneRequest, request=request)
            
            success = self.callService(service=self.solveSceneRequest, request=request).success
        
            sleep(5)
            
        response.success = True
        response.message = ''
        return response
        
    @property
    def record(self):
        return {
            "timestamp": self.get_clock().now(),
            
            "action": {}
        }

    def callService(self, service : Client, request, message : str | None = None):
        if isinstance(message, str):
            self.get_logger().info(message)
        self.get_logger().info('Calling ' + str(service.srv_name) + ' service.')
        # future  = service.call_async(request)
        # self.rate.sleep()
        # while not future.done():
        #     self.rate.sleep()
        #     sleep(0.01)
        # while isinstance(future.result(), type(None)):
        #     self.rate.sleep()
        #     sleep(0.01)
        # self.rate.sleep()
        # sleep(0.01)
        # self.rate.sleep()
        # response = future.result()
        response = service.call(request)
        self.get_logger().info('Called' + str(service.srv_name) + ' successfully.')
        return response

    def record(self):
        """
        Data collection loop. Runs until self.is_recording is False.
        """
        data_log = []

        print("Recording started...")
        while self.is_recording:
            sample = self.get_sample()
            data_log.append(sample)
            time.sleep(self.sample_interval)

        print(f"Recording stopped. Saving to {self.output_file}...")
        with open(self.output_file, 'w') as f:
            json.dump(data_log, f, indent=2)
        print("Save complete.")

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