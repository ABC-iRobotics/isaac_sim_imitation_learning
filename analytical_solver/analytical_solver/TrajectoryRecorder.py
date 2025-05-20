import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool

from std_srvs.srv import Trigger

import json, time

class TrajectoryRecorder(Node):
    
    def __init__(self, nodename : str):
        super().__init__(nodename)
        
        self.reentrantGroup = ReentrantCallbackGroup()
        self.mutuallyExclusiveGroup = MutuallyExclusiveCallbackGroup()
        
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
        self.solveSceneRequest = self.create_client(Trigger, '/AnalyticalSolver/SolveScene', callback_group=self.reentrantGroup)
        
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
        
    @property
    def record(self):
        return {
            "timestamp": self.get_clock().now(),
            
            "action": {}
        }

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