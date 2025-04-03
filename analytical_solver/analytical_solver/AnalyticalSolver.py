# ======================================================== #
# ===================== ROS 2 Imports ==================== #
# ======================================================== #
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor


# ======================================================== #
# ==================== Message Imports =================== #
# ======================================================== #
from std_srvs.srv import Trigger
from geometry_msgs.msg import Transform
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory
from isaac_sim_msgs.srv import PoseRequest


# ======================================================== #
# =================== MoveIt 2 Imports =================== #
# ======================================================== #
from moveit.planning import MoveItPy
from moveit.planning import PlanningComponent
from moveit.planning import TrajectoryExecutionManager


# ======================================================== #
# ==================== Python Imports ==================== #
# ======================================================== #
from time import sleep


class AnalyticalSolver(Node):

    # * Target prim's path to solve the task 
    target = '/World/Tubes/Tube_Target'
    
    def __init__(self, nodename : str, robot : MoveItPy):
        super().__init__(nodename)
        
        self.targetPose : Transform
        self.robotState : JointState = JointState()
        self.stopped = False
        
        # Wait for IsaacSim node to set up
        self.poseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestPose')
        self.tubeGraspPoseRequest = self.create_client(PoseRequest, '/IsaacSim/RequestTubeGraspPose')
        while not (self.poseRequest.wait_for_service(timeout_sec=30.0) and self.tubeGraspPoseRequest.wait_for_service(timeout_sec=30.0)):
            if not rclpy.ok():
                self.get_logger().error('Interruped while waiting for the server.')
                return
            else:
                self.get_logger().info('Server not available, waiting again...')
        
        self.move_to_tube = self.create_service(Trigger, '/AnalyticSolver/MoveToTube', self.moveToTubeCallback)
        
        self.robot_velocity = self.create_subscription(JointState, '/joint_states', self.refreshRobotValues, 10)

        sleep(5)
        
        self.robot = robot
        self.robot_arm : PlanningComponent = robot.get_planning_component('tmr_arm')
        self.robot_gripper : PlanningComponent = robot.get_planning_component('rg6')
        self.get_logger().info('MoveItPy instance created')
        self.execution_manager : TrajectoryExecutionManager = self.robot.get_trajectory_execution_manager()
        
        self.robot_arm.set_start_state_to_current_state()
        
        self.robot_arm.set_goal_state(configuration_name='ready1')
        
        self.move_robot_arm()

    def refreshRobotValues(self, msg : JointState):
            if any(joint_name == 'joint_1' for joint_name in msg.name):
                self.robotState = msg
                #self.get_logger().info('[' + ', '.join([str(self.robotState.name[idx]) + ': ' + str(velocity) for idx, velocity in enumerate(self.robotState.velocity) if 'joint_' in self.robotState.name[idx]]) +']')
                self.stopped = True if all(abs(velocity) < 5e-3 for idx, velocity in enumerate(self.robotState.velocity) if 'joint_' in self.robotState.name[idx]) else False

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
        
        request = Trigger.Request()
        self.future  = self.poseRequest.call_async(request)
        while not self.future.done:
            sleep(0.001)
            
        self.targetPose = self.future.result()
        
        self.robot_arm.set_start_state_to_current_state()
        
        goal : PoseStamped = PoseStamped()
        goal.header.frame_id = "link1"
        
        goal.pose.orientation.x = self.targetPose.rotation.x
        goal.pose.orientation.y = self.targetPose.rotation.y
        goal.pose.orientation.z = self.targetPose.rotation.z
        goal.pose.orientation.w = self.targetPose.rotation.w
        goal.pose.position.x = self.targetPose.translation.x
        goal.pose.position.y = self.targetPose.translation.y
        goal.pose.position.z = self.targetPose.translation.z + 0.3
        
        self.robot_arm.set_goal_state(pose_stamped_msg=goal, pose_link="link1")
        
        self.plan_and_execute('')
        
        response.success = True
        response.message = ''
        
        return response

    def move_robot_arm(
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
                self.robot.execute(robot_trajectory, controllers=[])
                while not self.stopped:
                    rclpy.spin_once(self, timeout_sec=0.1)  
                self.get_logger().info('Movement completed!')
        else:
                self.get_logger().error("Planning failed") 
                
def main():
    try:
        rclpy.init(args=None)
        
        executor = MultiThreadedExecutor()
        
        robot = MoveItPy(node_name='MoveItPy')
        solver = AnalyticalSolver('AnalyticalSolver', robot)
        
        executor.add_node(solver)
        
        executor.spin()
        
        solver.destroy_node()
        robot.shutdown()
        
        rclpy.shutdown()
        
    except(KeyboardInterrupt, ExternalShutdownException):
        pass

if __name__ == '__main__':
    main()
