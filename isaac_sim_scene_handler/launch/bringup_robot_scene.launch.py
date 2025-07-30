import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
import sys

import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    TextSubstitution
)

from launch_ros.actions import Node

import xacro
import yaml

from moveit_configs_utils import MoveItConfigsBuilder


def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except OSError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def generate_launch_description():
    # Declare arguments
    args = []
    length = len(sys.argv)
    if (len(sys.argv) >= 5):
        i = 4
        while i < len(sys.argv):
            args.append(sys.argv[i])
            i = i + 1

    # Configure robot_description
    tm_robot_type = 'tm5-900'
    description_path = 'tm_description'
    xacro_path = 'config/tm5-900.urdf.xacro'
    moveit_config_path = 'tm5-900_rg6_moveit_config'    
    srdf_path = 'config/tm5-900.srdf'
    rviz_path = '/config/moveit.rviz'
    controller_path = 'config/moveit_controllers.yaml'
    ros_controller_path = 'config/ros2_controllers.yaml'
    joint_limits_path = 'config/joint_limits.yaml'


    # * When using an external robot (or simulator), use mock_components
    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="mock_components",
        description="ROS2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )

    # MoveIt Configuration
    moveit_config = (
        MoveItConfigsBuilder(tm_robot_type, package_name="tm5-900_rg6_moveit_config")
        .robot_description(file_path=xacro_path,
            mappings={
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        .robot_description_semantic(file_path=srdf_path)
        .trajectory_execution(file_path=controller_path)
        .joint_limits(file_path=joint_limits_path)
        .planning_scene_monitor(
                publish_robot_description=True, publish_robot_description_semantic=True
            )
        .planning_pipelines(
                pipelines=["ompl"],
                default_planning_pipeline="ompl",
                load_all=True
            )
        .to_moveit_configs()
    )
    
    isaac_sim_launch = os.path.join(get_package_share_directory('isaac_sim_scene_handler'), 'launch', 'isaac_sim.launch.py')

    # Start the actual move_group node/action server
    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            #{'use_sim_time': True},
        ],
    )

    # RViz configuration
    rviz_config_file = (
        get_package_share_directory(moveit_config_path) + rviz_path
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,            
            moveit_config.joint_limits,
            #{'use_sim_time': False},
        ],
    )

    # Static TF Node: Publishes world -> base transform
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='log',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'world', 'base']
    )

    # Robot State Publisher Node: Publishes tf's for the robot
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            moveit_config.robot_description,
        ],
    )

    # ros2_control using FakeSystem as hardware
    ros2_controller_file = get_package_share_directory(moveit_config_path) + ros_controller_path

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[ros2_controller_file],
        remappings=[
            ('/controller_manager/robot_description', '/robot_description'),
        ],
        output='both',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager',
        ],
        remappings=[
            ('/joint_states', '/joint_command'),
        ]
    )

    tm_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            'tmr_arm_controller', 
            '--controller-manager',
            '/controller_manager',
        ],
    )

    # joint driver
    tm_driver_node = Node(
        package='tm_driver',
        executable='tm_driver',
        # name='tm_driver',
        output='log',
        respawn=True,
        arguments=args,
        remappings=[
            ('/joint_states', '/isaac_joint_commands'),
        ]
    )
    
    #Gripper type argument
    gripper_arg = DeclareLaunchArgument(
        'gripper',
        default_value=TextSubstitution(text='rg6')
    )
    
    control_arg = DeclareLaunchArgument(
        'control',
        default_value=TextSubstitution(text='isaac')
    )
    
    offset_arg = DeclareLaunchArgument(
        'offset',
        default_value='5'
    )
    
    isaac_joint_states_arg = DeclareLaunchArgument(
        'isaac_joint_states',
        default_value='/joint_states'
    )
    
    isaac_joint_commands_arg = DeclareLaunchArgument(
        'isaac_joint_commands',
        default_value='/isaac_joint_commands'
    )
    
    #Gripper node interfacing MoveIt2 and Isaac Sim
    rg6_node = Node(
        package='onrobot_rg_control',
        executable='OnRobotRGControllerServer',
        name='OnRobotRGControllerServer',
        output='screen',
        arguments=[],
        parameters=[{
            '/onrobot/gripper': LaunchConfiguration('gripper'),
            '/onrobot/control': LaunchConfiguration('control'),
            '/onrobot/isaac_joint_states': LaunchConfiguration('isaac_joint_states'),
            '/onrobot/isaac_joint_commands': LaunchConfiguration('isaac_joint_commands'),
        }],
    )
    
    analytic_solver = Node(
            package='analytic_solver',
            executable='AnalyticSolver',
            output='both',
            parameters=[
                moveit_config.to_dict(),
                {'use_sim_time': True},
                ],
    )
    
    trajector_recorder = Node(
        package='analytic_solver',
        executable='TrajectoryRecorder',
        output='both',
        parameters=[
            {'use_sim_time': True},
            ],
    )
    

    # Launching all the nodes
    return LaunchDescription(
        [
            # ~~~~~~~~~~~~~~~ Arguments ~~~~~~~~~~~~~~~ #
            gripper_arg,
            control_arg,
            offset_arg,
            isaac_joint_states_arg,
            isaac_joint_commands_arg,
            ros2_control_hardware_type,
            
            # ~~~~~~~~~~~~~~~~ Includes ~~~~~~~~~~~~~~~ #
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(isaac_sim_launch)
            ),

           # ~~~~~~~~~~~~~~~~~ Nodes ~~~~~~~~~~~~~~~~~ #
            launch_ros.actions.SetParameter(name='use_sim_time', value=True),
            #rviz_node,
            tm_driver_node,
            static_tf,
            robot_state_publisher,
            run_move_group_node,
            ros2_control_node,
            rg6_node,
            analytic_solver,
            trajector_recorder,
            # joint_state_broadcaster_spawner,
            # tm_arm_controller_spawner,
        ]
    )