import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    robot_launch = os.path.join(get_package_share_directory('tm5-900_rg6_moveit_config'), 'launch', 'tm5-900_isaac_sim.launch.py')
        
    isaac_sim_launch = os.path.join(get_package_share_directory('isaac_sim_imitation_learning'), 'launch', 'isaac_sim.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_launch)
        ),
            
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(isaac_sim_launch)
        ),
    ])
