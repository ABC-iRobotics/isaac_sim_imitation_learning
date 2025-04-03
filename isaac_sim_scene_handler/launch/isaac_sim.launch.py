import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    package_name = 'isaac_sim_scene_handler'
    
    package_share_dir = get_package_share_directory(package_name)
    node_path = os.path.join(os.path.dirname(os.path.dirname(package_share_dir)), 'lib', package_name, 'ScenePlanner')

    return LaunchDescription([
    
        DeclareLaunchArgument(
            'python_executable',
            default_value= os.path.join(os.path.expanduser("~"), 'isaacsim/python.sh'),
            description='Path to the Isaac Sim Python interpreter (python.sh file)'
        ),
        
        ExecuteProcess(
            cmd=[LaunchConfiguration('python_executable'), node_path],
            name='ScenePlanner',
            output='log',
        )
    ])