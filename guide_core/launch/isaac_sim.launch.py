import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    package_name = "guide_core"

    package_share_dir = get_package_share_directory(package_name)
    node_path = os.path.join(
        os.path.dirname(os.path.dirname(package_share_dir)),
        "lib",
        package_name,
        "ScenePlanner",
    )

    # Isaac Sim 6.0 runs on Python 3.12 (same as ROS 2 Jazzy) so ROS 2 works
    # natively — no bundled rclpy, no path scrubbing, no message overlay. Source
    # Jazzy + the workspace before launching. Override interpreter via ISAACSIM_PYTHON.
    default_python = os.environ.get(
        "ISAACSIM_PYTHON",
        os.path.join(os.path.expanduser("~"), "ros2_ws", ".venv", "bin", "python"),
    )

    env = {"OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA", "YES")}

    # Same-host DDS discovery needs a localhost cyclonedds config (loopback has no
    # MULTICAST flag + multiple NICs). Respect a user override.
    if "CYCLONEDDS_URI" not in os.environ:
        cdds_cfg = os.path.join(package_share_dir, "config", "cyclonedds_localhost.xml")
        if os.path.isfile(cdds_cfg):
            env["CYCLONEDDS_URI"] = f"file://{cdds_cfg}"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "python_executable",
                default_value=default_python,
                description="Path to the Isaac Sim Python interpreter (the '.venv' Python 3.12 "
                "for pip-installed Isaac Sim 6.0; override via ISAACSIM_PYTHON).",
            ),
            ExecuteProcess(
                cmd=[LaunchConfiguration("python_executable"), node_path],
                name="ScenePlanner",
                output="log",
                additional_env=env,
            ),
        ]
    )
