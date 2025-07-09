from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    package_name = 'openfusion_ros'

    # Find config and rviz path
    pkg_share = FindPackageShare(package_name).find(package_name)
    param_file = os.path.join(pkg_share, 'config', 'openfusion_ros.yml')

    # Rviz configuration
    rviz_config = os.path.join(pkg_share, 'rviz', 'rviz.rviz')

    openfusion_ros = Node(
        package=package_name,
        executable='openfusion_ros',
        name="openfusion_ros",
        output='screen',
        emulate_tty=True,
        parameters=[param_file],
        namespace='openfusion_ros',
        # arguments=['--ros-args', '--log-level', 'debug']
    )

    rviz_node = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        openfusion_ros,
        rviz_node
    ])
