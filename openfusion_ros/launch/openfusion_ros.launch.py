from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
from launch.actions import DeclareLaunchArgument, TimerAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

'''
Trigger Service to run semantic mapping for evaluation:
ros2 service call /openfusion_ros/run_semantic_map std_srvs/srv/Trigger
'''

def generate_launch_description():

    console_format = SetEnvironmentVariable(
        'RCUTILS_CONSOLE_OUTPUT_FORMAT', '{message}'
    )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Flag to enable use_sim_time'
    )

    openfusion_ros_default_config = os.path.join(
        get_package_share_directory("openfusion_ros"),
        'config',
        'openfusion_ros.yaml'
    )

    openfusion_config_arg = DeclareLaunchArgument(
        'openfusion_config',
        default_value=openfusion_ros_default_config,
        description='Path to OpenFusion ROS config file'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')


    openfusion_config = LaunchConfiguration('openfusion_config')

    openfusion_ros_node = Node(
        package='openfusion_ros',
        executable='openfusion_ros',
        name="openfusion_ros",
        namespace='openfusion_ros',
        output='screen',
        emulate_tty=True,
        # arguments=['--ros-args', '--log-level', 'debug'],
        parameters=[
            {'use_sim_time': use_sim_time},
            openfusion_config
        ]
    )

    openfusion_tf_bridge_node = Node(
        package='openfusion_tf_bridge',
        executable='openfusion_tf_bridge_node',
        name='openfusion_tf_bridge',
        namespace='openfusion_ros',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            openfusion_config
        ]
    )

    ld = LaunchDescription()
    ld.add_action(console_format)
    # Arguments
    ld.add_action(sim_time_arg)
    ld.add_action(openfusion_config_arg)
    # Nodes
    ld.add_action(openfusion_ros_node)
    ld.add_action(openfusion_tf_bridge_node)
    return ld