from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
from launch.actions import DeclareLaunchArgument, TimerAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    console_format = SetEnvironmentVariable(
        'RCUTILS_CONSOLE_OUTPUT_FORMAT', '{message}'
    )


    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Flag to enable use_sim_time'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')


    openfusion_ros_config = os.path.join(
        get_package_share_directory("openfusion_ros"),
        'config',
        'openfusion_ros.yaml'
    )

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
            openfusion_ros_config
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
            openfusion_ros_config
        ]
    )

    # lcm = Node(
    #     package='nav2_lifecycle_manager',
    #     executable='lifecycle_manager',
    #     name='lifecycle_manager_detection',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': use_sim_time,
    #         'autostart': True,
    #         'bond_timeout': 0.0,
    #         'node_names': [
    #             '/openfusion_ros/openfusion_ros',
    #         ]
    #     }]
    # )

    # # Delayed lifecycle manager launch (5 seconds delay)
    # delayed_lcm = TimerAction(
    #     period=5.0,
    #     actions=[lcm]
    # )

    ld = LaunchDescription()
    ld.add_action(console_format)
    ld.add_action(sim_time_arg)
    ld.add_action(openfusion_ros_node)
    ld.add_action(openfusion_tf_bridge_node)
    # ld.add_action(delayed_lcm)  # Add the delayed lifecycle manager
    return ld