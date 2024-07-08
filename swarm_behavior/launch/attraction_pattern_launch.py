import os
import launch_ros.actions
from webots_ros2_driver.webots_launcher import WebotsLauncher
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from webots_ros2_driver.utils import controller_url_prefix


def generate_launch_description():
    ld = LaunchDescription()

    config_dir = os.path.join(get_package_share_directory('swarm_behavior'), 'config')

    # Webots
    webots = WebotsLauncher(
        world=os.path.join(get_package_share_directory('swarm_behavior'), 'worlds', 'mul_epuck_world.wbt')
    )
    ld.add_action(webots)

    # Controller node
    for i in range(8):
        robot_name = 'epuck_' + str(i)
        controller = Node(
        package='swarm_behavior',
        executable='attraction_pattern',
        output='screen',
        additional_env={'WEBOTS_CONTROLLER_URL': controller_url_prefix() + robot_name},
        parameters=[PathJoinSubstitution([config_dir, 'attraction_pattern.yaml'])],
        )
        ld.add_action(controller)


    return ld