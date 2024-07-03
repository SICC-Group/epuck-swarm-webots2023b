import os
from webots_ros2_driver.webots_launcher import WebotsLauncher
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()

    # Webots
    webots = WebotsLauncher(
        world=os.path.join(get_package_share_directory('webots'), 'worlds', 'epuck_world.wbt')
    )
    ld.add_action(webots)

    # Controller node
    controller = Node(
        package='webots',
        executable='obstacle_avoidance',
        output='screen',
    )
    ld.add_action(controller)


    return ld