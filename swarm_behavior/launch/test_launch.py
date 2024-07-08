# import os
# import pathlib
# import launch
# from launch.substitutions import LaunchConfiguration
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# from launch.substitutions.path_join_substitution import PathJoinSubstitution
# from launch_ros.actions import Node
# from launch import LaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from ament_index_python.packages import get_package_share_directory
# from webots_ros2_driver.webots_launcher import WebotsLauncher
# from webots_ros2_driver.utils import controller_url_prefix

# def get_ros2_nodes(*args):
#     controller = Node(
#         package='webots',
#         executable='test',
#         output='screen',
#     )
#     return [
#         controller
#     ]


# def generate_launch_description():
#     package_dir = get_package_share_directory('webots')
#     world = LaunchConfiguration('world')

#     webots = WebotsLauncher(
#         world=PathJoinSubstitution([package_dir, 'worlds', world]),
#         ros2_supervisor=True
#     )

#     reset_handler = launch.actions.RegisterEventHandler(
#         event_handler=launch.event_handlers.OnProcessExit(
#             target_action=webots._supervisor,
#             on_exit=get_ros2_nodes,
#         )
#     )



#     return launch.LaunchDescription([
#         DeclareLaunchArgument(
#             'world',
#             default_value='epuck_world.wbt',
#             description='Choose one of the world files from `/webots_ros2_epuck/world` directory'
#         ),
#         webots,
#         webots._supervisor,
#         # This action will kill all nodes once the Webots simulation has exited
#         launch.actions.RegisterEventHandler(
#             event_handler=launch.event_handlers.OnProcessExit(
#                 target_action=webots,
#                 on_exit=[
#                     launch.actions.UnregisterEventHandler(
#                         event_handler=reset_handler.event_handler
#                     ),
#                     launch.actions.EmitEvent(event=launch.events.Shutdown())
#                 ],
#             )
#         ),

#         # Add the reset event handler
#         reset_handler
#     ] + get_ros2_nodes())


import os
import launch_ros.actions
import launch
import time
from launch.actions import TimerAction
#from webots_ros2_driver.utils import ControllerLauncher
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
        world=os.path.join(get_package_share_directory('swarm_behavior'), 'worlds', 'leader_followers.wbt')
    )
    ld.add_action(webots)

    # Controller node
    for i in range(3):
        robot_name = 'epuck_' + str(i)
        controller = Node(
        package='swarm_behavior',
        executable='test',
        output='screen',
        additional_env={'WEBOTS_CONTROLLER_URL': controller_url_prefix() + robot_name},
        parameters=[PathJoinSubstitution([config_dir, 'combined_pattern.yaml'])],
        )
        ld.add_action(controller)


    return ld