import rclpy
import os
import multiprocessing
import time

from swarm_frame.EpuckNode import EpuckNode
from swarm_frame.Behavior import Behavior

def main_robot(behavior_name:str):
    rclpy.init()
    print("环境变量：", os.environ["WEBOTS_CONTROLLER_URL"])

    robot = EpuckNode('robot')
    robot.start_device()

    behavior = Behavior(robot)
    behavior_method = getattr(behavior, behavior_name, None)
    if behavior_method:
        behavior_method()
    else:
        print(f"行为方法 {behavior_name} 不存在")
        
    rclpy.spin(robot)

def main():
    """
    behavior_name: random_pattern(), attraction_pattern(), 
                   dispersion_pattern(), obstacle_avoidance_pattern(),
                   formation_pattern(), combination_pattern().
    """

    os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://1234/epuck_0"
    robot1_behavior = 'random_pattern'
    p1 = multiprocessing.Process(target=main_robot, args=(robot1_behavior,))
    p1.start()

    time.sleep(0.5)

    os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://1234/epuck_1"
    robot2_behavior = 'attraction_pattern'
    p2 = multiprocessing.Process(target=main_robot, args=(robot2_behavior,))
    p2.start()

    time.sleep(0.5)

    os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://1234/epuck_2"
    robot3_behavior = 'dispersion_pattern'
    p3 = multiprocessing.Process(target=main_robot, args=(robot3_behavior,))
    p3.start()


if __name__ == '__main__':
    main()
