import rclpy
import os
import multiprocessing
import time

from swarm_frame.EpuckNode import EpuckNode

def main_robot():
    rclpy.init()
    print("环境变量：", os.environ["WEBOTS_CONTROLLER_URL"])

    robot = EpuckNode('robot')
    robot.start_device()
        
    rclpy.spin(robot)

def main():

    os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://1234/epuck_0"
    p1 = multiprocessing.Process(target=main_robot, args=())
    p1.start()

    time.sleep(0.5)

    os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://1234/epuck_1"
    p2 = multiprocessing.Process(target=main_robot, args=())
    p2.start()

    time.sleep(0.5)

    os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://1234/epuck_2"
    p3 = multiprocessing.Process(target=main_robot, args=())
    p3.start()


if __name__ == '__main__':
    main()
