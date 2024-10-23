import rclpy
from swarm_frame.EpuckNode import EpuckNode

def main():

    rclpy.init()                     # Initialize rclpy

    robot1 = EpuckNode('epuck')      # Initialize an EpuckNode node
    robot1.start_device()            # Launch robot devices in webots

    robot1.MyBehavior.start()        # Activate private robot behavior

    rclpy.spin(robot1)               # Continuously execute nodes
    robot1.destroy_node()            # Destroy node
    rclpy.shutdown()                 # Shutdown rclpy


if __name__ == '__main__':
    main()