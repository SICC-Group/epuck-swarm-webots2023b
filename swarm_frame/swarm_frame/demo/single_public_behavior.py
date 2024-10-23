import rclpy
from swarm_frame.EpuckNode import EpuckNode
from swarm_frame.Behavior import Behavior

def main():

    rclpy.init()                     # Initialize rclpy

    robot1 = EpuckNode('epuck')      # Initialize an EpuckNode node
    robot1.start_device()            # Launch robot devices in webots

    behavior = Behavior(robot1)      # Initialize a behavior instance
    behavior.random_pattern()        # Activate public robot behavior

    rclpy.spin(robot1)               # Continuously execute nodes
    robot1.destroy_node()            # Destroy node
    rclpy.shutdown()                 # Shutdown rclpy


if __name__ == '__main__':
    main()