from math import pi
import rclpy
from rclpy.time import Time
from .scan_calculation_functions import ScanCalculationFunctions
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from controller import Robot
from controller import Supervisor
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from webots_ros2_core.math.interpolation import interpolate_lookup_table
import math
import rclpy
import re
from geometry_msgs.msg import Twist
import numpy as np
from scipy.optimize import linear_sum_assignment
from turtlesim.msg import Pose
from interfaces.msg import OpinionMessage
import datetime
from collections import Counter
from webots.VoteList import VoteList

OUT_OF_RANGE = 0.0
INFRARED_MAX_RANGE = 0.04
INFRARED_MIN_RANGE = 0.009
TOF_MAX_RANGE = 1.0
DEFAULT_WHEEL_RADIUS = 0.02
DEFAULT_WHEEL_DISTANCE = 0.05685
NB_INFRARED_SENSORS = 8
SENSOR_DIST_FROM_CENTER = 0.035



class LineQueue(Node):

    def __init__(self, args):
        super().__init__(
            'combined_pattern',
        )

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())     

        self.start_device()


        #self.get_logger().info('{}'.format(dir(self.robot.getSelf())))
        self.get_logger().info('{}'.format(self.robot.getSelf().getProtoField('rotation').getSFRotation()))

        self.__timer = self.create_timer(0.001 * self.timestep, self.__timer_callback)

        self.cmd_vel_publisher = self.create_publisher(Twist, '/' + self.robot.getName() + '/cmd_vel', 10)
        self.cmd_vel_subscription = self.create_subscription(Twist, '/' + self.robot.getName() + '/cmd_vel', self.cmd_vel_callback, 1)

    def step(self): 
        self.robot.step(self.timestep)

    def __timer_callback(self):
        self.step()


    
    def cmd_vel_callback(self,twist):
        right_velocity = twist.linear.x + DEFAULT_WHEEL_DISTANCE * twist.angular.z / 2
        left_velocity = twist.linear.x - DEFAULT_WHEEL_DISTANCE * twist.angular.z / 2
        left_omega = left_velocity / (DEFAULT_WHEEL_RADIUS)
        right_omega = right_velocity / (DEFAULT_WHEEL_RADIUS)
        self.left_motor.setVelocity(left_omega)
        self.right_motor.setVelocity(right_omega)

    def start_device(self):
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        self.distance_sensors = {}
        for i in range(NB_INFRARED_SENSORS):
            sensor = self.robot.getDevice('ps{}'.format(i))
            sensor.enable(self.timestep)
            self.distance_sensors['ps{}'.format(i)] = sensor

        self.tof_sensor = self.robot.getDevice('tof')
        self.tof_sensor.enable(self.timestep)



def main(args=None):
    rclpy.init(args=args)
    exampleController = LineQueue(args=args)
    rclpy.spin(exampleController)
    rclpy.shutdown()


if __name__ == '__main__':
    main()