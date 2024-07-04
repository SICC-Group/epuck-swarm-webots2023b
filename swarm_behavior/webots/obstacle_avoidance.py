# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS2 example controller."""

from rclpy.node import Node
from controller import Robot
from controller import Supervisor

import rclpy
from geometry_msgs.msg import Twist


class Obstacleavoidance(Node):

    def __init__(self):

        super().__init__('obstacle')

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())


        self.current_msg = Twist()
        self.leftMotor = self.robot.getDevice('left wheel motor')
        self.rightMotor = self.robot.getDevice('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
       
        self.maxMotorVelocity = 4.2
        self.initialVelocity = 0.7 * self.maxMotorVelocity
        self.num_left_dist_sensors = 4
        self.num_right_dist_sensors = 4
        self.right_threshold = [75, 75, 75, 75]
        self.left_threshold = [75, 75, 75, 75]

        self.leftMotor.setVelocity(self.initialVelocity)
        self.rightMotor.setVelocity(self.initialVelocity)

        self.dist_left_sensors = [self.robot.getDevice('ps' + str(x)) for x in range(self.num_left_dist_sensors)]  # distance sensors
        # list(map((lambda s: s.enable(self.timestep)), self.dist_left_sensors))  # Enable all distance sensors

        self.dist_right_sensors = [self.robot.getDevice('ps' + str(x)) for x in range(self.num_right_dist_sensors,8)]  # distance sensors
        # list(map((lambda t: t.enable(self.timestep)), self.dist_right_sensors))  # Enable all distance sensors
        
        for i in range(8):
            sensor = self.robot.getDevice('ps{}'.format(i))
            sensor.enable(self.timestep)

        while self.robot.step(self.timestep) != -1:
            left_dist_sensor_values = [g.getValue() for g in self.dist_left_sensors]
            right_dist_sensor_values = [h.getValue() for h in self.dist_right_sensors]
            
            left_obstacle = [(x > y) for x, y in zip(left_dist_sensor_values, self.left_threshold)]
            right_obstacle = [(m > n) for m, n in zip(right_dist_sensor_values, self.right_threshold)]
        
            if True in left_obstacle:
                self.leftMotor.setVelocity(self.initialVelocity-(0.5*self.initialVelocity))
                self.rightMotor.setVelocity(self.initialVelocity+(0.5*self.initialVelocity))
            
            elif True in right_obstacle:
                self.leftMotor.setVelocity(self.initialVelocity+(0.5*self.initialVelocity))
                self.rightMotor.setVelocity(self.initialVelocity-(0.5*self.initialVelocity))

        self.__timer = self.create_timer(0.001 * self.timestep, self.__timer_callback)

    def step(self):
        self.robot.step(self.timestep)

    def __timer_callback(self):
        self.step()

def main(args=None):
    rclpy.init(args=args)

    exampleController = Obstacleavoidance()

    rclpy.spin(exampleController)
    rclpy.shutdown()


if __name__ == '__main__':
    main()