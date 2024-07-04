import rclpy
from rclpy.node import Node
from controller import Robot
from geometry_msgs.msg import Twist
import random


OUT_OF_RANGE = 0.0
INFRARED_MAX_RANGE = 0.04
INFRARED_MIN_RANGE = 0.009
TOF_MAX_RANGE = 1.0
DEFAULT_WHEEL_RADIUS = 0.02
DEFAULT_WHEEL_DISTANCE = 0.05685
NB_INFRARED_SENSORS = 8
SENSOR_DIST_FROM_CENTER = 0.035

class RandomPattern(Node):
    def __init__(self):
        super().__init__('random_pattern')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('random_walk_linear', 0.0),
                ('random_walk_angular', 0.0),
                ('random_walk_timer_period', 0.0),
                ('random_walk_rot_interval', 0.0),
                ('random_walk_lin_interval_min', 0.0),
                ('random_walk_lin_interval_max', 0.0)
            ])

        self.param_x = float(self.get_parameter("random_walk_linear").get_parameter_value().double_value)
        self.param_z = float(self.get_parameter("random_walk_angular").get_parameter_value().double_value)
        self.rot_interval = float(self.get_parameter("random_walk_rot_interval").get_parameter_value().double_value)
        self.lin_interval_min = float(self.get_parameter("random_walk_lin_interval_min")
                                      .get_parameter_value().double_value)
        self.lin_interval_max = float(self.get_parameter("random_walk_lin_interval_max")
                                      .get_parameter_value().double_value)
        self.turn = False if random.random() < 0.5 else True
        self.current_msg = Twist()
  

        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.start_device()

        self.timer = self.create_timer(
            self.get_parameter("random_walk_timer_period").get_parameter_value().double_value,
            self.random_callback)

        self.__timer = self.create_timer(0.001 * self.timestep, self.__timer_callback)
        self.walk = self.create_timer(3, self.random_speed)


        self.cmd_publisher = self.create_publisher(Twist, '/' + self.robot.getName() + '/cmd_vel',10)
        self.cmd_subscription = self.create_subscription(Twist, '/' + self.robot.getName() + '/cmd_vel', self.cmd_vel_callback, 10)


    def step(self):
        self.robot.step(self.timestep)

    def __timer_callback(self):
        self.step()

    def random_callback(self):
        self.cmd_publisher.publish(self.current_msg)
        right_velocity = self.current_msg.linear.x + DEFAULT_WHEEL_RADIUS * self.current_msg.angular.z / 2
        left_velocity = self.current_msg.linear.x - DEFAULT_WHEEL_RADIUS * self.current_msg.angular.z / 2
        left_omega = left_velocity / (DEFAULT_WHEEL_DISTANCE)
        right_omega = right_velocity / (DEFAULT_WHEEL_DISTANCE)
        self.left_motor.setVelocity(left_omega)
        self.right_motor.setVelocity(right_omega)


    def random_speed(self):
        msg = Twist()
        if self.turn:
            sign = 1 if random.random() < 0.5 else -1
            msg.angular.z = random.uniform(self.param_z  ,  2 * self.param_z ) * sign  
            msg.linear.x = 0.0
            self.walk.cancel()
            self.walk = self.create_timer(random.uniform(0, self.rot_interval), self.random_speed)
        else:
            msg.angular.z = 0.0
            msg.linear.x = self.param_x
            self.walk.cancel()
            bu = random.uniform(self.lin_interval_min, self.lin_interval_max)
            self.walk = self.create_timer(bu, self.random_speed)
        self.turn = not self.turn
        self.current_msg = msg
 

    def cmd_vel_callback(self, msg):
        right_velocity = msg.linear.x + DEFAULT_WHEEL_RADIUS * msg.angular.z / 2
        left_velocity = msg.linear.x - DEFAULT_WHEEL_RADIUS * msg.angular.z / 2
        left_omega = left_velocity / (DEFAULT_WHEEL_DISTANCE)
        right_omega = right_velocity / (DEFAULT_WHEEL_DISTANCE)
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
    epuck_controller = RandomPattern()
    rclpy.spin(epuck_controller)

    epuck_controller.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()