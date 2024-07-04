from math import pi
import rclpy
from rclpy.time import Time
from .scan_calculation_functions import ScanCalculationFunctions
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from controller import Robot
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from webots_ros2_core.math.interpolation import interpolate_lookup_table

OUT_OF_RANGE = 0.0
INFRARED_MAX_RANGE = 0.04
INFRARED_MIN_RANGE = 0.009
TOF_MAX_RANGE = 1.0
DEFAULT_WHEEL_RADIUS = 0.02
DEFAULT_WHEEL_DISTANCE = 0.05685
NB_INFRARED_SENSORS = 8
SENSOR_DIST_FROM_CENTER = 0.035

class AttractionPattern(Node):
    def __init__(self):
        super().__init__('attraction_pattern')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('attraction_max_range', None),
                ('attraction_min_range', None),
                ('attraction_front_attraction', None),
                ('attraction_threshold', None),
               #('attraction_stop_if_alone', None),
                ('attraction_linear_if_alone', 0.0),
                ('attraction_angular_if_alone', 0.0),
                ('max_translational_velocity', None),
                ('max_rotational_velocity', None),
            ])

        self.param_max_range = float(
            self.get_parameter("attraction_max_range").get_parameter_value().double_value)
        self.param_min_range = self.get_parameter(
            "attraction_min_range").get_parameter_value().double_value
        self.param_front_attraction = self.get_parameter(
            "attraction_front_attraction").get_parameter_value().double_value
        self.param_threshold = self.get_parameter(
            "attraction_threshold").get_parameter_value().integer_value
        #self.param_stop_if_alone = self.get_parameter(
            #"attraction_stop_if_alone").get_parameter_value().bool_value
        self.param_max_translational_velocity = self.get_parameter(
            "max_translational_velocity").get_parameter_value().double_value
        self.param_max_rotational_velocity = self.get_parameter(
            "max_rotational_velocity").get_parameter_value().double_value
        self.param_linear_if_alone = self.get_parameter(
            "attraction_linear_if_alone").get_parameter_value().double_value
        self.param_angular_if_alone = self.get_parameter(
            "attraction_angular_if_alone").get_parameter_value().double_value
        
        self.direction_if_alone = Twist()
        self.direction_if_alone.linear.x = self.param_linear_if_alone
        self.direction_if_alone.angular.z = self.param_angular_if_alone

        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.start_device()

        self.scan_publisher = self.create_publisher(LaserScan, '/' + self.robot.getName() + '/scan', 1)
        self.scan_subscription = self.create_subscription(LaserScan, '/' + self.robot.getName() + '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data)

        self.__timer = self.create_timer(0.001 * self.timestep, self.__timer_callback)
        self.create_timer(self.timestep / 1000, self.__publish_laserscan_data)

        self.cmd_publisher = self.create_publisher(Twist, '/' + self.robot.getName() + '/cmd_vel',10)
        self.cmd_subscription = self.create_subscription(Twist, '/' + self.robot.getName() + '/cmd_vel', self.cmd_vel_callback, 10)


    def step(self):
        self.robot.step(self.timestep)

    def __timer_callback(self):
        self.step()

    def scan_callback(self, incoming_msg):
        """Call back if a new scan msg is available."""
        direction = self.vector_calc(incoming_msg)
        self.cmd_publisher.publish(direction)

    def vector_calc(self, current_scan):
        """Calculate the direction vector for the current scan."""
        if current_scan is None:
            return Twist()

        direction, alone = ScanCalculationFunctions.repulsion_field(
            self.param_front_attraction,
            self.param_max_range,
            self.param_max_rotational_velocity,
            self.param_max_translational_velocity,
            self.param_min_range,
            current_scan,
            self.param_threshold)
        
        direction = self.direction_if_alone if alone else direction
            
        return direction


    def __publish_laserscan_data(self):
        stamp = Time(seconds=self.robot.getTime()).to_msg()
        dists = [OUT_OF_RANGE] * NB_INFRARED_SENSORS

        # Calculate distances
        for i, key in enumerate(self.distance_sensors):
            dists[i] = interpolate_lookup_table(
                self.distance_sensors[key].getValue(), self.distance_sensors[key].getLookupTable()
            )

        # Publish range: ToF
        if self.tof_sensor:
            dist_tof = interpolate_lookup_table(self.tof_sensor.getValue(), self.tof_sensor.getLookupTable())

        msg = LaserScan()
        msg.header.frame_id = 'laser_scanner'
        msg.header.stamp = stamp
        msg.angle_min = - 150 * pi / 180
        msg.angle_max = 150 * pi / 180
        msg.angle_increment = 15 * pi / 180
        msg.range_min = self.param_min_range
        msg.range_max = self.param_max_range
        msg.ranges = [
            0.0 ,                               # -150
            0.0 ,                               # -135
            0.0 ,                               # -120
            0.0 ,                               # -105
            0.0 ,                               # -90
            0.0 ,                               # -75
            0.0 ,                               # -60
            0.0 ,                               # -45
            0.0 ,                               # -30
            0.0 ,                               # -15
            dist_tof,                           # 0
            0.0 ,                               # 15
            0.0 ,                               # 30
            0.0 ,                               # 45
            0.0 ,                               # 60
            0.0 ,                               # 75
            0.0 ,                               # 90
            0.0 ,                               # 105
            0.0 ,                               # 120
            0.0 ,                               # 135
            0.0 ,                               # 150
        ]
        self.scan_publisher.publish(msg)

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
    epuck_controller = AttractionPattern()
    rclpy.spin(epuck_controller)

    epuck_controller.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()