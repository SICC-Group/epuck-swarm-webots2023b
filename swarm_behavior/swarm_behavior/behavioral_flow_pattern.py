import rclpy
from rclpy.time import Time
from .scan_calculation_functions import ScanCalculationFunctions
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from controller import Robot
from controller import Supervisor
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from .interpolation import interpolate_lookup_table
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from turtlesim.msg import Pose
from interfaces.msg import OpinionMessage
from collections import Counter
from .VoteList import VoteList
from interfaces.msg import TimesCount
import random


OUT_OF_RANGE = 0.0
INFRARED_MAX_RANGE = 0.04
INFRARED_MIN_RANGE = 0.009
TOF_MAX_RANGE = 1.0
DEFAULT_WHEEL_RADIUS = 0.02
DEFAULT_WHEEL_DISTANCE = 0.05685
NB_INFRARED_SENSORS = 8
SENSOR_DIST_FROM_CENTER = 0.035

class BehavioralFlowPattern(Node):
    def __init__(self):
        super().__init__('behavioral_flow_pattern')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('max_translational_velocity', None),
                ('max_rotational_velocity', None),
                ('direction_linear_if_alone', 0.0),
                ('direction_angular_if_alone', 0.0),
                ('attraction_max_range', None),
                ('attraction_min_range', None),
                ('attraction_front_attraction', None),
                ('attraction_threshold', None),
                ('dispersion_max_range', None),
                ('dispersion_min_range', None),
                ('dispersion_front_attraction', None),
                ('dispersion_threshold', None),
                ('dispersion_allow_dynamic_max_range_setting', False),
                ('random_walk_linear', 0.0),
                ('random_walk_angular', 0.0),
                ('random_walk_timer_period', 0.0),
                ('random_walk_rot_interval', 0.0),
                ('random_walk_lin_interval_min', 0.0),
                ('random_walk_lin_interval_max', 0.0)
            ])

        # global params
        self.param_max_translational_velocity = self.get_parameter(
            "max_translational_velocity").get_parameter_value().double_value
        self.param_max_rotational_velocity = self.get_parameter(
            "max_rotational_velocity").get_parameter_value().double_value
        self.param_linear_if_alone = self.get_parameter(
            "direction_linear_if_alone").get_parameter_value().double_value
        self.param_angular_if_alone = self.get_parameter(
            "direction_angular_if_alone").get_parameter_value().double_value

        # attraction_params
        self.attraction_param_max_range = float(
            self.get_parameter("attraction_max_range").get_parameter_value().double_value)
        self.attraction_param_min_range = self.get_parameter(
            "attraction_min_range").get_parameter_value().double_value
        self.attraction_param_front_attraction = self.get_parameter(
            "attraction_front_attraction").get_parameter_value().double_value
        self.attraction_param_threshold = self.get_parameter(
            "attraction_threshold").get_parameter_value().integer_value
        
        #dispersion_params
        self.dispersion_param_max_range = float(
            self.get_parameter("dispersion_max_range").get_parameter_value().double_value)
        self.dispersion_param_min_range = self.get_parameter(
            "dispersion_min_range").get_parameter_value().double_value
        self.dispersion_param_front_attraction = self.get_parameter(
            "dispersion_front_attraction").get_parameter_value().double_value
        self.dispersion_param_threshold = self.get_parameter(
            "dispersion_threshold").get_parameter_value().integer_value
        self.dispersion_param_allow_dynamic_max_range_setting = self.get_parameter(
            "dispersion_allow_dynamic_max_range_setting").get_parameter_value().bool_value
        
        #random_params
        self.param_x = float(self.get_parameter("random_walk_linear").get_parameter_value().double_value)
        self.param_z = float(self.get_parameter("random_walk_angular").get_parameter_value().double_value)
        self.rot_interval = float(self.get_parameter("random_walk_rot_interval").get_parameter_value().double_value)
        self.lin_interval_min = float(self.get_parameter("random_walk_lin_interval_min")
                                      .get_parameter_value().double_value)
        self.lin_interval_max = float(self.get_parameter("random_walk_lin_interval_max")
                                      .get_parameter_value().double_value)
        
        self.direction_if_alone = Twist()
        self.direction_if_alone.linear.x = self.param_linear_if_alone
        self.direction_if_alone.angular.z = self.param_angular_if_alone  

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())


        self.start_device()


        self.num_robot = 3

        self.id = int(self.robot.getName()[-1])
        self.opinion = np.random.randint(0, self.num_robot)
        
        # create reused OpinionMessage
        self.opinion_message = OpinionMessage()
        self.opinion_message.id = self.id
        self.opinion_message.opinion = self.opinion

        # list to store opinions
        self.opinion_list = []
        self.target_points_list = []

        self.__timer = self.create_timer(0.001 * self.timestep, self.__timer_callback)
        self.attraction_timer = self.create_timer(self.timestep / 1000, self.attraction_publish_laserscan_data)


        self.broadcast_publisher = self.create_publisher(OpinionMessage,
                                                         '/majority_broadcast',
                                                         10)

        self.opinion_publisher = self.create_publisher(OpinionMessage,
                                                       '/' + self.robot.getName() +  '/opinion',
                                                       10)

        self.broadcast_subscription = self.create_subscription(
            OpinionMessage,
            '/majority_broadcast',
            self.majority_broadcast_callback,
            10)
        
        self.scan_publisher = self.create_publisher(LaserScan, '/' + self.robot.getName() + '/scan', 1)
        self.scan_subscription = self.create_subscription(LaserScan, '/' + self.robot.getName() + '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data)
        

        self.cmd_publisher = self.create_publisher(Twist, '/' + self.robot.getName() + '/cmd_vel',10)
        self.cmd_subscription = self.create_subscription(Twist, '/' + self.robot.getName() + '/cmd_vel', self.cmd_vel_callback, 10)
        self.cmd = Twist()

        self.sequence_publisher = self.create_publisher(TimesCount, '/sequence', 10)
        self.sequence_subscription = self.create_subscription(TimesCount, '/sequence', self.sequence_callback, 10)


        self.position_publisher = self.create_publisher(Pose, '/' + self.robot.getName() + '/position', 10)
        self.position_subscriber = self.create_subscription(Pose, '/' + self.robot.getName() + '/position', self.position_callback, 1)
        ps = self.robot.getSelf().getPosition()
        x = ps[0]
        y = ps[1]
        pose_msg = Pose()
        pose_msg.x = x
        pose_msg.y = y
        self.position_publisher.publish(pose_msg)

        self.start_rotate_flag = True
        self.start_go_flag = False
        self.start_flag = True
        self.first_cal = 1
        self.first_broadcast_flag = False
        self.leader_id = -1
        self.first_pub = 0
        self.turn = False if random.random() < 0.5 else True

        self.my_sequence = TimesCount()
        self.my_sequence.times = 0
        self.i = 0
        self.first = 1
        self.dist = 100

        self.distance_matrix = np.full((self.num_robot, self.num_robot), np.inf)
        self.robot_positions = [None] * self.num_robot  


        self.p0 = self.create_subscription(Pose, '/epuck_0/position', self.ps0, 1)
        self.p1 = self.create_subscription(Pose, '/epuck_1/position', self.ps1, 1)
        self.p2 = self.create_subscription(Pose, '/epuck_2/position', self.ps2, 1)


    def step(self):
        self.robot.step(self.timestep)

    def __timer_callback(self):
        self.step()

    def ps0(self,position):
        self.update_distance_matrix(position, 0)

    def ps1(self,position):
        self.update_distance_matrix(position, 1)

    def ps2(self,position):
        self.update_distance_matrix(position, 2)

    def update_distance_matrix(self, position, robot_id):
        self.robot_positions[robot_id] = position

        for i in range(self.num_robot):
            if self.robot_positions[i] is not None:
                distance = self.calculate_distance(position, self.robot_positions[i])
                self.distance_matrix[robot_id, i] = distance
                self.distance_matrix[i, robot_id] = distance

    def calculate_distance(self, position1, position2):
        return round(math.sqrt((position1.x - position2.x)**2 + (position1.y - position2.y)**2), 3)

    def sequence_callback(self, all_sequence):     
        self.i = self.i + 1

    def position_callback(self, pose):
        ps = self.robot.getSelf().getPosition()
        self.position_x = ps[0]
        self.position_y = ps[1]

        q = self.robot.getSelf().getProtoField('rotation').getSFRotation()
        self.angle = q[3]

        pose_msg = Pose()
        pose_msg.x = self.position_x
        pose_msg.y = self.position_y
        pose_msg.theta = self.angle
        self.position_publisher.publish(pose_msg)

    def cmd_vel_callback(self,twist):
        right_velocity = twist.linear.x + DEFAULT_WHEEL_DISTANCE * twist.angular.z / 2
        left_velocity = twist.linear.x - DEFAULT_WHEEL_DISTANCE * twist.angular.z / 2
        left_omega = left_velocity / (DEFAULT_WHEEL_RADIUS)
        right_omega = right_velocity / (DEFAULT_WHEEL_RADIUS)
        self.left_motor.setVelocity(left_omega)
        self.right_motor.setVelocity(right_omega)

    def followers_ps_callback(self,pose):
        tar_position = Pose()
        tar_position = pose

        self.calculate_polygon_vertices(0.3535, 3, tar_position.x, tar_position.y)
        self.cal_target_point(self.id)
        
        self.tar_theta = tar_position.theta


        self.dist = math.sqrt((self.target_x - self.position_x)**2 + (self.target_y - self.position_y)**2)
        self.target_theta = math.atan2(self.target_y - self.position_y, self.target_x - self.position_x)

        if self.start_flag:
            if self.start_rotate_flag and not self.start_go_flag:
                self.rotate(self.angle,self.target_theta)

            if self.start_go_flag and not self.start_rotate_flag:
                if self.rotate2(self.angle,self.target_theta):
                    self.go()

            if not self.start_rotate_flag and not self.start_go_flag:
                self.rotate(self.angle,self.tar_theta)
                if abs(self.angle-self.tar_theta)<0.012:
                    self.start_flag = False
                    self.sequence_publisher.publish(self.my_sequence)
                   
    def rotate(self,curtheta,tartheta):  
        cmd = Twist()
        if tartheta>math.pi:
            tartheta-=math.pi*2
        if abs(tartheta-curtheta)<0.012:
            cmd.angular.z=0.0
            cmd.linear.x =0.0
            self.cmd_publisher.publish(cmd)   
            self.start_go_flag = True
            self.start_rotate_flag = False
            return True
        else:
            cmd.angular.z=1.0 if abs(tartheta-curtheta)>0.3 else 0.1    
            cmd.linear.x =0.0
            self.cmd_publisher.publish(cmd)
            return False
        
    def rotate2(self,curtheta,tartheta):
        cmd = Twist()
        if tartheta>math.pi:
            tartheta-=math.pi*2
        if abs(tartheta-curtheta)<0.05:
            return True
        else:
            cmd.angular.z=1.0 if abs(tartheta-curtheta)>0.2 else 0.1    
            cmd.linear.x =0.0
            self.cmd_publisher.publish(cmd)
            return False
        
    def go(self):
        cmd = Twist()
        if self.dist>=0.01:
            cmd.angular.z=0.0
            cmd.linear.x =0.25  
            self.cmd_publisher.publish(cmd)
        else:
            self.start_go_flag = False
            cmd.angular.z=0.0
            cmd.linear.x =0.0  
            self.cmd_publisher.publish(cmd)

    def calculate_polygon_vertices(self, base_length, num_sides, fix_x, fix_y ):
        angle_increment = 360 / num_sides
        angle_radians = math.radians(angle_increment / 2)
        sin_half_theta = math.sin(angle_radians)
        radius = base_length / (2 * sin_half_theta)

        for i in range(num_sides-1):
            if len(self.target_points_list) >= 2:
                self.target_points_list = []
            angle_deg = (i+1) * angle_increment
            angle_rad = math.radians(angle_deg)
            x = fix_x - radius + radius * math.cos(angle_rad)
            y = fix_y + radius * math.sin(angle_rad)
            self.target_points_list.append((x, y))
        # self.get_logger().info(f'{self.target_points_list}')

    def cal_target_point(self, id):
        num_followers = self.num_robot - 1  # 计算跟随者的数量
        distances = np.zeros((num_followers, len(self.target_points_list)))
        
        follower_ids = []  

        for i in range(self.num_robot):
            if i != self.leader_id:
                follower_ids.append(i)

        for i, robot_id in enumerate(follower_ids):
            pos = self.robot_positions[robot_id]
            x1, y1 = float(pos.x), float(pos.y)
            
            for j, target_point in enumerate(self.target_points_list):
                x2, y2 = target_point
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances[i, j] = distance

        # 选择目标点
        row_ind, col_ind = linear_sum_assignment(distances)

        # 输出分配结果
        for robot_id, target_id in zip(row_ind, col_ind):
            selected_follower_id = follower_ids[robot_id]  # 获取选中的跟随者的ID
            if id == selected_follower_id:
                target_tuple = self.target_points_list[target_id]
                self.target_x = target_tuple[0]
                self.target_y = target_tuple[1]
                
    def get_leader_callback(self):
        if len(self.opinion_list) == 3 and self.first_broadcast_flag:
            opinions = [e.opinion for e in self.opinion_list]
            distribution = Counter(opinions).most_common()
            maximum = distribution[0][1]
            maxima = []
            for e in distribution:
                if e[1] == maximum:
                    maxima.append(e[0])
                else:
                    break
            self.leader_id = maxima[0]
            self.get_logger().info('leader_id "{}"'
                                    .format(self.leader_id))          
            self.get_leader_timer.cancel()

            if self.id != self.leader_id:
                self.followers_ps_sub = self.create_subscription(Pose, '/epuck_' + str(self.leader_id) + '/position', self.followers_ps_callback, 1)

        self.opinion_message.opinion = self.opinion
        self.broadcast_publisher.publish(self.opinion_message)
        self.opinion_publisher.publish(self.opinion_message)

        self.first_broadcast_flag = True

    def majority_broadcast_callback(self, opinion_msg):   
        self.opinion_list = VoteList.update_opinion(self.opinion_list, opinion_msg, self.id)

    def start_device(self):
        self.left_motor = self.robot.getMotor('left wheel motor')
        self.right_motor = self.robot.getMotor('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        self.distance_sensors = {}
        for i in range(NB_INFRARED_SENSORS):
            sensor = self.robot.getDistanceSensor('ps{}'.format(i))
            sensor.enable(self.timestep)
            self.distance_sensors['ps{}'.format(i)] = sensor

        self.tof_sensor = self.robot.getDistanceSensor('tof')
        self.tof_sensor.enable(self.timestep)           

    def scan_callback(self, incoming_msg):
        if self.i < 3 and self.first == 1 :
            self.get_logger().info('attraction')
            direction = self.attraction_vector_calc(incoming_msg)
            self.cmd_publisher.publish(direction)
            condition_met = np.all(self.distance_matrix < 0.42)
            if condition_met :
                msg = Twist()
                self.cmd_publisher.publish(msg)
                self.sequence_publisher.publish(self.my_sequence)
                self.first = 2

        if self.i >=3 and self.i < 5 and self.first == 2:
            self.get_logger().info('formation')
            self.get_leader_timer = self.create_timer(3.0, self.get_leader_callback)
            self.first = 3

        if self.i >= 5  and self.i < 8 and self.first == 3:
            self.get_logger().info('dispersion')
            direction = self.dispersion_vector_calc(incoming_msg)
            self.cmd_publisher.publish(direction)
            indices = np.tril_indices(3, k=-1)
            elements = self.distance_matrix[indices]
            condition_met = np.all(elements > 0.75)
            if condition_met :
                msg = Twist()
                self.cmd_publisher.publish(msg)
                self.sequence_publisher.publish(self.my_sequence)
                self.first = 4

        if self.i >= 8 and self.first == 4:
            self.get_logger().info('random')
            if self.first_pub == 0:
                self.random_timer = self.create_timer(0.2, self.random_speed)
                self.first_pub =1
            indices = np.tril_indices(3, k=-1)
            elements = self.distance_matrix[indices]
            condition_met = np.all(elements > 1)
            if condition_met:
                self.random_timer.cancel()
                msg = Twist()
                self.cmd_publisher.publish(msg)
                self.first = 5

    
    def attraction_vector_calc(self, current_scan):
        if current_scan is None:
            return Twist()
        
        direction, alone = ScanCalculationFunctions.repulsion_field(
            self.attraction_param_front_attraction,
            self.attraction_param_max_range,
            self.param_max_rotational_velocity,
            self.param_max_translational_velocity,
            self.attraction_param_min_range,
            current_scan,
            self.attraction_param_threshold)
        
        direction = self.direction_if_alone if alone else direction  
        return direction

    def dispersion_vector_calc(self, current_scan):
        if current_scan is None:
            return Twist()
        
        direction, alone = ScanCalculationFunctions.potential_field(
            self.dispersion_param_front_attraction,
            self.dispersion_param_max_range,
            self.param_max_rotational_velocity,
            self.param_max_translational_velocity,
            self.dispersion_param_min_range,
            current_scan,
            self.dispersion_param_threshold)
        
        direction = self.direction_if_alone if alone else direction
        return direction
    
    def attraction_publish_laserscan_data(self):
        stamp = Time(seconds=self.robot.getTime()).to_msg()
        dists = [OUT_OF_RANGE] * NB_INFRARED_SENSORS

        for i, key in enumerate(self.distance_sensors):
            dists[i] = interpolate_lookup_table(
                self.distance_sensors[key].getValue(), self.distance_sensors[key].getLookupTable()
            )

        if self.tof_sensor:
            dist_tof = interpolate_lookup_table(self.tof_sensor.getValue(), self.tof_sensor.getLookupTable())

        msg = LaserScan()
        msg.header.frame_id = 'laser_scanner'
        msg.header.stamp = stamp
        msg.angle_min = - 150 * math.pi / 180
        msg.angle_max = 150 * math.pi / 180
        msg.angle_increment = 15 * math.pi / 180
        msg.range_min = self.attraction_param_min_range
        msg.range_max = self.attraction_param_max_range
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

    def dispersion_publish_laserscan_data(self):
        stamp = Time(seconds=self.robot.getTime()).to_msg()
        dists = [OUT_OF_RANGE] * NB_INFRARED_SENSORS

        for i, key in enumerate(self.distance_sensors):
            dists[i] = interpolate_lookup_table(
                self.distance_sensors[key].getValue(), self.distance_sensors[key].getLookupTable()
            )
  
        if self.tof_sensor:
            dist_tof = interpolate_lookup_table(self.tof_sensor.getValue(), self.tof_sensor.getLookupTable())

        msg = LaserScan()
        msg.header.frame_id = 'laser_scanner'
        msg.header.stamp = stamp
        msg.angle_min = - 150 * math.pi / 180
        msg.angle_max = 150 * math.pi / 180
        msg.angle_increment = 15 * math.pi / 180
        msg.range_min = self.dispersion_param_min_range
        msg.range_max = self.dispersion_param_max_range
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

    def random_speed(self):
        msg = Twist()
        if self.turn:
            sign = 1 if random.random() < 0.5 else -1
            msg.angular.z = random.uniform(self.param_z  ,  2 * self.param_z ) * sign  
            msg.linear.x = 0.0
            self.random_timer.cancel()
            self.random_timer = self.create_timer(random.uniform(0, self.rot_interval), self.random_speed)
        else:
            msg.angular.z = 0.0
            msg.linear.x = self.param_x
            self.random_timer.cancel()
            bu = random.uniform(self.lin_interval_min, self.lin_interval_max)
            self.random_timer = self.create_timer(bu, self.random_speed)
        self.turn = not self.turn
        self.cmd_publisher.publish(msg)
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
    epuck_controller = BehavioralFlowPattern()
    rclpy.spin(epuck_controller)

    epuck_controller.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()