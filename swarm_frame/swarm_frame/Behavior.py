from config.behavior_config import get_config
from swarm_frame.utils.scan_calculation_functions import ScanCalculationFunctions
from swarm_frame.utils.interpolation import interpolate_lookup_table
from swarm_interfaces.msg import TimesCount
from swarm_interfaces.msg import OpinionMessage

from geometry_msgs.msg import Twist
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from turtlesim.msg import Pose
from collections import Counter
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import random


OUT_OF_RANGE = 0.0
INFRARED_MAX_RANGE = 0.04
INFRARED_MIN_RANGE = 0.009
TOF_MAX_RANGE = 1.0
DEFAULT_WHEEL_RADIUS = 0.02
DEFAULT_WHEEL_DISTANCE = 0.05685
NB_INFRARED_SENSORS = 8
SENSOR_DIST_FROM_CENTER = 0.035

class Behavior(object):

    def __init__(self,robot):

        args = get_config().parse_known_args()[0]

        # Global_Params
        self.max_translational_velocity = args.max_translational_velocity
        self.max_rotational_velocity = args.max_rotational_velocity
        self.linear_if_alone = args.direction_linear_if_alone
        self.angular_if_alone = args.direction_angular_if_alone

        # Random_Pattern
        self.param_x = args.random_walk_linear
        self.param_z = args.random_walk_angular
        self.rot_interval = args.random_walk_rot_interval
        self.lin_interval_min = args.random_walk_lin_interval_min
        self.lin_interval_max = args.random_walk_lin_interval_max

        # Attraction_Pattern
        self.attraction_max_range = args.attraction_max_range
        self.attraction_min_range = args.attraction_min_range
        self.attraction_front_attraction = args.attraction_front_attraction
        self.attraction_threshold = args.attraction_threshold

        # Dispersion_Pattern
        self.dispersion_max_range = args.dispersion_max_range
        self.dispersion_min_range = args.dispersion_min_range
        self.dispersion_front_attraction = args.dispersion_front_attraction
        self.dispersion_threshold = args.dispersion_threshold

        self.direction_if_alone = Twist()
        self.direction_if_alone.linear.x = self.linear_if_alone
        self.direction_if_alone.angular.z = self.angular_if_alone  

        # Robot_Params
        self.robot = robot


    def random_pattern(self):
        self.turn = False if random.random() < 0.25 else True
        self.random_timer = self.robot.create_timer(0.2, self.random_speed) 
        self.robot.timer_name = "random_pattern"
        self.robot.pattern_timers.append(self.random_timer)

    def random_speed(self):
        msg = Twist()
        if self.turn:
            sign = 1 if random.random() < 0.5 else -1
            msg.angular.z = random.uniform(self.param_z , 3 * self.param_z) * sign * 1.5
            msg.linear.x = 0.0
            self.random_timer.cancel()
            self.random_timer = self.robot.create_timer(random.uniform(0, self.rot_interval), self.random_speed)
        else:
            msg.angular.z = 0.0
            msg.linear.x = self.param_x * 3.0
            self.random_timer.cancel()
            bu = random.uniform(self.lin_interval_min, self.lin_interval_max)
            self.random_timer = self.robot.create_timer(bu, self.random_speed)
        self.turn = not self.turn
        
        self.robot.cmd_publisher.publish(msg)

    def attraction_pattern(self):
        self.attraction_timer = self.robot.create_timer(0.02, self.attraction_publish_laserscan_data)
        self.robot.timer_name = "attraction_pattern"
        self.robot.pattern_timers.append(self.attraction_timer)

    def attraction_publish_laserscan_data(self):
        stamp = Time(seconds=self.robot.robot.getTime()).to_msg()
        dists = [OUT_OF_RANGE] * NB_INFRARED_SENSORS

        for i, key in enumerate(self.robot.distance_sensors):
            dists[i] = interpolate_lookup_table(
                self.robot.distance_sensors[key].getValue(), self.robot.distance_sensors[key].getLookupTable()
            )
        if self.robot.tof_sensor:
            dist_tof = interpolate_lookup_table(self.robot.tof_sensor.getValue(), self.robot.tof_sensor.getLookupTable())

        msg = LaserScan()
        msg.header.frame_id = 'laser_scanner'
        msg.header.stamp = stamp
        msg.angle_min = - 150 * math.pi / 180
        msg.angle_max = 150 * math.pi / 180
        msg.angle_increment = 15 * math.pi / 180
        msg.range_min = self.attraction_min_range
        msg.range_max = self.attraction_max_range
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
        self.robot.scan_publisher.publish(msg)

        if msg is None:
            self.robot.cmd_publisher.publish(Twist())
        
        direction, alone = ScanCalculationFunctions.repulsion_field(
            self.attraction_front_attraction,
            self.attraction_max_range,
            self.max_rotational_velocity,
            self.max_translational_velocity,
            self.attraction_min_range,
            msg,
            self.attraction_threshold)
        
        direction = self.direction_if_alone if alone else direction
        self.robot.cmd_publisher.publish(direction)

    def dispersion_pattern(self):
        self.dispersion_timer = self.robot.create_timer(0.02, self.dispersion_publish_laserscan_data)
        self.robot.timer_name = "dispersion_pattern"
        self.robot.pattern_timers.append(self.dispersion_timer)

    def dispersion_publish_laserscan_data(self):
        stamp = Time(seconds=self.robot.robot.getTime()).to_msg()
        dists = [OUT_OF_RANGE] * NB_INFRARED_SENSORS

        for i, key in enumerate(self.robot.distance_sensors):
            dists[i] = interpolate_lookup_table(
                self.robot.distance_sensors[key].getValue(), self.robot.distance_sensors[key].getLookupTable()
            )
        if self.robot.tof_sensor:
            dist_tof = interpolate_lookup_table(self.robot.tof_sensor.getValue(), self.robot.tof_sensor.getLookupTable())

        msg = LaserScan()
        msg.header.frame_id = 'laser_scanner'
        msg.header.stamp = stamp
        msg.angle_min = - 150 * math.pi / 180
        msg.angle_max = 150 * math.pi / 180
        msg.angle_increment = 15 * math.pi / 180
        msg.range_min = self.dispersion_min_range
        msg.range_max = self.dispersion_max_range
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
        self.robot.scan_publisher.publish(msg)

        if msg is None:
            self.robot.cmd_publisher.publish(Twist())
        
        direction, alone = ScanCalculationFunctions.potential_field(
            self.dispersion_front_attraction,
            self.dispersion_max_range,
            self.max_rotational_velocity,
            self.max_translational_velocity,
            self.dispersion_min_range,
            msg,
            self.dispersion_threshold)
        
        direction = self.direction_if_alone if alone else direction
        self.robot.cmd_publisher.publish(direction)
   
    def obstacle_avoidance_pattern(self):
        self.avoid_timer = self.robot.create_timer(0.01, self.avoidance)
        self.robot.timer_name = "obstacle_avoidance_pattern"
        self.robot.pattern_timers.append(self.avoid_timer) 

    def avoidance(self):
        self.maxMotorVelocity = 4.2
        self.initialVelocity = 0.7 * self.maxMotorVelocity
        self.num_left_dist_sensors = 4
        self.num_right_dist_sensors = 4
        self.right_threshold = [75, 75, 75, 75]
        self.left_threshold = [75, 75, 75, 75]

        self.dist_left_sensors = [self.robot.robot.getDevice('ps' + str(x)) for x in range(self.num_left_dist_sensors)]  
        self.dist_right_sensors = [self.robot.robot.getDevice('ps' + str(x)) for x in range(self.num_right_dist_sensors,8)]  

        left_dist_sensor_values = [g.getValue() for g in self.dist_left_sensors]
        right_dist_sensor_values = [h.getValue() for h in self.dist_right_sensors]
        
        left_obstacle = [(x > y) for x, y in zip(left_dist_sensor_values, self.left_threshold)]
        right_obstacle = [(m > n) for m, n in zip(right_dist_sensor_values, self.right_threshold)]
    
        if True in left_obstacle:
            self.robot.left_motor.setVelocity(self.initialVelocity-(0.5*self.initialVelocity))
            self.robot.right_motor.setVelocity(self.initialVelocity+(0.5*self.initialVelocity))
        
        elif True in right_obstacle:
            self.robot.left_motor.setVelocity(self.initialVelocity+(0.5*self.initialVelocity))
            self.robot.right_motor.setVelocity(self.initialVelocity-(0.5*self.initialVelocity))

    def formation_pattern(self):
        self.first_broadcast_flag = False

        self.start_rotate_flag = True
        self.start_go_flag = False
        self.start_flag = True

        self.my_sequence = TimesCount()
        self.my_sequence.times = 0

        self.target_points_list = []
        
        self.opinion_message = OpinionMessage()
        self.opinion_message.id = self.robot.id
        self.opinion_message.opinion = self.robot.opinion

        self.formation_timer = self.robot.create_timer(1, self.formation)  
        self.robot.timer_name = "formation_pattern"
        self.robot.pattern_timers.append(self.formation_timer) 

    def formation(self):
        if len(self.robot.opinion_list) == self.robot.robot_num and self.first_broadcast_flag:

            opinions = [e.opinion for e in self.robot.opinion_list]
            distribution = Counter(opinions).most_common()
            maximum = distribution[0][1]
            maxima = []
            for e in distribution:
                if e[1] == maximum:
                    maxima.append(e[0])
                else:
                    break

            self.robot.leader_id = maxima[0]
            self.robot.get_logger().info('leader_id "{}"'.format(self.robot.leader_id))
            
            self.formation_timer.cancel()

            if self.robot.id != self.robot.leader_id and self.robot.leader_id != -1 :
                self.followers_ps_sub = self.robot.create_subscription(Pose, '/epuck_' + str(self.robot.leader_id) + '/position', self.followers_ps_callback, 1)

        self.opinion_message.opinion = self.robot.opinion
        self.robot.broadcast_publisher.publish(self.opinion_message)
        self.robot.opinion_publisher.publish(self.opinion_message)

        self.first_broadcast_flag = True

    def followers_ps_callback(self, pose):
        tar_position = Pose()
        tar_position = pose

        self.calculate_polygon_vertices(0.3535, self.robot.robot_num, tar_position.x, tar_position.y)
        self.cal_target_point(self.robot.id)


        self.leader_theta = tar_position.theta

        self.dist = math.sqrt((self.target_x - self.robot.position_x)**2 + (self.target_y - self.robot.position_y)**2)
        self.target_theta = math.atan2(self.target_y - self.robot.position_y, self.target_x - self.robot.position_x)


        if self.start_flag:
            if self.start_rotate_flag and not self.start_go_flag:
                self.rotate(self.robot.angle,self.target_theta)


            if self.start_go_flag and not self.start_rotate_flag:
                if self.rotate2(self.robot.angle,self.target_theta):
                    self.go()

            if not self.start_rotate_flag and not self.start_go_flag:
                self.rotate(self.robot.angle,self.leader_theta)
                if abs(self.robot.angle-self.leader_theta)<0.02:
                    self.start_flag = False
                    self.robot.sequence_publisher.publish(self.my_sequence)

    def rotate(self,curtheta,tartheta):  
        cmd = Twist()
        if tartheta>math.pi:
            tartheta-=math.pi*2
        if abs(tartheta-curtheta)<0.02:
            cmd.angular.z=0.0
            cmd.linear.x =0.0
            self.robot.cmd_publisher.publish(cmd)   
            self.start_go_flag = True
            self.start_rotate_flag = False
            return True
        else:
            cmd.angular.z=3.0 if abs(tartheta-curtheta)>0.3 else 1.0 
            cmd.linear.x =0.0
            self.robot.cmd_publisher.publish(cmd)
            return False
       
    def rotate2(self,curtheta,tartheta):
        cmd = Twist()
        if tartheta>math.pi:
            tartheta-=math.pi*2
        if abs(tartheta-curtheta)<0.18:
            return True
        else:
            cmd.angular.z=2.0 if abs(tartheta-curtheta)>0.2 else 0.75    
            cmd.linear.x =0.0
            self.robot.cmd_publisher.publish(cmd)
            return False
        
    def go(self):
        cmd = Twist()
        if self.dist>=0.01:
            cmd.angular.z=0.0
            cmd.linear.x =0.3  
            self.robot.cmd_publisher.publish(cmd)
        else:
            self.start_go_flag = False
            cmd.angular.z=0.0
            cmd.linear.x =0.0  
            self.robot.cmd_publisher.publish(cmd)

    def calculate_polygon_vertices(self, base_length, num_sides, fix_x, fix_y ):
        angle_increment = 360 / num_sides

        angle_radians = math.radians(angle_increment / 2)

        sin_half_theta = math.sin(angle_radians)

        radius = base_length / (2 * sin_half_theta)

        for i in range(num_sides-1):
            if len(self.target_points_list) >= self.robot.robot_num-1:
                self.target_points_list = []
            angle_deg = (i+1) * angle_increment
            angle_rad = math.radians(angle_deg)
            x = fix_x - radius + radius * math.cos(angle_rad)
            y = fix_y + radius * math.sin(angle_rad)
            self.target_points_list.append((x, y))

    def cal_target_point(self, id):
        num_followers = self.robot.robot_num - 1  # 计算跟随者的数量
        distances = np.zeros((num_followers, len(self.target_points_list)))
        
        follower_ids = [int(robot_id[-1]) for robot_id in self.robot.robot_positions.keys() if int(robot_id[-1]) != self.robot.leader_id]  # 获取跟随者的ID列表

        for i, robot_id in enumerate(follower_ids):
            pos = self.robot.robot_positions['epuck_' + str(robot_id)]
            x1, y1 = float(pos['x']), float(pos['y'])
            
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

    def combination_pattern(self):
        self.turn = False if random.random() < 0.5 else True

        self.first_broadcast_flag = False

        self.start_rotate_flag = True
        self.start_go_flag = False
        self.start_flag = True

        self.target_points_list = []
        
        self.opinion_message = OpinionMessage()
        self.opinion_message.id = self.robot.id
        self.opinion_message.opinion = self.robot.opinion

        self.first_pub = 0
        self.my_sequence = TimesCount()
        self.my_sequence.times = 0
        self.first = 1
        self.dist = 100

        self.distance_matrix = np.full((self.robot.robot_num, self.robot.robot_num), np.inf)
        self.robot_positions = [None] * self.robot.robot_num 

        self.p0 = self.robot.create_subscription(Pose, '/epuck_0/position', self.ps0, 1)
        self.p1 = self.robot.create_subscription(Pose, '/epuck_1/position', self.ps1, 1)
        self.p2 = self.robot.create_subscription(Pose, '/epuck_2/position', self.ps2, 1)

        self.combination_timer = self.robot.create_timer(0.02, self.combination)
        self.robot.timer_name = "combination_pattern"
        self.robot.pattern_timers.append(self.combination_timer) 

    def combination(self):
        if self.robot.index < 3 and self.first == 1 :
            print('attraction')
            self.attraction_publish_laserscan_data()  
            condition_met = np.all(self.distance_matrix < 0.42)
            if condition_met :
                msg = Twist()
                self.robot.cmd_publisher.publish(msg)
                self.robot.sequence_publisher.publish(self.my_sequence)
                self.first = 2

        if self.robot.index >=3 and self.robot.index < 5 and self.first == 2:
            print('formation')
            self.formation_timer = self.robot.create_timer(1.0, self.formation)
            self.first = 3

        if self.robot.index >= 5  and self.robot.index < 8 and self.first == 3:
            print('dispersion')
            self.dispersion_publish_laserscan_data()
            indices = np.tril_indices(3, k=-1)
            elements = self.distance_matrix[indices]
            condition_met = np.all(elements > 0.75)
            if condition_met :
                msg = Twist()
                self.robot.cmd_publisher.publish(msg)
                self.robot.sequence_publisher.publish(self.my_sequence)
                self.first = 4

        if self.robot.index >= 8 and self.first == 4:
            print('random')
            if self.first_pub == 0:
                self.random_timer = self.robot.create_timer(3, self.random_speed)
                self.first_pub =1
            indices = np.tril_indices(3, k=-1)
            elements = self.distance_matrix[indices]
            condition_met = np.all(elements > 1)
            if condition_met:
                self.random_timer.cancel()
                msg = Twist()
                self.robot.cmd_publisher.publish(msg)
                self.first = 5

    def ps0(self,position):
        self.update_distance_matrix(position, 0)

    def ps1(self,position):
        self.update_distance_matrix(position, 1)

    def ps2(self,position):
        self.update_distance_matrix(position, 2)

    def update_distance_matrix(self, position, robot_id):
        self.robot_positions[robot_id] = position

        for i in range(self.robot.robot_num):
            if self.robot_positions[i] is not None:
                distance = self.calculate_distance(position, self.robot_positions[i])
                self.distance_matrix[robot_id, i] = distance
                self.distance_matrix[i, robot_id] = distance

    def calculate_distance(self, position1, position2):
        return round(math.sqrt((position1.x - position2.x)**2 + (position1.y - position2.y)**2), 3)
    
