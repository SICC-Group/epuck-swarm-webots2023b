from config.behavior_config import get_config 
from swarm_interfaces.msg import TimesCount
from swarm_interfaces.msg import OpinionMessage

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from collections import Counter
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import random
import re


OUT_OF_RANGE = 0.0
INFRARED_MAX_RANGE = 0.04
INFRARED_MIN_RANGE = 0.009
TOF_MAX_RANGE = 1.0
DEFAULT_WHEEL_RADIUS = 0.02
DEFAULT_WHEEL_DISTANCE = 0.05685
NB_INFRARED_SENSORS = 8
SENSOR_DIST_FROM_CENTER = 0.035

class ShareBehavior(object):

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


    def formation_pattern(self, list1, list2):
        self.formation_id = list1
        self.formation_name = list2

        self.num_robot = len(self.formation_id)

        # 读取 VRML_SIM 文件内容
        with open('/home/jzx/ros_webots/src/swarm_behavior/worlds/leader_followers.wbt', 'r') as file:
            data = file.read()

        self.robot_positions = {}
        # 查找所有 e-puck 机器人的位置信息
        matches = re.finditer(r'E-puck {([^}]*)}', data)

        for match in matches:
            robot_data = match.group(1)
            name_match = re.search(r'name "(epuck_\d+)"', robot_data)
            if name_match:
                robot_name = name_match.group(1)
                if robot_name in self.formation_name:  # 仅处理 formation_id 中的机器人
                    position_match = re.search(r'translation ([-0-9.]+) ([-0-9.]+) ([-0-9.]+)', robot_data)
                    if position_match:
                        x, y, z = position_match.groups()
                        self.robot_positions[robot_name] = {"x": x, "y": y, "z": z}

        self.first_broadcast_flag = False

        self.start_rotate_flag = True
        self.start_go_flag = False
        self.start_flag = True

        self.my_sequence = TimesCount()
        self.my_sequence.times = 0

        self.target_points_list = []
        
        self.opinion_message = OpinionMessage()
        self.opinion_message.id = self.robot.id
        self.opinion_message.opinion = int(np.random.choice(self.formation_id))

        ps = self.robot.robot.getSelf().getPosition()
        x = ps[0]
        y = ps[1]
        pose_msg = Pose()
        pose_msg.x = x
        pose_msg.y = y
        self.robot.position_publisher.publish(pose_msg)

        self.formation_timer = self.robot.create_timer(1, self.formation)  

    def formation(self):
        if len(self.robot.opinion_list) == self.num_robot and self.first_broadcast_flag:

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

        # self.opinion_message.opinion = self.robot.opinion
        self.robot.broadcast_publisher.publish(self.opinion_message)
        self.robot.opinion_publisher.publish(self.opinion_message)

        self.first_broadcast_flag = True

    def followers_ps_callback(self, pose):
        tar_position = Pose()
        tar_position = pose

        self.calculate_polygon_vertices(1, self.num_robot, tar_position.x, tar_position.y)
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
            if len(self.target_points_list) >= self.num_robot-1:
                self.target_points_list = []
            angle_deg = (i+1) * angle_increment
            angle_rad = math.radians(angle_deg)
            x = fix_x - radius + radius * math.cos(angle_rad)
            y = fix_y + radius * math.sin(angle_rad)
            self.target_points_list.append((x, y))

    def cal_target_point(self, id):
        num_followers = self.num_robot - 1  # 计算跟随者的数量
        distances = np.zeros((num_followers, len(self.target_points_list)))
        
        follower_ids = [int(robot_id[-1]) for robot_id in self.formation_name if int(robot_id[-1]) != self.robot.leader_id]  # 获取跟随者的ID列表

        for i, robot_id in enumerate(follower_ids):
            pos = self.robot_positions['epuck_' + str(robot_id)]
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
  

