from math import pi
import rclpy
from rclpy.node import Node
from controller import Robot
from controller import Supervisor
from geometry_msgs.msg import Twist
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

class FormationPattern(Node):
    def __init__(self):
        super().__init__('formation_pattern')

        # 读取 VRML_SIM 文件内容
        with open('/home/jzx/ros_webots/src/webots/worlds/mul_epuck_world.wbt', 'r') as file:
            data = file.read()

        self.robot_positions = {}
        # 查找所有 e-puck 机器人的位置信息
        matches = re.finditer(r'E-puck {([^}]*)}', data)

        for match in matches:
            robot_data = match.group(1)
            name_match = re.search(r'name "(epuck_\d+)"', robot_data)
            if name_match:
                robot_name = name_match.group(1)
                position_match = re.search(r'translation ([-0-9.]+) ([-0-9.]+) ([-0-9.]+)', robot_data)
                if position_match:
                    x, y, z = position_match.groups()
                    self.robot_positions[robot_name] = {"x": x, "y": y, "z": z}

  

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.start_device()


        self.num_robot = 8

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
        self.timer = self.create_timer(1.0, self.timer_callback)


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
        
        self.first_broadcast_flag = False

        self.leader_id = -1

        self.cmd_vel_publisher = self.create_publisher(Twist, '/' + self.robot.getName() + '/cmd_vel', 10)
        self.cmd_vel_subscription = self.create_subscription(Twist, '/' + self.robot.getName() + '/cmd_vel', self.cmd_vel_callback, 1)
        self.cmd = Twist()


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


    def step(self):
        self.robot.step(self.timestep)

    def __timer_callback(self):
        self.step()

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

        self.calculate_polygon_vertices(0.3535, self.num_robot, tar_position.x, tar_position.y)
        self.cal_target_point(self.id)


        self.leader_theta = tar_position.theta

        self.dist = math.sqrt((self.target_x - self.position_x)**2 + (self.target_y - self.position_y)**2)
        self.target_theta = math.atan2(self.target_y - self.position_y, self.target_x - self.position_x)


        if self.start_flag:
            if self.start_rotate_flag and not self.start_go_flag:
                self.rotate(self.angle,self.target_theta)


            if self.start_go_flag and not self.start_rotate_flag:
                if self.rotate2(self.angle,self.target_theta):
                    self.go()

            if not self.start_rotate_flag and not self.start_go_flag:
                self.rotate(self.angle,self.leader_theta)
                if abs(self.angle-self.leader_theta)<0.012:
                    self.start_flag = False
       

    def rotate2(self,curtheta,tartheta):
        cmd = Twist()
        if tartheta>math.pi:
            tartheta-=math.pi*2
        if abs(tartheta-curtheta)<0.18:
            return True
        else:
            cmd.angular.z=1.0 if abs(tartheta-curtheta)>0.2 else 0.1    
            cmd.linear.x =0.0
            self.cmd_vel_publisher.publish(cmd)
            return False
        
    def rotate(self,curtheta,tartheta):  
        if tartheta>math.pi:
            tartheta-=math.pi*2
        if abs(tartheta-curtheta)<0.012:
            self.cmd.angular.z=0.0
            self.cmd.linear.x =0.0
            self.cmd_vel_publisher.publish(self.cmd)   
            self.start_go_flag = True
            self.start_rotate_flag = False
            return True
        else:
            self.cmd.angular.z=0.5 if abs(tartheta-curtheta)>0.3 else 0.1    #0.2  0.1
            self.cmd.linear.x =0.0
            self.cmd_vel_publisher.publish(self.cmd)
            return False
        
    def go(self):
        if self.dist>=0.005:
            self.cmd.angular.z=0.0
            self.cmd.linear.x =0.25   #0.1
            self.cmd_vel_publisher.publish(self.cmd)
        else:
            self.start_go_flag = False
            self.cmd.angular.z=0.0
            self.cmd.linear.x =0.0  
            self.cmd_vel_publisher.publish(self.cmd)

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
        
        follower_ids = [int(robot_id[-1]) for robot_id in self.robot_positions.keys() if int(robot_id[-1]) != self.leader_id]  # 获取跟随者的ID列表

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
  
    def timer_callback(self):

        # update opinion if at least one opinion were received and initial opinion send once
        if len(self.opinion_list) == self.num_robot and self.first_broadcast_flag:
            self.get_logger().debug('Turtle "{}" reduce opinions "{}" at time "{}"'
                                    .format(self.id, self.opinion_list, datetime.datetime.now()))

            opinions = [e.opinion for e in self.opinion_list]
            # find max opinion
            distribution = Counter(opinions).most_common()
            # check the maximum is reached by more than one opinion
            maximum = distribution[0][1]
            maxima = []
            for e in distribution:
                if e[1] == maximum:
                    maxima.append(e[0])
                else:
                    # the input is ordered so no need for further search
                    break
            # choose randomly one of the maxima
            
            self.leader_id = maxima[0]

            self.get_logger().info('leader_id "{}"'
                                    .format(self.leader_id))
            
            self.timer.cancel()

            if self.id != self.leader_id and self.leader_id != -1 :
                self.followers_ps_sub = self.create_subscription(Pose, '/epuck_' + str(self.leader_id) + '/position', self.followers_ps_callback, 1)


        # emit opinion
        self.opinion_message.opinion = self.opinion
        self.broadcast_publisher.publish(self.opinion_message)
        self.opinion_publisher.publish(self.opinion_message)

        self.get_logger().debug('Robot "{}" send opinion "{}" at time "{}"'
                                .format(self.id, self.opinion, datetime.datetime.now()))
        self.first_broadcast_flag = True

    def majority_broadcast_callback(self, opinion_msg):
       
        self.opinion_list = VoteList.update_opinion(self.opinion_list, opinion_msg, self.id)

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
    epuck_controller = FormationPattern()
    rclpy.spin(epuck_controller)

    epuck_controller.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()