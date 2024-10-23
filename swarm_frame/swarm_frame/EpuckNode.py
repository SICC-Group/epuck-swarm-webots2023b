from swarm_frame.Agent import Agent
from swarm_frame.Behavior import Behavior
from swarm_frame.utils.VoteList import VoteList
from swarm_frame.utils.PoseDict import PoseDict
from swarm_interfaces.msg import TimesCount
from swarm_interfaces.msg import OpinionMessage
from swarm_interfaces.msg import PoseMessage

from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import re


OUT_OF_RANGE = 0.0
INFRARED_MAX_RANGE = 0.04
INFRARED_MIN_RANGE = 0.009
TOF_MAX_RANGE = 1.0
DEFAULT_WHEEL_RADIUS = 0.02
DEFAULT_WHEEL_DISTANCE = 0.05685
NB_INFRARED_SENSORS = 8
SENSOR_DIST_FROM_CENTER = 0.035

class EpuckNode(Agent):
    class MyBehavior():
        def start():
            print('My behavior is startting.')


    def __init__(self, robot_node):
        super().__init__(robot_node)

        # 读取 VRML_SIM 文件内容
        with open('/home/jzx/swarm_frame/src/swarm_frame/worlds/leader_followers.wbt', 'r') as file:
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


        self.cmd_publisher = self.create_publisher(Twist, '/' + self.robot.getName() + '/cmd_vel',1)
        self.cmd_subscription = self.create_subscription(Twist, '/' + self.robot.getName() + '/cmd_vel', self.cmd_vel_callback, 1)

        self.scan_publisher = self.create_publisher(LaserScan, '/' + self.robot.getName() + '/scan', 1)
        #self.scan_subscription = self.create_subscription(LaserScan, '/' + self.robot.getName() + '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data)
        
        self.position_publisher = self.create_publisher(Pose, '/' + self.robot.getName() + '/position', 10)
        self.position_subscriber = self.create_subscription(Pose, '/' + self.robot.getName() + '/position', self.position_callback, 1)

        self.broadcast_publisher = self.create_publisher(OpinionMessage, '/majority_broadcast', 10)
        self.broadcast_subscription = self.create_subscription(OpinionMessage, '/majority_broadcast', self.majority_broadcast_callback, 10)

        self.opinion_publisher = self.create_publisher(OpinionMessage, '/' + self.robot.getName() +  '/opinion', 10)  

        self.sequence_publisher = self.create_publisher(TimesCount, '/sequence', 10)
        self.sequence_subscription = self.create_subscription(TimesCount, '/sequence', self.sequence_callback, 10)

        self.all_pose_publisher = self.create_publisher(PoseMessage, '/all_pose', 10)
        self.all_pose_subscription = self.create_subscription(PoseMessage, '/all_pose', self.all_pose_callback, 10)

        
        self.robot_num = 3   # formation_pattern_robots_num
        self.opinion_list = []
        self.pose_dict = {}
        
        self.id = int(self.robot.getName()[-1])
        self.leader_id = -1
        self.opinion = np.random.randint(0, self.robot_num)

        self.index = 0
        self.pattern_timers = []
        self.timer_name = None

        ps = self.robot.getSelf().getPosition()
        x = ps[0]
        y = ps[1]
        pose_msg = Pose()
        pose_msg.x = x
        pose_msg.y = y
        self.position_publisher.publish(pose_msg)

    def all_pose_callback(self, pose_msg):
        self.pose_dict = PoseDict.update_pose(self.pose_dict, pose_msg)

    def majority_broadcast_callback(self, opinion_msg):
        self.opinion_list = VoteList.update_opinion(self.opinion_list, opinion_msg, self.id)

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

        all_pose_msg = PoseMessage()
        all_pose_msg.name = f'/epuck_{self.id}'
        all_pose_msg.id = self.id
        all_pose_msg.x = self.position_x
        all_pose_msg.y = self.position_y
        all_pose_msg.theta = self.angle
        self.all_pose_publisher.publish(all_pose_msg)

    def cmd_vel_callback(self, msg):
        right_velocity = msg.linear.x + DEFAULT_WHEEL_RADIUS * msg.angular.z / 2
        left_velocity = msg.linear.x - DEFAULT_WHEEL_RADIUS * msg.angular.z / 2
        left_omega = left_velocity / (DEFAULT_WHEEL_DISTANCE)
        right_omega = right_velocity / (DEFAULT_WHEEL_DISTANCE)
        self.left_motor.setVelocity(left_omega)
        self.right_motor.setVelocity(right_omega)

    def sequence_callback(self, all_sequence):     
        self.index = self.index + 1

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




