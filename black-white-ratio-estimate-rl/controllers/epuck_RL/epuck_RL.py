#!/usr/bin/python-deepbots
import os, sys
import random
import json
import random, math
import socket
import traceback
from copy import deepcopy
sys.path.append("/usr/local/webots/lib/controller/python")
os.environ['WEBOTS_HOME'] = '/usr/local/webots'
from deepbots.robots.controllers.csv_robot import CSVRobot

from controller import LED, Supervisor
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, "..", "..")
sys.path.append(exp_path)
from config import parser
# from torch_models import *


class RobotController(CSVRobot):
    def __init__(self, args, save_path) -> None:
        super().__init__(timestep=args.time_step  // args.frequency_ratio)
        self.args = args
        self.save_path = save_path
        self.f_ratio = self.args.frequency_ratio
        self.max_speed = 6.28
        self.speeds = [self.max_speed, self.max_speed]
        self.robot_name = self.getName()[-1]  # only id
        self.time_step = self.args.time_step
        self.num_agents = self.args.num_agents

        self.ranger_robots = list(range(self.args.ranger_robots))
        # self.comm_ranges = [
        #     self.args.range1 if id in self.ranger_robots else self.args.range0
        #     for id in range(self.num_agents)]
        # self.comm_range = self.comm_ranges[self.robot_id]
        # vel_actions = [[0,-1.57],[0,-0.785],[0,0],[0,0.785],[0,1.57],[0.1,-1.57],[0.1,-0.785],[0.1,0],[0.1,0.785],[0.1,1.57],[0,0]]
        # self.vel_actions = [[0, 0], [0.128, 0], [-0.128, 0], [0.1, 1.57], [0.1, -1.57]]  # stop, forward, backward, turn_left, turn_right
        
        # # forward, little left, little right, big left, big right, backward
        # self.vel_actions = [[0.128, 0], [0.1, 0.8], [0.1, -0.8], [0.05, 1.57], [0.05, -1.57], [-0.128, 0]]
        
        # forward, turn_left, turn_right, backward
        self.wheel_speeds_ratio = [[1, 1], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]

        # # accerleration config - forward, turn_left, turn_right, backward, keep
        # ratio = self.time_step / 1000
        # self.acc = [[1, 1], [-1, 1], [1, -1], [-1, -1], [0, 0]]
        # self.acc = (np.array(self.acc) * ratio).tolist()
        
        # self.acc = [[0.2, 0.2], [-0.2, 0.2], [0.2, -0.2], [-0.2, -0.2], [0, 0]]

        
        # states
        self.image_array = None  # 2-D rgb list
        self.ps_sensor_value = []  # 8 values
        self.position = []  # x, y
        self.rotation = []  # x, y, z, w
        self.gs_values = []

        # self.byz_robots = random.sample(
        #     list(range(self.num_agents)), self.args.byz_num)
        # self.byz_robots = [0, 1]
        # self.byz_style = self.args.byz_style

        # self.group_number = self.args.group_number
        # self.groups = {i: [] for i in range(self.group_number)}
        # for i in range(self.num_agents):
        #     self.groups[i % self.group_number].append(i)
        # self.group_id = self.robot_id % self.group_number
        
        # print info
        # print("========== robot info ==========")
        # print(f"ranger_robots: {self.ranger_robots}")
        # print(f"byzantine_robots: {self.byz_robots}")
        
        self.time_step = self.args.time_step  # in ms
        # self.steps = 0

        self.dist_threshold = 80
        self.neighbors = [0] * self.num_agents
        self.neighbors_dist = [0] * self.num_agents

        self.init_device_variables()

    def init_device_variables(self):
        # 8 E-puck distance sensors are named ranging from "ps0" to "ps7"
        self.ps_sensor = []
        for i in range(8):
            ps = self.getDevice(f"ps{i}")
            ps.enable(self.time_step // self.f_ratio)
            self.ps_sensor.append(ps)
        # use 4 LEDs named "led1", "led3", "led5" and "led7"
        self.leds = []
        for i in range(1,8,2):
            self.leds.append(LED(f"led{i}"))
        
        self.wheels = []
        for wheel_name in ['left wheel motor', 'right wheel motor']:
            wheel = self.getDevice(wheel_name)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

        self.receiver_rab = self.getDevice('receiver_rab')
        self.receiver_rab.enable(self.time_step // self.f_ratio)
        
        # gs "gs0", "gs1", "gs2" - left, middle, right
        self.gs = []
        for i in range(3):
            gs_ = self.getDevice(f"gs{i}")
            gs_.enable(self.time_step // self.f_ratio)
            self.gs.append(gs_)
        
    def create_message(self):
        # raito and ps sensor values
        # send to the supervisor
        msg = ['a' + self.robot_name]
        self.gs_values = [gs.getValue() for gs in self.gs]
        msg.append(self.gs_values[1])
        self.ps_sensor_value = [ps.getValue() for ps in self.ps_sensor]
        msg.extend(self.ps_sensor_value)
        return msg
    
    def use_message_data(self, message):
        # action
        # if self.robot_name in [0, "0"]: print("update action")
        action = int(message[int(self.robot_name)])
        # if self.robot_name in [0, "0"]: print(message)
        # speed = [0, 0]
        # linear_vel = self.vel_actions[action][0]
        # angle_vel = self.vel_actions[action][1]
        # speed[0] = ((2 * linear_vel) - (angle_vel * 0.053)) / (2 * 0.0205)
        # speed[1] = ((2 * linear_vel) + (angle_vel * 0.053)) / (2 * 0.0205)
        # self.speeds[0] += self.acc[action][0] * self.max_speed
        # self.speeds[1] += self.acc[action][1] * self.max_speed
        # self.speeds = np.clip(self.speeds, -self.max_speed * 0.7, self.max_speed)
        self.speeds[0] = self.max_speed * self.wheel_speeds_ratio[action][0]
        self.speeds[1] = self.max_speed * self.wheel_speeds_ratio[action][1]
        # if int(self.robot_name) == 0:
        #     print(self.speeds)
        for i in range(2):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(self.speeds[i])

    def handle_receiver_rab(self):
        # print("supervisor",self.receiver.getQueueLength())
        message = 0
        if self.receiver_rab.getQueueLength() > 0:
            str_message = self.receiver_rab.getString()
            #message = np.array(self.receiver_rab.getFloats()).reshape(self.num_agents,-1)
            message = np.array(str_message.split(","),dtype=np.float32).reshape(self.num_agents,-1)
            #print(message)
            self.receiver_rab.nextPacket()
        return message

    # main loop
    def run(self):
        i = 0
        try:
            while self.step(self.time_step//self.f_ratio) != -1:
                self.handle_receiver()
                if (i + 1) % self.f_ratio == 0:
                    self.handle_emitter()
                    i = -1
                i += 1
        except Exception as e:
            print(traceback.format_exc())
        # total_time = self.steps * self.time_step / 1000
        # print("total time is ", total_time)


if __name__ == '__main__':
    save_path = os.path.join(current_dir, "..", "..", "results")
    args, unknown = parser.parse_known_args()
    my_controller = RobotController(args, save_path)
    my_controller.run()
