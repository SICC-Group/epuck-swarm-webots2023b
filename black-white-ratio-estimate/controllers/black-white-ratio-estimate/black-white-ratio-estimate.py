#!/usr/bin/python-webots
import os, sys
import random
import yaml
import random, math
import socket
import threading
import traceback
from copy import deepcopy

from controller import LED, Supervisor
import numpy as np

current_dir = os.path.dirname(__file__)
exp_path = os.path.join(current_dir, "..", "..")
# sys.path.append(exp_path)
# from torch_models import *


class RobotController:
    def __init__(self, config, save_path) -> None:
        self.save_path = save_path
        self.robot = Supervisor()
        self.robot_id = int(self.robot.getCustomData())
        self.num_robots = config["numRobots"]
        self.ranger_robots = list(range(config["commRanger"]))
        self.comm_ranges = [
            config["range1"] if id in self.ranger_robots else config["range0"]
            for id in range(self.num_robots)]
        self.comm_range = self.comm_ranges[self.robot_id]
        self.speed_max = 7.5 if self.robot_id in self.ranger_robots else 5
        # action: left and right motor speed
        self.actions = {
            "forward": [self.speed_max, self.speed_max],
            "backward": [-self.speed_max, -self.speed_max],
            "turn_left": [-self.speed_max, self.speed_max],
            "turn_right": [self.speed_max, -self.speed_max],
            "stop": [0, 0],
            # "random_move": [0, 0]
        }
        self.last_action = "forward"
        self.turn_time = 2
        # states
        self.image_array = None  # 2-D rgb list
        self.ps_sensor_value = []  # 8 values
        self.position = []  # x, y, z

        # self.byz_robots = random.sample(
        #     list(range(self.num_robots)), config["byzNum"])
        self.byz_robots = [0, 1]
        self.byz_style = config["byzStyle"]

        self.group_number = config["groupNumber"]
        self.groups = {i: [] for i in range(self.group_number)}
        for i in range(self.num_robots):
            self.groups[i % self.group_number].append(i)
        self.group_id = self.robot_id % self.group_number
        
        # print info
        print("========== robot info ==========")
        print(f"ranger_robots: {self.ranger_robots}")
        print(f"byzantine_robots: {self.byz_robots}")
        
        self.time_step = config["timeStep"]  # in ms
        self.init_time_variables()
        self.init_env_variables()
        self.init_device_variables()
        self.init_socket()

    def init_time_variables(self):
        self.step = 0  # simulation steps
        # self.time_step = int(robot.getBasicTimeStep())  #  32 ms
        self.time_factor = self.time_step / 100.0
        self.turn = 45 / self.time_factor  # don't konw why it is named
        self._lambda = 100 / self.time_factor
        self.sigma = 30 / self.time_factor
        self.termination_time_ticks = 4000
        # remaining_sync_time = math.ceil(20 / time_factor)
        self.remaining_sync_time = -1
        self.remaining_time = math.ceil(np.random.exponential(self._lambda))
        self.remaining_exploration_time = math.ceil(self.sigma)
        self.action_time = 0

    def init_env_variables(self):
        # variables related to environment
        self.black_count = 0
        self.white_count = 0
        self.vote = 0
        self.vote_round = 0
        self.opinion = ""
        self.quality = random.random()
        self.direction = 0
        self.votes = []
        self.phase = "explore"  # ["explore", "diffuse"]
        self.dist_threshold = 100
        self.thread_currently_running = False
        self.res = ""
        self.stop_loop = False
        self.vote_flag = False
        self.sync_flag = False
        self.consensus_reached = False
        self.neighbors = [0] * self.num_robots
        self.neighbors_dist = [0] * self.num_robots
        self.last_front_obstacle = False

    def init_device_variables(self):
        # 8 E-puck distance sensors are named ranging from "ps0" to "ps7"
        self.ps_sensor = []
        for i in range(8):
            ps = self.robot.getDevice(f"ps{i}")
            ps.enable(self.time_step)
            self.ps_sensor.append(ps)
        # camera
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.time_step)  
        # use 4 LEDs named "led1", "led3", "led5" and "led7"
        self.leds = []
        for i in range(1,8,2):
            self.leds.append(LED(f"led{i}"))
        # motor
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
    
    def init_socket(self):
        self.ip = "172.17.0.1"
        self.port = 9955 + self.robot_id
        try:
            socket.setdefaulttimeout(10)
            self.s = socket.socket()
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 96)
            self.s.connect((self.ip, self.port))
        except Exception as e:
            print("connection failed")
            print(e)
            supvisornode = self.robot.getFromDef(f"epuck{self.robot_id}")
            supvisornode.restartController()
        else:
            print("connection success")
        
    # state
    def get_states(self):
        """get image_array, ps_sensor_value, position, neighbors, neighbors_dist"""
        self.image_array = self.camera.getImageArray()
        
        self.ps_sensor_value = [ps.getValue() for ps in self.ps_sensor]
        
        pos_list = []
        for i in range(self.num_robots):
            epuck = self.robot.getFromDef(f"epuck{i}")
            pos = epuck.getPosition()
            if i == self.robot_id:
                self.pos = pos
            pos_list.append(pos[:2])
        
        for i, (x, y) in enumerate(pos_list):
            if i != self.robot_id:
                dist = (x - self.pos[0])**2 + (y - self.pos[1])**2
                self.neighbors_dist[i] = dist
                self.neighbors[i] = 1 if dist < max(
                    self.comm_range, self.comm_ranges[i]
                ) else 0
    
    # action
    def obstacle_avoidance(self):
        right_obstacle = any(
            value > self.dist_threshold for value in self.ps_sensor_value[:3])
        left_obstacle = any(
            value > self.dist_threshold for value in self.ps_sensor_value[-3:])
        front_obstacle = right_obstacle and left_obstacle
        
        if front_obstacle:
            action = "backward"
        elif right_obstacle:
            action = "turn_left"
        elif left_obstacle:
            action = "turn_right"
        elif self.last_front_obstacle:
            action = random.choice(["turn_left", "turn_right"])
        else:
            action = "forward"
        
        if self.last_action in ["turn_left", "turn_right"]:
            if self.turn_time > 0:
                action = self.last_action
                self.turn_time -= 1
            else:
                self.turn_time = random.randint(1, 3)
        
        # if action == "random_move":
        #     l = random.uniform(-1, 1)
        #     r = random.uniform(-1, 1)
        #     self.walk_with_speed([l * self.speed_max, r * self.speed_max])
        # else:
        #     self.walk_with_speed(self.actions[action])
        self.walk_with_speed(self.actions[action])
        self.last_front_obstacle = deepcopy(front_obstacle)
        self.last_action = deepcopy(action)

    def walk_with_speed(self, speed_list: list):
        assert len(speed_list) == 2, "speed_list should have 2 elements"
        l_speed, r_speed = speed_list
        self.left_motor.setVelocity(l_speed)
        self.right_motor.setVelocity(r_speed)

    # main loop
    def run(self):
        while self.robot.step(self.time_step) != -1:
            try:
                self.step += 1
                self.termination_time_ticks -= 1
                self.get_states()
                if self.step > self.sigma + 1:
                    self.connect_and_listen()  # get neighbors and sync
                
                self.turn_leds_on()

                self.walk_with_speed(self.actions["forward"])
                                       
                if self.phase == "explore":
                    self.detect_cell()
                    self.explore()  # explore and estimate the ratio
                elif self.phase == "diffuse":
                    self.diffusing() # vote
            
                self.obstacle_avoidance()

                if self.stop_loop:
                    self.walk_with_speed(self.actions["stop"])
                    break
            except Exception as e:
                print(traceback.print_exc())
                      
        self.s.close()
        total_seconds = self.time_step * self.step / 1000.0
        print("total time is " + str(total_seconds) + "seconds")
    
    def turn_leds_on(self):
        if self.opinion == "white":
            self.leds[1].set(0x00ff00)  # green for white
        elif self.opinion == "black":
            self.leds[1].set(0x0000ff)  # blue for black
        else:
            pass

    def explore(self):
        self.opinion = "black" if self.black_count > self.white_count else "white"
        if self.robot_id in self.byz_robots:
            if self.byz_style == 0:
                self.quality = 0
            elif self.byz_style == 1:
                self.quality = 1
            else:
                self.quality = random.random()
        else:
            self.quality = self.black_count / (self.black_count + self.white_count)
        
        self.vote = format(self.quality, '.6f')
        if self.remaining_exploration_time > 0:
            self.remaining_exploration_time -= 1
        else:
            self.vote_flag = True
            self.remaining_exploration_time = math.ceil(self.sigma)
            self.black_count, self.white_count = 0, 0
            self.phase = "diffuse"

    # image process
    def detect_cell(self):
        flag = self.image_process(self.image_array, "black")
        if flag:
            self.black_count += 1
        else:
            self.white_count += 1
    
    @classmethod
    def image_process(cls, img_array, target_color) -> bool:
        black_num = 0
        row = len(img_array)
        colum = len(img_array[0])
        half = row * colum / 2
        for i in range(row):
            for j in range(colum):
                if cls.detect_color(img_array[i][j]) == target_color:
                    black_num = black_num + 1
            if black_num > half:
                return True
        return False
    
    @staticmethod
    def detect_color(pixel):
        color_thresholds = {
            "black": ([0, 0, 0], [10, 20, 20]),
            "white": ([200, 200, 200], [255, 255, 255]),
            # "red": ([187, 40, 40], [207, 80, 80]),
            # "blue": ([0, 187, 187], [20, 227, 227])
        }
        for color_name, thresholds in color_thresholds.items():
            min_threshold, max_threshold = thresholds
            if all(min_t <= p <= max_t for p, min_t, max_t in zip(
                pixel, min_threshold, max_threshold
            )):
                return color_name
        return "undefined"

    # Diffusing
    def diffusing(self):
        self.phase = "explore"
        if not self.thread_currently_running:
            self.thread_currently_running = True
            try:
                thread = threading.Thread(
                    target=self.wait_for_decision,
                    args=(f"Thread-{self.robot_id}",))
                thread.start()
            except Exception as e:
                print("threading error")
                print(e)
    
    def wait_for_decision(self, thread_name):
        try:
            result = self.get_result()
            if result != "":
                if result != "end":
                    print(f"id={self.robot_id},C:Response from Server:{result}")
                    resultList = eval(result)
                    if resultList[0] == True and self.consensus_reached == False:
                        print("consensusReached is {} for robot {:2d}".format(
                            resultList[0], self.robot_id))
                        
                        with open(
                            os.path.join(self.save_path, "all_result.txt"), "a+"
                        ) as f:
                            f.write(str(resultList[1:]))
                            f.write('\n')
                        
                        with open(
                            os.path.join(self.save_path, "tmp_result.txt"), "a+"
                        ) as f:
                            f.write(f"epuck{self.robot_id}\n")
                        self.consensus_reached = True
                else:
                    print(f"id={self.robot_id},C:end...")
                    with open(
                        os.path.join(self.save_path, "tmp_result.txt"), "a+"
                    ) as f:
                        f.write(f"epuck{self.robot_id}\n")
                    self.stop_loop = True
        except Exception as e:
            print("error in wait for decision: ", e)
            traceback.print_exc()
        self.thread_currently_running = False

    def get_result(self):
        try:
            res = self.s.recv(1024,0x40).decode()
            res = res.split("#")[1].strip("~")
        except Exception as e:
            res = ""
        return res

    def connect_and_listen(self):
        self.remaining_sync_time -= 1
        if (self.vote_flag and not self.consensus_reached 
            and self.termination_time_ticks > 0):
            # vote
            msg = "#[{}, {:2d}, {:>18}, {:6d}]~".format(
                self.neighbors, self.robot_id, self.vote, self.vote_round)
            print("vote: {}, length = {}".format(msg, len(msg)))
            self.votes.append(self.vote)
            print("id = {}, votes_length = {}, votes[:-10] = {}".format(
                self.robot_id, len(self.votes), self.votes[-10:]))
            self.vote_round += 1
            self.vote_flag = False
        else:
            self.remaining_sync_time = -1
            if self.consensus_reached:
                # sync stop signal
                msg = "#[{}, {:2d}, {:6d}, {:6d}]~".format(
                    self.neighbors, self.robot_id, -2, self.vote_round
                )
                print("sync: " + msg)
            else:
                # sync
                msg = "#[{}, {:2d}, {:>18}, {:6d}]~".format(
                    self.neighbors, self.robot_id, self.vote, self.vote_round
                )
        try:
            msg = msg.encode()
            self.s.send(msg)
        except Exception as e:
            print("error in connect_and_listen: ", e)
            traceback.print_exc()


if __name__ == '__main__':
    save_path = os.path.join(current_dir, "..", "..", "results")
    with open(os.path.join(exp_path, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    my_controller = RobotController(config, save_path)
    my_controller.run()
