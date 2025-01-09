#!/usr/bin/python-deepbots
import os, sys
import random
import math
import random
from itertools import combinations
import time
from datetime import datetime

from deepbots.supervisor.controllers.csv_supervisor_env import CSVSupervisorEnv
# from gym.spaces import Discrete
sys.path.append("/usr/local/webots/lib/controller/python")
os.environ['WEBOTS_HOME'] = '/usr/local/webots'
from controller import Supervisor, Robot
# from scipy.spatial.transform import Rotation as R
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, "..", "..")
sys.path.append(exp_path)
from config import parser

class LocalMap:
    def __init__(self, col, row, id, byz_robots, byz_style, threshold=0.04):
        self.col = col
        self.row = row
        self.id = id
        self.byz_robots = byz_robots
        self.byz_style = byz_style
        self.threshold = threshold
        self.black_count = 0
        self.white_count = 0
        self.offset_x = -0.1 * self.row / 2 - 0.05
        self.offset_y = -0.1 * self.col / 2 - 0.05
        
        self.tiles = [(self.offset_x + 0.1 * (j + 1), self.offset_y + 0.1 * (i + 1))
                      for i in range(self.col) for j in range(self.row)]
        self.tiles_visited = [False for _ in range(self.col * self.row)]

    def check(self, x, y, angle):
        # calculate the position of gs
        x = x + 0.03 * math.cos(angle)
        y = y + 0.03 * math.sin(angle)
        for idx, (tile_x, tile_y) in enumerate(self.tiles):
            if ((tile_x - self.threshold < x < tile_x + self.threshold) and 
                (tile_y - self.threshold < y < tile_y + self.threshold)):
                if not self.tiles_visited[idx]:
                    self.tiles_visited[idx] = True
                    return True, idx
        
        return False, None

    def get_ratio(self):
        try:
            if self.id in self.byz_robots and "ratio" in self.byz_style:
                style = self.byz_style.split("-")[1]
                if style.isdigit():
                    r = float(style)
                else:
                    r = random.uniform(0,1)
                return r
            else:
                return self.black_count / (self.black_count + self.white_count)
        except ZeroDivisionError:
            # print(f"agent {self.id} has no visited tiles - zero division error")
            return -1


class Epuck2Supervisor(CSVSupervisorEnv):
    def __init__(self, args, save_path, col, row) -> None:
        super().__init__(timestep=args.time_step // args.frequency_ratio)
        self.args = args
        self.save_path = save_path
        self.col, self.row = col, row
        self.start_time = 0
        self.num_agents = self.args.num_agents
        self.ps_threshold = self.args.ps_threshold
        self.ranger_robots = list(range(self.args.ranger_robots))
        self.comm_ranges = [
            self.args.range1 if id in self.ranger_robots else self.args.range0
            for id in range(self.num_agents)]
        self.byz_robots = [] if self.args.byz_num == 0 else list(range(self.num_agents))[-self.args.byz_num:]
        self.byz_style = self.args.byz_style
        
        self.group_number = self.args.group_number
        self.groups = {i: [] for i in range(self.group_number)}
        for i in range(self.num_agents):
            self.groups[i % self.group_number].append(i)
        self.swarm = []
        for g in self.groups.values():
            self.swarm.extend(g)
        self.init_combinations()
        self.all_states = None
        self.global_ratio_list = []  # list of tuples (step, ratio)
        self.global_ratio_estimation = 0
        self.is_update_global_ratio = False
        self.update_count = 0
        self.global_reward_total = 0
        self.global_reward_last = 0
        self.aggregation_count = 0
        self.local_maps = [
            LocalMap(self.col, self.row, i, self.byz_robots, self.byz_style, self.args.center_threshold)
            for i in range(self.num_agents)]
        self.exploration_ratios = None  # exploration ratio for all agents
        self.local_ratios = None  # local ratio estimation for all 
        self.local_ratio_dict = {i: [] for i in range(self.num_agents)}
        self.max_ratio_diff = max(abs(self.args.black_ratio - 0), abs(1 - self.args.black_ratio))
        
        # print info
        # print("time_step: ", self.time_step)
        print("========== supervisor info ==========")
        print(f"byzantine_robots: {self.byz_robots}")
        print(f"update method: {self.args.ratio_update_method}")
        print(f"groups: {self.groups}")
        print(f"swarm: {self.swarm}")
        self.start_flag = True
        self.time_step = self.args.time_step  # in ms
        self.f_ratio = self.args.frequency_ratio

        self.init_env(self.args.black_ratio)
        self.init_robots()

        self.robots = [self.getFromDef(f"epuck{i}") 
                       for i in range(self.num_agents)]
        self.emitter_rab = [self.getDevice(f"emitter0{i}") 
                            for i in range(self.num_agents)]
        # self.ps_sensor_mm = {'min': 0, 'max': 1023}
        # self.angle_mm = {'min': -np.pi, 'max': np.pi}
        # self.dis_mm = {'min': 0, 'max': 2.82}
        # self.pos_x_mm = {'min': -1, 'max': 1}
        # self.pos_y_mm = {'min': -1, 'max': 1}
        # self.m_cMuValues = [(0, 7.646890), (2, 7.596525), (5, 7.249550),
        #                     (10, 7.084636), (15, 6.984497), (30, 6.917447),
        #                     (45, 6.823188), (60, 6.828551), (80, 6.828551)]

        # self.m_cSigmaValues = [(0, 0.3570609), (2, 0.3192310), (5, 0.1926492),
        #                        (10, 0.1529397), (15, 0.1092330), (30, 0.1216533),
        #                        (45, 0.1531546), (60, 0.1418425), (80, 0.1418425)]

        # self.m_fExpA = 9.06422181283387
        # self.m_fExpB = -0.00565074879677167
        # self.radius_epuck = 0.035
        
        # RL info
        self.steps = 0
        self.reward_time = self.args.reward_time
        self.reward_exploration = self.args.reward_exploration
        self.reward_exp_ratio = self.args.reward_exploration_ratio
        self.reward_repeat = self.args.reward_repeat
        self.collision_distance = self.args.collision_distance
        self.reward_collision = self.args.reward_collision
        self.reward_local_ratio = self.args.reward_local_ratio
        self.reward_global_ratio = self.args.reward_global_ratio
        self.num_actions = 4
        self.num_states = 1 + 2 + 1 + 8 # ratio + 2 position + angle + 8 ps sensor values
    
    def init_env(self, black_ratio):
        """ In the two-dimensional space by deault
                    x
                    ^
                    |
            y       |
            <-------+ 
        - the x-coordinate is the same for each element in a row
        - the y-coordinate is the same for each element in a column
        the index of tiles is starting from the down-right corner, e.g.
        ..  ..  ..      ..     ..
        ..  ..  ..      ..      3
        ..  ..  ..      ..      2
        ..  ..  ..      1*row+1 1
        ..  ..  2*row+0 1*row+0 0"""
        blacktiles = random.sample(
            range(self.row * self.col), int(self.row * self.col * black_ratio))
        offset_x = -0.1 * self.row / 2 - 0.05
        offset_y = -0.1 * self.col / 2 - 0.05
        self.x_max, self.x_min, self.y_max, self.y_min= (
            offset_x + 0.1 * (self.row - 1), offset_x + 0.1 * 2,
            offset_y + 0.1 * (self.col - 1), offset_y + 0.1 * 2
        )
        print(f"x_max: {self.x_max}, x_min: {self.x_min}, y_max: {self.y_max}, y_min: {self.y_min}")
        tiles_list = [
            """
            Solid {
                translation %f %f 0.001
                rotation 1 0 0 1.5707963267948966
                children [
                    Shape {
                        appearance PBRAppearance {
                            baseColor %f %f %f
                            emissiveColor %f %f %f
                        }
                        geometry Box {
                            size 0.1 0.001 0.1
                        }
                    }
                ]
            }
            """ % (
                offset_x + 0.1 * (j + 1), offset_y + 0.1 * (i + 1),
                0, 0, 0, 0, 0, 0
            ) if self.row * i + j in blacktiles else
            """
            Solid {
                translation %f %f 0.001
                rotation 1 0 0 1.5707963267948966
                children [
                    Shape {
                        appearance PBRAppearance {
                            baseColor %f %f %f
                            emissiveColor %f %f %f
                        }
                        geometry Box {
                            size 0.1 0.001 0.1
                        }
                    }
                ]
            }
            """ % (
                offset_x + 0.1 * (j + 1), offset_y + 0.1 * (i + 1),
                1, 1, 1, 1, 1, 1
            ) for i in range(self.col) for j in range(self.row)
        ]
        children_string = "".join(tiles_list)
        line_string = """
            DEF Floor_Tiles Solid {
                children [
                    %s
                ]
            }
            """ % children_string
        root = self.getRoot()
        chFd = root.getField("children")
        chFd.importMFNodeFromString(-1, line_string)
        print("========== import tiles successfully ==========")

    def init_robots(self):
        offset_x = -0.1 * self.row / 2 - 0.05
        offset_y = -0.1 * self.col / 2 - 0.05
        x_y = [(offset_x + 0.1 * (j + 1), offset_y + 0.1 * (i + 1))
                for i in range(self.col) for j in range(self.row)]
        starts = random.sample(x_y, self.num_agents)
        # starts = [(i * 0.05, 0) for i in range(6)]
        z_ = [0] * self.num_agents
        rotation_init = [i / 100.0 for i in range(
            0, 628, int(628 / self.num_agents))]
        random.shuffle(rotation_init)
        rgblist = ['{:08b}'.format(i + 1) 
                   if i in self.ranger_robots else None for i in self.swarm]
        for i in range(self.num_agents):
            print(f"import robot {i:2d}")
            if rgblist[i] is None:
                line_string = """
                    DEF epuck%d E-puck {
                        translation %f %f %f
                        rotation 0 0 1 %f
                        name "e-puck%d"
                        controller "epuck_RL"
                        customData "%d"
                        supervisor TRUE
                        version "2"
                        emitter_channel 1
                        receiver_channel 2
                        receiver_rab_channel %d
                        camera_fieldOfView 0.5
                        camera_width 48
                        camera_height 48
                        camera_antiAliasing TRUE
                        camera_rotation 0 1 0 1.57
                        groundSensorsSlot [
                            E-puckGroundSensors {
                            }
                        ]
                    }""" % (i, starts[i][0], starts[i][1], z_[i], 
                            rotation_init[i], i, i, i + 3)
            else:
                line_string = """
                    DEF epuck%d E-puck{
                        translation %f %f %f
                        rotation 0 0 1 %f
                        name "e-puck%d"
                        controller "epuck_RL"
                        customData "%d"
                        supervisor TRUE
                        version "2"
                        emitter_channel 1
                        receiver_channel 2
                        receiver_rab_channel %d
                        camera_fieldOfView 0.5
                        camera_width 48
                        camera_height 48
                        camera_antiAliasing TRUE
                        camera_rotation 0 1 0 1.57
                        groundSensorsSlot [
                            E-puckGroundSensors {
                            }
                        ]
                        emissiveColor %s %s %s
                        emissiveColor2 %s %s %s
                    }""" % (i, starts[i][0], starts[i][1], z_[i], 
                            rotation_init[i], i, i, i + 3,
                            rgblist[-1], rgblist[-2], rgblist[-3],
                            rgblist[-4], rgblist[-5], rgblist[-6])
            
            root = self.getRoot()
            chFd = root.getField("children")
            chFd.importMFNodeFromString(-1,line_string)
        
        # start the controller
        super(Supervisor, self).step(self.time_step // self.f_ratio)
        print("========== import robots successfully ==========")

    def init_combinations(self):
        self.combinations = {}
        if self.num_agents > 2:
            for i in range(self.num_agents):
                remaining_agents = [x for x in range(self.num_agents) if x != i]
                comb_num_agents_minus_1 = list(combinations(remaining_agents, self.num_agents - 1))
                comb_num_agents_minus_2 = list(combinations(remaining_agents, self.num_agents - 2))
                self.combinations[i] = [list(c) for c in (comb_num_agents_minus_1 + comb_num_agents_minus_2)]
    
    def step(self, action, phase, train_count):
        self.handle_emitter(action)
        # for _ in range(self.f_ratio):
        #     super(Supervisor, self).step(self.time_step // self.f_ratio)
        super(Supervisor, self).step(self.time_step)
        msg = self.handle_receiver() # gs_values[1] and ps sensor values
        self.steps += 1
        self.gs_values_1 = msg[:, 0]  # shape of (num_agents,)
        ps_sensor_values = msg[:, 1:]  # shape of (num_agents, 8)
        
        epuck_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
        rotation_angle = np.zeros((self.num_agents, 1), dtype=np.float32)
        for i, robot in enumerate(self.robots):
            epuck_pos[i] = robot.getField('translation').getSFVec3f()[:2]
            rotation = robot.getField('rotation').getSFRotation()
            rotation_angle[i] = rotation[3] if rotation[2] > 0 else -rotation[3]
        
        is_new_tiles = self.update_env(epuck_pos, rotation_angle)
        self.exploration_ratios = self.get_exploration_ratio()
        self.local_ratios = self.get_ratio_estimation()
        for k, v in self.local_ratio_dict.items():
            v.append(self.local_ratios[k])
        local_ratios = np.expand_dims(self.local_ratios, axis=-1)
        exp_ratios = np.expand_dims(self.exploration_ratios, axis=-1)
        
        # is_new_tiles_np = np.expand_dims(is_new_tiles, axis=-1).astype(np.float32)
        # exploration_ratio = np.expand_dims([sum(self.env_tiles.tiles_visited)/(self.row * self.col)] * self.num_agents, axis=-1)
        # ps_values_normalized = (ps_sensor_values - 55) / (max(np.max(ps_sensor_values), 450) - 55)
        self.all_states = np.concatenate(
            [epuck_pos, rotation_angle, ps_sensor_values], axis=1
        )
        contributions = [0] * self.num_agents
        if self.steps % self.args.update_ratio_steps == 0:
            contributions = self._update_global_ratio(phase)
            self.is_update_global_ratio = True
        if self.steps % 2000 == 0:
            s = self.get_exploration_ratio()
            print(f"steps: {self.steps}")
            print(self.global_ratio_list[-2][1], self.global_ratio_list[-1][1])
            for i in range(self.num_agents):
                print(f"agent {i} - local ratio: {self.local_ratios[i]:.4f}, exploration ratio: {s[i]:.4f}")
        return (
            self.all_states,  # [num_agents x num_states], np.array
            self.get_map_info(),  # num_agents x [col, row], list of np.array
            self.get_reward(epuck_pos, exp_ratios, is_new_tiles, action),  # [num_agents,], np.array
            self.is_done(phase, train_count),  # [num_agents,], list
            contributions,
            self.get_info()  # [num_agents,]
        )
       
    def handle_receiver(self):
        message = np.zeros((self.num_agents, 9))
        for i in range(self.num_agents):
            if self.receiver.getQueueLength() > 0:
                try:
                    string_message = self.receiver.getString().split(',')
                except AttributeError:
                    string_message = self.receiver.getData().decode("utf-8")
                self.receiver.nextPacket()
                idx = int(string_message[0][1])
                message[idx] = np.array(string_message[1:]).astype(np.float32)
        return message
    
    def get_map_info(self):
        map_infos = []
        for local_map in self.local_maps:
            map_info = np.array(local_map.tiles_visited).astype(np.int32).reshape(local_map.col, local_map.row)
            map_infos.append(map_info)
        return map_infos
    
    
    def get_reward(self, positions, exp_ratios, is_new_tiles, action):
        # super().get_reward(action)
        '''if (self.message is None or len(self.message) == 0
                or self.observation is None):
            return 0'''
        assert len(is_new_tiles) == self.num_agents, "is_new_tiles should have the same length as num_agents"
        # exploration rewards
        exploration_r = np.zeros((self.num_agents,), dtype=np.float32) + self.reward_time
        collision_bounds = (
            (positions[:, 0] < self.x_min) | (positions[:, 0] > self.x_max) | 
            (positions[:, 1] < self.y_min) | (positions[:, 1] > self.y_max)
        )
        dist_matrix = np.sqrt(np.sum(
            (positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=-1
        ))
        collision_counts = np.sum(dist_matrix < self.collision_distance, axis=1) - 1 + collision_bounds.astype(int)
        for i in range(self.num_agents):
            exploration_r[i] += self.reward_exploration if is_new_tiles[i] else self.reward_repeat
            exploration_r[i] += collision_counts[i] * self.reward_collision
            # exploration_r[i] += self.reward_exp_ratio * exp_ratios[i]
        
        # ratio estimation rewards
        ratio_r = np.zeros((self.num_agents,), dtype=np.float32)
        # local
        for i in range(self.num_agents):
            ratio_r[i] += self.reward_local_ratio * (-abs(self.local_ratios[i] - self.args.black_ratio))
        # global
        if self.is_update_global_ratio:
            ratio_r += self.reward_global_ratio * (
                1 - abs(self.global_ratio_estimation - self.args.black_ratio)
            )
            self.global_reward_total += self.reward_global_ratio * (
                1 - abs(self.global_ratio_estimation - self.args.black_ratio)
            )
            self.global_reward_last = self.reward_global_ratio * (
                1 - abs(self.global_ratio_estimation - self.args.black_ratio)
            )
            self.aggregation_count += 1
            self.is_update_global_ratio = False
        
        return exploration_r + ratio_r
    
    def is_done(self, phase, train_count):
        # super().is_done() and 
            # abs(self.global_ratio_list[1] - self.global_ratio_list[0]) < 0.01
        # if (all([r > self.args.done_exploration for r in self.exploration_ratios])and 
        #     abs(self.global_ratio_list[1] - self.global_ratio_list[0]) < 0.01):
        if phase == "train":
            exploration = self.args.done_exploration - 0.01 * train_count
            if (np.mean(self.exploration_ratios) > max(exploration, self.args.min_exploration) and 
                abs(self.global_ratio_list[-1][1] - self.global_ratio_list[-2][1]) < self.args.done_ratio_difference):
                print(f"ratio -2:{self.global_ratio_list[-2][1]}, ratio -1:{self.global_ratio_list[-1][1]}") 
                return [True] * self.num_agents
            else:
                return [False] * self.num_agents
        elif phase == "eval":
            if (len(self.global_ratio_list) > 1 and
                abs(self.global_ratio_list[-1][1] - self.global_ratio_list[-2][1]) < self.args.done_ratio_difference):
                print(f"ratio -2:{self.global_ratio_list[-2][1]}, ratio -1:{self.global_ratio_list[-1][1]}") 
                return [True] * self.num_agents
            else:
                return [False] * self.num_agents
    
    def get_info(self):
        return None
    
    def reset_visited(self):
        for local_map in self.local_maps:
            local_map.tiles_visited = [False for _ in range(local_map.col * local_map.row)]
            local_map.black_count = 0
            local_map.white_count = 0

        floor_tiles_root = self.getFromDef("Floor_Tiles")
        floor_tiles_child = floor_tiles_root.getField("children")
        for idx in range(self.col * self.row):
            tile_ = floor_tiles_child.getMFNode(idx)
            tile_children = tile_.getField("children")
            appearance = tile_children.getMFNode(0).getField("appearance")
            base_color = appearance.getSFNode().getField("baseColor").getSFColor()
            appearance.getSFNode().getField("emissiveColor").setSFColor(base_color)
    
    def reset_state(self):
        offset_x = -0.1 * self.row / 2 - 0.05
        offset_y = -0.1 * self.col / 2 - 0.05
        x_y = [(offset_x + 0.1 * (j + 1), offset_y + 0.1 * (i + 1))
                for i in range(self.col) for j in range(self.row)]
        starts = random.sample(x_y, self.num_agents)
        for i in range(self.num_agents):
            robot = self.robots[i]
            epuck_default_pos = robot.getField('translation').getSFVec3f()
            epuck_default_pos[:2] = starts[i]
            robot.getField('translation').setSFVec3f(epuck_default_pos)
            rotation_angle = random.uniform(-np.pi, np.pi)
            robot.getField('rotation').setSFRotation([0, 0, 1, rotation_angle])
    
    def reset(self):
        if self.start_flag:
            for _ in range(self.args.exclude_steps):
                super(Supervisor, self).step(self.time_step)
                msg = self.handle_receiver()
            self.start_flag = False
        self.steps = 0
        self.global_ratio_estimation = 0
        self.global_reward_total = 0
        self.global_reward_last = 0
        self.aggregation_count = 0
        self.update_count = 0
        self.global_ratio_list.clear()
        self.local_ratio_dict = {i: [] for i in range(self.num_agents)}
        # self.simulationReset()
        # for robot in self.robots:
        #     robot.restartController()
        self.simulationResetPhysics()
        self.start_time = super(Supervisor, self).getTime()
        self.reset_state()
        self.reset_visited()
        super(Supervisor, self).step(self.time_step)
        msg = self.handle_receiver()
        ps_sensor_values = msg[:, 1:]
        # ps_values_normalized = (ps_sensor_values - 55) / (max(np.max(ps_sensor_values), 450) - 55)
        epuck_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
        rotation_angle = np.zeros((self.num_agents, 1), dtype=np.float32)
        for i, robot in enumerate(self.robots):
            epuck_pos[i] = robot.getField('translation').getSFVec3f()[:2]
            rotation = robot.getField('rotation').getSFRotation()
            rotation_angle[i] = rotation[3] if rotation[2] > 0 else -rotation[3]
        local_ratios = np.expand_dims(self.get_ratio_estimation(), axis=-1)
        init_state = np.concatenate([epuck_pos, rotation_angle, ps_sensor_values], axis=1)
        return init_state
    
    def _update_global_ratio(self, phase):
        shapley_value = [0] * self.num_agents

        if self.args.ratio_update_method == "threshold":
            selected_idx = self._threshold_select()
        elif self.args.ratio_update_method == "shapley":
            shapley_value = self.get_shapley_value(phase)
            selected_idx = np.argsort(shapley_value)[-int(2 / 3 * self.num_agents):]
        elif self.args.ratio_update_method == "all":
            selected_idx = list(range(self.num_agents))
        
        for i in selected_idx:
            delta = self.local_ratios[i] - self.global_ratio_estimation
            self.update_count += 1
            self.global_ratio_estimation += delta / self.update_count
        self.global_ratio_list.append((self.steps, self.global_ratio_estimation))
        
        return shapley_value

    def _threshold_select(self):
        if self.update_count == 0:
            selected_idx = list(range(self.num_agents))
        else:
            selected_idx = []
            for idx, r in enumerate(self.local_ratios):
                delta = r - self.global_ratio_estimation
                if abs(delta) < 0.1:
                    selected_idx.append(idx)
            if len(selected_idx) == 0:
                selected_idx = list(range(self.num_agents))
        return selected_idx
    
    def get_shapley_value(self, phase) -> np.ndarray:
        shapley_values = [0] * self.num_agents
        if phase == "train":
            for i in range(self.num_agents):
                for combination in self.combinations[i]:
                    global_reward_with_i = self.get_union_reward(combination, i)
                    global_reward_without_i = self.get_union_reward(combination, -1)
                    shapley_values[i] += global_reward_with_i - global_reward_without_i
                shapley_values[i] /= len(self.combinations[i])
        elif phase == "eval":
            tmp_local_ratios = np.array(self.local_ratios)
            for i in range(self.num_agents):
                for combination in self.combinations[i]:
                    consistency_without_i = -np.var(tmp_local_ratios[combination])
                    consistency_with_i = -np.var(tmp_local_ratios[combination + [i]])
                    shapley_values[i] += consistency_with_i - consistency_without_i
                shapley_values[i] /= len(self.combinations[i])
        return shapley_values
    
    def get_union_reward(self, combination: list, agent: int) -> float:
        # without agent
        tmp_count = 0
        tmp_global_ratio = self.global_ratio_estimation
        for other in combination:
            delta = self.local_ratios[other] - tmp_global_ratio
            tmp_count += 1
            tmp_global_ratio += delta / (self.update_count + tmp_count)
        if agent != -1:
            delta = self.local_ratios[agent] - tmp_global_ratio
            tmp_count += 1
            tmp_global_ratio += delta / (self.update_count + tmp_count)
        return self.reward_global_ratio * (1 - abs(tmp_global_ratio - self.args.black_ratio))
    
    def get_episode_time(self):
        return super(Supervisor, self).getTime() - self.start_time
    
    def get_ratio_estimation(self):
        """final ratio estimation"""
        return [local_map.get_ratio() for local_map in self.local_maps]
    
    def get_exploration_ratio(self): 
        return [sum(local_map.tiles_visited) / (local_map.row * local_map.col) 
                for local_map in self.local_maps]

    def update_env(self, epuck_pos, rotation_angle):
        is_new_tiles = []
        new_tiles_idxs = []
        for i in range(self.num_agents):
            x, y = epuck_pos[i]
            is_new, idx = self.local_maps[i].check(x, y, rotation_angle[i])
            is_new_tiles.append(is_new)
            if is_new:
                new_tiles_idxs.append(idx)
                if self.gs_values_1[i] > 900:
                    self.local_maps[i].white_count += 1
                else:
                    self.local_maps[i].black_count += 1
        
        if len(new_tiles_idxs) > 0:
            floor_tiles_root = self.getFromDef("Floor_Tiles")
            floor_tiles_child = floor_tiles_root.getField("children")
            for idx in new_tiles_idxs:
                tile_ = floor_tiles_child.getMFNode(idx)
                tile_children = tile_.getField("children")
                appearance = tile_children.getMFNode(0).getField("appearance")
                appearance.getSFNode().getField("emissiveColor").setSFColor([1, 0, 0])
        
        return is_new_tiles

    # def get_total_time(self):
    #     return super(Supervisor, self).getTime()


# if __name__ == '__main__':
#     save_path = os.path.join(current_dir, "..", "..", "results")
#     runs = Epuck2Supervisor.file_length(
#         os.path.join(save_path, "runs.txt"))
#     print(f"===== runs at {runs} =====")
#     args, unknown = parser.parse_known_args()
#     my_controller = Epuck2Supervisor(args, save_path, col=20, row=20)
#     try:
#         while True:
#             my_controller.step([1] * args.num_agents)
#     except Exception as e:
#         print(e)