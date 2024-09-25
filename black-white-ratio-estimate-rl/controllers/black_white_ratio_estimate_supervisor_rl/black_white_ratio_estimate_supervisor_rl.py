#!/usr/bin/python-deepbots
import os, sys
import random
import random
import time
from datetime import datetime

from deepbots.supervisor.controllers.csv_supervisor_env import CSVSupervisorEnv
from gym.spaces import Discrete
sys.path.append("/usr/local/webots/lib/controller/python")
os.environ['WEBOTS_HOME'] = '/usr/local/webots'
from controller import Supervisor, Robot
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, "..", "..")
sys.path.append(exp_path)
from config import parser

class Env:
    def __init__(self, col, row, threshold=0.04):
        self.col = col
        self.row = row
        self.threshold = threshold
        self.offset_x = -0.1 * self.row / 2 - 0.05
        self.offset_y = -0.1 * self.col / 2 - 0.05
        
        self.tiles = [(self.offset_x + 0.1 * (j + 1), self.offset_y + 0.1 * (i + 1))
                      for i in range(self.col) for j in range(self.row)]
        self.tiles_visited = [False for _ in range(self.col * self.row)]
        self.last_exploration = 0

    def check(self, x, y, angle):
        # calculate the position of gs
        x = x + 0.04 * math.cos(angle)
        y = y + 0.04 * math.sin(angle)
        for idx, (tile_x, tile_y) in enumerate(self.tiles):
            if tile_x - 0.03 < x < tile_x + 0.03 and tile_y - 0.03 < y < tile_y + 0.03:
                if not self.tiles_visited[idx]:
                    self.tiles_visited[idx] = True
                    return True, idx
        
        return False, None


class Epuck2Supervisor(CSVSupervisorEnv):
    def __init__(self, args, save_path, col, row) -> None:
        super().__init__(timestep=args.time_step)
        self.args = args
        self.save_path = save_path
        self.col, self.row = col, row
        self.black_count = 0
        self.white_count = 0
        self.start_time = 0
        self.num_agents = self.args.num_agents
        self.ps_threshold = self.args.ps_threshold
        self.dist_threshold = self.args.dist_threshold
        self.num_envs = 1
        self.ranger_robots = list(range(self.args.ranger_robots))
        self.comm_ranges = [
            self.args.range1 if id in self.ranger_robots else self.args.range0
            for id in range(self.num_agents)]
        
        self.group_number = self.args.group_number
        self.groups = {i: [] for i in range(self.group_number)}
        for i in range(self.num_agents):
            self.groups[i % self.group_number].append(i)
        self.swarm = []
        for g in self.groups.values():
            self.swarm.extend(g)
        self.all_states = None
        self.start_exploration = False
        self.env_tiles = Env(self.col, self.row, self.args.center_threshold)
        
        # print info
        # print("time_step: ", self.time_step)
        print("========== supervisor info ==========")
        print(f"groups: {self.groups}")
        print(f"swarm: {self.swarm}")
        
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
        self.reward_repeat = self.args.reward_repeat
        self.collision_distance = self.args.collision_distance
        self.reward_collision = self.args.reward_collision
        self.num_actions = 5  # The agent can perform 2 actions
        self.num_states = 2 + 1 + 8 # position + angle + 8 ps sensor values
    
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
            offset_x + 0.1 * self.row + self.dist_threshold, offset_x + 0.1 * 1 - self.dist_threshold,
            offset_y + 0.1 * self.col + self.dist_threshold, offset_y + 0.1 * 1 - self.dist_threshold
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
            # emitter_string = """
            #     Emitter {
            #         name "emitter0%d"
            #         channel %d
            #     }""" % (i, i + 3)
            # supervisor_node = self.getFromDef("EnvSet_supervisor")
            # children_field = supervisor_node.getField("children")
            # children_field.importMFNodeFromString(-1, emitter_string)
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
        
        print("========== import robots successfully ==========")

    def step(self, action):
        for _ in range(self.f_ratio):
            if super(
                Supervisor, self
            ).step(self.time_step // self.f_ratio) == -1:
                exit()
        self.handle_emitter(action)
        msg = self.handle_receiver() # gs_values[1] and ps sensor values
        self.steps += 1
        self.gs_values_1 = msg[:, 0]  # shape of (num_agents,)
        ps_sensor_values = msg[:, 1:]  # shape of (num_agents, 8)
        # if self.start_exploration:
        #     self.update_emissive_color([])
        
        epuck_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
        rotation_angle = np.zeros((self.num_agents, 1), dtype=np.float32)
        for i, robot in enumerate(self.robots):
            epuck_pos[i] = robot.getField('translation').getSFVec3f()[:2]
            rotation = robot.getField('rotation').getSFRotation()
            rotation_angle[i] = rotation[3] if rotation[2] > 0 else -rotation[3]
        is_new_tiles = [False] * self.num_agents
        if self.start_exploration:
            is_new_tiles = self.update_env(epuck_pos, rotation_angle)
        
        is_new_tiles_np = np.expand_dims(is_new_tiles, axis=-1).astype(np.float32)
        exploration_ratio = np.expand_dims([sum(self.env_tiles.tiles_visited)/(self.row * self.col)] * self.num_agents, axis=-1)
        ps_values_normalized = (ps_sensor_values - 55) / (max(np.max(ps_sensor_values), 450) - 55)
        self.all_states = np.concatenate(
            [is_new_tiles_np, exploration_ratio, epuck_pos, rotation_angle, ps_values_normalized], axis=1
        )
        return (
            self.all_states,  # [num_agents x num_states], np.array
            self.get_map_info(),  # [col, row], np.array
            self.get_reward(epuck_pos, is_new_tiles, action),  # [num_agents,], np.array
            self.is_done(),  # [num_agents,], list
            self.get_info(ps_sensor_values)  # [num_agents,]
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
        map_info = self.env_tiles.tiles_visited
        map_info = np.array(map_info).astype(np.int32).reshape(self.col, self.row)
        return map_info
    
    
    def get_reward(self, positions, is_new_tiles, action):
        # super().get_reward(action)
        '''if (self.message is None or len(self.message) == 0
                or self.observation is None):
            return 0'''
        rewards = np.zeros((self.num_agents,), dtype=np.float32) + self.reward_time
        collision_bounds = (
            (positions[:, 0] < self.x_min) | (positions[:, 0] > self.x_max) | 
            (positions[:, 1] < self.y_min) | (positions[:, 1] > self.y_max)
        )
        dist_matrix = np.sqrt(np.sum(
            (positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=-1
        ))
        collision_counts = np.sum(dist_matrix < self.collision_distance, axis=1) - 1 + collision_bounds.astype(int)
        
        for i in range(self.num_agents):
            rewards[i] += self.reward_exploration if is_new_tiles[i] else self.reward_repeat
            rewards[i] += collision_counts[i] * self.reward_collision
        return rewards
    
    def is_done(self):
        # super().is_done()
        if self.env_tiles.tiles_visited.count(True) == self.col * self.row:
            return [True] * self.num_agents
        if self.steps - 1 == self.args.episode_length:
            return [True] * self.num_agents
        else:
            return [False] * self.num_agents
    
    def get_info(self, values):
        # super().get_info()
        obstacles = []
        for i in range(self.num_agents):
            right_obstacle = any(
                value > self.ps_threshold for value in values[i][:3])
            left_obstacle = any(
                value > self.ps_threshold for value in values[i][-3:])
            front_obstacle = right_obstacle and left_obstacle
            back_obstacle = any(
                value > self.ps_threshold for value in values[i][3:-3]
            )
            if front_obstacle:
                obstacles.append("front")
            elif right_obstacle:
                obstacles.append("right")
            elif left_obstacle:
                obstacles.append("left")
            elif back_obstacle:
                obstacles.append("back")
            else:
                obstacles.append("none")
        return obstacles
    
    def reset_visited(self):
        self.start_exploration = True
        self.env_tiles.tiles_visited = [False for _ in range(self.col * self.row)]
        self.env_tiles.last_exploration = 0
        self.black_count = 0
        self.white_count = 0

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
        
        # for start, robot in zip(starts, self.robots):
        #     epuck_default_pos = robot.getField('translation').getSFVec3f()
        #     epuck_default_pos[:2] = start
        #     robot.getField('translation').setSFVec3f(epuck_default_pos)
        #     robot.getField('rotation').setSFRotation([0, 0, 1, random.uniform(-np.pi, np.pi)])
    
    def reset(self):
        self.steps = 0
        #self.simulationReset()
        self.simulationResetPhysics()
        self.start_time = super(Supervisor, self).getTime()
        for robot in self.robots:
            robot.restartController()
        super(Supervisor, self).step(self.time_step//self.f_ratio)

        self.reset_state()
        for _ in range(self.f_ratio-1):
            super(Supervisor, self).step(self.time_step//self.f_ratio)
        
        # test = self.handle_receiver()

        # now = datetime.now()
        # time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        # self.write_file(
        #     os.path.join(self.save_path, "runs.txt"),
        #     f"reset to next run {time_string} \n", mode="a+")
        # self.simulationReset()
        # supervisor_node = self.getFromDef("EnvSet_supervisor")
        # supervisor_node.restartController()
    
    def get_episode_time(self):
        return super(Supervisor, self).getTime() - self.start_time
    
    def get_ratio_estimation(self):
        """final ratio estimation"""
        try:
            ratio = self.black_count / (self.black_count + self.white_count)
        except ZeroDivisionError:
            ratio = 0
        return ratio
    
    def update_env(self, epuck_pos, rotation_angle):
        is_new_tiles = []
        new_tiles_idxs = []
        for i in range(self.num_agents):
            x, y = epuck_pos[i]
            is_new, idx = self.env_tiles.check(x, y, rotation_angle[i])
            is_new_tiles.append(is_new)
            if is_new:
                new_tiles_idxs.append(idx)
                if self.gs_values_1[i] > 900:
                    self.white_count += 1
                else:
                    self.black_count += 1
        
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