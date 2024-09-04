#!/usr/bin/python-webots
import os, sys
import random
import yaml
import random
from datetime import datetime
import socket
import threading
import traceback
from copy import deepcopy

from controller import LED, Supervisor
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, "..", "..")
sys.path.append(exp_path)
from config import parser
# from torch_models import *


class SupervisorController:
    def __init__(self, args, save_path, col, row) -> None:
        self.args = args
        self.save_path = save_path
        self.supervisor = Supervisor()
        self.num_robots = self.args.num_robots
        self.ranger_robots = list(range(self.args.ranger_robots))
        self.comm_ranges = [
            self.args.range1 if id in self.ranger_robots else self.args.range0
            for id in range(self.num_robots)]
        
        self.group_number = self.args.group_number
        self.groups = {i: [] for i in range(self.group_number)}
        for i in range(self.num_robots):
            self.groups[i % self.group_number].append(i)
        self.swarm = []
        for g in self.groups.values():
            self.swarm.extend(g)
        
        # print info
        print("========== supervisor info ==========")
        print(f"groups: {self.groups}")
        print(f"swarm: {self.swarm}")
        
        self.time_step = self.args.time_step  # in ms

        self.init_env(self.args.black_ratio, col, row)
        self.init_robots(col, row)
        self.clear_file(os.path.join(self.save_path, "tmp_result.txt"))
     
    def init_env(self, black_ratio, col, row):
        """ In the two-dimensional space by deault
                    x
                    ^
                    |
            y       |
            <-------+ 
        - the x-coordinate is the same for each element in a row
        - the y-coordinate is the same for each element in a column"""
        blacktiles = random.sample(
            range(row * col), int(row * col * black_ratio))
        offset_x = -0.1 * row / 2 - 0.05
        offset_y = -0.1 * col / 2 - 0.05
        tiles_list = [
            """
            Solid {
                translation %f %f 0.001
                rotation 1 0 0 1.5707963267948966
                children [
                    Shape {
                        appearance PBRAppearance {
                            baseColor 0 0 0
                            emissiveColor %f %f %f
                        }
                        geometry Box {
                            size 0.1 0.001 0.1
                        }
                    }
                ]
            }
            """ % (
                offset_x + 0.1 * (j + 1), offset_y + 0.1 * (i + 1), 0, 0, 0
            ) if row * i + j in blacktiles else
            """
            Solid {
                translation %f %f 0.001
                rotation 1 0 0 1.5707963267948966
                children [
                    Shape {
                        appearance PBRAppearance {
                            baseColor 0 0 0
                            emissiveColor %f %f %f
                        }
                        geometry Box {
                            size 0.1 0.001 0.1
                        }
                    }
                ]
            }
            """ % (
                offset_x + 0.1 * (j + 1), offset_y + 0.1 * (i + 1), 1, 1, 1
            ) for i in range(col) for j in range(row)
        ]
        children_string = "".join(tiles_list)
        line_string = """
            DEF Floor_Tiles Solid {
                children [
                    %s
                ]
            }
            """ % children_string
        root = self.supervisor.getRoot()
        chFd = root.getField("children")
        chFd.importMFNodeFromString(-1, line_string)
        print("========== import tiles successfully ==========")

    def init_robots(self, col, row):
        offset_x = -0.1 * row / 2 - 0.05
        offset_y = -0.1 * col / 2 - 0.05
        x_y = [(offset_x + 0.1 * (j + 1), offset_y + 0.1 * (i + 1))
                for i in range(col) for j in range(row)]
        starts = random.sample(x_y, self.num_robots)
        # starts = [(i * 0.05, 0) for i in range(6)]
        z_ = [0] * self.num_robots
        rotation_init = [i / 100.0 for i in range(
            0, 628, int(628 / self.num_robots))]
        random.shuffle(rotation_init)
        rgblist = ['{:08b}'.format(i + 1) 
                   if i in self.ranger_robots else None for i in self.swarm]
        for i in range(self.num_robots):
            print(f"import robot {i:2d}")
            if rgblist[i] is None:
                line_string = """
                    DEF epuck%d E-puck {
                        translation %f %f %f
                        rotation 0 0 1 %f
                        name "e-puck%d"
                        controller "black-white-ratio-estimate"
                        customData "%d"
                        supervisor TRUE
                        version "1"
                        camera_fieldOfView 0.5
                        camera_width 48
                        camera_height 48
                        camera_antiAliasing TRUE
                        camera_rotation 0 1 0 1.57
                    }""" % (i, starts[i][0], starts[i][1], z_[i], 
                            rotation_init[i], i, i)
            else:
                line_string = """
                    DEF epuck%d E-puck{
                        translation %f %f %f
                        rotation 0 0 1 %f
                        name "e-puck%d"
                        controller "black-white-ratio-estimate"
                        customData "%d"
                        supervisor TRUE
                        version "2"
                        camera_fieldOfView 0.5
                        camera_width 48
                        camera_height 48
                        camera_antiAliasing TRUE
                        camera_rotation 0 1 0 1.57
                        emissiveColor %s %s %s
                        emissiveColor2 %s %s %s
                    }""" % (i, starts[i][0], starts[i][1], z_[i], 
                            rotation_init[i], i, i,
                            rgblist[-1], rgblist[-2], rgblist[-3],
                            rgblist[-4], rgblist[-5], rgblist[-6])
            
            root = self.supervisor.getRoot()
            chFd = root.getField("children")
            chFd.importMFNodeFromString(-1,line_string)
        
        print("========== import robots successfully ==========")

    @staticmethod
    def clear_file(file):
        with open(file, "w") as f:
            f.write("")
    
    @staticmethod
    def write_file(file, content, mode):
        with open(file, mode=mode) as f:
            f.write(content)
    
    @staticmethod
    def file_length(file):
        with open(file, "r") as f:
            return len(f.readlines())
    
    @staticmethod
    def change_config(file, *args, **kwargs):
        pass

    def run(self):
        start_time = self.supervisor.getTime()
        while self.supervisor.step(self.time_step) != -1:
            total_time = self.supervisor.getTime() - start_time
            now = datetime.now()
            string = f"Exit Time: {total_time}\n"
            string += now.strftime("%Y-%m-%d %H:%M:%S")
            if total_time > 3000:
                print("Exceeded max time")
                string += "\n----------------Consensus NotReached! "
                string += "This experiment is time-exceeded-----------------\n"
                self.write_file(
                    os.path.join(self.save_path, "all_result.txt"),
                    string, mode="a+")
                self.reset_all()
            
            tmp_length = self.file_length(
                os.path.join(self.save_path, "tmp_result.txt")) 
            if tmp_length >= self.num_robots:
                print(f"There are {tmp_length} lines in tmp_result.txt")
                self.clear_file(os.path.join(self.save_path, "tmp_result.txt"))
                string += "\n----------------Consensus Reached! "
                string += "This experiment is finished-----------------\n"
                self.write_file(
                    os.path.join(self.save_path, "all_result.txt"),
                    string, mode="a+")
                self.reset_all()
    
    def reset_all(self):
        now = datetime.now()
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        self.write_file(
            os.path.join(self.save_path, "runs.txt"),
            f"reset to next run {time_string} \n", mode="a+")
        self.supervisor.simulationReset()
        supervisor_node = self.supervisor.getFromDef("EnvSet_supervisor")
        supervisor_node.restartController()


if __name__ == '__main__':
    save_path = os.path.join(current_dir, "..", "..", "results")
    runs = SupervisorController.file_length(
        os.path.join(save_path, "runs.txt"))
    print(f"===== runs at {runs} =====")
    args, unknown = parser.parse_known_args()
    my_controller = SupervisorController(args, save_path, col=20, row=20)
    my_controller.run()
