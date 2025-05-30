import socket
import json

import time, datetime
import multiprocessing as mp
import sys, os
import traceback
import random
from copy import deepcopy
from collections import deque
from itertools import islice

import numpy as np
import torch
import rospy
import pandas as pd
from std_msgs.msg import String
from tensorboardX import SummaryWriter

from adversary import Adversary
from aggregator import Aggregator
from server import Server
from model import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# mp.set_start_method('spawn', force=True)

class Leader(Server):
    def __init__(self, args, parameters):
        super().__init__()
        self.args = args
        self.byz_robots = [] if self.args.byz_num == 0 else list(range(self.args.num_agents))[-self.args.byz_num:]
        self.byz_style = self.args.byz_style
        self.aggregator = Aggregator(self.args.grad_aggregation)
        if "grad" in self.byz_style:
            self.adversary = Adversary(self.byz_style.split("-")[1:])
        self.shared_array = mp.Array('d', len(parameters))
        self.parameters = np.frombuffer(self.shared_array.get_obj(), dtype=np.float64)
        np.copyto(self.parameters, parameters)
        self.version = mp.Value('i', 0)
        self.last_time = time.time()
        manager = mp.Manager()
        self.buffer_grad = manager.list()  # (contribution and gradient)
        self.init_event = mp.Event()
        self.lock = mp.Lock()
        self.log_dir = None
        self.tb_writer = None
        
    def get_message(self):
        full_data = b""
        while True:
            data = self.clientsocket.recv(1024)
            if not data:
                break
            full_data += data
            if b"<END>" in full_data:
                full_data = full_data.replace(b"<END>", b"")
                break
        full_data = json.loads(full_data)
        if full_data["info"] == "access the latest model":
            print(f"==== sending the latest model to the worker {full_data['id']}")
            with self.lock:
                return self.parameters, self.version.value
        if 'gradients' in full_data['info']:
            self.add_grad(full_data['id'], full_data['info'])
            return None, full_data['id']
        # if full_data['info'] in self.byz_style:
        #     return None, full_data['id']
    
    def create_tb_writer(self):
        assert self.log_dir is not None
        tb_dir = os.path.join(self.log_dir, "leader")
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        self.tb_writer = SummaryWriter(tb_dir)
    
    def add_grad(self, id, infos: dict):
        """infos: {"gradients": list, "contribution": float}"""
        with self.lock:
            self.buffer_grad.append(
                (infos['contribution'], infos['gradients'], id)
            )
    
    def run(self):
        while True:
            self.accept_new_connection()
            p, v = self.get_message()
            if p is not None:
                msg = {
                    "parameters": p.tolist(),
                    "version": v,
                }
                self.send_message(msg)
            else:
                self.send_message(f"upload successfully - worker{v}")
            time.sleep(0.1)
    
    def aggregate(self):
        while True:
            t = time.time()
            if t - self.last_time >= self.args.aggregation_time:
                self.last_time = t
                with self.lock:
                    if len(self.buffer_grad) > (self.args.num_agents // 2):
                        # 针对是否存在梯度攻击，进行分类
                        if "grad" in self.byz_style:
                            contributions = [grad[0] for grad in self.buffer_grad if isinstance(grad[1], list)]
                            gradients = [grad[1] for grad in self.buffer_grad if isinstance(grad[1], list)]
                            tensor_gradients = torch.tensor(gradients, dtype=torch.float)
                            if "signflip" not in self.byz_style:
                                # adversary_grad = self.adversary(
                                #     tensor_gradients,
                                #     num_byz=self.args.byz_num
                                # )
                                # for _ in range(self.args.byz_num):
                                #     tensor_gradients = torch.cat((tensor_gradients, adversary_grad), dim=0)
                                adversary_grad = self.adversary(
                                    tensor_gradients,
                                    num_byz=self.args.byz_num
                                )
                                tensor_gradients = torch.cat((tensor_gradients, adversary_grad), dim=0)
                            contributions.extend([grad[0] for grad in self.buffer_grad if isinstance(grad[1], str)])
                            # aggregated_grad = self.aggregator(tensor_gradients)
                        # 没有存在梯度攻击
                        else:
                            contributions = np.array([grad[0] for grad in self.buffer_grad])
                            gradients = [grad[1] for grad in self.buffer_grad]
                            tensor_gradients = torch.tensor(gradients, dtype=torch.float)
                            # weights = self.softmax(contributions)
                            # print(self.args.ratio_update_method, weights)
                            # aggregated_grad = sum(w * g for w, g in zip(weights, gradients))
                        
                        aggregated_grad = self.aggregator(tensor_gradients, contributions=contributions)
                        aggregated_grad = self.clip_norm(aggregated_grad, self.args.grad_norm_init)
                        self.tb_writer.add_scalar("norm/aggregated", np.linalg.norm(aggregated_grad), self.version.value + 1)
                        self.tb_writer.flush()
                        self.parameters -= self.args.lr * aggregated_grad
                        self.version.value += 1
                        self.buffer_grad[:] = []
                        now = datetime.datetime.now()
                        print("aggregated with grad in norm {} successfully at {} and get version{}".format(
                            np.linalg.norm(aggregated_grad), now.strftime('%H:%M:%S'), self.version.value
                        ))
            time.sleep(0.1)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    @staticmethod
    def clip_norm(arr, max_norm):
        if max_norm == 0:
            return arr
        else:
            arr_norm = np.linalg.norm(arr)
            if arr_norm > max_norm:
                arr = arr * (max_norm / (arr_norm + 1e-6))
            return arr
        

class Worker:
    def __init__(self, args):
        self.id = args.id
        rospy.init_node(f'agent_{self.id}_node', anonymous=True)
        self.args = args
        self.byz_robots = [] if self.args.byz_num == 0 else list(range(self.args.num_agents))[-self.args.byz_num:]
        self.byz_style = self.args.byz_style
        if self.byz_style != "":
            assert self.args.byz_num > 0, "wrong byzantine numbers"
        self.ip, self.port = "localhost", 9801
        self.last_action = None
        self.obstacle = False
        self.steps = 0
        self.model = Model(
            11, 4, self.args.col, self.args.row,
            device, self.args.reward_normalize
        )
        if self.args.model_dir is not None:
            load_model = torch.load(self.args.model_dir, map_location=device)
            self.model.load_state_dict(load_model.state_dict())
            print("lode model successfully")
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.buffer_s, self.next_s = [], None
        self.buffer_map, self.next_map = [], None
        self.buffer_a = []
        self.buffer_r = []
        self.contribution = 0
        # self.buffer_done = []
        self.buffer_mask = []
        self.done = False
        self.save_dir = None
        self.log_dir = None

        self.publisher = rospy.Publisher(f'action_{self.id}', String, queue_size=10)
        self.subscriber_state = rospy.Subscriber(f'state_agent_{self.id}', String, self.state_callback)
        self.subscriber_reward = rospy.Subscriber(f'reward_agent_{self.id}', String, self.reward_callback)
    
    def state_callback(self, data):
        agent_data = json.loads(data.data)
        if self.save_dir is None:
            self.save_dir = agent_data['save_dir']
        if self.log_dir is None:
            self.log_dir = agent_data['log_dir']
            tb_dir = os.path.join(self.log_dir, f"worker_{self.id}")
            if not os.path.exists(tb_dir):
                os.makedirs(tb_dir)
            self.tb_writer = SummaryWriter(tb_dir)
            if self.id == 0 and self.leader.log_dir is None:
                self.leader.log_dir = agent_data['log_dir']
                self.leader.create_tb_writer()
                a = mp.Process(target=self.leader.aggregate)
                a.start()
        # sync with the global model at the beginning of each episode
        if agent_data['step'] == 1:
            self.sync_with_global_model()
            self.last_action = None
        
        
        if self.id in self.byz_robots and "action" in self.byz_style:
            action = int(self.byz_style.split("-")[1])
            mask = torch.ones(4, dtype=torch.int)
            if agent_data['phase'] == 'train':
                self.buffer_s.append(agent_data['state'])
                self.buffer_map.append(agent_data['map'])
        else:
            mask = self.get_mask(agent_data['state'][-8:], self.last_action)
            # print("mask: ", mask)
            if agent_data['phase'] == 'eval':
                # choose action for avoiding obstacles
                ps_values = agent_data['state'][-8:]
                right_obstacle = any(
                    value > self.args.ps_threshold for value in ps_values[:3])
                left_obstacle = any(
                    value > self.args.ps_threshold for value in ps_values[-3:])
                front_obstacle = right_obstacle and left_obstacle
                back_obstacle = any(
                    value > self.args.ps_threshold for value in ps_values[3:-3]
                )
                if self.last_action == 3 and self.obstacle:
                    action = random.choice([1, 2])  # turn right or left
                else:
                    if front_obstacle:
                        action = 3
                        self.obstacle = True
                    elif right_obstacle:
                        action = 1
                        self.obstacle = True
                    elif left_obstacle:
                        action = 2
                        self.obstacle = True
                    elif back_obstacle:
                        action = 0
                        self.obstacle = True
                    else:
                        self.obstacle = False
                        state = self.normalize_ps_values_of_state(agent_data['state'])
                        with torch.no_grad():
                            action = self.model.choose_action(
                                torch.tensor(state, dtype=torch.float, device=device),
                                torch.tensor(agent_data['map'], dtype=torch.float, device=device),
                                mask,
                            )
            
            elif agent_data['phase'] == 'train':
                state = self.normalize_ps_values_of_state(agent_data['state'])
                self.buffer_s.append(state)
                self.buffer_map.append(agent_data['map'])
                
                with torch.no_grad():
                    action = self.model.choose_action(
                        torch.tensor(state, dtype=torch.float, device=device),
                        torch.tensor(agent_data['map'], dtype=torch.float, device=device),
                        mask,
                    )
            self.last_action = action

        if agent_data['phase'] == 'train':
            self.buffer_a.append(action)
            self.buffer_mask.append(mask)
        message = json.dumps(action)
        # print(f"state: {agent_data['state']}")
        # print(f"action: {action}")
        # rospy.loginfo(f'Publishing action for agent_{self.id}: {message}')
        self.publisher.publish(message)
    
    def reward_callback(self, data):
        agent_data = json.loads(data.data)
        # rospy.loginfo(f"reward info: {agent_data['reward']}")
        if agent_data['phase'] == 'train':
            self.buffer_r.append(agent_data['reward'])
            self.contribution += agent_data['contribution']
            # self.buffer_done.append(agent_data['done'])
            self.next_s = agent_data['next_state']
            self.next_map = agent_data['next_map']
            self.done = agent_data['done']
    
    def get_message(self, connection) -> dict:
        full_data = b""
        while True:
            data = connection.recv(1024)
            if not data:
                break
            full_data += data
            if b"<END>" in full_data:
                full_data = full_data.replace(b"<END>", b"")
                break
        full_data = json.loads(full_data)

        if isinstance(full_data, str):
            print(full_data)
        elif isinstance(full_data, dict):
            assert "parameters" in full_data
            self.model.load_serializable_state_list(full_data["parameters"])
            print(f"\033[0;31maccess the latest model - version {full_data['version']}\033[0m")

    def send_message(self, message, connection):
        message = json.dumps(message) + "<END>"
        connection.sendall(message.encode())

    def init_model(self):
        if self.id == 0:
            self.leader = Leader(self.args, self.model.get_serializable_state_list(to_numpy=True))
            r = mp.Process(target=self.leader.run)
            r.start()
        time.sleep(2.0)
        # if self.id > 0:
        #     self.sync_with_global_model()
    
    def sync_with_global_model(self):
        if self.id == 0:
            self.model.load_serializable_state_list(self.leader.parameters)
            torch.save(self.model, self.save_dir + f'/version_{self.leader.version.value}.pt')
            print(f"\033[0;31maccess the latest model - version {self.leader.version.value}\033[0m")
        else:
            msg = {
                "id": self.id,
                "info": "access the latest model"
            }
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect((self.ip, self.port))
                self.send_message(msg, client)
                self.get_message(client)
    
    def normalize_ps_values_of_state(self, s):
        tmp_s = deepcopy(s)
        ps_values = tmp_s[-8:]
        ps_values_normalized = (np.array(ps_values) - 55) / (max(np.max(ps_values), 450) - 55)
        tmp_s[-8:] = ps_values_normalized
        return tmp_s
    
    def get_mask(self, ps_values, last_action):
        # print("ps_values: ", ps_values)
        mask = torch.ones(4, dtype=torch.int)
        if last_action == 3:
            mask[3] = 0
            mask[0] = 0
        right_obstacle = any(
            value > self.args.ps_threshold for value in ps_values[:3])
        left_obstacle = any(
            value > self.args.ps_threshold for value in ps_values[-3:])
        front_obstacle = right_obstacle and left_obstacle
        back_obstacle = any(
            value > self.args.ps_threshold for value in ps_values[3:-3]
        )
        if right_obstacle: mask[2] = 0
        if left_obstacle: mask[1] = 0
        if front_obstacle: mask[0] = 0
        if back_obstacle: mask[3] = 0
        if mask.sum() == 0:
            mask[random.choice([0, 1, 2, 3])] = 1
        return mask
    
    def train_and_update(self, bs, bmap, ba, br, bmask, done, s_, map_, gamma, opt):
        if self.id in self.byz_robots and "grad" in self.byz_style:
            if "signflip" in self.byz_style:
                grad, loss, c_loss, a_loss = self.model.train_and_get_grad(
                    bs, bmap, ba, br, bmask, done, s_, map_, gamma, opt, self.steps
                )
                grad *= -2
                msg = {
                    "id": self.id,
                    "info": {
                        "gradients": grad.tolist(),
                        "contribution": self.contribution
                    }
                }
            else:
                msg = {
                    "id": self.id,
                    "info": {
                        "gradients": self.byz_style.split("-")[1],
                        "contribution": self.contribution
                    }
                }
            self.tb_writer.add_scalar("contributions", self.contribution, self.steps)
            self.tb_writer.flush()
        else:
            grad, loss, c_loss, a_loss = self.model.train_and_get_grad(
                bs, bmap, ba, br, bmask, done, s_, map_, gamma, opt, self.steps
            )
            msg = {
                "id": self.id,
                "info": {
                    "gradients": grad.tolist(),
                    "contribution": self.contribution
                }
            }
            ##############
            # record the contribution
            ##############
            norm_ = np.linalg.norm(grad)
            self.tb_writer.add_scalar("loss/total", loss, self.steps)
            self.tb_writer.add_scalar("loss/c_loss", c_loss, self.steps)
            self.tb_writer.add_scalar("loss/a_loss", a_loss, self.steps)
            self.tb_writer.add_scalar("contributions", self.contribution, self.steps)
            self.tb_writer.add_scalar("norm/norm_local", norm_, self.steps)
            self.tb_writer.flush()
            info_record = [self.steps, loss, a_loss, c_loss, norm_, self.contribution]
            print(f"\033[33mstep: {self.steps}, loss: {loss:.6f}, a_loss: {a_loss:.6f}, c_loss:{c_loss:.6f}, grad_norm: {norm_:.6f}\033[0m")
            df = pd.DataFrame([info_record])
            df.to_csv(
                os.path.join(self.log_dir, f'../progress_train_{self.id}.csv'),
                mode='a', header=False, index=False
            )
        if self.id == 0:
            self.leader.add_grad(msg['id'], msg['info'])
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect((self.ip, self.port))
                self.send_message(msg, client)
                self.get_message(client)
    
    def run(self):
        self.init_model()
        while not rospy.is_shutdown():
            if self.done or all([
                len(self.buffer_s) >= self.args.buffer_length,
                len(self.buffer_map) >= self.args.buffer_length,
                len(self.buffer_a) >= self.args.buffer_length,
                len(self.buffer_r) >= self.args.buffer_length,
            ]):
            # if self.done:
                self.steps += 1
                # p = mp.Process(
                #     target=self.train_and_update,
                #     args=(
                #         self.buffer_s, self.buffer_map, self.buffer_a, self.buffer_r, self.buffer_mask,
                #         self.done, self.normalize_ps_values_of_state(self.next_s),
                #         self.next_map, self.args.gamma, self.optimizer
                #     )
                # )
                # p.start()
                self.train_and_update(
                    self.buffer_s, self.buffer_map, self.buffer_a, self.buffer_r, self.buffer_mask,
                    self.done, self.normalize_ps_values_of_state(self.next_s),
                    self.next_map, self.args.gamma, self.optimizer
                )
                self.done = False
                self.buffer_s = []
                self.buffer_map = []
                self.buffer_a = []
                self.buffer_r = []
                self.buffer_mask = []
                self.contribution = 0
                # self.buffer_done = []
                
            time.sleep(0.1)
        print("rospy is shutdown")


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(path, ".."))
    from config import parser
    args_ = sys.argv[1:]
    args, unknown = parser.parse_known_args(args_)
    np.random.seed(args.seed + args.id)
    torch.manual_seed(args.seed + args.id)
    torch.cuda.manual_seed_all(args.seed + args.id)
    worker = Worker(args)
    worker.run()