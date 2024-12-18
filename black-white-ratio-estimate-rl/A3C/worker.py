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

from server import Server
from model import Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Leader(Server):
    def __init__(self, args, parameters):
        super().__init__()
        self.args = args
        self.shared_array = mp.Array('d', len(parameters))
        self.parameters = np.frombuffer(self.shared_array.get_obj(), dtype=np.float64)
        np.copyto(self.parameters, parameters)
        self.version = mp.Value('i', 0)
        self.last_time = time.time()
        manager = mp.Manager()
        self.buffer_grad = manager.list()  # (contribution and gradient)
        self.lock = mp.Lock()
        
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
            self.add_grad(full_data['info'])
            return None, full_data['id']
    
    def add_grad(self, infos: dict):
        """infos: {"gradients": list, "contribution": float}"""
        with self.lock:
            self.buffer_grad.append(
                (infos['contribution'], infos['gradients'])
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
                    if len(self.buffer_grad) > 0:
                        contributions = np.array([grad[0] for grad in self.buffer_grad])
                        gradients = np.array([grad[1] for grad in self.buffer_grad])
                        weights = self.softmax(contributions)
                        print(weights)
                        aggregated_grad = sum(w * g for w, g in zip(weights, gradients))
                        self.parameters -= self.args.lr * aggregated_grad
                        self.version.value += 1
                        self.buffer_grad[:] = []
                        now = datetime.datetime.now()
                        print(f"aggregated successfully at {now.strftime('%H:%M:%S')} and get version{self.version.value}")
            time.sleep(0.1)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
        

class Worker:
    def __init__(self, args):
        self.id = args.id
        rospy.init_node(f'agent_{self.id}_node', anonymous=True)
        self.args = args
        # self.as_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.as_server.bind(("172.18.0.1", 12255 + self.id))
        # self.as_server.listen()
        # self.server_connection, self.server_address = self.as_server.accept()
        # print("connected to webots successfully")

        self.ip, self.port = "172.18.0.1", 9801
        # self.as_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.as_client.connect(("172.18.0.1", 9801))
        self.last_action = None
        self.steps = 0
        if self.args.model_dir is None:
            self.model = Model(
                11, 4, self.args.col, self.args.row, device,
                self.args.grad_norm_init, self.args.norm_decay_steps,
                self.args.grad_norm_min, self.args.reward_normalize
            )
        else:
            self.model = torch.load(self.args.model_dir)
            print("lode model successfully")
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.buffer_s, self.next_s = [], None
        self.buffer_map, self.next_map = [], None
        self.buffer_a = []
        self.buffer_r = []
        self.contribution = 0
        # self.buffer_done = []
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
        # sync with the global model at the beginning of each episode
        if agent_data['step'] == 1:
            self.sync_with_global_model()
        
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
            if self.last_action == 3:
                action = random.choice([1, 2])  # turn right or left
                self.last_action = None
            else:
                if front_obstacle:
                    action = 3
                    self.last_action = 3
                elif right_obstacle:
                    action = 1
                    self.last_action = None
                elif left_obstacle:
                    action = 2
                    self.last_action = None
                elif back_obstacle:
                    action = 0
                    self.last_action = None
                else:
                    state = self.normalize_ps_values_of_state(agent_data['state'])
                    with torch.no_grad():
                        action = self.model.choose_action(
                            torch.tensor(state, dtype=torch.float, device=device),
                            torch.tensor(agent_data['map'], dtype=torch.float, device=device)
                        )
                    self.last_action = None
            message = json.dumps(action)
            # rospy.loginfo(f'Publishing action for agent_{self.id}: {message}')
            self.publisher.publish(message)
        
        elif agent_data['phase'] == 'train':
            
            state = self.normalize_ps_values_of_state(agent_data['state'])
            self.buffer_s.append(state)
            self.buffer_map.append(agent_data['map'])
            
            with torch.no_grad():
                action = self.model.choose_action(
                    torch.tensor(state, dtype=torch.float, device=device),
                    torch.tensor(agent_data['map'], dtype=torch.float, device=device)
                )
            
            self.buffer_a.append(action)  # action type, should be `int`
            message = json.dumps(action)
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
            a = mp.Process(target=self.leader.aggregate)
            a.start()
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
        ps_values = s[-8:]
        ps_values_normalized = (np.array(ps_values) - 55) / (max(np.max(ps_values), 450) - 55)
        s[-8:] = ps_values_normalized
        return s
    
    def train_and_update(self, bs, bmap, ba, br, done, s_, map_, gamma, opt):
        grad, loss = self.model.train_and_get_grad(
            bs, bmap, ba, br, done, s_, map_, gamma, opt, self.steps
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
        self.tb_writer.add_scalar("loss", loss, self.steps)
        self.tb_writer.add_scalar("contributions", self.contribution, self.steps)
        norm_ = np.linalg.norm(grad)
        info_record = [self.steps, loss, norm_, self.contribution]
        print(f"\033[33mstep: {self.steps}, loss: {loss}, grad_norm: {norm_}\033[0m")
        df = pd.DataFrame([info_record])
        df.to_csv(
            os.path.join(self.log_dir, f'../progress_train_{self.id}.csv'),
            mode='a', header=False, index=False
        )
        if self.id == 0:
            self.leader.add_grad(msg['info'])
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
                p = mp.Process(
                    target=self.train_and_update,
                    args=(
                        self.buffer_s, self.buffer_map, self.buffer_a, self.buffer_r,
                        self.done, self.normalize_ps_values_of_state(self.next_s),
                        self.next_map, self.args.gamma, self.optimizer
                    )
                )
                p.start()
                self.done = False
                self.buffer_s = []
                self.buffer_map = []
                self.buffer_a = []
                self.buffer_r = []
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    worker = Worker(args)
    worker.run()