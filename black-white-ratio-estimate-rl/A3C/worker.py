import socket
import json

import time
import random
import sys, os
import traceback
from collections import deque
from itertools import islice

import numpy as np
import torch
import rospy
import pandas as pd
from std_msgs.msg import String
from tensorboard_logger import configure, log_value

from model import Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


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
        self.steps = 0
        self.model = Model(
            11, 4, self.args.col, self.args.row, device,
            self.args.grad_norm, self.args.reward_normalize
        )
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.buffer_s, self.next_s = [], None
        self.buffer_map, self.next_map = [], None
        self.buffer_a = []
        self.buffer_r = []
        # self.buffer_done = []
        self.done = False
        self.eval_sync = False
        self.save_dir = None
        self.log_dir = None

        self.publisher = rospy.Publisher(f'action_{self.id}', String, queue_size=10)
        self.subscriber_state = rospy.Subscriber(f'state_agent_{self.id}', String, self.state_callback)
        self.subscriber_reward = rospy.Subscriber(f'reward_agent_{self.id}', String, self.reward_callback)
    
    def state_callback(self, data):
        # try:
        agent_data = json.loads(data.data)
        if self.save_dir is None:
            self.save_dir = agent_data['save_dir']
        if self.log_dir is None:
            self.log_dir = agent_data['log_dir']
            tb_dir = os.path.join(self.log_dir, f"worker_{self.id}_train_info")
            if not os.path.exists(tb_dir):
                os.makedirs(tb_dir)
            configure(tb_dir)
            self.tb_logger = log_value
        if agent_data['phase'] == 'eval':
            # sync with the global model
            if not self.eval_sync:
                msg = {
                    "id": self.id,
                    "info": "access the latest model"
                }
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                    client.connect((self.ip, self.port))
                    self.send_message(msg, client)
                    model_info = self.get_message(client)
                self.model.load_serializable_state_list(model_info["parameters"])
                print(f"sync with the global model in eval phase - version {model_info['version']}")
                # print(f"len of the buffer_s: {len(self.buffer_s)}")
                self.eval_sync = True
                if self.id == 0:
                    torch.save(self.model, self.save_dir + f'/version_{model_info["version"]}.pt')
            
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
            if front_obstacle:
                action = 3
            elif right_obstacle:
                action = 1
            elif left_obstacle:
                action = 2
            elif back_obstacle:
                action = 0
            else:
                state = self.normalize_ps_values_of_state(agent_data['state'])
                with torch.no_grad():
                    action = self.model.choose_action(
                        torch.tensor(state, dtype=torch.float, device=device),
                        torch.tensor(agent_data['map'], dtype=torch.float, device=device)
                    )
            message = json.dumps(action)
            # rospy.loginfo(f'Publishing action for agent_{self.id}: {message}')
            self.publisher.publish(message)
        
        elif agent_data['phase'] == 'train':
            self.eval_sync = False
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
        # except Exception as e:
        #     print(traceback.format_exc())
    
    def reward_callback(self, data):
        agent_data = json.loads(data.data)
        # rospy.loginfo(f"reward info: {agent_data['reward']}")
        if agent_data['phase'] == 'train':
            self.buffer_r.append(agent_data['reward'])
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
        return full_data

    def send_message(self, message, connection):
        message = json.dumps(message) + "<END>"
        connection.sendall(message.encode())

    def init_model(self):
        if self.id == 0:
            model_info = self.model.get_serializable_state_list(to_list=True)
            msg = {
                "id": self.id,
                "info": {"parameters": model_info}
            }
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect((self.ip, self.port))
                self.send_message(msg, client)
                test_info = self.get_message(client)
            if isinstance(test_info, dict):
                self.model.load_serializable_state_list(test_info["parameters"])
                print("server model in not NONE, replaced by the latest model")
            elif isinstance(test_info, str):
                print(f"===== {test_info} =====")
        time.sleep(2.0)
        if self.id > 0:
            msg = {
                "id": self.id,
                "info": "access the latest model"
            }
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect((self.ip, self.port))
                self.send_message(msg, client)
                model_info = self.get_message(client)

            self.model.load_serializable_state_list(model_info["parameters"])
            print(f"access the latest model - version {model_info['version']}")
    
    def normalize_ps_values_of_state(self, s):
        ps_values = s[-8:]
        ps_values_normalized = (np.array(ps_values) - 55) / (max(np.max(ps_values), 450) - 55)
        s[-8:] = ps_values_normalized
        return s
    
    def run(self):
        self.init_model()
        # try:
        while not rospy.is_shutdown():
            if self.done or all([
                len(self.buffer_s) >= self.args.episode_length,
                len(self.buffer_map) >= self.args.episode_length,
                len(self.buffer_a) >= self.args.episode_length,
                len(self.buffer_r) >= self.args.episode_length,
            ]):
                print("done info: ", self.done)
                self.steps += 1
                # train
                grad, loss = self.model.train_and_get_grad(
                    self.buffer_s, self.buffer_map, self.buffer_a, self.buffer_r,
                    self.done, self.normalize_ps_values_of_state(self.next_s),
                    self.next_map, self.args.gamma, self.optimizer
                )
                self.done = False
                self.buffer_s = []
                self.buffer_map = []
                self.buffer_a = []
                self.buffer_r = []
                # self.buffer_done = []
                msg = {
                    "id": self.id,
                    "info": {
                        "gradients": grad,
                        "lr": self.args.lr
                    }
                }
                self.tb_logger(f"loss_{self.id}", loss, self.steps)
                norm_ = np.linalg.norm(grad)
                info_record = [self.steps, loss, norm_]
                print(f"step: {self.steps}, loss: {loss}, grad_norm: {norm_}")
                df = pd.DataFrame([info_record])
                df.to_csv(
                    os.path.join(self.log_dir, f'../progress_train_{self.id}.csv'),
                    mode='a', header=False, index=False
                )
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                    client.connect((self.ip, self.port))
                    self.send_message(msg, client)
                    model_info = self.get_message(client)
                self.model.load_serializable_state_list(model_info["parameters"])
                print(f"access the latest model - version {model_info['version']}")
            time.sleep(0.1)
        print("rospy is shutdown")
        # except Exception as e:
        #     print(traceback.format_exc())


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