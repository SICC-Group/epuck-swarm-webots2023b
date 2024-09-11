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
from std_msgs.msg import String

from model import Model

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
        
        self.model = Model(11, 4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.buffer_s = deque()  # avail info in [:-1]
        self.buffer_a = deque()  # avail info in [1:]
        self.buffer_r = deque()  # avail info in [:-1]
        self.buffer_info = deque()
        self.done = False
        self.eval_sync = False
        self.save_dir = None

        self.publisher = rospy.Publisher(f'action_{self.id}', String, queue_size=10)
        self.subscriber = rospy.Subscriber(f'agent_{self.id}', String, self.state_callback)
    
    def state_callback(self, data):
        try:
            agent_data = json.loads(data.data)
            if self.save_dir is None:
                self.save_dir = agent_data['save_dir']
            if agent_data['phase'] == 'eval':
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
                    print(f"len of the buffer_s: {len(self.buffer_s)}")
                    self.eval_sync = True
                    if self.id == 0:
                        torch.save(self.model, self.save_dir + f'/version_{model_info["version"]}.pt')
                with torch.no_grad():
                    action = self.model.choose_action(
                        torch.tensor(agent_data['state'], dtype=torch.float32)
                    )
                message = json.dumps(action)
                # rospy.loginfo(f'Publishing action for agent_{self.id}: {message}')
                self.publisher.publish(message)
            
            elif agent_data['phase'] == 'train':
                self.eval_sync = False
                if agent_data['done']:
                    self.buffer_s.append(agent_data['state'])
                    self.buffer_r.append(agent_data['reward'])
                    self.buffer_info.append(agent_data['info'])
                    self.buffer_a.append(None)
                    self.done = agent_data['done']
                    return
                
                if agent_data['info'] == 'none':
                    with torch.no_grad():
                        action = self.model.choose_action(
                            torch.tensor(agent_data['state'], dtype=torch.float32)
                        )
                elif agent_data['info'] == "front":
                    action = 3  # backward - 5,3
                elif agent_data['info'] == "right":
                    action = 1  # turn left - 3,2
                elif agent_data['info'] == "left":
                    action = 2  # turn right - 4,2
                elif agent_data['info'] == "back":
                    action = 0
                
                self.buffer_a.append(action)  # action type, should be `int`
                self.buffer_s.append(agent_data['state'])
                self.buffer_r.append(agent_data['reward'])
                self.buffer_info.append(agent_data['info'])
                message = json.dumps(action)
                # rospy.loginfo(f'Publishing action for agent_{self.id}: {message}')
                self.publisher.publish(message)
            
            else:
                if 'phase' not in agent_data:
                    print(f"none phase info")
                else:
                    print(f"unknown phase: {agent_data['phase']}")
        except Exception as e:
            print(traceback.format_exc())
    
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
    
    def run(self):
        self.init_model()
        try:
            # 获取来自webots-supervisor的数据
            while not rospy.is_shutdown():
                if self.done or all([
                    len(self.buffer_s) > self.args.buffer_length + 1,
                    len(self.buffer_a) > self.args.buffer_length + 1,
                    len(self.buffer_r) > self.args.buffer_length + 1
                ]):
                    print("done info: ", self.done)
                    # collect the buffer data
                    buffer_s = list(islice(self.buffer_s, self.args.buffer_length))
                    buffer_a = list(islice(self.buffer_a, self.args.buffer_length))
                    buffer_r = list(islice(self.buffer_r, self.args.buffer_length))
                    # update the buffer data
                    self.buffer_s = deque(islice(self.buffer_s, self.args.buffer_length, None))
                    self.buffer_a = deque(islice(self.buffer_a, self.args.buffer_length, None))
                    self.buffer_r = deque(islice(self.buffer_r, self.args.buffer_length, None))
                    # train
                    grad = self.model.train_and_get_grad(
                        buffer_s[:-1], buffer_a[:-1], buffer_r[1:], self.done,
                        buffer_s[-1], self.args.gamma, self.optimizer
                    )
                    self.done = False
                    msg = {
                        "id": self.id,
                        "info": {
                            "gradients": grad,
                            "lr": self.args.lr
                        }
                    }
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                        client.connect((self.ip, self.port))
                        self.send_message(msg, client)
                        model_info = self.get_message(client)
                    self.model.load_serializable_state_list(model_info["parameters"])
                    print(f"access the latest model - version {model_info['version']}")
                time.sleep(0.1)
            print("rospy is shutdown")
        except Exception as e:
            print(traceback.format_exc())


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