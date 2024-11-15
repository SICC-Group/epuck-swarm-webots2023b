import socket
import json
import time
import datetime
import traceback

from model import Model


class Server(object):
    def __init__(self, port=9801) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = "172.18.0.1"
        self.port = port
        self.server.bind((self.ip, self.port))
        self.server.listen(6)
        self.parameters = []
        self.version = 0

    def accept_new_connection(self):
        print(f"Waiting for clients, my ip is {self.ip} and my port is {self.port}")
        self.clientsocket, self.address = self.server.accept()
        print(f"connection from {self.address} has been established!")
    
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
            # while len(self.parameters) == 0:
            #     time.sleep(0.1)
            #     print("wait for the initialization of the parameters")
            print(f"==== sending the latest model to the worker {full_data['id']}")
            return self.parameters, self.version
        
        if len(self.parameters) == 0 and 'parameters' in full_data['info']:
            self.parameters = full_data['info']['parameters']
            print(f"parameters initialized by worker {full_data['id']}")
            return None, self.version
        else:
            if 'lr' in full_data['info']:
                lr = full_data['info']['lr']
                for i in range(len(self.parameters)):
                    self.parameters[i] -= lr * full_data['info']['gradients'][i]
                now = datetime.datetime.now()
                print(f"At {now.strftime('%H:%M:%S')} parameters updated by worker {full_data['id']}")
                self.version += 1
                return self.parameters, self.version
            else:
                print(f"`lr` is not in the msg, send the latest params to worker {full_data['id']}")
                return self.parameters, self.version
    
    def send_message(self, message):
        message = json.dumps(message) + "<END>"
        self.clientsocket.sendall(message.encode())


if __name__ == "__main__":
    server = Server()
    try:
        while True:
            server.accept_new_connection()
            data, version = server.get_message()
            if data is not None:
                params_to_worker = {
                    "parameters": data,
                    "version": version
                }
                server.send_message(params_to_worker)
            else:
                server.send_message("initialized by you")
            time.sleep(0.1)
    except Exception as e:
        print(traceback.format_exc())