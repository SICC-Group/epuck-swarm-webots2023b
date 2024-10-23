from rclpy.node import Node
from controller import Robot
from controller import Supervisor

class Agent(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.robot = Supervisor()

        self.timestep = int(self.robot.getBasicTimeStep())

        self.__timer = self.create_timer(0.001 * self.timestep, self.__timer_callback)


    def step(self):
        self.robot.step(self.timestep)

    def __timer_callback(self):
        self.step()


