import rclpy
import time
import threading
import multiprocessing
import os
from swarm_frame.EpuckNode import EpuckNode
from swarm_frame.BehaviorTree import NodeState, Rotate, Away, Close, Patrol, Selector, Sequence
     
class CheckAway():
    def __init__(self, start_time):
        self.start_time = start_time
        
    def run(self):
        current_time = time.time()  
        if current_time - self.start_time > 10 and current_time - self.start_time <20:                  
            return NodeState.SUCCESS
        else:  
            return NodeState.FAILURE
			
class CheckClose():
    def __init__(self, start_time):
        self.start_time = start_time
        
    def run(self):
        current_time = time.time()  
        if current_time - self.start_time < 5:                  
            return NodeState.SUCCESS
        else:  
            return NodeState.FAILURE
			

def spin_thread(node):
    rclpy.spin(node)

def robot_logic(robot_id):
    rclpy.init()
    robot_node = EpuckNode(f'epuck_{robot_id}')
    robot_node.start_device()
    
    start_time = time.time()

    root = Selector([
        Sequence([CheckAway(start_time), Away(robot_node)]),
        Sequence([CheckClose(start_time), Close(robot_node)]),
        Patrol(robot_node)
    ])

    spin_t = threading.Thread(target=spin_thread, args=(robot_node,))
    spin_t.start()

    try:
        while True:
            root.run()
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass  

    finally:
        robot_node.destroy_node()
        rclpy.shutdown()
        spin_t.join()

def main():
    num_robots = 3  
    threads = []

    for i in range(num_robots):
        os.environ["WEBOTS_CONTROLLER_URL"] = f"ipc://1234/epuck_{i}"
        t = multiprocessing.Process(target=robot_logic, args=(i,))
        t.start()
        threads.append(t)
        time.sleep(0.5)
    try:
        for t in threads:
            t.join()  

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()


