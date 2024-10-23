import rclpy
import time
import threading
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

def main():
    rclpy.init()

    robot1 = EpuckNode('epuck')
    robot1.start_device()

    start_time = time.time()

    root = Selector([
        Sequence([CheckAway(start_time), Away(robot1)]),
        Sequence([CheckClose(start_time), Close(robot1)]),
        Patrol(robot1)
        ])

    spin_t = threading.Thread(target=spin_thread, args=(robot1,))
    spin_t.start()

    try:
        while True:
            root.run()
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass  

    finally:
        robot1.destroy_node()
        rclpy.shutdown()
        spin_t.join() 
        
if __name__ == '__main__':
    main()

