import rclpy
import time
import threading
from swarm_frame.EpuckNode import EpuckNode
from swarm_frame.BehaviorTree import NodeState, Rotate, Away, Close, Patrol, Selector, Sequence
from swarm_behavior.interpolation import interpolate_lookup_table

class CheckAway():
    def __init__(self, children):
        self.children = children
        
    def run(self):
        tof_dist = interpolate_lookup_table(self.children.tof_sensor.getValue(), self.children.tof_sensor.getLookupTable())
        if tof_dist> 0.04 and tof_dist < 0.8 :                  
            return NodeState.SUCCESS
        else:  
            return NodeState.FAILURE
			
class CheckClose():
    def __init__(self, children):
        self.children = children
        
    def run(self):
        tof_dist = interpolate_lookup_table(self.children.tof_sensor.getValue(), self.children.tof_sensor.getLookupTable())
        if tof_dist > 1.0 and tof_dist < 1.5 :                  
            return NodeState.SUCCESS
        else:  
            return NodeState.FAILURE

def spin_thread(node):
    rclpy.spin(node)

def main():
    rclpy.init()

    robot1 = EpuckNode('epuck')
    robot1.start_device()

    root = Selector([
        Sequence([CheckAway(robot1), Away(robot1)]),
        Sequence([CheckClose(robot1), Close(robot1)]),
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


