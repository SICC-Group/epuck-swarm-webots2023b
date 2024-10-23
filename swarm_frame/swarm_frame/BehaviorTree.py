from swarm_frame.Behavior import Behavior
from geometry_msgs.msg import Twist
from enum import Enum

class NodeState(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3

class Rotate():
    def __init__(self, children):
        self.children = children

    def run(self):
        print('Rotate')
        msg = Twist()
        msg.angular.z = 3.0
        self.children.cmd_publisher.publish(msg)
        return NodeState.SUCCESS

class Away():
    def __init__(self, children):
        self.children = children 

    def run(self):
        print('Away')
        if self.children.timer_name == 'dispersion_pattern':
            return NodeState.SUCCESS
        else:
            for timer in self.children.pattern_timers:
                timer.cancel()
            self.children.pattern_timers.clear()
            beh = Behavior(self.children)
            beh.dispersion_pattern()
            return NodeState.SUCCESS

class Close():
    def __init__(self, children):
        self.children = children

    def run(self):
        print('Close')
        if self.children.timer_name == 'attraction_pattern':
            return NodeState.SUCCESS
        else:
            for timer in self.children.pattern_timers:
                timer.cancel()
            self.children.pattern_timers.clear()
            beh = Behavior(self.children)
            beh.attraction_pattern()
            return NodeState.SUCCESS
        
class Patrol():
    def __init__(self, children):
        self.children = children     

    def run(self):
        print('Patrol')
        if self.children.timer_name == 'random_pattern':
            return NodeState.SUCCESS
        else:
            for timer in self.children.pattern_timers:
                timer.cancel()
            self.children.pattern_timers.clear()
            beh = Behavior(self.children)
            beh.random_pattern()  
            return NodeState.SUCCESS  
        
class Selector():
    def __init__(self, children):
        self.children = children

    def run(self):
        for child in self.children:
            result = child.run()
            if result == NodeState.FAILURE:
                continue
            elif result == NodeState.SUCCESS:
                self.state = NodeState.SUCCESS
                return self.state
            elif result == NodeState.RUNNING:
                self.state = NodeState.RUNNING
                return self.state
            
        self.state = NodeState.FAILURE
        return self.state

class Sequence():
    def __init__(self, children):
        self.children = children
        
    def run(self):
        for child in self.children:
            result = child.run()
            if result == NodeState.SUCCESS:
               continue
            elif result == NodeState.RUNNING:
                self.state = NodeState.RUNNING
                return self.state
            elif result == NodeState.FAILURE:
                self.state = NodeState.FAILURE
                return self.state
            
        self.state = NodeState.SUCCESS
        return self.state