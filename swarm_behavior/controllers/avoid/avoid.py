from controller import Robot
import time

robot = Robot()

timestep = int(robot.getBasicTimeStep())

speed = 6.0

lm = robot.getMotor('left_motor')
rm = robot.getMotor('right_motor')

lm.setPosition(float('inf'))
lm.setVelocity(0.0)

rm.setPosition(float('inf'))
rm.setVelocity(0.0)

dist = robot.getDistanceSensor("ds_front")
dist.enable(timestep)

distance_left = robot.getDistanceSensor("ds_left")
distance_left.enable(timestep)

distance_right = robot.getDistanceSensor("ds_right")
distance_right.enable(timestep)

def Left():
    for i in range(5):
        if robot.step(timestep) != -1:
            ls = -speed
            rs = speed
                
            lm.setVelocity(ls)
            rm.setVelocity(rs)

def Right():
    for i in range(5):
        if robot.step(timestep) != -1:
            ls = speed
            rs = -speed
                
            lm.setVelocity(ls)
            rm.setVelocity(rs)

def Forward():
    while True:
        if robot.step(timestep) != -1:
            ls = -speed
            rs = -speed
                
            lm.setVelocity(ls)
            rm.setVelocity(rs)
            
            ds = dist.getValue()
            ds_left = distance_left.getValue()
            ds_right = distance_right.getValue()  
            
        if float(ds)<700.0 or float(ds_left)<200.0 or float(ds_right)<200.0:# or float(ds_left)<1000.0 or float(ds_right)<1000.0:
            break
    
    if ds_left>ds_right:
        return "L"
    return "R"
  
while True:
    direction = Forward()
    if direction == "L":
        Left()
    else:
        Right()
