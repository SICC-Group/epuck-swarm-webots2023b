from controller import Robot

robot = Robot()

timestep = int(robot.getBasicTimeStep())

#获取键盘输入
key = robot.getKeyboard()
key.enable(timestep)

#获取电机
motor_right = robot.getMotor("right_motor")
motor_left = robot.getMotor("left_motor")

#设置电机模式与电机初始速度
motor_right.setPosition(float("inf"))
motor_right.setVelocity(0.0)

motor_left.setPosition(float("inf"))
motor_left.setVelocity(0.0)

while robot.step(timestep) != -1:
    
    key_ = key.getKey()
    
    if key_ == ord('W'):
        motor_right.setVelocity(-7.0)
        motor_left.setVelocity(-7.0)
    if key_ == ord('S'):
        motor_right.setVelocity(7.0)
        motor_left.setVelocity(7.0)
    if key_ == ord('D'):
        motor_right.setVelocity(4.0)
        motor_left.setVelocity(-4.0)
    if key_ == ord('A'):
        motor_right.setVelocity(-4.0)
        motor_left.setVelocity(4.0)
    if key_ == ord(' '):
        motor_right.setVelocity(0.0)
        motor_left.setVelocity(0.0)
