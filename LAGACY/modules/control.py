from simple_pid import PID
import time
import drone

control_loop_active = True
#Yaw
max_rotation = 8 #degree
x_scalar = max_rotation / 460

movementJawAngle = 0
inputValueYaw = 0
debug_enableYaw = True
Yaw_PID_Active = True
file_path = ""

#velocity
max_speed = 3 #m/s
z_scalar = max_speed / 10

velocityXCommand = 0
inputValueVelocityX = 0
debug_enableVelocityX = False
Velocity_X_PID_Active = False

# PID_Config Yaw

pidYaw = PID(0.03, 0, 0, setpoint=0)  #I = 0.001
pidYaw.output_limits = (-15, 15)
p, i, d = pidYaw.components  # The separate terms are now in p, i, d

# end PID_Config Yaw


# PID_Config Velocity_X

pidVelocityX = PID(0.03, 0, 0, setpoint=0)  #I = 0.001
pidVelocityX.output_limits = (-1.5, 1.5)  #values to determain
p, i, d = pidVelocityX.components  # The separate terms are now in p, i, d

# end PID_Config Velocity_X

debug_fileYaw = open(file_path + "_yaw.txt", "a")
debug_fileYaw.write("P: I: D: Error: command:\n")

debug_file = open(file_path + "_velocity.txt", "a")
debug_file.write("P: I: D: Error: command:\n")

# Logging_config
def debug_writerYaw():
    global debug_fileYaw
    debug_fileYaw.write(str(p) + "," + str(i) + "," + str(d) + "," + str(inputValueYaw) + "," + str(movementJawAngle) + "\n")

def debug_writerVelocityX():
    global debug_fileVelocityX
    debug_fileVelocityX.write(str(p) + "," + str(i) + "," + str(d) + "," + str(inputValueVelocityX) + "," + str(velocityXCommand) + "\n")

# end Logging_Config

def close_control_loop():
    global control_loop_active, debug_fileYaw, debug_fileVelocityX
    control_loop_active = False
    debug_fileYaw.close()
    debug_fileVelocityX.close()

def getMovementJawAngle():
    global movementJawAngle
    return movementJawAngle

def setXdelta(XDelta):
    global inputValueYaw
    inputValueYaw = XDelta

def getMovementVelocityXCommand():
    global velocityXCommand
    return velocityXCommand

def setZDelta(ZDelta):
    global inputValueVelocityX
    inputValueVelocityX = ZDelta

def main(filename):
    global movementJawAngle, velocityXCommand, file_path

    file_path = filename

    while control_loop_active:

        if Yaw_PID_Active == True:
            movementJawAngle = (pidYaw(inputValueYaw) * -1)
            drone.send_movement_command_YAW(movementJawAngle)

        if debug_enableYaw == True:
            debug_writerYaw()

       # if Yaw_PID_Active == False:
       #     movementJawAngle = inputValueYaw * x_scalar
       #     drone.send_movement_command_YAW(movementJawAngle)

      #  if Velocity_X_PID_Active == True:
      #      velocityXCommand = pidVelocityX(inputValueVelocityX)
      #      drone.send_movement_command_XYZ(velocityXCommand, 0, 0)

       # if debug_enableVelocityX == True:
       #     debug_writerVelocityX()

       # if Velocity_X_PID_Active == False:
       #     velocityXCommand = inputValueVelocityX * z_scalar
       #     drone.send_movement_command_XYZ(velocityXCommand, 0, 0)

        time.sleep(1)
