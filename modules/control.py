from logging import debug
from modules import drone
from simple_pid import PID
import time

USE_PID_YAW = True
USE_PID_ROLL = False

MAX_SPEED = 3       # M / s
MAX_YAW = 15        # Degrees / s 

P_YAW = 0.015
I_YAW = 0
D_YAW = 0

P_ROLL = 0.15
I_ROLL = 0
D_ROLL = 0

control_loop_active = True
pidYaw = None
pidRoll = None
movementYawAngle = 0
inputValueYaw = 0
inputValueVelocityX = 0
velocityXCommand = 0
control_loop_active = True

debug_yaw = None
debug_velocity = None


def configure_PID(control):
    """ Creates a new PID object depending on whether or not the PID or P is used """ 
    if control == "PID":
        pidYaw = PID(P_YAW, I_YAW, D_YAW, setpoint=0)       # I = 0.001
        pidYaw.output_limits = (-MAX_YAW, MAX_YAW)          # PID Range
        pidRoll = PID(P_ROLL, I_ROLL, D_ROLL, setpoint=0)   # I = 0.001
        pidRoll.output_limits = (-MAX_SPEED, MAX_SPEED)     # PID Range
    elif control == "P":
        pidYaw = PID(P_YAW, 0, 0, setpoint=0)               # I = 0.001
        pidYaw.output_limits = (-MAX_YAW, MAX_YAW)          # PID Range
        pidRoll = PID(P_ROLL, 0, 0, setpoint=0)             # I = 0.001
        pidRoll.output_limits = (-MAX_SPEED, MAX_SPEED)     # PID Range

def connect_drone(drone_location):
    drone.connect_drone(drone_location) #'/dev/ttyACM0'

def getMovementYawAngle():
    return movementYawAngle

def setXdelta(XDelta):
    global inputValueYaw
    inputValueYaw = XDelta

def getMovementVelocityXCommand():
    return velocityXCommand

def setZDelta(ZDelta):
    global inputValueVelocityX
    inputValueVelocityX = ZDelta

def set_system_state(current_state):
    global state
    state = current_state
# end control functions

#drone functions
def arm_and_takeoff(max_height):
    drone.arm_and_takeoff(max_height)

def land():
    drone.land()

def print_drone_report():
    print(drone.get_EKF_status())
    print(drone.get_battery_info())
    print(drone.get_version())
#end drone functions

def initialize_debug_logs(DEBUG_FILEPATH):
    global debug_yaw
    global debug_velocity
    debug_yaw = open(DEBUG_FILEPATH + "_yaw.txt", "a")
    debug_yaw.write("P: I: D: Error: command:\n")

    debug_velocity = open(DEBUG_FILEPATH + "_velocity.txt", "a")
    debug_velocity.write("P: I: D: Error: command:\n")

# Logging_config
def debug_writer(value, file):
    file.write(str(0) + "," + str(0) + "," + str(0) + "," + str(inputValueYaw) + "," + str(value) + "\n")

def control_drone():
    global debug_yaw, debug_velocity, movementRollAngle, movementYawAngle
    if inputValueYaw == 0:
        drone.send_movement_command_YAW(0)
    else:
        movementYawAngle = (pidYaw(inputValueYaw) * -1)
        drone.send_movement_command_YAW(movementYawAngle)
        debug_writer(movementYawAngle, debug_yaw)

    if inputValueVelocityX == 0:
        drone.send_movement_command_XYZ(0,0,0)
    else:
        movementRollAngle = pidRoll(inputValueVelocityX)
        drone.send_movement_command_XYZ(movementRollAngle, 0, 0)
        debug_writer(movementRollAngle, debug_velocity)

def stop_drone():
    drone.send_movement_command_YAW(0)
    drone.send_movement_command_XYZ(0,0,0)
    