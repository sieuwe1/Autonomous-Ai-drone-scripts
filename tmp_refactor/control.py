import drone
from simple_pid import PID

USE_PID_YAW = True
USE_PID_ROLL = False

MAX_SPEED = 3 #m/s

P_YAW = 0.03
I_YAW = 0
D_YAW = 0

P_ROLL = 0
I_ROLL = 0
D_ROLL = 0


DEBUG_FILEPATH = ""


control_loop_active = True
pidYaw = None
pidRoll = None
movementJawAngle = 0
inputValueYaw = 0
inputValueVelocityX = 0
velocityXCommand = 0

control_loop_active = True
state = ""

debug_yaw = None
debug_velocity = None


def configure_PID():
    """ Creates a new PID object depending on whether or not the PID or P is used """ 
    if USE_PID_YAW:
        pidYaw = PID(P_YAW, I_YAW, D_YAW, setpoint=0)  #I = 0.001
        pidYaw.output_limits = (-15, 15) # PID Range
    else:
        pidYaw = PID(P_YAW, 0, 0, setpoint=0)  #I = 0.001
        pidYaw.output_limits = (-15, 15) # PID Range

    if USE_PID_ROLL:
        pidRoll = PID(P_ROLL, I_ROLL, D_ROLL, setpoint=0)  #I = 0.001
        pidRoll.output_limits = (-15, 15) # PID Range
    else:
        pidRoll = PID(P_YAW, 0, 0, setpoint=0)  #I = 0.001
        pidRoll.output_limits = (-15, 15) # PID Range

def connect_drone(drone_location):
    drone.connect_drone(drone_location) #'/dev/ttyACM0'

def initialize_debug_logs():
    global debug_yaw
    global debug_velocity
    debug_yaw = open(DEBUG_FILEPATH + "_yaw.txt", "a")
    debug_yaw.write("P: I: D: Error: command:\n")

    debug_file = open(DEBUG_FILEPATH + "_velocity.txt", "a")
    debug_file.write("P: I: D: Error: command:\n")

# Logging_config
def debug_writerYaw():
    global debug_yaw
    debug_yaw.write(str(p) + "," + str(i) + "," + str(d) + "," + str(inputValueYaw) + "," + str(movementJawAngle) + "\n")

def debug_writerVelocityX():
    global debug_velocity
    debug_velocity.write(str(p) + "," + str(i) + "," + str(d) + "," + str(inputValueVelocityX) + "," + str(velocityXCommand) + "\n")

# control functions
def close_control_loop():
    global control_loop_active, debug_yaw, debug_velocity
    control_loop_active = False
    debug_yaw.close()
    debug_velocity.close()

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

def main(filename):
    global movementJawAngle, velocityXCommand, file_path

    file_path = filename

    while control_loop_active:
        
        if state == "track":
            if inputValueYaw != 0:          #only start control if drone and camera is ready

                movementJawAngle = (pidYaw(inputValueYaw) * -1)
                drone.send_movement_command_YAW(movementJawAngle)

                debug_writerYaw()

        elif state == "search":
            start = time.time()
            while time.time() - start < 40:
                if time.time() - start > 10:
                    drone.send_movement_command_YAW(1)

        else:
            print("control waiting for state update")
        
        time.sleep(0.1) # 10hz
