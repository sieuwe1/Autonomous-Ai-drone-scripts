from simple_pid import PID
import time
import logging
import drone

movementJawAngle = 0
inputValue = 0
debug_enable = True
control_loop_active = True

# PID_Config Yaw

pid = PID(0.03, 0, 0, setpoint=0)  #I = 0.001
pid.output_limits = (-15, 15)
p, i, d = pid.components  # The separate terms are now in p, i, d

# end PID_Config_Yaw

debug_file = open("run2.txt", "a")
debug_file.write("P: I: D: Error: command:\n")

# Logging_config
def debug_writer():
    global debug_file
    debug_file.write(str(p) + "," + str(i) + "," + str(d) + "," + str(inputValue) + "," + str(movementJawAngle) + "\n")

# end Logging_Config

def close_control_loop():
    global control_loop_active, debug_file
    control_loop_active = False
    debug_file.close()

def getMovementJawAngle():
    global movementJawAngle
    return movementJawAngle

def setXdelta(XDelta):
    global inputValue
    inputValue = XDelta

def main():
    global movementJawAngle
    while control_loop_active:
        movementJawAngle = pid(inputValue)

        drone.send_movement_command_YAW(movementJawAngle)

        if debug_enable == True:
            debug_writer()
        time.sleep(1/20)
