from simple_pid import PID
import logging

max_rotation = 8
x_scalar = max_rotation / 460
movementJawAngle = 0
inputValue = 0
debug = True

# PID_Config Yaw

pid = PID(x_scalar, 0.05, 0, setpoint=0)
pid.output_limits(-15, 15)
pid.sample_time = 0.05
p, i, d = pid.components  # The separate terms are now in p, i, d

# end PID_Config_Yaw


# Logging_config
if debug == True:
    logging.basicConfig(filename='PIDDebug.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.debug('p: ', p)
    logging.debug('i: ', i)
    logging.debug('d: ', d)
    logging.debug('movementJawAngle: ', movementJawAngle)

# end Logging_Config

def getMovementJawAngle():
    global movementJawAngle
    return movementJawAngle

def setXdelta(XDelta):
    global inputValue
    inputValue = XDelta

def main():

    while True:
        movementJawAngle = pid(inputValue)
