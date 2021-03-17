import sys, time
sys.path.insert(1, 'modules')

import drone

#config 
speed = 3 #m/s
size = 165 #10 meter  
#end config

drone.connect_drone('/dev/ttyACM0')

drone.arm_and_takeoff(10)

time.sleep(4)

#fly recangle with XYZ
for i in range(size)
    drone.send_movement_command_XYZ(speed,0,0)
    time.sleep(0.02)

for i in range(size)
    drone.send_movement_command_XYZ(0,speed,0)
    time.sleep(0.02)

for i in range(size)
    drone.send_movement_command_XYZ(-speed,0,0)
    time.sleep(0.02)

for i in range(size)
    drone.send_movement_command_XYZ(0,-speed,0)
    time.sleep(0.02)

#fly rectangle with YAW 

for i in range(4):
    if i > 0:
        send_movement_command_YAW(90)

    for i in range(size)
        drone.send_movement_command_XYZ(speed,0,0)
        time.sleep(0.02)

drone.land()