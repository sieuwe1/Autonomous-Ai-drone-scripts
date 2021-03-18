import sys, time
sys.path.insert(1, 'modules')

import drone

#config 
height = 10
speed = 3 #m/s
size = 165 #10 meter  
#end config

drone.connect_drone('/dev/ttyACM0')
#drone.connect_drone('127.0.0.1:14551')

drone.arm_and_takeoff(height)

time.sleep(5)

#fly recangle with XYZ
for i in range(size):
    drone.send_movement_command_XYZ(speed,0,0)
    time.sleep(0.02)

time.sleep(5)

for i in range(size):
    drone.send_movement_command_XYZ(0,speed,0)
    time.sleep(0.02)

time.sleep(5)

for i in range(size):
    drone.send_movement_command_XYZ(-speed,0,0)
    time.sleep(0.02)

time.sleep(5)

for i in range(size):
    drone.send_movement_command_XYZ(0,-speed,0)
    time.sleep(0.02)

time.sleep(5)

#fly rectangle with YAW 

for i in range(4):
    if i > 0:
        drone.send_movement_command_YAW(90)

    for i in range(size):
        drone.send_movement_command_XYZ(speed,0,0)
        time.sleep(0.02)

    time.sleep(5)

drone.land()