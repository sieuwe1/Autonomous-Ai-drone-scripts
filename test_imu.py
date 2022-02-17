# test imu on drone and jetson nano
import sys
sys.path.insert(1, '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules')
import drone
import time

print("Starting imu test!")

drone.connect_drone('/dev/ttyACM0') 

while True:
    velx, vely, velz =  drone.get_velocity()
    print("velx: " + str(velx) + " vely: " + str(vely) + " velz: " + str(velz))
    time.sleep(0.1)

drone.disconnect_drone()
print("finished!")

