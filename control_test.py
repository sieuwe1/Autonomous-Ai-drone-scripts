# Fly fully autonomous from starting point to target! Dont worry about obstacles our AI should avoid them!
import sys
sys.path.insert(1, '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules')
import drone
import time 
#-------config------
record_button_channel = 6
#-------------------

print("Starting control test!")

drone.connect_drone('/dev/ttyACM0')
#drone.connect_drone('/dev/ttyACM0', False, 921600)

#set SR_CTRL param and read param to check set

drone.arm()

while True:
    if drone.read_channel(record_button_channel) < 1500:
        
        for pwm in range(1000,2000,100): 
            drone.set_channel('3', pwm)
            print("written overwrite: ", pwm)
            print("readed overwrite: ", drone.get_channel_override('3'))
            print("readed pwm value: ", drone.read_channel(3))
            time.sleep(1/12)

        for pwm in range(1900,1000,100): 
            drone.set_channel('3', pwm)
            print("written overwrite: ", pwm)
            print("readed overwrite: ", drone.get_channel_override('3'))
            print("readed pwm value: ", drone.read_channel(3))
            time.sleep(1/12)

    else: 
        #clear overwrites needs to be made!
        drone.disarm()
        drone.disconnect_drone()
        print("finished!")
        break
