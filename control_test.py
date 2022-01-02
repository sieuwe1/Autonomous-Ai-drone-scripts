# test control link between drone and jetson nano
import sys
sys.path.insert(1, '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules')
import drone
import time 
import numpy as np
import matplotlib.pyplot as plt
#-------config------
record_button_channel = 6
#-------------------

def read_flight_data():
    file1 = open('/home/drone/Desktop/Autonomous-Ai-drone-scripts/mapped_predictions_log.txt', 'r')
    Lines = file1.readlines()

    data = []

    line_count = 0

    for line in Lines:
        if line_count > 0:
            section_count = 0
            sections = line.split(',') 
            
            if line_count == 1:
                for i in range(len(sections)):
                    data.append([])

            for section in sections:
                if '\n' in section:
                    section = section[0:-1]
                point = section
                if point != 0:
                    data[section_count].append(point)
                
                section_count+=1
        line_count+=1

    return np.transpose(data)

flight_data = read_flight_data()
print("Starting control test!")

#drone.connect_drone('/dev/ttyACM0') 
drone.connect_drone('/dev/ttyACM0', True, 921600) 
#drone.set_param('SR2_RAW_CTRL', 30) #change default override frequency
print("param set to: ", drone.get_param('SR2_RAW_CTRL'))
#drone.connect_drone('/dev/ttyACM0', False, 921600)

while drone.get_mode() != "ALT_HOLD":
    print(drone.get_mode())
    drone.set_flight_mode("ALT_HOLD")

drone.arm()

print("Flight mode: " + str(drone.get_mode()))

results = []
for data in flight_data:

    pwm = int(data[3])
    print("written overwrite: ", pwm)
    drone.set_channel('3', pwm)
    print("readed overwrite: ", drone.get_channel_override('3'))
    print("readed pwm value: ", drone.read_channel(3))
    results.append((drone.get_channel_override('3'), drone.read_channel(3)))
    time.sleep(1/6)


drone.clear_channel('3')
drone.disarm()
drone.disconnect_drone()
print("finished!")

# while True:
#     if drone.read_channel(record_button_channel) < 1500:
        
#         for pwm in range(1000,2000,100): 
#             drone.set_channel('3', pwm)
#             print("written overwrite: ", pwm)
#             print("readed overwrite: ", drone.get_channel_override('3'))
#             print("readed pwm value: ", drone.read_channel(3))
#             results.append((drone.get_channel_override('3'), drone.read_channel(3)))
#             time.sleep(1/12)
#     else: 
#         drone.clear_channel('3')
#         drone.disarm()
#         drone.disconnect_drone()
#         print("finished!")
#         break

plt.plot(results)
plt.legend(['overwrite', 'actual'])
plt.ylabel('some numbers')
plt.show()
