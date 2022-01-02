#plot PID data drom person follow script
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.arrayprint import DatetimeFormat

#file1 = open('/home/drone/Desktop/Autonomous-Ai-drone-scripts/predictions_log.txt', 'r')
file1 = open('/home/drone/Desktop/Autonomous-Ai-drone-scripts/data_log.txt', 'r')
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

#-------use this when plotting auto flight data-------
np_data = np.transpose(data)
print(np_data[:,3])
plt.plot(np_data)
#plt.legend(['roll', 'pitch', 'yaw', 'throttle'])
plt.legend(['speed', 'target_distance', 'path_distance', 'heading_delta', 'altitude', 'vel_x', 'vel_y', 'vel_z'])
#-----------------------------------------------------

#plt.plot(data)
plt.ylabel('some numbers')
plt.show()
