#plot data from auto mission
from matplotlib import lines
import matplotlib.pyplot as plt
import collections

file1 = open('/home/sieuwe/drone/Autonomous-Ai-drone-scripts/predictions_log.txt', 'r')
Lines = file1.readlines()

moving_averages = []

for i in range(4):
    moving_averages.append(collections.deque(maxlen=8))

def average(lst):
    return sum(lst) / len(lst)

def get_num(data):
    num = ""
    for c in data:
        if c == '.' or c.isdigit():
            num = num + c
    return float(num)

sections = Lines[0].split('dtype') 
filtered = [x.replace('float32', '') for x in sections]
data = []
for i in range(4,len(filtered)-3,4):

    roll = get_num(filtered[i-4])
    pitch = get_num(filtered[i-3])
    yaw = get_num(filtered[i-2])
    throttle = get_num(filtered[i-1])

    data.append((roll, pitch, yaw, throttle))

data2 = []

for d in data:
    moving_averages[0].append(d[0]) 
    moving_averages[1].append(d[1]) 
    moving_averages[2].append(d[2])
    moving_averages[3].append(d[3]) 

    data2.append((average(moving_averages[0]), average(moving_averages[1]), average(moving_averages[2]), average(moving_averages[3])))

plt.plot(data2)
plt.ylabel('some numbers')
plt.legend(['roll', 'pitch', 'yaw', 'throttle'])
plt.show()
