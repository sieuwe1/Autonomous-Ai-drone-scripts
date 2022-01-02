import cv2
import json
import time 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
folder = "/home/drone/Desktop/dataset_POC/Testing/Run1" #input("Please type root direction of data folder: ")
data_log = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/data_log.txt"
preds_log = "predictions_log.txt"
mapped_preds_log = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/mapped_predictions_log.txt"
camera_log_dir = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/camera_log/"

playback_speed = 0.75 #ms

count = 1
plot_data = []

fig, ax = plt.subplots()

def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return round(rightMin + (valueScaled * rightSpan))

def load_data(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()

    data = []

    line_count = 0

    for line in Lines:
        if line_count > 0:
            section_count = 0
            sections = line.split(',') 
            #print(sections)
            
            if line_count == 1:
                for i in range(len(sections)):
                    data.append([])

            for section in sections:
                if '\n' in section:
                    section = section[0:-1]
                point = float(section)
                if point != 0:
                    data[section_count].append(point)
                
                section_count+=1

        line_count+=1
    print("shape data: ",len(data))
    return data

def drawUI(img, data, controls):
    #print(data)
    #print(controls)
    cv2.putText(img, "TargetDistance: " + str(data[0]), (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA)     
    cv2.putText(img, "PathDistance: " + str(data[1]), (50,100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "HeadingDelta: " + str(data[2]), (50,150), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "Speed: " + str(data[3]), (50,200), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "altitude: " + str(data[4]), (50,250), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 

    cv2.putText(img, "x: " + str(data[5]), (1000,150), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "y: " + str(data[6]), (1000,200), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "z: " + str(data[7]), (1000,250), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 


    cv2.rectangle(img, (50, 500), (250, 700), (255,0,0), 2)
    cv2.rectangle(img, (1000, 500), (1200, 700), (255,0,0), 2)

    throttle = map(controls[3], 1000, 2000, 500, 700)
    yaw = map(controls[2], 1000, 2000, 50, 250)

    pitch = map(controls[1], 1000, 2000, 500, 700)
    roll = map(controls[0], 1000, 2000, 1000, 1200)

    cv2.circle(img,(yaw,throttle),20,(0,0,255),-1)
    cv2.circle(img,(roll,pitch),20,(0,0,255),-1)

    return img

def plot(data):
    plot_data.append(data)
    ax.cla()
    ax.plot(plot_data)

controls_log = np.transpose(load_data(mapped_preds_log))
data_log = np.transpose(load_data(data_log))

print("data log shape!!!!!!!!!!!!!!! " + str(len(data_log[0])))

def main(i):
    global count
    img = cv2.imread(camera_log_dir + str(count) + '_cam-image.jpg')

    #@print(data_log[count])
    img = drawUI(img, data_log[count], controls_log[count])

    cv2.imshow("data visualizer", img)
    cv2.waitKey(1)
    #time.sleep(playback_speed)
    plot(controls_log[count])
    count +=1

ani = FuncAnimation(fig, main, interval=playback_speed)

plt.show()


# while True:

#     f = open(folder + "/control/record_" + str(count) + ".json")
#     data = json.load(f)
#     img = cv2.imread(folder + "/left_camera/" + data['cam/image_name'])

#     img = drawUI(img, data)

#     cv2.imshow("data visualizer", img)
#     cv2.waitKey(1)
#     time.sleep(playback_speed)
#     count +=1