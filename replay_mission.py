# Fly fully autonomous from starting point to target! Dont worry about obstacles our AI should avoid them!
import sys

from modules.drone import get_mode
sys.path.insert(1, '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules')
import drone
import camera
import gps
import cv2
import json
import time 
import numpy as np
import collections
from tensorflow import keras
from keras.applications import imagenet_utils
import network
from adabelief_tf import AdaBeliefOptimizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import onnxruntime as rt
import os
#-------config----------
target_location = (51.4504757, 5.4546918) 
target_reached_bubble = 5 #when vehicle is within 5m range from target it has reached the target
model_dir ="/home/drone/Desktop/Autonomous-Ai-drone-scripts/Tools/Donkeycar.onnx"
data_log = "data_log.txt"
preds_log = "predictions_log.txt"
mapped_preds_log = "mapped_predictions_log.txt"
camera_log_dir = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/camera_log"
moving_average_length = 3
record_button_channel = 6
debug = True
#-----------------------

results = [] #TODO remove !

STATE = "init"
sess = None
inputs = None
moving_averages = []
frame_count = 0

for i in range(4):
    moving_averages.append(collections.deque(maxlen=moving_average_length))

def read_flight_data():
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

    return np.transpose(data)

flight_data = read_flight_data()

def warmup():
    #perform random prediction to warmup system
    global sess, inputs
    data = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    img = camera.get_video(0) 
    predict(img,data)

def scale_z_score(data, mean, std):
    return (data - mean) / std

def map_decimal(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return round(rightMin + (valueScaled * rightSpan),5)

def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return round(rightMin + (valueScaled * rightSpan))

def average(lst):
    return sum(lst) / len(lst)

def write_image(img):
    global frame_count
    cam_name = str(frame_count) + '_cam-image.jpg'
    cam_path = os.path.join(camera_log_dir, cam_name)
    cv2.imwrite(cam_path, img)
    frame_count += 1


def predict(img, data):
    global sess, inputs 
    
    if debug:
        write_image(img)

    # scale image from (0 to 255) to (0 to 1)
    img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    normalizedImg = imagenet_utils.preprocess_input(img, data_format=None, mode='tf') #inception uses 'tf' mode

    # scale data from ranges to (0 to 1)
    speed = map_decimal(data[0],0.038698211312294006, 3.2179622650146484 , 0,1)
    target_distance = map_decimal(data[1], 2.450411854732669, 80.67362590478994, 0,1)
    path_distance = map_decimal(data[2], 0, 13.736849872936324, 0,1)
    heading_delta = map_decimal(data[3], 0.012727423912849645, 155.99795402967004, 0,1)
    altitude = map_decimal(data[4], 0.093, 6.936, 0,1)
    vel_x = map_decimal(data[5], -3.19, 3.03, 0,1)
    vel_y = map_decimal(data[6], -3.03, 1.46, 0,1)
    vel_z = map_decimal(data[7], -1.17, 1.07, 0,1)
    
    #predict
    #data = np.array([speed, target_distance, path_distance, heading_delta, altitude, vel_x, vel_y, vel_z])
    data = np.array(flight_data[frame_count])
    print(data)

    if debug:
        with open(data_log, 'a') as f:
            message = str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "," + str(data[6]) + "," + str(data[7]) +"\n"
            f.write(message)
        f.close()
    
    print("input data: ", data)
    sample_to_predict = [normalizedImg.reshape((1,300,300,3)), data.reshape((1,8))]
    
    preds = sess.run(None, {inputs[0].name: sample_to_predict[0].astype(np.float32), inputs[1].name: sample_to_predict[1].astype(np.float32)})

    if debug:
        with open(preds_log, 'a') as f:
            message = str(preds[0][0][0]) + "," + str(preds[1][0][0]) + "," + str(preds[2][0][0]) + "," + str(preds[3][0][0]) +"\n"
            f.write(message)
        f.close()

    moving_averages[0].append(preds[0][0][0]) 
    moving_averages[1].append(preds[1][0][0]) 
    moving_averages[2].append(preds[2][0][0])
    moving_averages[3].append(preds[3][0][0]) 

    smooth_predicted_roll = average(moving_averages[0])
    smooth_predicted_pitch = average(moving_averages[1])
    smooth_predicted_yaw = average(moving_averages[2])
    smooth_predicted_throttle = average(moving_averages[3])

    #scale predicted (0 to 1) to actual scale (1000 to 2000)
    predicted_roll = map(smooth_predicted_roll, 0, 1, 1000, 2000)
    predicted_pitch = map(smooth_predicted_pitch, 0, 1, 1000, 2000)
    predicted_yaw = map(smooth_predicted_yaw, 0, 1, 1000, 2000)
    predicted_throttle = map(smooth_predicted_throttle, 0, 1, 1000, 2000)

    if debug:
        with open(mapped_preds_log, 'a') as f:
            message = str(predicted_roll) + "," + str(predicted_pitch) + "," + str(predicted_yaw) + "," + str(predicted_throttle) +"\n"
            f.write(message)
        f.close()

    print("/predicted throttle: " +  str(predicted_throttle) + " /predicted yaw: " + str(predicted_yaw) + " /predicted pitch: " + str(predicted_pitch) + " /predicted roll: " + str(predicted_roll))

    return (predicted_roll, predicted_pitch, predicted_yaw, predicted_throttle)

def init():
    global sess, inputs 

    print("State = INIT -> " + STATE)

    time.sleep(20)
    # create onnx session
    sess = rt.InferenceSession(model_dir)
    inputs = sess.get_inputs()

    camera.create_camera(0)
    warmup()

    print("FLYING NOW")
    drone.connect_drone('/dev/ttyACM0')

    drone.set_param('SR2_RAW_CTRL', 30) #change default override frequency

    #drone.arm_and_takeoff(4)
    drone.arm()
    #drone.connect_drone('127.0.0.1:14551')
    return "flight"


def flight():

    print("State = FLIGHT -> " + STATE)

    start_cordinate = drone.get_location()
    start = (start_cordinate.lat,start_cordinate.lon)

    print("start location: " + str(start))
    print("target location: " + str(target_location))

    while True:
        print("Flight mode: " + str(drone.get_mode()))
        if drone.read_channel(record_button_channel) != None:
            if drone.read_channel(record_button_channel) < 1500:
                if drone.get_mode() != "STABILIZE":
                    drone.set_flight_mode("STABILIZE")

                start_time = time.time()
                current_cordinate = drone.get_location()
                current_location = (current_cordinate.lat,current_cordinate.lon)

                target_distance, path_distance = gps.calculate_path_distance(target_location, start, current_location)

                if target_distance < target_reached_bubble: #check if target is reached 
                    break
                
                else:
                    #get data
                    current_heading = drone.get_heading()
                    img = camera.get_video(0)

                    heading_delta = gps.calculate_heading_difference(current_heading, target_location, current_location)
                    vel_x, vel_y, vel_z = drone.get_velocity()
                    altitude = drone.get_altitude()
                    speed = drone.get_ground_speed()
                    data = np.array([speed, target_distance, path_distance, heading_delta, altitude, vel_x, vel_y, vel_z])

                    #predict
                    predicted = predict(img,data)
                    predicted_roll = predicted[0]
                    predicted_pitch = predicted[1]
                    predicted_yaw = predicted[2]
                    predicted_throttle = predicted[3]

                    #write to drone 
                    drone.set_channel('1', predicted_roll)
                    drone.set_channel('2', predicted_pitch)
                    drone.set_channel('3', predicted_throttle)
                    drone.set_channel('4', predicted_yaw)

                    print("written overwrite: ", predicted_throttle)
                    print("readed overwrite: ", drone.get_channel_override('3'))
                    print("readed pwm value: ", drone.read_channel(3))
                    results.append((drone.get_channel_override('3'), drone.read_channel(3)))

                end_time = time.time()
                print("FPS: " + str(1/(end_time-start_time)))

#Maak systeem om overrides te plotten om te kijken of de overrides optijs geschreven worden

#Maak mode die de hoogt uit zichzelf behoud

#Maak mobilenet met onnx

#misschien arduiono die pwm signalen stuurt met usb verbinding aan jetson

            else: 
                drone.clear_channel('1')
                drone.clear_channel('2')
                drone.clear_channel('3')
                drone.clear_channel('4')
                if drone.get_mode() != "LOITER":
                    drone.set_flight_mode("LOITER")
                print("PAUSED")
                print("/current throttle: ", str(drone.read_channel(3)) + " /current yaw: ", str(drone.read_channel(4)) + " /current pitch: ", str(drone.read_channel(2)) + " /current roll: ", str(drone.read_channel(1)))
                
                plt.plot(results)
                plt.legend(['overwrite', 'actual'])
                plt.ylabel('some numbers')
                plt.show()

                time.sleep(0.1)
        else:
            print("record button not working!")

    return "goal_reached"


def goal_reached():
    global frame_count
    print("State = GOAL_REACHED -> " + STATE)
    print("Goal reached! Landing in 20 seconds!")
    drone.set_flight_mode("GUIDED")

    drone.clear_channel('1')
    drone.clear_channel('2')
    drone.clear_channel('3')
    drone.clear_channel('4')
    
    time.sleep(20)
    drone.land()

    return "shutdown"

def shutdown():
    print("State = SHUTDOWN -> " + STATE)
    camera.close_cameras()
    sys.exit(0)


while True:
    # main program loop

    if STATE == "init":
        STATE = init()

    elif STATE == "flight":
        STATE = flight()

    elif STATE == "goal_reached":
        STATE = goal_reached()

    elif STATE == "shutdown":
        STATE = shutdown()
