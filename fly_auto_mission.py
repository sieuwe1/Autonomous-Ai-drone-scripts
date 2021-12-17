# Fly fully autonomous from starting point to target! Dont worry about obstacles our AI should avoid them!
import sys
sys.path.insert(1, 'modules')
import drone
import camera
import gps

import cv2
import json
import time 
import numpy as np
from tensorflow import keras
from keras.applications import imagenet_utils
import network
from adabelief_tf import AdaBeliefOptimizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

target_location = () 
target_reached_bubble = 5 #when vehicle is within 5m range from target it has reached the target
front_camera = None

#model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules/trained_best_model_full_set.h5'
model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/sets/full_set_shuffle_linear/Inception_new_preprocessor_shuffled_occi_linear_in_linear_out_sigmoid_lr_0.001.h5'
model = None

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

def predict(img, data):

    # scale image from (0 to 255) to (0 to 1)
    img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    normalizedImg = imagenet_utils.preprocess_input(img, data_format=None, mode='tf') #inception uses 'tf' mode

    # scale data from ranges to (0 to 1)
    speed = map_decimal(data[0],0.0, 4.58 , 0,1)
    target_distance = map_decimal(data[1], 0.88, 90.37, 0,1)
    path_distance = map_decimal(data[2], 0, 18.71, 0,1)
    heading_delta = map_decimal(data[3], 0.0025, 178.60, 0,1)
    altitude = map_decimal(data[4], 0.093, 7.285, 0,1)
    vel_x = map_decimal(data[5], -3.19, 3.45, 0,1)
    vel_y = map_decimal(data[6], -4.2, 3.23, 0,1)
    vel_z = map_decimal(data[7], -1.17, 1.22, 0,1)
    
    #predict
    data = np.array([speed, target_distance, path_distance, heading_delta, altitude, vel_x, vel_y, vel_z])
    print("input data: ", data)
    sample_to_predict = [normalizedImg.reshape((1,300,300,3)), data.reshape((1,8))]
    preds = model.predict(sample_to_predict)

    #scale predicted (0 to 1) to actual scale (1000 to 2000)
    predicted_throttle = map(predicted[3], 0, 1, 1000, 2000)
    predicted_yaw = map(predicted[2], 0, 1, 1000, 2000)
    predicted_pitch = map(predicted[1], 0, 1, 1000, 2000)
    predicted_roll = map(predicted[0], 0, 1, 1000, 2000)

    print("predicted throttle: ", predicted_throttle)
    print("predicted yaw: ", predicted_yaw)
    print("predicted pitch: ", predicted_pitch)
    print("predicted roll: ", predicted_roll)

    return (predicted_throttle, predicted_yaw, predicted_pitch, predicted_roll)

def init():
    global front_camera, model

    print("State = INIT -> " + STATE)

    #create model
    custom_objects = {"AdaBeliefOptimizer": AdaBeliefOptimizer}
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(model_dir)

    front_camera = camera.create_camera(0)

    drone.connect_drone('/dev/ttyACM0')
    drone.arm_and_takeoff(4)
    #drone.connect_drone('127.0.0.1:14551')

    return "flight"


def flight():
    global front_camera
    print("State = FLIGHT -> " + STATE)

    while True:
        current_cordinate = drone.get_location()
        current_location = (current_cordinate.lat,current_cordinate.lon)

        if gps.calculate_target_reached(current_location, target_location, target_reached_bubble): #check if target is reached 
            break
        
        else:
            #get data
            current_heading = drone.get_heading()
            img = front_camera.get_video(0)

            target_distance, path_distance = gps.calculate_path_distance(target, start, current_location)
            heading_delta = gps.calculate_heading_difference(current_heading, target_location, current_location)
            vel_x, vel_y, vel_z = drone.get_velocity()
            altitude = drone.get_altitude()
            speed = drone.get_ground_speed()
            data = np.array([speed, target_distance, path_distance, heading_delta, altitude, vel_x, vel_y, vel_z])

            #predict
            predicted = predict(img,data)
            predicted_throttle = predicted[0]
            predicted_yaw = predicted[1]
            predicted_pitch = predicted[2]
            predicted_roll = predicted[3]

            #write to drone 
            set_channel('1', predicted_roll)
            set_channel('2', predicted_pitch)
            set_channel('3', predicted_throttle)
            set_channel('4', predicted_yaw)

    return "goal_reached"


def goal_reached():
    global frame_count
    print("State = GOAL_REACHED -> " + STATE)
    print("Goal reached! Landing in 20 seconds!")
    drone.set_flight_mode("LOITER")
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
