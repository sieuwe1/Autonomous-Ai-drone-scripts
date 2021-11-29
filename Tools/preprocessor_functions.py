import os
import glob
import json
import cv2
import numpy as np
from tensorflow import keras
from keras.applications import imagenet_utils

def normalize_inception_image(img):
    return imagenet_utils.preprocess_input(img, data_format=None, mode='tf') #inception uses 'tf' mode

def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def load_folder(data_folder):
    folder_pattern = os.path.join(data_folder,'*/')
    folders = glob.glob(folder_pattern)
    data = []
    images = []
    print(folders)

    for folder in folders:
        json_pattern = os.path.join(folder + str("control/"),'*.json')
        control_files = glob.glob(json_pattern)

        #print(control_files)

        for control_file in control_files:
            if os.path.getsize(control_file) > 250: #check if file is empty
                sample = []
                f = open(control_file)
                json_data = json.load(f)
                sample.append(json_data['user/roll'])
                sample.append(json_data['user/pitch'])
                sample.append(json_data['user/yaw'])
                sample.append(json_data['user/throttle'])
                #sample.append(json_data['gps/latitude'])
                #sample.append(json_data['gps/longtitude'])
                sample.append(json_data['gps/speed'])
                sample.append(json_data['gps/target_distance'])
                sample.append(json_data['gps/path_distance'])
                sample.append(json_data['gps/heading_delta'])
                sample.append(json_data['gps/altitude'])
                sample.append(json_data['imu/vel_x'])
                sample.append(json_data['imu/vel_y'])
                sample.append(json_data['imu/vel_z'])
                image_path = json_data['cam/image_name']
                #print(folder + str("left_camera/") + image_path)
                img = cv2.imread(folder + str("left_camera/") + image_path)
                images.append(img)
                data.append(sample)

    return (data,images)

def normalize_data(data):

    bounds_data = []
    bounds_data.append((min([a[0] for a in data]), max([a[0] for a in data]))) #roll
    bounds_data.append((min([a[1] for a in data]), max([a[1] for a in data]))) #pitch
    bounds_data.append((min([a[2] for a in data]), max([a[2] for a in data]))) #yaw 
    bounds_data.append((min([a[3] for a in data]), max([a[3] for a in data]))) #throttle
    bounds_data.append((min([a[4] for a in data]), max([a[4] for a in data]))) #speed
    bounds_data.append((min([a[5] for a in data]), max([a[5] for a in data]))) #target_distance
    bounds_data.append((min([a[6] for a in data]), max([a[6] for a in data]))) #path_distance
    bounds_data.append((min([a[7] for a in data]), max([a[7] for a in data]))) #heading_delta
    bounds_data.append((min([a[8] for a in data]), max([a[8] for a in data]))) #altitude
    bounds_data.append((min([a[9] for a in data]), max([a[9] for a in data]))) #vel_x
    bounds_data.append((min([a[10] for a in data]), max([a[10] for a in data]))) #vel_y
    bounds_data.append((min([a[11] for a in data]), max([a[11] for a in data]))) #vel_z

    data_normalized = []
    for sample in data:
        count = 0
        data_normalized_sample = []
        for val in sample:
            if count > 3:
                data_normalized_sample.append(map(val,bounds_data[count][0],bounds_data[count][1],0,1))
            count+=1
        data_normalized.append(data_normalized_sample)

    roll_normalized = [map(a[0],bounds_data[0][0],bounds_data[0][1],0,1) for a in data]
    pitch_normalized = [map(a[1],bounds_data[1][0],bounds_data[1][1],0,1) for a in data]
    yaw_normalized = [map(a[2],bounds_data[2][0],bounds_data[2][1],0,1) for a in data]
    throttle_normalized = [map(a[3],bounds_data[3][0],bounds_data[3][1],0,1) for a in data]

    return (bounds_data, data_normalized, roll_normalized, pitch_normalized, yaw_normalized, throttle_normalized)

def normalize_images(imagedata):
    normalized_images = []
    for image in imagedata:
        normalized_images.append(normalize_inception_image(image))
    
    return normalized_images

def get_input_data_from_data(data):
    input_data = []
    for sample in input_data:
        count = 0
        input_data_sample = []
        for val in sample:
            if count > 3:
                input_data_sample.append(val)
            count+=1
        input_data.append(input_data_sample)

    return input_data