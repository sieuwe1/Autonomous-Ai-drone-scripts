import os
import glob
import json
import cv2
from keras.backend import dtype
import numpy as np
from numpy.random.mtrand import shuffle
from tensorflow import keras
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def augment_images(images, data):
    print("before augmentation data length: ", len(images))
    #print(np.array(images).shape)

    datagen = ImageDataGenerator(width_shift_range=[-30,30], height_shift_range=[-30,30],rotation_range=5,brightness_range=[0.2,1.0],)
    # prepare iterator
    it = datagen.flow(np.array(images), batch_size=1)

    for x in range(500):

        
        plt.imshow(images[x])
        for i in range(5):
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imshow(image)

        plt.show()


    #return images, data

def normalize_inception_image(img):
    result = imagenet_utils.preprocess_input(img, data_format=None, mode='tf') #inception uses 'tf' mode
    return result

def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def load_folder(data_folder):

    #file_count = sum([len(files) for r, d, files in os.walk(data_folder)])
    #print(file_count)
    
    folder_pattern = os.path.join(data_folder,'*/')
    folders = sorted(glob.glob(folder_pattern))

    sample_count = 0
    data = []
    images = []
    print(folders)

    for folder in folders:
        json_pattern = os.path.join(folder + str("control/"),'*.json')
        control_files = sorted(glob.glob(json_pattern))

        print(folder)

        for control_file in control_files:
            if os.path.getsize(control_file) > 250: #check if file is empty
                sample = np.zeros(shape=(12,1))
                f = open(control_file)
                json_data = json.load(f)
                sample[0] = json_data['user/roll']
                sample[1] = json_data['user/pitch']
                sample[2] = json_data['user/yaw']
                sample[3] = json_data['user/throttle']
                sample[4] = json_data['gps/speed']
                sample[5] = json_data['gps/target_distance']
                sample[6] = json_data['gps/path_distance']
                sample[7] = json_data['gps/heading_delta']
                sample[8] = json_data['gps/altitude']
                sample[9] = json_data['imu/vel_x']
                sample[10] = json_data['imu/vel_y']
                sample[11] = json_data['imu/vel_z']
                #sample.append(json_data['user/roll'])
                #sample.append(json_data['user/pitch'])
                #sample.append(json_data['user/yaw'])
                #sample.append(json_data['user/throttle'])
                #sample.append(json_data['gps/latitude'])
                #sample.append(json_data['gps/longtitude'])
                #sample.append(json_data['gps/speed'])
                #sample.append(json_data['gps/target_distance'])
                #sample.append(json_data['gps/path_distance'])
                #sample.append(json_data['gps/heading_delta'])
                #sample.append(json_data['gps/altitude'])
                #sample.append(json_data['imu/vel_x'])
                #sample.append(json_data['imu/vel_y'])
                #sample.append(json_data['imu/vel_z'])
                image_path = json_data['cam/image_name']
                #print(folder + str("left_camera/") + image_path)
                img = cv2.imread(folder + str("left_camera/") + image_path)
                img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
                images.append(img)
                data.append(sample)
                sample_count+=1

    return (data,images, sample_count)

def normalize_data(data, sample_count):

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

    data_normalized = np.zeros((len(data),8),dtype=np.float32)

    for i in range(len(data)):
        data_normalized[i][0] = map(data[i][4],bounds_data[4][0],bounds_data[4][1],0,1)
        data_normalized[i][1] = map(data[i][5],bounds_data[5][0],bounds_data[5][1],0,1)
        data_normalized[i][2] = map(data[i][6],bounds_data[6][0],bounds_data[6][1],0,1)
        data_normalized[i][3] = map(data[i][7],bounds_data[7][0],bounds_data[7][1],0,1)
        data_normalized[i][4] = map(data[i][8],bounds_data[8][0],bounds_data[8][1],0,1)
        data_normalized[i][5] = map(data[i][9],bounds_data[9][0],bounds_data[9][1],0,1)
        data_normalized[i][6] = map(data[i][10],bounds_data[10][0],bounds_data[10][1],0,1)
        data_normalized[i][7] = map(data[i][11],bounds_data[11][0],bounds_data[11][1],0,1)


    roll_normalized = np.array([map(a[0],bounds_data[0][0],bounds_data[0][1],0,1) for a in data])
    pitch_normalized = np.array([map(a[1],bounds_data[1][0],bounds_data[1][1],0,1) for a in data])
    yaw_normalized = np.array([map(a[2],bounds_data[2][0],bounds_data[2][1],0,1) for a in data])
    throttle_normalized = np.array([map(a[3],bounds_data[3][0],bounds_data[3][1],0,1) for a in data])

    #for i in range(8):
    #    fill_list(data_normalized[i], sample_count)

    #fill_list(roll_normalized, sample_count)
    #fill_list(pitch_normalized, sample_count)
    #fill_list(yaw_normalized, sample_count)
    #fill_list(throttle_normalized, sample_count)

    return bounds_data, data_normalized, roll_normalized, pitch_normalized, yaw_normalized, throttle_normalized

def normalize_images(imagedata):
    for i in range(len(imagedata)):
        image = imagedata[i]
        imagedata[i] = normalize_inception_image(image)
    return np.array(imagedata,dtype=np.float16)

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
