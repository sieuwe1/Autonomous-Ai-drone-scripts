import os
import glob
import json
import cv2
import numpy as np

data_folder = '/home/drone/Desktop/dataset_POC/Testing'
#data_folder = '/home/drone/Desktop/dataset_POC/Training/shortend'

def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

# Loading in control dataframes
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
            img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
            normalizedImg = np.zeros((300, 300))
            normalizedImg = cv2.normalize(img,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
            images.append(normalizedImg)
            data.append(sample)


print("testing size: " , len(data))

#split and normalize train data into the seperate inputs and outputs ?? is linear distrubtion the best ??
img_x_train = images
train_data = data

bounds_train = []
bounds_train.append((min([a[0] for a in train_data]), max([a[0] for a in train_data]))) #roll
bounds_train.append((min([a[1] for a in train_data]), max([a[1] for a in train_data]))) #pitch
bounds_train.append((min([a[2] for a in train_data]), max([a[2] for a in train_data]))) #yaw 
bounds_train.append((min([a[3] for a in train_data]), max([a[3] for a in train_data]))) #throttle
bounds_train.append((min([a[4] for a in train_data]), max([a[4] for a in train_data]))) #speed
bounds_train.append((min([a[5] for a in train_data]), max([a[5] for a in train_data]))) #target_distance
bounds_train.append((min([a[6] for a in train_data]), max([a[6] for a in train_data]))) #path_distance
bounds_train.append((min([a[7] for a in train_data]), max([a[7] for a in train_data]))) #heading_delta
bounds_train.append((min([a[8] for a in train_data]), max([a[8] for a in train_data]))) #altitude
bounds_train.append((min([a[9] for a in train_data]), max([a[9] for a in train_data]))) #vel_x
bounds_train.append((min([a[10] for a in train_data]), max([a[10] for a in train_data]))) #vel_y
bounds_train.append((min([a[11] for a in train_data]), max([a[11] for a in train_data]))) #vel_z

data_x_train = []
for sample in train_data:
    count = 0
    data_x_train_sample = []
    for val in sample:
        if count > 3:
            data_x_train_sample.append(map(val,bounds_train[count][0],bounds_train[count][1],0,1))
        count+=1
    data_x_train.append(data_x_train_sample)

y_roll_train = [map(a[0],bounds_train[0][0],bounds_train[0][1],0,1) for a in train_data]
y_pitch_train = [map(a[1],bounds_train[1][0],bounds_train[1][1],0,1) for a in train_data]
y_yaw_train = [map(a[2],bounds_train[2][0],bounds_train[2][1],0,1) for a in train_data]
y_throttle_train = [map(a[3],bounds_train[3][0],bounds_train[3][1],0,1) for a in train_data]

data = np.array([img_x_train,data_x_train,y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train],dtype=object)
np.save('eval_data.npy', data)