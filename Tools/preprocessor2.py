
#%%
import os
import glob
import json
from typing import Protocol
import cv2
import numpy as np
import preprocessor_functions
import h5py
import random

steps = 2 #how maney samples to skip
batch_size = 30
preprocessed_data_name_train = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/Tools/train/"
preprocessed_data_name_val = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/Tools/val/"

processed_data_name_non_batch = "data.h5"

#data_folder = '/home/drone/Desktop/dataset_POC/Training/shortend'
data_folder = '/home/drone/Desktop/dataset_POC/Training'

evaluation_data = True #weather the data is for training or for evaluation script 

if evaluation_data:
    steps = 1

data, images, sample_count= preprocessor_functions.load_folder(data_folder,steps)

if not evaluation_data:
    for i in range(8):
        random.shuffle([a[i] for a in data])

if evaluation_data:
    print("testing size: " , len(data))

    #bounds_test, data_x_train, remove1, remove2, remove3, remove4 = preprocessor_functions.normalize_data_z_score(data,sample_count)
    #bounds_train, remove5, y_roll_train, y_pitch_train, y_yaw_train, y_throttle_train = preprocessor_functions.normalize_data_linear(data,sample_count)
    
    bounds_train, data_x_train, n, n, yaw_train, n = preprocessor_functions.normalize_data_linear(data,sample_count)

    img_x_train = preprocessor_functions.normalize_images(images)

    print("bounds_test: ", bounds_train)

    #shift all outputs to t+1 to make model predict future values
    y_vel_x_train =  np.roll(data_x_train[:,5],1)
    y_vel_y_train =  np.roll(data_x_train[:,6],1)
    y_yaw_train = np.roll(yaw_train,1)

    data_x_train = np.delete(data_x_train, [0,4], 1)
    data_x_train = np.hstack((data_x_train,yaw_train))
 
    print(data_x_train.shape)
    print("random sample: " + str(data_x_train[2]))

    hf = h5py.File('eval_' + processed_data_name_non_batch, 'w')
    hf.create_dataset('img_x_train', data=img_x_train)
    hf.create_dataset('data_x_train', data=data_x_train)
    hf.create_dataset('y_vel_x_train', data=y_vel_x_train)
    hf.create_dataset('y_vel_y_train', data=y_vel_y_train)
    hf.create_dataset('y_yaw_train', data=y_yaw_train)
    hf.close()

else:
    #split data into val and train 
    print("splitting")
    data_length = len(data)
    split_index = round(data_length * 0.7)

    #train_img, train_data = preprocessor_functions.augment_images(images[:split_index],data[:split_index])
    #validate_img, validate_data = preprocessor_functions.augment_images(images[split_index + 1:], data[split_index + 1:])
    train_img = images[:split_index]
    train_data = data[:split_index]
    validate_img = images[split_index + 1:]
    validate_data = data[split_index + 1:]

    print("training size: " , len(train_data))
    print("validation size: " ,len(validate_data))

    print("normalizing images")
    img_x_train = preprocessor_functions.normalize_images(train_img)
    img_x_val = preprocessor_functions.normalize_images(validate_img)

    print("normalizing variables")
    print("val")

    bounds_val, data_x_val, n, n, yaw_val, n = preprocessor_functions.normalize_data_linear(validate_data,sample_count)
    print("train")  
    bounds_train, data_x_train, n, n, yaw_train, n = preprocessor_functions.normalize_data_linear(train_data,sample_count)

    #shift all outputs to t+1 to make model predict future values
    y_vel_x_val = np.roll(data_x_val[:,5],1)
    y_vel_y_val =  np.roll(data_x_val[:,6],1)
    y_vel_x_train =  np.roll(data_x_train[:,5],1)
    y_vel_y_train =  np.roll(data_x_train[:,6],1)
    y_yaw_val = np.roll(yaw_val,1)
    y_yaw_train = np.roll(yaw_train,1)

    print("y_vel_x_train ", y_vel_x_val)
    print("y_vel_x_train shape", y_vel_x_val.shape)

    data_x_val = np.delete(data_x_val, [0,4], 1)
    data_x_train = np.delete(data_x_train, [0,4], 1)
    data_x_val = np.hstack((data_x_val,yaw_val))
    data_x_train = np.hstack((data_x_train,yaw_train))
    print(data_x_train.shape)
    print("random sample: " + str(data_x_train[2]))

    print("bounds_train: ", bounds_train)
    print("bounds_val: ", bounds_val)
    
#%%    
    training_len = len(img_x_train)
    validation_len = len(img_x_val)

    training_file_count = round(training_len / batch_size)
    validation_file_count = round(validation_len / batch_size)

    print("training file count: ", training_file_count)
    print("validation file count: ", validation_file_count)

    last_end_index_train = 0
    for i in range(training_file_count):
        start = last_end_index_train
        end = last_end_index_train + batch_size + 1
        
        hf = h5py.File(preprocessed_data_name_train+"_training_"+str(i)+'.h5', 'w')
        hf.create_dataset('img_x_train', data=img_x_train[start:end])
        hf.create_dataset('data_x_train', data=data_x_train[start:end])
        hf.create_dataset('y_vel_x_train', data=y_vel_x_train[start:end])
        hf.create_dataset('y_vel_y_train', data=y_vel_y_train[start:end])
        hf.create_dataset('y_yaw_train', data=y_yaw_train[start:end])
        hf.close()

        last_end_index_train = end

    last_end_index_val = 0
    for i in range(validation_file_count):

        start = last_end_index_val
        end = last_end_index_val + batch_size + 1

        hf = h5py.File(preprocessed_data_name_val+"_validation_"+str(i)+'.h5', 'w')
        hf.create_dataset('img_x_val', data=img_x_val[start:end])
        hf.create_dataset('data_x_val', data=data_x_val[start:end,])
        hf.create_dataset('y_vel_x_val', data=y_vel_x_val[start:end])
        hf.create_dataset('y_vel_y_val', data=y_vel_y_val[start:end])
        hf.create_dataset('y_yaw_val', data=y_yaw_val[start:end])
        hf.close()
    
        last_end_index_val = end
    