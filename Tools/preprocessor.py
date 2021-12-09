#%%
import os
import glob
import json
from typing import Protocol
import cv2
import numpy as np
import preprocessor_functions
import h5py
import config as c

batch_size = 32
preprocessed_data_name_train = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/Tools/train/"
preprocessed_data_name_val = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/Tools/val/"

processed_data_name_non_batch = "data.h5"


evaluation_data = True #weather the data is for training or for evaluation script 
positional_embedding = False #weather the data is for training and evaluating postional embedding enabled model which does not need data normalization. Only images

data, images, sample_count= preprocessor_functions.load_folder(c.data_dir)

if positional_embedding:
    if evaluation_data:
        processed_data = np.zeros((sample_count,6))
        print("testing size: " , len(data))
        
        data_x_train = preprocessor_functions.get_input_data_from_data(data)
        y_roll_train = [a[0] for a in data]
        y_pitch_train = [a[1] for a in data]
        y_yaw_train = [a[2] for a in data]
        y_throttle_train = [a[3] for a in data]
        img_x_train = preprocessor_functions.normalize_images(images)

        hf = h5py.File('positional_embedding_eval_' + processed_data_name_non_batch, 'w')
        hf.create_dataset('img_x_train', data=img_x_train)
        hf.create_dataset('data_x_train', data=data_x_train)
        hf.create_dataset('y_roll_train', data=y_roll_train)
        hf.create_dataset('y_pitch_train', data=y_pitch_train)
        hf.create_dataset('y_yaw_train', data=y_yaw_train)
        hf.create_dataset('y_throttle_train', data=y_throttle_train)
        hf.close()

    else:
        #split data into val and train 
        data_length = len(data)
        split_index = round(data_length * 0.7)

        train_data = data[:split_index]
        train_img = images[:split_index]
        validate_data = data[split_index + 1:]
        validate_img = images[split_index + 1:]

        print("training size: " , len(train_data))
        print("validation size: " ,len(validate_data))

        data_x_train = preprocessor_functions.get_input_data_from_data(train_data)
        y_roll_train = [a[0] for a in train_data]
        y_pitch_train = [a[1] for a in train_data]
        y_yaw_train = [a[2] for a in train_data]
        y_throttle_train = [a[3] for a in train_data]
        img_x_train = preprocessor_functions.normalize_images(train_img)

        data_x_val = preprocessor_functions.get_input_data_from_data(validate_data)
        y_roll_val = [a[0] for a in validate_data]
        y_pitch_val = [a[1] for a in validate_data]
        y_yaw_val = [a[2] for a in validate_data]
        y_throttle_val = [a[3] for a in validate_data]
        img_x_val = preprocessor_functions.normalize_images(validate_img)

        hf = h5py.File('positional_embedding_' + processed_data_name_non_batch, 'w')
        hf.create_dataset('img_x_train', data=img_x_train)
        hf.create_dataset('data_x_train', data=data_x_train)
        hf.create_dataset('y_roll_train', data=y_roll_train)
        hf.create_dataset('y_pitch_train', data=y_pitch_train)
        hf.create_dataset('y_yaw_train', data=y_yaw_train)
        hf.create_dataset('y_throttle_train', data=y_throttle_train)

        hf.create_dataset('img_x_val', data=img_x_val)
        hf.create_dataset('data_x_val', data=data_x_val)
        hf.create_dataset('y_roll_val', data=y_roll_val)
        hf.create_dataset('y_pitch_val', data=y_pitch_val)
        hf.create_dataset('y_yaw_val', data=y_yaw_val)
        hf.create_dataset('y_throttle_val', data=y_throttle_val)

        hf.close()

else:
    if evaluation_data:
        print("testing size: " , len(data))
        
        bounds_test, data_x_train, y_roll_train, y_pitch_train, y_yaw_train, y_throttle_train = preprocessor_functions.normalize_data(data,sample_count)
        img_x_train = preprocessor_functions.normalize_images(images)

        print("bounds_test: ", bounds_test)

        hf = h5py.File('eval_' + processed_data_name_non_batch, 'w')
        hf.create_dataset('img_x_train', data=img_x_train)
        hf.create_dataset('data_x_train', data=data_x_train)
        hf.create_dataset('y_roll_train', data=y_roll_train)
        hf.create_dataset('y_pitch_train', data=y_pitch_train)
        hf.create_dataset('y_yaw_train', data=y_yaw_train)
        hf.create_dataset('y_throttle_train', data=y_throttle_train)
        hf.close()

    else:
        #processed_data = np.zeros((sample_count,12),dtype=object)
        #print(processed_data.shape)
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

        print("training size: " , len(data[:split_index]))
        print("validation size: " ,len(data[split_index + 1:]))

        img_x_train = preprocessor_functions.normalize_images(train_img)
        img_x_val = preprocessor_functions.normalize_images(validate_img)

        #bounds_train,processed_data[:,1],processed_data[:,2],processed_data[:,3],processed_data[:,4],processed_data[:,5] = preprocessor_functions.normalize_data(train_data,sample_count)
        #bounds_val,processed_data[7],processed_data[8],processed_data[9],processed_data[10],processed_data[11] =  preprocessor_functions.normalize_data(validate_data,sample_count)
        bounds_train, data_x_train, y_roll_train, y_pitch_train, y_yaw_train, y_throttle_train = preprocessor_functions.normalize_data(train_data,sample_count)
        bounds_val, data_x_val, y_roll_val, y_pitch_val, y_yaw_val, y_throttle_val = preprocessor_functions.normalize_data(validate_data,sample_count)

        #processed_data[0] = preprocessor_functions.normalize_images(train_img)
        #processed_data[6] = preprocessor_functions.normalize_images(validate_img)

        #print("bounds_train: ", bounds_train)
        #print("bounds_val: ", bounds_val)

        #print(len(img_x_train))
        #print(len(img_x_val))
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
            hf.create_dataset('y_roll_train', data=y_roll_train[start:end])
            hf.create_dataset('y_pitch_train', data=y_pitch_train[start:end])
            hf.create_dataset('y_yaw_train', data=y_yaw_train[start:end])
            hf.create_dataset('y_throttle_train', data=y_throttle_train[start:end])
            hf.close()

            last_end_index_train = end

        last_end_index_val = 0
        for i in range(validation_file_count):

            start = last_end_index_val
            end = last_end_index_val + batch_size + 1

            hf = h5py.File(preprocessed_data_name_val+"_validation_"+str(i)+'.h5', 'w')
            hf.create_dataset('img_x_val', data=img_x_val[start:end])
            hf.create_dataset('data_x_val', data=data_x_val[start:end])
            hf.create_dataset('y_roll_val', data=y_roll_val[start:end])
            hf.create_dataset('y_pitch_val', data=y_pitch_val[start:end])
            hf.create_dataset('y_yaw_val', data=y_yaw_val[start:end])
            hf.create_dataset('y_throttle_val', data=y_throttle_val[start:end])
            hf.close()
        
            last_end_index_val = end
        #data = np.array([img_x_train,data_x_train,y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train,img_x_val,data_x_val,y_roll_val,y_pitch_val,y_yaw_val,y_throttle_val],dtype=object)
        
# %%
