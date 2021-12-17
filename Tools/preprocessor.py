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

batch_size = 32
preprocessed_data_name_train = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/Tools/train/"
preprocessed_data_name_val = "/home/drone/Desktop/Autonomous-Ai-drone-scripts/Tools/val/"

processed_data_name_non_batch = "data.h5"

#data_folder = '/home/drone/Desktop/dataset_POC/Training/shortend'
data_folder = '/home/drone/Desktop/dataset_POC/Testing'

evaluation_data = True #weather the data is for training or for evaluation script 
sinusoidal_embedding = False #weather the data is for training and evaluating postional embedding enabled model which does not need data normalization. Only images#shuffle = True


data, images, sample_count= preprocessor_functions.load_folder(data_folder)

if not evaluation_data:
    for i in range(8):
        random.shuffle([a[i] for a in data])

if sinusoidal_embedding:
    if evaluation_data:
        processed_data = np.zeros((sample_count,6))
        print("testing size: " , len(data))
        
        data_x_train = preprocessor_functions.get_input_data_from_data(data)
        data_x_train_ = np.array([sinusoidal_embedding(_, max_value=4000) for _ in data_x_train])
        y_roll_train = preprocessor_functions.normalize_data_linear_single([a[0] for a in data])
        y_pitch_train = preprocessor_functions.normalize_data_linear_single([a[1] for a in data])
        y_yaw_train = preprocessor_functions.normalize_data_linear_single([a[2] for a in data])
        y_throttle_train = preprocessor_functions.normalize_data_linear_single([a[3] for a in data])
        img_x_train = preprocessor_functions.normalize_images(images)

        hf = h5py.File('positional_embedding_eval_' + processed_data_name_non_batch, 'w')
        hf.create_dataset('img_x_train', data=img_x_train)
        hf.create_dataset('data_x_train', data=data_x_train_)
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
        data_x_train_ = np.array([sinusoidal_embedding(_, max_value=4000) for _ in data_x_train])
        y_roll_train = preprocessor_functions.normalize_data_linear_single([a[0] for a in train_data])
        y_pitch_train = preprocessor_functions.normalize_data_linear_single([a[1] for a in train_data])
        y_yaw_train = preprocessor_functions.normalize_data_linear_single([a[2] for a in train_data])
        y_throttle_train = preprocessor_functions.normalize_data_linear_single([a[3] for a in train_data])
        img_x_train = preprocessor_functions.normalize_images(train_img)

        data_x_val = preprocessor_functions.get_input_data_from_data(validate_data)
        data_x_val_ = np.array([sinusoidal_embedding(_, max_value=4000) for _ in data_x_val])
        y_roll_val = preprocessor_functions.normalize_data_linear_single([a[0] for a in validate_data])
        y_pitch_val = preprocessor_functions.normalize_data_linear_single([a[1] for a in validate_data])
        y_yaw_val = preprocessor_functions.normalize_data_linear_single([a[2] for a in validate_data])
        y_throttle_val = preprocessor_functions.normalize_data_linear_single([a[3] for a in validate_data])
        img_x_val = preprocessor_functions.normalize_images(validate_img)

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
            hf.create_dataset('data_x_train', data=data_x_train_[start:end])
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
            hf.create_dataset('data_x_val', data=data_x_val_[start:end])
            hf.create_dataset('y_roll_val', data=y_roll_val[start:end])
            hf.create_dataset('y_pitch_val', data=y_pitch_val[start:end])
            hf.create_dataset('y_yaw_val', data=y_yaw_val[start:end])
            hf.create_dataset('y_throttle_val', data=y_throttle_val[start:end])
            hf.close()
        
            last_end_index_val = end

else:
    if evaluation_data:
        print("testing size: " , len(data))
  
        #bounds_test, data_x_train, remove1, remove2, remove3, remove4 = preprocessor_functions.normalize_data_z_score(data,sample_count)
        #bounds_train, remove5, y_roll_train, y_pitch_train, y_yaw_train, y_throttle_train = preprocessor_functions.normalize_data_linear(data,sample_count)
        
        bounds_train, data_x_train, y_roll_train, y_pitch_train, y_yaw_train, y_throttle_train = preprocessor_functions.normalize_data_linear(data,sample_count)
        
        print(data_x_train.shape)
        print("random sample: " + str(data_x_train[2]))

        img_x_train = preprocessor_functions.normalize_images(images)

        print("bounds_test: ", bounds_train)

        hf = h5py.File('eval_' + processed_data_name_non_batch, 'w')
        hf.create_dataset('img_x_train', data=img_x_train)
        hf.create_dataset('data_x_train', data=data_x_train)
        hf.create_dataset('y_roll_train', data=y_roll_train)
        hf.create_dataset('y_pitch_train', data=y_pitch_train)
        hf.create_dataset('y_yaw_train', data=y_yaw_train)
        hf.create_dataset('y_throttle_train', data=y_throttle_train)
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
#%%
        print("normalizing images")
        img_x_train = preprocessor_functions.normalize_images(train_img)
        img_x_val = preprocessor_functions.normalize_images(validate_img)

#%%
        print("normalizing variables")
        print("val")
        bounds_val, data_x_val, y_roll_val, y_pitch_val, y_yaw_val, y_throttle_val = preprocessor_functions.normalize_data_linear(validate_data,sample_count)
        print("train")  
        bounds_train, data_x_train, y_roll_train, y_pitch_train, y_yaw_train, y_throttle_train = preprocessor_functions.normalize_data_linear(train_data,sample_count)

        bounds_train2, remove14, remove1, remove2, remove3, remove4 = preprocessor_functions.normalize_data_z_score(train_data,sample_count)
        #bounds_val, data_x_val, remove5, remove6, remove7, remove8 = preprocessor_functions.normalize_data_z_score(validate_data,sample_count)
        
        #bounds_train, remove9, y_roll_train, y_pitch_train, y_yaw_train, y_throttle_train = preprocessor_functions.normalize_data_linear(train_data,sample_count)
        #bounds_val, remove10, y_roll_val, y_pitch_val, y_yaw_val, y_throttle_val = preprocessor_functions.normalize_data_linear(validate_data,sample_count)

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
        
# %%
