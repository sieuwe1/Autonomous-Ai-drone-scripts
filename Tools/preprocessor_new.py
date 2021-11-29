import os
import glob
import json
import cv2
import numpy as np
import preprocessor_functions

preprocessed_data_name = "data.npy"
data_folder = '/home/drone/Desktop/dataset_POC/Training'

evaluation_data = False #weather the data is for training or for evaluation script 
positional_embedding = False #weather the data is for training and evaluating postional embedding enabled model which does not need data normalization. Only images

data, images = preprocessor_functions.load_folder(data_folder)

if positional_embedding:
    if evaluation_data:
        print("testing size: " , len(data))
        
        data_x_train = preprocessor_functions.get_input_data_from_data(data)
        y_roll_train = [a[0] for a in data]
        y_pitch_train = [a[1] for a in data]
        y_yaw_train = [a[2] for a in data]
        y_throttle_train = [a[3] for a in data]
        img_x_train = preprocessor_functions.normalize_images(images)

        data = np.array([img_x_train,data_x_train,y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train],dtype=object)
        np.save('positional_embedding_eval_' + preprocessed_data_name, data)

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

        data = np.array([img_x_train,data_x_train,y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train,img_x_val,data_x_val,y_roll_val,y_pitch_val,y_yaw_val,y_throttle_val],dtype=object)
        np.save('positional_embedding_' + preprocessed_data_name, data)

else:
    if evaluation_data:
        print("testing size: " , len(data))
        
        bounds_test, data_x_train, y_roll_train, y_pitch_train, y_yaw_train, y_throttle_train = preprocessor_functions.normalize_data(data)
        img_x_train = preprocessor_functions.normalize_images(images)
        
        print("bounds_test: ", bounds_test)
        data = np.array([img_x_train,data_x_train,y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train],dtype=object)
        np.save('eval_' + preprocessed_data_name, data)

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

        bounds_train,data_x_train,y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train = preprocessor_functions.normalize_data(train_data)
        bounds_val,data_x_val,y_roll_val,y_pitch_val,y_yaw_val,y_throttle_val =  preprocessor_functions.normalize_data(validate_data)

        img_x_train = preprocessor_functions.normalize_images(train_img)
        img_x_val = preprocessor_functions.normalize_images(validate_img)

        print("bounds_train: ", bounds_train)
        print("bounds_val: ", bounds_val)

        data = np.array([img_x_train,data_x_train,y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train,img_x_val,data_x_val,y_roll_val,y_pitch_val,y_yaw_val,y_throttle_val],dtype=object)
        np.save(preprocessed_data_name, data)