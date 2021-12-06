import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.core.shape_base import vstack

training_size = 17571 #get these values from preprocessor
validation_size = 7529 #get these values from preprocessor
batch_size = 32

train_folder = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/train'
val_folder = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/val'
folder = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/eval_data.h5'

train_file_names = os.listdir(train_folder)
val_file_names = os.listdir(val_folder)

multi_file = True #set True if multi file dataset needs to be plotted
training_data = True #set False if validation data needs to be plotted from multi file 

value_to_plot =  1   # -1 
value_to_plot2 = 3 
value_to_plot3 = -1

# 0 'speed'             
# 1 'target_distance'
# 2 'path_distance'
# 3 'heading_delta'
# 4 'altitude'
# 5 'vel_x'
# 6 'vel_y'
# 7 'vel_z'
# 8 'roll'
# 9 'pitch'
# 10 'yaw'
# 11 'throttle'
# -1 ''

names = np.array(['speed', 'target_distance', 'path_distance', 'heading_delta', 'altitude', 
                    'vel_x', 'vel_y', 'vel_z', 'roll', 'pitch', 'yaw', 'throttle', ''])

def plot_in_time(data, results, key, key2, key3):

    if key != -1:
        if key > 7:
            plt.plot(results[key-8,:])
        else:
            plt.plot(data[:,key])

    if key2 != -1:
        if key2 > 7:
            plt.plot(results[key2-8,:])
        else:
            plt.plot(data[:,key2])

    if key3 != -1:    
        if key3 > 7:
            plt.plot(results[key3-8,:])
        else:
            plt.plot(data[:,key3])

    plt.ylabel('value')
    plt.xlabel('sample')
    plt.legend([names[key], names[key2], names[key3]], loc='upper left')
    plt.show()

    return None

def plot_distrubution(data, results, key, bins_count=60): 
    if key != -1:
        if key > 7:
            plt.hist(results[key-8,:], density=True, bins=bins_count)
        else:
            plt.hist(data[:,key], density=True, bins=bins_count) 

    plt.ylabel('Probability')
    plt.xlabel(names[key])
    plt.show()
    return None

def reshape(data):
    data_reshaped = []
    for file in data:
        for sample in range(file.shape[0]):
            data_reshaped.append(file[sample])
    return data_reshaped

if multi_file:

    data = []
    roll = []
    pitch = []
    yaw = []
    throttle = []

    if training_data:
        for i in range(round(training_size / batch_size)-1):
            hf = h5py.File(train_folder + "/"  + train_file_names[i], 'r')
            #img = np.array(hf.get('img_x_train'))
            data.append(np.array(hf.get('data_x_train')))
            roll.append(np.array(hf.get('y_roll_train')))
            pitch.append(np.array(hf.get('y_pitch_train')))
            yaw.append(np.array(hf.get('y_yaw_train')))
            throttle.append(np.array(hf.get('y_throttle_train')))
            hf.close()
    else:
        for i in range(round(validation_size / batch_size)-1):
            hf = h5py.File(val_folder + "/" + val_file_names[i], 'r')
            #img = np.array(hf.get('img_x_val'))
            data = np.array(hf.get('data_x_val'))
            roll = np.array(hf.get('y_roll_val'))
            pitch = np.array(hf.get('y_pitch_val'))
            yaw = np.array(hf.get('y_yaw_val'))
            throttle = np.array(hf.get('y_throttle_val'))
            hf.close()

    #reshape into usable format
    data_reshaped = np.array(reshape(data))
    roll_reshaped = np.array(reshape(roll))
    pitch_reshaped = np.array(reshape(pitch))
    yaw_reshaped = np.array(reshape(yaw))
    throttle_reshaped = np.array(reshape(throttle))

    results = vstack((roll_reshaped, pitch_reshaped, yaw_reshaped, throttle_reshaped))

    plot_in_time(data_reshaped, results,value_to_plot, value_to_plot2, value_to_plot3)
    
    #plot_distrubution(data_reshaped, results, value_to_plot)
    #plot_distrubution(data_reshaped, results, 8)
    #plot_distrubution(data_reshaped, results, 9)
    #plot_distrubution(data_reshaped, results, 10)
    #plot_distrubution(data_reshaped, results, 11)

else:
    hf = h5py.File(folder, 'r')
    data = np.array(hf.get('data_x_train'))
    roll = np.array(hf.get('y_roll_train'))
    pitch = np.array(hf.get('y_pitch_train'))
    yaw = np.array(hf.get('y_yaw_train'))
    throttle = np.array(hf.get('y_throttle_train'))
    hf.close()

    results = np.vstack((roll, pitch, yaw, throttle))

    #plot_in_time(data, results,value_to_plot, value_to_plot2, value_to_plot3)
    plot_distrubution(data, results, value_to_plot)