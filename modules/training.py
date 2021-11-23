#%%
import os
import glob
import random
import json
import time
import zlib
import matplotlib.pyplot as plt
from os.path import basename, join, splitext, dirname
import cv2
from matplotlib import image
from matplotlib import pyplot
from tensorflow import keras
import numpy as np
import math
import os
from time import time
import network
import tensorflow as tf

transfer = True
data_folder = '/home/drone/Desktop/dataset_POC/Training'
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

            #print(control_file)

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

#%%
#split data into val and train 

data_length = len(data)
split_index = round(data_length * 0.7)

train_data = data[:split_index]
train_img = images[:split_index]
validate_data = data[split_index + 1:]
validate_img = images[split_index + 1:]

print(train_data)

print("training size: " , len(train_data))
print("validation size: " ,len(validate_data))

#split and normalize train data into the seperate inputs and outputs ?? is linear distrubtion the best ??
img_x_train = train_img

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



#split and normalize val data into the seperate inputs and outputs !!! needs to happen saperatly to avoid data leak between val and train sets !!!
img_x_val = validate_img

bounds_val = []
bounds_val.append((min([a[0] for a in validate_data]), max([a[0] for a in validate_data]))) #roll
bounds_val.append((min([a[1] for a in validate_data]), max([a[1] for a in validate_data]))) #pitch
bounds_val.append((min([a[2] for a in validate_data]), max([a[2] for a in validate_data]))) #yaw 
bounds_val.append((min([a[3] for a in validate_data]), max([a[3] for a in validate_data]))) #throttle
bounds_val.append((min([a[4] for a in validate_data]), max([a[4] for a in validate_data]))) #speed
bounds_val.append((min([a[5] for a in validate_data]), max([a[5] for a in validate_data]))) #target_distance
bounds_val.append((min([a[6] for a in validate_data]), max([a[6] for a in validate_data]))) #path_distance
bounds_val.append((min([a[7] for a in validate_data]), max([a[7] for a in validate_data]))) #heading_delta
bounds_val.append((min([a[8] for a in validate_data]), max([a[8] for a in validate_data]))) #altitude
bounds_val.append((min([a[9] for a in validate_data]), max([a[9] for a in validate_data]))) #vel_x
bounds_val.append((min([a[10] for a in validate_data]), max([a[10] for a in validate_data]))) #vel_y
bounds_val.append((min([a[11] for a in validate_data]), max([a[11] for a in validate_data]))) #vel_z

data_x_val = []
for sample in validate_data:
    count = 0
    data_x_val_sample = []
    for val in sample:
        if count > 3:
            data_x_val_sample.append(map(val,bounds_val[count][0],bounds_val[count][1],0,1))
        count+=1
    data_x_val.append(data_x_val_sample)

y_roll_val = [map(a[0],bounds_val[0][0],bounds_val[0][1],0,1) for a in validate_data]
y_pitch_val = [map(a[1],bounds_val[1][0],bounds_val[1][1],0,1) for a in validate_data]
y_yaw_val = [map(a[2],bounds_val[2][0],bounds_val[2][1],0,1) for a in validate_data]
y_throttle_val = [map(a[3],bounds_val[3][0],bounds_val[3][1],0,1) for a in validate_data]

#print(data_x_train)
#print(y_roll_train)
#print(y_pitch_train)
#print(y_yaw_train)
#print(y_throttle_train)
print("--------------------------------------------------------------------------------")
#print(data_x_val)
#print(y_roll_val)
#print(y_pitch_val)
#print(y_yaw_val)
#print(y_throttle_val)

print("bounds_train: ", bounds_train)
print("bounds_val: ", bounds_val)

model = None
#create model
if transfer:
    model, base_model = network.create_transfer_model()

    for layer in base_model.layers:
        layer.trainable = False

else:
    model = network.create_model()

#compile model
crit = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
opt = tf.keras.optimizers.Nadam(learning_rate=0.0001)
model.compile(loss=crit, optimizer=opt, metrics=['accuracy'])

#train model
callback_early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta= .0005)

#save logs with tensorflow
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

#train model
history = model.fit(x=[np.array(img_x_train), np.array(data_x_train)], y=[np.array(y_roll_train),np.array(y_pitch_train),np.array(y_yaw_train),np.array(y_throttle_train)],
	validation_data=([np.array(img_x_val), np.array(data_x_val)], [np.array(y_roll_val),np.array(y_pitch_val),np.array(y_yaw_val),np.array(y_throttle_val)]),
	epochs=150, batch_size=8, callbacks=[callback_early_stop], shuffle=True)

#history = model.fit(x=[img_x_train, data_x_train], y=[y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train],
#	validation_data=([img_x_val, data_x_val], [y_roll_val,y_pitch_val,y_yaw_val,y_throttle_val]),
#	epochs=150, batch_size=8, callbacks=[callback_early_stop], shuffle=True)


#save model
model.save_weights('trained_best_model_full_set_vgg16_tranfer_weights_2.h5')
#model.save('trained_best_model_full_set_vgg16_tranfer.h5')

print(history.history)

#plot summary
plt.plot(history.history['out_0_accuracy'])
plt.plot(history.history['val_out_0_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#print bounds for use in interference
print("bounds_train: ", bounds_train)
print("bounds_val: ", bounds_val)

# %%
