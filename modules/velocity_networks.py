#these networks predict velocity components and use other inputs then the pwm based networks
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3, VGG16, MobileNetV3Large, MobileNetV3Small
from keras_pos_embd import PositionEmbedding
from tensorflow.python.keras.backend import flatten
from tensorflow.python.keras.utils.layer_utils import print_summary

# CNN + Dense models

#---------------------------------------------------------------------------------
# Basic model based on Donkeycar

def donkeycar_model_no_height_control():  
    input_shape = (300, 300, 3)
    
    img_in = layers.Input(shape=input_shape, name='img_in')

    #targetdistance, pathdistance, headingdelta, x at T, y at T, yaw_pwm at T
    metadata = layers.Input(shape=(7,), name="path_distance_in")

    x = img_in
    x = layers.Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = layers.Flatten(name='flattened')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.1)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.1)(x)

    y = metadata
    y = layers.Dense(14, activation='relu')(y)
    y = layers.Dense(28, activation='relu')(y)
    y = layers.Dense(56, activation='relu')(y)

    z = layers.concatenate([x, y])
    z = layers.Dense(50, activation='relu')(z)
    z = layers.Dropout(.1)(z)
    z = layers.Dense(50, activation='relu')(z)
    z = layers.Dropout(.1)(z)

    outputs = []  # will be vel_x at T+1, vel_y at T+1, yaw pwm at T+1

    for i in range(3):
        a = layers.Dense(64, activation='relu')(z)
        a = layers.Dropout(.1)(a)
        a = layers.Dense(64, activation='relu')(a)
        a = layers.Dropout(.1)(a)
        outputs.append(layers.Dense(1, activation='linear', name='output' + str(i))(a)) 

    model = models.Model(inputs=[img_in, metadata], outputs=outputs)

    return model

#---------------------------------------------------------------------------------
# Basic model without velocity help based on Donkeycar

def donkeycar_model_no_height_control_no_velocity():  
    input_shape = (300, 300, 3)
    
    img_in = layers.Input(shape=input_shape, name='img_in')

    #targetdistance, pathdistance, headingdelta, x at T, y at T, yaw_pwm at T
    metadata = layers.Input(shape=(3,), name="path_distance_in")

    x = img_in
    x = layers.Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = layers.Flatten(name='flattened')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.1)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.1)(x)

    y = metadata
    y = layers.Dense(14, activation='relu')(y)
    y = layers.Dense(28, activation='relu')(y)
    y = layers.Dense(56, activation='relu')(y)

    z = layers.concatenate([x, y])
    z = layers.Dense(50, activation='relu')(z)
    z = layers.Dropout(.1)(z)
    z = layers.Dense(50, activation='relu')(z)
    z = layers.Dropout(.1)(z)

    outputs = []  # will be vel_x at T+1, vel_y at T+1, yaw pwm at T+1

    for i in range(3):
        a = layers.Dense(64, activation='relu')(z)
        a = layers.Dropout(.1)(a)
        a = layers.Dense(64, activation='relu')(a)
        a = layers.Dropout(.1)(a)
        outputs.append(layers.Dense(1, activation='linear', name='output' + str(i))(a)) 

    model = models.Model(inputs=[img_in, metadata], outputs=outputs)

    return model

#---------------------------------------------------------------------------------
# series model based on Donkeycar

def create_image_encoder(img_in):
    x = img_in
    x = layers.Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = layers.Flatten(name='flattened')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.1)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.1)(x)
    return x

def create_mlp(data_in):
    y = data_in
    y = layers.Dense(14, activation='relu')(y)
    y = layers.Dense(28, activation='relu')(y)
    y = layers.Dense(56, activation='relu')(y)
    return y

def series_donkeycar_model_no_height_control():  
    input_shape = (300, 300, 3)
    
    img_t2 = layers.Input(shape=input_shape, name='img_t-2')
    img_t1 = layers.Input(shape=input_shape, name='img_t-1')
    img_t0 = layers.Input(shape=input_shape, name='img_t0')

    #targetdistance, pathdistance, headingdelta, x, y 
    metadata_t2 = layers.Input(shape=(7,), name="metadata_t-2")
    metadata_t0 = layers.Input(shape=(7,), name="metadata_t0")

    x2 = create_image_encoder(img_t2)
    x1 = create_image_encoder(img_t1)
    x0 = create_image_encoder(img_t0)

    y2 = create_mlp(metadata_t2)    
    y0 = create_mlp(metadata_t0)

    z = layers.concatenate([x2, x1, x0, y2, y0])
    z = layers.Dense(100, activation='relu')(z)
    z = layers.Dropout(.1)(z)
    z = layers.Dense(50, activation='relu')(z)
    z = layers.Dropout(.1)(z)
    z = layers.Dense(50, activation='relu')(z)
    z = layers.Dropout(.1)(z)

    outputs = []  # will be vel_x, vel_y, yaw pwm

    for i in range(3):
        a = layers.Dense(64, activation='relu')(z)
        a = layers.Dropout(.1)(a)
        a = layers.Dense(64, activation='relu')(a)
        a = layers.Dropout(.1)(a)
        outputs.append(layers.Dense(1, activation='linear', name='output' + str(i))(a)) 

    model = models.Model(inputs=[img_t2,img_t1,img_t0,metadata_t2,metadata_t0], outputs=outputs)


    return model

#---------------------------------------------------------------------------------
# Transfer learning model with LSTM

def mobilenet_LSTM_MLP_2(metadata):
    y = metadata
    y = layers.Dense(14, activation='relu')(y)
    y = layers.Dense(28, activation='relu')(y)
    y = layers.Dense(56, activation='relu')(y)
    return y

def mobilenet_LSTM():  
    input_shape = (300, 300, 3)

    #targetdistance, pathdistance, headingdelta, x at T, y at T, yaw_pwm at T
    metadata1 = layers.Input(shape=(3,), name="meta1")
    metadata2 = layers.Input(shape=(3,), name="meta2")
    metadata3 = layers.Input(shape=(3,), name="meta3")
    metadata4 = layers.Input(shape=(3,), name="meta4")

    img_in = layers.Input(shape=(300, 300, 3), name='image1')

    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(300, 300, 3), pooling='avg')
    x = base_model(img_in)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(.1)(x)

    y1 = mobilenet_LSTM_MLP_2(metadata1)
    y2 = mobilenet_LSTM_MLP_2(metadata2)
    y3 = mobilenet_LSTM_MLP_2(metadata3)
    y4 = mobilenet_LSTM_MLP_2(metadata4)

    y = layers.concatenate([y1, y2, y3, y4]) 
    y = layers.Reshape((4,56))(y)

    y = layers.LSTM(128, input_shape=(4,56))(y)

    z = layers.concatenate([x,y])

    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dropout(.1)(z)
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dropout(.1)(z)

    outputs = []  # will be vel_x at T+1, vel_y at T+1, yaw pwm at T+1

    for i in range(3):
        a = layers.Dense(64, activation='relu')(z)
        a = layers.Dropout(.1)(a)
        a = layers.Dense(64, activation='relu')(a)
        a = layers.Dropout(.1)(a)
        outputs.append(layers.Dense(1, activation='linear', name='output' + str(i))(a)) 

    model = models.Model(inputs=[img_in, metadata1, metadata2, metadata3, metadata4], outputs=outputs)

    return model


#---------------------------------------------------------------------------------
# Serie Transfer learning model 



# CNN + Transformer models

model = mobilenet_LSTM()

utils.plot_model(model,"model.png",show_shapes=True)