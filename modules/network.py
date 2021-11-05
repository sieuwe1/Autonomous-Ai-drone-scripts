# network file. Based on donkeycar network.py file.
# %%

import os
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense, Add
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose


def create_encoder(img_in):
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    #X_shortcut = x
    #X_shortcut = Convolution2D(64, (5, 5), strides=(4, 4), activation='relu')(x)  should take into resnet like skips to improve performance once model becomes deeper
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    #x = Add()([x, X_shortcut])
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    return x

def create_model():  
    input_shape = (300, 300, 3)
    
    img_depth_in = Input(shape=input_shape, name='img_depth_in')

    #targetdistance, pathdistance, headingdelta, speed, altitude, x, y, z
    metadata = Input(shape=(8,), name="path_distance_in")

    x = create_encoder(img_depth_in)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)

    y = metadata
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)

    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = []  # will be throttle, yaw, pitch, roll

    for i in range(4):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))

    model = Model(inputs=[img_depth_in, metadata], outputs=outputs)

    return model


model = create_model()

dot_img_file = 'model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

# %%
