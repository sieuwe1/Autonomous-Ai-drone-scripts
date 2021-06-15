import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.layers import Dense
from tensorflow.keras.layers import Input, Dropout, Convolution2D, Flatten
from tensorflow.python.keras.layers.merge import concatenate

INPUT_SHAPE = (300, 300, 3)  # 300x300 RGB


def create_dense(numerical_in):
    y = numerical_in
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    return y


def create_encoder(img_in):
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    return x


def create_model():  # img_right_in, img_left_in, img_front_in

    # Dense layer inputs
    path_distance_in = Input(shape=(1,), name="path_distance_in")
    bearing_in = Input(shape=(1,), name="bearing_in")
    acc_x_in = Input(shape=(1,), name="acc_x_in")
    acc_y_in = Input(shape=(1,), name="acc_y_in")
    acc_z_in = Input(shape=(1,), name="acc_z_in")
    alt_in = Input(shape=(1,), name="alt_in")

    # Define camera inputs
    camera_DEPTH_in = Input(shape=INPUT_SHAPE, name='img_depth_in')
    camera_LEFT_in = Input(shape=INPUT_SHAPE, name='camera_LEFT_in')
    camera_RIGHT_in = Input(shape=INPUT_SHAPE, name='camera_RIGHT_in')

    outputs = []

    # Create layers
    cam_D = create_encoder(camera_DEPTH_in)
    cam_L = create_encoder(camera_LEFT_in)
    cam_R = create_encoder(camera_RIGHT_in)
    metadata = create_dense(6)

    z = concatenate([cam_D, cam_L, cam_R, metadata])

    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    for i in range(3):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))

    model = Model(inputs=[camera_DEPTH_in, camera_LEFT_in,
                  camera_RIGHT_in, path_distance_in, bearing_in, acc_x_in, acc_y_in, acc_z_in, alt_in], outputs=outputs)

    return model
