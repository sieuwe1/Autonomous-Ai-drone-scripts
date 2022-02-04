#these networks predict velocity components and use other inputs then the pwm based networks
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import layers, backend
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
        a = layers.Dense(64, activation='calize using gps datarelu')(a)
        a = layers.Dropout(.1)(a)
        outputs.append(layers.Dense(1, activation='linear', name='output' + str(i))(a)) 

    model = models.Model(inputs=[img_t2,img_t1,img_t0,metadata_t2,metadata_t0], outputs=outputs)


    return model

#CNN + time series models. 

# learning in the wild model with changes. conv1d + cnn 

def conv1d_mobilenet_serie():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    image_input = layers.Input(shape=(300, 300, 3), name='image')
    data_input = layers.Input(shape=(32,7), name="data")

    f = 1.0
    g = 1.0
    latent_dim = 3
    output_length = 4

    # image encoder
    x = base_model(image_input)
    x = layers.Conv1D(128, kernel_size=1, strides=1, padding='valid', dilation_rate=1, use_bias=True)(x)
    x = layers.Flatten()(x)
    x = layers.Reshape(target_shape=(1,backend.int_shape(x)[1]))(x)

    x = layers.Conv1D(int(128 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1)(x)
    x = layers.LeakyReLU(alpha=1e-2)(x)
    x = layers.Conv1D(int(64 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1)(x)
    x = layers.LeakyReLU(alpha=1e-2)(x)
    x = layers.Conv1D(int(64 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1)(x)
    x = layers.LeakyReLU(alpha=1e-2)(x)
    x = layers.Conv1D(int(32 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1)(x)
    x = layers.LeakyReLU(alpha=1e-2)(x)

    x = layers.Permute((2,1))(x)
    x = layers.Conv1D(latent_dim, kernel_size=3, strides=1, padding='valid', dilation_rate=1, use_bias=True)(x)
    x = layers.Permute((2,1))(x)

    #data encoder
    y = data_input
    y = layers.Conv1D(int(64 * g), kernel_size=2, strides=1, padding='same',dilation_rate=1)(y)
    y = layers.LeakyReLU(alpha=.5)(y)
    y = layers.Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same',dilation_rate=1)(y)
    y = layers.LeakyReLU(alpha=.5)(y)
    y = layers.Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same',dilation_rate=1)(y)
    y = layers.LeakyReLU(alpha=.5)(y)
    y = layers.Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same',dilation_rate=1)(y)
    y = layers.LeakyReLU(alpha=.5)(y)

    y = layers.Permute((2,1))(y)
    y = layers.Conv1D(latent_dim, kernel_size=3, strides=1, padding='valid', dilation_rate=1, use_bias=True)(y)
    y = layers.Permute((2,1))(y)

    #head
    z = layers.concatenate([x,y])
    print(backend.int_shape(z))
    z = layers.Conv1D(int(64 * g), kernel_size=1, strides=1, padding='valid')(z)
    z = layers.LeakyReLU(alpha=.5)(z)
    z = layers.Conv1D(int(128 * g), kernel_size=1, strides=1, padding='valid')(z)
    z = layers.LeakyReLU(alpha=.5)(z)
    z = layers.Conv1D(int(128 * g), kernel_size=1, strides=1, padding='valid')(z)
    z = layers.LeakyReLU(alpha=.5)(z)
    z = layers.Conv1D(output_length, kernel_size=1, strides=1, padding='same')(z)

    print(backend.int_shape(z))

    model = models.Model(inputs=[image_input,data_input], outputs=z)
    return model

#----------------------------------------------------------------------------------
# LSTM time serie prediction model.

def LSTM_time_serie():

    input_shape = (300, 300, 3)
    
    img_in = layers.Input(shape=input_shape, name='img_in')

    #targetdistance, pathdistance, headingdelta, x at T, y at T, yaw_pwm at T
    metadata = layers.Input(shape=(10,7,), name="path_distance_in")

    x = img_in
    x = layers.Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = layers.Flatten(name='flattened')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.1)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.1)(x)
    
    y = layers.LSTM(20, input_shape=(10,7))(metadata) #GRU? #statefull = true en reset model state na epoch! want data is over de hele datatset gekoppeld aan elkaar. Elke batch is geen losse date point
    y = layers.Dense(1, activation='relu')(y)

    z = layers.concatenate([x, y])
    z = layers.Dense(50, activation='relu')(z)
    z = layers.Dropout(.1)(z)
    z = layers.Dense(50, activation='relu')(z)
    z = layers.Dropout(.1)(z)

    outputs = []  # will be vel_x at T+1, vel_y at T+1, yaw pwm at T+1
    a = layers.Dropout(.1)(z)
    outputs.append(layers.Dense(1, activation='linear', name='output' + str(0))(z)) 

    #for i in range(3):
    #    outputs.append(layers.TimeDistributed(layers.Dense(activation='linear', units=10))(z))
    #     a = layers.Dense(64, activation='relu')(z)
    #     a = layers.Dropout(.1)(a)
    #     a = layers.Dense(64, activation='relu')(a)
    #     a = layers.Dropout(.1)(a)
    #     outputs.append(layers.Dense(1, activation='linear', name='output' + str(i))(a)) 

    model = models.Model(inputs=[img_in, metadata], outputs=outputs)

    return model

# CNN + Transformer models

model = conv1d_mobilenet_serie()

dot_img_file = 'FAKKA_model.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

utils.plot_model(model,"model.png",show_shapes=True)