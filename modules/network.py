# network file. Based on donkeycar keras.py file.
# %%

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose
from tensorflow.keras.applications import InceptionV3, VGG16


"""
Inception/MobileNet -> Transformer (one-image) -> LSTM

Inception/MobileNet(t) --\
Inception/MobileNet(t-1) --> Transformer (multi-image features)

Real Encoder (positional encoding) {8,256} --\
Inception/MobileNet(t-1) -----------------------> (concat,-2{8*k+n*3,256})---> Transformer (8+n,256)[0] -> MLP{1,4}

Real Encoder (positional encoding) {8,256} ---(concat,-2{8*k+n*3,256})---> Transformer (8+n,256)[0] -> MLP{1,4}
Inception/MobileNet(t){n,256} --------------/
Inception/MobileNet(t-1){n,256} -----------/
Inception/MobileNet(t-2){n,256} ----------/
"""

def create_transfer_model():

    #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = VGG16(weights='imagenet',include_top=False,input_shape=(300, 300, 3))


    image_input = Input(shape=(300, 300, 3), name='image')
    vgg16 = base_model(image_input)
    x = Flatten()(vgg16) 
    x = Dense(256, activation='relu')(x)

    #targetdistance, pathdistance, headingdelta, speed, altitude, x, y, z
    # Input(shape(8, 256)) # 1500 -> [0.5, 0.1, 0.3, 0.1 0.6] # 0.5 -> [0.1, 0.1]
    metadata = Input(shape=(8,), name="path_distance_in")

    y = metadata
    y = Dense(14, activation='relu')(y)
    y = Dense(28, activation='relu')(y)
    y = Dense(46, activation='relu')(y)

    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = []  # will be throttle, yaw, pitch, roll

    for i in range(4):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))

    model = Model(inputs=[image_input, metadata], outputs=outputs)

    return model, base_model

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
    
    img_in = Input(shape=input_shape, name='img_in')

    #targetdistance, pathdistance, headingdelta, speed, altitude, x, y, z
    metadata = Input(shape=(8,), name="path_distance_in")

    x = create_encoder(img_in)
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
        outputs.append(Dense(1, activation='sigmoid', name='out_' + str(i))(z))

    model = Model(inputs=[img_in, metadata], outputs=outputs)

    return model



#model = create_model()]
#model, base_model= create_transfer_model()

#print(base_model.summary())

dot_img_file = 'model_432.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

# %%
