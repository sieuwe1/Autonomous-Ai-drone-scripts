import os
import glob
import random
import json
import time
import zlib
from os.path import basename, join, splitext, dirname
import pickle
import pandas as pd
import cv2
from matplotlib import image
from matplotlib import pyplot
from tensorflow.python import keras
import numpy as np

# Loading JSON into dataframe
path_to_json = '/home/koen/Downloads/run3_jeton/control'
path_to_img = '/home/koen/Downloads/run3_jeton/left_camera'

# Loading in control dataframes

json_pattern = os.path.join(path_to_json,'*.json')
control_fl = glob.glob(json_pattern)

control_dfs = []
for file in control_fl:
    with open(file) as f:
        json_data = pd.json_normalize(json.loads(f.read()))
    control_dfs.append(json_data)
control_df = pd.concat(control_dfs, sort=False)

# Filtering out input data
control_df = \
    control_df[[
        "user/roll",
        "user/pitch"
    ]]

# Loading in left_camera images into dataframes

# img_pattern = os.path.join(path_to_img,'*.jpg')
# img_fl = glob.glob(img_pattern)
#
# img_df = [cv2.imread(file) for file in img_fl]
#
# print(control_df.head())
#
# cv2.imshow('image',img_df[0])
# cv2.waitKey(0)
# Creating validation and training dataframe
# val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
# train_dataframe = dataframe.drop(val_dataframe.index)



