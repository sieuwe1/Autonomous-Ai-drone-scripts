#convert .h5 model to .pb model 
from adabelief_tf import AdaBeliefOptimizer
import tensorflow as tf
import os
import tensorflow as tf
from tensorflow.python.framework import graph_io

model_fname = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/model/Donkeycar_new_preprocessor_shuffled_occi_linear_in_linear_out_sigmoid_lr_0.001.h5'
model_name = os.path.splitext(os.path.basename(model_fname))[0]
print(model_name)
name = "CONVERTED_" + model_name
converted_model_dir = os.path.join(os.path.dirname(model_fname),name)

#load
custom_objects = {"AdaBeliefOptimizer": AdaBeliefOptimizer}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(model_fname)

os.mkdir(converted_model_dir)
tf.saved_model.save(model, converted_model_dir)

params = tf.experimental.tensorrt.ConversionParams(
    precision_mode='FP16')
converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=converted_model_dir, conversion_params=params)

converter.convert()
converter.save("tensorRT_model")
