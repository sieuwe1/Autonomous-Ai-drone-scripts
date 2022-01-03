#convert .h5 model to .pb model 
from adabelief_tf import AdaBeliefOptimizer
import tensorflow as tf
import os
import tf2onnx

model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/sets/VELOCITY_full_set_shuffle_linear/Donkeycar_VELOCITY.h5'
model_name = os.path.splitext(os.path.basename(model_dir))[0]
print(model_name)
name = "CONVERTED_" + model_name
converted_model_dir = os.path.join(os.path.dirname(model_dir),name)

#load
custom_objects = {"AdaBeliefOptimizer": AdaBeliefOptimizer}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(model_dir)

(onnx_model_proto, storage) = tf2onnx.convert.from_keras(model)
with open('Donkeycar_velocity.onnx', "wb") as f:
    f.write(onnx_model_proto.SerializeToString())

print("done")

#save
#os.mkdir(converted_model_dir)
#tf.saved_model.save(model, converted_model_dir)

# now run 