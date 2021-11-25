#%%
import os
import matplotlib.pyplot as plt
from os.path import basename, join, splitext, dirname
from matplotlib import image
from matplotlib import pyplot
from tensorflow import keras
import numpy as np
import os
from time import time
import network
import tensorflow as tf

transfer = True
model_name = 'Inception_sigmoid_MLP.h5'
data = np.load('/home/drone/Desktop/data.npy',allow_pickle=True)

#load train data
img_x_train = data[0]
data_x_train = data[1]
y_roll_train = data[2]
y_pitch_train = data[3]
y_yaw_train = data[4]
y_throttle_train = data[5]

#load val data
img_x_val = data[6]
data_x_val = data[7]
y_roll_val = data[8]
y_pitch_val = data[9]
y_yaw_val = data[10]
y_throttle_val = data[11]

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
	epochs=150, batch_size=8, callbacks=[callback_early_stop,tensorboard_callback], shuffle=True)

#history = model.fit(x=[img_x_train, data_x_train], y=[y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train],
#	validation_data=([img_x_val, data_x_val], [y_roll_val,y_pitch_val,y_yaw_val,y_throttle_val]),
#	epochs=150, batch_size=8, callbacks=[callback_early_stop], shuffle=True)


#save model
#model.save_weights(model_name)
model.save(model_name)

print(history.history)

#plot summary
plt.plot(history.history['out_0_accuracy'])
plt.plot(history.history['val_out_0_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(model_name + '_accuracy.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(model_name + '_loss.png')

# %%
