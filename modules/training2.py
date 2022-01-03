# %%
import os
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import pyplot
from tensorflow import keras
from keras.models import Model
import numpy as np
import os
from time import time
from datetime import datetime
import velocity_networks
import tensorflow as tf
import h5py
from adabelief_tf import AdaBeliefOptimizer

transfer = False
model_name = 'Donkeycar_VELOCITY.h5'

training_size = 13648 #get these values from preprocessor
validation_size = 5848 #get these values from preprocessor
batch_size = 30

train_folder = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/sets/VELOCITY_full_set_shuffle_linear/train'
val_folder = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/sets/VELOCITY_full_set_shuffle_linear/val'

train_file_names = os.listdir(train_folder)
val_file_names = os.listdir(val_folder)

checkpoint_path = f'./checkpoints/{datetime.now()}_checkpoint'

class data_generator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, train):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.train = train

    def __len__(self):
        return int(np.ceil(self.x / float(self.batch_size))) # -2

    def __getitem__(self, index):
        if self.train == True:
            #load train data
            hf = h5py.File(train_folder + "/"  + train_file_names[index], 'r')
            img_x_train = np.array(hf.get('img_x_train'))
            data_x_train = np.array(hf.get('data_x_train'))
            y_vel_x_train = np.array(hf.get('y_vel_x_train'))
            y_vel_y_train = np.array(hf.get('y_vel_y_train'))
            y_yaw_train = np.array(hf.get('y_yaw_train'))
            hf.close()

            #print(data_x_train[:,:,0])
            
            #if np.prod(data_x_train.shape) == 0:
            #    print(train_file_names[index], data_x_train)
            #    return [np.zeros((0,300,300,3)),np.zeros((0,8,256))], [np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)]
            return [img_x_train, data_x_train], [y_vel_x_train,y_vel_y_train,y_yaw_train]
            ####################        [:,:,0]
            #return [img_x_train], [y_roll_train,y_pitch_train,y_yaw_train,y_throttle_train]

        else:
            hf = h5py.File(val_folder + "/" + val_file_names[index], 'r')
            img_x_val = np.array(hf.get('img_x_val'))
            data_x_val = np.array(hf.get('data_x_val'))
            y_vel_x_val = np.array(hf.get('y_vel_x_val'))
            y_vel_y_val = np.array(hf.get('y_vel_y_val'))
            y_yaw_val = np.array(hf.get('y_yaw_val'))
            hf.close()
            
            #if np.prod(data_x_val.shape) == 0:
            #    print(val_file_names[index], data_x_val)
            #    return [np.zeros((0,300,300,3)),np.zeros((0,8,256))], [np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)]
            return [img_x_val, data_x_val], [y_vel_x_val,y_vel_y_val,y_yaw_val]
            #return [img_x_val], [y_roll_val,y_pitch_val,y_yaw_val,y_throttle_val]

model = None

#create model
if transfer:
    #model, base_model = network.create_transformer_model()
    #model, base_model = network.create_transfer_model()
    #model, base_model = network.create_transfer_image_only_model()

    #for layer in base_model.layers:
    #    layer.trainable = False
    test = 0

else:
    model = velocity_networks.donkeycar_model_no_height_control()

# compile model
crit = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
#opt = tf.keras.optimizers.Nadam(learning_rate=0.001)
opt = AdaBeliefOptimizer(learning_rate=1e-4, epsilon=1e-14, rectify=True, weight_decay=1e-4)

model.compile(loss=crit, optimizer=opt, metrics=['accuracy'])

# train model
callback_early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=.001)

# Creates checkpoint of the model that has achieved the best performance
callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# save logs with tensorflow
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f"./logs/{int(time())}")

#create generator
train_generator = data_generator(training_size,training_size,batch_size,True)
val_generator = data_generator(validation_size,validation_size,batch_size,False)

#train model
# history = model.fit_generator(generator=train_generator, steps_per_epoch = int(training_size // batch_size) - 1,epochs = 150,
#                    verbose = 1,validation_data = val_generator, validation_steps = int(validation_size // batch_size) - 1,     #Why -1? can someone research this?
#                    callbacks=[callback_early_stop,tensorboard_callback,callback_model_checkpoint], shuffle=False)

history = model.fit_generator(generator=train_generator, steps_per_epoch = int(training_size // batch_size) - 1,epochs = 500,
                   verbose = 1,validation_data = val_generator, validation_steps = int(validation_size // batch_size) - 1,     #Why -1? can someone research this?
                   callbacks=[callback_early_stop,tensorboard_callback,callback_model_checkpoint], shuffle=False)


# %%

# save model
def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

#model_freezed = freeze_layers(model)
model.save_weights("weights_" + model_name)
model.save(model_name)

print(history.history)

# plot summary
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
