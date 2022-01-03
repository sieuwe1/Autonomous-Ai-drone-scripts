#%%
from numpy.lib.function_base import average, median
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from adabelief_tf import AdaBeliefOptimizer
from sklearn.metrics import mean_squared_error
import time

transfer = False
model_name = 'Donkeycar_VELOCITY.001.h5'
model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/sets/VELOCITY_full_set_shuffle_linear/Donkeycar_VELOCITY.h5'
test_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/sets/VELOCITY_full_set_shuffle_linear/eval_data.h5'
#data = np.load(test_dir,allow_pickle=True)

#load train data
#img_x_test = data[0]
#data_x_test = data[1]
#y_roll_test = data[2]
#y_pitch_test = data[3]
#y_yaw_test= data[4]
#y_throttle_test = data[5]

hf = h5py.File(test_dir, 'r')
img_x_test = np.array(hf.get('img_x_train'))
data_x_test = np.array(hf.get('data_x_train'))
y_vel_x_test = np.array(hf.get('y_vel_x_train'))
y_vel_y_test = np.array(hf.get('y_vel_y_train'))
y_yaw_test = np.array(hf.get('y_yaw_train'))
hf.close()

model = None

custom_objects = {"AdaBeliefOptimizer": AdaBeliefOptimizer}
with keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model(model_dir)

print(model.summary())

print("Evaluate on test data")

#sample_to_predict = [normalizedImg.reshape((1,300,300,3)), data.reshape((1,8))]
#%%
results = []
for i in range(len(img_x_test)):
    print("data: ", data_x_test[i])
    start_time = time.time()
    sample_to_predict = [img_x_test[i].reshape((1,300,300,3)), np.array(data_x_test[i]).reshape((1,7))]
    
    preds = model.predict(sample_to_predict)
    end_time = time.time()
    fps = 1/(end_time-start_time)

    print("result: ", (y_vel_x_test[i],preds[0][0][0],y_vel_y_test[i], preds[1][0][0], y_yaw_test[i], preds[2][0][0], fps, i))

    results.append((y_vel_x_test[i],preds[0][0][0],y_vel_y_test[i], preds[1][0][0], y_yaw_test[i], preds[2][0][0], fps, i))
#%%
results = np.array(results)

roll_mse = mean_squared_error(results[:,0],results[:,1])
pitch_mse = mean_squared_error(results[:,2],results[:,3])
yaw_mse = mean_squared_error(results[:,4],results[:,5])
average_fps = average(results[:,6])

average_mse = (roll_mse+pitch_mse+yaw_mse) / 3

print("")
print("")
print("Average error: " + str(average_mse) + "   sample count: " + str(len(results)))
print("")
print("Average roll error: " + str(roll_mse))
print("")
print("Average pitch error: " + str(pitch_mse))
print("")
print("Average yaw error: " + str(yaw_mse))
print("")
print("FPS: " + str(average_fps ))
print("")
print("Memory peak: " + str(tf.config.experimental.get_memory_info('GPU:0')['peak']))

fig, axs = plt.subplots(2, 2)
fig.suptitle(model_name, fontsize=16)
fig.patch.set_facecolor('white')
axs[0, 0].plot(results[:,0])
axs[0, 0].plot(results[:,1])
#axs[0, 0].plot(np.abs(results[:,0] - results[:,1]))
axs[0, 0].legend(['actual', 'predicted', 'error'], loc='upper left')
axs[0, 0].set_title('Roll')
axs[0, 1].plot(results[:,2])
axs[0, 1].plot(results[:,3])
#axs[0, 1].plot(np.abs(results[:,2] - results[:,3]))
axs[0, 1].set_title('Pitch')
axs[1, 0].plot(results[:,4])
axs[1, 0].plot(results[:,5])
#axs[1, 0].plot(np.abs(results[:,4] - results[:,5]))
axs[1, 0].set_title('yaw')

plt.savefig(model_dir + '_accuracy.png')
plt.show()


#plt.plot(results[:,0]-results[:,1])
#plt.title('model error')
#plt.legend(['roll', 'pitch', 'yaw', 'throttle'], loc='upper left')
#plt.ylabel('mse')
#plt.xlabel('sample')

#results = model.evaluate([np.array(img_x_test).astype(np.float32), data_x_test], 
#[np.array(y_roll_test).astype(np.float32),np.array(y_pitch_test).astype(np.float32),
#np.array(y_yaw_test).astype(np.float32),np.array(y_throttle_test).astype(np.float32)],verbose=True, batch_size=128)

#print("test loss, test acc:", results)


# %%
