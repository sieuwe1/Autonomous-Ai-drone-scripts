#%%
from numpy.lib.function_base import average, median
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

model_name = 'trained_best_model_full_set_LINEAR.h5'
model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/models/trained_best_model_full_set_LINEAR.h5'
test_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/data/eval_data.npy'
data = np.load(test_dir,allow_pickle=True)

#load train data
img_x_test = data[0]
data_x_test = data[1]
y_roll_test = data[2]
y_pitch_test = data[3]
y_yaw_test= data[4]
y_throttle_test = data[5]

model = keras.models.load_model(model_dir)

print("Evaluate on test data")

#sample_to_predict = [normalizedImg.reshape((1,300,300,3)), data.reshape((1,8))]
#%%
results = []
for i in range(len(img_x_test)):
    sample_to_predict = [img_x_test[i].reshape((1,300,300,3)), np.array(data_x_test[i]).reshape((1,8))]
    preds = model.predict(sample_to_predict)

    pitch_error = abs(y_pitch_test[i]- preds[1][0])  
    yaw_error = abs(y_yaw_test[i] - preds[2][0])
    throttle_error = abs(y_throttle_test[i] - preds[3][0])

    results.append((y_roll_test[i],preds[0][0][0],y_pitch_test[i], preds[1][0][0], y_yaw_test[i], preds[2][0][0], y_throttle_test[i], preds[3][0][0], i))
#%%
mse = keras.losses.MeanSquaredError()

results = np.array(results)

roll_mse = mse(results[:,0],results[:,1]).numpy()
pitch_mse = mse(results[:,2],results[:,3]).numpy()
yaw_mse = mse(results[:,4],results[:,5]).numpy()
throttle_mse = mse(results[:,6],results[:,7]).numpy()

average_mse = (roll_mse+pitch_mse+yaw_mse+throttle_mse) / 4

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
print("Average throttle error: " + str(throttle_mse))

fig, axs = plt.subplots(2, 2)
fig.suptitle(model_name, fontsize=16)
fig.patch.set_facecolor('white')
axs[0, 0].plot(results[:,0])
axs[0, 0].plot(results[:,1])
axs[0, 0].legend(['actual', 'predicted'], loc='upper left')
axs[0, 0].set_title('Roll')
axs[0, 1].plot(results[:,2])
axs[0, 1].plot(results[:,3])
axs[0, 1].set_title('Pitch')
axs[1, 0].plot(results[:,4])
axs[1, 0].plot(results[:,5])
axs[1, 0].set_title('yaw')
axs[1, 1].plot(results[:,6])
axs[1, 1].plot(results[:,7])
axs[1, 1].set_title('throttle')

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
