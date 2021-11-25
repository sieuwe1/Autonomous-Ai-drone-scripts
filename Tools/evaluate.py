from numpy.lib.function_base import average, median
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts_new/models/trained_best_model_full_set_LINEAR.h5'
test_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts_new/Tools/eval_data.npy'
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

results = []
for i in range(len(img_x_test)):
    sample_to_predict = [img_x_test[i].reshape((1,300,300,3)), np.array(data_x_test[i]).reshape((1,8))]
    preds = model.predict(sample_to_predict)

    mse = keras.losses.MeanSquaredError()
    roll_error = mse(y_roll_test[i], preds[0][0]).numpy()  
    pitch_error = mse(y_pitch_test[i], preds[1][0]).numpy()  
    yaw_error = mse(y_yaw_test[i], preds[2][0]).numpy()  
    throttle_error = mse(y_throttle_test[i], preds[3][0]).numpy()  

    results.append((roll_error,pitch_error,yaw_error,throttle_error))

average_mse = average(results)
roll_average_mse = average(results[0])
pitch_average_mse = average(results[1])
yaw_average_mse = average(results[2])
throttle_average_mse = average(results[3])

print("")
print("")
print("Average error: " + str(average_mse) + "   sample count: " + str(len(results)))
print("")
print("Average roll error: " + str(roll_average_mse))
print("")
print("Average pitch error: " + str(pitch_average_mse))
print("")
print("Average yaw error: " + str(yaw_average_mse))
print("")
print("Average throttle error: " + str(throttle_average_mse))

plt.plot(results)
plt.title('model mse')
plt.legend(['roll', 'pitch', 'yaw', 'throttle'], loc='upper left')
plt.ylabel('mse')
plt.xlabel('sample')
plt.savefig(model_dir + '_accuracy.png')
plt.show()

#results = model.evaluate([np.array(img_x_test).astype(np.float32), data_x_test], 
#[np.array(y_roll_test).astype(np.float32),np.array(y_pitch_test).astype(np.float32),
#np.array(y_yaw_test).astype(np.float32),np.array(y_throttle_test).astype(np.float32)],verbose=True, batch_size=128)

#print("test loss, test acc:", results)

