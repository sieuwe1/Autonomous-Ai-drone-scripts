import cv2
import json
import time 
import numpy as np
from tensorflow import keras
import network
import h5py

#folder = "/home/drone/Desktop/dataset_POC/Training/Run6" #input("Please type root direction of data folder: ")
#folder = '/home/drone/Desktop/dataset_POC/Testing/Run1'
folder = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/Tools/eval_data.h5'

playback_speed = 0.5 #0.03
count = 0
transfer = False
#model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules/trained_best_model_full_set.h5'
model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules/donkeycar_new_preprocessor_sigmoid.h5'

def predict(model, img_x_test, data_x_test):

    sample_to_predict = [img_x_test.reshape((1,300,300,3)), np.array(data_x_test).reshape((1,8))]
    preds = model.predict(sample_to_predict)

    #print(preds)

    return preds

def map_decimal(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return round(rightMin + (valueScaled * rightSpan),5)

def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return round(rightMin + (valueScaled * rightSpan))

def drawUI(img_norm, y_roll_test,y_pitch_test,y_yaw_test,y_throttle_test, predicted):

    img_norm+=1.
    img_norm*=127.5
    img = np.array(img_norm,dtype=np.uint8)
    img = cv2.resize(img, (1080,720), interpolation = cv2.INTER_AREA)

    #draw original
    cv2.rectangle(img, (50, 500), (250, 700), (255,0,0), 2)
    cv2.rectangle(img, (1000, 500), (1200, 700), (255,0,0), 2)

    throttle = map(y_throttle_test, 0, 1, 500, 700)
    yaw = map(y_yaw_test, 0, 1, 50, 250)

    pitch = map(y_pitch_test, 0, 1, 500, 700)
    roll = map(y_roll_test, 0, 1, 1000, 1200)

    cv2.circle(img,(yaw,throttle),20,(255,0,0),-1)
    cv2.circle(img,(roll,pitch),20,(255,0,0),-1)

    #draw predicted
    predicted_throttle = predicted[3]
    predicted_yaw = predicted[2]
    predicted_pitch = predicted[1]
    predicted_roll = predicted[0]

    #print("throttle: ", predicted_throttle)
    #print("yaw: ", predicted_yaw)
    #print("pitch: ", predicted_pitch)
    #print("roll: ", predicted_roll)

    throttle = map(predicted_throttle, 0, 1, 500, 700)
    yaw = map(predicted_yaw, 0, 1, 50, 250)

    pitch = map(predicted_pitch, 0, 1, 500, 700)
    roll = map(predicted_roll, 0, 1, 1000, 1200)

    cv2.circle(img,(yaw,throttle),20,(0,0,255),-1)
    cv2.circle(img,(roll,pitch),20,(0,0,255),-1)

    return img

model = None

#create model
if transfer:
    model, base_model = network.create_transfer_model()
    model.load_weights(model_dir)

else:
    model = keras.models.load_model(model_dir)

hf = h5py.File(folder, 'r')
img_x_test = np.array(hf.get('img_x_train'))
data_x_test = np.array(hf.get('data_x_train'))
y_roll_test = np.array(hf.get('y_roll_train'))
y_pitch_test = np.array(hf.get('y_pitch_train'))
y_yaw_test = np.array(hf.get('y_yaw_train'))
y_throttle_test = np.array(hf.get('y_throttle_train'))
hf.close()

while True:

    predicted = predict(model,img_x_test[count],data_x_test[count])

    img = drawUI(img_x_test[count], y_roll_test[count],y_pitch_test[count],y_yaw_test[count],y_throttle_test[count], predicted)

    cv2.imshow("data visualizer", img)
    cv2.waitKey(1)
    time.sleep(playback_speed)
    print(count)
    count +=1