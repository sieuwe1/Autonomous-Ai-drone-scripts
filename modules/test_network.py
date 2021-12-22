import cv2
import json
import time 
import numpy as np
import collections
from tensorflow import keras
from keras.applications import imagenet_utils
import network
from adabelief_tf import AdaBeliefOptimizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#folder = "/home/drone/Desktop/dataset_POC/Training/Run6" #input("Please type root direction of data folder: ")
folder = '/home/drone/Desktop/recordings/RunOcciTest'
playback_speed = 0.03 #0.03
count = 1
transfer = False
#model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules/trained_best_model_full_set.h5'
model_dir = '/home/drone/Desktop/Autonomous-Ai-drone-scripts/model/Inception_new_preprocessor_shuffled_occi_linear_in_linear_out_sigmoid_lr_0.001.h5'

plot_data = []
fig, ax = plt.subplots()

moving_averages = []
moving_average_length = 4

for i in range(4):
    moving_averages.append(collections.deque(maxlen=moving_average_length))

def scale_z_score(data, mean, std):
    return (data - mean) / std

def average(lst):
    return sum(lst) / len(lst)

def predict(model, img, json_data):

    img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    normalizedImg = imagenet_utils.preprocess_input(img, data_format=None, mode='tf') #inception uses 'tf' mode

    #used for z score input
    #speed = scale_z_score(json_data['gps/speed'], 0, 1)
    #target_distance = scale_z_score(json_data['gps/target_distance'], 0, 1.0)
    #path_distance = scale_z_score(json_data['gps/path_distance'], 0, 1.0)
    #heading_delta = scale_z_score(json_data['gps/heading_delta'], 0, 1.0)
    #altitude = scale_z_score(json_data['gps/altitude'], 0, 1.254)
    #vel_x = scale_z_score(json_data['gps/vel_x'], 0.081, 1.195)
    #vel_y = scale_z_score(json_data['gps/vel_y'], -0.415, 1.307)
    #vel_z = scale_z_score(json_data['gps/vel_z'], -0.022, 0.332)


    #used for linear input

    #sample.append(json_data['gps/latitude'])
    #sample.append(json_data['gps/longtitude'])
    speed = map_decimal(json_data['gps/speed'],0.038698211312294006, 3.2179622650146484 , 0,1)
    target_distance = map_decimal(json_data['gps/target_distance'], 2.450411854732669, 80.67362590478994, 0,1)
    path_distance = map_decimal(json_data['gps/path_distance'], 0, 13.736849872936324, 0,1)
    heading_delta = map_decimal(json_data['gps/heading_delta'], 0.012727423912849645, 155.99795402967004, 0,1)
    altitude = map_decimal(json_data['gps/altitude'], 0.093, 6.936, 0,1)
    vel_x = map_decimal(json_data['imu/vel_x'], -3.19, 3.03, 0,1)
    vel_y = map_decimal(json_data['imu/vel_y'], -3.03, 1.46, 0,1)
    vel_z = map_decimal(json_data['imu/vel_z'], -1.17, 1.07, 0,1)
    
    data = np.array([speed, target_distance, path_distance, heading_delta, altitude, vel_x, vel_y, vel_z])
    print("data: ", data)
    sample_to_predict = [normalizedImg.reshape((1,300,300,3)), data.reshape((1,8))]
    preds = model.predict(sample_to_predict)

    moving_averages[0].append(preds[0][0]) 
    moving_averages[1].append(preds[1][0]) 
    moving_averages[2].append(preds[2][0]) 
    moving_averages[3].append(preds[3][0]) 

    smooth_predicted_throttle = average(moving_averages[3])
    smooth_predicted_yaw = average(moving_averages[2])
    smooth_predicted_pitch = average(moving_averages[1])
    smooth_predicted_roll = average(moving_averages[0])



    return (smooth_predicted_roll,smooth_predicted_pitch,smooth_predicted_yaw,smooth_predicted_throttle)

def scale_z_score(data, mean, std):
    return mean + std * data

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

def drawUI(img, data, predicted):
    cv2.putText(img, "TargetDistance: " + str(data['gps/target_distance']), (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA)     
    cv2.putText(img, "PathDistance: " + str(data['gps/path_distance']), (50,100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "HeadingDelta: " + str(data['gps/heading_delta']), (50,150), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "Speed: " + str(data['gps/speed']), (50,200), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "altitude: " + str(data['gps/altitude']), (50,250), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 

    cv2.putText(img, "x: " + str(data['imu/vel_x']), (1000,150), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "y: " + str(data['imu/vel_y']), (1000,200), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.putText(img, "z: " + str(data['imu/vel_z']), (1000,250), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 

    #draw original
    cv2.rectangle(img, (50, 500), (250, 700), (255,0,0), 2)
    cv2.rectangle(img, (1000, 500), (1200, 700), (255,0,0), 2)

    throttle = map(data['user/throttle'], 1000, 2000, 500, 700)
    yaw = map(data['user/yaw'], 1000, 2000, 50, 250)

    pitch = map(data['user/pitch'], 1000, 2000, 500, 700)
    roll = map(data['user/roll'], 1000, 2000, 1000, 1200)

    cv2.circle(img,(yaw,throttle),20,(255,0,0),-1)
    cv2.circle(img,(roll,pitch),20,(255,0,0),-1)

    #draw predicted
    predicted_throttle = map(predicted[3], 0, 1, 1374, 1787)
    predicted_yaw = map(predicted[2], 0, 1, 1159, 1990)
    predicted_pitch = map(predicted[1], 0, 1, 1127, 2000)
    predicted_roll = map(predicted[0], 0, 1, 1091, 1826)

    #print("throttle: ", predicted_throttle)
    #print("yaw: ", predicted_yaw)
    #print("pitch: ", predicted_pitch)
    #print("roll: ", predicted_roll)

    throttle = map(predicted_throttle, 1374, 1787, 500, 700)
    yaw = map(predicted_yaw, 1159, 1990, 50, 250)

    pitch = map(predicted_pitch, 1127, 2000, 500, 700)
    roll = map(predicted_roll, 1091, 1826, 1000, 1200)

    cv2.circle(img,(yaw,throttle),20,(0,0,255),-1)
    cv2.circle(img,(roll,pitch),20,(0,0,255),-1)

    return img

model = None

#create model
if transfer:
    model, base_model = network.create_transfer_model()
    model.load_weights(model_dir)

else:
    custom_objects = {"AdaBeliefOptimizer": AdaBeliefOptimizer}
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(model_dir)

def plot(data):
    plot_data.append(data)
    ax.cla()
    ax.plot(plot_data)

def main(i):
    global count
    f = open(folder + "/control/record_" + str(count) + ".json")
    data = json.load(f)
    img = cv2.imread(folder + "/left_camera/" + data['cam/image_name'])

    predicted = predict(model,img,data)

    #img = drawUI(img, data, predicted)

    #cv2.imshow("data visualizer", img)
    #cv2.waitKey(1)
    #time.sleep(playback_speed)
    print(predicted)
    plot(predicted[0])
    count +=1

ani = FuncAnimation(fig, main, interval=playback_speed)

plt.show()