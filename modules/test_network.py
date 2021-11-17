import cv2
import json
import time 
import numpy as np
import attention
from tensorflow import keras

folder = "/home/sieuwe/Desktop/dataset_POC/Training/temp/Run5" #input("Please type root direction of data folder: ")

playback_speed = 0.04

count = 1

def predict(model, img, json_data):

    img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    normalizedImg = np.zeros((300, 300))
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 1, cv2.NORM_MINMAX)

    #sample.append(json_data['gps/latitude'])
    #sample.append(json_data['gps/longtitude'])
    speed = map(json_data['gps/speed'],0.09596491605043411, 3.1905386447906494 , 0,1)
    target_distance = map(json_data['gps/target_distance'], 6.205801853336117, 70.78423058915499, 0,1)
    path_distance = map(json_data['gps/path_distance'], 7.262124709253662e-10, 8.144912221746539, 0,1)
    heading_delta = map(json_data['gps/heading_delta'], 0.03087504148265907, 110.53662845988049, 0,1)
    altitude = map(json_data['gps/altitude'], 0.263, 5.052, 0,1)
    vel_x = map(json_data['imu/vel_x'], -2.84, 1.52, 0,1)
    vel_y = map(json_data['imu/vel_y'], -2.71, 1.41, 0,1)
    vel_z = map(json_data['imu/vel_z'], -1.15, 1.33, 0,1)
    
    data = np.array([speed, target_distance, path_distance, heading_delta, altitude, vel_x, vel_y, vel_z])
    #print(img.shape)

    preds = model.predict([[normalizedImg] , [data]])


    #last_conv_layer_name = "conv2d_11"
    #img_array = attention.get_img_array(normalizedImg, size=(300,300))
    #heatmap = attention.make_gradcam_heatmap(img_array, data, model, last_conv_layer_name)
    # Display heatmap
    #plt.matshow(heatmap)
    #plt.show()


    print(preds)
    
    return preds

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

model = keras.models.load_model('trained_best_model.h5')

while True:
    f = open(folder + "/control/record_" + str(count) + ".json")
    data = json.load(f)
    img = cv2.imread(folder + "/left_camera/" + data['cam/image_name'])
    
    predicted = predict(model,img,data)

    img = drawUI(img, data, predicted)

    cv2.imshow("data visualizer", img)
    cv2.waitKey(1)
    time.sleep(playback_speed)
    count +=1