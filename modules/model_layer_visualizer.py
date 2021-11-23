import sys
sys.path.insert(1, 'modules')
import attention
import cv2
import json
import matplotlib.pyplot as plt
import tensorflow.keras as keras


folder = "/home/drone/Desktop/dataset_POC/Training/Run3"
sample_count = 25

def predict(model, img, json_data):

    img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    normalizedImg = np.zeros((300, 300))
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 1, cv2.NORM_MINMAX)

    #sample.append(json_data['gps/latitude'])
    #sample.append(json_data['gps/longtitude'])
    speed = map(json_data['gps/speed'],0.038698211312294006, 3.2179622650146484 , 0,1)
    target_distance = map(json_data['gps/target_distance'], 2.450411854732669, 80.67362590478994, 0,1)
    path_distance = map(json_data['gps/path_distance'], 0, 13.736849872936324, 0,1)
    heading_delta = map(json_data['gps/heading_delta'], 0.012727423912849645, 155.99795402967004, 0,1)
    altitude = map(json_data['gps/altitude'], 0.093, 6.936, 0,1)
    vel_x = map(json_data['imu/vel_x'], -3.19, 3.03, 0,1)
    vel_y = map(json_data['imu/vel_y'], -3.03, 1.46, 0,1)
    vel_z = map(json_data['imu/vel_z'], -1.17, 1.07, 0,1)
    
    data = np.array([speed, target_distance, path_distance, heading_delta, altitude, vel_x, vel_y, vel_z])

    sample_to_predict = [normalizedImg.reshape((1,300,300,3)), data.reshape((1,8))]

    preds = model.predict(sample_to_predict)

    print(preds)
    
    return preds

#load 
f = open(folder + "/control/record_" + str(sample_count) + ".json")
data = json.load(f)
img = cv2.imread(folder + "/left_camera/" + data['cam/image_name'])

model = keras.models.load_model('/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules/trained_best_model_first.h5')

#predict
preds = predict(model,img,data)

#visualize
last_conv_layer_name = "conv2d_11"
img_array = attention.get_img_array(normalizedImg, size=(300,300))
heatmap = attention.make_gradcam_heatmap(img_array, data, model, last_conv_layer_name)

# Display heatmap
plt.matshow(heatmap)
plt.show()
