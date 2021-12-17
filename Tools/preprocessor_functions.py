import os
import glob
import json
import cv2
from keras.backend import dtype
import numpy as np
from numpy.random.mtrand import shuffle
from tensorflow import keras
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def augment_images(images, data):
    print("before augmentation data length: ", len(images))
    #print(np.array(images).shape)

    datagen = ImageDataGenerator(width_shift_range=[-30,30], height_shift_range=[-30,30],rotation_range=5,brightness_range=[0.2,1.0],)
    # prepare iterator
    it = datagen.flow(np.array(images), batch_size=1)

    for x in range(500):

        
        plt.imshow(images[x])
        for i in range(5):
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imshow(image)

        plt.show()


    #return images, data

##folder loading methods

def load_folder(data_folder):

    #file_count = sum([len(files) for r, d, files in os.walk(data_folder)])
    #print(file_count)
    
    folder_pattern = os.path.join(data_folder,'Run*/')
    folders = sorted(glob.glob(folder_pattern))
    print("folders: ", str(folders))

    sample_count = 0
    data = []
    images = []
    for folder in folders:
        print("loading: ", str(folder))
        fcount = 1
        while True:
            json_pattern = os.path.join(folder + f"control/record_{fcount}.json")
            if not os.path.exists(json_pattern):
                break
            
            if fcount < 4000:
                print("loading file: ", json_pattern)

            if os.path.getsize(json_pattern) > 250: #check if file is empty
                sample = np.zeros(shape=(12,1))
                f = open(json_pattern)
                json_data = json.load(f)
                sample[0] = json_data['user/roll']
                sample[1] = json_data['user/pitch']
                sample[2] = json_data['user/yaw']
                sample[3] = json_data['user/throttle']
                sample[4] = json_data['gps/speed']
                sample[5] = json_data['gps/target_distance']
                sample[6] = json_data['gps/path_distance']
                sample[7] = json_data['gps/heading_delta']
                sample[8] = json_data['gps/altitude']
                sample[9] = json_data['imu/vel_x']
                sample[10] = json_data['imu/vel_y']
                sample[11] = json_data['imu/vel_z']
                image_path = json_data['cam/image_name']
                img = cv2.imread(folder + str("left_camera/") + image_path)#float16 to use less memory 
                img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA).astype(np.float16) 
                images.append(img)
                data.append(sample)
                sample_count+=1
            fcount += 1

    return (data,images, sample_count)

##normalization methods

def normalize_inception_image(img):
    result = imagenet_utils.preprocess_input(img, data_format=None, mode='tf') #inception uses 'tf' mode
    return result

def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def sinusoidal_embedding(x: np.array, dim=256, max_value=10000):
    position_enc = np.array([[_x / np.power(max_value, 2*i/dim) for i in range(dim)] for _x in x])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return position_enc

def normalize_data_z_score_single(data):
    data = np.array(data)
    data_normalized = (data[:,0] - data[:,0].mean()) / data[:,0].std()
    return data_normalized

def normalize_data_linear_single(data):
    bounds = (min([a[0] for a in data]), max([a[0] for a in data]))
    data_normalized = np.zeros((len(data),),dtype=np.float32)

    for i in range(len(data)):
        data_normalized[i] = map(data[i],bounds[0],bounds[1],0,1)

    return data_normalized


def normalize_data_z_score(data, sample_count):

    data = np.array(data)
    bounds_data = [0,0,0,0,0,0,0,0]

    roll_normalized = (data[:,0] - data[:,0].mean()) / data[:,0].std()              #roll 
    pitch_normalized = (data[:,1] - data[:,1].mean()) / data[:,1].std()             #pitch
    yaw_normalized = (data[:,2] - data[:,2].mean()) / data[:,2].std()               #yaw 
    throttle_normalized = (data[:,3] - data[:,3].mean()) / data[:,3].std()          #throttle

    data[:,0] = (data[:,4] - data[:,4].mean()) / data[:,4].std()                    #speed
    data[:,1] = (data[:,5] - data[:,5].mean()) / data[:,5].std()                    #target_distance
    data[:,2] = (data[:,6] - data[:,6].mean()) / data[:,6].std()                    #path_distance
    data[:,3] = (data[:,7] - data[:,7].mean()) / data[:,7].std()                    #heading_delta
    data[:,4] = (data[:,8] - data[:,8].mean()) / data[:,8].std()                    #altitude
    data[:,5] = (data[:,9] - data[:,9].mean()) / data[:,9].std()                    #vel_x
    data[:,6] = (data[:,10] - data[:,10].mean()) / data[:,10].std()                  #vel_y
    data[:,7] = (data[:,11] - data[:,11].mean()) / data[:,11].std()                  #vel_z

    print("z_scores")
    print("speed mean: " + str(data[:,4].mean()) + "std: " + str(data[:,4].std()))
    print("target_distance mean: " + str(data[:,5].mean()) + "std: " + str(data[:,5].std()))
    print("path_distance mean: " + str(data[:,6].mean()) + "std: " + str(data[:,6].std()))
    print("heading_delta mean: " + str(data[:,7].mean()) + "std: " + str(data[:,7].std()))
    print("altitude mean: " + str(data[:,8].mean()) + "std: " + str(data[:,8].std()))
    print("vel_x mean: " + str(data[:,9].mean()) + "std: " + str(data[:,9].std()))
    print("vel_y mean: " + str(data[:,10].mean()) + "std: " + str(data[:,10].std()))
    print("vel_z mean: " + str(data[:,11].mean()) + "std: " + str(data[:,11].std()))

    return bounds_data, data[:,0:8], roll_normalized, pitch_normalized, yaw_normalized, throttle_normalized

def normalize_min_max(data, sample_count):
    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)

    return (min_max_scaler.data_min_ ,min_max_scaler.data_max_ ), data_scaled[:,4:11], data_scaled[:,0], data_scaled[:,1], data_scaled[:,2], data_scaled[:,3]


def normalize_data_linear(data, sample_count):

    bounds_data = []
    bounds_data.append((min([a[0] for a in data]), max([a[0] for a in data]))) #roll
    bounds_data.append((min([a[1] for a in data]), max([a[1] for a in data]))) #pitch
    bounds_data.append((min([a[2] for a in data]), max([a[2] for a in data]))) #yaw 
    bounds_data.append((min([a[3] for a in data]), max([a[3] for a in data]))) #throttle
    bounds_data.append((min([a[4] for a in data]), max([a[4] for a in data]))) #speed
    bounds_data.append((min([a[5] for a in data]), max([a[5] for a in data]))) #target_distance
    bounds_data.append((min([a[6] for a in data]), max([a[6] for a in data]))) #path_distance
    bounds_data.append((min([a[7] for a in data]), max([a[7] for a in data]))) #heading_delta
    bounds_data.append((min([a[8] for a in data]), max([a[8] for a in data]))) #altitude
    bounds_data.append((min([a[9] for a in data]), max([a[9] for a in data]))) #vel_x
    bounds_data.append((min([a[10] for a in data]), max([a[10] for a in data]))) #vel_y
    bounds_data.append((min([a[11] for a in data]), max([a[11] for a in data]))) #vel_z

    data_normalized = np.zeros((len(data),8),dtype=np.float32)

    for i in range(len(data)):
        data_normalized[i][0] = map(data[i][4],bounds_data[4][0],bounds_data[4][1],0,1)
        data_normalized[i][1] = map(data[i][5],bounds_data[5][0],bounds_data[5][1],0,1)
        data_normalized[i][2] = map(data[i][6],bounds_data[6][0],bounds_data[6][1],0,1)
        data_normalized[i][3] = map(data[i][7],bounds_data[7][0],bounds_data[7][1],0,1)
        data_normalized[i][4] = map(data[i][8],bounds_data[8][0],bounds_data[8][1],0,1)
        data_normalized[i][5] = map(data[i][9],bounds_data[9][0],bounds_data[9][1],0,1)
        data_normalized[i][6] = map(data[i][10],bounds_data[10][0],bounds_data[10][1],0,1)
        data_normalized[i][7] = map(data[i][11],bounds_data[11][0],bounds_data[11][1],0,1)

    roll_normalized = np.vstack(np.array([map(a[0],bounds_data[0][0],bounds_data[0][1],0,1) for a in data]))
    pitch_normalized = np.vstack(np.array([map(a[1],bounds_data[1][0],bounds_data[1][1],0,1) for a in data]))
    yaw_normalized = np.vstack(np.array([map(a[2],bounds_data[2][0],bounds_data[2][1],0,1) for a in data]))
    throttle_normalized = np.vstack(np.array([map(a[3],bounds_data[3][0],bounds_data[3][1],0,1) for a in data]))

    return bounds_data, data_normalized, roll_normalized, pitch_normalized, yaw_normalized, throttle_normalized

def normalize_images(imagedata):
    for i in range(len(imagedata)):
        image = imagedata[i]
        imagedata[i] = normalize_inception_image(image)
    return np.array(imagedata,dtype=np.float16)

#data sorting methods
def get_input_data_from_data(data):
    input_data = []
    for sample in input_data:
        count = 0
        input_data_sample = []
        for val in sample:
            if count > 3:
                input_data_sample.append(val)
            count+=1
        input_data.append(input_data_sample)

    return input_data
