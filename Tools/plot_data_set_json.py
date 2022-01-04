import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler

data_folder = "/home/drone/Desktop/dataset_POC/Training"
#data_folder = "/home/drone/Desktop/dataset_POC/Testing"
#data_folder = "/home/drone/Desktop/dataset_POC/recordings"

def load_folder(data_folder):

    folder_pattern = os.path.join(data_folder,'Run*/')
    folders = sorted(glob.glob(folder_pattern))

    print("folders: ", str(folders))

    sample_count = 0
    data = []
    for folder in folders:
        print("loading: ", str(folder))
        fcount = 1
        while True:
            json_pattern = os.path.join(folder + f"control/record_{fcount}.json")
            if not os.path.exists(json_pattern):
                break
            
            print("loading file: ", json_pattern)

            if os.path.getsize(json_pattern) > 250: #check if file is empty
                sample = np.zeros(shape=(12,))
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
                data.append(sample)
                sample_count+=1
            
            fcount += 1

    return np.array(data)

def plot_histogram(data,key):
    trans = RobustScaler(quantile_range=(50, 50))
    data2 = trans.fit_transform(data) 

    min_max_scaler = MinMaxScaler()
    data3 = min_max_scaler.fit_transform(data2)   

    min_max_scaler2 = MinMaxScaler()
    data4 = min_max_scaler2.fit_transform(data)

    fig, axs = plt.subplots(2)

    axs[0].hist(data[:,key], density=True, bins=60)

    axs[1].hist(data3[:,key], density=True, bins=60)

    plt.show()

data = load_folder(data_folder)


plot_histogram(data,9)
plot_histogram(data,10)

# plt.plot(data[:,5])

#plot_data = (data[:,key] - data[:,key].min()) / data[:,key].max()
#plt.hist(plot_data, density=True, bins=int(100 * plot_data.std()))
#plt.hist(plot_data, density=True, bins=60)

