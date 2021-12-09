import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import config as c

data_folder = c.data_dir

def load_folder(data_folder):

    folder_pattern = os.path.join(data_folder,'*/')
    folders = sorted(glob.glob(folder_pattern))

    sample_count = 0
    data = []
    images = []
    print(folders)

    for folder in folders:
        json_pattern = os.path.join(folder + str("control/"),'*.json')
        control_files = sorted(glob.glob(json_pattern))

        print(folder)

        for control_file in control_files:
            if os.path.getsize(control_file) > 250: #check if file is empty
                sample = np.zeros(shape=(12,1))
                f = open(control_file)
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
                #sample.append(json_data['user/roll'])
                #sample.append(json_data['user/pitch'])
                #sample.append(json_data['user/yaw'])
                #sample.append(json_data['user/throttle'])
                #sample.append(json_data['gps/latitude'])
                #sample.append(json_data['gps/longtitude'])
                #sample.append(json_data['gps/speed'])
                #sample.append(json_data['gps/target_distance'])
                #sample.append(json_data['gps/path_distance'])
                #sample.append(json_data['gps/heading_delta'])
                #sample.append(json_data['gps/altitude'])
                #sample.append(json_data['imu/vel_x'])
                #sample.append(json_data['imu/vel_y'])
                #sample.append(json_data['imu/vel_z'])
                data.append(sample)
                sample_count+=1

    return np.array(data)

data = load_folder(data_folder)

print(data.shape)
print(data[5].shape)

plt.plot(data[:,7])
plt.ylabel('value')
plt.xlabel('sample')
plt.legend(['yolo'], loc='upper left')
plt.show()