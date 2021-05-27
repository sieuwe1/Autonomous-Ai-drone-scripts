import sys
sys.path.insert(1, 'modules')
import drone
import camera
import zed
import json
import collections
import cv2
import time
import os
import argparse
import cv2
import collections
import json
import numpy as np
import math
import threading
import concurrent

record_button_channel = 8

STATE = "init"  # init, flight, flight_record, shutdown

frame_count = 0
left_camera = None
right_camera = None
data_dir = None
cam_left_dir = None
cam_right_dir = None
cam_depth_dir = None
control_dir = None

image_queue = []

def calculate_path_distance(target, start):
    current_cordinate = drone.get_location()
    current = np.asarray((current_cordinate.lat,current_cordinate.lon))
    path_distance = np.linalg.norm(np.cross(start-target,target-current))/np.linalg.norm(start-target)
    target_distance = math.sqrt(((target[0]- start[0])**2)+((target[1]- start[1])**2))
    print("target distance: " + str(target_distance))    
    print("path distance: " + str(path_distance))
    return target_distance, path_distance
    

def calculate_target(start):
    distance_to_target = 50 #meter
    heading = drone.get_heading() #0 = north. Value 0 to 360
    print("heading: " + str(heading))

    lat0 = math.cos(math.pi / 180.0 * start[0])
   
    a = math.radians(heading);  

    x = start[0] + (180/math.pi) * (distance_to_target / 6378137) / math.cos(lat0) * math.cos(a)
    y = start[1] + (180/math.pi) * (distance_to_target / 6378137) * math.sin(a)

    return (x,y)

def setup_writer():
    global data_dir, cam_left_dir, cam_right_dir, cam_depth_dir, control_dir

    dataName = input(
        "please type name of this run. This will be the name of the Data folder")
    print("you entered: " + str(dataName))

    current_directory = os.getcwd()
    data_dir = os.path.join(current_directory, dataName)
    try:
        os.mkdir(data_dir)
        print("Directory ", data_dir,  " Created ")
    except FileExistsError:
        print("Directory " , data_dir ,  " already exists") 
    
    cam_left_dir = data_dir + "/left_camera"
    cam_right_dir = data_dir + "/right_camera"
    cam_depth_dir = data_dir + "/depth_camera"
    control_dir = data_dir + "/control"
 
    os.mkdir(cam_left_dir)
    os.mkdir(cam_right_dir)
    os.mkdir(cam_depth_dir)
    os.mkdir(control_dir)

def write_image(img, path, framecount):
    print("I AM BEING CALLED")
    #img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    cam_name = str(framecount) + '_cam-image_cam_array.jpg'
    cam_path = os.path.join(path, cam_name)
    cv2.imwrite(cam_path, img) 
    return cam_name

def write_train_data(left_img, right_img, depth_img, target_distance, path_distance, roll, pitch, throttle, yaw, framecount):
    left_cam =  write_image(left_img,cam_left_dir,framecount)
    right_cam = write_image(right_img,cam_right_dir,framecount)
    depth_cam = write_image(depth_img,cam_depth_dir,framecount) 
    
    # x = threading.Thread(target=write_image, args=(left_img,cam_left_dir,framecount))
    # y = threading.Thread(target=write_image, args=(right_img,cam_right_dir,framecount))
    # z = threading.Thread(target=write_image, args=(depth_img, cam_depth_dir, framecount))
    # x.start()
    # y.start()
    # z.start()
    # print(roll)
    # print(pitch)
    # print(throttle)
    # print(yaw)
    # print(framecount) 

    json_data = {"user/roll": roll, "user/pitch": pitch, "user/throttle": throttle, "user/yaw": yaw, "gps/target_distance": target_distance, "gps/path_distance": path_distance,"cam/image_array_left": left_cam, "cam/image_array_right": right_cam, "cam/image_array_depth": depth_cam, "frame_count": framecount , "user/mode": "user"}
    
    jsonName = "record_" + str(framecount) + '.json'
    jsonPath = os.path.join(control_dir, jsonName)
    with open(jsonPath, "w") as write_file:
        json.dump(json_data, write_file)


def init():
    global depth_camera
    global left_camera, right_camera

    print("State = INIT -> " + STATE)

    drone.connect_drone('/dev/ttyACM0')

    setup_writer()  # file path or something as param

    left_camera = camera.create_camera(0)
    right_camera = camera.create_camera(1)
    depth_camera = zed.init_zed(performance_mode=True)

    drone.connect_drone('/dev/ttyACM0')
    #drone.connect_drone('127.0.0.1:14551')

    return "flight"


def flight():
    print("State = FLIGHT -> " + STATE)

    # 3000 good PWM value?
    while drone.read_channel(record_button_channel) < 1500:
        print("record channel output: " +
              str(drone.read_channel(record_button_channel)))
        time.sleep(0.1)

    return "flight_record"


def flight_record():
    global frame_count

    print("State = FLIGHT_RECORD -> " + STATE) 

    start_cordinate = drone.get_location()
    start = np.asarray((start_cordinate.lat,start_cordinate.lon))

    target = calculate_target(start) 

    print("start location: " + str(start))
    print("target location: " + str(target))

    # 3000 good PWM value?
    while drone.read_channel(record_button_channel) > 1500:
        start_time = time.time()
        frame_count += 1

        left_img = camera.get_video(0)
        right_img = camera.get_video(1)
        depth_img = zed.get_depth_image()
       
        #cv2.imshow("left", left_img)
        #cv2.imshow("right", right_img)
        #cv2.imshow("depth", depth_img)

        #cv2.waitKey(1)
        
        roll = drone.read_channel(1)  # not needed
        pitch = drone.read_channel(2)  # forward/backward
        throttle = drone.read_channel(3)  # up/down
        yaw = drone.read_channel(4)  # yaw
        target_distance, path_distance = calculate_path_distance(target,start)

        write_train_data(left_img, right_img, depth_img, target_distance, path_distance, roll, pitch, throttle, yaw, frame_count)
        
        end_time = time.time()
        print("FPS: " + str(1/(end_time-start_time)))

    return "flight"


def bulk_write_images():
    

def shutdown():
    print("State = SHUTDOWN -> " + STATE)
    depth_camera.close()
    camera.close_cameras()
    sys.exit(0)


while True:
    # main program loop

    if STATE == "init":
        STATE = init()

    elif STATE == "flight":
        STATE = flight()

    elif STATE == "flight_record":
        STATE = flight_record()

    elif STATE == "shutdown":
        STATE = shutdown()