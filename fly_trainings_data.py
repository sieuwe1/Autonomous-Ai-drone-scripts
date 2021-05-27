import sys, time, os
import argparse
sys.path.insert(1, 'modules')

import cv2
import collections
import json
import numpy as np
import math

import camera
import drone

STATE = "init" # init, flight, flight_record, shutdown
record_button_channel = 8

frame_count = 0
left_camera = None
right_camera = None
data_dir = None
cam_left_dir = None
cam_right_dir = None
cam_depth_dir = None
control_dir = None

positions = collections.deque(maxlen=60) #if system runs 30fps then 60fps is 2 seconds 


#def calculate_path():
#    positions.append(drone.get_location())
#    print(positions[0])
#    #if len(positions) == 60:
#    return
#    #calculate straight line trought points. Then add end point to the end of the line + 50 meters.         

#def calculate_path_distance():
#    #calcualte distance between the line and drone and between end point and drone using path calcualted from calulate_path()
#    return

def calculate_path_distance(target, start):
    current_cordinate = drone.get_location()
    current = np.asarray((current_cordinate.lat,current_cordinate.lon))
    path_distance = np.linalg.norm(np.cross(start-target,target-current))/np.linalg.norm(start-target)
    target_distance = math.sqrt(((target[0]- start[0])**2)+((target[1]- start[1])**2))
    print("target distance: " + str(target_distance))    
    print("path distance: " + str(path_distance))
    return target_distance + path_distance
    

def setup_writer():
    global data_dir, cam_left_dir, cam_right_dir, cam_depth_dir, control_dir

    dataName = input("please type name of this run. This will be the name of the Data folder > ")
    print("you entered: " + str(dataName))

    current_directory = os.getcwd()
    data_dir = os.path.join(current_directory, dataName)
    try:
        os.mkdir(data_dir)
        print("Directory " , data_dir ,  " Created ") 
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
    cam_name = str(framecount) + '_cam-image_cam_array.jpg'
    cam_path = os.path.join(path, cam_name)
    cv2.imwrite(cam_path, img) 
    return cam_name

def write_train_data(left_img, right_img, depth_img, roll, pitch, throttle, yaw, distance, framecount):

    left_cam = write_image(left_img,cam_left_dir,framecount)
    right_cam = write_image(right_img,cam_right_dir,framecount)
    depth_cam = "yoloooo" #write_image(depth_img,cam_depth_dir,framecount)   
    print(roll)
    print(pitch)
    print(throttle)
    print(yaw)
    print(framecount) 

    json_data = {"user/roll": roll, "user/pitch": pitch, "user/throttle": throttle, "user/yaw": yaw, "gps/distance": distance, "cam/image_array_left": left_cam, "cam/image_array_right": right_cam, "cam/image_array_depth": depth_cam, "user/mode": "user"}
    
    jsonName = "record_" + str(framecount) + '.json'
    jsonPath = os.path.join(control_dir, jsonName)
    with open(jsonPath, "w") as write_file:
        json.dump(json_data, write_file)

def init():
    global left_camera, right_camera

    print("State = INIT -> " + STATE)

    setup_writer() #file path or something as param

    left_camera = camera.create_camera(0)
    right_camera = camera.create_camera(1)

    #init zed camera

    #drone.connect_drone('/dev/ttyACM0')
    drone.connect_drone('127.0.0.1:14551')

    return "flight_record"

def flight():
    print("State = FLIGHT -> " + STATE)

    while (drone.read_channel(record_button_channel) < 1500): #3000 good PWM value?
        print("record channel output: " + str(drone.read_channel(record_button_channel)))
        time.sleep(0.1)

    return "flight_record"

def flight_record():
    global frame_count

    print("State = FLIGHT_RECORD -> " + STATE) 

    start_cordinate = drone.get_location()
    start = np.asarray((start_cordinate.lat,start_cordinate.lon))
    
    #mission = drone.get_mission()
    #print(mission)
    #target_cordinate = 
    target = np.asarray((51.4519143,5.4537731))

    #while (drone.read_channel(record_button_channel) > 1500): #3000 good PWM value?
    while True:
        frame_count += 1

        left_img = camera.get_video(0)
        right_img = camera.get_video(1)
        
        depth_img = None

        roll = 0 # drone.read_channel(1) # not needed
        pitch = 0 # drone.read_channel(2) #forward/backward
        throttle = 0 # drone.read_channel(3) #up/down
        yaw = 0 # drone.read_channel(4) #yaw

        distance = calculate_path_distance(target,start)

        write_train_data(left_img, right_img, depth_img, roll, pitch, throttle, yaw, distance, frame_count)

    return "flight"

def shutdown():
    print("State = SHUTDOWN -> " + STATE)

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