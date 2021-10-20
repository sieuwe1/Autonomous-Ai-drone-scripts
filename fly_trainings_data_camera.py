import sys
sys.path.insert(1, 'modules')
import drone
import camera
import json
import gps
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
 
target = (51.45047075041008, 5.454691543317034) #first run get_gps.py to get target location

record_button_channel = 6

STATE = "init"  # init, flight, flight_record, shutdown

frame_count = 0
left_camera = None
data_dir = None
cam_left_dir = None
control_dir = None
throttle_threshold = 500

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
    control_dir = data_dir + "/control"
 
    os.mkdir(cam_left_dir)
    os.mkdir(control_dir)

def write_image(img, path, cam_name):
    #img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    cam_path = os.path.join(path, cam_name)
    cv2.imwrite(cam_path, img) 

def write_train_data(left_img, vel_x, vel_y, vel_z, speed, alt, target_distance, path_distance, heading_delta, roll, pitch, throttle, yaw, coords, framecount,  debug=True):

    cam_name = str(framecount) + '_cam-image.jpg'

    write_image(left_img,cam_left_dir,cam_name)
    if debug:

        print("[GPS] target_distance: " + str(target_distance) + "path_distance: " + str(path_distance) + "heading_delta: " + str(heading_delta))
        print("[RC] roll: " + str(roll) + " pitch: " + str(pitch) + " throttle: ", str(throttle)+ " yaw: ", str(yaw))
        print("[IMU] x: " + str(vel_x) + " y: " + str(vel_y) + " z: " + str(vel_z))
        print("[PIX] speed: " +  str(speed) + " altitude: " + str(alt)) 
    
    print(coords)
    json_data = {"user/roll": roll, "user/pitch": pitch, "user/throttle": throttle, "user/yaw": yaw, "imu/vel_x": vel_x, "imu/vel_y": vel_y, "imu/vel_z": vel_z, "gps/latitude":coords[0], "gps/longtitude":coords[1], "gps/speed": speed, "gps/target_distance": target_distance, "gps/path_distance": path_distance, "gps/path_distance": heading_delta, "gps/altitude": alt, "cam/image_name": cam_name, "framecount/count": framecount , "user/mode": "user"}
    
    jsonName = "record_" + str(framecount) + '.json'
    jsonPath = os.path.join(control_dir, jsonName)
    with open(jsonPath, "w") as write_file:
        json.dump(json_data, write_file)


def init():
    global depth_camera
    global left_camera, right_camera

    print("State = INIT -> " + STATE)

    setup_writer()  # file path or something as param

    left_camera = camera.create_camera(0)

    drone.connect_drone('/dev/ttyACM0')
    drone.arm()
    #drone.connect_drone('127.0.0.1:14551')

    return "flight"


def flight():
    global frame_count
    print("State = FLIGHT -> " + STATE)

    # 3000 good PWM value?
    while drone.read_channel(record_button_channel) < 1500:
        print("WAITING")
        time.sleep(0.1)

    return "flight_record"


def flight_record():
    global frame_count

    print("State = FLIGHT_RECORD -> " + STATE) 

    start_cordinate = drone.get_location()
    start = (start_cordinate.lat,start_cordinate.lon)

    print("start location: " + str(start))
    print("target location: " + str(target))

    # 3000 good PWM value?
    while drone.read_channel(record_button_channel) > 1500:
        start_time = time.time()
        frame_count += 1

        current_cordinate = drone.get_location()
        current_location = (current_cordinate.lat,current_cordinate.lon)
        current_heading = drone.get_heading()

        #print("[CURRENT GPS] location " + str(current_location) + " heading " + str(current_heading)) #remove later

        left_img = camera.get_video(0)

        #cv2.imshow("left", left_img)
        #cv2.imshow("right", right_img)
        #cv2.imshow("depth", depth_img)

        #cv2.waitKey(1)
        
        roll = drone.read_channel(1)  # not needed
        pitch = drone.read_channel(2)  # forward/backward
        throttle = drone.read_channel(3)  # up/down
        yaw = drone.read_channel(4)  # yaw

        target_distance, path_distance = gps.calculate_path_distance(target, start, current_location)
        heading_delta = gps.calculate_heading_difference(current_heading,target,current_location)

        vel_x, vel_y, vel_z = drone.get_velocity()
        
        alt = drone.get_altitude()

        speed = drone.get_ground_speed()

        write_train_data(left_img, vel_x, vel_y, vel_z, speed, alt, target_distance, path_distance, heading_delta, roll, pitch, throttle, yaw, current_location, frame_count)
        
        end_time = time.time()
        print("RECORDING > FPS: " + str(1/(end_time-start_time)))

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
