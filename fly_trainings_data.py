import sys, time
import argparse
sys.path.insert(1, 'modules')

import cv2
import collections
import json

import camera
import drone

state = "init" # init, flight, flight_record, shutdown

left_camera = None
right_camera = None

record_button_channel = 6

positions = collections.deque(maxlen=60) #if system runs 30fps then 60fps is 2 seconds 


def calculate_path():
    positions.append(drone.get_location())

    if len(positions) == 60:
    #calculate straight line trought points. Then add end point to the end of the line + 50 meters.         

def calculate_path_distance():
    #calcualte distance between the line and drone and between end point and drone using path calcualted from calulate_path()

def setup_writer()
    return 

def writeTrainData(angle, throttle, image, framecount):

    camName = str(framecount) + '_cam-image_array_.jpg'
    camPath = os.path.join(data_dir, camName)
    cv2.imwrite(camPath, image) 
    json_data = {"user/angle": angle, "cam/image_array": camName, "user/throttle": throttle, "user/mode": "user"}
    
    jsonName = "record_" + str(framecount) + '.json'
    jsonPath = os.path.join(data_dir, jsonName)
    with open(jsonPath, "w") as write_file:
        json.dump(json_data, write_file)

def init():
    global left_camera, right_camera

    print("State = INIT -> " + STATE)

    drone.connect_drone('/dev/ttyACM0')

    setup_writer() #file path or something as param

    left_camera = camera.create_camera(0)
    right_camera = camera.create_camera(1)

    #init zed camera

    return "flight"

def flight():
    print("State = FLIGHT -> " + STATE)

    while drone.read_channel(record_button_channel) < 3000 #3000 good PWM value?
        time.sleep(0.1)

    return "flight_record"

def flight_record():
    print("State = FLIGHT_RECORD -> " + STATE)

    while drone.read_channel(record_button_channel) > 3000 #3000 good PWM value?

        left_img = camera.get_video(0)
        right_img = camera.get_video(1)
        
        #depth_img = 

        roll = drone.read_channel(1) # not needed
        pitch = drone.read_channel(2) #forward/backward
        throttle = drone.read_channel(3) #up/down
        yaw = drone.read_channel(4) #yaw
    
        writeTrainData(left_img, right_img,)

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