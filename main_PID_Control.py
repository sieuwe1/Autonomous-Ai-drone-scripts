import sys, time
sys.path.insert(1, 'modules')

import cv2
from simple_pid import PID

import lidar
import detector_mobilenet as detector
import drone
import vision

from states import *

print("connecting lidar")
lidar.connect_lidar("/dev/ttyTHS1")

print("connecting to drone")
drone.connect_drone('/dev/ttyACM0')
#drone.connect_drone('127.0.0.1:14551')

print(drone.get_EKF_status())
print(drone.get_battery_info())
print(drone.get_version())

#config
follow_distance =1.5 #meter
max_height =  3  #m
max_speed = 3 #m/s
max_rotation = 8 #degree
#end config


x_scalar = max_rotation / 460 
z_scalar = max_speed / 10
state = "takeoff" # takeoff land track search
image_width, image_height = detector.get_image_size()
drone_image_center = (image_width / 2, image_height / 2)

debug_image_writer = cv2.VideoWriter("debug/run3.avi",cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25.0,(image_width,image_height))


def search():
    print("State = SEARCH")
    start = time.time()
    while time.time() - start < 40:
        detections, fps, image = detector.get_detections()
        print("searching: " + str(len(detections)))
        if len(detections) > 0:
            return "track"
        
        if time.time() - start > 10:
            drone.send_movement_command_YAW(1)

        if vis:
            cv2.putText(image, "searching target. Time left: " + str(40 - (time.time() - start)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
            visualize(image)

    return "land"

def visualize(img):
    debug_image_writer.write(img)
    return

while True:
    # main program loop

    if state == "track":
        state = track()

    elif state == "search":
        state = search()
    
    elif state == "takeoff":
        state = takeoff()

    elif state == "land":
        state = land()