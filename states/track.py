""" Additional module regarding the smooth controlling of the drone """
from simple_pid import PID
#TODO does the docstring make sense?

""" Import all required vision modules relating image detection """
from modules import vision
from modules import detector_mobilenet as detector
from modules import lidar
from modules import drone

""" Library imports used for displaying and calculating purposes """
import cv2

import main_PID_Control as main

""" Global variables, used throughout the tracking script """
IMAGE_WIDTH, IMAGE_HEIGHT = None, None
DRONE_IMAGE_CENTER = (0,0)
MOVEMENT_X_EN = False
MOVEMENT_YAW_EN = True
VIS = True
PID_YAW = PID(0.03, 0, 0, setpoint=0)
PID_YAW.output_limits = (-15, 15)
P, I, D = PID_YAW.components

DEBUG_FILEYAW = None
DEBUG_INITIALIZED = False


# Logging_config
def write_yaw_debug(inputValueYaw,movementJawAngle):
    if not DEBUG_INITIALIZED:
        initialize_debugger()
    DEBUG_FILEYAW.write(str(P) + "," + str(I) + "," + str(D) + "," + str(inputValueYaw) + "," + str(movementJawAngle) + "\n")

def initialize_detector():
    """ Initializes the detector and define the image variables """
    global IMAGE_WIDTH
    global IMAGE_HEIGHT
    global DRONE_IMAGE_CENTER
    detector.initialize_detector()
    IMAGE_WIDTH, IMAGE_HEIGHT = detector.get_image_size()
    DRONE_IMAGE_CENTER = (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)

def initialize_debugger():
    """ Initial code which will be ran before the debug file can be accessed """
    global DEBUG_INITIALIZED
    global DEBUG_FILEYAW
    DEBUG_FILEYAW = open("PID_IN_MAIN_yaw.txt", "a")
    DEBUG_FILEYAW.write("P: I: D: Error: command:\n")
    DEBUG_INITIALIZED = True


def track():
    """ Tracks the object and returns the coordinates to the main loop """
    print("State = TRACKING")

    """ Will loop indefenitely until the tracked object can no longer be tracked """
    while True:
        detections, fps, image = detector.get_detections()

        if len(detections) > 0:
            person_to_track = detections[0] # only track 1 person
            
            person_to_track_center = person_to_track.Center # get center of person to track

            x_delta = vision.get_single_axis_delta(DRONE_IMAGE_CENTER[0],person_to_track_center[0]) # get x delta 
            y_delta = vision.get_single_axis_delta(DRONE_IMAGE_CENTER[1],person_to_track_center[1]) # get y delta

            lidar_on_target = vision.point_in_rectangle(DRONE_IMAGE_CENTER,person_to_track.Left,
                                person_to_track.Right, person_to_track.Top, person_to_track.Bottom) #check if lidar is pointed on target
            lidar_distance = lidar.read_lidar_distance()[0] # get lidar distance in meter
            velocity_x_command = 0

            yaw_command = 0
            if MOVEMENT_YAW_EN:
                yaw_command = (PID_YAW(y_delta) * -1)
                write_yaw_debug(x_delta, yaw_command)
                drone.send_movement_command_YAW(yaw_command)

            if VIS:
                #draw lidar distance
                lidar_vis_x = IMAGE_WIDTH - 50
                lidar_vis_y = IMAGE_HEIGHT - 50
                lidar_vis_y2 = int(IMAGE_HEIGHT - lidar_distance * 200)
                cv2.line(image, (lidar_vis_x,lidar_vis_y), (lidar_vis_x, lidar_vis_y2), (0, 255, 0), thickness=10, lineType=8, shift=0)
                cv2.putText(image, "distance: " + str(round(lidar_distance,2)), (IMAGE_WIDTH- 300, 200), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 

                #draw path
                cv2.line(image, (int(DRONE_IMAGE_CENTER[0]), int(DRONE_IMAGE_CENTER[1])), (int(person_to_track_center[0]), int(person_to_track_center[1])), (255, 0, 0), thickness=10, lineType=8, shift=0)

                #draw bbox around target
                cv2.rectangle(image,(int(person_to_track.Left),int(person_to_track.Bottom)), (int(person_to_track.Right),int(person_to_track.Top)), (0,0,255), thickness=10)

	            #show drone center
                cv2.circle(image, (int(DRONE_IMAGE_CENTER[0]), int(DRONE_IMAGE_CENTER[1])), 20, (0, 255, 0), thickness=-1, lineType=8, shift=0)

               #show trackable center
                cv2.circle(image, (int(person_to_track_center[0]), int(person_to_track_center[1])), 20, (0, 0, 255), thickness=-1, lineType=8, shift=0)

                #show stats
                cv2.putText(image, "fps: " + str(round(fps,2)) + " yaw: " + str(round(yaw_command,2)) + " forward: " + str(round(velocity_x_command,2)) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
                cv2.putText(image, "lidar_on_target: " + str(lidar_on_target), (50, 100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
                cv2.putText(image, "x_delta: " + str(round(x_delta,2)) + " y_delta: " + str(round(y_delta,2)), (50, 150), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 

                main.visualize(image)

        else:
            return "search"

