# Fly fully autonomous from starting point to target! Dont worry about obstacles our AI should avoid them!
import sys
from modules.drone import get_mode
sys.path.insert(1, '/home/drone/Desktop/Autonomous-Ai-drone-scripts/modules')
import drone
import camera
import landing
import gps
import cv2
import json
import time 
import numpy as np
import collections
from tensorflow import keras
from keras.applications import imagenet_utils
from adabelief_tf import AdaBeliefOptimizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import onnxruntime as rt
import os
#-------config----------
flight_altitude = 10 #meters
debug = True
#-----------------------

STATE = "init"




def warmup():
    print("Warming UP")

def init():
    global sess, inputs 

    print("State = INIT -> " + STATE)

    time.sleep(20)

    warmup()

    print("FLYING NOW")
    drone.connect_drone('/dev/ttyACM0')

    drone.arm_and_takeoff(flight_altitude)
    #drone.connect_drone('127.0.0.1:14551')
    return "flight"


def flight():
    while drone.get_altitude() < 10:
        print("not high enough")

    landing.land()


def goal_reached():
    global frame_count
    print("State = GOAL_REACHED -> " + STATE)
    print("Goal reached! Landing in 20 seconds!")
    drone.send_movement_command_XYA(0, 0, flight_altitude)
    drone.clear_channel('4')
    time.sleep(20)
    drone.land()

    return "shutdown"

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

    elif STATE == "goal_reached":
        STATE = goal_reached()

    elif STATE == "shutdown":
        STATE = shutdown()
