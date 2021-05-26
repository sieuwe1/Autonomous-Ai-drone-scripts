import drone
import camera
import json
import collections
import cv2
import sys
import time
import os
import argparse
import pyzed.sl as sl

sys.path.insert(1, 'modules')


state = "init"  # init, flight, flight_record, shutdown
record_button_channel = 6

# Zed 2 required variables
cam = None
mat = None
runtime = None
init_parameters = None
mat = None


frame_count = None
left_camera = None
right_camera = None
data_dir = None
cam_left_dir = None
cam_right_dir = None
cam_depth_dir = None
control_dir = None

# if system runs 30fps then 60fps is 2 seconds
positions = collections.deque(maxlen=60)


def calculate_path():
    positions.append(drone.get_location())

    if len(positions) == 60:
        # calculate straight line trought points. Then add end point to the end of the line + 50 meters.
        return NotImplementedError
    return NotImplementedError


def calculate_path_distance():
    # calcualte distance between the line and drone and between end point and drone using path calcualted from calulate_path()
    return NotImplementedError


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
        print("Directory ", data_dir,  " already exists")

    cam_left_dir = data_dir + "left_camera"
    cam_right_dir = data_dir + "right_camera"
    cam_depth_dir = data_dir + "depth_camera"
    control_dir = data_dir + "control"

    os.mkdir(cam_left_dir)
    os.mkdir(cam_right_dir)
    os.mkdir(cam_depth_dir)
    os.mkdir(control_dir)


def write_image(img, path, framecount, resize=False):
    if resize:
        # Resizes the image using the INTER_CUBIC interpolation, taking the average of the 4 pixels surrounding it
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    cam_name = str(framecount) + '_cam-image_cam_array.jpg'
    cam_path = os.path.join(path, cam_name)
    cv2.imwrite(cam_path, img)


def write_train_data(left_img, right_img, depth_img, roll, pitch, throttle, yaw, framecount):

    write_image(left_img, cam_left_dir, framecount)
    write_image(right_img, cam_right_dir, framecount)
    write_image(depth_img, cam_depth_dir, framecount, resize=True)

    json_data = {"user/roll": roll, "user/pitch": pitch, "user/throttle": throttle, "user/yaw": yaw,
                 "cam/image_array_left": left_img, "cam/image_array_right": right_img, "cam/image_array_depth": depth_img, "user/mode": "user"}

    jsonName = "record_" + str(framecount) + '.json'
    jsonPath = os.path.join(data_dir, jsonName)
    with open(jsonPath, "w") as write_file:
        json.dump(json_data, write_file)


def init():
    global cam
    global left_camera, right_camera

    print("State = INIT -> " + STATE)

    drone.connect_drone('/dev/ttyACM0')

    setup_writer()  # file path or something as param

    left_camera = camera.create_camera(0)
    right_camera = camera.create_camera(1)

    cam = init_zed()

    return "flight"


def flight():
    print("State = FLIGHT -> " + STATE)

    # 3000 good PWM value?
    while drone.read_channel(record_button_channel) < 3000:
        print("record channel output: " +
              str(drone.read_channel(record_button_channel)))
        time.sleep(0.1)

    return "flight_record"


def flight_record():
    global frame_count

    print("State = FLIGHT_RECORD -> " + STATE)

    # 3000 good PWM value?
    while drone.read_channel(record_button_channel) > 3000:
        frame_count += 1

        left_img = camera.get_video(0)
        right_img = camera.get_video(1)

        depth_img = cam.retrieve_measure(
            mat, sl.MEASURE.DEPTH).get_data()   # Retrieves the depth image

        roll = drone.read_channel(1)  # not needed
        pitch = drone.read_channel(2)  # forward/backward
        throttle = drone.read_channel(3)  # up/down
        yaw = drone.read_channel(4)  # yaw

        write_train_data(left_img, right_img, depth_img, roll,
                         pitch, throttle, yaw, frame_count)

    return "flight"


# TODO discuss extra parameters?
def init_zed(performance_mode=True):
    global mat
    init_parameters = sl.InitParameters()
    init_parameters.depth_mode = sl.DEPTH_MODE.PERFORMANCE if performance_mode else sl.DEPTH_MODE.ULTRA
    init_parameters.depth_minimum_distance = 1
    mat = sl.Mat()
    return sl.Camera(init_parameters)


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
