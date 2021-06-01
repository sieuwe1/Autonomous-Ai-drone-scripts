import pyzed.sl as sl
import time
import cv2

# Zed 2 required variables
depth_camera = None
runtime = None
depth_mat = None
rgb_mat = None
init_parameters = None
initialized = False

MAX_RANGE = 6
MIN_RANGE = 1

def set_params(performance_mode=True):
    global init_parameters
    init_parameters = sl.InitParameters()
    init_parameters.depth_mode = sl.DEPTH_MODE.PERFORMANCE if performance_mode else sl.DEPTH_MODE.ULTRA
    init_parameters.depth_minimum_distance = MIN_RANGE #1 Meter minimum detection distance
    init_parameters.coordinate_units = sl.UNIT.METER


def set_runtime_params():
    global runtime
    runtime = sl.RuntimeParameters()

def close():
    global depth_camera
    depth_camera.close()


def init_zed(performance_mode=True):
    global depth_mat, rgb_mat, depth_camera, initialized

    set_params(performance_mode)
    depth_camera = sl.Camera(init_parameters)

    if not depth_camera.is_opened():
        status = depth_camera.open()
        initialized = True
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))

    set_runtime_params()
    #depth_camera.set_depth_max_range_value(MAX_RANGE)
    depth_camera.open()

    depth_mat = sl.Mat()
    rgb_mat = sl.Mat()
    return depth_camera


def get_depth_image():
    global depth_mat
    image = None
    err = depth_camera.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        depth_camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        image = depth_mat.get_data()
    return image

def get_rgbd_image():
    global depth_mat, rgb_mat
    depth_image = None
    rgb_image = None
    err = depth_camera.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        depth_camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth_camera.retrieve_image(rgb_mat, sl.VIEW.LEFT)
        rgb_image = rgb_mat.get_data()
        depth_image = depth_mat.get_data()
        depth_image = cv2.normalize(depth_image,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        splitted = cv2.split(rgb_image)
        bgrd = cv2.merge((splitted[0],splitted[1],splitted[2],depth_image))
        return bgrd
    return None

        
if __name__ == "__main__":
    init_zed()
    while True:
        bgrd = get_rgbd_image()
        cv2.imshow("BGRD" , bgrd)
        cv2.waitKey(1)
