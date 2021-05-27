import pyzed.sl as sl
import time
import cv2

# Zed 2 required variables
depth_camera = None
runtime = None
mat = None
init_parameters = None
initialized = False

def set_params(performance_mode=True):
    global init_parameters
    init_parameters = sl.InitParameters()
    init_parameters.depth_mode = sl.DEPTH_MODE.PERFORMANCE if performance_mode else sl.DEPTH_MODE.ULTRA
    init_parameters.depth_minimum_distance = 1

def set_runtime_params():
    global runtime
    runtime = sl.RuntimeParameters()

def close():
    global depth_camera
    depth_camera.close()


def init_zed(performance_mode=True):
    global mat, depth_camera, initialized

    set_params(performance_mode)
    depth_camera = sl.Camera(init_parameters)

    if not depth_camera.is_opened():
        status = depth_camera.open()
        initialized = True
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))

    set_runtime_params()
    depth_camera.open()
    return depth_camera


def get_depth_image():
    global mat
    mat = sl.Mat()
    image = None
    err = depth_camera.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        depth_camera.retrieve_measure(mat, sl.MEASURE.DEPTH)
        image = mat.get_data()
    return image
        
if __name__ == "__main__":
    init_zed()
    while True:
        cv2.imshow("ZED" , get_depth_image())
        cv2.waitKey(1)