from simple_pid import PID

from modules import drone


def land():
    print("State = LAND")
    drone.land()
    detector.close_camera()
    sys.exit(0)