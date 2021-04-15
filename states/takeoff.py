from simple_pid import PID
from modules import drone

TARGET_ALT = 10

def takeoff():
    print("State = TAKEOFF")
    drone.arm_and_takeoff(TARGET_ALT)
    return "search"