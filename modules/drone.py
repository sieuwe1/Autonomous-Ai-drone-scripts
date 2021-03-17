from dronekit import *

vehicle = None 

# Connect to the Vehicle (in this case a UDP endpoint)
def connect_drone(connection_string):
    global vehicle
    if vehicle == None:
        vehicle = connect('/dev/ttyACM0', wait_ready=True, baud=57600)

def get_version():
    global vehicle
    return vehicle.version

def get_location():
    global vehicle
    return vehicle.location.global_frame

def get_altitude():
    global vehicle
    return vehicle.attitude

def get_velocity():
    global vehicle
    return vehicle.velocity

def get_battery_info():
    global vehicle
    return vehicle.battery

def get_mode():
    global vehicle
    return vehicle.mode.name

def get_home_location():
    global vehicle
    return vehicle.home_location

def set_gimbal_angle(angle):
    global vehicle
    print("gimbal angle set to: " % angle)
    return vehicle.gimbal.rotate(0, angle, 0)

def set_groundspeed(speed):
    global vehicle
    print("groundspeed set to: " % speed)
    vehicle.groundspeed = speed

def arm_and_takeoff(aTargetAltitude):
    global vehicle

    #set default groundspeed low for safety 
    print ("setting groundspeed to 3")
    vehicle.groundspeed = 3

    print ("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print (" Waiting for vehicle to initialise...")
        time.sleep(1)

    print ("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode    = VehicleMode("GUIDED")
    vehicle.armed   = True

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print (" Waiting for arming...")
        time.sleep(1)

    print ("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print (" Altitude: ", vehicle.location.global_relative_frame.alt)
        #Break and return from function just below target altitude.
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95:
            print ("Reached target altitude")
            break
        time.sleep(1)

def send_movement_command_YAW_SPEED_HEIGHT(velocity):


def send_movement_command_XYZ(velocity_x, velocity_y, velocity_z):
    global vehicle

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # relative to drone heading
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    vehicle.send_mavlink(msg)




SOUTH=-2
UP=-0.5   #NOTE: up is negative!

#Fly south and up.
send_ned_velocity(SOUTH,0,UP)