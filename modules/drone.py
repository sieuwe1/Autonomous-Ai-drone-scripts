from dronekit import *

vehicle = None 

# Connect to the Vehicle (in this case a UDP endpoint)
def connect_drone(connection_string, waitready=True, baudrate=57600):
    global vehicle
    if vehicle == None:
        vehicle = connect(connection_string, wait_ready=waitready, baud=baudrate)
    print("drone connected")

def disconnect_drone():
    vehicle.close()

def get_version():
    global vehicle
    return vehicle.version

def get_mission():
    global vehicle
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    return 

def get_location():
    global vehicle
    return vehicle.location.global_frame

def get_altitude():
    global vehicle
    return vehicle.location.global_relative_frame.alt

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

def get_heading():
    global vehicle
    return vehicle.heading

def get_EKF_status():
    return vehicle.ekf_ok

def get_ground_speed():
    return vehicle.groundspeed

def read_channel(channel):
    return vehicle.channels[str(channel)]

def set_gimbal_angle(angle):
    global vehicle
    print("gimbal angle set to: " % angle)
    return vehicle.gimbal.rotate(0, angle, 0)

def set_groundspeed(speed):
    global vehicle
    print("groundspeed set to: " % speed)
    vehicle.groundspeed = speed

def set_flight_mode(f_mode):
    global vehicle
    vehicle.mode = VehicleMode(f_mode)

def set_param(param, value):
    global vehicle
    vehicle.parameters[param] = value

def get_param(param):
    return vehicle.parameters[param] 

def set_channel(channel, value):
    global vehicle
    vehicle.channels.overrides[channel] = value

def clear_channel(channel):
    global vehicle
    vehicle.channels.overrides[channel] = None
    
def get_channel_override(channel):
    return vehicle.channels.overrides[channel]
          
def disarm():
    global vehicle
    vehicle.armed = False

def arm():
    global vehicle
    vehicle.groundspeed = 3

    print ("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print (" Waiting for vehicle to initialise...")
        time.sleep(1)

    print ("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode    = VehicleMode("STABILIZE")
    vehicle.armed   = True

    while not vehicle.armed:
        print (" Waiting for arming...")
        time.sleep(1)

    print ("ARMED! Let's take OFF")

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

def land():
    global vehicle
    print("Setting LAND mode...")
    vehicle.mode = VehicleMode("LAND")

def return_to_launch_location():
    #carefull with using this! It wont detect obstacles!
    vehicle.mode = VehicleMode("RTL")

def send_movement_command_YAW(heading):
    global vehicle
    speed = 0 
    direction = 1 #direction -1 ccw, 1 cw
    
    #heading 0 to 360 degree. if negative then ccw 
    
    print("Sending YAW movement command with heading: %f" % heading)

    if heading < 0:
        heading = heading*-1
        direction = -1

    #point drone into correct heading 
    msg = vehicle.message_factory.command_long_encode(
        0, 0,       
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 
        0,          
        heading,    
        speed,      #speed deg/s
        direction,  
        1,          #relative offset 1
        0, 0, 0)    

    # send command to vehicle
    vehicle.send_mavlink(msg)
    #Vehicle.commands.flush()

def send_movement_command_XYA(velocity_x, velocity_y, altitude):
    global vehicle

    #velocity_x positive = forward. negative = backwards
    #velocity_y positive = right. negative = left
    #velocity_z positive = down. negative = up (Yes really!)

    print("Sending XYZ movement command with v_x(forward/backward): %f v_y(right/left): %f " % (velocity_x,velocity_y))

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,      
        0, 0,    
        mavutil.mavlink.MAV_FRAME_BODY_NED,  #relative to drone heading pos relative to EKF origin
        0b0000111111100011, #ignore velocity z and other pos arguments
        0, 0, altitude,
        velocity_x, velocity_y, 0, 
        0, 0, 0, 
        0, 0)    

    vehicle.send_mavlink(msg)
    #Vehicle.commands.flush()

