import sys, time
sys.path.insert(1, 'modules')

import cv2
from simple_pid import PID
import collections

import lidar
import detector_mobilenet as detector
import drone
import vision

print("connecting lidar")
lidar.connect_lidar("/dev/ttyTHS1")

print("setting up detector")
detector.initialize_detector()

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
vis = True
movement_x_en = True
movement_yaw_en = True
#end config

pidYaw = PID(0.02, 0, 0, setpoint=0)  #I = 0.001
pidYaw.output_limits = (-15, 15)

maxlength = 5
moving_average_z = collections.deque(maxlen=maxlength)

maxlength = 5
moving_average_x = collections.deque(maxlen=maxlength)

z_scalar = max_speed / 50
state = "takeoff" # takeoff land track search
image_width, image_height = detector.get_image_size()
drone_image_center = (image_width / 2, image_height / 2)

debug_image_writer = cv2.VideoWriter("debug/new_drone_2.avi",cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25.0,(image_width,image_height))
debug_fileYaw = open("new_drone_2_yaw.txt", "a")
debug_fileYaw.write("P: I: D: Error: command:\n")

debug_fileDepth= open("new_drone_2_Depth.txt", "a")
debug_fileDepth.write("P: I: D: Error: command:\n")

# Logging_config
def debug_writerYaw(inputValueYaw,movementJawAngle):
    global debug_fileYaw
    debug_fileYaw.write("0" + "," + "0" + "," + "0" + "," + str(inputValueYaw) + "," + str(movementJawAngle) + "\n")

def debug_writerDepth(inputValueYaw,movementJawAngle):
    global debug_fileDepth
    debug_fileDepth.write("0" + "," + "0" + "," + "0" + "," + str(inputValueYaw) + "," + str(movementJawAngle) + "\n")


def track():
    print("State = TRACKING")

    while True:
        detections, fps, image = detector.get_detections()

        if len(detections) > 0:
            person_to_track = detections[0] # only track 1 person
            
            person_to_track_center = person_to_track.Center # get center of person to track

            x_delta = vision.get_single_axis_delta(drone_image_center[0],person_to_track_center[0]) # get x delta 
            y_delta = vision.get_single_axis_delta(drone_image_center[1],person_to_track_center[1]) # get y delta

            lidar_on_target = vision.point_in_rectangle(drone_image_center,person_to_track.Left, person_to_track.Right, person_to_track.Top, person_to_track.Bottom) #check if lidar is pointed on target

            lidar_distance = lidar.read_lidar_distance()[0] # get lidar distance in meter

            moving_average_z.append(lidar_distance)
            moving_average_x.append(x_delta)
            
            #control section 
            #x_delta max=620 min= -620
            #y_delta max=360 min= -360
            #z_delta max=8   min= 2

            #depth x command > manual P and moving average
            velocity_x_command = 0
            if movement_x_en and lidar_distance > 0 and lidar_on_target and len(moving_average_z) > 0: #only if a valid lidar value is given change the forward velocity. Otherwise keep previos velocity (done by arducopter itself)
                
                sum_moving_avg = 0
                for i in moving_average_z:
                    sum_moving_avg = sum_moving_avg + i

                lider_distance_moving_avg = sum_moving_avg / maxlength
                z_delta = lider_distance_moving_avg - follow_distance
                velocity_x_command = z_delta * z_scalar
                debug_writerDepth(z_delta, velocity_x_command)
                drone.send_movement_command_XYZ(velocity_x_command,0,0)

            #yaw command > PID and moving average
            yaw_command = 0
            if movement_yaw_en and len(moving_average_x) > 0:
                
                sum_moving_avg = 0
                for i in moving_average_x:
                    sum_moving_avg = sum_moving_avg + i

                delta_x_moving_avg = sum_moving_avg / maxlength
                yaw_command = (pidYaw(delta_x_moving_avg) * -1)
                debug_writerYaw(x_delta, yaw_command)
                drone.send_movement_command_YAW(yaw_command)

            if vis:
                #draw lidar distance
                lidar_vis_x = image_width - 50
                lidar_vis_y = image_height - 50
                lidar_vis_y2 = int(image_height - lidar_distance * 200)
                cv2.line(image, (lidar_vis_x,lidar_vis_y), (lidar_vis_x, lidar_vis_y2), (0, 255, 0), thickness=10, lineType=8, shift=0)
                cv2.putText(image, "distance: " + str(round(lidar_distance,2)), (image_width - 300, 200), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 

                #draw path
                cv2.line(image, (int(drone_image_center[0]), int(drone_image_center[1])), (int(person_to_track_center[0]), int(person_to_track_center[1])), (255, 0, 0), thickness=10, lineType=8, shift=0)

                #draw bbox around target
                cv2.rectangle(image,(int(person_to_track.Left),int(person_to_track.Bottom)), (int(person_to_track.Right),int(person_to_track.Top)), (0,0,255), thickness=10)

	            #show drone center
                cv2.circle(image, (int(drone_image_center[0]), int(drone_image_center[1])), 20, (0, 255, 0), thickness=-1, lineType=8, shift=0)

                #show trackable center
                cv2.circle(image, (int(person_to_track_center[0]), int(person_to_track_center[1])), 20, (0, 0, 255), thickness=-1, lineType=8, shift=0)

                #show stats
                cv2.putText(image, "fps: " + str(round(fps,2)) + " yaw: " + str(round(yaw_command,2)) + " forward: " + str(round(velocity_x_command,2)) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
                cv2.putText(image, "lidar_on_target: " + str(lidar_on_target), (50, 100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
                cv2.putText(image, "x_delta: " + str(round(x_delta,2)) + " y_delta: " + str(round(y_delta,2)), (50, 150), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 

                visualize(image)

        else:
            return "search"

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

def takeoff():
    print("State = TAKEOFF")
    drone.arm_and_takeoff(max_height)
    return "search"

def land():
    print("State = LAND")
    drone.land
    detector.close_camera()
    sys.exit(0)

def visualize(img):
    #cv2.imshow("out", img)
    
    #cv2.waitKey(1)
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
