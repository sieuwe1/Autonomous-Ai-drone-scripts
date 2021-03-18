import sys, time
sys.path.insert(1, 'modules')

import cv2

import lidar
import detector_mobilenet as detector
import drone
import vision

vis = True

#lidar.connect_lidar("/dev/ttyTHS1")

#lidar.read_lidar_distance()
#lidar.read_lidar_temperature()

detector.initialize_detector()

#drone.connect_drone('/dev/ttyACM0')

#drone.arm_and_takeoff(height)

state = "takeoff" # takeoff land track search
image_width, image_height = detector.get_image_size()
drone_image_center = (image_width / 2, image_height / 2)

def track():
    print("State = TRACKING")

    while True:
        detections, fps, image = detector.get_detections(vis)

        if len(detections) > 0:
            person_to_track = detections[0] # only track 1 person
            
            person_to_track_center = person_to_track.Center

            x_delta = vision.get_single_axis_delta(drone_image_center[0],person_to_track_center[0])
            y_delta = vision.get_single_axis_delta(drone_image_center[1],person_to_track_center[1])

            print(x_delta)
            print(y_delta)

            if vis:
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

        if vis:
            visualize(image)

    return "land"

def takeoff():
    print("State = TAKEOFF")

    return "search"

def land():
    print("State = LAND")

    sys.exit(0)

def visualize(img):
    cv2.imshow("out", img)
    
    cv2.waitKey(1)

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
    
    #cv2.imshow("out", image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

    #print(fps)