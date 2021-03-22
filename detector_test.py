import sys, time
sys.path.insert(1, 'modules')

import cv2

import detector_mobilenet as detector

print("setting up detector")
detector.initialize_detector()

while True:
    detections, fps, image = detector.get_detections()
    cv2.putText(image, str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    cv2.imshow("out", image)
    cv2.waitKey(1)