
import cv2

import argparse
import sys

import math

#def generateMovementCommand(delta):

def getCenter(contour):
	M = cv2.moments(contour)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return (cx,cy)

def getDelta(point1,point2):
	return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def get_single_axis_delta(value1,value2):
    return value2 - value1


def process(output_img):

    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)	
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    centerpoint = (round(output_img.shape[1] / 2), round(output_img.shape[0] / 2))

    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area > 100000.0:
            filtered_contours.append(contour)

    count = 0
    target = ((centerpoint[0],centerpoint[1]),0)
    if len(filtered_contours) > 1:
    #decision needs to be made to which target drone needs to go
        for contour in filtered_contours:
            targetCenter = getCenter(contour)
            if count == 0:
                targetCenter = getCenter(contour)
                delta = getDelta(targetCenter, centerpoint) 
                target = (targetCenter,delta)
            else:
                currentDelta = getDelta(targetCenter, centerpoint) 
                if abs(target[1]) > abs(currentDelta):
                    targetCenter = getCenter(contour)
                    target = (targetCenter,currentDelta)
            count+=1
            

    elif len(filtered_contours) == 1:
        targetCenter = getCenter(filtered_contours[0])
        delta = getDelta(targetCenter, centerpoint) 
        target = (targetCenter,delta)

    else:
        cv2.putText(output_img, "NO TARGET", (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 

    #show target
    cv2.circle(output_img, target[0], 20, (0, 0, 255), thickness=-1, lineType=8, shift=0)
    cv2.putText(output_img, str(target[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 3, cv2.LINE_AA) 
    #show path
    cv2.line(output_img, centerpoint, target[0], (255, 0, 0), thickness=10, lineType=8, shift=0)

    #show center
    cv2.circle(output_img, centerpoint, 20, (0, 255, 0), thickness=-1, lineType=8, shift=0)

    #show contours
    cv2.drawContours(output_img, filtered_contours, -1, (255, 255, 255), 3)

    return output_img


