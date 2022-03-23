from re import A
import cv2
import math
from simple_pid import PID
import camera
import time 
import numpy as np
import drone




arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters_create()


MAXSPEEDXY = 2 * 100
MAXSPEEDZ = 1 * 100
MAXSPEEDYAW = 10 
YAWMAX = 90
FOUNDSMALLEST = False

pidX = PID(0.25, 0, 0, setpoint=0, output_limits=(MAXSPEEDXY*-1,MAXSPEEDXY))
pidY = PID(0.25, 0, 0, setpoint=0, output_limits=(MAXSPEEDXY*-1,MAXSPEEDXY))
pidZ = PID(0.25, 0, 0, setpoint=250, output_limits=(0,MAXSPEEDZ))
pidYaw = PID(0.25, 0, 0, setpoint=0, output_limits=(MAXSPEEDYAW*-1,MAXSPEEDYAW))

cap = cv2.VideoCapture(0x0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
center = (int(width / 2), int(height / 2))

Time = 0

LenX = None
LenY = None
LenZ = None


#Difine Visualisation
VIS = True




def calc_angle(B, A, C):
    return np.degrees(np.arccos((((B ** 2) + (C ** 2) - (A ** 2))/ (2*B*C))))


#Calculate the Length
def calc_length(loc1, loc2):    
    return math.sqrt(abs(loc1[0] - loc2[0]) ** 2 + abs(loc1[1] - loc2[1]) ** 2)

#Check if the value is positive or negative
def NegOrPos(loc1, loc2, number):
    if loc1[0] > loc2[0] or loc1[1] > loc2[1]:
        return number
    return -abs(number)

def NegOrPosDegrees(loc1, loc2, radius):
    if loc1[0] > loc2[0]:
        return -abs(radius)
    return radius

def CalculatePIDs(corner, img):
        #Just print the box
        (TL, TR, BR, BL) = corner.reshape((4,2))
        TL = (int(TL[0]), int(TL[1]))
        TR = (int(TR[0]), int(TR[1]))
        BR = (int(BR[0]), int(BR[1]))
        BL = (int(BL[0]), int(BL[1]))
        cv2.line(img, TL, TR, (0,255,0), 2)
        cv2.line(img, TR, BR, (0,255,255), 2)
        cv2.line(img, BR, BL, (0,255,0), 2)
        cv2.line(img, BL, TL, (0,255,0), 2)
        #Get average of the position, maybe remove the INT() for finer control
        CX = int((TL[0] + BR[0]) / 2)
        CY = int((TL[1] + BR[1]) /  2 )


        #Print the ID of the marker
        #cv2.putText(img, str(markerID), (TL[0], TL[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)


        #Calculate all the lengths 
        MAINLength = calc_length((CX, CY), center)
        XLength = calc_length((center[0],CY), (CX,CY))
        YLength = calc_length((center[0], CY), (center[0],center[1]))



        #Z?

        ZLength = calc_length(TR, TL)

        #See if the lengths need to be converted to negative
        #Swap these
        XLength = NegOrPos((center[0],CY), (CX,CY), XLength)
        YLength = NegOrPos((center[0], CY), (center[0],center[1]), YLength)

        LenX = pidX(XLength)
        LenY = pidY(YLength)
        LenZ = pidZ(ZLength)
        if VIS is True:
                    #Print the lines and the text 
            #Main Line
            cv2.line(img, center, (CX,CY), (255,0,0),2)
            cv2.putText(img, ("Main: {}".format(str(MAINLength))), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

            #X Line
            cv2.line(img, (center[0],CY), (CX,CY), (0,255,0),2)
            cv2.putText(img, ("X: {}".format(str(YLength))), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

            #Y Line
            cv2.line(img, (center[0], CY), (center[0],center[1]), (0,0,255),2)
            cv2.putText(img, ("Y: {}".format(str(XLength))), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)

            #Calculate Z

            cv2.putText(img, ("Z: {}".format(str(ZLength))), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        
        return LenX, LenY, LenZ, img



def RotateIfNeeded(corner, img):

    (TL, TR, BR, BL) = corner.reshape((4,2))
    TL = (int(TL[0]), int(TL[1]))
    TR = (int(TR[0]), int(TR[1]))
    BR = (int(BR[0]), int(BR[1]))
    BL = (int(BL[0]), int(BL[1]))

    CX = int((TL[0] + BR[0]) / 2)
    CY = int((TL[1] + BR[1]) /  2 )

    LX = int((TR[0] + BR[0]) / 2)
    LY = int((TR[1] + BR[1]) / 2)

    A = calc_length((CX, CY), (CX, 0))
    B = calc_length((CX, 0), (LX, LY))
    C = calc_length((CX, CY), (LX, LY))
    Yaw =  NegOrPosDegrees((CX, CY), (LX, LY), calc_angle(A,B,C))


    print(A)
    print(B)
    print(C)
    print(Yaw)
    print(Yaw < YAWMAX and Yaw > YAWMAX * -1)
    print()

    if VIS is True:
        cv2.line(img, (CX, CY), (CX, 0), (255,255,255), 2)
        cv2.line(img, (CX, 0), (LX, LY), (255,255,255), 2)
        cv2.line(img, (CX, CY), (LX, LY), (255,255,255), 2)
        cv2.putText(img, ("ANGLE: {}".format(str(Yaw))), (0, 300  ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

    return img, Yaw


def Land():
    camera.create_camera(1)
    FOUNDSMALLEST = False
    Time = 0
    while True:
        img = camera.get_video(1)
        #Remove Extra platform if needed

    #Check for markers
        (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
             
        if len(corners) > 0:
            ids = ids.flatten()

            #Run this to motor
            img, yaw = RotateIfNeeded(corners[0], img)
            RYaw = pidYaw(yaw)
            drone.send_movement_command_YAW(yaw)
            if yaw < YAWMAX and (yaw > (YAWMAX * -1)):

                if 1 in ids:
                    FOUNDSMALLEST = True
                if FOUNDSMALLEST:
                    if 1 in ids:
                        LenX, LenY, LenZ, img = CalculatePIDs(corners[np.where(ids == 1)[0][0]], img)
                        Time = time.time()
                        
                else:
                    LenX, LenY, LenZ, img = CalculatePIDs(corners[np.where(ids == 0)[0][0]], img)

                drone.send_movement_command_XYA(LenY /  100, LenX / 100, 0, 1)
                if VIS is True:
                    cv2.putText(img, ("PIDX: {}".format(str(LenX / 100))), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
                    cv2.putText(img, ("PIDY: {}".format(str(LenY / 100))), (0, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
                    cv2.putText(img, ("PIDZ: {}".format(str(LenZ / 100))), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,),2)
                    cv2.putText(img, ("PIDYaw: {}".format(str(RYaw))), (0, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,),2)
                    cv2.putText(img, str(time.time() - Time), (0,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                if  time.time() - Time > 1:
                    FOUNDSMALLEST = False

        if VIS is True:
            #Print middle circlex
            cv2.circle(img, (int(width /2) , int(height / 2)),2, (0,255,0), -1)
            
            #Show the image
            cv2.imshow("Webcam", img) # This will open an independent window
            cv2.waitKey(1)




Land()




