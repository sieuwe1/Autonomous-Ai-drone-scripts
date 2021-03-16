import jetson.inference
import jetson.utils
import cv2
import numpy as np

net = jetson.inference.detectNet("ssd-mobilenet-v2")
camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file ("display://0")

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	jetson.utils.cudaDeviceSynchronize()
	aimg = jetson.utils.cudaToNumpy (img, 1280, 720, 4)
	aimg1 = cv2.cvtColor (aimg.astype (np.uint8), cv2.COLOR_RGBA2BGR)
	
	cv2.imshow("outputa",aimg1)
	#https://dronekit-python.readthedocs.io/en/latest/guide/copter/guided_mode.html
	#display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
