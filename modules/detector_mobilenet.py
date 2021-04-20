import jetson.inference
import jetson.utils
import cv2
import numpy as np

net = None
camera = None

def initialize_detector():
	global net, camera
	net = jetson.inference.detectNet("ssd-mobilenet-v2")
	camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
	print("fakka")

def get_image_size():
	return camera.GetWidth(), camera.GetHeight()

def close_camera():
	camera.Close()

def get_detections():
	person_detections = []
	img = camera.Capture()
	detections = net.Detect(img)
	for detection in detections:
		if detection.ClassID == 1: #remove unwanted classes
			person_detections.append(detection)
	fps = net.GetNetworkFPS()

	return person_detections, fps, jetson.utils.cudaToNumpy(img)

