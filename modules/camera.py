import jetson.utils
import cv2

cams = []

def create_camera(csi_port): 
    cams.append(jetson.utils.videoSource("csi://" + str(csi_port)))

def get_image_size(camera_id):
	return cams[camera_id].GetWidth(), cams[camera_id].GetHeight()

def get_video(camera_id):
    return jetson.utils.cudaToNumpy(cams[camera_id].Capture())

def close_cameras():
    for cam in cams:
        cam.Close()    

if __name__ == "__main__":
    create_camera(0)
    
    while True:
        img = get_video(0)
        cv2.imshow("camera", img)
        cv2.waitKey(1)
    
    close_cameras()