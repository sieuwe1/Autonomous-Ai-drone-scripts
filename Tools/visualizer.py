import cv2
import json
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import config as c

count = 1
plot_data = []

fig, ax = plt.subplots()


def map(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return round(rightMin + (valueScaled * rightSpan))


def drawUI(img, data):
    cv2.putText(img, "TargetDistance: " + str(data['gps/target_distance']), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "PathDistance: " + str(data['gps/path_distance']), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "HeadingDelta: " + str(data['gps/heading_delta']), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "Speed: " + str(data['gps/speed']), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                cv2.LINE_AA)
    cv2.putText(img, "altitude: " + str(data['gps/altitude']), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                cv2.LINE_AA)

    cv2.putText(img, "x: " + str(data['imu/vel_x']), (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                cv2.LINE_AA)
    cv2.putText(img, "y: " + str(data['imu/vel_y']), (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                cv2.LINE_AA)
    cv2.putText(img, "z: " + str(data['imu/vel_z']), (1000, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                cv2.LINE_AA)

    cv2.rectangle(img, (50, 500), (250, 700), (255, 0, 0), 2)
    cv2.rectangle(img, (1000, 500), (1200, 700), (255, 0, 0), 2)

    throttle = map(data['user/throttle'], 1000, 2000, 500, 700)
    yaw = map(data['user/yaw'], 1000, 2000, 50, 250)

    pitch = map(data['user/pitch'], 1000, 2000, 500, 700)
    roll = map(data['user/roll'], 1000, 2000, 1000, 1200)

    cv2.circle(img, (yaw, throttle), 20, (255, 0, 0), -1)
    cv2.circle(img, (roll, pitch), 20, (255, 0, 0), -1)

    return img


def plot(data):
    plot_data.append(data)
    ax.cla()
    ax.plot(plot_data)


def main(i):
    global count
    f = open(c.visualizer_folder + "/control/record_" + str(count) + ".json")
    data = json.load(f)
    img = cv2.imread(c.visualizer_folder + "/left_camera/" + data['cam/image_name'])

    img = drawUI(img, data)

    cv2.imshow("data visualizer", img)
    cv2.waitKey(1)
    # time.sleep(playback_speed)
    plot(data['gps/path_distance'])
    count += 1


ani = FuncAnimation(fig, main, interval=c.playback_speed)

plt.show()

# while True:

#     f = open(folder + "/control/record_" + str(count) + ".json")
#     data = json.load(f)
#     img = cv2.imread(folder + "/left_camera/" + data['cam/image_name'])

#     img = drawUI(img, data)

#     cv2.imshow("data visualizer", img)
#     cv2.waitKey(1)
#     time.sleep(playback_speed)
#     count +=1
