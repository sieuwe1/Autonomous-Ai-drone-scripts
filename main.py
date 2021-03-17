import sys
sys.path.insert(1, 'modules')

import lidar

lidar.connect_lidar("/dev/ttyTHS1")

lidar.read_lidar_distance()
lidar.read_lidar_temperature()

