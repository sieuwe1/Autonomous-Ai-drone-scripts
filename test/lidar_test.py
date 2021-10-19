import sys
sys.path.insert(1, 'modules')

import lidar

print(lidar.connect_lidar("/dev/ttyTHS1"))

print(lidar.check_connection())

print(lidar.read_lidar_distance())
print(lidar.read_lidar_temperature())

lidar.disconnect_lidar()

print(lidar.check_connection())