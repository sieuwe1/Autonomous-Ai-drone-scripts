import numpy as np
from numpy.linalg import lstsq
import math
import scipy.odr as odr
import matplotlib.pyplot as plt


def calculate_path_distance(target, start, current, vis=False): #current_cordinate = drone.get_location()  current = np.asarray((current_cordinate.lat,current_cordinate.lon))
    
    #calculate equation for line between gps points
    points = [target,start]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m1, b1 = lstsq(A, y_coords)[0]
    
    #calculate equation for line perpendicular to gps line
    m2 = (-1.0 / m1)
    b2 = current[1] - (m2 * current[0]) 
    
    #calculate intersection
    x_intersect = (b1-b2) / (m2-m1)
    y_intersect = m1 * x_intersect + b1

    if vis:
        fig,ax = plt.subplots()
        plt.plot(target[0],target[1],'ro')
        plt.plot(start[0],start[1],'ro')

        x = np.linspace([start[0]],target[0])
        ax.plot(x,m1*x+b1,'r')
        ax.plot(x,m2*x+b2,'g')
        ax.plot(current[0],current[1],'bo')

        ax.plot(x_intersect,y_intersect,'go')

        #plt.plot(xi, yi, 'bo')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    #calcualte distances between points in meters
    R = 6373.0

    target_r = (math.radians(target[0]),math.radians(target[1]))
    current_r = (math.radians(current[0]),math.radians(current[1]))
    intersection_r = (math.radians(x_intersect),math.radians(y_intersect))

    target_d_lon = target_r[0] - current_r[0] 
    target_d_lat = target_r[1] - current_r[1] 

    a = math.sin(target_d_lat / 2)**2 + math.cos(target_r[1]) * math.cos(current_r[1]) * math.sin(target_d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    target_distance = (R * c) * 1000

    path_d_lon = intersection_r[0] - current_r[0] 
    path_d_lat = intersection_r[1] - current_r[1] 

    a = math.sin(path_d_lat / 2)**2 + math.cos(intersection_r[1]) * math.cos(current_r[1]) * math.sin(path_d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    path_distance = (R * c) * 1000

   # print("target distance: " + str(target_distance))    
   # print("path distance: " + str(path_distance))

    return target_distance, path_distance
    

def calculate_heading_difference(heading,target,current):
    brng = calculate_initial_compass_bearing(current,target)
    #print("bearing: " + str(brng))
    #print("heading: " + str(heading))

    delta = abs(brng - heading)
    if delta > 180:
        return 360 - delta
    else:
        return delta 

def calculate_target(start,heading):
    distance_to_target = 50 #meter
    print("heading: " + str(heading))
    lat0 = math.cos(math.pi / 180.0 * start[0])
    a = math.radians(heading);  
    x = start[0] + (180/math.pi) * (distance_to_target / 6378137) / math.cos(lat0) * math.cos(a)
    y = start[1] + (180/math.pi) * (distance_to_target / 6378137) * math.sin(a)

    return (x,y)

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

#if __name__ == "__main__":
    #target = calculate_target((51.45068,5.45525),45)
    #calculate_path_distance(target,(51.45068,5.45525),(51.4509297236778, 5.455395981176845))
    #calculate_path_distance((5,10),(20,40),(10,35))
                                        #heading, target, current 
    #print(calculate_heading_difference(270,(51.45152420417108, 5.4547439964670446), (51.45224546110723, 5.454700931022534)))
