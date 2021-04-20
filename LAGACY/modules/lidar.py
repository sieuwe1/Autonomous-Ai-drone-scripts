import serial,time
import numpy as np

#https://www.robotshop.com/media/files/content/b/ben/pdf/tf-luna-8m-lidar-distance-sensor-instructions-manual.pdf
ser = None 

def connect_lidar(serialString):
    global ser
    ser = serial.Serial(serialString, 115200,timeout=0) 
    if ser.isOpen() == False:
        ser.open() 
        return "succes"
    else:
        return "port already open"

def disconnect_lidar():
    global ser
    ser.close()

def check_connection():
    global ser
    return ser.isOpen()

def read_lidar_distance():
    global ser
    while True:
        counter = ser.in_waiting 
        if counter > 6:
            bytes_serial = ser.read(7) 
            ser.reset_input_buffer() 
            if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59: 
                distance = bytes_serial[2] + bytes_serial[3]*256 
                strength = bytes_serial[4] + bytes_serial[5]*256 
                return distance/100.0,strength

def read_lidar_temperature():
    global ser
    while True:
        counter = ser.in_waiting 
        if counter > 8:
            bytes_serial = ser.read(9) 
            ser.reset_input_buffer() 

            if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59: 
                temperature = bytes_serial[6] + bytes_serial[7]*256 
                temperature = (temperature/8.0) - 256.0 
                return temperature
