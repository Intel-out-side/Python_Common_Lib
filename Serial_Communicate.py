"""
Note:
    1. I use the 6-axis IMU sensor (MPU-6050)
    2. If you use other types, then you need to check the data sheet.
"""

import serial
import math


class MySerial:

    def __init__(self, COM, BaudRate=19200):
        self.ser = serial.Serial(COM, BaudRate)
        # construct with COM number and BaudRate
        # use this class like ser = Serial(14, 9600)
        # COM number is

    def get_raw(self):
        """
        data sent from the microcomputer must be:
        [ax; ay; az; gx; gy; gz; t]

        :return:
        """

        try:
            data = self.ser.readline()
            data = data.decode().split(';')

        except:
            data = [0, 0, 0, 0, 0, 0, 0]
            print("Oops, something going wrong.\n")

        return data

    def get_angle(self):
        """
        input sent by the microcomputer must be like:
        [ax, ay, az, gx, gy, gz, t]
        :return: pitch, roll
        """
        try:
            data = self.ser.readline()
            data = data.decode().split(';')

        except:
            data = [0, 0, 0, 0, 0, 0, 0]
            print("Oops, something going wrong.\n")

        ax = data[0]/16384
        ay = data[1]/16384
        az = data[2]/16384

        acc_pitch = math.atan2(ax, math.sqrt(az**2 + ay**2)) * 360 / 2 / math.pi
        acc_roll = math.atan2(ay, math.sqrt(ax**2 + az**2)) * 360 / 2 / math.pi

        return acc_pitch, acc_roll