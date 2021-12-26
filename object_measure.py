import cv2
import numpy as np
import math

Angle = 50
Distance = 100

def measure():
    tan = math.tan(math.pi * (Angle/180))
    print(tan)
    standard_size = tan*Distance

if __name__ == '__main__':
    measure()
