
"""
The goal is to automatically detect equilateral triangles
in natural images using a variant of the Hough Transform.
"""
# import cv2 # C:\Users\English user\anaconda3\Lib\site-packages\cv2
import numpy as np
from math import sqrt, pi, cos, sin, atan2, ceil, floor, radians
from PIL import Image, ImageDraw
import random
import find_equilateral_triangles

input_image = Image.open('C:/Users/English user/Desktop/image001.jpg')

