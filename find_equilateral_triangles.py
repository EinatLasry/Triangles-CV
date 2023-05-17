import cv2
import numpy as np
from math import sqrt, pi, cos, sin, atan2, ceil, floor, radians
from PIL import Image, ImageDraw
import random

def find_equilateral_triangles(img, length_side):

    input_image = img
    # Output image:
    output_image = Image.new("RGB", input_image.size)
    output_image.paste(input_image)
    draw_result = ImageDraw.Draw(output_image)

    height = length_side*0.5*sqrt(3)  # Height in equilateral triangle
    cols, rows = img.shape
    points = np.zeros((rows+1, cols+1, 120)) # In equilateral triangle, all sides have the same angle, if we apply modulo 120
    # 1. canny
    # 2. calculate gradient and theta
    edges_arr = cv2.Canny(img, 450, 500)
    edges_list, gradient, direction = canny_edge_detector_einat(input_image)
    # binary img
    edges_arr[edges_arr < 127] = 0
    edges_arr[edges_arr >= 127] = 255
    edges = []
    """ 
    IN PROCESS
    """


# find points in a specific distance
def find_points(x, y, m, distance):
    point_b = (x + dx(distance, m), y + dy(distance, m))
    other_possible_point_b = (x - dx(distance, m), y - dy(distance, m))  # going the other way
    return point_b, other_possible_point_b


def dy(distance, m):
    return m * dx(distance, m)


def dx(distance, m):
    return sqrt(distance ** 2 / (m ** 2 + 1))
