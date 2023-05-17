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
def canny_edge_detector_einat(input_image):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    grayscaled = compute_grayscale(input_pixels, width, height)

    # Blur it to remove noise
    blurred = compute_blur(grayscaled, width, height)

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep, gradient, direction

def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = pixel
            # grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale

def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred

"""*********"""

# find points in a specific distance
def find_points(x, y, m, distance):
    point_b = (x + dx(distance, m), y + dy(distance, m))
    other_possible_point_b = (x - dx(distance, m), y - dy(distance, m))  # going the other way
    return point_b, other_possible_point_b


def dy(distance, m):
    return m * dx(distance, m)


def dx(distance, m):
    return sqrt(distance ** 2 / (m ** 2 + 1))
