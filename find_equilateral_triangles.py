import cv2
import numpy as np
from math import sqrt, pi, cos, sin, atan2, ceil, floor, radians
from PIL import Image, ImageDraw
import random

def find_equilateral_triangles(img, length_side):

    input_image = Image.open('Q1/triangles_2/image008.jpg')
    # Output image:
    output_image = Image.new("RGB", input_image.size)
    output_image.paste(input_image)
    draw_result = ImageDraw.Draw(output_image)

    height = length_side*0.5*sqrt(3)
    cols, rows = img.shape
    points = np.zeros((rows+1, cols+1, 120))
    # 1. canny
    # 2. calculate gradient and theta
    edges_arr = cv2.Canny(img, 450, 500)
    edges_list, gradient, direction = canny_edge_detector_einat(input_image)
    # binary img
    edges_arr[edges_arr < 127] = 0
    edges_arr[edges_arr >= 127] = 255
    edges = []

    for y in range(cols):
        for x in range(rows):
            if edges_arr[y, x] == 255:
                edges.append((x, y))

    # save edges img
    edges_img = Image.new("RGB", input_image.size)
    draw = ImageDraw.Draw(edges_img)
    for x, y in edges:
        draw.point((x, y), (255, 255, 255))
    edges_img.save("edges_img.jpg")

    # 4. for edge: for (x-side/2:x+side/2): for (y-side/2:y+side/2): if you didnt vote yet, vote to: (x+height/3, y+height/3)
    parallel_y = 0  # flag
    parallel_x = 0  # flag
    for x, y in edges:
        # find m
        theta = round(np.rad2deg(direction[x, y])) % 120
        if theta == 90:  # gradient theta (Vertical to the rib)
            m = 0
            parallel_x = 1
        elif theta == 0:
            m = 0
            parallel_y = 1
        else:
            m = np.tan(direction[x, y])
            m = -1 / m

        n = y - m * x  # y=mx+n

        # find points_b
        if parallel_x == 1:
            point_b = (x + length_side / 2, y)
            other_possible_point_b = (x - length_side / 2, y)
        elif parallel_y == 1:
            point_b = (x, y + length_side / 2)
            other_possible_point_b = (x, y - length_side / 2)
        else:
            point_b, other_possible_point_b = find_points(x, y, m, length_side/2)
            if point_b[0] < other_possible_point_b[0]:
                point = point_b
                point_b = other_possible_point_b
                other_possible_point_b = point

        # votes :
        if parallel_y == 1:
            for y1 in range(floor(other_possible_point_b[1]), ceil(point_b[1])):
                x1 = x  # parallel_y = x is constant
                # VOTES
                center1 = (x1 + height / 3, y1)
                center2 = (x1 - height / 3, y1)
                if 0 <= center1[0] < rows and 0 <= center1[1] < cols:
                    points[round(center1[0]), round(center1[1]), theta] += 1
                if 0 <= center2[0] < rows and 0 <= center2[1] < cols:
                    points[round(center2[0]), round(center2[1]), theta] += 1
        elif parallel_x == 1:
            for x1 in range(floor(other_possible_point_b[0]), ceil(point_b[0])):
                y1 = y  # parallel_x = y is constant
                # VOTES
                center1 = (x1, y1 + height / 3)
                center2 = (x1, y1 - height / 3)
                if 0 <= center1[0] < rows and 0 <= center1[1] < cols:
                    points[round(center1[0]), round(center1[1]), theta] += 1
                if 0 <= center2[0] < rows and 0 <= center2[1] < cols:
                    points[round(center2[0]), round(center2[1]), theta] += 1
        else:
            for x1 in range(floor(other_possible_point_b[0]), ceil(point_b[0])):
                y1 = m * x1 + n
                # VOTES
                if m == 0:
                    m2 = 0
                else:
                    m2 = -1 / m
                center1, center2 = find_points(x1, y1, m2, height / 3)
                if 0 <= center1[0] < rows and 0 <= center1[1] < cols:
                    points[round(center1[0]), round(center1[1]), theta] += 1
                if 0 <= center2[0] < rows and 0 <= center2[1] < cols:
                    points[round(center2[0]), round(center2[1]), theta] += 1
        parallel_y = 0  # flag
        parallel_x = 0  # flag

    # complete threshold and winsize
    threshold = np.amax(points)-3
    winsize = [1097, 835]
    triangles = []
    # non max suppression
    i = 1
    print("threshold: ", threshold)
    # NON MAX SUPPRESSION
    for y in range(0, cols, winsize[0]):
        for x in range(0, rows, winsize[1]):
            y2 = min(y+winsize[0], cols-1)
            x2 = min(x+winsize[1], rows-1)
            a, b, t = np.unravel_index(points[x:x2, y:y2, :].argmax(), points[x:x2, y:y2, :].shape)
            max_val = points[x+a, y+b, t]
            points[x:x2, y:y2, :] = 0
            points[x+a, y+b, t] = max_val
            if max_val >= threshold:
                print("Found triangle: ", i)
                print(x+a, y+b, t, "votes:", max_val)
                i += 1
                triangles.append((x+a, y+b, t))

    # Draw

    for x, y, theta in triangles:
        # find a center of a side in the triangle
        if theta == 90:  # parallel_x
            m = 0
            center1 = (x, y - height/3)
            center2 = (x, y + height/3)
        elif theta == 0:  # parallel_y
            m = 0
            center1 = (x + height / 3, y)
            center2 = (x - height / 3, y)
        else:
            m = np.tan(theta)
            center1, center2 = find_points(x, y, m, height/3)
        a = b = c = []
        if center2 in edges:
            center = center2  # midpoint
        else:
            center = center1  # midpoint

        if theta == 90:  # parallel_x
            c = (center[0] + length_side/2, center[1])
            b = (center[0] - length_side/2, center[1])
        elif theta == 0:  # parallel_y
            c = (center[0], center[1] + length_side/2)
            b = (center[0], center[1] - length_side/2)
        else:
            c, b = find_points(center[0], center[1], -1/m, length_side/2)
        a = (3*x - 2*center[0], 3*y - 2*center[1])
        draw_result.polygon((a, b, c), fill=None, outline=(255, 0, 0))
    # Save output image
    output_image.save("result.jpg")

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

def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction

def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0

def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)

# find points in a specific distance
def find_points(x, y, m, distance):
    point_b = (x + dx(distance, m), y + dy(distance, m))
    other_possible_point_b = (x - dx(distance, m), y - dy(distance, m))  # going the other way
    return point_b, other_possible_point_b


def dy(distance, m):
    return m * dx(distance, m)


def dx(distance, m):
    return sqrt(distance ** 2 / (m ** 2 + 1))
