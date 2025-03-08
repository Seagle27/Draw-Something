"""
shape detection functions

"""

import numpy as np
import cv2
import math

from DrawSomething.error_matrices import error_line,error_circle,error_ellipse,error_triangle,error_rectangle,count_close_point_triangle
from DrawSomething.geometry_utils import rotate_point, chaikin_smoothing
from DrawSomething.constants import  THRESHOLD_abstract,THRESHOLD, NUM_BINS,GAUSS_BLUR_K,GAUSS_BLUR_SIGMA,SOBEL_K

def detect_shape_by_gradient_orientations_from_points(points,
                                                      low_threshold=100,
                                                      peak_threshold_ratio=0.2):
    if len(points) < 2:
        return "Unknown"

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = int(min(xs)), int(max(xs))
    min_y, max_y = int(min(ys)), int(max(ys))

    width = max_x - min_x + 20
    height = max_y - min_y + 20
    if width < 2 or height < 2:
        return "Unknown"

    img = np.ones((height, width), dtype=np.uint8) * 255
    shift_x = min_x - 10
    shift_y = min_y - 10

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        cv2.line(img,
                 (int(x1 - shift_x), int(y1 - shift_y)),
                 (int(x2 - shift_x), int(y2 - shift_y)),
                 color=(0,),
                 thickness=4)

    blurred = cv2.GaussianBlur(img, (GAUSS_BLUR_K, GAUSS_BLUR_K), GAUSS_BLUR_SIGMA)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=SOBEL_K)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=SOBEL_K)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    angle = np.arctan2(sobel_y, sobel_x)
    angle_deg = np.degrees(angle)

    valid_mask = magnitude > low_threshold
    valid_angles = angle_deg[valid_mask]
    if len(valid_angles) == 0:
        return "abstract"

    bins = NUM_BINS
    hist, bin_edges = np.histogram(valid_angles, bins=bins, range=(-180, 180))
    max_val = np.max(hist)
    if max_val == 0:
        return "abstract"

    peak_threshold = peak_threshold_ratio * max_val
    peaks = []
    for i in range(1, bins - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] >= peak_threshold:
            peaks.append(i)

    i = bins - 1
    if hist[i] > hist[i - 1] and hist[i] >= peak_threshold:
        peaks.append(i)

    num_peaks = len(peaks)
    if num_peaks == 6:
        shape_name = "Triangle Shape"
    elif num_peaks == 4:
        shape_name = "Square Shape"
    elif num_peaks == 5:
        shape_name = "Triangle or Square Shape"
    elif num_peaks >= 9:
        shape_name = "Circular Shape"
    elif num_peaks in [7, 8]:
        shape_name = "Unknown Shape"
    else:
        shape_name = "abstract"
    print(f"Detected shape: {shape_name}, with {num_peaks} major peaks in orientation.")
    return shape_name


def best_fit_shape(points):
    smoothed = chaikin_smoothing(points, iterations=2)
    aprrox_shape = detect_shape_by_gradient_orientations_from_points(smoothed)

    best_shape = "abstract"
    best_error = float("inf")
    best_count = 0
    best_angle = 0
    aligned_edge = "None"

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    center = (cx, cy)

    # for debug
    candidate = []
    if aprrox_shape == "Triangle Shape":
        candidate = ["triangle"]
    elif aprrox_shape == "Square Shape":
        candidate = ["rectangle"]
    elif aprrox_shape == "Circular Shape":
        candidate = ["circle","ellipse"]
    elif aprrox_shape == "Triangle or Square Shape":
        candidate = ["triangle", "rectangle"]
    elif aprrox_shape == "Unknown Shape":
        candidate = ["line","circle", "ellipse","triangle","rectangle"]
    else:
        candidate = ["line", "ellipse"]
    print(f"Shape name: {aprrox_shape},Candidates:{candidate}")



    for angle in range(0, 91, 5):
        rotated_points = []
        theta_rad = math.radians(angle)
        for point in points:
            rotated_point = rotate_point(point[0], point[1], -theta_rad, center)
            rotated_points.append(rotated_point)

        if aprrox_shape == "Triangle Shape":
            candidates_rotates = {"triangle": count_close_point_triangle(rotated_points)}
        elif aprrox_shape == "Square Shape":
            candidates_rotates = {"rectangle": error_rectangle(rotated_points)}
        elif aprrox_shape == "Circular Shape":
            candidates_rotates = {
                "circle": error_circle(rotated_points),
                "ellipse": error_ellipse(rotated_points)
            }
        elif aprrox_shape =="Triangle or Square Shape":
            candidates_rotates = {
                "rectangle": error_rectangle(rotated_points),
                "triangle": count_close_point_triangle(rotated_points)
            }
        elif aprrox_shape == "Unknown Shape":
            candidates_rotates = {
                "line": error_line(rotated_points),
                "circle": error_circle(rotated_points),
                "ellipse": error_ellipse(rotated_points),
                "triangle1": error_triangle(rotated_points),
                "rectangle": error_rectangle(rotated_points)
            }
        else:
            candidates_rotates = {
                "line": error_line(rotated_points),
                "ellipse": error_ellipse(rotated_points)
            }
        #print(f"\nAngle: {angle}°,Shape name: {aprrox_shape},Candidates:{candidates_rotates}")

        for shape_name, err in candidates_rotates.items():
            if shape_name != "triangle":
                if err < best_error:
                    best_error = err
                    best_shape = shape_name
                    best_angle = angle
                    aligned_edge = "None"
            else:
                count, side = err
                if count > best_count and side != "None":
                    best_count = count
                    best_shape = shape_name
                    best_angle = angle
                    aligned_edge = side

        if best_error < 7:
            break

    if aprrox_shape == "Triangle Shape":
        return best_shape, best_angle, aligned_edge
    elif aprrox_shape == "Unknown Shape" and best_shape == "triangle":
        for angle in range(0, 91, 5):
            rotated_points = []
            theta_rad = math.radians(angle)
            for point in points:
                rotated_point = rotate_point(point[0], point[1], -theta_rad, center)
                rotated_points.append(rotated_point)
                count, side = count_close_point_triangle(rotated_points)
                if count > best_count and side != "None":
                    best_count = count
                    best_shape = shape_name
                    best_angle = angle
                    aligned_edge = side
        return best_shape, best_angle, aligned_edge
    elif ((aprrox_shape == "abstract" and best_error < THRESHOLD_abstract) or
          (aprrox_shape == "Unknown Shape" and best_error < THRESHOLD_abstract) or
          (aprrox_shape in ["Circular Shape", "Square Shape"] and best_error < THRESHOLD)):
        return best_shape, best_angle, aligned_edge

    return "abstract", 0, "None"