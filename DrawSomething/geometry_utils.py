"""
geometry utils - helper function (using for detecting shape)

"""
# geometry_utils.py

import math
import numpy as np
from typing import List, Tuple

def rotate_point(x: float, y: float, theta: float, center: Tuple[float,float]) -> Tuple[float,float]:
    """
    Rotate (x, y) around 'center' by angle 'theta' (radians).
    """
    # Extract the coordinates of the center of rotation.
    a, b = center

    # Translate the point so that the center of rotation becomes the origin.
    # This is achieved by subtracting the center's coordinates from the point's coordinates.
    x_rel = x - a
    y_rel = y - b

    # Apply the rotation transformation using the standard rotation formulas:
    #   x_rot = x_rel * cos(theta) - y_rel * sin(theta)
    #   y_rot = x_rel * sin(theta) + y_rel * cos(theta)
    # These formulas compute the coordinates of the rotated point relative to the origin.
    x_rot = x_rel * math.cos(theta) - y_rel * math.sin(theta)
    y_rot = x_rel * math.sin(theta) + y_rel * math.cos(theta)

    # Translate the rotated coordinates back to the original coordinate system by adding the center's coordinates.
    x_new = x_rot + a
    y_new = y_rot + b

    # Return the new coordinates of the rotated point.
    return x_new, y_new

def line_segment_distance(px: float, py: float,
                         x1: float, y1: float,
                         x2: float, y2: float) -> float:
    """
    Returns minimum distance from point (px, py) to the line segment
    defined by (x1,y1)->(x2,y2).
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    # If the segment length is zero
    if seg_len_sq == 0:
        # The distance is simply from the point to (x1, y1)
        return math.hypot(px - x1, py - y1)

    # Compute t, the normalized projection of (px, py) onto the segment
    # t = 0 => perpendicular to (x1, y1), t = 1 => perpendicular to (x2, y2), between 0 and 1 => lies on segment
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq

    if t < 0:
        # Closest to (x1, y1)
        return math.hypot(px - x1, py - y1)
    elif t > 1:
        # Closest to (x2, y2)
        return math.hypot(px - x2, py - y2)
    else:
        # Projection falls within the segment
        projx = x1 + t * dx
        projy = y1 + t * dy
        return math.hypot(px - projx, py - projy)


def chaikin_smoothing(points, iterations=2):
    if len(points) < 3:
        return points
    new_points = points
    for _ in range(iterations):
        new_points = chaikin_once(new_points)
    return new_points

def chaikin_once(points):
    if len(points) < 3:
        return points

    new_pts = [points[0]]
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        qx = 0.75 * p1[0] + 0.25 * p2[0]
        qy = 0.75 * p1[1] + 0.25 * p2[1]
        rx = 0.25 * p1[0] + 0.75 * p2[0]
        ry = 0.25 * p1[1] + 0.75 * p2[1]
        new_pts.append((qx, qy))
        new_pts.append((rx, ry))
    new_pts.append(points[-1])
    return new_pts