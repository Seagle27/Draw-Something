from functools import wraps
from time import time
import cv2
import numpy as np
import math



def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r, took: %2.4f sec' % \
              (f.__name__, te - ts))
        return result

    return wrap


def compute_edge_orientation(gray_img, low_threshold=50, high_threshold=100):
    """
    Run the Canny edge detector on 'gray_img' and compute gradient orientation
    (via Sobel) only at edge pixels.

    Returns:
      - orientation_masked: array of same shape as gray_img, containing gradient
        orientation (in radians) where edges exist, and 0 elsewhere.
      - edges: binary edge map from Canny.
    """
    # Run Canny to get edges
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)

    # Compute gradients using Sobel
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)

    # Compute orientation (in radians)
    orientation = cv2.phase(gx, gy, angleInDegrees=False)

    # Mask orientation so that only edge pixels have non-zero values
    orientation_masked = np.where(edges > 0, orientation, 0)
    cv2.imshow("orientation", orientation_masked)

    return orientation_masked, edges


def angle_difference(a, b):
    """
    Compute minimal angular difference (in radians) between angles a and b.
    The result is in [0, pi].
    """
    a_mod = a % math.pi
    b_mod = b % math.pi
    diff = np.abs(a_mod - b_mod)
    diff = np.where(diff > math.pi / 2, math.pi - diff, diff)
    return diff

