"""
fingertip detection - helper functions

"""

from DrawSomething import constants
import cv2
import numpy as np

def preprocess_mask(mask, kernel_size=3, iterations=1):
    """
    Apply basic morphological operations to remove noise and small artifacts.
    Adjust the kernel size and iterations based on your particular needs.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Morphological opening: removes small white noise
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    # Morphological closing: closes small holes inside foreground objects
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return cleaned

def find_fingertip(mask):
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    max_contour = max(contours, key=cv2.contourArea)
    topmost = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    return topmost[0], topmost[1]


def detect_fingertip(mask):
    """
    Detect the fingertip by:
    1. Finding the largest contour in the mask
    2. Computing its centroid
    3. Building the convex hull
    4. Selecting the hull point farthest from the centroid and above it
    Returns (x, y) or None if no valid fingertip is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the largest contour - presumed to be the hand
    hand_contour = max(contours, key=cv2.contourArea)
    # Find the convex hull
    hull = cv2.convexHull(hand_contour)

    # Calculate centroid of the hand
    M = cv2.moments(hand_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    max_distance = 0
    fingertip = None

    # Scan hull points, choose farthest from centroid and above it
    for point in hull[:, 0]:
        x, y = point
        # You can also try removing the "y > cy" check if you want the
        # absolute farthest point in any direction. For a typical hand pose,
        # ignoring points below the centroid often helps find the top fingertip.
        if y > cy:
            continue
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if distance > max_distance:
            max_distance = distance
            fingertip = (x, y)

    return fingertip


def is_valid_fingertip(new_tip, prev_tip, threshold=constants.JUMP_THRESHOLD):
    """
    Decide if the new fingertip is valid by comparing distance
    with the previous known fingertip. If distance exceeds the jump threshold,
    it is considered invalid (spurious).
    """
    if prev_tip is None or new_tip is None:
        # If we don't have a previous tip or a new tip is None, just accept it
        return True
    distance = np.linalg.norm(np.array(new_tip) - np.array(prev_tip))
    return distance < threshold

def dynamic_ema_alpha(speed):
    """
    Get a smoothing factor alpha based on the movement speed of the fingertip.
    Slower movement => more smoothing; faster movement => less smoothing.
    """
    if speed < constants.SPEED_THRESHOLD_LOW:
        return constants.EMA_ALPHA_MIN
    elif speed > constants.SPEED_THRESHOLD_HIGH:
        return constants.EMA_ALPHA_MAX
    else:
        # Interpolate alpha based on speed
        return constants.EMA_ALPHA_MIN + (constants.EMA_ALPHA_MAX - constants.EMA_ALPHA_MIN) * (
            (speed - constants.SPEED_THRESHOLD_LOW)
            / float(constants.SPEED_THRESHOLD_HIGH - constants.SPEED_THRESHOLD_LOW)
        )

def smooth_fingertip(curr_tip, prev_tip):
    """
    Applies exponential smoothing to the current fingertip estimate
    based on how quickly it's moving.
    """
    if prev_tip is None:
        return curr_tip

    speed = np.linalg.norm(np.array(prev_tip) - np.array(curr_tip))
    alpha = dynamic_ema_alpha(speed)

    x_smooth = int(alpha * curr_tip[0] + (1 - alpha) * prev_tip[0])
    y_smooth = int(alpha * curr_tip[1] + (1 - alpha) * prev_tip[1])
    return (x_smooth, y_smooth)