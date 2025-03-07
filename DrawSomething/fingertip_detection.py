from DrawSomething import constants
import cv2
import numpy as np


def find_fingertip(mask):
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    max_contour = max(contours, key=cv2.contourArea)
    topmost = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    return topmost[0], topmost[1]


def detect_fingertip(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hand_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(hand_contour)
    M = cv2.moments(hand_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    max_distance = 0
    fingertip = None
    for point in hull[:, 0]:
        x, y = point
        if y > cy:
            continue
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if distance > max_distance:
            max_distance = distance
            fingertip = (x, y)
    return fingertip


def is_valid_fingertip(new_tip, prev_tip, threshold=constants.JUMP_THRESHOLD):
    if prev_tip is None:
        return True
    distance = np.linalg.norm(np.array(new_tip) - np.array(prev_tip))
    return distance < threshold


def dynamic_ema_alpha(speed):
    if speed < constants.SPEED_THRESHOLD_LOW:
        return constants.EMA_ALPHA_MIN
    elif speed > constants.SPEED_THRESHOLD_HIGH:
        return constants.EMA_ALPHA_MAX
    else:
        return constants.EMA_ALPHA_MIN + (constants.EMA_ALPHA_MAX - constants.EMA_ALPHA_MIN) * (
                (speed - constants.SPEED_THRESHOLD_LOW)
                / (constants.SPEED_THRESHOLD_HIGH - constants.SPEED_THRESHOLD_LOW)
        )


def smooth_fingertip(curr_tip, prev_tip):
    if prev_tip is None:
        return curr_tip
    speed = np.linalg.norm(np.array(prev_tip) - np.array(curr_tip))
    alpha = dynamic_ema_alpha(speed)
    return (
        int(alpha * curr_tip[0] + (1 - alpha) * prev_tip[0]),
        int(alpha * curr_tip[1] + (1 - alpha) * prev_tip[1])
    )
