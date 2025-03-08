"""
print on frame - helper functions

"""
import cv2

def print_gesture_on_frame(frame, gesture="no gesture"):
    text = f"Detected Gesture: {gesture}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 1
    text_color = (255, 255, 255)
    cv2.putText(frame, text, (40, 440), font, font_scale, text_color, thickness, cv2.LINE_AA)


def print_color_on_frame(frame,color):
    text = f"Detected Color: {color}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 1
    text_color = (255, 255, 255)
    cv2.putText(frame, text, (40, 460), font, font_scale, text_color, thickness, cv2.LINE_AA)

