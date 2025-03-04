import cv2
import numpy as np

def fuse_color_motion(ratio_map_online, p_skin_offline, motion_prob, w_color=0.7, w_motion=0.6, threshold=0.6):
    """
    Fuse color and motion probabilities.
    Both inputs should be normalized to [0,1].
    Returns a binary mask.
    """
    motion_prob = motion_prob/255
    x = ratio_map_online + p_skin_offline
    p_skin_combined = (x-np.min(x))/(np.max(x)-np.min(x))
    combined = (w_color * p_skin_combined + w_motion * motion_prob) / (w_color + w_motion)
    return (combined > threshold).astype(np.uint8) * 255, combined


def get_high_motion_mask(fg_mask, high_thresh=60):
    """
    Apply a high threshold to the foreground mask (from MOG2) to keep only strongly moving pixels.
    """
    _, high_motion = cv2.threshold(fg_mask, high_thresh, 255, cv2.THRESH_BINARY)
    return high_motion