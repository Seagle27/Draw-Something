import numpy as np
import cv2


def segment_hand_only(final_mask, face_mask):
    """
    Given:
      final_mask  -> a binary image (0/255) from your hybrid skin segmentation
      frame_bgr   -> original frame in BGR
      face_mask   -> a binary image

    Returns a mask containing ONLY the largest skin region outside the face region,
    presumably the user's hand.
    """

    # Exclude the face region(s) from the mask
    hand_mask = final_mask.copy()  # keep a working copy
    hand_mask[face_mask] = 0

    # Ensure hand_mask is a proper single-channel 8-bit
    # It should already be, but just in case:
    if hand_mask.ndim == 3:
        hand_mask = cv2.cvtColor(hand_mask, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, hierarchy = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No contours found -> return an empty mask
        return np.zeros_like(hand_mask)

    # Pick the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create an output mask with ONLY that largest contour
    hand_only_mask = np.zeros_like(hand_mask)
    cv2.drawContours(hand_only_mask, [largest_contour], contourIdx=-1, color=255, thickness=-1)

    return hand_only_mask