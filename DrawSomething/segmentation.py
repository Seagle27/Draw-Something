import numpy as np
import cv2


def segment_hand_only(final_mask, face_mask):
    """
    Given:
      final_mask  -> a binary image (0/255) from your hybrid skin segmentation
      face_mask   -> a binary image

    Returns a mask containing ONLY the largest skin region outside the face region,
    presumably the user's hand.
    """

    # Exclude the face region(s) from the mask
    hand_mask = final_mask.copy()  # keep a working copy
    hand_mask[face_mask] = 0
    return largest_contour_segmentation(hand_mask)


def segment_hand_with_face_overlap(
        frame_bgr,
        final_skin_mask,
        face_bbox,
        face_buffer,
        movement_threshold=20,
        min_motion_area=50
):
    """
    Given:
      - frame_bgr: current color frame (BGR).
      - final_skin_mask: the skin mask from the hybrid approach (0/255).
      - face_bbox: (x, y, w, h) for face region, or None if no face found.
      - face_buffer: grayscale face buffer from the previous frames' face region
                        (or None if no face).
      - movement_threshold: absolute difference threshold for considering a pixel "moving".
      - min_motion_area: minimum # of changed pixels in the face region to consider.

    Returns:
      - updated_mask: an updated final skin mask that re-includes
                      any "moving" region in the face box.
      - current_face_gray: the grayscale patch from this frame, to store for the next loop.
    """
    updated_mask = final_skin_mask.copy()
    current_face_gray = None

    if face_bbox is None:
        # No face found or tracker lost => can't do partial face removal
        return updated_mask, None

    x, y, w, h = map(int, face_bbox)

    # Make sure the bounding box is in frame bounds
    H, W = frame_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x1 >= x2 or y1 >= y2:
        # invalid region
        return updated_mask, None

    # Extract the face region in grayscale
    face_region_bgr = frame_bgr[y1:y2, x1:x2]
    face_region_gray = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2GRAY)
    current_face_gray = face_region_gray.copy()

    # If we have a prev_face_gray from the previous frame,
    # compute the absolute difference
    if all(prev is not None and prev.shape == face_region_gray.shape for prev in face_buffer):
        diffs = [cv2.absdiff(face_region_gray, prev) for prev in face_buffer]

        # Threshold the difference
        motion_masks = [(diff > movement_threshold).astype(np.uint8) * 255 for diff in diffs]
        bitwise_and_result = motion_masks[0]
        for motion_mask in motion_masks[1:]:
            bitwise_and_result = cv2.bitwise_and(bitwise_and_result, motion_mask)

        # Optional: morphological ops to remove noise
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # bitwise_and_result = cv2.morphologyEx(bitwise_and_result, cv2.MORPH_OPEN, kernel)
        # bitwise_and_result = cv2.morphologyEx(bitwise_and_result, cv2.MORPH_CLOSE, kernel)

        face_area_mask = np.where(bitwise_and_result > 0, 255, 0).astype(np.uint8)

        updated_mask[y1:y2, x1:x2] = face_area_mask

    else:
        print("No prev_face_gray or shape mismatch => no motion check.")

    return updated_mask, current_face_gray


def largest_contour_segmentation(binary_img):
    """
    find the largest contour in a segmentation img, and return a mask just with the segmented object
    """
    largest_contour_mask = np.zeros_like(binary_img)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No contours found -> return an empty mask
        return largest_contour_mask

    # Pick the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create an output mask with ONLY that largest contour
    cv2.drawContours(largest_contour_mask, [largest_contour], contourIdx=-1, color=255, thickness=-1)
    return largest_contour_mask
