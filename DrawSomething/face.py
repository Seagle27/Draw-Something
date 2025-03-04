import cv2
import numpy as np
from DrawSomething import constants
from DrawSomething.utils import utility_functions


def initialize_face_template(frame_bgr, face_bbox):
    """
    Extract the face region from 'frame_bgr' and compute:
      - The gray-scale face patch.
      - The edge orientation map using Canny (for later occlusion detection).
      - The original color patch.
    Returns a dictionary with template info.
    """
    (x, y, w, h) = face_bbox
    face_bgr = frame_bgr[y:y + h, x:x + w]

    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    orientation_map, _ = utility_functions.compute_edge_orientation(face_gray)

    template = {
        'bbox': (x, y, w, h),  # initial face bounding box
        'color': face_bgr,  # face color patch
        'orientation': orientation_map  # edge orientation map from Canny
    }
    return template


def manage_face_detection_and_tracking(
        frame,
        frame_count,
        skip_interval,
        face_cascade,
        face_tracker,
        face_bbox,
        update_flag
):
    """
    Detect or track a face in the given frame.

    Args:
        frame (np.ndarray): BGR color image from the webcam or video.
        frame_count (int): Current frame index (increment each loop).
        skip_interval (int): How many frames to skip before re-running Haar detection.
        face_cascade (cv2.CascadeClassifier): Pre-loaded Haar face detector.
        face_tracker (cv2.legacy.Tracker or cv2.Tracker): Current face tracker, or None if not used yet.
        face_bbox (tuple or None): Current bounding box for the face (x, y, w, h), or None.

    Returns:
        (face_tracker, face_bbox): Updated tracker and bounding box.
    """
    # Decide whether to run face detection on this frame
    run_detection = (frame_count % skip_interval == 0) or (face_tracker is None) or update_flag
    face_temp = {}

    if run_detection:
        # Convert to gray for faster/more typical Haar detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=constants.SCALE_FACTOR, minNeighbors=5)

        if len(faces) > 0:
            # Pick the first face in the list
            face_bbox = faces[0]  # (x, y, w, h)

            face_temp = initialize_face_template(frame, face_bbox)

            # Create/recreate the tracker
            face_tracker = cv2.legacy.TrackerKCF_create()
            face_tracker.init(frame, tuple(face_bbox))
        else:
            # No face found
            face_tracker = None
            face_bbox = None
    else:
        # We have an existing tracker -> update it
        if face_tracker is not None:
            success, bbox = face_tracker.update(frame)
            if success:
                # Tracker succeeded
                face_bbox = tuple(map(int, bbox))
            else:
                # Tracker failed to locate the face
                face_tracker = None
                face_bbox = None

    return face_tracker, face_bbox, face_temp


def compute_orientation_difference_map(template, face_roi_gray):
    """
    Compute the edge orientation map for the current face ROI and compare it
    with the stored template orientation. Non-edge pixels (via Canny) are ignored.
    """
    current_orient, current_edges = utility_functions.compute_edge_orientation(face_roi_gray)
    template_orient = template['orientation']
    # Compute per-pixel orientation difference (only where edges exist)
    diff_map = utility_functions.angle_difference(current_orient, template_orient)
    diff_map = diff_map * (current_edges > 0).astype(np.float32)
    return diff_map


def track_face(template, frame_bgr,
               search_range=15,
               angle_range=10,
               angle_step=2):
    """
    Search over small translations and rotations to align the current face region with the template.
    Uses the edge orientation maps to compute the alignment error.

    Returns (best_x, best_y, best_angle) that minimizes the orientation difference.
    """
    (x0, y0, w, h) = template['bbox']
    template_orient = template['orientation']

    best_score = float('inf')
    best_params = (x0, y0, 0)

    angle_min = -angle_range * np.pi / 180.0
    angle_max = angle_range * np.pi / 180.0
    angle_step_rad = angle_step * np.pi / 180.0

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    for dy in range(-search_range, search_range + 1, 2):
        for dx in range(-search_range, search_range + 1, 2):
            for angle in np.arange(angle_min, angle_max + 0.0001, angle_step_rad):
                x_new = x0 + dx
                y_new = y0 + dy

                H, W = frame_gray.shape[:2]
                if (x_new < 0 or y_new < 0 or x_new + w >= W or y_new + h >= H):
                    continue

                face_patch_gray = frame_gray[y_new:y_new + h, x_new:x_new + w]

                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle * 180 / np.pi, 1.0)
                face_patch_rot = cv2.warpAffine(face_patch_gray, M, (w, h),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)

                # Compute Canny-based edge orientation for the rotated patch
                orient_patch, _ = utility_functions.compute_edge_orientation(face_patch_rot)

                diff_map = utility_functions.angle_difference(orient_patch, template_orient)
                score = np.sum(diff_map)  # sum or mean difference

                if score < best_score:
                    best_score = score
                    best_params = (x_new, y_new, angle)

    return best_params


def compute_color_difference_map(template, face_roi_bgr):
    """
    Compute a simple per-pixel color difference (in BGR) between the current face ROI
    and the stored template color patch.
    """
    template_bgr = template['color']
    diff = cv2.absdiff(face_roi_bgr, template_bgr)
    color_diff_map = diff.sum(axis=2).astype(np.float32)
    color_diff_map = color_diff_map * (255.0 / 765.0)   # Scale the difference to the range 0-255
    return color_diff_map


def combine_edges_and_color(orient_diff, color_diff,
                            low_thresh=120, high_thresh=200,
                            min_area=150):
    """
    Combine the orientation difference and color difference maps using hysteresis thresholding.
    The orientation difference is first normalized (assuming maximum π), then the two cues are fused.
    Hysteresis thresholding is applied on the combined score, and then only the largest connected component is retained.

    Parameters:
      - orient_diff: Orientation difference map (in radians)
      - color_diff: Color difference map (e.g., sum of absolute differences in BGR)
      - low_thresh: Lower threshold for hysteresis (0-255 scale)
      - high_thresh: Higher threshold for hysteresis (0-255 scale)
      - min_area: Minimum area for a valid connected region

    Returns:
      A binary (0/255) mask for the occluding hand region.
    """
    # Normalize orient_diff: assume max possible difference is π, map to 0-255.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    orient_diff = (orient_diff / (np.pi/2)) * 255
    orient_norm = np.clip(orient_diff, 0, 255)
    orient_norm = orient_norm.astype(np.uint8)
    # orient_norm = cv2.morphologyEx(orient_norm, cv2.MORPH_CLOSE, kernel)

    # Assume color_diff is already on a 0-255 scale; if not, scale appropriately.
    color_norm = np.clip(color_diff, 0, 255).astype(np.uint8)
    # orient_norm = np.zeros_like(color_norm)
    # color_norm = np.zeros_like(orient_norm)

    # Fuse the cues. Here we use the maximum value at each pixel.
    combined_score = np.maximum(orient_norm, color_norm)
    # idx_max_mat = ((color_norm >= orient_norm).astype(int) * 255).astype(np.uint8)
    # cv2.imshow("idx_max", idx_max_mat)

    # Apply hysteresis thresholding on the combined score.
    hyst_mask = hysteresis_thresholding(combined_score, low_thresh, high_thresh)

    # Optionally perform a morphological closing to fill small gaps.
    closed = cv2.morphologyEx(hyst_mask, cv2.MORPH_CLOSE, kernel)

    # Use connected component analysis to keep only the largest region.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    max_area = 0
    best_label = 0
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            best_label = label_id
    if max_area < min_area:
        final_mask = np.zeros_like(closed)
    else:
        final_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)

    return final_mask, max_area


def hysteresis_thresholding(img, low_thresh, high_thresh):
    """
    Perform hysteresis thresholding on a single-channel image.
    Pixels with value >= high_thresh are considered strong.
    Pixels between low_thresh and high_thresh are weak.
    Weak pixels are kept if connected to strong pixels.
    """
    # Create masks for strong and weak pixels
    strong = (img >= high_thresh).astype(np.uint8)
    weak = ((img >= low_thresh) & (img < high_thresh)).astype(np.uint8)

    # Initialize the final edge map with strong edges
    final = strong.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    changed = True
    while changed:
        prev = final.copy()
        # Dilate final edges
        dilated = cv2.dilate(final, kernel)
        # If a weak pixel is adjacent (in the dilated image) to a strong pixel, keep it
        final = np.where((weak == 1) & (dilated == 1), 1, final)
        changed = not np.array_equal(prev, final)
    return (final * 255).astype(np.uint8)
