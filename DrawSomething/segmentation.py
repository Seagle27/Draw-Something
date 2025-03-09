import numpy as np
import cv2
import time

from DrawSomething import face
from DrawSomething.offline_model import OfflineModel
from DrawSomething.online_model import OnlineModel
from DrawSomething.constants import *
from DrawSomething.utils import bg_and_motion


class HandSegmentation:
    BUFFER_LEN = 3
    HAND_BUFFER_LEN = 3

    def __init__(self, cap):
        self.background = self.capture_background(cap)
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_FACE_DETECTOR)
        self.face_mask, frame = face.get_initial_face_mask(cap, self._face_cascade)
        self.offline_model = OfflineModel(OFFLINE_THRESHOLD)

        offline_mask, _ = self.offline_model.compute_offline_mask(frame)
        fg = self.background_subtraction(frame)

        non_skin = cv2.bitwise_not(cv2.bitwise_and(fg, offline_mask))
        self.online_model = OnlineModel(ONLINE_THRESHOLD, self.face_mask, non_skin, frame)
        self.face_tracker = None
        self.gray_face_buffer = []
        self.hand_over_face_buffer = []
        self.face_template = {}
        self.last_bbox = None
        self.last_segmentation = None

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    def background_subtraction(self, frame, threshold=10):
        diff = cv2.absdiff(self.background, frame)

        # Convert difference to grayscale
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold the difference to obtain a binary mask
        _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

        return mask

    def proc_frame(self, frame):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = self.background_subtraction(frame)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        face_bbox = self.manage_face_detection_and_tracking(frame)
        if face_bbox is not None:
            face_bbox = self.extend_bbox(face_bbox, frame.shape[:2], extend_ratio=(0.8, 0.3))
            self.last_bbox = face_bbox
            self.face_mask = self.bbox_to_mask(face_bbox, frame.shape[:2])
            face_mask_extended = self.bbox_to_mask(face_bbox, frame.shape[:2], (1.5, 1.5))
            if not self.face_template:
                self.face_template = face.initialize_face_template(frame, face_bbox)
        else:
            face_mask_extended = self.face_mask

        # hybrid_mask, online_mask, offline_mask = self.get_hybrid_mask(frame, frame_hsv)
        hybrid_mask = self.get_hybrid_mask(frame, frame_hsv)
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_CLOSE, kernel)
        hybrid_mask = cv2.bitwise_and(fg_mask, hybrid_mask)
        # contours_count, _ = self.count_large_contours(hybrid_mask, min_area=500)
        # print("contours count: ", contours_count)%
        # new_hybrid_mask = cv2.bitwise_and(hybrid_mask, fg_frame)
        # new_hybrid_mask = cv2.medianBlur(new_hybrid_mask, 3)

        updated_color_mask, current_face_gray = segment_hand_with_face_overlap(
            frame_bgr=frame,
            final_skin_mask=hybrid_mask,
            face_bbox=face_bbox,
            face_buffer=self.gray_face_buffer,
            movement_threshold=8,  # tune
            min_motion_area=50  # tune
        )
        if current_face_gray is not None:
            self.gray_face_buffer.append(current_face_gray)
            if len(self.gray_face_buffer) > self.BUFFER_LEN:
                self.gray_face_buffer.pop(0)
        updated_color_mask = cv2.medianBlur(updated_color_mask, 3)
        skin_mask = self.largest_contour_segmentation(updated_color_mask)

        non_skin_mask = cv2.bitwise_not(hybrid_mask)
        non_skin_mask[face_mask_extended == 255] = 0
        self.online_model.update(frame_hsv, skin_mask, non_skin_mask)
        # combined_mask = self.hand_over_face(frame, skin_mask)
        fg_mask2 = self.bg_subtractor.apply(frame, learningRate=7e-2)
        motion_mask_high = bg_and_motion.get_high_motion_mask(fg_mask2, high_thresh=70)
        motion_mask = cv2.morphologyEx(motion_mask_high, cv2.MORPH_CLOSE, kernel)

        face_area_mask = cv2.bitwise_and(motion_mask, hybrid_mask)  # motion_mask_high
        face_area_mask = cv2.medianBlur(face_area_mask, 5)
        face_area_mask = face_area_mask * (self.face_mask > 0).astype(np.uint8)

        not_face_area_mask = hybrid_mask * (self.face_mask == 0).astype(np.uint8)

        final_mask = cv2.bitwise_or(face_area_mask, not_face_area_mask)
        final_mask = cv2.medianBlur(final_mask, 5)
        final_hand_mask = self.fill_large_holes(final_mask)
        final_hand_mask = self.largest_contour_segmentation(final_hand_mask)
        self.last_segmentation = final_hand_mask
        return final_hand_mask, motion_mask, hybrid_mask

        # Probability map and motion filters:
        # # ___________________________________

        # #
        # # # Optionally smooth the fg_mask:
        # # # fg_mask = cv2.medianBlur(fg_mask, 5)
        # # # Use a high threshold to keep only strong motion:
        # motion_mask_high = bg_and_motion.get_high_motion_mask(fg_mask, high_thresh=70)  # 30
        # motion_prob = motion_mask_high.astype(np.float32)
        # fused_mask_temp, combined_prob = bg_and_motion.fuse_color_motion(ratio_map_online, p_skin_offline, motion_prob,
        #                                                                  w_color=0.22, w_motion=0.9, threshold=0.75)
        # mask1 = cv2.bitwise_and(fused_mask_temp, hybrid_mask)  # motion_mask_high
        # # # mask1 = cv2.medianBlur(mask1, 5)
        # fused_mask_temp = cv2.medianBlur(fused_mask_temp, 5)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # fused_mask_temp = cv2.morphologyEx(fused_mask_temp, cv2.MORPH_CLOSE, kernel)
        # hybrid_mask_copy = hybrid_mask.copy()
        # hybrid_mask_copy[face_mask_extended == 255] = 0
        # fused_mask = cv2.bitwise_or(mask1, hybrid_mask_copy)
        # hand_mask_fused = segmentation.largest_contour_segmentation(fused_mask_temp)
        # # ___________________________________

    def get_hybrid_mask(self, frame, frame_hsv):
        # mask_online, ratio_map_online = self.online_model.compute_online_mask(frame_hsv)
        mask_offline, p_skin_offline = self.offline_model.compute_offline_mask(frame, self.last_segmentation)
        # return cv2.bitwise_and(mask_online, mask_offline), mask_online, mask_offline
        return mask_offline


    @staticmethod
    def capture_background(cap, num_frames=30):
        """
        Capture and compute an average background frame.
        Assumes the scene is static for the first 'num_frames' frames.
        """
        print("Capturing background. Please make sure the background is static")
        time.sleep(2)
        background = None
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Flip horizontally if needed
            # Initialize the background with the first frame
            if background is None:
                background = frame.astype("float")
            else:
                # Accumulate weighted average
                cv2.accumulateWeighted(frame, background, 0.1)
        # Convert the accumulated background to an 8-bit image
        background = cv2.convertScaleAbs(background)
        return background

    def manage_face_detection_and_tracking(self, frame, run_detection=False):
        """
        Detect or track a face in the given frame.

        Args:
            frame (np.ndarray): BGR color image from the webcam or video.
            face_cascade (cv2.CascadeClassifier): Pre-loaded Haar face detector.
            face_tracker (cv2.legacy.Tracker or cv2.Tracker): Current face tracker, or None if not used yet.
            run_detection: Run detection or use tracker

        Returns:
            (face_tracker, face_bbox): Updated tracker and bounding box.
        """
        # Decide whether to run face detection on this frame
        run_detection = self.face_tracker is None or run_detection
        face_bbox = None

        if run_detection:
            # Convert to gray for faster/more typical Haar detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=5)
            if len(faces) > 0:
                # Pick the first face in the list
                face_bbox = faces[0]  # (x, y, w, h)

                # Create/recreate the tracker
                self.face_tracker = cv2.legacy.TrackerKCF_create()
                self.face_tracker.init(frame, tuple(face_bbox))
            else:
                # No face found
                self.face_tracker = None
        else:
            # We have an existing tracker -> update it
            if self.face_tracker is not None:
                success, bbox = self.face_tracker.update(frame)
                if success:
                    # Tracker succeeded
                    face_bbox = tuple(map(int, bbox))
                else:
                    # Tracker failed to locate the face
                    self.face_tracker = None

        return face_bbox

    def hand_over_face(self, frame, hand_mask):
        # Ensure face_mask is in shape (480, 640, 1) for broadcasting
        x, y, w, h = self.mask_to_bbox(self.face_mask)

        # Apply the mask to the frame
        face_roi_bgr = frame[y: y + h, x: x+w]
        orient_diff = face.compute_orientation_difference_map(self.face_template, face_roi_bgr)
        color_diff = face.compute_color_difference_map(self.face_template, face_roi_bgr)
        occluding_hand_mask, max_area = face.combine_edges_and_color(orient_diff, color_diff)

        face_area = np.zeros_like(frame[:, :, 0])
        face_area[y: y + h, x: x+w] = occluding_hand_mask
        hand_mask_local = hand_mask.copy()
        hand_mask_local[y: y + h, x: x+w] = 0

        combined_mask = cv2.bitwise_or(hand_mask, face_area)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)



    @staticmethod
    def bbox_to_mask(bbox, frame_shape, ratio=(1, 1)):
        face_mask = np.zeros(frame_shape, dtype=np.uint8)
        x, y, w, h = bbox
        h = np.int32(h * ratio[0])
        w = np.int32(w * ratio[1])
        face_mask[y:y + h, x:x + w] = 255
        return face_mask

    @staticmethod
    def mask_to_bbox(mask):
        y_indices, x_indices = np.where(mask > 0)

        if y_indices.size > 0 and x_indices.size > 0:  # Ensure face region exists
            # Get the bounding box around the nonzero region
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            y_max += 1
            x_max += 1
            return x_min, y_min, x_max - x_min, y_max - y_min

    @staticmethod
    def extend_bbox(bbox, frame_shape, extend_ratio=(0.3, 0.3)):
        """
        Extend the bounding box downward to include the neck.

        Parameters:
        - bbox: (x, y, w, h) tuple representing the face bounding box.
        - img_height: Height of the image to prevent out-of-bounds errors.
        - extend_ratio: The fraction of height to extend downward.

        Returns:
        - New bounding box (x, y, w, new_h)
        """
        x, y, w, h = bbox
        new_x = int(0.9 * x)
        new_y = int(0.75 * y)
        extra_height = int(h * extend_ratio[0])  # Extend by a percentage of original height
        extra_width = int(h * extend_ratio[1])
        new_h = h + extra_height
        new_w = w + extra_width
        # Ensure the box does not exceed the image height
        new_x = 0 if new_x < 0 else new_x
        new_y = 0 if new_y < 0 else new_y

        if new_y + new_h > frame_shape[0]:
            new_h = frame_shape[0] - new_y

        if new_x + new_w > frame_shape[1]:
            new_w = frame_shape[1] - new_x

        return (new_x, new_y, new_w, new_h)

    @staticmethod
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

    import cv2
    import numpy as np

    @staticmethod
    def count_large_contours(mask, min_area=100):
        """
        Count the number of contours in a binary mask that have an area greater than min_area.

        Parameters:
        - mask: Binary mask (0 and 255) of shape (H, W).
        - min_area: Minimum contour area to be counted.

        Returns:
        - count: Number of contours above the minimum area.
        - valid_contours: The filtered contours list.
        """
        # Ensure the mask is binary (values are 0 or 255)
        mask = (mask > 0).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on the minimum area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        return len(valid_contours), valid_contours
    @staticmethod
    def fill_large_holes(mask):
        """
        Fills large holes inside a binary mask using flood fill.

        Parameters:
        - mask: Binary mask (0 and 255) of shape (H, W).

        Returns:
        - filled_mask: Mask with large holes filled.
        """
        # Copy the mask and convert to 3 channels (needed for flood fill)
        h, w = mask.shape
        filled_mask = mask.copy()
        mask_floodfill = mask.copy()

        # Create a mask for flood fill (2 pixels larger)
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill from point (0,0) - assumes black background
        cv2.floodFill(mask_floodfill, flood_mask, (0, 0), 255)

        # Invert flood-filled mask and combine with the original
        inverted_flood_fill = cv2.bitwise_not(mask_floodfill)
        filled_mask = mask | inverted_flood_fill

        return filled_mask


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
      - face_buffer: grayscale face buffer from the previous frames' face region.
      - movement_threshold: absolute difference threshold for considering a pixel "moving".
      - min_motion_area: minimum # of changed pixels in the face region to consider.

    Returns:
      - updated_mask: an updated final skin mask that re-includes
                      any "moving" region in the face box.
      - current_face_gray: the grayscale patch from this frame, to store for the next loop.
    """
    updated_mask = final_skin_mask.copy()

    if face_bbox is None:
        # No face found or tracker lost => can't do partial face removal
        return updated_mask, None

    # Extract original bounding box and extend downward
    x, y, w, h = face_bbox

    # Ensure bounding box remains within image bounds
    H, W = frame_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)  # Width remains the same
    y2 = min(H, y + h)  # Extended height

    if x1 >= x2 or y1 >= y2:
        print("Warning: Invalid face bounding box.")
        return updated_mask, None

    # Extract the face region in grayscale
    face_region_bgr = frame_bgr[y1:y2, x1:x2]
    face_region_gray = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2GRAY)
    current_face_gray = face_region_gray.copy()

    # If face_buffer exists, compute motion detection
    diffs = []
    for prev in face_buffer:
        if prev.shape == face_region_gray.shape:
            diffs.append(cv2.absdiff(face_region_gray, prev))
        else:
            print(f"Warning: Skipping prev_face_gray due to shape mismatch: {prev.shape}")

    # Compute motion masks
    motion_masks = [((diff > movement_threshold).astype(np.uint8) * 255) for diff in diffs]

    if len(motion_masks) >= 2:
        bitwise_and_result = motion_masks[0]
        for motion_mask in motion_masks[1:]:
            bitwise_and_result = cv2.bitwise_and(bitwise_and_result, motion_mask)

        # Ensure face area mask has the same size as face bounding box region
        face_area_mask = np.where(bitwise_and_result > 0, 255, 0).astype(np.uint8)

        # Check shape alignment before updating mask
        if face_area_mask.shape == (y2 - y1, x2 - x1):
            updated_mask[y1:y2, x1:x2] = face_area_mask
        else:
            print("Error: Shape mismatch when updating mask.")
            print(f"Face Area Mask Shape: {face_area_mask.shape}")
            print(f"Target Region Shape: {(y2 - y1, x2 - x1)}")

    else:
        print("No previous frames detected or shape mismatch prevented motion check.")

    return updated_mask, current_face_gray

