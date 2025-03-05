import collections
from DrawSomething import constants as const
from DrawSomething import classifier as svm
import joblib
import cv2


class GestureStabilizer:
    def __init__(self, window_size=60, min_change_frames=60):
        """
        window_size: number of frames to consider (e.g. 60 frames ~ 2 seconds at 30 fps)
        min_change_frames: minimum number of consecutive frames (or frames window) with a new majority before updating the stable label.
        """
        self.window_size = window_size
        self.min_change_frames = min_change_frames
        self.predictions = []  # sliding window of recent predictions
        self.stable_label = None
        self.frames_since_change = 0  # counter for consecutive frames with new majority

    def update(self, prediction):
        """
        Add a new frame prediction, update the sliding window, and return the current stable label.
        """
        # Append new prediction
        self.predictions.append(prediction)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)

        # Compute the majority (mode) from the sliding window
        counter = collections.Counter(self.predictions)
        majority_label, majority_count = counter.most_common(1)[0]

        # If no stable label yet, initialize it.
        if self.stable_label is None:
            self.stable_label = majority_label
            self.frames_since_change = 0
        else:
            if majority_label != self.stable_label:
                self.frames_since_change += 1
                # If the new majority persists for enough frames, update stable label.
                if self.frames_since_change >= self.min_change_frames:
                    self.stable_label = majority_label
                    self.frames_since_change = 0
            else:
                # If majority equals the current stable label, reset the counter.
                self.frames_since_change = 0

        return self.stable_label


class SvmModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        return

    def extract_features_from_mask(self, hand_mask):
        hog_features = svm.extract_hog_features(hand_mask).reshape(1, -1)
        return hog_features

    def predict(self, hand_mask):
        hog_features = self.extract_features_from_mask(hand_mask)
        gest_prediction = self.get_prediction_from_classifier(hog_features)
        return gest_prediction

    # Initialize video capture, your model, etc.
    def get_prediction_from_classifier(self, hog_features):
        prediction_num = self.model.predict(hog_features)[0]
        file_names = ('index_finger', 'up_thumb', 'open_hand', 'close_hand', 'three_fingers')
        gest_prediction = file_names[prediction_num - 1]
        return gest_prediction
# Placeholder functions for demonstration:



# Example usage in a main loop:
if __name__ == '__main__':
    # Assume 'model' is your trained SVM gesture classifier,
    # and 'extract_features_from_frame()' is a function that extracts the features for classification.
    import cv2
    import numpy as np
    import classifier as svm
    # # hand_mask = segmentation.largest_contour_segmentation(updated_color_mask)
    # hand_mask = segmentation.largest_contour_segmentation(fused_mask_temp)
    # if flag_save_gest:
    #     if hand_mask is not None and (frame_counter > 10):
    #         # Convert from OpenCV contour format to a NumPy array of shape (N, 2)
    #         contour_array = hand_mask  # shape => (N, 2)
    #         all_contours.append(contour_array)
    #     if len(all_contours) == 500:
    #         break
    #
    # s_old, n_old = online_model.update_histograms(frame_hsv, hand_mask, 1 - hybrid_mask, s_old, n_old)



    model = SvmModel(const.SVM_MODEL_PATH)

    # Initialize the stabilizer: use 60 frames window and require 60 consistent frames for change.
    stabilizer = GestureStabilizer(const.WIN_SIZE, const.MIN_CHANGE_FRAME)

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        prediction = model.predict(mask)
        # Update the stabilizer with the new prediction
        stable_label = stabilizer.update(prediction)
        # Display the stable label on the frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Gesture: {stable_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Gesture Prediction", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
