import collections
import classifier as svm
import joblib


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

        if majority_count >= self.window_size // 4:
            self.stable_label = majority_label
            return majority_label
        elif majority_count >= self.window_size // 6 and self.stable_label==majority_label:
            return majority_label
        self.stable_label=None
        return None

        # # If no stable label yet, initialize it.
        # if self.stable_label is None:
        #     self.stable_label = majority_label
        #     self.frames_since_change = 0
        # else:
        #     if majority_label != self.stable_label:
        #         self.frames_since_change += 1
        #         # If the new majority persists for enough frames, update stable label.
        #         if self.frames_since_change >= self.min_change_frames:
        #             self.stable_label = majority_label
        #             self.frames_since_change = 0
        #     else:
        #         # If majority equals the current stable label, reset the counter.
        #         self.frames_since_change = 0

        # return self.stable_label


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
