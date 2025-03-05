import numpy as np
import cv2
import joblib
from DrawSomething import constants, segmentation, track, face
from DrawSomething.online_model import OnlineModel
from DrawSomething.offline_model import OfflineModel
from DrawSomething.utils import bg_and_motion
from DrawSomething.gesture_recognition import SvmModel, GestureStabilizer
ONLINE_THRESHOLD = 1.5
OFFLINE_THRESHOLD = 0.4
VIDEO_SOURCE = 0

def main_loop_wrapper():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + constants.CASCADE_FACE_DETECTOR)
    face_mask, frame = face.get_initial_face_mask(VIDEO_SOURCE, face_cascade)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=8, detectShadows=True)

    # Create Online and Offline models:
    online_model = OnlineModel(ONLINE_THRESHOLD, face_mask, frame)
    offline_model = OfflineModel(OFFLINE_THRESHOLD)
    svm_model = SvmModel(constants.SVM_MODEL_PATH)
    stabilizer = GestureStabilizer(constants.WIN_SIZE, constants.MIN_CHANGE_FRAME)
    # cap = cv2.VideoCapture(VIDEO_SOURCE)

    # Parameters used in main loop:
    frame_counter = 0
    face_tracker = None
    gray_face_buffer = [None]
    face_mask_extended = face_mask

    def main_loop(frame):
        """
        TODO: Update doc here...
        """
        # Init...
        nonlocal face_cascade, face_mask, bg_subtractor, online_model, offline_model, svm_model, stabilizer, cap, \
            frame_counter, face_tracker, face_mask_extended
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Face detection
        face_tracker, face_bbox = track.manage_face_detection_and_tracking(frame, face_cascade, face_tracker)
        if face_bbox is not None:
            face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            face_mask_extended = np.zeros(frame.shape[:2], dtype=np.uint8)
            x, y, w, h = face_bbox
            face_mask[y:y + h, x:x + w] = 1
            h_new = np.int32(h * 1.5)
            w_new = np.int32(w * 1.5)
            face_mask_extended[y:y + h_new, x:x + w_new] = 255

        # Generate Hybrid mask
        mask_online, ratio_map_online = online_model.compute_online_mask(frame_hsv)
        mask_offline, p_skin_offline = offline_model.compute_offline_mask(frame)
        hybrid_mask = cv2.bitwise_and(mask_online, mask_offline)
        hybrid_mask = cv2.medianBlur(hybrid_mask, 3)

        updated_color_mask, current_face_gray = segmentation.segment_hand_with_face_overlap(
            frame_bgr=frame,
            final_skin_mask=hybrid_mask,
            face_bbox=face_bbox,
            face_buffer=gray_face_buffer,
            movement_threshold=8,  # tune
            min_motion_area=50  # tune
        )
        gray_face_buffer[frame_counter % len(gray_face_buffer)] = current_face_gray
        updated_color_mask = cv2.medianBlur(updated_color_mask, 5)
        skin_mask = segmentation.largest_contour_segmentation(updated_color_mask)

        # Probability map and motion filters:
        # ___________________________________
        fg_mask = bg_subtractor.apply(frame)  # , learningRate=5e-3)
        # Optionally smooth the fg_mask:
        # fg_mask = cv2.medianBlur(fg_mask, 5)
        # Use a high threshold to keep only strong motion:
        motion_mask_high = bg_and_motion.get_high_motion_mask(fg_mask, high_thresh=70)  # 30
        motion_prob = motion_mask_high.astype(np.float32)
        fused_mask_temp, combined_prob = bg_and_motion.fuse_color_motion(ratio_map_online, p_skin_offline, motion_prob,
                                                                         w_color=0.22, w_motion=0.9, threshold=0.75)
        mask1 = cv2.bitwise_and(fused_mask_temp, hybrid_mask)  # motion_mask_high
        # mask1 = cv2.medianBlur(mask1, 5)
        fused_mask_temp = cv2.medianBlur(fused_mask_temp, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused_mask_temp = cv2.morphologyEx(fused_mask_temp, cv2.MORPH_CLOSE, kernel)
        hybrid_mask_copy = hybrid_mask.copy()
        hybrid_mask_copy[face_mask_extended == 255] = 0
        fused_mask = cv2.bitwise_or(mask1, hybrid_mask_copy)
        hand_mask_fused = segmentation.largest_contour_segmentation(fused_mask_temp)
        # Update online model
        non_skin_mask = cv2.bitwise_not(hybrid_mask)
        online_model.update(frame_hsv, skin_mask, non_skin_mask)

        # ___________________________________
        prediction = svm_model.predict(hand_mask_fused)
        # Update the stabilizer with the new prediction
        stable_label = stabilizer.update(prediction)
        # Display the stable label on the frame
        display_frame = frame.copy()

        return hand_mask_fused, stable_label

        # cv2.putText(display_frame, f"Gesture: {stable_label}", (10, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # cv2.imshow("Gesture Prediction", display_frame)

                # Display
        # cv2.imshow("Original", frame)
        # cv2.imshow("Offline mask", mask_offline)
        # cv2.imshow("Online mask", mask_online)
        # cv2.imshow("Hybrid mask", hybrid_mask)
        # cv2.imshow("combined", hand_mask_fused)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()
    return main_loop

#
# if __name__ == '__main__':
#     main_loop()
