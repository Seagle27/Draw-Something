import numpy as np
import cv2
from functools import partial
import joblib

from DrawSomething import online_model, offline_model, constants, segmentation, track
import time


def main_hybrid(skin_gmm, non_skin_gmm, video_source=0,
                threshold_gmm=0.4, threshold_hist=1.5):
    """
    Hybrid approach:
    1) Initialize online hist from first frame face detection
    2) For each new frame:
       a) Face detect => update s_new
       b) Build ratio LUT => produce mask_online
       c) Produce mask_offline from GMM => combine => final_mask
       d) Update non-skin hist => blend with old
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + constants.CASCADE_FACE_DETECTOR)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Unable to open video source:", video_source)
        return

    # Initialize online histograms as zeros
    skin_hist = np.zeros((constants.H_BINS, constants.S_BINS, constants.V_BINS), dtype=np.float32)
    non_hist = np.zeros((constants.H_BINS, constants.S_BINS, constants.V_BINS), dtype=np.float32)

    discovered_face = False
    while not discovered_face:
        # Step A: Use first frame to prime hist
        ret, frame = cap.read()
        if not ret:
            print("No frame for initialization!")
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        H, W = frame.shape[:2]
        face_mask_init = np.zeros((H, W), dtype=np.uint8)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=constants.SCALE_FACTOR, minNeighbors=5,
                                              minSize=(50, 50))
        for (x, y, w, h) in faces:
            face_mask_init[y:y + h, x:x + w] = 1
            discovered_face = True

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Build skin / non-skin hist from the first frame
    skin_hist = online_model.hist_update_vectorized(skin_hist, frame_hsv, face_mask_init)
    non_hist = online_model.hist_update_vectorized(non_hist, frame_hsv, 1 - face_mask_init)

    s_old = skin_hist.copy()
    n_old = non_hist.copy()
    time_lst = []
    frame_counter = 0
    face_tracker = None
    face_bbox = None
    gray_face_buffer = [None, None]

    while True:
        frame_counter = (frame_counter + 1) % 1000
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Face detection
        face_tracker, face_bbox = track.manage_face_detection_and_tracking(frame, frame_counter, 250, face_cascade,
                                                                           face_tracker, face_bbox)
        face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_mask[y:y + h, x:x + w] = 1
        else:
            print(1)

        # 1) Build new accumulators
        s_new = np.zeros_like(s_old)
        n_new = np.zeros_like(n_old)

        # 2) Update s_new with face region
        s_new = online_model.hist_update_vectorized(s_new, frame_hsv, face_mask)

        # 3) Build ratio LUT => produce mask_online
        ratio_lut = online_model.build_ratio_lut(s_old, n_old)
        mask_online = online_model.compute_online_mask(frame_hsv, ratio_lut, threshold_hist=threshold_hist)

        # 4) Get mask_offline from GMM
        mask_offline = offline_model.compute_offline_mask(frame, skin_gmm, non_skin_gmm, threshold=threshold_gmm)

        # Combine with AND
        final_mask = cv2.bitwise_and(mask_online, mask_offline)
        # final_mask = cv2.medianBlur(final_mask, 3)

        # 5) Update non-skin with newly classified non-skin
        # Convert final_mask to boolean
        final_bool = (final_mask > 0)
        non_skin_bool = ~final_bool
        # But also exclude the face region from updating non-skin:
        non_skin_bool[face_mask > 0] = False

        # Make an 8-bit mask for hist_update
        non_skin_mask = non_skin_bool.astype(np.uint8)
        n_new = online_model.hist_update_vectorized(n_new, frame_hsv, non_skin_mask)

        # 6) Exponential smoothing: 90/10
        s_old = 0.9 * s_old + 0.1 * s_new
        n_old = 0.9 * n_old + 0.1 * n_new

        # 7) Hand segmentation:
        updated_mask, current_face_gray = segmentation.segment_hand_with_face_overlap(
            frame_bgr=frame,
            final_skin_mask=final_mask,
            face_bbox=face_bbox,
            face_buffer=gray_face_buffer,
            movement_threshold=8,  # tune
            min_motion_area=50  # tune
        )

        # x, y, w, h = map(int, face_bbox)

        # update prev_face_gray
        gray_face_buffer[frame_counter % len(gray_face_buffer)] = current_face_gray
        updated_mask = cv2.medianBlur(updated_mask, 3)
        hand_mask = segmentation.largest_contour_segmentation(updated_mask)

        # Display
        cv2.imshow("Original", frame)
        # cv2.imshow("Offline mask", mask_offline)
        # cv2.imshow("Online mask", mask_online)
        cv2.imshow("Hybrid mask", final_mask)
        cv2.imshow("Updated mask", updated_mask)
        cv2.imshow("hand mask", hand_mask)
        end = time.time()
        iter_time = end - start
        time_lst.append(iter_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average iteration time {np.mean(time_lst)}")


if __name__ == '__main__':
    skin_gmm_model = joblib.load(constants.SKIN_GMM)
    non_skin_gmm_model = joblib.load(constants.NON_SKIN_GMM)
    main_hybrid(skin_gmm_model, non_skin_gmm_model, video_source=0, threshold_gmm=0.4,
                threshold_hist=1.4)
