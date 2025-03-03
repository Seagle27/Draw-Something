import numpy as np
import cv2
import joblib

from DrawSomething import online_model, offline_model, constants, segmentation, track
import time


def main_hybrid(skin_gmm, non_skin_gmm, video_source=0,
                threshold_gmm=0.4, threshold_hist=1.5, save_hand_mask_video=False):
    """
    Hybrid approach:
    1) Initialize online hist from first frame face detection
    2) For each new frame:
       a) Face detect => update s_new
       b) Build ratio LUT => produce mask_online
       c) Produce mask_offline from GMM => combine => hybrid_mask
       d) Update non-skin hist => blend with old
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + constants.CASCADE_FACE_DETECTOR)
    offline_prob_lut = offline_model.build_rg_probability_lut(skin_gmm, non_skin_gmm, 256)
    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

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
        frame = cv2.flip(frame, 1)

        if not ret:
            print("No frame for initialization!")
            exit(1)

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
    frame_counter = 0
    face_tracker = None
    face_bbox = None
    gray_face_buffer = [None]

    # Video writer setup
    if save_hand_mask_video:
 
        # Get video properties
        frame_width = int(cap.get(3))  # Width
        frame_height = int(cap.get(4)) # Height
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default FPS

        output_source = video_source.split('.')[0] + "_mask.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(output_source, fourcc, fps, (frame_width, frame_height))

    while True:
        frame_counter = (frame_counter + 1) % 1000
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Face detection
        face_tracker, face_bbox = track.manage_face_detection_and_tracking(frame, frame_counter, 50, face_cascade,
                                                                           face_tracker, face_bbox)
        face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_mask[y:y + h, x:x + w] = 1

        # Build ratio LUT => produce mask_online
        ratio_lut = online_model.build_ratio_lut(s_old, n_old)
        mask_online = online_model.compute_online_mask(frame_hsv, ratio_lut, threshold_hist=threshold_hist)

        # Get offline mask from GMM
        mask_offline = offline_model.compute_offline_mask(frame, offline_prob_lut, threshold=threshold_gmm)
        # Combine with AND
        hybrid_mask = cv2.bitwise_and(mask_online, mask_offline)
        # hybrid_mask = cv2.bitwise_and(hybrid_mask, fg_frame)
        hybrid_mask = cv2.medianBlur(hybrid_mask, 3)

        # Hand segmentation:
        updated_mask, current_face_gray = segmentation.segment_hand_with_face_overlap(
            frame_bgr=frame,
            final_skin_mask=hybrid_mask,
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

        # Save to video
        if save_hand_mask_video:
            if hand_mask is not None and hand_mask.size > 0:
                hand_mask = np.clip(hand_mask, 0, 255).astype(np.uint8)
                hand_mask_bgr = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
                out.write(hand_mask_bgr)
        s_old, n_old = online_model.update_histograms(frame_hsv, hand_mask, 1 - hybrid_mask, s_old, n_old)

        # Display
        cv2.imshow("Original", frame)
        cv2.imshow("Offline mask", mask_offline)
        cv2.imshow("Online mask", mask_online)
        # cv2.imshow("fg frame", fg_frame)
        cv2.imshow("Hybrid mask", hybrid_mask)
        cv2.imshow("Updated mask", updated_mask)
        cv2.imshow("hand mask", hand_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_hand_mask_video:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    skin_gmm_model = joblib.load(constants.SKIN_GMM)
    non_skin_gmm_model = joblib.load(constants.NON_SKIN_GMM)
    video_source = "records/closed_hand.mp4"
    save_hand_mask_video = True
    main_hybrid(skin_gmm_model, non_skin_gmm_model, video_source=video_source, threshold_gmm=0.4,
                threshold_hist=1.4, save_hand_mask_video=save_hand_mask_video)
