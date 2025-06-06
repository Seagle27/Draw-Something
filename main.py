import cv2
from DrawSomething import constants, track, face
from DrawSomething.gesture_recognition import SvmModel, GestureStabilizer
from DrawSomething.segmentation import HandSegmentation
from DrawSomething.constants import *
from DrawSomething import detect_fingertip


def equalize_frame(frame):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the L channel in LAB color space and return the equalized BGR frame.
    """
    # Convert frame from BGR to LAB
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # v_eq = np.where(v > 240, 127 , v)
    # cv2.imshow("s", s)
    # Create CLAHE object (tune clipLimit and tileGridSize if necessary)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    # Merge equalized L with original a and b channels
    hsv_eq = cv2.merge((h, s, v_eq))
    # Convert back to BGR
    frame_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return frame_eq


def main_loop():
    """
    TODO: Update doc here...
    """
    # Init...
    # bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=8, detectShadows=True)
    # bgfg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    # Create Online and Offline models:

    svm_model = SvmModel(constants.SVM_MODEL_PATH)
    stabilizer = GestureStabilizer(constants.WIN_SIZE, constants.MIN_CHANGE_FRAME)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    segmentation_handler = HandSegmentation(cap)


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        final_hand_mask, motion_mask, hybrid_mask, fg_mask = segmentation_handler.proc_frame(frame)

        # Display
        cv2.imshow("Original", frame)
        cv2.imshow("Hand mask", final_hand_mask)
        cv2.imshow("Hybrid mask", hybrid_mask)
        # # cv2.imshow("Final mask", final_mask)
        cv2.imshow("Motion mask", motion_mask)
        # Display
        cv2.imshow("fg_mask", fg_mask)
        # cv2.imshow("combined", combined_mask)
        # cv2.imshow("motion", fused_mask_temp)
        # cv2.imshow("fg frame", fg_frame)

        # ___________________________________
        prediction = svm_model.predict(final_hand_mask)
        # # Update the stabilizer with the new prediction
        stable_label = stabilizer.update(prediction)
        # # Display the stable label on the frame
        display_frame = frame.copy()
        curr_fingertip = detect_fingertip.detect_fingertip(final_hand_mask)
        pos_x, pos_y = curr_fingertip[0], curr_fingertip[1]
        cv2.circle(final_hand_mask, (int(pos_x), int(pos_y)), 10, (255, 255, 255))

        # cv2.putText(display_frame, f"Gesture: {stable_label}", (10, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # cv2.imshow("Gesture Prediction", display_frame)
        cv2.imshow("Hand mask", final_hand_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_loop()
