import cv2
import numpy as np
import time
from DrawSomething import segmentation

# List of gestures and number of frames per gesture
gestures = ['index_finger', 'up_thumb', 'open_hand', 'close_hand', 'three_fingers', 'nonsense']
num_frames_per_gesture = 700
# Open the webcam
cap = cv2.VideoCapture(0)
segmentation_handler = segmentation.HandSegmentation(cap)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Loop over each gesture
for gesture in gestures:
    print(f"\nPrepare for gesture: {gesture}")
    print("Move out of the frame to allow background stabilization...")
    time.sleep(3)  # pause to let user clear the FOV
    print(f"Now, place your hand for '{gesture}' and get ready!")
    time.sleep(2)  # additional pause before starting recording

    all_contours = []
    frame_counter = 0

    while frame_counter < num_frames_per_gesture:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed, exiting...")
            break

        frame = cv2.flip(frame, 1)
        frame_counter += 1

        # Process the frame using your segmentation method
        # Assume segmentation_handler.proc_frame returns the following masks:
        # (hybrid_mask, motion_mask, fg_mask, final_mask, final_hand_mask)
        hybrid_mask, motion_mask, fg_mask, final_mask, final_hand_mask = segmentation_handler.proc_frame(frame)

        # If a valid hand mask is found and we have passed an initial frame warm-up
        if frame_counter >= 100:
            # Here final_hand_mask is expected to be a NumPy array (or convert if needed)
            all_contours.append(final_hand_mask)

        # Optionally display the frame and mask for debugging
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Convert the list of masks to a NumPy array and save to a file with the gesture name
    gesture_data = np.array(all_contours)
    np.save(f"data/recordings/{gesture}.npy", gesture_data)
    print(f"Saved {gesture_data.shape[0]} frames for gesture: '{gesture}' to {gesture}.npy")

    # Pause before moving on to the next gesture, unless it's the last one
    if gesture != gestures[-1]:
        print("Recording complete for this gesture. Please move out of the frame.")
        time.sleep(5)  # pause to let the user reposition
        print("Get ready for the next gesture...")
        time.sleep(2)

print("All gestures recorded. Exiting...")
cap.release()
cv2.destroyAllWindows()
