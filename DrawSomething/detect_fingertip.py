import cv2
import numpy as np

from classifier import extract_hog_features, train

# Threshold for detecting a "jump" (in pixels)
JUMP_THRESHOLD = 50  # Adjust this value as needed
STABILITY_FRAMES = 30  # Number of frames needed to accept a new position
EMA_ALPHA_MIN = 0.25  # Minimum smoothing factor (very smooth, slow response)
EMA_ALPHA_MAX = 0.9  # Maximum smoothing factor (fast response)
SPEED_THRESHOLD_LOW = 10   # Movement speed below this uses max smoothing
SPEED_THRESHOLD_HIGH = 30  # Movement speed above this uses min smoothing

def detect_fingertip(mask):
    """Detects the most probable fingertip from a binary hand mask using convex hull."""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # No hand detected

    # Get the largest contour (assuming it's the hand)
    hand_contour = max(contours, key=cv2.contourArea)

    # Compute convex hull (this ensures a smooth outer boundary)
    hull = cv2.convexHull(hand_contour)

    # Compute hand center (palm center)
    M = cv2.moments(hand_contour)
    if M["m00"] == 0:
        return None  # Avoid division by zero
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Identify potential fingertips: Convex hull points that are FAR from the palm center
    max_distance = 0
    fingertip = None

    for point in hull[:, 0]:  # Iterate over hull points
        x, y = point
        if y > cy: continue
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Condition: Fingertip is a point with the **maximum distance from the palm center**
        if distance > max_distance:
            max_distance = distance
            fingertip = (x, y)

    return fingertip  # Returns (x, y) coordinates of the detected fingertip


def is_valid_fingertip(new_tip, prev_tip, threshold=JUMP_THRESHOLD):
    """Checks if the new fingertip is within a reasonable distance of the previous one."""
    if prev_tip is None:
        return True  # Always accept the first detected point
    distance = np.linalg.norm(np.array(new_tip) - np.array(prev_tip))
    return distance < threshold  # Accept only if within threshold

def dynamic_ema_alpha(speed):
    """Adjusts EMA_ALPHA dynamically based on movement speed."""
    if speed < SPEED_THRESHOLD_LOW:
        return EMA_ALPHA_MIN  # Very smooth for slow movement
    elif speed > SPEED_THRESHOLD_HIGH:
        return EMA_ALPHA_MAX  # Fast response for rapid movement
    else:
        # Linearly interpolate between min and max alpha
        return EMA_ALPHA_MIN + (EMA_ALPHA_MAX - EMA_ALPHA_MIN) * ((speed - SPEED_THRESHOLD_LOW) / (SPEED_THRESHOLD_HIGH - SPEED_THRESHOLD_LOW))

def smooth_fingertip(curr_tip, prev_tip):
    """Smooths fingertip position using an Exponential Moving Average (EMA)."""
    speed = np.linalg.norm(np.array(prev_tip) - np.array(curr_tip))
    if prev_tip is None:
        return curr_tip
    
    alpha = dynamic_ema_alpha(speed)
    return (
        int(alpha * curr_tip[0] + (1 - alpha) * prev_tip[0]),
        int(alpha * curr_tip[1] + (1 - alpha) * prev_tip[1])
    )


def stable_detect_fingertip(mask):
    detected_fingertip = detect_fingertip(mask)

    if detected_fingertip:
        if is_valid_fingertip(detected_fingertip, prev_fingertip):
            # If it's a small movement, accept immediately
            curr_fingertip = smooth_fingertip(detected_fingertip,
                                              prev_fingertip) if prev_fingertip else detected_fingertip
            stability_counter = 0
            if prev_fingertip:
                cv2.line(canvas, curr_fingertip, prev_fingertip, (0, 255, 0), thickness=4, lineType=cv2.LINE_AA,
                         shift=0)

        else:
            stability_counter += 1  # Continue tracking if it's stable
            cv2.putText(frame, "Big Jump", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if stability_counter >= STABILITY_FRAMES:
                curr_fingertip = detected_fingertip  # Confirm new position
                stability_counter = 0

    prev_fingertip = curr_fingertip


def test_video(video_source, model):
    cap = cv2.VideoCapture(video_source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('fingertip_video.mp4', fourcc, fps, (frame_width, frame_height))

    prev_fingertip = None
    stability_counter = 0  # Counts how many frames the candidate has been stable
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Blank black canvas

    i = 0
    while True:
        i += 1
        ret, frame = cap.read()
        if not ret: break
        # if i < 100: continue
        if i < 130 or (i>300 and i<530) or (i>600 and i<700) or (i>750 and i<840) or (i>900 and i<1050) or (i>1200 and i<2000): continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features from hand mask
        hog_features = extract_hog_features(frame).reshape(1, -1)  # Reshape for SVM

        # Predict gesture
        prediction = model.predict(hog_features)[0]

        # Display prediction on frame
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, prediction, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if prediction == "point":
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_fingertip = detect_fingertip(mask)

            if detected_fingertip:
                if is_valid_fingertip(detected_fingertip, prev_fingertip):
                    # If it's a small movement, accept immediately
                    curr_fingertip = smooth_fingertip(detected_fingertip, prev_fingertip) if prev_fingertip else detected_fingertip
                    stability_counter = 0
                    if prev_fingertip:
                        cv2.line(canvas, curr_fingertip, prev_fingertip, (0, 255, 0), thickness=4, lineType=cv2.LINE_AA, shift=0)

                else:
                    stability_counter += 1  # Continue tracking if it's stable
                    cv2.putText(frame, "Big Jump", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if stability_counter >= STABILITY_FRAMES:
                        curr_fingertip = detected_fingertip  # Confirm new position
                        stability_counter = 0
            
            prev_fingertip = curr_fingertip

            # Draw the fingertip only if we have a valid detection
            cv2.circle(frame, curr_fingertip, 4, (0, 255, 0), -1)
            cv2.circle(frame, detected_fingertip, 3, (0, 0, 255), -1)
        cv2.putText(frame, "EMA Tip", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Detected Tip", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Fingertip Points on Blank Canvas", canvas)
        cv2.imshow("Live Finger Detection", frame)
        out.write(frame)
        if cv2.waitKey(int(2000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_source = "records/test_mask.mp4"
    gesture_paths = "data/HandGestures"
    model = train(gesture_paths)    
    test_video(video_source, model)


