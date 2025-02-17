import cv2
import numpy as np

def detect_fingertip(mask):
    """
    Detects the single highest fingertip from a binary hand mask.

    Parameters:
        mask (numpy array): Binary image where the hand is white (255) and background is black (0).

    Returns:
        fingertip (tuple): The (x, y) coordinate of the highest fingertip.
        edge_image (numpy array): Image with detected finger edges.
    """
    
    # Find contours of the hand
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None  # No hand detected

    # Get the largest contour (assuming it's the hand)
    hand_contour = max(contours, key=cv2.contourArea)

    # Compute convex hull (outer boundary of hand)
    hull = cv2.convexHull(hand_contour)

    # Find the highest point (lowest y-coordinate) in the convex hull
    fingertip = min(hull[:, 0, :], key=lambda p: p[1])  # Min Y value
    fingertip = (int(fingertip[0]), int(fingertip[1]))

    return fingertip


if __name__ == '__main__':
    # Open a connection to the default camera
    cap = cv2.VideoCapture(0)
    
    # Create the background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=90, detectShadows=True)

    bg_subtractor.setShadowThreshold(0.7)  # Lower value = stricter shadow detection (detects fewer shadows)
    
    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    while True:
        ret, frame = cap.read()
        H, W = frame.shape[:2]
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction with the chosen learning rate
        mask = bg_subtractor.apply(gray, learningRate=0.0) 

        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
        fingertip = detect_fingertip(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        cv2.circle(mask, fingertip, 5, (0, 255, 0), -1)

        cv2.imshow("Finger Tips", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    