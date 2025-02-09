import cv2
import numpy as np


def find_fingertip(fg_mask, frame, H, W):
    # 1. Find largest contour
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None  # No person found
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    if area < 1000:  # or some threshold
        return None, None  # Not a valid shape
    
    # 2. Convex hull and defects
    hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
    if len(hull_indices) < 3:
        return None, None
    
    hull_points = cv2.convexHull(largest_contour, returnPoints=True)
    defects = cv2.convexityDefects(largest_contour, hull_indices)
    
    # 3. Centroid (palm center approx)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    palm_center = (cx, cy)
    
    # 4. Analyze hull + (optionally) defects to find the fingertip
    fingertip_point = None
    max_dist = -1
    
    if defects is not None:
        # Optionally use defects to pick start/end points that are far from center
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(largest_contour[s][0])
            end = tuple(largest_contour[e][0])
            depth = d / 256.0
            
            if depth > 10:  # choose a threshold
                for candidate in [start, end]:
                    c_dist = (candidate[0] - cx)**2 + (candidate[1] - cy)**2
                    if c_dist > max_dist:
                        max_dist = c_dist
                        fingertip_point = candidate
    
    # If no fingertip found from defects, fallback to just the farthest hull point
    # if fingertip_point is None:
    #     for pt in hull_points:
    #         x, y = pt[0]
    #         dist = (x - cx)**2 + (y - cy)**2
    #         if dist > max_dist:
    #             max_dist = dist
    #             fingertip_point = (x, y)
    
    # Draw info
    cv2.drawContours(frame, [largest_contour], -1, (0,255,0), 2)
    cv2.circle(frame, palm_center, 5, (255, 0, 0), -1)  # palm center
    if fingertip_point is not None:
        cv2.circle(frame, fingertip_point, 8, (0, 0, 255), -1)
        cv2.putText(frame, "Fingertip", (fingertip_point[0], fingertip_point[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    return fingertip_point, palm_center

def main():
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
        fg_mask = bg_subtractor.apply(gray, learningRate=0.0) 

        # Perform morphological operations to remove noise and fill holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)   # remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # fill holes

        # fg_mask = cv2.medianBlur(fg_mask, 5)  # Kernel size 5 (adjust if needed)

        # Find external contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Pick the largest contour by area (likely the person)
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 1000:  # filter out very small contours
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(frame, [max_contour], -1, (255, 0, 0), 2)
        # fingertip, palm_center = find_fingertip(fg_mask, frame, H, W)
        # Show the foreground mask (for debugging)
        cv2.imshow('Foreground Mask', fg_mask)
        cv2.imshow('Frame', frame)
        
        # Press 'Esc' to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
