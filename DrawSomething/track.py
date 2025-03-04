import cv2
import numpy as np
from DrawSomething import constants


def manage_face_detection_and_tracking(
        frame,
        face_cascade,
        face_tracker,
        run_detection=False
):
    """
    Detect or track a face in the given frame.

    Args:
        frame (np.ndarray): BGR color image from the webcam or video.
        face_cascade (cv2.CascadeClassifier): Pre-loaded Haar face detector.
        face_tracker (cv2.legacy.Tracker or cv2.Tracker): Current face tracker, or None if not used yet.
        run_detection: Run detection or use tracker

    Returns:
        (face_tracker, face_bbox): Updated tracker and bounding box.
    """
    # Decide whether to run face detection on this frame
    run_detection = face_tracker is None or run_detection

    if run_detection:
        # Convert to gray for faster/more typical Haar detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=constants.SCALE_FACTOR, minNeighbors=5)

        if len(faces) > 0:
            # Pick the first face in the list
            face_bbox = faces[0]  # (x, y, w, h)

            # Create/recreate the tracker
            face_tracker = cv2.legacy.TrackerKCF_create()
            face_tracker.init(frame, tuple(face_bbox))
        else:
            # No face found
            face_tracker = None
            face_bbox = None
    else:
        # We have an existing tracker -> update it
        if face_tracker is not None:
            success, bbox = face_tracker.update(frame)
            if success:
                # Tracker succeeded
                face_bbox = tuple(map(int, bbox))
            else:
                # Tracker failed to locate the face
                face_tracker = None
                face_bbox = None

    return face_tracker, face_bbox
