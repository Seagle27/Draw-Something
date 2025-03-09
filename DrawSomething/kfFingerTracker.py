import numpy as np
from filterpy.kalman import KalmanFilter
from DrawSomething import constants, fingertip_detection

class kfFingerTracker:
    def __init__(self):
        # Initialize your Kalman filter when the detector is created.
        self.kf = self.initialize_kalman_filter()
        self.prev_fingertip = None
        self.curr_fingertip = None
        self.fingertip_history = []
        self.stability_counter = 0


    def initialize_kalman_filter(self):
        """
        Initializes a Kalman filter with:
          - state vector: [x, y, dx, dy]
          - state transition matrix F assuming a constant velocity model
          - measurement matrix H (we only measure [x, y])
          - Q: process noise covariance
          - R: measurement noise covariance
          - P: error covariance matrix
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)
        # Initial state: position (0, 0) and zero velocity.
        kf.x = np.array([0, 0, 0, 0], dtype=np.float32)

        # State transition matrix: Predict next state (constant velocity model)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)

        # Measurement matrix: we can only measure x and y
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance matrix: tune these values to your system
        kf.Q = np.eye(4, dtype=np.float32) * 0.01

        # Measurement noise covariance matrix: tune based on sensor noise
        kf.R = np.eye(2, dtype=np.float32) * 0.1

        # Error covariance matrix: initial uncertainty
        kf.P = np.eye(4, dtype=np.float32) * 1.0

        return kf

    def stable_detect_fingertip(self, mask, history_size=constants.HISTORY_MAX_LENGTH):
        # Preprocess the mask to remove noise
        cleaned_mask = fingertip_detection.preprocess_mask(mask)

        # Get the new measurement from the current frame
        measurement = fingertip_detection.detect_fingertip(cleaned_mask)  # Expected to be a tuple (x, y) or None

        # Predict the next state using the Kalman filter
        predicted_state = self.kf.predict()  # self.kf.x is updated with the prediction

        # Decide whether to update the Kalman filter with the new measurement
        if measurement is not None:
            if fingertip_detection.is_valid_fingertip(measurement, self.prev_fingertip):
                self.kf.update(np.array(measurement, dtype=np.float32))
                self.stability_counter = 0
            else:
                self.stability_counter += 1
                if self.stability_counter < constants.STABILITY_FRAMES:
                    # Do not update the filter, rely on the prediction
                    pass
                else:
                    self.kf.update(np.array(measurement, dtype=np.float32))
                    self.stability_counter = 0
        else:
            # No measurement detected; rely on the prediction (no update)
            pass

        # Extract the filtered (smoothed) position from the state vector ([x, y, dx, dy])
        new_tip = self.kf.x[:2]
        new_tip = (int(new_tip[0]), int(new_tip[1]))

        # Maintain a history to further reduce jitter via median filtering
        self.fingertip_history.append(new_tip)
        if len(self.fingertip_history) > history_size:
            self.fingertip_history.pop(0)
        xs = [pt[0] for pt in self.fingertip_history]
        ys = [pt[1] for pt in self.fingertip_history]
        median_tip = (int(np.median(xs)), int(np.median(ys)))

        self.curr_fingertip = median_tip
        self.prev_fingertip = median_tip

        return self.curr_fingertip
