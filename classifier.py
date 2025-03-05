import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def preprocess_and_crop(binary_mask):
    """
    1. Find bounding box around the largest contour (the hand).
    2. Crop the image so only the hand region remains.
    3. Optionally resize to a fixed dimension.
    """
    # Find contours
    # contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Largest contour => hand
    # cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(binary_mask)

    # Crop
    cropped = binary_mask[y:y + h, x:x + w]

    # Resize to a standard size (e.g., 64x64) to keep HOG dimension consistent
    resized = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_LINEAR)

    return resized


def extract_hog_features(mask):
    """
    Extract Histogram of Oriented Gradients (HOG) features from a binary mask shaped (H, W). output shape (N, )
    """
    # Ensure it's truly binary (0 or 255)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Preprocess and crop
    processed_mask = preprocess_and_crop(mask)

    # skimage.feature.hog returns the HOG descriptor as a 1D array by default
    # You can tune orientations, pixels_per_cell, and cells_per_block for best results
    hog_features = hog(processed_mask,
                       orientations=8,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(1, 1),
                       block_norm='L2-Hys',
                       visualize=False)
    return hog_features


def load_gesture_data(file_path, label):
    """
    Load a recording file (saved with np.save) containing an array of masks.
    Extracts multiple features per mask and assigns the provided label.
    """
    masks = np.load(file_path, allow_pickle=True)
    X = []
    y = []
    for mask in masks:
        feature = extract_hog_features(mask)
        X.append(feature)
        y.append(label)
    return X, y


def load_all_gesture_data(base_dir, file_names):
    """
    Assumes that you have 5 files named 'gesture1.npy', 'gesture2.npy', ..., 'gesture5.npy'.
    Each file contains a list/array of masks corresponding to one gesture.
    Returns:
      - X_all: a NumPy array of shape (total_samples, 7)
      - y_all: a NumPy array of labels.
    """
    X_all = []
    y_all = []
    base_dir = "C:/BGU/Git/DrawSomething/data"
    file_names = ('index_finger', 'up_thumb', 'open_hand', 'close_hand', 'three_fingers')
    num_gestures = 5  # Adjust if necessary
    for gesture_label in range(0, num_gestures):
        file_name = f"{file_names[gesture_label]}.npy"
        file_path = os.path.join(base_dir, file_name)
        X, y = load_gesture_data(file_path, label=gesture_label + 1)
        X_all.extend(X)
        y_all.extend(y)
    return np.array(X_all), np.array(y_all)


def train(X, y):
    # Load training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='rbf', C=1, gamma='scale')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return svm_model


def test_frames(gesture_paths, model):
    # Load testing data
    X_test, y_test = load_all_gesture_data(base_dir, file_names, train=False)

    # Evaluate model
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")


def test_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if not ret:
            break
        if i < 130 or (i > 300 and i < 530) or (i > 600 and i < 700) or (i > 750 and i < 840) or (
                i > 900 and i < 1050) or (i > 1200 and i < 2000): continue
        print(i)
        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract HOG features from hand mask
        hog_features = extract_hog_features(frame).reshape(1, -1)  # Reshape for SVM

        # Predict gesture
        prediction = model.predict(hog_features)[0]

        # Display prediction on frame
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    base_dir = "C:/BGU/Git/DrawSomething/data"
    file_names = ('index_finger', 'up_thumb', 'open_hand', 'close_hand', 'three_fingers')
    X, y = load_all_gesture_data(base_dir, file_names)
    model = train(X, y)
    joblib.dump(model, "gesture_svm_model.pkl")

    # test_frames(gesture_paths, model)
    # test_video("records/test_mask.mp4", model)
