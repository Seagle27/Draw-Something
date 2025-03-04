import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC


def preprocess_and_crop(binary_mask):
    """
    1. Find bounding box around the largest contour (the hand).
    2. Crop the image so only the hand region remains.
    3. Optionally resize to a fixed dimension.
    """
    # Find contours
    contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Largest contour => hand
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Crop
    cropped = binary_mask[y:y+h, x:x+w]
    
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


def load_data(gesture_paths, train=True):
    X, y = [], []
    suffix = "train" if train else "test"
    for gesture_name in os.listdir(gesture_paths):
        set_path = os.path.join(gesture_paths, gesture_name, suffix)
        for img_name in os.listdir(set_path):
            mask = cv2.imread(os.path.join(set_path, img_name), cv2.IMREAD_GRAYSCALE)
            features = extract_hog_features(mask)
            X.append(features)
            y.append(gesture_name)
    return np.array(X), np.array(y)


def train(gesture_paths):
    # Load training data
    X_train, y_train = load_data(gesture_paths, train=True)
    # Train SVM
    svm_model = SVC(kernel='rbf', C=1, gamma='scale')
    svm_model.fit(X_train, y_train)
    return svm_model


def test_frames(gesture_paths, model):
    # Load testing data
    X_test, y_test = load_data(gesture_paths, train=False)

    # Evaluate model
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")


def test_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    i=0
    while cap.isOpened():
        i+=1
        ret, frame = cap.read()
        if not ret:
            break
        if i < 130 or (i>300 and i<530) or (i>600 and i<700) or (i>750 and i<840) or (i>900 and i<1050) or (i>1200 and i<2000): continue
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
    gesture_paths = "data/HandGestures"
    model = train(gesture_paths)
    # test_frames(gesture_paths, model)
    test_video("records/test_mask.mp4", model)
