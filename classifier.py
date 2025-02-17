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

def extract_hog_features(img_path):
    """
    Extract Histogram of Oriented Gradients (HOG) features from a binary mask.
    """
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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
        if gesture_name != "point": continue
        set_path = os.path.join(gesture_paths, gesture_name, suffix)
        for img_name in os.listdir(set_path):
            features = extract_hog_features(os.path.join(set_path, img_name))
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

def test(gesture_paths, model):
    # Load testing data
    X_test, y_test = load_data(gesture_paths, train=False)

    # Evaluate model
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")



if __name__ == '__main__':
    gesture_paths = "data/HandGestures"
    model = train(gesture_paths)
    test(gesture_paths, model)
