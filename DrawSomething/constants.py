from enum import Enum


class ColorSpace(Enum):
    RGB = 1
    BGR = 2
    rg = 3
    HSV = 4


VIDEO_SOURCE = 0

# Thresholds
ONLINE_THRESHOLD = 0.5
OFFLINE_THRESHOLD = 0.4


# Online model:
H_BINS = 8
S_BINS = 4
V_BINS = 2


# Face Detection:
CASCADE_FACE_DETECTOR = 'haarcascade_frontalface_default.xml'
SCALE_FACTOR = 1.3

# Offline model:
SKIN_GMM = "data/raw/skin_gmm.joblib"
NON_SKIN_GMM = "data/raw/non_skin_gmm.joblib"

# SVM model:
SVM_MODEL_PATH = "data/raw/gesture_svm_model.pkl"

# gesture recognition cfg:
WIN_SIZE = 40
MIN_CHANGE_FRAME = 10