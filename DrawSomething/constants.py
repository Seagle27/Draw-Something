from enum import Enum


class ColorSpace(Enum):
    RGB = 1
    BGR = 2
    rg = 3
    HSV = 4


# Online model:
H_BINS = 8
S_BINS = 4
V_BINS = 2
# For a standard frontal-face detector, ship with OpenCV:
CASCADE_PATH = r"C:\BGU\Year_4\ImageProcessing\Draw-Something\data\raw\haarcascade_frontalcatface.xml"

# Offline model:
SKIN_GMM = "data/raw/skin_gmm.joblib"
NON_SKIN_GMM = "data/raw/non_skin_gmm.joblib"
