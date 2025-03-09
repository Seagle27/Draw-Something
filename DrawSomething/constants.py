from enum import Enum


class ColorSpace(Enum):
    RGB = 1
    BGR = 2
    rg = 3
    HSV = 4


VIDEO_SOURCE = 0

# Thresholds
ONLINE_THRESHOLD = 0.5
OFFLINE_THRESHOLD = 0.35


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
NUM_GESTURES = 5
file_names = ('index_finger', 'two_fingers', 'three_fingers', 'open_hand', 'nonsense')

# gesture recognition cfg:
WIN_SIZE = 40
MIN_CHANGE_FRAME = 10


# Threshold for detecting a "jump" (in pixels)
JUMP_THRESHOLD = 50
# Number of frames needed to accept a new position
STABILITY_FRAMES = 12
# Minimum smoothing factor (very smooth, slow response)
EMA_ALPHA_MIN = 0.25
# Maximum smoothing factor (fast response)
EMA_ALPHA_MAX = 0.9
# Movement speed below this uses max smoothing
SPEED_THRESHOLD_LOW = 10
# Movement speed above this uses min smoothing
SPEED_THRESHOLD_HIGH = 30

# GUI Parameters
FRAME_WIDTH =  640
FRAME_HEIGHT = 480

# Color Bar Constants
START_X_POS = 110 # 30
Y_POS = 30
BUTTON_RADIUS = 20
BUTTON_ROAIUS_OF_SELECTED = 30
BUTTON_SPACING = 70 # 70

BRUSH_BLACK_ICON = "GUI_photos/brush_black.png"
ERASER_BLACK_ICON = "GUI_photos/eraser_black.png"
ERASER_WHITE_ICON = "GUI_photos/eraser_white.png"
ERASER_S_BLACK_ICON = "GUI_photos/eraserS_black.png"
ERASER_S_WHITE_ICON = "GUI_photos/eraser_white.png"

COLORS_OPTIONS = {
            "black": (0, 0, 0),
            "red": (0, 0, 255),
            "salmon": (122, 158, 227),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "blue": (255, 0, 0)
        }

# Width Bar Constanst
WIDTH_OPTIONS = [2,4,8,12]
#START_Y_POS = 110
START_Y_POS = 100
X_POS = 110 # 30 #610
#X_CLEAR = 30
#X_CLEAR = START_X_POS + BUTTON_SPACING*(len(WIDTH_OPTIONS)) # with the width
X_CLEAR = START_X_POS + BUTTON_SPACING*(len(COLORS_OPTIONS)) # with color
#Y_CLEAR = START_Y_POS # with the width
Y_CLEAR = Y_POS # with color

#Y_CLEAR = START_Y_POS + BUTTON_SPACING*(len(WIDTH_OPTIONS))
ERASER_RADIUS = 10
HISTORY_MAX_LENGTH = 5 # 10 # was 1
HISTORY_MAX_LENGTH_3FINGERS = 5

# Detection Shapes
NUM_BINS = 36
THRESHOLD_abstract = 7
THRESHOLD = 7
GAUSS_BLUR_K = 7
GAUSS_BLUR_SIGMA = 10
SOBEL_K = 3

