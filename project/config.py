import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
SCREENSHOT_DIR = os.path.join(PROJECT_DIR, 'screenshots')
ASSETS_DIR = os.path.join(PROJECT_DIR, 'assets')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

CAMERA_INDEX = 0

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

MAX_HANDS = 1

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

FINGER_UP_THRESHOLD = 0.7
FINGER_BENT_THRESHOLD = 0.3

GESTURE_CONFIRM_FRAMES = 5

COOLDOWN_DURATION = {
    'play': 1.5,
    'pause': 1.5,
    'next_page': 1.0,
    'screenshot': 2.0,
    'click': 0.5,
}

SMOOTHING_ALPHA = 0.3

SCREENSHOT_FORMAT = 'png'

LOG_LEVEL = 'INFO'
LOG_FILE_NAME = 'gesture_control.log'

MOUSE_MOVE_SCALE = 2.0

PINCH_THRESHOLD = 0.05
