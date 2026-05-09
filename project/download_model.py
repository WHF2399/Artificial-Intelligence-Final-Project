import os
import urllib.request
import zipfile

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
MODEL_PATH = os.path.join(ASSETS_DIR, 'hand_landmarker.task')

print(f"Assets directory: {ASSETS_DIR}")
os.makedirs(ASSETS_DIR, exist_ok=True)

if os.path.exists(MODEL_PATH):
    print(f"Model file already exists: {MODEL_PATH}")
    print(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
else:
    print(f"Downloading model from: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded successfully!")
        print(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually download the model from:")
        print(MODEL_URL)
        print(f"And place it in: {ASSETS_DIR}")
