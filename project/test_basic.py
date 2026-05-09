import cv2
import mediapipe as mp
import numpy as np

print("Testing OpenCV...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)
print("Camera opened successfully")

print("\nTesting MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=1)
print("MediaPipe Hands initialized successfully")

print("\nTesting numpy...")
arr = np.array([1, 2, 3])
print(f"numpy array: {arr}")

print("\nAll basic tests passed!")
cap.release()
hands.close()
