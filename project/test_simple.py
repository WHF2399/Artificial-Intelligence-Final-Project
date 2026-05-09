import sys
print(f"Python version: {sys.version}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV import error: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")

try:
    import pyautogui
    print("PyAutoGUI imported successfully")
except ImportError as e:
    print(f"PyAutoGUI import error: {e}")

print("\nBasic imports completed!")
