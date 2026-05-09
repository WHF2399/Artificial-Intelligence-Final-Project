import mediapipe
print(f"MediaPipe version: {mediapipe.__version__}")
print("\nMediaPipe attributes:")
print([attr for attr in dir(mediapipe) if not attr.startswith('_')])

try:
    from mediapipe import solutions
    print("\nsolutions module found!")
except ImportError as e:
    print(f"\nsolutions import error: {e}")

import os
print(f"\nMediaPipe path: {mediapipe.__file__}")
