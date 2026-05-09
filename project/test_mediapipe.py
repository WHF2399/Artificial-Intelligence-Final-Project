import sys
import time

print("Starting MediaPipe test...")
start_time = time.time()

try:
    import mediapipe as mp
    print(f"MediaPipe imported successfully")
    
    print("Initializing Hands model (may take time to download)...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    elapsed = time.time() - start_time
    print(f"Hands model initialized in {elapsed:.2f} seconds")
    
    hands.close()
    print("Test completed successfully!")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"Error after {elapsed:.2f} seconds: {e}")
    import traceback
    traceback.print_exc()
