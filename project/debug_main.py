import cv2
import time
import sys
from config import (
    CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT,
    COOLDOWN_DURATION
)

print("Importing modules...")

try:
    from hand_tracker import HandTracker
    print("HandTracker imported successfully")
except Exception as e:
    print(f"HandTracker import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from gesture_recognizer import GestureRecognizer
    print("GestureRecognizer imported successfully")
except Exception as e:
    print(f"GestureRecognizer import error: {e}")
    sys.exit(1)

try:
    from gesture_mapper import GestureMapper
    print("GestureMapper imported successfully")
except Exception as e:
    print(f"GestureMapper import error: {e}")
    sys.exit(1)

try:
    from mouse_controller import MouseController
    print("MouseController imported successfully")
except Exception as e:
    print(f"MouseController import error: {e}")
    sys.exit(1)

try:
    from visualizer import Visualizer
    print("Visualizer imported successfully")
except Exception as e:
    print(f"Visualizer import error: {e}")
    sys.exit(1)

try:
    from logger import Logger
    print("Logger imported successfully")
except Exception as e:
    print(f"Logger import error: {e}")
    sys.exit(1)

print("\nInitializing components...")

try:
    logger = Logger()
    logger.log_info("Starting Gesture Control System...")
    print("Logger initialized")
except Exception as e:
    print(f"Logger init error: {e}")
    sys.exit(1)

try:
    hand_tracker = HandTracker()
    print("HandTracker initialized")
except Exception as e:
    print(f"HandTracker init error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    gesture_recognizer = GestureRecognizer()
    print("GestureRecognizer initialized")
except Exception as e:
    print(f"GestureRecognizer init error: {e}")
    sys.exit(1)

try:
    gesture_mapper = GestureMapper()
    print("GestureMapper initialized")
except Exception as e:
    print(f"GestureMapper init error: {e}")
    sys.exit(1)

try:
    mouse_controller = MouseController()
    print("MouseController initialized")
except Exception as e:
    print(f"MouseController init error: {e}")
    sys.exit(1)

try:
    visualizer = Visualizer()
    print("Visualizer initialized")
except Exception as e:
    print(f"Visualizer init error: {e}")
    sys.exit(1)

print("\nOpening camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open camera")
    sys.exit(1)
print(f"Camera opened successfully: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")

print("\nRegistering handlers...")
def handle_play(landmarks=None):
    mouse_controller.play_media()
    logger.log_operation_executed('play', 'fist')

def handle_pause(landmarks=None):
    mouse_controller.pause_media()
    logger.log_operation_executed('pause', 'open_palm')

def handle_next_page(landmarks=None):
    mouse_controller.next_page()
    logger.log_operation_executed('next_page', 'scissors')

def handle_screenshot(landmarks=None):
    filepath = mouse_controller.take_screenshot()
    if filepath:
        logger.log_screenshot(filepath)
        logger.log_operation_executed('screenshot', 'ok')

def handle_mouse_move(landmarks=None):
    if landmarks and len(landmarks) > 8:
        index_tip = landmarks[8]
        mouse_controller.move_mouse(index_tip[0], index_tip[1], CAPTURE_WIDTH, CAPTURE_HEIGHT)

def handle_click(landmarks=None):
    mouse_controller.click()
    logger.log_operation_executed('click', 'pinch')

gesture_mapper.register_operation_handler('play', handle_play)
gesture_mapper.register_operation_handler('pause', handle_pause)
gesture_mapper.register_operation_handler('next_page', handle_next_page)
gesture_mapper.register_operation_handler('screenshot', handle_screenshot)
gesture_mapper.register_operation_handler('mouse_move', handle_mouse_move)
gesture_mapper.register_operation_handler('click', handle_click)
print("Handlers registered")

print("\nStarting main loop...")
frame_count = 0
start_time = time.time()

try:
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)

        success, landmarks, world_landmarks = hand_tracker.process(frame)

        if success and landmarks:
            gesture = gesture_recognizer.recognize(landmarks)
            stable_gesture = gesture_recognizer.get_stable_gesture()
            
            operation = gesture_mapper.process_gesture(stable_gesture, landmarks)
            
            frame = hand_tracker.draw_landmarks(frame)

        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('AI Gesture Control', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit key pressed")
            break

except KeyboardInterrupt:
    print("Keyboard interrupt")
except Exception as e:
    print(f"Error in main loop: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    hand_tracker.release()
    cv2.destroyAllWindows()
    print("System closed")
