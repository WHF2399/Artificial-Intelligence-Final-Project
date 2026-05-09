import cv2
import time
import sys
from config import (
    CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT,
    COOLDOWN_DURATION
)
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from gesture_mapper import GestureMapper
from mouse_controller import MouseController
from visualizer import Visualizer
from logger import Logger

def main():
    logger = Logger()
    logger.log_info("Starting Gesture Control System...")

    hand_tracker = HandTracker()
    gesture_recognizer = GestureRecognizer()
    gesture_mapper = GestureMapper()
    mouse_controller = MouseController()
    visualizer = Visualizer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    if not cap.isOpened():
        logger.log_error("Failed to open camera")
        print("Error: Could not open camera")
        return

    logger.log_info(f"Camera opened successfully: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")

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
            mx, my = mouse_controller.get_mouse_position()
            logger.log_mouse_move(mx, my)

    def handle_click(landmarks=None):
        mouse_controller.click()
        logger.log_operation_executed('click', 'pinch')

    gesture_mapper.register_operation_handler('play', handle_play)
    gesture_mapper.register_operation_handler('pause', handle_pause)
    gesture_mapper.register_operation_handler('next_page', handle_next_page)
    gesture_mapper.register_operation_handler('screenshot', handle_screenshot)
    gesture_mapper.register_operation_handler('mouse_move', handle_mouse_move)
    gesture_mapper.register_operation_handler('click', handle_click)

    logger.log_info("Operation handlers registered")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.log_warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            success, landmarks, world_landmarks = hand_tracker.process(frame)

            if success and landmarks:
                gesture = gesture_recognizer.recognize(landmarks)
                stable_gesture = gesture_recognizer.get_stable_gesture()
                confidence = gesture_recognizer.get_gesture_confidence(stable_gesture)

                if gesture != "unknown":
                    print(f"Detected gesture: {gesture} (stable: {stable_gesture}, confidence: {confidence:.2f})")

                if stable_gesture != "unknown":
                    logger.log_gesture_detected(stable_gesture, confidence)

                operation = gesture_mapper.process_gesture(stable_gesture, landmarks)
                if operation:
                    logger.log_operation_executed(operation, stable_gesture)
                    print(f"Executed operation: {operation}")

                frame = hand_tracker.draw_landmarks(frame)

                if gesture == 'index_pointing' and landmarks:
                    index_tip = landmarks[8]
                    print(f"Index pointing detected, moving mouse to: ({index_tip[0]:.2f}, {index_tip[1]:.2f})")
                    mouse_controller.move_mouse(index_tip[0], index_tip[1], CAPTURE_WIDTH, CAPTURE_HEIGHT)
                    mx, my = mouse_controller.get_mouse_position()
                    logger.log_mouse_move(mx, my)
                    frame = visualizer.draw_finger_tip(frame, index_tip[0], index_tip[1], 
                                                       CAPTURE_WIDTH, CAPTURE_HEIGHT)

            else:
                gesture = "unknown"
                stable_gesture = "unknown"
                confidence = 0.0

            fps = visualizer.calculate_fps()
            frame = visualizer.draw_fps(frame, fps)
            frame = visualizer.draw_gesture(frame, gesture_recognizer.get_gesture_description(stable_gesture), confidence)
            frame = visualizer.draw_state(frame, gesture_mapper.get_state())

            cooldown_info = {}
            for operation in COOLDOWN_DURATION:
                remaining = gesture_mapper.get_operation_cooldown(operation)
                if remaining > 0:
                    cooldown_info[operation] = remaining
            frame = visualizer.draw_cooldown(frame, cooldown_info)

            frame = visualizer.draw_help_info(frame)

            mx, my = mouse_controller.get_mouse_position()
            frame = visualizer.draw_mouse_position(frame, mx, my)

            statistics = gesture_mapper.get_statistics()
            frame = visualizer.draw_statistics(frame, statistics)

            cv2.imshow('AI Gesture Control', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.log_info("Quit key pressed")
                break
            elif key == ord('r'):
                gesture_recognizer.reset()
                gesture_mapper.reset_statistics()
                logger.log_info("Statistics reset")
            elif key == ord('s'):
                logger.print_statistics()

    except KeyboardInterrupt:
        logger.log_info("Keyboard interrupt received")
    except Exception as e:
        logger.log_error_occurred(e, "in main loop")
        print(f"Error: {e}")
    finally:
        cap.release()
        hand_tracker.release()
        cv2.destroyAllWindows()
        logger.print_statistics()
        logger.close()
        logger.log_info("Gesture Control System closed")

if __name__ == "__main__":
    main()
