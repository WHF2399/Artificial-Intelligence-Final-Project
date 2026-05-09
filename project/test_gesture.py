import cv2
import numpy as np
import os

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_TASKS_API = True
except ImportError:
    USE_TASKS_API = False

def main():
    print("Initializing HandTracker...")
    
    detector = None
    hands = None
    
    if USE_TASKS_API:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5
            )
        )
        print("Using Tasks API")
    else:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("Using legacy API")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera opened, press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        landmarks = None
        
        if USE_TASKS_API and detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_image)
            
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        elif hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        
        if landmarks:
            print(f"Landmarks detected: {len(landmarks)} points")
            
            if len(landmarks) >= 21:
                index_tip = landmarks[8]
                print(f"Index tip: ({index_tip[0]:.3f}, {index_tip[1]:.3f})")
                
                fingers_extended = []
                for i, finger_name in enumerate(['index', 'middle', 'ring', 'pinky']):
                    tip_idx = 8 + i * 4
                    base_idx = 5 + i * 4
                    if tip_idx < len(landmarks) and base_idx < len(landmarks):
                        tip_y = landmarks[tip_idx][1]
                        base_y = landmarks[base_idx][1]
                        is_extended = tip_y < base_y - 0.05
                        fingers_extended.append(is_extended)
                        print(f"  {finger_name}: {'extended' if is_extended else 'bent'} (tip_y={tip_y:.3f}, base_y={base_y:.3f})")
                
                thumb_tip = landmarks[4]
                wrist = landmarks[0]
                thumb_dist = np.linalg.norm(np.array(thumb_tip[:2]) - np.array(wrist[:2]))
                print(f"  thumb: distance to wrist = {thumb_dist:.3f}")
                
                extended_count = sum(fingers_extended)
                print(f"Extended fingers: {extended_count}/4")
                
                if extended_count >= 4:
                    print("  -> Gesture: Open Palm")
                elif extended_count == 0:
                    print("  -> Gesture: Fist")
                elif extended_count == 1 and fingers_extended[0]:
                    print("  -> Gesture: Index Pointing")
                elif extended_count == 2 and fingers_extended[0] and fingers_extended[1]:
                    print("  -> Gesture: Scissors")
                else:
                    print("  -> Gesture: Unknown")
            
            cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Gesture Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if detector:
        detector.close()
    if hands:
        hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
