import cv2
import numpy as np
import os
from typing import Optional, Tuple, List
from config import (
    MAX_HANDS, MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_TASKS_API = True
except ImportError:
    USE_TASKS_API = False

class HandTracker:
    def __init__(self):
        self.detector = None
        self.hands = None
        self._initialize_detector()
        
        self.frame_width = 0
        self.frame_height = 0
        self.last_landmarks = None
        self.last_world_landmarks = None
        self.detection_confidence = 0.0

    def _initialize_detector(self):
        try:
            if USE_TASKS_API:
                project_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(project_dir, 'assets', 'hand_landmarker.task')
                
                if os.path.exists(model_path):
                    print(f"Using local model: {model_path}")
                    base_options = python.BaseOptions(model_asset_path=model_path)
                else:
                    print("Downloading model... This may take a moment.")
                    base_options = python.BaseOptions()
                
                self.detector = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=base_options,
                        num_hands=MAX_HANDS,
                        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE
                    )
                )
                print("HandLandmarker initialized successfully!")
            else:
                print("Using legacy solutions API")
                mp_hands = mp.solutions.hands
                self.hands = mp_hands.Hands(
                    model_complexity=0,
                    max_num_hands=MAX_HANDS,
                    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
                )
                print("Hands detector initialized successfully!")
                
        except Exception as e:
            print(f"Detector init error: {e}")
            self.detector = None
            self.hands = None

    def process(self, frame: np.ndarray) -> Tuple[bool, Optional[List[Tuple[float, float, float]]], Optional[List[Tuple[float, float, float]]]]:
        self.frame_height, self.frame_width, _ = frame.shape
        
        if USE_TASKS_API and self.detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.detector.detect(mp_image)
            
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                self.last_landmarks = self._normalize_landmarks(hand_landmarks)
                
                if result.hand_world_landmarks:
                    world_landmarks = result.hand_world_landmarks[0]
                    self.last_world_landmarks = self._normalize_world_landmarks(world_landmarks)
                
                self.detection_confidence = 1.0
                return True, self.last_landmarks, self.last_world_landmarks
        elif self.hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.hands.process(frame_rgb)
            frame_rgb.flags.writeable = True
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.last_landmarks = self._normalize_landmarks_legacy(hand_landmarks)
                
                if results.multi_hand_world_landmarks:
                    world_landmarks = results.multi_hand_world_landmarks[0]
                    self.last_world_landmarks = self._normalize_world_landmarks_legacy(world_landmarks)
                
                self.detection_confidence = 1.0
                return True, self.last_landmarks, self.last_world_landmarks
        
        self.last_landmarks = None
        self.last_world_landmarks = None
        return False, None, None

    def _normalize_landmarks(self, hand_landmarks) -> List[Tuple[float, float, float]]:
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

    def _normalize_world_landmarks(self, world_landmarks) -> List[Tuple[float, float, float]]:
        landmarks = []
        for landmark in world_landmarks:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

    def _normalize_landmarks_legacy(self, hand_landmarks) -> List[Tuple[float, float, float]]:
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

    def _normalize_world_landmarks_legacy(self, world_landmarks) -> List[Tuple[float, float, float]]:
        landmarks = []
        for landmark in world_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        if self.last_landmarks is None:
            return frame
        
        annotated_frame = frame.copy()
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (9, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
        for start, end in connections:
            if start < len(self.last_landmarks) and end < len(self.last_landmarks):
                x1, y1, z1 = self.last_landmarks[start]
                x2, y2, z2 = self.last_landmarks[end]
                
                px1 = int(x1 * self.frame_width)
                py1 = int(y1 * self.frame_height)
                px2 = int(x2 * self.frame_width)
                py2 = int(y2 * self.frame_height)
                
                cv2.line(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        
        for x, y, z in self.last_landmarks:
            px = int(x * self.frame_width)
            py = int(y * self.frame_height)
            cv2.circle(annotated_frame, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(annotated_frame, (px, py), 8, (0, 255, 0), 2)
        
        return annotated_frame

    def get_hand_center(self) -> Optional[Tuple[float, float]]:
        if self.last_landmarks is None:
            return None
        wrist = self.last_landmarks[0]
        return (wrist[0], wrist[1])

    def get_finger_tip(self, finger_index: int) -> Optional[Tuple[float, float, float]]:
        if self.last_landmarks is None:
            return None
        
        tip_indices = [4, 8, 12, 16, 20]
        
        if finger_index < 0 or finger_index >= len(tip_indices):
            return None
        
        return self.last_landmarks[tip_indices[finger_index]]

    def release(self):
        if self.detector:
            self.detector.close()
        if self.hands:
            self.hands.close()

    def get_confidence(self) -> float:
        return self.detection_confidence

    def is_hand_detected(self) -> bool:
        return self.last_landmarks is not None

    def is_detector_ready(self) -> bool:
        return self.detector is not None or self.hands is not None
