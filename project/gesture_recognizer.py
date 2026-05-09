import numpy as np
from typing import List, Optional, Tuple
from collections import deque

class GestureRecognizer:
    def __init__(self):
        self.finger_landmark_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20],
        }
        
        self.gesture_history = deque(maxlen=10)
        self.last_detected_gesture = "unknown"
        self.gesture_counts = {}
    
    def _get_finger_angle(self, landmarks: List[Tuple[float, float, float]], finger_name: str) -> float:
        if finger_name not in self.finger_landmark_indices:
            return 0.0
        
        indices = self.finger_landmark_indices[finger_name]
        if len(landmarks) <= max(indices):
            return 0.0
        
        p1 = np.array(landmarks[indices[0]])
        p2 = np.array(landmarks[indices[1]])
        p3 = np.array(landmarks[indices[2]])
        p4 = np.array(landmarks[indices[3]])
        
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        if finger_name == 'thumb':
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
            return np.degrees(angle)
        else:
            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)
            d3 = np.linalg.norm(v3)
            
            total_length = d1 + d2 + d3
            tip_to_base = np.linalg.norm(p4 - p1)
            
            if total_length < 0.001:
                return 0.0
            
            ratio = tip_to_base / total_length
            return ratio
    
    def _is_finger_straight(self, landmarks: List[Tuple[float, float, float]], finger_name: str) -> bool:
        ratio = self._get_finger_angle(landmarks, finger_name)
        if finger_name == 'thumb':
            return ratio < 50
        else:
            return ratio > 0.75
    
    def _is_finger_bent(self, landmarks: List[Tuple[float, float, float]], finger_name: str) -> bool:
        return not self._is_finger_straight(landmarks, finger_name)
    
    def _get_finger_tip_distance(self, landmarks: List[Tuple[float, float, float]], 
                                 finger1: str, finger2: str) -> float:
        if finger1 not in self.finger_landmark_indices or finger2 not in self.finger_landmark_indices:
            return float('inf')
        
        idx1 = self.finger_landmark_indices[finger1][-1]
        idx2 = self.finger_landmark_indices[finger2][-1]
        
        if len(landmarks) <= max(idx1, idx2):
            return float('inf')
        
        tip1 = np.array(landmarks[idx1])
        tip2 = np.array(landmarks[idx2])
        
        return np.linalg.norm(tip1 - tip2)
    
    def _detect_open_palm(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        fingers = ['index', 'middle', 'ring', 'pinky']
        straight_count = sum(1 for f in fingers if self._is_finger_straight(landmarks, f))
        
        palm_width = self._get_finger_tip_distance(landmarks, 'index', 'pinky')
        
        return straight_count >= 4 and palm_width > 0.25
    
    def _detect_fist(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        fingers = ['index', 'middle', 'ring', 'pinky']
        bent_count = sum(1 for f in fingers if self._is_finger_bent(landmarks, f))
        
        wrist = np.array(landmarks[0])
        index_mcp = np.array(landmarks[5])
        index_tip = np.array(landmarks[8])
        
        tip_to_wrist = np.linalg.norm(index_tip - wrist)
        mcp_to_wrist = np.linalg.norm(index_mcp - wrist)
        
        return bent_count >= 4 and tip_to_wrist < mcp_to_wrist * 1.2
    
    def _detect_scissors(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        index_straight = self._is_finger_straight(landmarks, 'index')
        middle_straight = self._is_finger_straight(landmarks, 'middle')
        ring_bent = self._is_finger_bent(landmarks, 'ring')
        pinky_bent = self._is_finger_bent(landmarks, 'pinky')
        
        distance = self._get_finger_tip_distance(landmarks, 'index', 'middle')
        
        return index_straight and middle_straight and ring_bent and pinky_bent and distance > 0.08
    
    def _detect_ok_gesture(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        thumb_tip = np.array(landmarks[4])
        index_tip = np.array(landmarks[8])
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        middle_straight = self._is_finger_straight(landmarks, 'middle')
        ring_straight = self._is_finger_straight(landmarks, 'ring')
        pinky_straight = self._is_finger_straight(landmarks, 'pinky')
        
        return distance < 0.08 and middle_straight and ring_straight and pinky_straight
    
    def _detect_index_pointing(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        index_straight = self._is_finger_straight(landmarks, 'index')
        middle_bent = self._is_finger_bent(landmarks, 'middle')
        ring_bent = self._is_finger_bent(landmarks, 'ring')
        pinky_bent = self._is_finger_bent(landmarks, 'pinky')
        
        thumb_tip = np.array(landmarks[4])
        index_mcp = np.array(landmarks[5])
        thumb_distance = np.linalg.norm(thumb_tip - index_mcp)
        
        index_tip = np.array(landmarks[8])
        middle_mcp = np.array(landmarks[9])
        index_height = index_tip[1] - middle_mcp[1]
        
        return index_straight and middle_bent and ring_bent and pinky_bent and \
               thumb_distance < 0.12 and index_height < -0.1
    
    def _detect_pinch(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        thumb_tip = np.array(landmarks[4])
        index_tip = np.array(landmarks[8])
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        middle_bent = self._is_finger_bent(landmarks, 'middle')
        ring_bent = self._is_finger_bent(landmarks, 'ring')
        pinky_bent = self._is_finger_bent(landmarks, 'pinky')
        
        return distance < 0.06 and middle_bent and ring_bent and pinky_bent
    
    def recognize(self, landmarks: Optional[List[Tuple[float, float, float]]]) -> str:
        if landmarks is None or len(landmarks) < 21:
            self._update_history("unknown")
            return "unknown"
        
        gesture_priority = [
            ('ok', self._detect_ok_gesture),
            ('pinch', self._detect_pinch),
            ('scissors', self._detect_scissors),
            ('index_pointing', self._detect_index_pointing),
            ('open_palm', self._detect_open_palm),
            ('fist', self._detect_fist),
        ]
        
        for gesture_name, detector_func in gesture_priority:
            if detector_func(landmarks):
                self._update_history(gesture_name)
                self.last_detected_gesture = gesture_name
                return gesture_name
        
        self._update_history("unknown")
        return "unknown"
    
    def _update_history(self, gesture: str):
        self.gesture_history.append(gesture)
        self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
    
    def get_stable_gesture(self) -> str:
        if len(self.gesture_history) < 5:
            return "unknown"
        
        gesture_counts = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        max_count = max(gesture_counts.values())
        if max_count >= 6:
            for gesture, count in gesture_counts.items():
                if count == max_count:
                    return gesture
        
        return "unknown"
    
    def get_gesture_confidence(self, gesture: str) -> float:
        if gesture == "unknown":
            return 0.0
        
        count = self.gesture_history.count(gesture)
        return count / len(self.gesture_history)
    
    def get_gesture_description(self, gesture: str) -> str:
        descriptions = {
            'open_palm': 'Open Palm',
            'fist': 'Fist',
            'scissors': 'Scissors',
            'ok': 'OK',
            'index_pointing': 'Index Pointing',
            'pinch': 'Pinch',
            'unknown': 'Unknown'
        }
        return descriptions.get(gesture, gesture)
