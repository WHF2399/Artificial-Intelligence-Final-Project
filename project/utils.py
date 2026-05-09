import numpy as np
import cv2
from typing import Tuple, List

def calculate_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(p1: Tuple[float, float, float], p2: Tuple[float, float, float], p3: Tuple[float, float, float]) -> float:
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norm_product == 0:
        return 0.0
    
    angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return np.degrees(angle)

def normalize_landmarks(landmarks: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if not landmarks:
        return []
    
    wrist = landmarks[0]
    normalized = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in landmarks]
    
    max_dist = max(np.linalg.norm(np.array(p)) for p in normalized)
    if max_dist > 0:
        normalized = [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in normalized]
    
    return normalized

def resize_frame(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    height, width = frame.shape[:2]
    
    scale = min(max_width / width, max_height / height)
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    return frame

def flip_frame(frame: np.ndarray, flip_code: int = 1) -> np.ndarray:
    return cv2.flip(frame, flip_code)

def add_mirror_effect(frame: np.ndarray) -> np.ndarray:
    return cv2.flip(frame, 1)

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))

def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return ((value - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

def smooth_value(current: float, target: float, alpha: float = 0.5) -> float:
    return alpha * target + (1 - alpha) * current

def is_point_inside_bounds(x: float, y: float, width: int, height: int, margin: int = 20) -> bool:
    return (margin <= x <= width - margin) and (margin <= y <= height - margin)

def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def get_finger_names() -> List[str]:
    return ['thumb', 'index', 'middle', 'ring', 'pinky']

def get_landmark_indices(finger_name: str) -> List[int]:
    indices = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }
    return indices.get(finger_name, [])

def calculate_hand_size(landmarks: List[Tuple[float, float, float]]) -> float:
    if len(landmarks) < 21:
        return 0.0
    
    wrist = landmarks[0]
    all_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
    
    max_dist = max(calculate_distance(wrist, tip) for tip in all_tips)
    return max_dist

def create_overlay(frame: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int] = (0, 255, 0), font_scale: float = 0.6,
                   thickness: int = 2, bg_alpha: float = 0.6) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    bg_x1 = max(0, x - 5)
    bg_y1 = max(0, y - text_height - 5)
    bg_x2 = min(frame.shape[1], x + text_width + 5)
    bg_y2 = min(frame.shape[0], y + 5)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    
    return frame
