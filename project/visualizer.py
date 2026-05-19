import cv2
import numpy as np
import time
from typing import Optional, Dict

class Visualizer:
    def __init__(self):
        self.fps_history = []
        self.max_fps_history = 30
        self.last_frame_time = time.time()
        
        self.text_color = (0, 255, 0)
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.6
        self.text_thickness = 2
        
        self.bg_color = (0, 0, 0)
        self.bg_alpha = 0.6
        
        self.gesture_colors = {
            'open_palm': (0, 255, 0),
            'fist': (0, 0, 255),
            'scissors': (255, 0, 0),
            'thumbs_up': (0, 165, 255),
            'ok': (0, 255, 255),
            'index_pointing': (255, 255, 0),
            'pinch': (255, 0, 255),
            'unknown': (128, 128, 128)
        }

    def calculate_fps(self) -> float:
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_fps_history:
            self.fps_history.pop(0)
        
        return np.mean(self.fps_history) if self.fps_history else 0.0

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        fps_text = f"FPS: {fps:.1f}"
        frame = self._draw_text_with_background(frame, fps_text, (10, 30))
        return frame

    def draw_gesture(self, frame: np.ndarray, gesture: str, confidence: float = 1.0) -> np.ndarray:
        gesture_text = f"Gesture: {gesture} ({confidence:.2f})"
        color = self.gesture_colors.get(gesture, (128, 128, 128))
        frame = self._draw_text_with_background(frame, gesture_text, (10, 60), color)
        return frame

    def draw_state(self, frame: np.ndarray, state: str) -> np.ndarray:
        state_text = f"State: {state}"
        frame = self._draw_text_with_background(frame, state_text, (10, 90))
        return frame

    def draw_statistics(self, frame: np.ndarray, statistics: Dict[str, int]) -> np.ndarray:
        y_offset = 120
        for gesture, count in statistics.items():
            stat_text = f"{gesture}: {count}"
            frame = self._draw_text_with_background(frame, stat_text, (10, y_offset))
            y_offset += 30
            if y_offset > frame.shape[0] - 30:
                break
        return frame

    def draw_cooldown(self, frame: np.ndarray, cooldown_info: Dict[str, float]) -> np.ndarray:
        y_offset = frame.shape[0] - 30
        for operation, remaining in cooldown_info.items():
            if remaining > 0:
                cooldown_text = f"{operation}: {remaining:.1f}s"
                frame = self._draw_text_with_background(frame, cooldown_text, (10, y_offset), (0, 0, 255))
                y_offset -= 30
                if y_offset < 150:
                    break
        return frame

    def draw_mouse_position(self, frame: np.ndarray, x: int, y: int) -> np.ndarray:
        mouse_text = f"Mouse: ({x}, {y})"
        frame = self._draw_text_with_background(frame, mouse_text, (10, frame.shape[0] - 30))
        return frame

    def _draw_text_with_background(self, frame: np.ndarray, text: str, position: tuple, color: tuple = None) -> np.ndarray:
        if color is None:
            color = self.text_color
        
        (text_width, text_height), _ = cv2.getTextSize(text, self.text_font, self.text_scale, self.text_thickness)
        
        bg_x1 = max(0, position[0] - 5)
        bg_y1 = max(0, position[1] - text_height - 5)
        bg_x2 = min(frame.shape[1], position[0] + text_width + 5)
        bg_y2 = min(frame.shape[0], position[1] + 5)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.bg_color, -1)
        cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0, frame)
        
        cv2.putText(frame, text, position, self.text_font, self.text_scale, color, self.text_thickness)
        
        return frame

    def draw_help_info(self, frame: np.ndarray) -> np.ndarray:
        help_lines = [
            "Open Palm - Pause",
            "Fist - Play",
            "Scissors - Next Page",
            "Thumbs Up - Previous Page",
            "Three Fingers - Scroll (non-PPT)",
            "OK - Screenshot",
            "Index Pointing - Mouse Move",
            "Pinch - Click"
        ]
        
        start_x = frame.shape[1] - 180
        start_y = 30
        
        for i, line in enumerate(help_lines):
            y = start_y + i * 25
            if y > frame.shape[0] - 20:
                break
            frame = self._draw_text_with_background(frame, line, (start_x, y), (128, 128, 255))
        
        return frame

    def draw_landmarks(self, frame: np.ndarray, landmarks, connections=None) -> np.ndarray:
        if landmarks is None:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        for i, landmark in enumerate(landmarks):
            x, y, z = landmark
            px = int(x * frame_width)
            py = int(y * frame_height)
            
            cv2.circle(annotated_frame, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(annotated_frame, (px, py), 8, (0, 255, 0), 2)
        
        if connections is not None:
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    x1, y1, _ = landmarks[start_idx]
                    x2, y2, _ = landmarks[end_idx]
                    px1 = int(x1 * frame_width)
                    py1 = int(y1 * frame_height)
                    px2 = int(x2 * frame_width)
                    py2 = int(y2 * frame_height)
                    cv2.line(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        
        return annotated_frame

    def draw_finger_tip(self, frame: np.ndarray, x: float, y: float, frame_width: int, frame_height: int) -> np.ndarray:
        px = int(x * frame_width)
        py = int(y * frame_height)
        
        cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)
        cv2.circle(frame, (px, py), 15, (0, 0, 255), 2)
        
        return frame

    def get_average_fps(self) -> float:
        return np.mean(self.fps_history) if self.fps_history else 0.0

    def reset(self):
        self.fps_history = []
        self.last_frame_time = time.time()
