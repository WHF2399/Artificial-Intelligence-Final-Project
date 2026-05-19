import time
from typing import Optional, Dict, Callable, Any
from config import COOLDOWN_DURATION, GESTURE_CONFIRM_FRAMES

class GestureMapper:
    def __init__(self):
        self.state = "idle"
        self.last_operation_time = {}
        self.gesture_statistics = {}
        self.confirm_counter = 0
        self.last_confirmed_gesture = "unknown"
        
        self.gesture_to_operation = {
            'open_palm': 'pause',
            'fist': 'play',
            'scissors': 'next_page',
            'thumbs_up': 'previous_page',
            'three_finger_scroll': 'scroll',
            'ok': 'screenshot',
            'index_pointing': 'mouse_move',
            'pinch': 'click',
        }
        
        self.operation_handlers = {}
        
        self.state_transitions = {
            'idle': ['gesture_detected'],
            'gesture_detected': ['operation_executed', 'idle'],
            'operation_executed': ['cooldown', 'idle'],
            'cooldown': ['idle'],
        }
        
        self.media_playing = True

    def register_operation_handler(self, operation: str, handler: Callable):
        self.operation_handlers[operation] = handler

    def _can_execute_operation(self, operation: str) -> bool:
        if operation not in COOLDOWN_DURATION:
            return True
        
        last_time = self.last_operation_time.get(operation, 0)
        current_time = time.time()
        elapsed = current_time - last_time
        
        return elapsed >= COOLDOWN_DURATION[operation]

    def _transition_state(self, new_state: str) -> bool:
        if new_state in self.state_transitions.get(self.state, []):
            self.state = new_state
            return True
        return False

    def process_gesture(self, gesture: str, landmarks=None) -> Optional[str]:
        if gesture == "unknown":
            self.confirm_counter = 0
            self._transition_state("idle")
            return None

        if gesture == self.last_confirmed_gesture:
            self.confirm_counter += 1
        else:
            self.confirm_counter = 1
            self.last_confirmed_gesture = gesture

        if self.confirm_counter < GESTURE_CONFIRM_FRAMES:
            self._transition_state("gesture_detected")
            return None

        operation = self.gesture_to_operation.get(gesture)
        if operation is None:
            return None

        if gesture == 'open_palm':
            if not self.media_playing:
                return None
            self.media_playing = False
        elif gesture == 'fist':
            if self.media_playing:
                return None
            self.media_playing = True
        elif gesture == 'index_pointing':
            if not self._can_execute_operation('mouse_move'):
                return None
        else:
            if not self._can_execute_operation(operation):
                return None

        self._transition_state("operation_executed")
        
        self.last_operation_time[operation] = time.time()
        
        self._update_statistics(gesture)
        
        if operation in self.operation_handlers:
            try:
                self.operation_handlers[operation](landmarks)
            except Exception as e:
                print(f"Error executing operation {operation}: {e}")
        
        self._transition_state("cooldown")
        
        return operation

    def get_media_playing(self) -> bool:
        return self.media_playing

    def _update_statistics(self, gesture: str):
        if gesture not in self.gesture_statistics:
            self.gesture_statistics[gesture] = 0
        self.gesture_statistics[gesture] += 1

    def get_statistics(self) -> Dict[str, int]:
        return self.gesture_statistics.copy()

    def get_state(self) -> str:
        return self.state

    def get_operation_cooldown(self, operation: str) -> float:
        if operation not in COOLDOWN_DURATION:
            return 0.0
        
        last_time = self.last_operation_time.get(operation, 0)
        elapsed = time.time() - last_time
        remaining = max(0, COOLDOWN_DURATION[operation] - elapsed)
        return remaining

    def is_in_cooldown(self) -> bool:
        return self.state == "cooldown"

    def reset_statistics(self):
        self.gesture_statistics = {}

    def get_gesture_operation(self, gesture: str) -> Optional[str]:
        return self.gesture_to_operation.get(gesture)

    def get_all_operations(self) -> list:
        return list(self.gesture_to_operation.values())

    def get_operation_description(self, operation: str) -> str:
        descriptions = {
            'pause': '暂停播放',
            'play': '播放',
            'next_page': '向下滚动',
            'previous_page': '向上滚动',
            'screenshot': '截图',
            'mouse_move': '鼠标移动',
            'click': '鼠标点击',
        }
        return descriptions.get(operation, operation)
