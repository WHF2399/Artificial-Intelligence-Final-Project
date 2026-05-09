import os
import time
import logging
from typing import Optional, Dict
from config import LOG_DIR, LOG_FILE_NAME, LOG_LEVEL

class Logger:
    def __init__(self):
        self.log_file = os.path.join(LOG_DIR, LOG_FILE_NAME)
        
        os.makedirs(LOG_DIR, exist_ok=True)
        
        self.logger = logging.getLogger('gesture_control')
        self.logger.setLevel(self._get_log_level(LOG_LEVEL))
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.gesture_statistics = {}
        self.session_start_time = time.time()
        self.total_operations = 0
        
        self.log_info("Logger initialized")
        self.log_info(f"Session started at {time.ctime(self.session_start_time)}")

    def _get_log_level(self, level_str: str) -> int:
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(level_str.upper(), logging.INFO)

    def log_debug(self, message: str):
        self.logger.debug(message)

    def log_info(self, message: str):
        self.logger.info(message)

    def log_warning(self, message: str):
        self.logger.warning(message)

    def log_error(self, message: str):
        self.logger.error(message)

    def log_critical(self, message: str):
        self.logger.critical(message)

    def log_gesture_detected(self, gesture: str, confidence: float):
        self.log_info(f"Gesture detected: {gesture} (confidence: {confidence:.2f})")
        
        if gesture not in self.gesture_statistics:
            self.gesture_statistics[gesture] = {
                'count': 0,
                'total_confidence': 0.0,
                'last_detected': None
            }
        
        self.gesture_statistics[gesture]['count'] += 1
        self.gesture_statistics[gesture]['total_confidence'] += confidence
        self.gesture_statistics[gesture]['last_detected'] = time.time()

    def log_operation_executed(self, operation: str, gesture: str):
        self.total_operations += 1
        self.log_info(f"Operation executed: {operation} (triggered by: {gesture})")

    def log_mouse_move(self, x: int, y: int):
        self.log_debug(f"Mouse moved to: ({x}, {y})")

    def log_screenshot(self, filepath: str):
        self.log_info(f"Screenshot saved: {filepath}")

    def log_error_occurred(self, error: Exception, context: str = ""):
        self.log_error(f"Error occurred {context}: {str(error)}")

    def get_statistics(self) -> Dict[str, dict]:
        stats = {}
        for gesture, data in self.gesture_statistics.items():
            avg_confidence = data['total_confidence'] / data['count'] if data['count'] > 0 else 0.0
            stats[gesture] = {
                'count': data['count'],
                'average_confidence': round(avg_confidence, 2),
                'last_detected': time.ctime(data['last_detected']) if data['last_detected'] else None
            }
        return stats

    def get_total_operations(self) -> int:
        return self.total_operations

    def get_session_duration(self) -> float:
        return time.time() - self.session_start_time

    def print_statistics(self):
        print("\n=== Gesture Control Statistics ===")
        print(f"Session duration: {self.get_session_duration():.2f} seconds")
        print(f"Total operations executed: {self.total_operations}")
        print("\nGesture detection statistics:")
        
        stats = self.get_statistics()
        for gesture, data in stats.items():
            print(f"  {gesture}:")
            print(f"    Count: {data['count']}")
            print(f"    Average confidence: {data['average_confidence']}")
            print(f"    Last detected: {data['last_detected']}")
        
        print("===================================\n")

    def log_statistics(self):
        self.log_info("=== Session Statistics ===")
        self.log_info(f"Session duration: {self.get_session_duration():.2f} seconds")
        self.log_info(f"Total operations executed: {self.total_operations}")
        
        stats = self.get_statistics()
        for gesture, data in stats.items():
            self.log_info(f"Gesture {gesture}: count={data['count']}, avg_confidence={data['average_confidence']}")
        
        self.log_info("============================")

    def reset_statistics(self):
        self.gesture_statistics = {}
        self.total_operations = 0
        self.session_start_time = time.time()
        self.log_info("Statistics reset")

    def close(self):
        self.log_statistics()
        self.log_info("Logger closed")
        self.log_info(f"Session ended at {time.ctime(time.time())}")
        
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
