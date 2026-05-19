import pyautogui
import numpy as np
import os
import time
import sys
from typing import Optional, Tuple
from config import (
    SMOOTHING_ALPHA, MOUSE_MOVE_SCALE, MOUSE_SCREEN_MARGIN, SCREENSHOT_DIR
)


try:
    from PIL import ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not available, screenshot feature will be limited")

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

class MouseController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        
        self.smooth_x = self.screen_width / 2
        self.smooth_y = self.screen_height / 2
        
        self.last_mouse_x = self.screen_width / 2
        self.last_mouse_y = self.screen_height / 2
        
        self.is_moving = False
        self.movement_threshold = 0.01

        self.last_screenshot_time = 0
        self.screenshot_interval = 1.0

        # three-finger scroll tracking
        self.last_scroll_y = None
        self.last_scroll_time = 0.0

    def _apply_smoothing(self, target_x: float, target_y: float) -> Tuple[float, float]:
        self.smooth_x = SMOOTHING_ALPHA * target_x + (1 - SMOOTHING_ALPHA) * self.smooth_x
        self.smooth_y = SMOOTHING_ALPHA * target_y + (1 - SMOOTHING_ALPHA) * self.smooth_y
        
        return self.smooth_x, self.smooth_y

    def move_mouse(self, normalized_x: float, normalized_y: float, frame_width: int, frame_height: int):
        usable_width = max(1, self.screen_width - 2 * MOUSE_SCREEN_MARGIN)
        usable_height = max(1, self.screen_height - 2 * MOUSE_SCREEN_MARGIN)

        target_x = normalized_x * usable_width + MOUSE_SCREEN_MARGIN
        target_y = normalized_y * usable_height + MOUSE_SCREEN_MARGIN

        center_x = self.screen_width / 2
        center_y = self.screen_height / 2

        target_x = (target_x - center_x) * MOUSE_MOVE_SCALE + center_x
        target_y = (target_y - center_y) * MOUSE_MOVE_SCALE + center_y

        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))
        
        smooth_x, smooth_y = self._apply_smoothing(target_x, target_y)
        
        dx = abs(smooth_x - self.last_mouse_x)
        dy = abs(smooth_y - self.last_mouse_y)
        
        if dx > self.movement_threshold or dy > self.movement_threshold:
            try:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
                self.last_mouse_x = smooth_x
                self.last_mouse_y = smooth_y
                self.is_moving = True
            except Exception as e:
                print(f"Mouse move error: {e}")
        else:
            self.is_moving = False

    def click(self, button: str = 'left'):
        try:
            pyautogui.click(button=button)
            print(f"Mouse click: {button}")
        except Exception as e:
            print(f"Mouse click error: {e}")

    def double_click(self):
        try:
            pyautogui.doubleClick()
            print("Mouse double click")
        except Exception as e:
            print(f"Double click error: {e}")

    def play_media(self):
        try:
            pyautogui.press('playpause')
            print("Play media")
        except Exception as e:
            pyautogui.press('space')
            print("Play media (space fallback)")

    def pause_media(self):
        try:
            pyautogui.press('playpause')
            print("Pause media")
        except Exception as e:
            pyautogui.press('space')
            print("Pause media (space fallback)")

    def next_page(self):
        try:
            pyautogui.press('pagedown')
            print("Next page (PageDown)")
        except Exception as e:
            print(f"Next page error: {e}")

    def previous_page(self):
        try:
            pyautogui.press('pageup')
            print("Previous page (PageUp)")
        except Exception as e:
            print(f"Previous page error: {e}")

    def take_screenshot(self) -> Optional[str]:
        current_time = time.time()
        if current_time - self.last_screenshot_time < self.screenshot_interval:
            print("Screenshot skipped (cooldown)")
            return None

    def _get_active_window_title(self) -> str:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value
        except Exception:
            return ""

    def scroll(self, landmarks, threshold: float = None, amount: int = None):
        # landmarks: list of normalized (x,y,z); use index(8), middle(12), ring(16)
        from config import THREE_FINGER_SCROLL_THRESHOLD, MOUSE_SCROLL_AMOUNT
        if threshold is None:
            threshold = THREE_FINGER_SCROLL_THRESHOLD
        if amount is None:
            amount = MOUSE_SCROLL_AMOUNT

        title = self._get_active_window_title().lower()
        # avoid scrolling in PowerPoint / slideshow viewers
        if any(k in title for k in ['powerpoint', 'ppt', 'slide show', 'pptx']):
            return

        if landmarks is None or len(landmarks) < 17:
            self.last_scroll_y = None
            return

        y8 = landmarks[8][1]
        y12 = landmarks[12][1]
        y16 = landmarks[16][1]
        cur_y = float((y8 + y12 + y16) / 3.0)

        current_time = time.time()
        if self.last_scroll_y is None:
            self.last_scroll_y = cur_y
            self.last_scroll_time = current_time
            return

        dy = cur_y - self.last_scroll_y
        if abs(dy) < threshold:
            # not enough movement
            return

        # enforce cooldown
        from config import COOLDOWN_DURATION
        cooldown = COOLDOWN_DURATION.get('scroll', 0.1)
        if current_time - self.last_scroll_time < cooldown:
            return

        # moving down (cur_y > last) -> three-finger down -> content should move up -> positive scroll
        try:
            scroll_amount = int(max(1, round(abs(dy) / threshold * amount)))
            if dy > 0:
                pyautogui.scroll(scroll_amount)
            else:
                pyautogui.scroll(-scroll_amount)
            self.last_scroll_time = current_time
            self.last_scroll_y = cur_y
            print(f"Three-finger scroll: dy={dy:.4f}, amount={scroll_amount}")
        except Exception as e:
            print(f"Scroll error: {e}")
        
        if not HAS_PIL:
            print("Screenshot failed: PIL/ImageGrab not available")
            return None
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(SCREENSHOT_DIR, filename)
            
            screenshot = ImageGrab.grab()
            screenshot.save(filepath)
            
            self.last_screenshot_time = current_time
            print(f"Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"Screenshot error: {e}")
            return None



    def get_mouse_position(self) -> Tuple[int, int]:
        try:
            return pyautogui.position()
        except Exception as e:
            print(f"Get mouse position error: {e}")
            return (0, 0)

    def is_mouse_moving(self) -> bool:
        return self.is_moving

    def reset_smoothing(self):
        self.smooth_x, self.smooth_y = pyautogui.position()
        self.last_mouse_x, self.last_mouse_y = self.smooth_x, self.smooth_y

    def move_relative(self, dx: int, dy: int):
        try:
            pyautogui.moveRel(dx, dy)
            print(f"Move relative: ({dx}, {dy})")
        except Exception as e:
            print(f"Move relative error: {e}")
