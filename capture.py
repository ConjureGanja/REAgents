import logging
import mss
import numpy as np
import cv2
import yaml
import time
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class ScreenCapture:
    """
    High-speed screen capture using MSS.
    Targets 60+ FPS with minimal CPU overhead.
    """
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.sct = mss.mss()
        monitor_idx = self.config["game_settings"].get("monitor_index", 1)

        # Validate monitor index
        if monitor_idx >= len(self.sct.monitors):
            logger.warning("Monitor %d not found — defaulting to Monitor 1.", monitor_idx)
            monitor_idx = 1

        self.monitor = self.sct.monitors[monitor_idx]
        self._blank: Optional[np.ndarray] = None  # lazily created fallback frame
        logger.info("Capture initialised on Monitor %d: %s", monitor_idx, self.monitor)

    def get_frame(self) -> np.ndarray:
        """
        Captures the screen region and returns a BGR NumPy array.
        Returns a black frame on failure so callers always receive a valid array.
        """
        try:
            sct_img = self.sct.grab(self.monitor)
            frame = np.array(sct_img)
            return frame[:, :, :3]  # Drop alpha → BGR
        except Exception as exc:
            logger.warning("Screen capture failed: %s — returning blank frame.", exc)
            if self._blank is None:
                h = self.monitor.get("height", 1080)
                w = self.monitor.get("width", 1920)
                self._blank = np.zeros((h, w, 3), dtype=np.uint8)
            return self._blank

    def test_fps(self, duration: int = 5):
        """Prints the achieved FPS over a given duration."""
        print(f"Testing capture speed for {duration} seconds...")
        start_time = time.time()
        frames = 0
        while time.time() - start_time < duration:
            _ = self.get_frame()
            frames += 1
        
        fps = frames / duration
        print(f"Captured {frames} frames in {duration}s. Average FPS: {fps:.2f}")
        return fps

if __name__ == "__main__":
    # Quick test if run directly
    cap = ScreenCapture()
    cap.test_fps()
    
    # Show one frame
    frame = cap.get_frame()
    cv2.imshow("Capture Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
