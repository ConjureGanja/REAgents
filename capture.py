import ctypes
import logging
import threading
import time
from typing import Optional

import cv2
import mss
import numpy as np
import yaml

# Must be called before any mss monitor enumeration.
# Without this, Windows DPI virtualisation causes mss to report scaled-up
# coordinates (e.g. 2400×1350 instead of 1920×1080 on a 125%-DPI monitor),
# which shifts the capture region and produces the half-black frame bug.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    pass

logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    High-speed screen capture using MSS, with a background grabber thread.

    WHY A BACKGROUND THREAD:
      The RL step loop also runs YOLO + OCR + a policy forward pass, so if it
      grabbed the screen inline it would only ever see a frame as often as the
      whole loop completes (~3-4 fps).  Instead, a dedicated thread continuously
      grabs frames into a latest-frame buffer at `capture_fps` (default 30), and
      get_frame() returns that buffer instantly.  This guarantees the agent's
      vision is always fresh (≤ 1/capture_fps old) no matter how slow the rest
      of the step is — the requested "minimum 20 fps vision".

      Analogy: a security guard watching a live monitor (the thread) vs. someone
      who only glances at the camera once they've finished their paperwork.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        game_cfg = self.config["game_settings"]
        monitor_idx = game_cfg.get("monitor_index", 1)
        # Target capture frame rate for the background grabber.  Minimum 20 so
        # the agent's vision is always at least real-time-ish.
        self._target_fps = max(int(game_cfg.get("capture_fps", 30)), 20)

        self.sct = mss.mss()
        if monitor_idx >= len(self.sct.monitors):
            logger.warning("Monitor %d not found — defaulting to Monitor 1.", monitor_idx)
            monitor_idx = 1
        self.monitor = self.sct.monitors[monitor_idx]

        self._latest: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self._blank: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info(
            "Capture initialised on Monitor %d: %s (target %d fps)",
            monitor_idx, self.monitor, self._target_fps,
        )
        self.start()

    # ── Background grabber ────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the background capture thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CaptureThread"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _capture_loop(self) -> None:
        # mss instances are NOT safe to share across threads — create a fresh
        # one bound to this thread.
        local_sct = mss.mss()
        period = 1.0 / self._target_fps
        while self._running:
            t0 = time.time()
            try:
                img = local_sct.grab(self.monitor)
                # ascontiguousarray does two jobs here:
                #   1. [:, :, :3] alone is a VIEW that keeps the whole BGRA
                #      buffer alive (33% wasted RAM per retained frame).
                #   2. OpenCV ops expect contiguous memory; a strided view can
                #      force hidden copies (or errors) downstream.
                # Analogy: tearing the page out of the notebook instead of
                # carrying the whole notebook around to show one page.
                frame = np.ascontiguousarray(np.array(img)[:, :, :3])  # Drop alpha → BGR
                with self._latest_lock:
                    self._latest = frame
            except Exception as exc:
                logger.warning("Capture thread grab failed: %s", exc)
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    # ── Public frame access ───────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray:
        """
        Return the most recent captured frame (BGR NumPy array) instantly.

        Falls back to a direct grab if the background thread hasn't produced a
        frame yet, and to a black frame only if even that fails — so callers
        always receive a valid array.
        """
        with self._latest_lock:
            if self._latest is not None:
                return self._latest.copy()
        # Thread not warmed up yet — grab directly this once.
        try:
            sct_img = self.sct.grab(self.monitor)
            return np.array(sct_img)[:, :, :3]
        except Exception as exc:
            logger.warning("Screen capture failed: %s — returning blank frame.", exc)
            if self._blank is None:
                h = self.monitor.get("height", 1080)
                w = self.monitor.get("width", 1920)
                self._blank = np.zeros((h, w, 3), dtype=np.uint8)
            # Return a COPY — callers (annotate_frame, recorder) draw on frames
            # in place.  Handing out the cached array itself meant one caller's
            # scribbles appeared in every later "blank" frame.
            # Analogy: give visitors a photocopy of the blueprint, not the
            # original — or everyone's pencil marks end up on the master.
            return self._blank.copy()

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
    time.sleep(0.5)  # let the grabber warm up
    cap.test_fps()

    # Show one frame
    frame = cap.get_frame()
    cv2.imshow("Capture Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
