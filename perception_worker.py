"""
Dedicated perception worker thread.

WHY THIS EXISTS:
  Previously the RL step loop ran YOLO + EasyOCR + health-mask inline every
  perception tick (~40-150 ms), throttled to `perception_fps`.  That meant the
  policy's decision rate dipped every time perception re-ran.  Now a dedicated
  thread owns the PerceptionSystem and publishes results into SharedState:
  the step loop just adopts the newest snapshot and never blocks on vision.

  Analogy: the capture thread is the camera operator; this worker is the
  analyst in the back room annotating the footage — the pilot (RL loop) only
  ever looks at the latest annotated report.

THREAD SAFETY:
  YOLO (ultralytics) and EasyOCR are NOT thread-safe.  Every call into the
  PerceptionSystem funnels through this worker's lock — the background loop
  AND the synchronous read_sync() path used by env.reset() / startup fallback.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PerceptionWorker:
    """
    Runs YOLO + OCR + health + death-screen detection on a background thread
    at `fps`, publishing results to SharedState.

    Published fields (all written atomically via one update() call):
      detections     — list of YOLO boxes
      hud            — parsed HUD dict (health_pct may be None)
      death_screen   — True when the "YOU ARE DEAD" screen is detected
      perception_at  — time.time() of this publish (freshness marker)
    """

    def __init__(self, eyes, cap, shared, fps: float = 10.0):
        self._eyes     = eyes
        self._cap      = cap
        self._shared   = shared
        self._interval = 1.0 / max(float(fps), 1.0)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="PerceptionWorker"
        )
        self._thread.start()
        logger.info("PerceptionWorker started (%.1f Hz).", 1.0 / self._interval)

    def stop(self) -> None:
        self._running = False

    # ── Synchronous path (reset seeding, startup fallback) ────────────────────

    def read_sync(
        self, frame: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict], Dict[str, Any], bool]:
        """
        Run one full perception pass NOW, serialized with the background loop.

        Returns (detections, hud, death_screen).  Used by env.reset() to seed
        the caches and by step() as a fallback before the worker's first
        publish.  All PerceptionSystem access must go through this lock.
        """
        with self._lock:
            if frame is None:
                frame = self._cap.get_frame()
            return self._perceive(frame)

    # ── Background loop ───────────────────────────────────────────────────────

    def _perceive(self, frame: np.ndarray) -> Tuple[List[Dict], Dict[str, Any], bool]:
        detections = self._eyes.detect_objects(frame)
        hud        = self._eyes.read_hud(frame)
        death      = (
            self._eyes.detect_death_screen(frame)
            if hasattr(self._eyes, "detect_death_screen") else False
        )
        return detections, hud, death

    def _loop(self) -> None:
        while self._running and not self._shared.stop_requested:
            t0 = time.time()
            try:
                with self._lock:
                    detections, hud, death = self._perceive(self._cap.get_frame())
                self._shared.update(
                    detections=detections,
                    hud=hud,
                    death_screen=death,
                    perception_at=time.time(),
                )
            except Exception as exc:
                logger.warning("PerceptionWorker tick failed: %s", exc)
            dt = time.time() - t0
            if dt < self._interval:
                time.sleep(self._interval - dt)
        logger.info("PerceptionWorker stopped.")
