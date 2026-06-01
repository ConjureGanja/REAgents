# recorder.py
import cv2
import threading
import time
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class GameplayRecorder:
    """
    Saves gameplay as .mp4 files in the background.
    Pulls frames from SharedState so the training loop is never blocked.
    
    Usage:
        recorder = GameplayRecorder(shared, output_dir="footage")
        recorder.start()          # begin recording
        ...
        recorder.stop()           # flush and close the file
    """

    def __init__(
        self,
        shared,                       # SharedState
        output_dir: str = "footage",
        fps: int = 20,                # lower than game FPS is fine — saves disk
        width: int = 1920,
        height: int = 1080,
        max_minutes_per_file: int = 10,  # auto-split so files stay manageable
    ):
        self._shared = shared
        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._size = (width, height)
        self._max_frames = fps * 60 * max_minutes_per_file

        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_recording(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_recording:
            logger.warning("Recorder already running — ignoring start().")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._record_loop, daemon=True, name="RecorderThread"
        )
        self._thread.start()
        logger.info("Gameplay recorder started → %s", self._out_dir)

    def stop(self) -> None:
        if not self.is_recording:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _new_writer(self) -> cv2.VideoWriter:
        filename = self._out_dir / f"ep_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filename), fourcc, self._fps, self._size)
        logger.info("Recording → %s", filename)
        return writer

    def _record_loop(self) -> None:
        interval = 1.0 / self._fps
        writer = self._new_writer()
        frame_count = 0

        while self._running:
            t0 = time.monotonic()

            frame = self._shared.frame
            if frame is not None:
                # Resize to target in case capture res differs
                if frame.shape[1] != self._size[0] or frame.shape[0] != self._size[1]:
                    frame = cv2.resize(frame, self._size)
                writer.write(frame)
                frame_count += 1

            # Auto-split files
            if frame_count >= self._max_frames:
                writer.release()
                writer = self._new_writer()
                frame_count = 0

            elapsed = time.monotonic() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

        writer.release()
        logger.info("Gameplay recorder stopped.")