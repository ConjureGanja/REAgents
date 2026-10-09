import logging
import cv2
import torch
import numpy as np
import yaml
import easyocr
from ultralytics import YOLO
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class PerceptionSystem:
    """
    The 'Eyes' of the Agent.
    Processes raw frames to extract HUD data and object detections.
    """
    def __init__(self, config_path: str = "config.yaml", config: Optional[Dict[str, Any]] = None):
        from config_loader import ensure_config
        self.config = ensure_config(config, config_path)

        perc = self.config["perception"]
        self.model = YOLO(perc["yolo_model"])
        self.conf_threshold = float(perc.get("confidence_threshold", 0.40))

        # Inference device / precision — default to CUDA + FP16 when available.
        self._device = perc.get("yolo_device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._half   = bool(perc.get("yolo_half", True)) and self._device == "cuda"
        self._imgsz  = int(perc.get("yolo_imgsz", 640))

        # GPU sanity check — warn loudly if CUDA exists but isn't being used.
        logger.info(
            "YOLO device=%s half=%s imgsz=%d (cuda_available=%s, torch=%s)",
            self._device, self._half, self._imgsz,
            torch.cuda.is_available(), torch.__version__,
        )
        if torch.cuda.is_available() and self._device != "cuda":
            logger.warning("CUDA is available but YOLO is configured for %s!", self._device)

        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        logger.info("EasyOCR gpu=%s", torch.cuda.is_available())
        self.ocr_regions = perc["ocr_regions"]

        # Warmup: the first YOLO inference costs ~16 s (model load + CUDA kernel
        # init).  Burn it here at startup instead of inside the first RL step.
        try:
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, imgsz=self._imgsz,
                       device=self._device, half=self._half)
            logger.info("YOLO warmup inference complete.")
        except Exception as exc:
            logger.warning("YOLO warmup failed (non-fatal): %s", exc)

        # ── GPU sanity check (T002 / plan P1.2) ──────────────────────────────
        # A silent CPU fallback is the single most common reason perception
        # crawls: YOLO and EasyOCR both "work" on CPU, just 5-10× slower, and
        # nothing in the default logs tells you it happened.  Check once at
        # startup and warn loudly so the fix (install CUDA torch) is obvious.
        _cuda = torch.cuda.is_available()
        _yolo_dev = str(getattr(self.model, "device", "unknown"))
        _ocr_dev = str(getattr(self.reader, "device", "unknown"))
        logger.info("Perception devices — torch.cuda=%s, YOLO=%s, EasyOCR=%s",
                    _cuda, _yolo_dev, _ocr_dev)
        if not _cuda:
            logger.warning(
                "CUDA is NOT available to PyTorch — YOLO/EasyOCR will run on CPU "
                "(~5-10× slower). Install GPU torch first: pip install torch "
                "torchvision --index-url https://download.pytorch.org/whl/cu121"
            )

        # New: Health Circle configuration
        self.health_region = self.config["perception"]["health_circle"]
        self.health_colors = self.config["perception"]["health_colors"]
        # Below this fraction of the full-health pixel count the ring is treated
        # as NOT VISIBLE (returns None) rather than "nearly dead".  Stray warm
        # pixels (torches, fires, sunlit walls) leak a few dozen pixels through
        # the colour mask when Leon isn't aiming — reading those as 1-5% health
        # was the root of the phantom-death reset loop.
        self.min_visible_frac = float(
            self.config["perception"].get("health_min_visible_frac", 0.08)
        )

        # Persistence & Smoothing
        self.last_ammo = {"clip": "0", "reserve": "0"}
        self.health_buffer = [] # For smoothing jumps
        self.buffer_size = 5
        self._last_ammo_fp = None   # ROI fingerprint for skip-unchanged OCR

    def reset_state(self) -> None:
        """Clear per-episode buffers.  Called by the env on reset so stale
        low-health readings from the previous episode can't bleed into the
        new one and re-trigger a death."""
        self.health_buffer.clear()
        self.last_ammo = {"clip": "0", "reserve": "0"}

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detects enemies, items, and doors using YOLO."""
        if frame is None or frame.size == 0:
            return []
        # imgsz letterboxes the 1080p frame down for the network — the single
        # biggest YOLO speed win (inference cost scales with pixel count).
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            imgsz=self._imgsz,
            device=self._device,
            half=self._half,
        )[0]
        detections = []

        for box in results.boxes:
            # Extract coordinates and class info
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = results.names[cls_id]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        return detections

    def get_health_percentage(self, frame: np.ndarray) -> Optional[float]:
        """
        Calculates health 0.0-1.0 and applies smoothing.

        Returns None when the HUD ring is NOT VISIBLE (Leon not aiming, menu,
        cutscene, loading screen).  The RE4R health radial only renders while
        aiming, so most frames legitimately have no ring — a handful of warm
        background pixels used to be misread as "1-5% health" and trip phantom
        deaths.  Invalid frames are NOT pushed into the smoothing buffer, so
        they can neither drag the average down nor decay it through the
        low-health zone.
        """
        y1, x1, y2, x2 = self.health_region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        total_pixels = 0
        for status, (low, high) in self.health_colors.items():
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            total_pixels += cv2.countNonZero(mask)

        # Normalise by the colored-pixel count at FULL health (config-tuned),
        # not the box area — the health ring is a thin arc, so area-based scaling
        # wildly under-read.  health_full_px is measured from a full-health frame.
        max_possible_pixels = float(self.config["perception"].get("health_full_px", 1300))
        if max_possible_pixels <= 0:
            return self.health_buffer[-1] if self.health_buffer else None

        # Visibility floor: below this the ring isn't on screen at all.
        if total_pixels < self.min_visible_frac * max_possible_pixels:
            return None

        current_val = min(1.0, total_pixels / max_possible_pixels)

        # Smoothing logic (valid readings only)
        self.health_buffer.append(current_val)
        if len(self.health_buffer) > self.buffer_size:
            self.health_buffer.pop(0)

        return sum(self.health_buffer) / len(self.health_buffer)

    def detect_death_screen(self, frame: np.ndarray) -> bool:
        """
        Heuristic detector for the RE4R "YOU ARE DEAD" screen: a mostly-black
        centre with a modest patch of saturated red lettering.

        This is the PRIMARY death signal — at the moment of death the HUD ring
        vanishes, so health-based detection is structurally unreliable.  The
        env requires several consecutive positive reads before terminating, so
        a single dark-and-red cutscene frame can't trigger a reset.
        """
        if frame is None or frame.size == 0:
            return False
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.25):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
        if crop.size == 0:
            return False
        # Downscale for speed — colour statistics survive resizing.
        crop = cv2.resize(crop, (192, 96), interpolation=cv2.INTER_AREA)
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        dark_frac = float(np.mean(vch < 40))
        red_mask  = ((hch <= 10) | (hch >= 170)) & (sch >= 100) & (vch >= 60)
        red_frac  = float(np.mean(red_mask))

        return dark_frac > 0.70 and 0.002 <= red_frac <= 0.15

    def read_hud(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extracts text from screen regions with splitting and persistence.

        health_pct may be None — meaning "HUD ring not visible this frame".
        Callers must treat None as unknown, NOT as zero health.
        """
        hud_data = {}

        # 1. Health (None = ring not visible)
        health_val = self.get_health_percentage(frame)
        hud_data["health_pct"] = health_val

        # 2. Ammo (Split into Clip/Reserve)
        y1, x1, y2, x2 = self.ocr_regions["ammo"]
        roi = frame[y1:y2, x1:x2]

        # Skip OCR entirely when the region is pixel-identical to the last read
        # (ammo only changes on shots/pickups — most ticks would re-read the
        # same digits).  Cheap shape+sum fingerprint is enough here.
        roi_fp = (roi.shape, int(roi.sum())) if roi.size else None
        if roi_fp is not None and roi_fp == self._last_ammo_fp:
            result = None   # unchanged — keep cached last_ammo values
        else:
            self._last_ammo_fp = roi_fp
            # Preprocess: grayscale → 2× upscale → Otsu threshold.  The ammo
            # counter is white text on a dark HUD; binarising removes antialias
            # noise that used to produce "l|" style garbage reads.
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # allowlist constrains recognition to digits + the clip/reserve
            # separator — both faster and immune to letter-shaped artifacts.
            result = self.reader.readtext(binary, detail=0, allowlist="0123456789/")

        if result:
            full_text = " ".join(result).replace(" ", "")
            # Look for common separators or patterns like "10/12"
            if "/" in full_text:
                parts = full_text.split("/")
            else:
                # Fallback: Guess split point if numbers are long
                parts = [full_text[:2], full_text[2:]] if len(full_text) >= 3 else [full_text, "0"]

            # Only OVERWRITE the cached value when the OCR read actually
            # contained digits.  Previously a garbled read (e.g. "l|" → "")
            # stored an empty string, wiping the last good count until the next
            # clean read.  Analogy: if the scoreboard flickers, keep showing the
            # last score you saw — don't display a blank.
            clip_digits = "".join(filter(str.isdigit, parts[0]))
            if clip_digits:
                self.last_ammo["clip"] = clip_digits
            if len(parts) >= 2:
                res_digits = "".join(filter(str.isdigit, parts[1]))
                if res_digits:
                    self.last_ammo["reserve"] = res_digits

        # Return INTs, not strings.  Every consumer (environment.py reward math,
        # the LLM prompt, memory.py logging) expects numbers; returning "12"
        # worked only because callers defensively wrapped values in int(... or 0).
        # Typed boundaries beat defensive casts scattered across the codebase.
        hud_data["ammo_clip"] = int(self.last_ammo["clip"] or 0)
        hud_data["ammo_res"]  = int(self.last_ammo["reserve"] or 0)

        return hud_data

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict], hud: Dict) -> np.ndarray:
        """Draws bounding boxes, HUD data, and search regions for the user."""
        annotated = frame.copy()

        # 1. Draw HUD Search Regions (Calibration Guide)
        for key, region in [("AMMO_BOX", self.ocr_regions["ammo"]), ("HEALTH_BOX", self.health_region)]:
            y1, x1, y2, x2 = region
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1) # Blue thin line
            cv2.putText(annotated, key, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # 2. Draw Detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{det['label']} {det['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw HUD info on top left
        y_offset = 30
        for key, val in hud.items():
            cv2.putText(annotated, f"{key.upper()}: {val}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

        return annotated

if __name__ == "__main__":
    # Test script - assumes capture.py is working
    from capture import ScreenCapture
    import time

    cap = ScreenCapture()
    eyes = PerceptionSystem()

    print("Perception System Live. Press 'q' to quit.")

    while True:
        frame = cap.get_frame()

        # Process Frame
        t1 = time.time()
        detections = eyes.detect_objects(frame)
        hud = eyes.read_hud(frame)
        latency = (time.time() - t1) * 1000

        # Display
        display_frame = eyes.annotate_frame(frame, detections, hud)
        cv2.putText(display_frame, f"Latency: {latency:.1f}ms", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Agent Perception View", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
