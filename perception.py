import logging
import cv2
import torch
import numpy as np
import yaml
import easyocr
from ultralytics import YOLO
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class PerceptionSystem:
    """
    The 'Eyes' of the Agent.
    Processes raw frames to extract HUD data and object detections.
    """
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLO(self.config["perception"]["yolo_model"])
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.ocr_regions = self.config["perception"]["ocr_regions"]
        
        # New: Health Circle configuration
        self.health_region = self.config["perception"]["health_circle"]
        self.health_colors = self.config["perception"]["health_colors"]
        
        # Persistence & Smoothing
        self.last_ammo = {"clip": "0", "reserve": "0"}
        self.health_buffer = [] # For smoothing jumps
        self.buffer_size = 5

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detects enemies, items, and doors using YOLO."""
        if frame is None or frame.size == 0:
            return []
        results = self.model(frame, verbose=False)[0]
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

    def get_health_percentage(self, frame: np.ndarray) -> float:
        """
        Calculates health 0.0-1.0 and applies smoothing.
        """
        y1, x1, y2, x2 = self.health_region
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        total_pixels = 0
        for status, (low, high) in self.health_colors.items():
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            total_pixels += cv2.countNonZero(mask)
            
        # Tuned for RE4 Remake circle area
        max_possible_pixels = (y2-y1) * (x2-x1) * 0.15
        if max_possible_pixels <= 0:
            return self.health_buffer[-1] if self.health_buffer else 1.0
        current_val = min(1.0, total_pixels / max_possible_pixels)
        
        # Smoothing logic
        self.health_buffer.append(current_val)
        if len(self.health_buffer) > self.buffer_size:
            self.health_buffer.pop(0)
            
        return sum(self.health_buffer) / len(self.health_buffer)

    def read_hud(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extracts text from screen regions with splitting and persistence."""
        hud_data = {}
        
        # 1. Health
        health_val = self.get_health_percentage(frame)
        hud_data["health_pct"] = health_val
        
        # 2. Ammo (Split into Clip/Reserve)
        y1, x1, y2, x2 = self.ocr_regions["ammo"]
        roi = frame[y1:y2, x1:x2]
        result = self.reader.readtext(roi, detail=0)
        
        if result:
            full_text = " ".join(result).replace(" ", "")
            # Look for common separators or patterns like "10/12"
            if "/" in full_text:
                parts = full_text.split("/")
            else:
                # Fallback: Guess split point if numbers are long
                parts = [full_text[:2], full_text[2:]] if len(full_text) >= 3 else [full_text, "0"]
            
            if len(parts) >= 2:
                self.last_ammo["clip"] = "".join(filter(str.isdigit, parts[0]))
                self.last_ammo["reserve"] = "".join(filter(str.isdigit, parts[1]))
            elif len(parts) == 1:
                self.last_ammo["clip"] = "".join(filter(str.isdigit, parts[0]))

        hud_data["ammo_clip"] = self.last_ammo["clip"]
        hud_data["ammo_res"] = self.last_ammo["reserve"]
            
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
