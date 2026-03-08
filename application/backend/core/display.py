"""
Display and visualization module.
Handles drawing on frames, overlays, stats panels, alerts.
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Dict


class FrameDisplay:
    """Draw bounding boxes, stats, and alerts on frames."""
    
    def __init__(self, anomaly_threshold: float):
        self.anomaly_threshold = anomaly_threshold
        self.start_time = datetime.now()
        self.total_frames = 0
    
    def update_frame(
        self,
        frame: np.ndarray,
        boxes: list,
        classes: list,
        confidences: list,
        disease_count: int,
        total_count: int,
        anomaly_pct: float,
        should_alert: bool
    ) -> np.ndarray:
        """Draw detections and stats on frame."""
        h, w = frame.shape[:2]
        
        # Draw detections
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if i < len(classes):
                cls = classes[i]
                conf = confidences[i] if i < len(confidences) else 0.0
                
                # Color by class
                color_map = {
                    'disease': (0, 0, 255),    # Red
                    'healthy': (0, 255, 0),   # Green
                    'uncertain': (0, 255, 255)  # Yellow
                }
                color = color_map.get(cls, (0, 255, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw stats panel
        self._draw_stats(frame, disease_count, total_count, anomaly_pct, should_alert)
        
        return frame
    
    def _draw_stats(
        self,
        frame: np.ndarray,
        disease_count: int,
        total_count: int,
        anomaly_pct: float,
        should_alert: bool
    ):
        """Draw statistics panel and alert banner."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Title
        cv2.putText(frame, "TOKKATOT EDGE MONITOR", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Stats
        y_offset = 60
        elapsed = (datetime.now() - self.start_time).total_seconds()
        fps = self.total_frames / max(1, elapsed)
        
        stats_text = [
            f"Total Detections: {total_count}",
            f"Disease Markers: {disease_count}",
            f"Anomaly Rate: {anomaly_pct:.1f}%",
            f"Threshold: {self.anomaly_threshold:.1f}%",
            f"FPS: {fps:.1f}",
        ]
        
        for text in stats_text:
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Alert banner
        if should_alert:
            color = (0, 0, 255)  # Red
            text = f"⚠ ALERT: {anomaly_pct:.1f}% > {self.anomaly_threshold}%"
            cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), color, -1)
            cv2.putText(frame, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            color = (0, 255, 0)  # Green
            text = "✓ SAFE: All systems normal"
            cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), color, -1)
            cv2.putText(frame, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
