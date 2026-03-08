"""
Frame processing pipeline.
Handles YOLO detection, ensemble classification, tracking, and aggregation.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List

try:
    from models.inference import ChickenDiseaseDetector
except ImportError:
    from ..models.inference import ChickenDiseaseDetector


class FrameProcessor:
    """Process video frames: detect → classify → track → aggregate."""
    
    def __init__(self, detector, yolo, tracker, aggregator, conf_threshold=0.5):
        self.detector = detector
        self.yolo = yolo
        self.tracker = tracker
        self.aggregator = aggregator
        self.conf_threshold = conf_threshold
        
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'disease_detections': 0,
            'healthy_detections': 0,
            'uncertain_detections': 0,
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, List, List, List]:
        """
        Process single frame: detect feces, classify samples, track objects.
        
        Returns:
            (frame_stats, boxes, classes, confidences)
        """
        self.stats['total_frames'] += 1
        
        # YOLO Detection
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)
        detection_boxes = []
        classes = []
        confidences = []
        
        # Process each detection
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = box.conf[0].item()
                
                # Skip tiny boxes
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                
                detection_boxes.append((x1, y1, x2, y2))
                
                # Classify with ensemble
                try:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        pred_class = 'uncertain'
                    else:
                        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        prediction = self.detector.predict(pil_image)
                        pred_class = prediction['class']
                        
                        # Update aggregator
                        is_disease = (pred_class == 'disease')
                        self.aggregator.add_detection(is_disease)
                        
                        # Update stats
                        self.stats['total_detections'] += 1
                        if pred_class == 'disease':
                            self.stats['disease_detections'] += 1
                        elif pred_class == 'healthy':
                            self.stats['healthy_detections'] += 1
                        else:
                            self.stats['uncertain_detections'] += 1
                except Exception:
                    pred_class = 'uncertain'
                    self.stats['uncertain_detections'] += 1
                
                classes.append(pred_class)
                confidences.append(conf)
        
        # Update tracker
        self.tracker.update(detection_boxes, classes, confidences)
        
        # Get anomaly info
        disease_count, total_count, anomaly_pct = self.aggregator.get_anomaly_rate()
        should_alert = self.aggregator.should_alert(anomaly_pct)
        
        frame_stats = {
            'disease_count': disease_count,
            'total_count': total_count,
            'anomaly_pct': anomaly_pct,
            'should_alert': should_alert,
            'healthy_count': self.stats['healthy_detections'],
            'uncertain_count': self.stats['uncertain_detections']
        }
        
        return frame_stats, detection_boxes, classes, confidences
