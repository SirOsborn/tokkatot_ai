"""
Frame processing pipeline.
Handles YOLO detection, ensemble classification, tracking, and aggregation.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List

# Detector is used only if ensemble is needed (optional on edge now)
# Class naming from YOLO perspective:
# 0 = healthy_feces
# 1 = suspicious_feces


class FrameProcessor:
    """Process video frames: detect → classify → track → aggregate."""
    
    def __init__(self, detector, yolo, tracker, aggregator, conf_threshold=0.5, anomaly_threshold=10.0, cloud_service=None):
        self.detector = detector
        self.yolo = yolo
        self.tracker = tracker
        self.aggregator = aggregator
        self.conf_threshold = conf_threshold
        self.anomaly_threshold = anomaly_threshold
        self.cloud_service = cloud_service
        
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
                
                # Process based on YOLO class ID (from the custom trained model)
                # Class 0 = Healthy, Class 1 = Suspicious
                yolo_cls_id = int(box.cls[0].item())
                
                if yolo_cls_id == 0:
                    pred_class = 'healthy'
                else:
                    # Suspicious! Trigger Cloud Verification
                    pred_class = 'suspicious'
                    
                    if self.cloud_service:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            # Run cloud ensemble verification
                            result = self.cloud_service.verify_sample(crop)
                            
                            # If cloud is 100% sure it's healthy, we override
                            if result.get('is_healthy', False) or result.get('classification') == 'Healthy':
                                pred_class = 'healthy'
                            else:
                                # Confirmed disease or unsure
                                pred_class = 'disease'
                                # Update aggregator (only for confirmed diseases)
                                self.aggregator.add_detection(is_disease=True)
                        else:
                            pred_class = 'uncertain'
                    else:
                        pred_class = 'suspicious' # Fallback if no cloud service
                
                detection_boxes.append((x1, y1, x2, y2))
                
                # Update aggregator for healthy results to track total samples
                if pred_class == 'healthy':
                    self.aggregator.add_detection(is_disease=False)
                
                # Update stats
                self.stats['total_detections'] += 1
                if pred_class == 'disease':
                    self.stats['disease_detections'] += 1
                elif pred_class == 'healthy':
                    self.stats['healthy_detections'] += 1
                else:
                    self.stats['uncertain_detections'] += 1
                
                classes.append(pred_class)
                confidences.append(conf)
        
        # Update tracker
        self.tracker.update(detection_boxes, classes, confidences)
        
        # Get anomaly info
        disease_count, total_count, anomaly_pct = self.aggregator.get_anomaly_rate()
        should_alert = self.aggregator.should_alert(self.anomaly_threshold)
        
        frame_stats = {
            'disease_count': disease_count,
            'total_count': total_count,
            'anomaly_pct': anomaly_pct,
            'should_alert': should_alert,
            'healthy_count': self.stats['healthy_detections'],
            'uncertain_count': self.stats['uncertain_detections']
        }
        
        return frame_stats, detection_boxes, classes, confidences
