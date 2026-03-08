"""
EdgeApp - Coordinator class.
Ties together processor, display, and interface modules.
"""

import torch
from datetime import datetime
from typing import Dict

from ..services.inference import ChickenDiseaseDetector

from ultralytics import YOLO
from .tracker import CentroidTracker
from .aggregator import AnomalyAggregator
from .processor import FrameProcessor
from .display import FrameDisplay
from .interface import CameraInterface, VideoInterface, DemoInterface


class EdgeApp:
    """Edge application coordinator."""
    
    def __init__(
        self,
        ensemble_model_path: str = "outputs/ensemble_model.pth",
        yolo_model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        anomaly_threshold_pct: float = 10.0,
        device: str = 'auto'
    ):
        """Initialize with models and defaults."""
        # Device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print("\n" + "=" * 60)
        print("TOKKATOT EDGE APPLICATION - INITIALIZING")
        print("=" * 60)
        print(f"Device: {self.device.upper()}")
        
        # Load models
        print("\n[1/3] Loading Ensemble Model...")
        self.detector = ChickenDiseaseDetector(
            model_path=ensemble_model_path,
            device=self.device
        )
        
        print("\n[2/3] Loading YOLO Model...")
        self.yolo = YOLO(yolo_model_path)
        print("✓ YOLO loaded!")
        
        print("\n[3/3] Initializing Tracking & Aggregation...")
        self.tracker = CentroidTracker(maxDisappeared=30)
        self.aggregator = AnomalyAggregator(window_size_minutes=5)
        print("✓ Ready!")
        
        # Processor and display
        self.processor = FrameProcessor(
            self.detector, self.yolo, self.tracker, 
            self.aggregator, conf_threshold
        )
        self.display = FrameDisplay(anomaly_threshold_pct)
        self.anomaly_threshold = anomaly_threshold_pct
        
        print("\n" + "=" * 60)
        print("Ready to start monitoring!")
        print("=" * 60 + "\n")
    
    def _process_and_display(self, frame):
        """Process frame and add display."""
        stats, boxes, classes, confs = self.processor.process_frame(frame)
        self.display.total_frames = self.processor.stats['total_frames']
        
        frame = self.display.update_frame(
            frame, boxes, classes, confs,
            stats['disease_count'], stats['total_count'],
            stats['anomaly_pct'], stats['should_alert']
        )
        
        return frame, stats
    
    def run_webcam(self, camera_id: int = 0):
        """Run webcam monitoring."""
        interface = CameraInterface(camera_id)
        interface.run(
            lambda f: self._process_and_display(f)[1],
            lambda f: self._process_and_display(f)[0],
            self._print_final_stats
        )
        self._print_final_stats()
    
    def run_video_file(self, video_path: str):
        """Run video file processing."""
        interface = VideoInterface(video_path)
        interface.run(
            lambda f: self._process_and_display(f)[1],
            lambda f: self._process_and_display(f)[0],
            self._print_final_stats
        )
        self._print_final_stats()
    
    def capture_and_analyze(self, image_path: str = None):
        """Run demo analysis mode with uploaded image."""
        interface = DemoInterface(image_path)
        interface.run(
            lambda f: self._process_and_display(f)[1],
            lambda f: self._process_and_display(f)[0],
            self._print_demo_analysis
        )
    
    def _print_demo_analysis(self, stats: Dict, capture_number: int):
        """Print demo analysis."""
        print("\n" + "="*60)
        print(f"CAPTURE #{capture_number} - RESULTS")
        print("="*60)
        print(f"Total Detections: {stats['total_count']}")
        print("-"*60)
        
        if stats['total_count'] > 0:
            print(f"Healthy: {stats['healthy_count']}")
            print(f"Disease: {stats['disease_count']}")
            print(f"Uncertain: {stats['uncertain_count']}")
            print("-"*60)
            print(f"Disease Rate: {(stats['disease_count']/stats['total_count'])*100:.1f}%")
            print(f"Anomaly: {stats['anomaly_pct']:.1f}%")
            print(f"Threshold: {self.anomaly_threshold:.1f}%")
            print("-"*60)
            
            if stats['should_alert']:
                print(f"⚠️  ALERT TRIGGERED!")
            else:
                print(f"✓ SAFE")
        else:
            print("No fecal samples detected")
        
        print("="*60 + "\n")
    
    def _print_final_stats(self):
        """Print session statistics."""
        elapsed = (datetime.now() - self.display.start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("SESSION STATISTICS")
        print("=" * 60)
        print(f"Duration: {elapsed:.1f}s")
        print(f"Frames: {self.processor.stats['total_frames']}")
        print(f"FPS: {self.processor.stats['total_frames'] / max(1, elapsed):.1f}")
        print(f"Detections: {self.processor.stats['total_detections']}")
        print(f"  - Healthy: {self.processor.stats['healthy_detections']}")
        print(f"  - Disease: {self.processor.stats['disease_detections']}")
        print(f"  - Uncertain: {self.processor.stats['uncertain_detections']}")
        
        if self.processor.stats['total_detections'] > 0:
            rate = (self.processor.stats['disease_detections'] / 
                   self.processor.stats['total_detections']) * 100
            print(f"Disease Rate: {rate:.1f}%")
        
        print("=" * 60 + "\n")
