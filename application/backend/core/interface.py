"""
Interface modes: webcam, video file, demo/capture.
Handles different input sources and user interaction.
"""

import cv2
from datetime import datetime
from pathlib import Path
from typing import Callable


class CameraInterface:
    """Handle real-time webcam monitoring."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
    
    def run(self, process_fn: Callable, display_fn: Callable, output_stats_fn: Callable):
        """Run webcam monitoring loop."""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"✗ Cannot open camera {self.camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✓ Camera {self.camera_id} opened")
        print("Press 'q' to quit, 's' to save frame\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                stats = process_fn(frame)
                frame = display_fn(frame)
                
                # Print stats every 30 frames
                if frame_count % 30 == 0:
                    print(f"[Frame {frame_count}] "
                          f"Detections: {stats['total_count']} | "
                          f"Disease: {stats['disease_count']} | "
                          f"Anomaly: {stats['anomaly_pct']:.1f}%")
                
                cv2.imshow('TOKKATOT Edge Monitoring', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"edge_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Frame saved: {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            output_stats_fn()


class VideoInterface:
    """Handle video file processing."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
    
    def run(self, process_fn: Callable, display_fn: Callable, output_stats_fn: Callable):
        """Run video file processing."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"✗ Cannot open video: {self.video_path}")
            return
        
        print(f"✓ Video loaded: {self.video_path}")
        print("Press 'q' to quit, 's' to save frame\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                stats = process_fn(frame)
                frame = display_fn(frame)
                
                # Print stats every 30 frames
                if frame_count % 30 == 0:
                    print(f"[Frame {frame_count}] "
                          f"Detections: {stats['total_count']} | "
                          f"Disease: {stats['disease_count']} | "
                          f"Anomaly: {stats['anomaly_pct']:.1f}%")
                
                cv2.imshow('TOKKATOT Edge Monitoring', frame)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"edge_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Frame saved: {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            output_stats_fn()


class DemoInterface:
    """Demo mode: single image upload & analysis."""
    
    def __init__(self, image_path: str = None):
        """
        Initialize demo interface.
        
        Args:
            image_path: Path to image file to analyze
        """
        self.image_path = image_path
    
    def run(self, process_fn: Callable, display_fn: Callable, print_analysis_fn: Callable):
        """Run demo analysis mode with uploaded image."""
        
        if not self.image_path:
            print("✗ No image path provided")
            return
        
        image_path = Path(self.image_path)
        if not image_path.exists():
            print(f"✗ Image not found: {self.image_path}")
            return
        
        print("\n" + "="*60)
        print("DEMO MODE - Image Upload & Analysis")
        print("="*60)
        print(f"Loading image: {image_path.name}\n")
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"✗ Failed to load image: {self.image_path}")
            return
        
        print("✓ Image loaded successfully")
        print("="*60 + "\n")
        
        try:
            # Process frame
            print("Analyzing image...\n")
            stats = process_fn(frame)
            annotated_frame = display_fn(frame)
            
            # Display results
            cv2.imshow('TOKKATOT Edge - Analysis Results', annotated_frame)
            print_analysis_fn(stats, 1)
            
            print("\nPress any key to close...")
            cv2.waitKey(0)
        
        finally:
            cv2.destroyAllWindows()
