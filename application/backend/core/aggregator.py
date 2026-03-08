"""
Anomaly Aggregator for Rolling Time Window Detection
Tracks disease detections in a rolling window (default: 5 minutes)
for outbreak detection and alert triggering.
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Tuple


class AnomalyAggregator:
    """
    Track detected anomalies in a rolling time window.
    Maintains statistics for alerting.
    """
    
    def __init__(self, window_size_minutes=5):
        self.window_size = window_size_minutes * 60  # Convert to seconds
        self.detections = deque()  # (timestamp, is_disease)
        self.total_detections = 0
        
    def add_detection(self, is_disease: bool):
        """Add a detection to the anomaly window."""
        self.total_detections += 1
        self.detections.append((datetime.now(), is_disease))
        self._cleanup_old()
        
    def _cleanup_old(self):
        """Remove detections outside the rolling window."""
        cutoff_time = datetime.now() - timedelta(seconds=self.window_size)
        while self.detections and self.detections[0][0] < cutoff_time:
            self.detections.popleft()
    
    def get_anomaly_rate(self) -> Tuple[int, int, float]:
        """
        Get anomaly statistics.
        
        Returns:
            (disease_count, total_count, anomaly_percentage)
        """
        self._cleanup_old()
        if len(self.detections) == 0:
            return 0, 0, 0.0
        
        disease_count = sum(1 for _, is_disease in self.detections if is_disease)
        total_count = len(self.detections)
        anomaly_pct = (disease_count / total_count) * 100
        
        return disease_count, total_count, anomaly_pct
    
    def should_alert(self, threshold_pct: float) -> bool:
        """Check if anomaly rate exceeds threshold."""
        _, _, anomaly_pct = self.get_anomaly_rate()
        return anomaly_pct >= threshold_pct
