"""
Cloud Sync Service - Sends triggered image crops to the cloud for ensemble verification.
"""

import cv2
import requests
import json
import numpy as np
from typing import Dict, Optional
from pathlib import Path

class CloudSyncService:
    """Handles communication with the Tokkatot Cloud API."""
    
    def __init__(self, api_url: str = "http://api.tokkatot.com/api/v1", api_key: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }

    def verify_sample(self, image: np.ndarray) -> Dict:
        """
        Send an image sample to the cloud for ensemble verification.
        
        Args:
            image: Image crop as numpy array (BGR)
            
        Returns:
            Dictionary containing result from cloud
        """
        # Encode image to JPG
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        
        try:
            files = {'file': ('sample.jpg', img_bytes, 'image/jpeg')}
            response = requests.post(
                f"{self.api_url}/verify",
                files=files,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Cloud API returned status {response.status_code}",
                    "is_healthy": False, # Assume not healthy for safety
                    "classification": "CLOUD_ERROR",
                    "should_isolate": True
                }
        except Exception as e:
            return {
                "error": f"Connection failed: {str(e)}",
                "is_healthy": False,
                "classification": "CONNECTION_ERROR",
                "should_isolate": True
            }

    def send_metrics(self, stats: Dict) -> bool:
        """Send session metrics to the cloud dashboard."""
        try:
            response = requests.post(
                f"{self.api_url}/metrics",
                json=stats,
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
