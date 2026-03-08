# 🎯 Services - Business Logic

**Higher-level functionality above core ML pipeline**

---

## 📂 Key Services

### inference.py - ML Inference Wrapper

Wraps the ensemble model for easy inference:

```python
from backend.services.inference import ChickenDiseaseDetector

detector = ChickenDiseaseDetector("ensemble_model.pth", device="cuda")

# Single image
disease, confidence = detector.predict(image)

# Batch inference
results = detector.predict_batch(images)

# Get model info
info = detector.get_model_info()
# {"name": "Ensemble", "accuracy": 0.97, "classes": [...]}
```

### metrics.py - Metrics Collection

Aggregates and tracks system metrics:

```python
from backend.services.metrics import MetricsCollector

metrics = MetricsCollector()

# Record each detection
metrics.record_detection(
    disease_type="Coccidiosis",
    confidence=0.96,
    processing_time_ms=45.2
)

# Get current statistics
stats = metrics.get_stats()
# {
#   "fps": 22.3,
#   "total_detections": 457,
#   "disease_count": 23,
#   "anomaly_rate": 5.04,
#   "avg_confidence": 0.94,
#   "uptime_seconds": 86400
# }

# Send to cloud
metrics.send_to_cloud()
```

### alerts.py - Alert Management

Generates alerts based on thresholds:

```python
from backend.services.alerts import AlertManager

alerts = AlertManager(
    anomaly_threshold_pct=10,
    confidence_threshold=0.7
)

# Check conditions and generate alerts
alert = alerts.check_status(metrics)

if alert:
    print(f"Alert: {alert.type} - {alert.message}")
    # Could email, SMS, or trigger webhook
```

### health_check.py - System Health

Monitors overall system health:

```python
from backend.services.health_check import HealthChecker

health = HealthChecker()

status = health.check_all()
# {
#   "models_loaded": true,
#   "camera_available": true,
#   "disk_space_gb": 5.2,
#   "memory_available_mb": 420,
#   "cpu_percent": 65,
#   "temperature_celsius": 52,
#   "network_connected": true
# }

if not status["models_loaded"]:
    raise RuntimeError("Models failed to load!")
```

---

## 🔄 Integration with Core

Services sit between API layer and core ML pipeline:

```
API Request
    ↓
Service Layer (validates, augments data)
    ↓
Core ML Pipeline
    ↓
Service Layer (formats result, records metrics)
    ↓
API Response
```

Example:

```python
# From API endpoint
from backend.services.inference import ChickenDiseaseDetector
from backend.services.metrics import MetricsCollector

detector = ChickenDiseaseDetector()
metrics_collector = MetricsCollector()

# Receive image from API
image = request.files["image"]

# Use inference service
start_time = time.time()
disease, confidence = detector.predict(image)
latency_ms = (time.time() - start_time) * 1000

# Record metrics
metrics_collector.record_detection(disease, confidence, latency_ms)

# Return result
return {
    "disease": disease,
    "confidence": confidence.item(),
    "processing_time_ms": latency_ms,
    "anomaly_rate": metrics_collector.get_anomaly_rate()
}
```

---

## 📊 Service Architecture

```
┌─────────────────────────────────┐
│        API Layer                │
│   (routes.py, handlers.py)      │
└──────────────┬──────────────────┘
               │
      ┌────────┼────────┐
      │                 │
      ↓                 ↓
┌──────────────┐  ┌──────────────┐
│ MetricsService│  │AlertService  │
├──────────────┤  ├──────────────┤
│• Collect     │  │• Check       │
│• Aggregate   │  │  thresholds  │
│• Report      │  │• Trigger     │
└────────┬─────┘  └──────┬───────┘
         │               │
         └─────┬───┬─────┘
               │   │
        ┌──────┘   └──────┐
        ↓                 ↓
┌────────────────┐   ┌──────────────┐
│ Inference      │   │Health        │
│Service         │   │Checker       │
├────────────────┤   ├──────────────┤
│• Model wrapper │   │• CPU, Mem    │
│• Predict       │   │• Camera ok  │
│• Batch process │   │• Disk space  │
└────────┬───────┘   └──────┬───────┘
         │                  │
         └────────┬─────────┘
                  │
           ┌──────┴──────┐
           │             │
           ↓             ↓
    ┌─────────────────────────────┐
    │   Core ML Pipeline          │
    │  (core/processor.py, etc.)  │
    └─────────────────────────────┘
```

---

## 🚀 Adding New Service

Example: Slack alerts

```python
# backend/services/slack_alerts.py

import requests
from typing import Dict

class SlackAlertService:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert_type: str, message: str):
        """Send alert to Slack channel"""
        payload = {
            "text": f"🚨 Alert {alert_type}",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message}
                }
            ]
        }
        response = requests.post(self.webhook_url, json=payload)
        return response.status_code == 200
```

Then use in edge_app.py:

```python
from backend.services.slack_alerts import SlackAlertService

class EdgeApp:
    def __init__(self, ...):
        self.slack_alerts = SlackAlertService(
            webhook_url=os.getenv("SLACK_WEBHOOK_URL")
        )
    
    def run_webcam(self):
        # ... main loop
        if self.aggregator.should_alert():
            self.slack_alerts.send_alert(
                "HIGH_ANOMALY",
                f"Disease rate at {rate:.1f}%"
            )
```

---

## 📈 Metrics Tracking

Services automatically track:

**Performance Metrics:**
- FPS (frames per second)
- Latency per detection (ms)
- Batch throughput (images/sec)
- Model load time (ms)

**Business Metrics:**
- Total detections (all-time count)
- Disease count (current window)
- Healthy samples (%)
- Anomaly rate (%)
- Last alert timestamp

**System Metrics:**
- CPU usage (%)
- Memory usage (MB)
- Disk space available (GB)
- Temperature (°C)
- Uptime (hours)

---

## 🆘 Service Error Handling

Services should gracefully handle errors:

```python
class ChickenDiseaseDetector:
    def predict(self, image):
        try:
            # Validate input
            if not self._validate_image(image):
                raise ValueError("Invalid image format")
            
            # Model inference
            result = self.model(image)
            
            # Validate output
            if result.confidence < 0.3:
                logger.warning("Low confidence prediction")
            
            return result.disease, result.confidence
            
        except RuntimeError as e:
            logger.error(f"Model inference failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return "UNKNOWN", 0.0
```

---

## 📊 Service Configuration

Services read from config:

```python
# backend/utils/config.py

CONFIG = {
    "INFERENCE": {
        "CONF_THRESHOLD": 0.5,
        "ANOMALY_THRESHOLD": 10.0,
        "MODEL_PATH": "ensemble_model.pth"
    },
    "METRICS": {
        "WINDOW_SIZE_MINUTES": 5,
        "REPORT_INTERVAL_MINUTES": 5,
        "ENABLE_CLOUD_UPLOAD": True
    },
    "ALERTS": {
        "ANOMALY_THRESHOLD": 10.0,
        "CONFIDENCE_THRESHOLD": 0.7,
        "ALERT_CHANNELS": ["slack", "email"]
    }
}

# Usage
from backend.utils.config import CONFIG

anomaly_threshold = CONFIG["ALERTS"]["ANOMALY_THRESHOLD"]
```

---

## 🧪 Testing Services

```python
# backend/services/test_inference.py

import pytest
from PIL import Image
import numpy as np

def test_predict():
    detector = ChickenDiseaseDetector("ensemble_model.pth")
    
    # Create dummy image
    image = Image.new('RGB', (224, 224))
    
    disease, confidence = detector.predict(image)
    
    assert disease in ["Healthy", "Coccidiosis", "Salmonella", "New Castle"]
    assert 0 <= confidence <= 1

def test_predict_batch():
    detector = ChickenDiseaseDetector()
    images = [Image.new('RGB', (224, 224)) for _ in range(5)]
    
    results = detector.predict_batch(images)
    
    assert len(results) == 5
    assert all(0 <= conf <= 1 for _, conf in results)
```

Run tests:
```bash
pytest backend/services/ -v
```

---

## 📚 Service Responsibilities

| Service | Responsibility | Deployed |
|---------|-----------------|----------|
| **inference** | ML model wrapping | ✅ |
| **metrics** | Data collection & aggregation | ✅ |
| **alerts** | Alert triggering logic | ✅ |
| **health_check** | System monitoring | ✅ |

---

## 🔗 Next Steps

1. **Review** existing services in `backend/services/`
2. **Extend** with custom business logic
3. **Integrate** with API layer
4. **Test** all services
5. **Monitor** in production

**Services are the bridge between API and ML.** 🌉
