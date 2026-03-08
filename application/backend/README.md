# 🔧 Backend - Server Logic

**Backend contains all server-side logic:**
- ML inference pipeline
- REST API endpoints  
- Business logic & services
- Utilities & helpers

---

## 📂 Structure

```
backend/
├── core/        # ML pipeline (550 lines, focused modules)
├── api/         # REST API endpoints (FUTURE)
├── services/    # Business logic
└── utils/       # Helper functions
```

---

## 🎯 core/ - ML Inference Pipeline

**The actual disease detection system** (550 lines across 7 modules)

### Modules
| File | Lines | Purpose |
|------|-------|---------|
| edge_app.py | 158 | Coordinator |
| processor.py | 103 | Frame processing |
| display.py | 101 | Visualization |
| interface.py | 189 | I/O modes |
| tracker.py | 94 | Tracking |
| aggregator.py | 52 | Anomaly detection |
| __init__.py | 5 | Exports |

### Data Flow
```
Input Frame
    ↓
Processor: YOLO detect + Ensemble classify + Track + Aggregate
    ↓
Display: Draw boxes + stats overlay + alerts
    ↓
Interface: Output to screen or file
```

### Usage
```python
from backend.core import EdgeApp

app = EdgeApp(
    ensemble_model_path="ensemble_model.pth",
    yolo_model_path="yolov8n.pt",
    device="cuda"
)

# Run with webcam
app.run_webcam()

# Or process single frame
stats = app.capture_and_analyze()
```

---

## 🔌 api/ - REST API (FUTURE)

**For integration with Tokkatot web app**

### Planned Endpoints

```
GET  /health                    # System status
POST /detect                    # Detect in image
GET  /metrics                   # Current metrics
GET  /metrics/history           # 24h history
POST /config/update             # Change settings
GET  /models/info               # Model details
```

### Example: Detect Disease

```python
import requests

# Send image
files = {'image': open('test.jpg', 'rb')}
response = requests.post('http://localhost:5000/detect', files=files)

result = response.json()
# {
#   "disease": "Coccidiosis",
#   "confidence": 0.96,
#   "boxes": [[x1,y1,x2,y2], ...],
#   "anomaly_rate": 5.04,
#   "timestamp": "2026-03-08T12:30:00Z"
# }
```

### API Specification
See [api/README.md](api/README.md) when implemented

---

## 🎯 services/ - Business Logic

**Higher-level functionality**

### Key Classes

```python
# Inference wrapper
from backend.services.inference import ChickenDiseaseDetector
detector = ChickenDiseaseDetector("ensemble_model.pth")
disease, confidence = detector.predict(image)

# Metrics collection
from backend.services.metrics import MetricsCollector
metrics = MetricsCollector()
metrics.record_detection(disease_type, confidence)
stats = metrics.get_stats()

# Alert management
from backend.services.alerts import AlertManager
alerts = AlertManager()
if metrics['anomaly_rate'] > 10:
    alerts.trigger("HIGH_DISEASE_RATE")

# Health checks
from backend.services.health_check import HealthChecker
health = HealthChecker()
status = health.check_all()
```

### Files

| File | Purpose |
|------|---------|
| inference.py | ML model wrapper |
| metrics.py | Metrics aggregation |
| alerts.py | Alert triggering |
| health_check.py | System status |

---

## 🛠️ utils/ - Utilities

**Helper functions and configuration**

### Key Modules

```python
# Configuration
from backend.utils.config import CONFIG
print(CONFIG['CONF_THRESHOLD'])

# Image transforms
from backend.utils.transforms import get_transforms
transform = get_transforms()
processed = transform(image)

# Logging
from backend.utils.logger import get_logger
logger = get_logger(__name__)
logger.info("Starting app")

# Validation
from backend.utils.validators import validate_config
validate_config(config_dict)
```

### Files

| File | Purpose |
|------|---------|
| config.py | Settings & defaults |
| transforms.py | Image preprocessing |
| logger.py | Logging setup |
| validators.py | Input validation |

---

## 🔄 Development Workflow

### Add New Feature to Backend

**Example: Add email alerts**

```python
# 1. Create service
# backend/services/email_alerts.py
class EmailAlertService:
    def send_alert(self, alert_message):
        # Implementation
        pass

# 2. Import in edge_app.py
from backend.services.email_alerts import EmailAlertService

# 3. Use in EdgeApp
class EdgeApp:
    def __init__(self, ...):
        self.email_alerts = EmailAlertService()
    
    def _process_and_display(self):
        if self.aggregator.should_alert():
            self.email_alerts.send_alert("Disease detected!")

# 4. Test
python -m pytest backend/services/test_email_alerts.py
```

---

## 📊 Module Interactions

```
app.py (entry point)
    ↓
edge_app.py (coordinator)
    ├→ processor.py (detect + classify)
    ├→ display.py (draw frame)
    ├→ interface.py (get input)
    │
    ├→ [services layer]
    │   ├→ inference.py (ML wrapper)
    │   ├→ metrics.py (collect stats)
    │   ├→ alerts.py (trigger alerts)
    │   └→ health_check.py (check status)
    │
    └→ [utils layer]
        ├→ config.py (settings)
        ├→ transforms.py (preprocessing)
        ├→ logger.py (logging)
        └→ validators.py (validation)
```

---

## ✅ Testing

### Test individual modules
```bash
# Test processor
python -m pytest backend/core/test_processor.py

# Test services
python -m pytest backend/services/test_inference.py

# Test utils
python -m pytest backend/utils/test_config.py
```

### Integration test
```bash
python -m pytest backend/test_integration.py
```

---

## 🚀 Deployment

All of `backend/` gets deployed:

```bash
# Copy to Raspberry Pi
scp -r backend pi@raspberrypi.local:/home/pi/application/

# Or in Docker
COPY backend /app/backend
```

---

## 📝 Code Style

- **Python 3.9+**
- **PEP 8** formatting
- **Type hints** for all functions
- **Docstrings** for modules and classes
- **Max 100 chars** per line
- **Single responsibility** per module

---

## 🔗 Next Steps

- **Configure**: Edit `utils/config.py`
- **Extend**: Add services in `services/`
- **Test**: Add tests in `test_*.py`
- **Deploy**: Follow deployment/README.md

**Backend is production-ready.** 🎯
