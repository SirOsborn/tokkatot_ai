# 🔌 API - REST Endpoints

**Integration with Tokkatot web dashboard**

## ⚠️ Status

- **Status**: Placeholder / Future implementation
- **Framework**: FastAPI or Flask (recommended: FastAPI)
- **Deployment**: Optional (can use without it)

---

## 📋 Planned Endpoints

### Health & Status

```
GET /health
├─ Returns: System status
├─ Response: {"status": "ok", "uptime": 24.5}
└─ Used by: Monitoring, health checks
```

### Detection / Inference

```
POST /api/v1/detect
├─ Input: Image file (multipart/form-data)
├─ Returns: Detection results
├─ Response: {
│   "disease": "Coccidiosis",
│   "confidence": 0.96,
│   "boxes": [[x1,y1,x2,y2], ...],
│   "processing_time_ms": 45,
│   "anomaly_rate": 5.04,
│   "timestamp": "2026-03-08T12:30:00Z"
│ }
└─ Used by: Web UI, integrations
```

### Metrics

```
GET /api/v1/metrics
├─ Returns: Current metrics
├─ Response: {
│   "fps": 22.3,
│   "total_detections": 457,
│   "disease_count": 23,
│   "anomaly_rate": 5.04,
│   "last_alert": "2026-03-08T12:25:00Z",
│   "uptime_hours": 24.5,
│   "latency_ms": 45.2,
│   "cpu_percent": 65,
│   "memory_mb": 512
│ }
└─ Used by: Dashboard, monitoring
```

```
GET /api/v1/metrics/history
├─ Query: ?hours=24  (default 24h)
├─ Returns: Historical metrics
├─ Response: [{metrics_point}, ...]
└─ Used by: Graphs, analytics
```

### Configuration

```
GET /api/v1/config
├─ Returns: Current configuration
└─ Response: {config_dict}
```

```
POST /api/v1/config
├─ Input: {threshold: 0.5, disease_threshold: 10}
├─ Returns: Success/error
└─ Response: {"status": "ok"}
```

### Models

```
GET /api/v1/models
├─ Returns: Available models and info
├─ Response: {
│   "ensemble": {
│     "version": "1.0.0",
│     "accuracy": 0.97,
│     "classes": ["Healthy", "Coccidiosis", "Salmonella", "New Castle"]
│   },
│   "yolo": {
│     "version": "8n",
│     "purpose": "feces detection"
│   }
│ }
└─ Used by: UI, diagnostics
```

---

## 🛠️ Implementation Guide

### Using FastAPI (Recommended)

```python
# backend/api/app.py

from fastapi import FastAPI, File, UploadFile
from backend.core import EdgeApp
from backend.services.inference import ChickenDiseaseDetector

app = FastAPI(title="Tokkatot Edge API")

# Initialize models
edge_app = EdgeApp()
detector = ChickenDiseaseDetector()

@app.get("/health")
def health():
    return {"status": "ok", "uptime": edge_app.get_uptime()}

@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...)):
    image = Image.open(file.file)
    disease, confidence = detector.predict(image)
    
    return {
        "disease": disease,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/metrics")
def metrics():
    return edge_app.get_current_metrics()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

### Using Flask

```python
# backend/api/app.py

from flask import Flask, request, jsonify
from backend.core import EdgeApp
from backend.services.inference import ChickenDiseaseDetector

app = Flask(__name__)

edge_app = EdgeApp()
detector = ChickenDiseaseDetector()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/v1/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    image = Image.open(file.stream)
    disease, confidence = detector.predict(image)
    
    return jsonify({
        "disease": disease,
        "confidence": confidence
    })

@app.route("/api/v1/metrics", methods=["GET"])
def metrics():
    return jsonify(edge_app.get_current_metrics())

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
```

---

## 📡 Usage Examples

### Python Client

```python
import requests

# Detect in image
files = {'image': open('test.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/v1/detect', files=files)
result = response.json()
print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### JavaScript Client

```javascript
// Fetch metrics
async function getMetrics() {
    const response = await fetch('http://localhost:5000/api/v1/metrics');
    const metrics = await response.json();
    console.log(`FPS: ${metrics.fps}`);
    console.log(`Disease Rate: ${metrics.anomaly_rate}%`);
}

// Detect in image
async function detectDisease(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await fetch('http://localhost:5000/api/v1/detect', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result;
}
```

### cURL

```bash
# Get metrics
curl http://localhost:5000/api/v1/metrics

# Detect in image
curl -X POST -F "image=@test.jpg" \
  http://localhost:5000/api/v1/detect
```

---

## 🔐 Authentication (Optional)

```python
# backend/api/auth.py

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials = Depends(security)):
    token = credentials.credentials
    if not validate_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...), token = Depends(verify_token)):
    # ... implementation
    pass
```

---

## 📝 Request/Response Models (Pydantic)

```python
# backend/api/models.py

from pydantic import BaseModel
from typing import List

class DetectionResult(BaseModel):
    disease: str
    confidence: float
    boxes: List[List[int]]
    processing_time_ms: float
    anomaly_rate: float
    timestamp: str

class MetricsResponse(BaseModel):
    fps: float
    total_detections: int
    disease_count: int
    anomaly_rate: float
    uptime_hours: float
    
class ConfigUpdate(BaseModel):
    conf_threshold: float = 0.5
    anomaly_threshold_pct: float = 10.0
```

---

## 🚀 Running API Server

### Development

```bash
# FastAPI
pip install fastapi uvicorn
python backend/api/app.py

# Flask
pip install flask
python backend/api/app.py
```

### Production

```bash
# FastAPI with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.api.app:app

# Or systemd service
sudo systemctl start tokkatot-api
```

### Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "backend.api.app:app"]
```

---

## 🧪 Testing API

```python
# backend/api/test_api.py

import pytest
from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_detect():
    with open("test_image.jpg", "rb") as f:
        response = client.post(
            "/api/v1/detect",
            files={"image": f}
        )
    assert response.status_code == 200
    assert "disease" in response.json()

def test_metrics():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert "fps" in response.json()
```

Run tests:
```bash
pytest backend/api/test_api.py -v
```

---

## 📊 API Performance

| Endpoint | Response Time | Throughput |
|----------|---------------|-----------|
| /health | <10ms | Unlimited |
| /api/v1/metrics | <50ms | Unlimited |
| /api/v1/detect | ~45ms | 20 req/s |
| /api/v1/config | <10ms | Unlimited |

---

## 🔗 Integration with Tokkatot

Your Tokkatot web app connects like:

```javascript
// In Tokkatot dashboard
const EDGE_API_URL = "http://raspberrypi.local:5000";

async function refreshMetrics() {
    const response = await fetch(`${EDGE_API_URL}/api/v1/metrics`);
    const metrics = await response.json();
    
    // Update dashboard UI
    updateDashboard(metrics);
}

// Refresh every 5 seconds
setInterval(refreshMetrics, 5000);
```

---

## 📖 API Documentation

Once running:

### FastAPI (Auto-generated)
```
http://localhost:5000/docs          # Interactive Swagger UI
http://localhost:5000/redoc         # ReDoc documentation
```

### Manual
See full spec in [API_SPEC.md](API_SPEC.md) (to be created)

---

## ⚠️ Note

This is a **placeholder** for future implementation. Currently, the system works perfectly fine without an API server. The API becomes useful when:

1. ✅ Multiple Pis need to be coordinated
2. ✅ Cloud dashboard needs real-time data
3. ✅ External integrations are needed
4. ✅ Web UI is deployed

For basic Raspberry Pi operation, the CLI interface is sufficient.

---

## 🔗 Next Steps

1. Choose framework (FastAPI recommended)
2. Implement endpoints
3. Add authentication
4. Write tests
5. Deploy with Gunicorn/Uvicorn

**API implementation is optional but recommended for Tokkatot integration.** 🚀
