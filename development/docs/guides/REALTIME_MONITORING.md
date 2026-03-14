# Hybrid Cloud-Edge Monitoring Architecture

The Tokkatot AI system uses a **safety-first hybrid architecture** to provide 24/7 real-time monitoring of chicken health via manure conveyor belts.

## 1. Split-Logic Strategy

We balance high-speed edge detection with high-accuracy cloud verification:

### Edge: High-Speed Screening (Local)
- **Model**: Custom-trained **YOLOv8n** (`application/yolov8_custom.pt`).
- **Function**: Scans every frame (24/7 runtime) to detect fecal samples and perform binary screening.
- **Decision Logic**:
    - **Healthy**: Detected feces look normal. Monitoring continues silently.
    - **Suspicious**: Detected feces show potential disease markers. The image crop is immediately sent to the Cloud for verification.

### Cloud: High-Accuracy Verification (Remote)
- **Model**: Full **Ensemble AI** (EfficientNetB0 + DenseNet121 + Safety Vote).
- **Function**: Receives "suspicious" crops from the edge.
- **Decision Logic**:
    - Runs a multi-model "sanity check" with 99% accuracy.
    - If confirmed as "Disease", it logs the event and triggers a high-priority farmer alert.
    - If "Healthy", it overrides the edge's suspicion to prevent false alarms.

## 2. Production Assets

| Asset | Location | Role |
| :--- | :--- | :--- |
| **Edge Model** | `application/yolov8_custom.pt` | Binary detection (Healthy vs Suspicious) |
| **Ensemble Model** | `application/ensemble_model.pth` | High-accuracy verification (Cloud-side) |
| **Edge App** | `application/app.py` | Main coordinator (Streams video/cam) |
| **Cloud API** | `application/backend/api/main.py` | FastAPI server for verification |

## 3. Data Flow

1.  **Detection**: Edge YOLO detects feces on the belt.
2.  **Screening**: If categorized as `suspicious_feces` (Class 1), `cloud_sync.py` triggers.
3.  **Transmission**: Image crop is POSTed to the Cloud API `/verify` endpoint.
4.  **Verification**: Cloud Ensemble runs and returns a final classification.
5.  **Action**: If "Disease" is confirmed, the anomaly rate in `aggregator.py` increases, eventually triggering a local visual alert and a cloud dashboard update.

## 4. Hardware Deployment

- **Mounting**: Camera must be top-down, fixed distance from the conveyor belt (50-80cm recommended).
- **Lighting**: Constant, daylight-balanced LED lighting. The model was trained on vibrant, well-lit samples.
- **Maintenance**: Use a protective glass cover for the camera lens to prevent dust buildup.

---
See [EDGE_APP_GUIDE.md](EDGE_APP_GUIDE.md) for local testing instructions.
