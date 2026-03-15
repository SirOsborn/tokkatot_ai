# Architecture

## Data Flow (Hierarchical Trigger)

```
[EDGE: Raspberry Pi]
    Manure Conveyor Video Stream
    ↓
    YOLOv8 (Fast Screening: Healthy vs. Unhealthy)
    ↓
    (IF UNHEALTHY DETECTED)
    ↓
    Capture High-Res Image → Upload to Cloud
    ↓
[CLOUD: Tokkatot Server]
    Ensemble Inference (EfficientNetB0 + DenseNet121)
    ↓
    Voting Architecture (Safety-First Consensus)
    ↓
    Result: [Coccidiosis | Newcastle | Salmonella | Healthy | Uncertain]
    ↓
    Services (Farmer Alerts, IoT Dashboard, Metric Logging)
```

## Component Details

### Edge Layer (Screening)
- **Model**: Custom YOLOv8 (TFLite/INT8 Quantized)
- **Frequency**: 24/7 continuous monitoring.
- **Goal**: Minimize false negatives at the source with zero cloud cost for healthy samples.

### Cloud Layer (Diagnosis)
- **Model**: Ensemble (EfficientNetB0 + DenseNet121)
- **Mechanism**: Hard voting with confidence thresholding.
- **Goal**: High-precision classification of specific diseases.

### Services
- **Cloud Sync**: Manages the upload of suspicious frames.
- **Alert System**: Real-time push notifications to the Tokkatot Web App.
- **Inference Service**: Wraps the ensemble model for scalable cloud processing.

## Performance Targets

| Layer | Metric | Target |
|-------|--------|--------|
| **Edge** | Inference Latency | <50ms (YOLO) |
| **Edge** | 24/7 Stability | 100% Uptime |
| **Cloud** | Ensemble Accuracy | >99% |
| **Cloud** | Processing Time | <2s per upload |
| **Alerts** | Notification Delay | <5s from detection |

## Files NOT Changed

Core modules (`application/backend/core/*.py`) are stable and should not be modified without careful testing.

Add new features in `services/` or `api/`, never in core.
