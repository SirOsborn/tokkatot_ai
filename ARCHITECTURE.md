# Architecture

## Data Flow

```
Camera Input (webcam/video/demo)
    ↓
Interface (get frame)
    ↓
Processor (YOLO detect → DenseNet classify)
    ↓
Tracker (assign IDs to birds)
    ↓
Aggregator (collect 5-min anomaly window)
    ↓
Display (annotate frame with results)
    ↓
Services (send alerts, metrics, logs)
    ↓
Output (stream/save/display)
```

## Component Details

### Frontend (Optional)
- **streaming/**: Live video feed with overlays
- **upload/**: Single image analysis
- **components/**: Shared UI elements

### Backend Required
- **core/**: ML pipeline (550 lines, immutable)
- **services/**: Business logic (alerts, metrics, logging)
- **utils/**: Config, transforms, logging
- **api/**: REST endpoints (optional)

### Deployment
- **Docker/**: Container config
- **systemd/**: Linux service file
- **Cloud/**: Monitoring integration

## Performance Targets

| Metric | Target |
|--------|--------|
| Inference | 15-30 FPS |
| GPU Memory | <2 GB |
| CPU | <60% |
| Disk | <500 MB |
| Alerts | Real-time +0ms |

## Files NOT Changed

Core modules (`application/backend/core/*.py`) are stable and should not be modified without careful testing.

Add new features in `services/` or `api/`, never in core.
