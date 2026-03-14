# Tokkatot Edge Application - Local Testing Guide

The `application/app.py` is the core real-time monitoring application. While designed for Raspberry Pi, you can test the full pipeline locally on your laptop.

## 1. Quick Start

### Install Dependencies
```bash
cd application
pip install -r requirements.txt
```

### Run with Webcam (Hybrid Mode)
```bash
# Start Cloud API in one terminal
python start_api.py

# Run Edge app in another terminal
python app.py
```

## 2. Testing Scenarios

### Offline Video Test
Use this to simulate the conveyor belt without a live camera:
```bash
python app.py --video your_conveyor_clip.mp4
```

### Demo Mode (Single Image)
Analyze a single suspected case manually:
```bash
python app.py --demo --image suspicious_sample.jpg
```

### Sensitivity Adjustment
Change the alert threshold (default is 10.0%):
```bash
# Alert if only 5% of samples are suspicious
python app.py --anomaly-threshold 5.0
```

## 3. Deployment Controls

| Key | Action |
| :--- | :--- |
| `q` | Quit application |
| `s` | Save current frame (for debugging edge cases) |

## 4. Understanding the Display

- **🟢 SAFE**: Anomaly rate is below threshold.
- **🔴 ALERT**: Outbreak detected! Anomaly rate exceeded threshold.
- **Panel Metrics**:
    - **Disease**: Counts confirmed by Cloud Ensemble.
    - **Anomaly %**: 5-minute rolling average.

## 5. Troubleshooting

- **"Model not found"**: Ensure `yolov8_custom.pt` and `ensemble_model.pth` are inside the `application/` folder.
- **Cloud Connection Error**: Ensure `start_api.py` is running before starting `app.py`.
- **Low FPS**: Set `--device cpu` if you don't have a CUDA-compatible GPU.

---
See [REALTIME_MONITORING.md](REALTIME_MONITORING.md) for architecture details.
