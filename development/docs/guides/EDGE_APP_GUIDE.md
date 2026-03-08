# Tokkatot Edge Application - Local Testing Guide

## Overview

The `edge_app.py` is the core real-time monitoring application that simulates Raspberry Pi deployment. You can test it locally using:
- **Laptop webcam** (default)  
- **Video file** (for offline testing)

## What It Does

### Real-Time Pipeline:
1. **YOLOv8n Detection** - Detects fecal samples in video stream
2. **EfficientNetB0 Classification** - Classifies each detected sample for disease markers
3. **Ensemble Voting** - Applies safety-first logic to prevent false positives
4. **Centroid Tracking** - Avoids counting the same sample twice as it moves through frame
5. **Anomaly Aggregation** - 5-minute rolling window to detect disease outbreaks
6. **Real-time Alerts** - Visual alerts when anomaly rate exceeds threshold

### Key Features:
- ✅ Parallel model execution (both EfficientNetB0 and DenseNet121)
- ✅ Safety-first voting (high confidence + agreement required)
- ✅ Robust tracking with centroid matching
- ✅ Configurable thresholds for anomaly detection
- ✅ Real-time statistics display overlay
- ✅ Frame capture for debugging

---

## Installation

### Step 1: Install Dependencies
```bash
pip install opencv-python scipy
```

### Step 2: Verify Models Exist
Make sure you have:
- `outputs/ensemble_model.pth` - Ensemble model (99% accuracy)
- `yolov8n.pt` - YOLO detection model (should be in project root)

```bash
# Check what you have
ls -la outputs/ensemble_model.pth
ls -la yolov8n.pt
```

---

## Usage

### Basic: Run with Webcam
```bash
python edge_app.py
```

**Default settings:**
- Camera ID: 0 (built-in webcam)
- YOLO confidence: 0.5
- Anomaly threshold: 10% (alert if >10% of detected samples show disease)

### Alternative 1: Test with Video File
```bash
python edge_app.py --video /path/to/your/video.mp4
```

### Alternative 2: Adjust Thresholds
```bash
# More sensitive anomaly detection (alert at 5%)
python edge_app.py --anomaly-threshold 5.0

# More strict YOLO detection (ignore low-conf boxes)
python edge_app.py --conf-threshold 0.7

# Less strict (catch more potential samples)
python edge_app.py --conf-threshold 0.3
```

### Alternative 3: Use GPU or CPU
```bash
# Force GPU
python edge_app.py --device cuda

# Force CPU
python edge_app.py --device cpu

# Auto (default)
python edge_app.py --device auto
```

### Alternative 4: Custom Camera
```bash
# If you have multiple cameras, try:
python edge_app.py --camera-id 1
```

---

## Controls During Monitoring

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame to disk |

---

## Understanding the Output

### On-Screen Display:

**Top-left panel shows:**
- Total Detections (in 5-min window)
- Disease Markers (count)
- Anomaly Rate (%)
- Threshold (%)
- FPS (frames per second)

**Bottom banner shows:**
- **🟢 SAFE**: All systems normal (green)
- **🔴 ALERT**: Anomaly rate exceeded threshold (red)

### Console Output (every 30 frames):
```
[Frame 240] Detections: 15 | Disease: 2 | Anomaly: 13.3% | 🚨 ALERT
```

### Final Statistics:
```
============================================================
SESSION STATISTICS
============================================================
Duration: 45.2 seconds
Total Frames Processed: 1350
Average FPS: 29.9
Total Detections: 87
  - Healthy: 75
  - Disease: 10
  - Uncertain: 2
Disease Detection Rate: 11.5%
============================================================
```

---

## Example Scenarios

### Scenario 1: Normal Operation
```bash
python edge_app.py --anomaly-threshold 10.0
```
Expected: Green "SAFE" banner, disease rate < 10%

### Scenario 2: Sensitive Monitoring
```bash
python edge_app.py --anomaly-threshold 5.0
```
Expected: More frequent alerts, catches subtle issues early

### Scenario 3: Offline Video Testing
```bash
python edge_app.py --video test_video.mp4 --anomaly-threshold 8.0
```
Expected: Processes entire video, repeatable results

### Scenario 4: Debug Mode (CPU + Low Conf)
```bash
python edge_app.py --device cpu --conf-threshold 0.3
```
Expected: Catches more samples (even weak detections), slower FPS

---

## Troubleshooting

### Issue: "Cannot open camera 0"
**Solution:**
- Check if webcam is available: `ls /dev/video*`
- Try different camera ID: `python edge_app.py --camera-id 1`
- Some systems require permissions: May need to run with `sudo`

### Issue: Very Low FPS (<5)
**Solution:**
- Use CPU instead of GPU: `--device cpu` or `--device gpu`
- Reduce frame resolution in code (edit line ~280)
- Skip frames: Change `if frame_count % 1 == 0:` to `if frame_count % 3 == 0:` (process every 3rd frame)

### Issue: No Detections
**Solution:**
- Lower YOLO threshold: `--conf-threshold 0.3`
- Check models are correct: `python inference.py` to test manually
- Check video/camera is working: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

### Issue: Too Many False Positives
**Solution:**
- Increase YOLO threshold: `--conf-threshold 0.7`
- Increase safety voting threshold in code (edit line ~210): `healthy_threshold=0.90`
- Increase uncertainty threshold: `uncertainty_threshold=0.60`

---

## Next Steps: Integration with Cloud

Once local testing is working, you'll:

1. **Add MQTT Client** - Send alerts to cloud
2. **Add Local Logging** - Store detection history locally
3. **Add Webserver** - View status from phone/tablet
4. **Deploy to Raspberry Pi** - Transfer after testing

See [REALTIME_MONITORING.md](REALTIME_MONITORING.md) for cloud integration details.

---

## Performance Tips

- **For real-time monitoring**: Aim for 20+ FPS. Use GPU if available.
- **For accuracy**: Use lower confidence thresholds, willing to sacrifice some FPS
- **For throughput**: Increase `--conf-threshold` to 0.7+, process every 3rd frame

---

## File Output

Saved frames go to the project root with names like:
```
edge_frame_20260308_214530.jpg
```

Use these for debugging or building a dataset of edge cases.

---

## Questions?

Refer to:
- [GEMINI.md](GEMINI.md) - Project context
- [README.md](README.md) - Model cards & architecture
- [inference.py](inference.py) - Ensemble voting logic
- [models.py](models.py) - Model architecture
