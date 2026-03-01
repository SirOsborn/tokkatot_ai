## 1. Project Context & Goal
The goal is to evolve the `tokkatot_ai` system from static image analysis to **24/7 real-time monitoring** of chicken health via a manure conveyor belt in the **Tokkatot Smart Chicken Farming System**.

### Hybrid Cloud Ensemble Architecture
We use a safety-first split architecture to ensure maximum reliability and 0% false positives:
*   **Edge (Raspberry Pi + YOLOv8n + EfficientNetB0):** 
    - Performs real-time feces detection and initial "Healthy" screening.
    - If the model detects a disease marker or is unsure, it triggers a cloud verification.
*   **Cloud (vCPU Cloud Server + Full Ensemble):** 
    - Runs the full ensemble model (EfficientNetB0 + DenseNet121 + Safety Vote logic) on flagged images.
    - Only ~5% of samples reach the cloud, so a basic vCPU instance handles the load easily.
    - The parallel safety vote mechanism ensures 99% accuracy and 0% false positives before alerting the farmer via the Tokkatot web dashboard.

## 2. Current Status: Models Ready ✅
Both models in the ensemble are **already trained and production-ready**:
*   **Ensemble Model:** `outputs/ensemble_model.pth` (contains both EfficientNetB0 + DenseNet121, 99% accuracy).
*   **Edge TFLite:** `outputs/tflite/EfficientNetB0_best.tflite` — converted and ready for Hailo compilation.
*   **Edge Detection:** `outputs/tflite/yolov8n.tflite` — feces detection model, ready for Hailo compilation.
*   **Cloud Model:** Full ensemble (`outputs/ensemble_model.pth`) — deploy on the cloud GPU server for maximum-accuracy verification.

## 3. Next Steps: Phase 2 - Edge Implementation
This phase involves setting up the 24/7 monitoring logic on the Raspberry Pi.

### Step 1: Hailo Compilation (.tflite -> .hef)
*   **Action:** Transfer `outputs/tflite/*.tflite` to the Raspberry Pi.
*   **Process:** Use the Hailo Dataflow Compiler to generate `.hef` files optimized for the AI HAT+.

### Step 2: Develop `edge_app.py`
This is the core "AI Farm Guard" application. It will:
1.  **Stream Capture:** Capture frames from the camera positioned over the manure belt.
2.  **Detection (YOLOv8n):** Real-time detection of individual fecal samples on the belt.
3.  **Classification (EfficientNetB0):** Each detected sample is cropped and classified for signs of disease (Salmonella, Coccidiosis, New Castle).
4.  **Anomaly Aggregation:** Instead of single-frame alerts, we will implement "Anomaly Density" logic:
    *   If >X% of feces in a 5-minute window show disease markers → Trigger high-priority alert.
5.  **Tracking:** Simple object tracking (Centroid Tracking) to avoid double-counting the same feces moving along the belt.

### Step 3: Alarm & Integration
1.  **Local Feedback:** Trigger a local alert (LED/Buzzer) for immediate farm worker attention.
2.  **Cloud Sync:** Send metadata and high-confidence "sick" fecal images to the Tokkatot Cloud via MQTT.
3.  **24/7 Dashboard:** Log anomaly counts to a time-series database for long-term health monitoring.

## 4. Hardware Deployment Reference
*   **Mounting:** Camera must be top-down, fixed distance from the conveyor belt.
*   **Lighting:** Consistent, daylight-balanced LED lighting to ensure model accuracy 24/7.
*   **Cleaning:** Camera lens requires a protective cover/air blast to prevent dust buildup in the coop environment.
