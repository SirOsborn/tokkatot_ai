# Tokkatot AI - Chicken Disease Detection

**Real-time disease detection for poultry farms using ensemble ML**

**Backend AI Engine for Tokkatot Smart Chicken Farming**

**© 2026 Tokkatot. All Rights Reserved.**

## Quick Start

```bash
# Test locally (demo mode)
python application/app.py --demo

# Run with webcam
python application/app.py --camera-id 0

# Deploy to Raspberry Pi
scp -r application/ pi@raspberrypi.local:/home/pi/
```

## What It Does

- ✅ Detects 4 health states from chicken fecal images (Coccidiosis, Salmonella, New Castle, Healthy)
- ✅ **Hierarchical Trigger Architecture**: 24/7 YOLO monitoring on Edge triggers Cloud Ensemble analysis
- ✅ 99% accuracy via ensemble safety voting (EfficientNetB0 + DenseNet121) in the cloud
- ✅ Lightweight YOLOv8 runs 24/7 on Raspberry Pi monitoring manure conveyors
- ✅ Sends high-precision health metrics to Tokkatot Cloud only when anomalies are detected

## Architecture

```
Conveyor Belt → YOLO (Edge: Healthy/Unhealthy?) → [If Unhealthy] → Ensemble (Cloud: Specific Disease?) → Alert
```

### Hybrid Edge-Cloud Pipeline
| Layer | Model | Role |
|-------|-------|------|
| **Edge** (Raspberry Pi) | YOLOv8 (Custom Trained) | 24/7 Gatekeeper: Monitors manure conveyor, detects 'Unhealthy' feces |
| **Cloud** (Tokkatot Server) | Ensemble (EfficientNetB0 + DenseNet121) | High-precision diagnosis: Triggered only for suspicious samples |

→ This architecture ensures **low power consumption** on edge devices while maintaining **maximum safety** through cloud-based ensemble verification.

### Safety-First Decision Rules
The system prioritizes **Recall** (Flock Safety) over general accuracy:
1. **YOLO Trigger**: If the edge model detects ANY sign of abnormality, the sample is escalated.
2. **Ensemble Agreement**: In the cloud, both EfficientNet and DenseNet must agree on a "Healthy" status with high confidence.
3. **Isolation Rule**: If models disagree OR confidence is < 85%, the sample is flagged as "Uncertain" for manual veterinarian review.

## Project Structure

```
tokkatot_ai/
│
├── application/          ← DEPLOY THIS (backend engine)
│   ├── app.py            (main entry point)
│   ├── backend/core/     (ML pipeline: detect → classify → track → alert)
│   ├── backend/services/ (inference wrapper, metrics, alerts)
│   └── backend/api/      (REST API for Tokkatot Web App integration)
│
├── development/          ← TRAINING ONLY (not deployed)
│   ├── training/         (train.py, export scripts)
│   ├── models/           (EfficientNetB0, DenseNet121, Ensemble architectures)
│   ├── data_prep/        (dataset utilities)
│   ├── evaluation/       (test metrics & plots)
│   ├── outputs/          (trained models, TFLite, ONNX)
│   └── docs/             (model cards, guides, research paper)
│
├── STRUCTURE.md          Quick structure reference
├── ARCHITECTURE.md       Data flow & component details
├── DEPLOYMENT_GUIDE.md   Raspberry Pi & cloud deployment
└── YOLO_TRAINING_TODO.md Custom YOLO training roadmap
```

See [STRUCTURE.md](STRUCTURE.md) for detailed module breakdown.

## Models

| Model | Accuracy | Recall | Use |
|-------|----------|--------|-----|
| **Ensemble** | **99%** | **99%** | Final decision (safety voting) |
| EfficientNetB0 | 98.05% | 98.05% | Edge classification (fast) |
| DenseNet121 | 96.69% | 96.69% | Deep classification (powerful) |
| YOLOv8n | — | — | Feces detection on conveyor belt |

**Test Results (70,677 samples):**
- Classified: 67,137 (94.99%) at 99% accuracy
- Isolated: 3,540 (5.01%) for safety verification
- Per-class recall: Coccidiosis 100%, Healthy 98%, Newcastle 100%, Salmonella 100%

### Model Documentation
- [EfficientNetB0 Model Card](development/docs/reports/MODEL_CARD_EfficientNetB0.md)
- [DenseNet121 Model Card](development/docs/reports/MODEL_CARD_DenseNet121.md)
- [Ensemble Model Card](development/docs/reports/MODEL_CARD_ENSEMBLE.md)

## Deploy to Raspberry Pi

```bash
# 1. Copy application to Pi
scp -r application/ pi@raspberrypi.local:/home/pi/tokkatot_ai/

# 2. Install & start
ssh pi@raspberrypi.local
cd tokkatot_ai/application && pip install -r requirements.txt
sudo systemctl enable tokkatot-edge && sudo systemctl start tokkatot-edge

# 3. Monitor
sudo journalctl -u tokkatot-edge -f
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for full instructions.

## Train Models

```bash
cd development/training
python train.py --epochs 50 --device cuda
python evaluate.py
cp ../outputs/ensemble_model.pth ../../application/
```

See [development/README.md](development/README.md) for full training pipeline.

## Cloud Integration

Every 5 minutes the edge device sends metrics to the Tokkatot Cloud:
- FPS, latency, detection count
- Disease rate, anomaly percentage
- System health (CPU, memory, uptime)

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for cloud configuration.

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| [STRUCTURE.md](STRUCTURE.md) | Quick folder & module reference |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Data flow & component details |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Pi deployment & cloud setup |
| [YOLO_TRAINING_TODO.md](YOLO_TRAINING_TODO.md) | Custom YOLO training roadmap |
| [application/README.md](application/README.md) | Production system guide |
| [development/README.md](development/README.md) | Training & research guide |
| [Edge App Guide](development/docs/guides/EDGE_APP_GUIDE.md) | Local testing with webcam/video |
| [Real-Time Monitoring](development/docs/guides/REALTIME_MONITORING.md) | Hybrid cloud architecture |
| [Docker Deployment](development/docs/guides/DOCKER_DEPLOYMENT.md) | Container deployment |

## License

**© 2026 Tokkatot. All Rights Reserved.**

This software is proprietary and confidential. It is part of Tokkatot's Smart Chicken Farming Solutions and may not be copied, modified, distributed, or used without explicit written permission from Tokkatot.

## Contact

**Tokkatot Smart Chicken Farming Solutions**

- **Email**: [tokkatot.info@gmail.com](mailto:tokkatot.info@gmail.com)
- **Website**: [tokkatot.aztrolabe.com](https://tokkatot.aztrolabe.com)
- **AI Engineer**: [Sun Heng](mailto:sunhenglong@outlook.com)

---

**Proprietary Notice**: This software is part of Tokkatot's integrated smart farming ecosystem and is protected by intellectual property rights.
