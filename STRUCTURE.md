# Project Structure

## Two Main Folders

```
tokkatot_ai/
│
├── application/      ← DEPLOY THIS (backend engine)
│   ├── app.py        (main entry point)
│   ├── backend/core/ (ML pipeline)
│   ├── backend/api/  (REST API for Tokkatot Web App)
│   ├── backend/services/ (inference, metrics, alerts)
│   └── requirements.txt
│
└── development/      ← TRAINING ONLY
    ├── training/     (train.py, evaluate.py)
    ├── data_prep/
    ├── models/
    ├── outputs/      (ensemble_model.pth)
    └── docs/
```

## Quick Commands

```bash
# Test
python application/app.py --demo

# Deploy
scp -r application/ pi@raspberrypi.local:/home/pi/

# Train
cd development/training && python train.py --epochs 50

# Copy model
cp development/outputs/ensemble_model.pth application/

# Start on Pi
sudo systemctl start tokkatot-edge

# View logs
sudo journalctl -u tokkatot-edge -f
```

## Backend Structure

| Module | Purpose | Lines |
|--------|---------|-------|
| core/edge_app.py | Coordinator | 158 |
| core/processor.py | Detection + Classification | 103 |
| core/display.py | Visualization | 101 |
| core/interface.py | I/O modes | 189 |
| core/tracker.py | Tracking | 94 |
| core/aggregator.py | Anomaly detection | 52 |
| services/ | Business logic | ~100 |

**Total: 550-700 lines, 7 focused modules**
