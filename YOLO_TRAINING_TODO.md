# YOLO Training

## Approach
Train YOLOv8n with **2 classes** for edge-only detection and classification:

| Class ID | Name | Action |
|----------|------|--------|
| 0 | `healthy_feces` | Continue monitoring |
| 1 | `suspicious_feces` | Send to cloud for ensemble verification |

This eliminates the need for EfficientNetB0 on edge — **one model does detection + classification**.

## Dataset
Auto-generated YOLO labels from existing classification folders:
- `Healthy/` → class 0 (healthy_feces)
- `Coccidiosis/` + `Salmonella/` + `New Castle Disease/` → class 1 (suspicious_feces)

Bounding boxes cover the full image with 5% margin (each image = single sample).

## Steps

### 1. Generate labels
```bash
cd development/training
python generate_yolo_labels.py --data-dir ../archive/data
```

### 2. Train YOLO
```bash
python train_yolo.py --epochs 50 --device cuda --batch 16
```

### 3. Deploy to application
```bash
cp runs/detect/train/weights/best.pt ../../application/yolov8_custom.pt
```

### 4. Test
```bash
python ../../application/app.py --demo --image <image_path> --yolo-model ../../application/yolov8_custom.pt
```

## Files
- [x] `development/training/generate_yolo_labels.py` - Auto-annotation script
- [x] `development/training/data.yaml` - Dataset config (2 classes)
- [x] `development/training/train_yolo.py` - Training script
- [ ] `development/archive/data/train_labels/` - Generated labels (run step 1)
- [ ] `application/yolov8_custom.pt` - Trained model (run steps 2-3)

## Status
✅ **Core Architecture Shift**: YOLOv8 is now the primary Edge Gatekeeper.
⏳ Scripts ready — Generating labels from the 400k image dataset, then training for binary Healthy/Unhealthy classification.
