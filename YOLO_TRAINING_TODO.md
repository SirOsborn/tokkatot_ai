# YOLO Training - TODO

## Issue
Demo mode works but YOLO detects 0 fecal samples because the generic `yolov8n.pt` model isn't trained on your dataset.

**Current flow:**
1. ✅ User uploads image
2. ❌ YOLO detects 0 samples (not trained on fecals)
3. ❌ Ensemble has nothing to classify
4. ❌ Result: 0 detections, 0 classifications

## Solution
Train a custom YOLO model on the chicken fecal dataset.

**Dataset available:**
- `development/archive/data/train/` - Training images (Coccidiosis, Healthy, New Castle Disease, Salmonella)
- `development/archive/data/test/` - Test images
- `development/archive/data/val/` - Validation images

## Implementation Steps

### 1. Create YOLO dataset config
File: `development/training/data.yaml`
```yaml
path: ../archive/data
train: train
val: val
test: test
nc: 1  # 1 class: "fecal_sample"
names: ['fecal_sample']
```

### 2. Generate YOLO labels
Convert image folders to YOLO format:
- Need bounding box annotations for fecal samples in each image
- Format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)

### 3. Train YOLO
```bash
cd development/training
python train_yolo.py --epochs 50 --device cuda --batch 16
```

### 4. Deploy
```bash
cp development/runs/detect/train/weights/best.pt application/yolov8_custom.pt
```

### 5. Test
```bash
python application/app.py --demo --image <image_path> --yolo-model application/yolov8_custom.pt
```

## Files Created
- [ ] `development/training/train_yolo.py` - Training script
- [ ] `development/training/data.yaml` - Dataset config
- [ ] `development/archive/labels/` - YOLO format annotations

## Expected Result
After training and deployment:
```
✓ Image loaded successfully
Analyzing image...
✓ YOLO detects fecal samples (boxes)
✓ Ensemble classifies each detection (disease type)
✓ Results: "X samples detected, Y diseased"
```

## Status
⏳ Pending - Mark as done after training completes
