# 🔬 Development - Model Training & Research

**This folder contains everything for developing and training the AI models.**

## ⚠️ Important

- **Status**: Local development environment only
- **Deployed**: ❌ NO (only trained models deployed)
- **Size**: Several GB (raw images, checkpoints, logs)
- **Cleanup**: Delete after training and validation

---

## 📂 What's Inside

| Folder | Purpose | Size | Deploy |
|--------|---------|------|--------|
| **data_prep/** | Dataset organization & validation | Large | ❌ |
| **training/** | Training scripts & configs | Small | ❌ |
| **models/** | Model architecture definitions | Small | ❌ |
| **evaluation/** | Evaluation metrics & plots | Small | ❌ |
| **outputs/** | Checkpoints, logs, trained models | Large | ✅ Selective |
| **archive/** | Raw images & original dataset | Very Large | ❌ |
| **docs/** | Reports, model cards, paper | Small | ℹ️ Reference |

---

## 🚀 Quick Start - Training

```bash
# 1. Prepare dataset
cd data_prep
python prepare_dataset.py
python verify_dataset.py

# 2. Train models
cd ../training
python train.py \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --device cuda

# 3. Evaluate
python evaluate.py

# 4. Export for production
python export_models.py    # → ONNX
python export_tflite.py    # → TFLite
```

---

## 📊 Dataset Structure

```
archive/data/
├── train/        ← 70% of data
│   ├── Healthy/        (560 images)
│   ├── Coccidiosis/    (200 images)
│   ├── Salmonella/     (200 images)
│   └── New Castle/     (160 images)
├── val/         ← 15% of data
└── test/        ← 15% of data (hold-out)
```

---

## 🤖 Models Trained

| Model | Accuracy | Size | Speed |
|-------|----------|------|-------|
| **EfficientNetB0** | 94% | 27 MB | Fast |
| **DenseNet121** | 96% | 32 MB | Slower |
| **Ensemble** (voting) | 97% | 59 MB | Medium |

See `docs/reports/MODEL_CARD_*.md` for details.

---

## 📈 Training Pipeline

```
Raw Images
    ↓
Dataset Preparation (data_prep/)
    ├─ Image validation
    ├─ Create train/val/test splits
    └─ Store in archive/data/
    ↓
Training (training/)
    ├─ Load model (models/)
    ├─ Train with augmentation
    ├─ Validate each epoch
    └─ Save checkpoints (outputs/checkpoints/)
    ↓
Evaluation (evaluation/)
    ├─ Test on held-out set
    ├─ Generate metrics
    └─ Create plots (docs/figures/)
    ↓
Export (training/export_*.py)
    ├─ PyTorch → ONNX
    ├─ PyTorch → TFLite
    └─ Save to outputs/
    ↓
DEPLOY: Copy outputs/ensemble_model.pth → ../application/
```

---

## 📝 Key Files

### training/train.py
Main training script. Usage:
```bash
python train.py --help
```

Features:
- ✅ Multi-GPU support
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ TensorBoard logging
- ✅ Model checkpointing

### training/evaluate.py
Evaluate on test set:
```bash
python evaluate.py --model-path ../outputs/ensemble_model.pth
```

Outputs:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Per-class metrics
- ROC curves

### models/architectures.py
Model definitions (PyTorch):
- `EfficientNetB0Classifier` - Lightweight
- `DenseNet121Classifier` - Powerful
- `EnsembleModel` - Voting ensemble

### data_prep/data/utils.py
Dataset utilities:
- Image transforms (augmentation)
- DataLoaders
- Normalization
- Class mappings

---

## 📊 Outputs Structure

```
outputs/
├── ensemble_model.pth           # Final trained model ⭐
├── checkpoints/
│   ├── DenseNet121_best.pth     # Best DenseNet
│   └── EfficientNetB0_best.pth  # Best EfficientNet
├── logs/
│   ├── DenseNet121/
│   │   └── events.out.tfevents  # TensorBoard logs
│   └── EfficientNetB0/
├── evaluation/
│   ├── evaluation_report.txt    # Test results
│   └── confusion_matrix.png
├── onnx/
│   ├── EfficientNetB0_best.onnx
│   └── yolov8n.onnx
└── tflite/
    ├── EfficientNetB0_best.tflite
    └── yolov8n.tflite
```

**What to Deploy**: Only `ensemble_model.pth` to `../application/`

---

## 🔍 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir outputs/logs
# Open: http://localhost:6006
```

View:
- Training loss curve
- Validation accuracy
- Learning rate changes
- Gradient norms

### Console Output
Training shows:
```
Epoch 1/50
  Train Loss: 0.234 | Train Acc: 92.5%
  Val Loss: 0.189 | Val Acc: 94.2%
```

---

## ⚙️ Configuration

Edit `training/config.yaml` to adjust:
- Batch size
- Learning rate
- Number of epochs
- Augmentation strength
- Model architecture
- Device (GPU/CPU)

Example:
```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  device: cuda

augmentation:
  rotation: 15
  flip: true
  brightness: 0.2
```

---

## 🔄 Iterative Development

### Experiment Workflow

1. **Change config** → `training/config.yaml`
2. **Train model** → `python train.py`
3. **Evaluate** → `python evaluate.py`
4. **Analyze results** → Check `outputs/evaluation/`
5. **Iterate** → Go back to step 1

### Debugging

If training fails:
```bash
# Check dataset
python data_prep/verify_dataset.py

# Check model
from models.architectures import EfficientNetB0Classifier
model = EfficientNetB0Classifier()
print(model)  # Print architecture

# Check data
from data_prep.data.utils import get_transforms, ChickenFecalDataset
```

---

## 📚 Documentation

### Model Cards
- `docs/reports/MODEL_CARD_EfficientNetB0.md`
- `docs/reports/MODEL_CARD_DenseNet121.md`
- `docs/reports/MODEL_CARD_ENSEMBLE.md`

Each includes:
- Architecture details
- Training data used
- Performance metrics
- Limitations and bias analysis
- Intended use

### Academic Paper
- `docs/reports/main.tex` - Full research paper
- `docs/reports/reference.bib` - References

### Evaluation Results
- `docs/figures/` - Plots and visualizations
- `outputs/evaluation/` - Numerical results

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce batch size, use smaller model |
| Training too slow | Use GPU, increase batch size |
| Low accuracy | Check dataset, increase epochs, try different model |
| Validation accuracy plateaus | Early stopping triggered, increase epochs |
| Dataset not found | Run prepare_dataset.py first |

---

## 🗑️ Cleanup

After training and deploying:

```bash
# Delete raw images (keep archive if needed for reference)
rm -rf archive/data/train archive/data/val archive/data/test

# Delete checkpoints (keep only best)
rm outputs/checkpoints/DenseNet121_*.pth
rm outputs/checkpoints/EfficientNetB0_*.pth
# Keep: ensemble_model.pth

# Delete logs (if no longer needed)
rm -rf outputs/logs/*

# Delete training artifacts
rm -rf training/__pycache__ models/__pycache__
```

**Keep**: `outputs/ensemble_model.pth` + `outputs/onnx/` + `outputs/tflite/`  
**Delete**: Everything else if space is needed

---

## 📝 References

- PyTorch docs: https://pytorch.org
- EfficientNet: https://arxiv.org/abs/1905.11946
- DenseNet: https://arxiv.org/abs/1608.06993
- YOLO: https://docs.ultralytics.com

---

## 🔗 Next Steps

1. **Prepare dataset** → See `data_prep/README.md`
2. **Train models** → Run `python training/train.py`
3. **Evaluate** → Run `python training/evaluate.py`
4. **Deploy** → Copy models to `../application/`
5. **Monitor** → See `application/README.md`

**Development is complete when `outputs/ensemble_model.pth` exists and test accuracy > 95%.** ✅
