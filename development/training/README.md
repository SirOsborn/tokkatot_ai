# 📚 Training & Model Development

**Status**: Not deployed to production (local development only)

This folder contains scripts for training and evaluating models. 
**You don't need these on the Raspberry Pi or in production.**

---

## 🎯 What's Here?

| File | Purpose |
|------|---------|
| `train.py` | Train classification models (EfficientNetB0, DenseNet121) |
| `evaluate.py` | Evaluate trained models on test set |
| `export_models.py` | Convert PyTorch models to ONNX format |
| `export_tflite.py` | Convert to TFLite for Raspberry Pi |

---

## 🚀 Training Workflow

```bash
# 1. Prepare dataset (see ../data_prep/)
python ../data_prep/prepare_dataset.py

# 2. Train models
python train.py \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --device cuda

# 3. Evaluate on test set
python evaluate.py \
  --model-path ../outputs/ensemble_model.pth

# 4. Export for deployment
python export_models.py   # → outputs/ONNX models
python export_tflite.py   # → outputs/TFLite models
```

---

## 📊 Output

All trained models saved to `../outputs/`:

```
outputs/
├── ensemble_model.pth          # PyTorch ensemble (production)
├── checkpoints/
│   ├── DenseNet121_best.pth    # Best DenseNet checkpoint
│   └── EfficientNetB0_best.pth # Best EfficientNet checkpoint
├── onnx/
│   └── *.onnx                  # ONNX format (compatibility)
├── tflite/
│   └── *.tflite                # TFLite (for Raspberry Pi)
└── evaluation/
    └── evaluation_report.txt   # Test set results
```

---

## ⚙️ Training Scripts Details

### train.py

Trains and validates models with:
- ✅ Data augmentation
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Model checkpointing
- ✅ TensorBoard logging

```bash
python train.py --help          # See all options
```

### evaluate.py

Tests trained models on held-out test set:
- ✅ Accuracy, Precision, Recall, F1
- ✅ Confusion matrix
- ✅ Per-class metrics
- ✅ ROC curves (if binary)

```bash
python evaluate.py --model-path ../outputs/ensemble_model.pth
```

### export_models.py

Converts PyTorch to ONNX for:
- ✅ Compatibility with other frameworks
- ✅ Quantization options
- ✅ Mobile deployment

### export_tflite.py

Converts to TFLite for:
- ✅ Raspberry Pi inference
- ✅ Reduced model size
- ✅ Faster inference

---

## 📈 Model Performance

See model cards in `../docs/reports/`:

- `MODEL_CARD_DenseNet121.md` - Deep network, high accuracy
- `MODEL_CARD_EfficientNetB0.md` - Lightweight, balanced
- `MODEL_CARD_ENSEMBLE.md` - Voting ensemble, safest

---

## 🎓 Typical Training Time

| Model | Device | Time | Size |
|-------|--------|------|------|
| EfficientNetB0 | GPU | 20 min | 27 MB |
| DenseNet121 | GPU | 30 min | 32 MB |
| Ensemble | GPU | 50 min | 59 MB |

On CPU: ~3x slower

---

## ❌ Don't Deploy This Folder

These scripts are for **development only**. Production only needs:
- ✅ `../outputs/*.pth` (trained models)
- ✅ `../core/` (application)
- ✅ `../models/` (inference wrapper)

Training scripts add unnecessary dependencies and complexity to production.

---

## 🔗 References

- Model training details: See `train.py` docstring
- Dataset format: See `../data_prep/README.md`
- Production deployment: See `../DEPLOYMENT_GUIDE.md`
