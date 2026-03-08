# 🔧 Data Preparation

**Status**: Not deployed to production (local development only)

This folder contains scripts for preparing and organizing datasets.
**You don't need these on the Raspberry Pi or in production.**

---

## 🎯 Purpose

Before training models, raw fecal sample images must be:
1. ✅ Collected from cameras
2. ✅ Manually labeled (disease type or healthy)
3. ✅ Organized into train/val/test splits
4. ✅ Normalized and augmented
5. ✅ Stored as PyTorch datasets

---

## 📂 Dataset Organization

**Expected format**:

```
archive/data/
├── train/
│   ├── Healthy/          # 1000+ images
│   ├── Coccidiosis/      # 200+ images
│   ├── Salmonella/       # 200+ images
│   └── New Castle Disease/ # 200+ images
├── val/
│   ├── Healthy/
│   ├── Coccidiosis/
│   ├── Salmonella/
│   └── New Castle Disease/
└── test/
    ├── Healthy/
    ├── Coccidiosis/
    ├── Salmonella/
    └── New Castle Disease/
```

---

## 🚀 Data Preparation Steps

```bash
# 1. Download raw images
# (Manual: capture from cameras or download from archive)

# 2. Create train/val/test split
python prepare_dataset.py \
  --input-folder ./raw_images \
  --output-folder archive/data \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15

# 3. Check dataset integrity
python verify_dataset.py

# 4. Generate statistics
python dataset_stats.py
```

---

## 📊 Dataset Statistics

After preparation, calculate:

```
Total samples: 1600
├── Train: 1120 (70%)
├── Validation: 240 (15%)
└── Test: 240 (15%)

Class distribution (train):
├── Healthy: 560 samples
├── Coccidiosis: 200 samples
├── Salmonella: 200 samples
└── New Castle Disease: 160 samples
```

---

## 🎨 Data Augmentation

During training, images are augmented:

- ✅ Random rotation ±15°
- ✅ Random horizontal flip
- ✅ Random brightness/contrast
- ✅ Random crop and resize
- ✅ Normalization (ImageNet stats)

See `../data/utils.py` for implementation.

---

## ❌ Don't Deploy This Folder

These scripts are for **development only**. Production only needs:
- ✅ `../data/utils.py` (for transforms during inference)
- ✅ `../outputs/` (trained models ready to use)

Raw data and preparation scripts add gigabytes of storage and aren't needed on Raspberry Pi.

---

## 📝 Class Definitions

The application recognizes 4 classes:

| ID | Class | Description |
|----|----|------------|
| 0 | **Healthy** | Normal fecal samples |
| 1 | **Coccidiosis** | Parasitic infection (blood in feces) |
| 2 | **Salmonella** | Bacterial infection (watery diarrhea) |
| 3 | **New Castle Disease** | Viral infection (yellow droppings) |

---

## 🔗 References

- Dataset location: `archive/data/`
- Training process: `../training/train.py`
- Runtime transforms: `../data/utils.py`
- Model training: `../DEPLOYMENT_GUIDE.md`
