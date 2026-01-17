# Tokkatot AI - Chicken Disease Detection System

**Safety-First Ensemble AI for Chicken Disease Detection via Fecal Images**

**© 2026 Tokkatot. All Rights Reserved.**  
*Part of Tokkatot Smart Chicken Farming Solutions*

## 🎯 Overview

Tokkatot AI is a safety-first ensemble machine learning system designed to detect chicken diseases through fecal matter analysis. The system prioritizes **100% recall** to ensure no diseased chickens are falsely classified as healthy, using a parallel safety vote mechanism with two complementary deep learning models.

This proprietary system is developed exclusively for Tokkatot's integrated smart farming ecosystem and is protected under intellectual property rights.

## 🏗️ Architecture

### Ensemble Approach: Parallel Safety Vote

The system combines two state-of-the-art neural networks:

1. **EfficientNetB0** (v1.0.0 - Released Jan 16, 2026)
   - Fast, lightweight model optimized for edge deployment (Raspberry Pi)
   - General-purpose feature detection
   - Efficient inference with minimal computational requirements
   - **98.05% validation recall** (90 epochs)
   - [📄 Model Card](MODEL_CARD_EfficientNetB0.md)

2. **DenseNet121** (v1.0.0 - Released Jan 17, 2026)
   - Superior feature reuse through dense connections
   - Robust gradient flow for stable training
   - Excellent at capturing fine-grained patterns
   - **96.69% validation recall** (20 epochs)
   - [📄 Model Card](MODEL_CARD_DenseNet121.md)

3. **Ensemble Model** (v1.0.0 - Released Jan 17, 2026)
   - Combines both models for maximum safety and accuracy
   - **99% test accuracy** (67,137 classified samples)
   - **5.01% isolation rate** (3,540 samples for safety)
   - Production-ready system
   - [📄 Model Card](MODEL_CARD_ENSEMBLE.md)

### Safety-First Logic

```
┌───────────────────────────────┐
│         Input Image           │
└──────────┬────────────────────┘
           │
     ┌─────▼─────┐
     │   YOLO    │ (Optional ROI Extraction)
     │  (Feces)  │
     └─────┬─────┘
           │
    ┌──────┴───────┐
    │              │
┌───▼─────┐    ┌───▼────┐
│Efficient│    │DenseNet│
│ NetB0   │    │  121   │
└───┬─────┘    └───┬────┘
    │              │
    └──────┬───────┘
           │
    ┌──────▼──────┐
    │Safety Vote: │
    │ If EITHER   │
    │ model not   │
    │ confident   │
    │ → ISOLATE   │
    └──────┬──────┘
           │
    ┌──────▼───────┐
    │   Decision   │
    └──────────────┘
```

### Decision Rules

The system isolates chickens if **ANY** of the following conditions are met:

1. **Uncertainty Check**: Either model's maximum confidence < 50% → **ISOLATE** (unknown/out-of-distribution)
2. **Safety Vote**: Either model's healthy confidence < 80% → **ISOLATE** (potential disease)
3. **Disagreement**: Models predict different classes and either predicts disease → **ISOLATE**

## 🦠 Target Classes

| Class | Type | Description |
|-------|------|-------------|
| **Healthy** | Baseline | Normal fecal matter (high prevalence) |
| **Salmonella** | Bacterial | High contagion risk, gut health impact |
| **Coccidiosis** | Parasitic | Gut health issue, common in flocks |
| **New Castle Disease** | Viral | Respiratory/nervous system, highly contagious |

## 📊 Key Features

- **100% Recall Target**: Prevents false negatives (diseased → healthy)
- **Focal Loss**: Emphasizes hard examples and rare disease classes
- **False Negative Penalty**: 5x loss penalty for misclassifying diseased chickens as healthy
- **Class Weighting**: 2x emphasis on disease classes during training
- **Early Stopping**: Patience-based stopping on recall metric
- **Comprehensive Metrics**: Per-class recall, precision, F1, confusion matrices

## 🚀 Installation

### Prerequisites

- Python >= 3.12
- CUDA-capable GPU (recommended) or CPU
- 10GB+ free disk space

### Setup

**Using uv (recommended - faster):**
```bash
cd tokkatot_ai

# Install dependencies
uv pip install -e .

# For development
uv pip install -e ".[dev]"

# Or use uv sync
uv sync
```

**Using pip:**
```bash
cd tokkatot_ai

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Download Pre-trained Models

**Latest Release (v1.0.0 - January 17, 2026):**

1. **Ensemble Model** (Recommended for Production)
   - File: `ensemble_model.pth`
   - Size: ~200MB
   - Includes: EfficientNetB0 + DenseNet121
   - Performance: 99% test accuracy
   - Download from: [GitHub Releases](https://github.com/tokkatot/tokkatot_ai/releases)

2. **Individual Models** (Optional)
   - `EfficientNetB0_best.pth` - 98.05% recall (90 epochs)
   - `DenseNet121_best.pth` - 96.69% recall (20 epochs)
   - Use for edge deployment or custom ensemble configurations

**Place downloaded models in:**
```
tokkatot_ai/outputs/
├── ensemble_model.pth          # Main ensemble model
└── checkpoints/
    ├── EfficientNetB0_best.pth # Individual model
    └── DenseNet121_best.pth    # Individual model
```

### Verify the setup

```bash
python setup_check.py
```


## 📁 Project Structure

```
tokkatot_ai/
├── main.py              # Entry point
├── train.py             # Training script with recall-focused loss
├── inference.py         # Ensemble inference with safety logic
├── models.py            # EfficientNetB0 & DenseNet121 architectures
├── data_utils.py        # Data loading and preprocessing
├── pyproject.toml       # Dependencies
├── README.md            # This file
├── archive/
│   └── data/
│       ├── train/       # Training images
│       │   ├── Coccidiosis/
│       │   ├── Healthy/
│       │   ├── New Castle Disease/
│       │   └── Salmonella/
│       ├── val/         # Validation images
│       └── test/        # Test images
└── outputs/
    ├── checkpoints/     # Saved model weights
    ├── logs/            # Tensorboard logs
    └── ensemble_model.pth  # Final ensemble model
```

## 🎓 Training

### Check GPU

```bash
python check_gpu.py
```

### Start Training

```bash
python main.py train
```


### Continue Training
Resume from where left off:

```powershell
python main.py train --resume
```

### Training Configuration

The training script includes:
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Learning Rate**: 1e-4 (with ReduceLROnPlateau)
- **Loss Function**: RecallFocusedLoss (5x false negative penalty)
- **Optimizer**: AdamW with weight decay
- **Data Augmentation**: Rotation, flip, color jitter, affine transforms

### Monitoring Training

```bash
# View training logs with TensorBoard
tensorboard --logdir outputs/logs
```

Metrics tracked:
- Loss (train/val)
- Accuracy
- **Recall** (primary metric)
- Precision
- F1 Score
- Per-class recall

## 🔍 Inference

### Single Image Prediction

```bash
python main.py test path/to/image.jpg
```

### Programmatic Usage

```python
from inference import ChickenDiseaseDetector

# Initialize detector
detector = ChickenDiseaseDetector(
    model_path='outputs/ensemble_model.pth',
    healthy_threshold=0.80,      # 80% confidence required for healthy
    uncertainty_threshold=0.50    # 50% minimum confidence
)

# Simple prediction
result = detector.predict('image.jpg')
print(result)  # 'Healthy', 'Salmonella', 'ISOLATE', etc.

# Detailed prediction
detailed = detector.predict('image.jpg', return_details=True)
print(f"Classification: {detailed['classification']}")
print(f"Should Isolate: {detailed['should_isolate']}")
print(f"Action: {detailed['action']}")

# Both model predictions
print(f"EfficientNet: {detailed['models']['efficientnet']['prediction']}")
print(f"DenseNet: {detailed['models']['densenet']['prediction']}")

# Safety evaluation
is_safe, reason = detector.evaluate_safety('image.jpg')
print(f"Safe: {is_safe}, Reason: {reason}")
```

### Batch Prediction

```python
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.predict_batch(images, return_details=True)
```

## 📈 Performance Metrics

### Achieved Performance (Test Set)

| Model | Accuracy | Recall | Epochs | Status |
|-------|----------|--------|--------|--------|
| **Ensemble** | **99%** | **99%** | - | ✅ Production Ready |
| EfficientNetB0 | 98.05% | 98.05% | 90 | ✅ Production Ready |
| DenseNet121 | 96.69% | 96.69% | 20 | ✅ Production Ready |

**Ensemble Test Results (70,677 samples):**
- **Classified:** 67,137 (94.99%) with 99% accuracy
- **Isolated:** 3,540 (5.01%) for safety verification
- **Per-Class Recall:** Coccidiosis 100%, Healthy 98%, Newcastle 100%, Salmonella 100%

### Target Metrics

- **Recall (Disease Classes)**: > 99% ✅ **ACHIEVED**
- **Recall (Healthy)**: ≥ 85% ✅ **ACHIEVED (98%)**
- **Overall Accuracy**: ≥ 90% ✅ **ACHIEVED (99%)**
- **Isolation Rate**: 10-20% ✅ **ACHIEVED (5.01%)**

### Evaluation

The system provides:
- Confusion matrices per model
- Per-class recall scores
- Isolation statistics
- Model agreement rates

### Model Documentation

- [EfficientNetB0 Model Card](MODEL_CARD_EfficientNetB0.md) - Fast, lightweight model (v1.0.0)
- [DenseNet121 Model Card](MODEL_CARD_DenseNet121.md) - Dense connectivity model (v1.0.0)
- [Ensemble Model Card](MODEL_CARD_ENSEMBLE.md) - Combined system (v1.0.0)

## 🔧 Configuration

### Adjust Safety Thresholds

```python
detector = ChickenDiseaseDetector(
    model_path='outputs/ensemble_model.pth',
    healthy_threshold=0.85,      # Stricter: require 85% for healthy
    uncertainty_threshold=0.60    # Stricter: require 60% min confidence
)
```

**Threshold Guidelines:**
- **Higher healthy_threshold** (e.g., 0.85-0.90): More cautious, more isolations
- **Lower healthy_threshold** (e.g., 0.70-0.75): Less cautious, fewer isolations
- **Higher uncertainty_threshold** (e.g., 0.60-0.70): Reject more uncertain predictions
- **Lower uncertainty_threshold** (e.g., 0.40-0.45): Accept more uncertain predictions

### Training Hyperparameters

Edit [train.py](train.py):

```python
BATCH_SIZE = 32           # Increase for faster training (if GPU allows)
NUM_EPOCHS = 100          # Maximum epochs
LEARNING_RATE = 1e-4      # Base learning rate
IMG_SIZE = 224            # Input image size
```

## 🎯 Use Cases

### Farm Deployment

1. **Automated Monitoring**: Integrate with camera system for continuous fecal monitoring
2. **Early Detection**: Identify sick chickens before symptoms spread
3. **Quarantine Protocol**: Automatic isolation alerts for farm workers

### Edge Deployment (Raspberry Pi)

The EfficientNetB0 model can run independently on resource-constrained devices:

```python
# Use only EfficientNet for edge deployment
from models import EfficientNetB0Classifier

model = EfficientNetB0Classifier(num_classes=4)
model.load_state_dict(torch.load('outputs/checkpoints/EfficientNetB0_best.pth')['model_state_dict'])
```


## 📄 License

**© 2026 Tokkatot. All Rights Reserved.**

This software is proprietary and confidential. It is part of Tokkatot's Smart Chicken Farming Solutions and may not be copied, modified, distributed, or used without explicit written permission from Tokkatot.

## 🙏 Acknowledgments

- Developed by: Tokkatot Smart Farming Team
- Built with: PyTorch, torchvision (EfficientNet, DenseNet pretrained weights)
- Framework: PyTorch, scikit-learn

## 📧 Contact

**Tokkatot Smart Chicken Farming Solutions**

For business inquiries, technical support, or partnership opportunities:
- **Email**: [tokkatot.info@gmail.com](tokkatot.info@gmail.com)
- **Website**: [tokkatot.aztrolabe.com](tokkatot.aztrolabe.com)
- **AI Engineer**: [Sun Heng](sunhenglong@outlook.com)

---

**Proprietary Notice**: This software is part of Tokkatot's integrated smart farming ecosystem and is protected by intellectual property rights.
