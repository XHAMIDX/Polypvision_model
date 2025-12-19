# AI Agent Instructions for Polyp Classification Pipeline

This document provides comprehensive guidance for AI agents (Copilot, Claude, etc.) to effectively maintain, extend, and debug this multi-stage polyp classification and segmentation system.

---

## 1. Project Architecture Overview

### 1.1 High-Level Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    5-Stage Training Pipeline                    │
└─────────────────────────────────────────────────────────────────┘

STAGE 1: BACKBONE PRETRAINING
├─ Task: Binary classification (Adenomatous vs Hyperplastic)
├─ Dataset: Dataset A (5,843 images, 43.18% / 56.82% split)
├─ Model: EfficientNetV2-M (52.8M parameters)
├─ Loss: Focal Loss (α=[weight_class0, weight_class1], γ=2.0)
├─ Sampling: WeightedRandomSampler (per-sample weight = 1/class_frequency)
├─ Output: checkpoints/stage1/backbone_pretrained_onA.pth (encoder) + checkpoints/stage1/best_model.pth (full checkpoint)
└─ Duration: ~50 epochs, converges in 2-3 epochs

STAGE 2: SEGMENTATION HEAD TRAINING
├─ 2A: Frozen encoder (backbone frozen, only UNet head trains)
├─ 2B: Full fine-tune (entire UNet model trains)
├─ Task: Pixel-level polyp boundary detection
├─ Model: UNet with EfficientNetV2-M encoder
├─ Loss: Dice + BCE combined
├─ Dataset: Segmentation pairs (images + masks)
└─ Output: checkpoints/stage2/backbone_after_seg.pth, checkpoints/stage2/seg_best.pth

STAGE 3: ADENOMA SUBTYPE CLASSIFICATION
├─ Task: Classify adenomatous polyps into tubular / tubulovillous / villous
├─ Dataset: Dataset C (adenoma-only samples)
├─ Model: EfficientNetV2-M classifier initialised from Stage 2 encoder
├─ Loss: Focal Loss with MixUp augmentation + weighted sampling
├─ Sampling: WeightedRandomSampler across adenoma subtypes
└─ Output: checkpoints/stage3/adenoma_subtype_classifier_best.pth (state dict) + checkpoints/stage3/best_model.pth (full checkpoint)

INFERENCE: Hierarchical 2-Stage Classification
├─ Input: Single image or batch directory
├─ Stage 1: Binary decision (Is this polyp adenomatous?)
├─ Stage 3: If adenomatous → Classify adenoma subtype
└─ Output: JSON with class probabilities and confidence scores
```

### 1.2 Directory Structure

```
e:\dr\CODES/
├── config.yaml                 ← CENTRAL CONFIG (hyperparameters)
├── requirements.txt            ← Python dependencies
├── README.md                   ← Project overview
├── QUICKSTART.md              ← Getting started guide
├── GET_STARTED.md             ← Detailed setup
├── PROJECT_SUMMARY.md         ← Architecture details
│
├── datasets/                  ← DATA HANDLING
│   ├── classification_dataset.py    (274 lines) - Image classification datasets
│   ├── segmentation_dataset.py      (165 lines) - Image-mask pair datasets
│   └── __init__.py
│
├── models/                    ← ARCHITECTURES
│   ├── backbone.py            (234 lines) - EfficientNetV2-M classifier wrapper
│   ├── segmentation.py        (178 lines) - UNet/UNet++ segmentation models
│   └── __init__.py
│
├── utils/                     ← UTILITIES
│   ├── losses.py              (304 lines) - FocalLoss, DiceLoss, combined losses
│   ├── metrics.py             (291 lines) - MetricsTracker, SegmentationMetrics
│   ├── augmentations.py       (253 lines) - Albumentations pipelines
│   ├── checkpoint.py          (258 lines) - CheckpointManager, EarlyStopping
│   ├── samplers.py            (89 lines)  - WeightedRandomSampler, StratifiedBatchSampler
│   ├── scheduler.py           (78 lines)  - CosineWarmup scheduler
│   ├── visualization.py       (156 lines) - Plotting utilities
│   └── __init__.py
│
├── training/                  ← TRAINING SCRIPTS
│   ├── train_stage1.py        (212 lines) - Backbone pretraining
│   ├── train_stage2.py        (174 lines) - Segmentation training
│   ├── train_stage3.py        (363 lines) - Adenoma subtype classifier
│   └── __init__.py
│
├── inference/                 ← INFERENCE PIPELINE
│   ├── hierarchical_inference.py (274 lines) - 2-stage inference
│   └── __init__.py
│
├── checkpoints/               ← MODEL STORAGE
│   ├── stage1/backbone_pretrained_onA.pth
│   ├── stage1/best_model.pth
│   ├── stage2/backbone_after_seg.pth
│   ├── stage2/seg_best.pth
│   ├── stage3/adenoma_subtype_classifier_best.pth
│   └── stage3/best_model.pth
│
└── test_imports.py            ← Quick validation script
```

### 1.3 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deep Learning** | PyTorch 2.0+ | GPU-accelerated tensor operations & training |
| **Backbone** | timm (EfficientNetV2-M) | Pre-trained 52.8M parameter feature extractor |
| **Segmentation** | segmentation-models-pytorch | UNet/UNet++ with arbitrary encoders |
| **Data Loading** | torch.utils.data | Efficient batching with worker processes |
| **Augmentation** | Albumentations | Fast, differentiable image transformations |
| **Metrics** | sklearn.metrics | Confusion matrix, ROC-AUC, precision-recall |
| **Visualization** | Matplotlib, TensorBoard | Training monitoring & diagnosis |
| **Config** | PyYAML | Centralized hyperparameter management |

---

## 2. Critical Code Patterns & Conventions

### 2.1 Module Imports (CRITICAL for sys.path)

**ALL training and inference scripts MUST include sys.path handling:**

```python
import os
import sys

# Add parent directory to path - REQUIRED for modular imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_efficientnetv2_classifier
from datasets import create_classification_dataset
from utils import get_loss_function, MetricsTracker
```

**Why:** This ensures the script can find sibling packages (models, datasets, utils) regardless of working directory.

**When modifying:** Any new training script or utility that does inter-package imports MUST have this at the top.

---

### 2.2 Focal Loss with Flexible Alpha Parameter

**Pattern (in utils/losses.py):**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        # alpha can be:
        # - None: uniform weights
        # - float/int: scalar weight for positive class
        # - list/tuple: [weight_class0, weight_class1, ...]
        # - torch.Tensor: pre-created weight tensor
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # CRITICAL: Handle alpha type flexibility
        if self.alpha is not None and isinstance(self.alpha, torch.Tensor):
            # Use tensor directly as weight
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                     weight=self.alpha)
        elif self.alpha is not None and isinstance(self.alpha, (list, tuple)):
            # Convert list/tuple to tensor
            alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32, 
                                       device=inputs.device)
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                     weight=alpha_tensor)
        else:
            # No alpha weighting
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weight: (1 - p_t)^gamma
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()
```

**Key Points:**
- **Always check alpha type** before using in cross_entropy
- **Never pass float alpha directly** to cross_entropy weight parameter
- **Convert lists/tuples to tensors** on correct device

---

### 2.3 Tensor to Numpy Conversion

**CRITICAL: Always use .detach() before .numpy()**

```python
# ❌ WRONG - causes RuntimeError if tensor requires_grad=True
pred = model(x)
numpy_array = pred.cpu().numpy()  # ERROR!

# ✅ CORRECT
pred = model(x)
numpy_array = pred.detach().cpu().numpy()

# In metrics.py - ALWAYS detach tensors:
predictions = preds.detach().cpu().numpy().tolist()
targets = targets.detach().cpu().numpy().tolist()
```

**Why:** PyTorch tracks gradients for tensors with `requires_grad=True`. These can't convert to numpy directly.

---

### 2.4 Dataset Folder Structure Convention

**MUST follow this exact structure:**

```
/path/to/dataset/
├── ClassName1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── ClassName2/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── ClassName3/
    └── ...
```

**Key Rules:**
1. **Folder names ARE class labels** - Code reads `os.listdir()` for class names
2. **All images in one folder** - No subdirectories inside class folders
3. **Class folders must match exactly** across different datasets if merging
4. **Supported formats** - .jpg, .jpeg, .png, .tif, .tiff (case-insensitive)

**In config.yaml:**
```yaml
dataset:
  root_dir: /path/to/dataset              # Parent folder containing class folders
  classification_type: binary              # 'binary' or 'multiclass'
```

**Dataset class will:**
1. Read all directories in `root_dir`
2. Treat each directory name as a class label
3. Assign class indices in sorted order (alphabetical)
4. Print class distribution on load

---

### 2.5 Training Loop Pattern

**Standard pattern (used in all train_stage*.py files):**

```python
def main(args):
    # 1. Load configuration
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Create dataset and dataloader
    dataset = create_classification_dataset(config, split='train')
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2)
    
    # 3. Create model
    model = create_efficientnetv2_classifier(num_classes=config['num_classes'])
    model = model.to(device)
    
    # 4. Setup training components
    loss_fn = get_loss_function(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, 
                                      total_epochs=config['epochs'], 
                                      num_batches=len(train_loader))
    
    # 5. Training loop
    for epoch in range(config['epochs']):
        # Train phase
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            # Validation logic
            pass
        
        # Checkpointing
        checkpoint_manager.save(epoch, metrics, model, optimizer)
```

---

### 2.6 Common Configuration Pattern

**In config.yaml - all hyperparameters centralized:**

```yaml
model:
  backbone: "efficientnetv2_m"
  num_classes: 2

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  warmup_epochs: 5

loss:
  name: "focal"
  focal_gamma: 2.0
  focal_alpha: [0.6, 0.4]  # Per-class weights (inverse frequency)

augmentation:
  input_size: 224
  standard_pipeline: true  # Use standard or advanced augmentations
  mixup: false

checkpoint:
  save_dir: "checkpoints"
  patience: 5  # Early stopping patience
```

**Access in code:**
```python
with open('config.yaml') as f:
    config = yaml.safe_load(f)

num_classes = config['model']['num_classes']
learning_rate = config['training']['learning_rate']
```

---

## 3. How to Extend the Pipeline

### 3.1 Add a New Classification Stage

**Steps:**

1. **Create new training script**: `training/train_stage4.py`
  - Copy structure from `train_stage3.py`
   - Add sys.path handling at top
   - Modify dataset creation to load your data
   - Update checkpoint saving path

2. **Add config entries** to `config.yaml`:
   ```yaml
   stage4:
     epochs: 50
     batch_size: 32
     learning_rate: 1e-5
   ```

3. **Create dataset** (if needed):
   - Ensure folder structure matches convention
   - Verify class distribution
   - Update `datasets/classification_dataset.py` if special handling needed

4. **Run training**:
  ```bash
  python training/train_stage4.py --config config.yaml
  ```

---

### 3.2 Add a New Loss Function

**Steps:**

1. **Add to `utils/losses.py`**:
   ```python
   class MyCustomLoss(nn.Module):
       def __init__(self, **kwargs):
           super().__init__()
           self.param = kwargs.get('param', default_value)
       
       def forward(self, inputs, targets):
           # Implementation
           loss = ...
           return loss.mean()
   ```

2. **Register in `get_loss_function()`**:
   ```python
   def get_loss_function(config):
       loss_name = config['loss']['name']
       if loss_name == 'my_custom_loss':
           return MyCustomLoss(**config['loss'])
   ```

3. **Add config entry**:
   ```yaml
   loss:
     name: "my_custom_loss"
     param: value
   ```

---

### 3.3 Add New Metrics

**Steps:**

1. **Add method to `MetricsTracker` in `utils/metrics.py`**:
   ```python
   def update(self, predictions, targets, probabilities=None):
       self.predictions.extend(preds.detach().cpu().numpy().tolist())
       # Add new metric calculation
       self.my_new_metric = calculate_my_metric(...)
   ```

2. **Return in `get_metrics()`**:
   ```python
   def get_metrics(self):
       return {
           'macro_f1': self.macro_f1,
           'my_new_metric': self.my_new_metric
       }
   ```

3. **Use in training loop**:
   ```python
   metrics = metrics_tracker.get_metrics()
   print(f"My New Metric: {metrics['my_new_metric']}")
   ```

---

## 4. Debugging Common Issues

### 4.1 Import Errors

**Error:** `ModuleNotFoundError: No module named 'models'`

**Solution:** Add sys.path handling at script top:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**Verify:** Run `test_imports.py` to validate all imports work.

---

### 4.2 CUDA / Device Errors

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size in config.yaml
2. Use mixed precision training
3. Reduce image size in augmentations

**Error:** `Expected all tensors to be on the same device`

**Solution:** Ensure inputs are on same device as model:
```python
images = images.to(device)
labels = labels.to(device)
outputs = model(images)  # All on same device
```

---

### 4.3 Class Distribution Warnings

**When running training, expect to see:**
```
Loaded 5843 images from /path/to/dataset
Classes: {'Adenomatous': 0, 'Hyperplastic': 1}
Class distribution:
  Adenomatous: 2523 samples (43.18%)
  Hyperplastic: 3320 samples (56.82%)
```

This is **EXPECTED and GOOD** - shows data loading works correctly.

---

### 4.4 DataLoader Worker Warnings

**Warning:** `This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2`

**Solution:** In config or training script, reduce num_workers:
```python
train_loader = DataLoader(dataset, batch_size=batch_size, 
                         num_workers=2, shuffle=True)  # Was 4
```

---

### 4.5 Tensor Type Mismatches

**Error:** `TypeError: cross_entropy_loss() weight must be Tensor, not float`

**Solution:** Check FocalLoss alpha parameter - ensure it's converted to tensor:
```python
if isinstance(self.alpha, (list, tuple)):
    alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32, 
                               device=inputs.device)
    # Use alpha_tensor, not self.alpha
```

---

### 4.6 Numpy Conversion Errors

**Error:** `RuntimeError: Can't call numpy() on Tensor that requires grad`

**Solution:** Always use .detach():
```python
# ❌ WRONG
numpy_data = tensor.cpu().numpy()

# ✅ CORRECT  
numpy_data = tensor.detach().cpu().numpy()
```

---

## 5. Dataset Management

### 5.1 Using Different Datasets

**Change dataset in config.yaml:**
```yaml
dataset:
  root_dir: /path/to/new/dataset  # ← Only change needed
  batch_size: 32
  classification_type: binary
```

**Folder structure must be:**
```
/path/to/new/dataset/
├── Adenomatous/        ← Folder name = class label
├── Hyperplastic/       ← Folder name = class label
└── OtherClass/         ← Optional additional classes
```

**Then run:**
```bash
python training/train_stage1.py --config config.yaml
```

✅ **Automatic fine-tuning:** Will load pretrained weights and adapt to new dataset

---

### 5.2 Creating Segmentation Datasets

**Required structure:**
```
/path/to/segmentation_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── masks/
    ├── image_001_mask.png  ← Matching name with suffix
    ├── image_002_mask.png
    └── ...
```

**Code automatically:**
- Pairs images with masks by filename
- Converts masks to 0-1 range
- Verifies all images have corresponding masks

---

## 6. Quick Reference Commands

```bash
# Test that all imports work
python test_imports.py

# Stage 1: Backbone pretraining
python training/train_stage1.py --config config.yaml

# Stage 2: Segmentation training
python training/train_stage2.py --config config.yaml --stage 2a
python training/train_stage2.py --config config.yaml --stage 2b

# Stage 3: Adenoma subtype classification
python training/train_stage3.py --config config.yaml

# Run inference on single image
python inference/hierarchical_inference.py \
  --image /path/to/image.jpg \
  --config config.yaml

# Run inference on directory
python inference/hierarchical_inference.py \
  --image_dir /path/to/images/ \
  --config config.yaml \
  --output results.json
```

---

## 7. Performance Optimization Tips

### 7.1 Training Speedup

1. **Reduce num_workers if system is slow:**
   - Default: 4 workers
   - For slower systems: 2 workers
   - For very slow systems: 0 workers

2. **Use mixed precision** (if GPU supports it):
   ```python
   from torch.cuda.amp import GradScaler, autocast
   scaler = GradScaler()
   with autocast():
       outputs = model(images)
       loss = loss_fn(outputs, labels)
   ```

3. **Reduce image size in augmentations:**
   ```yaml
   augmentation:
     input_size: 224  # Reduce to 192 or 160 for faster training
   ```

---

### 7.2 Memory Optimization

1. **Gradient checkpointing:**
   ```python
   model = torch.utils.checkpoint.checkpoint(model, images)
   ```

2. **Smaller batch size:**
   ```yaml
   training:
     batch_size: 16  # Reduce from 32
   ```

3. **Freeze backbone in Stage 2A:**
   - Already done in code - freezing encoder reduces memory

---

## 8. Production Checklist

- [ ] All imports pass `python test_imports.py`
- [ ] Training runs without errors for at least 5 epochs
- [ ] Validation metrics improve across epochs
- [ ] Checkpoints save correctly
- [ ] Inference outputs valid JSON
- [ ] Cross-validate with external dataset
- [ ] Document any dataset-specific quirks
- [ ] Version control config.yaml changes

---

## 9. Contact Points for Future Modifications

### 9.1 If adding new model architecture:
- Modify: `models/backbone.py` and/or `models/segmentation.py`
- Register in: `models/__init__.py`
- Test: Create new model instance in training script

### 9.2 If adding new augmentation:
- Modify: `utils/augmentations.py` (get_transforms function)
- Register in: Config for standard vs advanced pipeline

### 9.3 If changing dataset structure:
- Modify: `datasets/classification_dataset.py`
- Key method: `_load_images()` - adjust if folder structure changes

### 9.4 If adding new training stage:
- Template: Copy `training/train_stage1.py` structure
- Don't forget: sys.path handling at top
- Register: In config.yaml with stage-specific hyperparameters

---

## 10. Final Notes for AI Agents

1. **Always prioritize .detach() before .numpy()** - This is the most common tensor error
2. **Check sys.path at top of any new script** - Prevents ModuleNotFoundError
3. **Respect the config.yaml pattern** - All hyperparameters should be configurable
4. **Test imports first** - Run test_imports.py before major changes
5. **Preserve checkpoint compatibility** - Don't change model architecture without migration logic
6. **Document dataset assumptions** - If code assumes specific class names, document them clearly
7. **Use Focal Loss alpha properly** - It's the most complex parameter that needs type flexibility

**Happy coding!** 🚀
