# Multi-Stage Polyp Classification Pipeline

Complete implementation of the multi-stage training pipeline for polyp classification and segmentation as specified.

## ✅ Implementation Status

All stages have been implemented:

- ✅ **Stage 1**: Backbone pretraining on binary classification
- ✅ **Stage 2A**: Segmentation head training (frozen encoder)
- ✅ **Stage 2B**: Full segmentation fine-tuning
- ✅ **Stage 3**: Adenoma subtype classifier (leverages Stage 2 encoder)
- ✅ **Inference**: Hierarchical classification pipeline

## 📁 Project Structure

```
CODES/
├── README.md                      # Main documentation
├── QUICKSTART.md                  # Quick start guide
├── config.yaml                    # Configuration file
├── requirements.txt               # Dependencies
│
├── datasets/                      # Dataset loaders
│   ├── __init__.py
│   ├── classification_dataset.py # Classification dataset
│   └── segmentation_dataset.py   # Segmentation dataset
│
├── models/                        # Model architectures
│   ├── __init__.py
│   ├── backbone.py               # EfficientNetV2-M classifier
│   └── segmentation.py           # UNet/UNet++ models
│
├── utils/                         # Utilities
│   ├── __init__.py
│   ├── augmentations.py          # Data augmentation
│   ├── checkpoint.py             # Checkpoint management
│   ├── losses.py                 # Loss functions
│   ├── metrics.py                # Medical metrics
│   ├── samplers.py               # Imbalanced samplers
│   └── scheduler.py              # LR schedulers
│
├── training/                      # Training scripts
│   ├── __init__.py
│   ├── train_stage1.py           # Stage 1 training
│   ├── train_stage2.py           # Stage 2 training
│   └── train_stage3.py           # Stage 3 training
│
├── inference/                     # Inference
│   ├── __init__.py
│   └── hierarchical_inference.py # Hierarchical classifier
│
├── checkpoints/                   # Model checkpoints (created during training)
├── logs/                          # TensorBoard logs (created during training)
└── data/                          # Datasets (user provided)
    ├── dataset_a/                # Binary classification
    ├── dataset_b/                # Segmentation
    └── dataset_c/                # Adenoma subtypes
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Prepare your datasets** (see QUICKSTART.md for structure)

3. **Update config.yaml** with your dataset paths

4. **Run training pipeline**:
   ```powershell
   # Stage 1: Backbone pretraining
   python training/train_stage1.py --config config.yaml
   
   # Stage 2A: Segmentation head training
   python training/train_stage2.py --config config.yaml --stage 2a
   
   # Stage 2B: Full fine-tuning
   python training/train_stage2.py --config config.yaml --stage 2b
   
   # Stage 3: Adenoma subtype classifier
   python training/train_stage3.py --config config.yaml
   ```

5. **Run inference**:
   ```powershell
   python inference/hierarchical_inference.py --image path/to/image.jpg --config config.yaml
   ```

## 📊 Key Features

### Data Handling
- ✅ Weighted/Stratified sampling for imbalanced data
- ✅ Comprehensive augmentation (standard + advanced with RandAugment)
- ✅ MixUp support for Stage 3 adenoma subtype classification
- ✅ Configurable dataset filtering for adenoma-only training

### Loss Functions
- ✅ Focal Loss for classification (handles imbalance)
- ✅ Dice Loss for segmentation
- ✅ Combined losses (Dice+BCE, Dice+Focal)
- ✅ Label Smoothing

### Medical Metrics
- ✅ Macro-F1, Balanced Accuracy
- ✅ Per-class Recall, Precision, F1
- ✅ ROC-AUC (binary and multi-class)
- ✅ Confusion matrices
- ✅ Dice coefficient, IoU for segmentation

### Training Utilities
- ✅ Early stopping (patience-based)
- ✅ Checkpoint management (best model tracking)
- ✅ Multiple LR schedulers (Cosine, Warmup, Polynomial)
- ✅ TensorBoard integration
- ✅ Differential learning rates (Stage 2B)

### Model Features
- ✅ EfficientNetV2-M backbone
- ✅ UNet/UNet++ segmentation models
- ✅ Encoder freezing/unfreezing utilities
- ✅ Pretrained weight loading

### Inference
- ✅ Hierarchical classification pipeline
- ✅ Single image and batch processing
- ✅ Adjustable threshold for sensitivity
- ✅ JSON output format

## 🎯 Expected Outputs

After completing all stages, you will have:

1. **checkpoints/stage1/backbone_pretrained_onA.pth** - Stage 1 backbone state dict
2. **checkpoints/stage1/best_model.pth** - Stage 1 binary classifier checkpoint
3. **checkpoints/stage2/backbone_after_seg.pth** - Stage 2 encoder weights
4. **checkpoints/stage2/seg_best.pth** - Best segmentation model
5. **checkpoints/stage3/adenoma_subtype_classifier_best.pth** - Stage 3 subtype classifier state dict
6. **checkpoints/stage3/best_model.pth** - Stage 3 checkpoint with optimizer state

## 📈 Performance Expectations

### Stage 1 (Binary Classification)
- Balanced Accuracy: 0.85-0.90
- Macro-F1: 0.83-0.88
- ROC-AUC: 0.90-0.95

### Stage 2 (Segmentation)
- Dice Coefficient: 0.75-0.85
- IoU: 0.65-0.75

### Stage 3 (Adenoma Subtypes)
- Macro-F1: 0.70-0.80

*Note: Actual performance depends on dataset quality and size*

## ⚙️ Configuration

All hyperparameters are configured in `config.yaml`:

- Dataset paths
- Model architectures
- Batch sizes and epochs
- Learning rates and optimizers
- Loss functions and weights
- Augmentation strategies
- Early stopping criteria

See comments in `config.yaml` for detailed explanations.

## 📚 Documentation

- **README.md** - This file (overview)
- **QUICKSTART.md** - Detailed quick start guide
- **config.yaml** - Fully commented configuration

## 🔬 Medical Context

This pipeline is designed for polyp classification with:
- **High recall/sensitivity** prioritized (medical safety)
- **Imbalanced class handling** (focal loss, weighted sampling)
- **Hierarchical classification** (reduces complexity)
- **Medical metrics** (macro-F1, balanced accuracy)

## 🛠️ Customization

### Adjust for Your Dataset

1. **Update paths** in `config.yaml`
2. **Adjust class names** if different
3. **Tune batch sizes** based on GPU memory
4. **Modify input sizes** (224 for classification, 512 for segmentation)
5. **Adjust thresholds** for recall/precision trade-off

### Extend Functionality

- Add more augmentations in `utils/augmentations.py`
- Implement new loss functions in `utils/losses.py`
- Add custom metrics in `utils/metrics.py`
- Try different architectures (update `models/`)

## 📦 Dependencies

- PyTorch ≥ 2.0.0
- timm (for EfficientNetV2)
- segmentation-models-pytorch (for UNet)
- Albumentations (for augmentation)
- scikit-learn (for metrics)
- TensorBoard (for logging)

See `requirements.txt` for complete list.


## 📄 License

This implementation is provided for research and educational purposes.

