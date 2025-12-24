# Multi-Stage Polyp Classification & Segmentation Pipeline

A comprehensive deep learning pipeline for polyp classification and segmentation with hierarchical classification approach.

## Pipeline Overview

### Stage 1: Backbone Pretraining
- **Dataset**: Dataset A (binary classification: hyperplastic vs adenoma)
- **Model**: EfficientNetV2-M
- **Goal**: Build strong general backbone representation and binary classifier
- **Output**: `checkpoints/stage1/backbone_pretrained_onA.pth` (state dict) + `checkpoints/stage1/best_model.pth`

### Stage 2: Segmentation Training
- **Dataset**: Dataset B (segmentation masks)
- **Architecture**: UNet/UNet++ with EfficientNetV2-M encoder
- **Stage 2A**: Freeze backbone, train decoder only
- **Stage 2B**: Unfreeze and fine-tune entire network
- **Outputs**: `checkpoints/stage2/backbone_after_seg.pth`, `checkpoints/stage2/seg_best.pth`

### Stage 3: Adenoma Subtype Classification
- **Input**: Stage 2 encoder finetuned during segmentation
- **Dataset**: Dataset C (adenoma morphology subtypes only)
- **Goal**: Classify adenomas into tubular, tubulovillous, or villous
- **Output**: `checkpoints/stage3/adenoma_subtype_classifier_best.pth`
- **Inference flow**:
  1. Stage 1 binary classifier decides Hyperplastic vs Adenoma
  2. If Adenoma → Stage 3 assigns final subtype label

## Project Structure

```
CODES/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── data/                       # Dataset directories
│   ├── dataset_a/             # Binary classification
│   ├── dataset_b/             # Segmentation
│   └── dataset_c/             # Adenoma subtypes
├── models/                     # Model architectures
│   ├── backbone.py            # EfficientNetV2 backbone wrapper
│   └── segmentation.py        # UNet/UNet++ models
├── datasets/                   # Dataset classes
│   ├── classification_dataset.py
│   └── segmentation_dataset.py
├── utils/                      # Utilities
│   ├── losses.py              # Focal, Dice, combined losses
│   ├── metrics.py             # Medical metrics
│   ├── samplers.py            # Weighted/stratified samplers
│   ├── scheduler.py           # Custom schedulers
│   ├── checkpoint.py          # Save/load utilities
│   └── augmentations.py       # Data augmentation
├── training/                   # Training scripts
│   ├── train_stage1.py        # Backbone pretraining (binary classifier)
│   ├── train_stage2.py        # Segmentation training
│   └── train_stage3.py        # Adenoma subtype classifier
├── inference/                  # Inference pipeline
│   └── hierarchical_inference.py
├── checkpoints/                # Model checkpoints
└── logs/                       # Training logs
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Stage 1: Pretrain Backbone
```bash
python training/train_stage1.py --config config.yaml
```

### Stage 2: Segmentation Training
```bash
# Stage 2A: Head-only
python training/train_stage2.py --config config.yaml --stage 2a

# Stage 2B: Full fine-tuning
python training/train_stage2.py --config config.yaml --stage 2b
```

### Stage 3: Hierarchical Classification
```bash
python training/train_stage3.py --config config.yaml
```

### Inference
```bash
python inference/hierarchical_inference.py --image path/to/image.jpg --config config.yaml
```

## Key Features

- **Stratified/Weighted Sampling**: Handles class imbalance effectively
- **Focal Loss**: Improves learning on hard examples and imbalanced classes
- **Medical Metrics**: Macro-F1, balanced accuracy, per-class recall
- **Early Stopping**: Prevents overfitting with patience-based monitoring
- **Hierarchical Inference**: Binary classification followed by subtype classification
- **Checkpoint Management**: Saves the best models based on validation metrics
- **Comprehensive Logging**: TensorBoard integration with confusion matrices

## Expected Outputs

All trained models will be saved in `checkpoints/`:
- `checkpoints/stage1/backbone_pretrained_onA.pth` - Stage 1 backbone (also used for Stage 2 init)
- `checkpoints/stage1/best_model.pth` - Stage 1 binary classifier checkpoint for inference
- `checkpoints/stage2/backbone_after_seg.pth` - Stage 2 encoder (finetuned)
- `checkpoints/stage2/seg_best.pth` - Best segmentation model weights
- `checkpoints/stage3/adenoma_subtype_classifier_best.pth` - Stage 3 classifier state dict
- `checkpoints/stage3/best_model.pth` - Stage 3 best checkpoint (with optimizer state)

## Configuration

Edit `config.yaml` to customize:
- Dataset paths
- Hyperparameters (learning rates, batch sizes, epochs)
- Model architectures
- Loss functions and weights
- Augmentation strategies
- Early stopping criteria
- Checkpoint locations

## Metrics Tracked

- **Classification**: macro-F1, balanced accuracy, ROC-AUC, per-class recall/precision
- **Segmentation**: Dice coefficient, IoU
- **Confusion matrices** for all classification tasks

## 🔬 Project Attribution & Credits

This project is powered by DataBioX (https://databiox.com), an AI-driven biomedical research initiative focused on developing clinically meaningful deep learning solutions for medical imaging and pathology.

### Principal Investigator (PI)
- Mojgan Forootan, MD  
  Principal Investigator (PI), clinical leadership, medical oversight, and strategic guidance  

### Core Contributors
- Hamidreza Rastad  
  Primary model concept, deep learning architecture design, and end-to-end development  

- Amir Akbari  
  Supporting development role, backend collaboration, and technical support  

- Mohammad Tashakoripour  
  Specialized nurse, procedural collaboration, data acquisition, and metadata generation  

### Co-Principal Investigator (Co-PI)
- Hamidreza Bolhasani, PhD  
  Co-Principal Investigator (Co-PI), AI and data leadership, data governance, dataset and metadata design, research ideation, and overall scientific coordination  

This repository represents an ongoing research and development effort. Additional contributors may be acknowledged as the project evolves.
