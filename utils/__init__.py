"""
Utility functions initialization
"""

from .losses import (
    FocalLoss,
    DiceLoss,
    DiceBCELoss,
    DiceFocalLoss,
    LabelSmoothingCrossEntropy,
    get_loss_function
)

from .metrics import (
    MetricsTracker,
    SegmentationMetrics,
    compute_dice_coefficient,
    compute_iou
)

from .samplers import (
    WeightedRandomSampler,
    StratifiedBatchSampler,
    create_weighted_sampler,
    create_stratified_sampler
)

from .augmentations import (
    get_classification_train_transform,
    get_classification_val_transform,
    get_segmentation_train_transform,
    get_segmentation_val_transform,
    MixUpTransform
)

from .checkpoint import (
    CheckpointManager,
    EarlyStopping
)

from .scheduler import (
    CosineWarmupScheduler,
    PolynomialLR,
    get_scheduler
)

__all__ = [
    # Losses
    'FocalLoss',
    'DiceLoss',
    'DiceBCELoss',
    'DiceFocalLoss',
    'LabelSmoothingCrossEntropy',
    'get_loss_function',
    
    # Metrics
    'MetricsTracker',
    'SegmentationMetrics',
    'compute_dice_coefficient',
    'compute_iou',
    
    # Samplers
    'WeightedRandomSampler',
    'StratifiedBatchSampler',
    'create_weighted_sampler',
    'create_stratified_sampler',
    
    # Augmentations
    'get_classification_train_transform',
    'get_classification_val_transform',
    'get_segmentation_train_transform',
    'get_segmentation_val_transform',
    'MixUpTransform',
    
    # Checkpoint
    'CheckpointManager',
    'EarlyStopping',
    
    # Scheduler
    'CosineWarmupScheduler',
    'PolynomialLR',
    'get_scheduler',
]
