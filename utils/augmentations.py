"""
Data augmentation pipelines using Albumentations (v1.4+)
Updated for new Albumentations API
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


def get_classification_train_transform(input_size: int = 224, advanced: bool = False):
    """
    Training augmentation pipeline for classification
    
    Args:
        input_size: Target image size
        advanced: Use more aggressive augmentations
    
    Returns:
        Albumentations transform
    """
    if advanced:
        # Advanced augmentations for Stage 3 adenoma subtype classification
        transforms = [
            A.Compose([
                A.RandomCrop(height=int(input_size*0.9), width=int(input_size*0.9), p=0.5),
                A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=0, p=1.0),
            ]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=45, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=7),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05),
            ], p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=4.0),
                A.Equalize(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    else:
        # Standard augmentations for Stage 1 and 3A
        transforms = [
            A.Resize(height=input_size, width=input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=30, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=3),
            ], p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms)


def get_classification_val_transform(input_size: int = 224):
    """
    Validation/test transform for classification (no augmentation)
    
    Args:
        input_size: Target image size
    
    Returns:
        Albumentations transform
    """
    return A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_segmentation_train_transform(input_size: int = 512):
    """
    Training augmentation for segmentation (must preserve mask alignment)
    
    Args:
        input_size: Target image size
    
    Returns:
        Albumentations transform (handles both image and mask)
    """
    return A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=30, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=5),
        ], p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_segmentation_val_transform(input_size: int = 512):
    """
    Validation/test transform for segmentation (no augmentation)
    
    Args:
        input_size: Target image size
    
    Returns:
        Albumentations transform (handles both image and mask)
    """
    return A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class MixUpTransform:
    """
    MixUp augmentation for classification
    Apply this after batching
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        """
        Apply MixUp to a batch
        
        Args:
            batch_images: Tensor of shape [B, C, H, W]
            batch_labels: Tensor of shape [B] (class indices)
        
        Returns:
            Mixed images and soft labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = batch_images.size(0)
        index = np.random.permutation(batch_size)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        labels_a, labels_b = batch_labels, batch_labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def test_augmentations():
    """Test augmentation pipelines"""
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    
    # Test classification transforms
    print("Testing classification transforms...")
    clf_train = get_classification_train_transform(224, advanced=False)
    clf_train_adv = get_classification_train_transform(224, advanced=True)
    clf_val = get_classification_val_transform(224)
    
    result = clf_train(image=dummy_img)
    print(f"Classification train output shape: {result['image'].shape}")
    
    result = clf_train_adv(image=dummy_img)
    print(f"Classification train (advanced) output shape: {result['image'].shape}")
    
    result = clf_val(image=dummy_img)
    print(f"Classification val output shape: {result['image'].shape}")
    
    # Test segmentation transforms
    print("\nTesting segmentation transforms...")
    seg_train = get_segmentation_train_transform(512)
    seg_val = get_segmentation_val_transform(512)
    
    result = seg_train(image=dummy_img, mask=dummy_mask)
    print(f"Segmentation train - image shape: {result['image'].shape}, mask shape: {result['mask'].shape}")
    
    result = seg_val(image=dummy_img, mask=dummy_mask)
    print(f"Segmentation val - image shape: {result['image'].shape}, mask shape: {result['mask'].shape}")
    
    print("\nAll augmentation tests passed!")


if __name__ == '__main__':
    test_augmentations()
