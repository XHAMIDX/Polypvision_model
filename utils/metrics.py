"""
Medical imaging metrics for classification and segmentation
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    precision_score, 
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple


class MetricsTracker:
    """Track and compute medical imaging metrics"""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        """
        Args:
            num_classes: Number of classes
            class_names: Names of classes (optional)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all tracked values"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, probs: torch.Tensor = None):
        """
        Update metrics with batch results
        
        Args:
            preds: Predicted class indices [B]
            targets: Ground truth labels [B]
            probs: Class probabilities [B, C] (optional, for AUC)
        """
        self.predictions.extend(preds.detach().cpu().numpy().tolist())
        self.targets.extend(targets.detach().cpu().numpy().tolist())
        
        if probs is not None:
            self.probabilities.extend(probs.detach().cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metric names and values
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Safety check: if no predictions, return zeros
        if len(preds) == 0 or len(targets) == 0:
            return {
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'macro_recall': 0.0,
                'macro_precision': 0.0,
                'roc_auc': 0.0,
            }
        
        # Overall accuracy
        metrics['accuracy'] = (preds == targets).mean()
        
        # Balanced accuracy (important for imbalanced data)
        metrics['balanced_accuracy'] = balanced_accuracy_score(targets, preds)
        
        # Macro F1 (treats all classes equally)
        metrics['macro_f1'] = f1_score(targets, preds, average='macro', zero_division=0)
        
        # Weighted F1
        metrics['weighted_f1'] = f1_score(targets, preds, average='weighted', zero_division=0)
        
        # Macro recall (sensitivity)
        metrics['macro_recall'] = recall_score(targets, preds, average='macro', zero_division=0)
        
        # Macro precision
        metrics['macro_precision'] = precision_score(targets, preds, average='macro', zero_division=0)
        
        # Per-class metrics
        per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
        per_class_recall = recall_score(targets, preds, average=None, zero_division=0)
        per_class_precision = precision_score(targets, preds, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(per_class_f1):
                metrics[f'f1_{class_name}'] = per_class_f1[i]
                metrics[f'recall_{class_name}'] = per_class_recall[i]
                metrics[f'precision_{class_name}'] = per_class_precision[i]
        
        # ROC-AUC (if probabilities available)
        if len(self.probabilities) > 0:
            probs = np.array(self.probabilities)
            
            if self.num_classes == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(targets, probs[:, 1])
            else:
                # Multi-class (OvR)
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        targets, probs, multi_class='ovr', average='macro'
                    )
                except ValueError:
                    # Not all classes present in targets
                    metrics['roc_auc_ovr'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.targets, self.predictions)
    
    def print_report(self):
        """Print classification report"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            self.targets, 
            self.predictions, 
            target_names=self.class_names,
            zero_division=0
        ))
        
        print("\nConfusion Matrix:")
        cm = self.get_confusion_matrix()
        print(cm)
        
        metrics = self.compute()
        print(f"\nBalanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


class SegmentationMetrics:
    """Metrics for segmentation tasks"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Threshold for converting probabilities to binary masks
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset tracked values"""
        self.dice_scores = []
        self.iou_scores = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch results
        
        Args:
            preds: Predicted masks [B, 1, H, W] (probabilities)
            targets: Ground truth masks [B, 1, H, W]
        """
        preds = (preds > self.threshold).float()
        
        batch_size = preds.shape[0]
        
        for i in range(batch_size):
            pred = preds[i].view(-1)
            target = targets[i].view(-1)
            
            # Dice coefficient
            intersection = (pred * target).sum()
            dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
            self.dice_scores.append(dice.item())
            
            # IoU (Jaccard)
            union = pred.sum() + target.sum() - intersection
            iou = intersection / (union + 1e-8)
            self.iou_scores.append(iou.item())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics
        
        Returns:
            Dictionary of metric names and values
        """
        return {
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
        }
    
    def print_report(self):
        """Print segmentation metrics"""
        metrics = self.compute()
        print("\n" + "="*60)
        print("SEGMENTATION METRICS")
        print("="*60)
        print(f"Dice Coefficient: {metrics['dice']:.4f}")
        print(f"IoU (Jaccard): {metrics['iou']:.4f}")


def compute_dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Compute Dice coefficient for a single prediction
    
    Args:
        pred: Predicted mask [H, W] or [1, H, W]
        target: Ground truth mask [H, W] or [1, H, W]
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient value
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-8) -> float:
    """
    Compute Intersection over Union
    
    Args:
        pred: Predicted mask [H, W] or [1, H, W]
        target: Ground truth mask [H, W] or [1, H, W]
        smooth: Smoothing factor
    
    Returns:
        IoU value
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def test_metrics():
    """Test metrics computation"""
    print("=== Testing Classification Metrics ===")
    
    # Create dummy predictions and targets
    num_samples = 100
    num_classes = 3
    
    preds = torch.randint(0, num_classes, (num_samples,))
    targets = torch.randint(0, num_classes, (num_samples,))
    probs = torch.softmax(torch.randn(num_samples, num_classes), dim=1)
    
    tracker = MetricsTracker(num_classes, class_names=['Class_A', 'Class_B', 'Class_C'])
    tracker.update(preds, targets, probs)
    
    metrics = tracker.compute()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    tracker.print_report()
    
    print("\n=== Testing Segmentation Metrics ===")
    
    # Create dummy segmentation outputs
    batch_size = 4
    pred_masks = torch.sigmoid(torch.randn(batch_size, 1, 64, 64))
    gt_masks = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
    
    seg_metrics = SegmentationMetrics(threshold=0.5)
    seg_metrics.update(pred_masks, gt_masks)
    
    seg_results = seg_metrics.compute()
    print(f"Dice: {seg_results['dice']:.4f}")
    print(f"IoU: {seg_results['iou']:.4f}")
    
    seg_metrics.print_report()
    
    print("\nAll metrics tests passed!")


if __name__ == '__main__':
    test_metrics()
