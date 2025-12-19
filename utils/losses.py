"""
Loss functions for classification and segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape [B, C] where C = num_classes
            targets: Ground truth labels of shape [B]
        
        Returns:
            Focal loss value
        """
        # Compute base cross entropy loss
        if self.alpha is not None and isinstance(self.alpha, torch.Tensor):
            # Use weighted cross entropy if alpha is a tensor
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        else:
            # Use unweighted cross entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            
            # Apply per-sample alpha weighting if provided as scalar or list
            if self.alpha is not None:
                if isinstance(self.alpha, (list, tuple)):
                    alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
                    ce_loss = ce_loss * alpha_tensor[targets]
        
        # Apply focal term
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape [B, 1, H, W] (after sigmoid)
            targets: Ground truth masks of shape [B, 1, H, W]
        
        Returns:
            Dice loss value
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross Entropy Loss
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        """
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            smooth: Smoothing factor for Dice
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits of shape [B, 1, H, W]
            targets: Ground truth masks of shape [B, 1, H, W]
        
        Returns:
            Combined loss value
        """
        # Apply sigmoid for Dice loss
        inputs_sigmoid = torch.sigmoid(inputs)
        dice = self.dice_loss(inputs_sigmoid, targets)
        
        # Use raw logits for BCE
        bce = self.bce_loss(inputs, targets)
        
        return self.dice_weight * dice + self.bce_weight * bce


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss for segmentation with severe imbalance
    """
    
    def __init__(
        self, 
        dice_weight: float = 0.5, 
        focal_weight: float = 0.5,
        gamma: float = 2.0,
        smooth: float = 1.0
    ):
        """
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            gamma: Focal loss gamma parameter
            smooth: Smoothing factor for Dice
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits of shape [B, 1, H, W]
            targets: Ground truth masks of shape [B, 1, H, W]
        
        Returns:
            Combined loss value
        """
        # Dice loss
        inputs_sigmoid = torch.sigmoid(inputs)
        dice = self.dice_loss(inputs_sigmoid, targets)
        
        # Focal loss for binary segmentation
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal = ((1 - pt) ** self.gamma * bce_loss).mean()
        
        return self.dice_weight * dice + self.focal_weight * focal


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Helps prevent overconfidence
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (0-1)
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape [B, C]
            targets: Ground truth labels of shape [B]
        
        Returns:
            Label smoothed cross entropy loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * (self.smoothing / (num_classes - 1))
        
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss


def get_loss_function(loss_config: dict, device: Union[str, torch.device] = 'cuda'):
    """
    Factory function to create loss function from config
    
    Args:
        loss_config: Loss configuration dictionary
        device: Device to place tensors on
    
    Returns:
        Loss function instance
    """
    # Support torch.device inputs transparently
    device_name = device.type if isinstance(device, torch.device) else device

    loss_name = loss_config['name'].lower()
    
    if loss_name == 'crossentropy' or loss_name == 'ce':
        # Handle class weights if provided
        if loss_config.get('use_class_weights', False) and 'class_weights' in loss_config:
            weights = torch.FloatTensor(loss_config['class_weights']).to(device_name)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()
    
    elif loss_name == 'focalloss' or loss_name == 'focal':
        alpha = None
        if loss_config.get('use_class_weights', False) and 'class_weights' in loss_config:
            alpha = torch.FloatTensor(loss_config['class_weights']).to(device_name)
        elif 'focal_alpha' in loss_config:
            # Single alpha value or list
            alpha = loss_config['focal_alpha']
            if isinstance(alpha, (list, tuple)):
                alpha = torch.FloatTensor(alpha).to(device_name)
        
        gamma = loss_config.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_name == 'diceloss' or loss_name == 'dice':
        smooth = loss_config.get('smooth', 1.0)
        return DiceLoss(smooth=smooth)
    
    elif loss_name == 'dicebce':
        dice_weight = loss_config.get('dice_weight', 0.5)
        bce_weight = loss_config.get('bce_weight', 0.5)
        smooth = loss_config.get('smooth', 1.0)
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight, smooth=smooth)
    
    elif loss_name == 'dicefocal':
        dice_weight = loss_config.get('dice_weight', 0.5)
        focal_weight = loss_config.get('focal_weight', 0.5)
        gamma = loss_config.get('focal_gamma', 2.0)
        smooth = loss_config.get('smooth', 1.0)
        return DiceFocalLoss(
            dice_weight=dice_weight, 
            focal_weight=focal_weight,
            gamma=gamma,
            smooth=smooth
        )
    
    elif loss_name == 'labelsmoothing':
        smoothing = loss_config.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def test_losses():
    """Test loss functions"""
    batch_size = 4
    num_classes = 3
    
    # Test classification losses
    print("=== Testing Classification Losses ===")
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    ce_loss = nn.CrossEntropyLoss()
    print(f"CrossEntropyLoss: {ce_loss(logits, targets).item():.4f}")
    
    focal_loss = FocalLoss(gamma=2.0)
    print(f"FocalLoss: {focal_loss(logits, targets).item():.4f}")
    
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    print(f"LabelSmoothingCE: {ls_loss(logits, targets).item():.4f}")
    
    # Test segmentation losses
    print("\n=== Testing Segmentation Losses ===")
    pred_masks = torch.randn(batch_size, 1, 64, 64)
    gt_masks = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
    
    dice_loss = DiceLoss()
    print(f"DiceLoss: {dice_loss(torch.sigmoid(pred_masks), gt_masks).item():.4f}")
    
    dice_bce = DiceBCELoss()
    print(f"DiceBCELoss: {dice_bce(pred_masks, gt_masks).item():.4f}")
    
    dice_focal = DiceFocalLoss()
    print(f"DiceFocalLoss: {dice_focal(pred_masks, gt_masks).item():.4f}")
    
    print("\nAll loss tests passed!")


if __name__ == '__main__':
    test_losses()
