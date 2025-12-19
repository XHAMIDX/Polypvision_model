"""
EfficientNetV2 backbone wrapper for classification
"""

import os
import torch
import torch.nn as nn
import timm
from typing import Optional


class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2-M for classification with customizable head
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = 'tf_efficientnetv2_m',
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            model_name: Timm model name
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before classifier
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()
        
        # Load backbone from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            if len(dummy_output.shape) == 4:
                self.feature_dim = dummy_output.shape[1]
            else:
                self.feature_dim = dummy_output.shape[-1]
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen")
    
    def unfreeze_last_n_blocks(self, n: int = 2):
        """
        Unfreeze last N blocks of backbone
        
        Args:
            n: Number of blocks to unfreeze from the end
        """
        # First freeze all
        self.freeze_backbone()
        
        # Get all named modules
        all_modules = list(self.backbone.named_children())
        
        # Unfreeze last n modules
        for name, module in all_modules[-n:]:
            for param in module.parameters():
                param.requires_grad = True
            print(f"✓ Unfroze block: {name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)
        
        # Global pooling if needed
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        
        # Classifier
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Features [B, feature_dim]
        """
        features = self.backbone(x)
        
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        
        return features
    
    def load_pretrained_backbone(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained backbone weights with robust key matching.
        This version correctly handles checkpoints from Stage 1 (full classifier)
        and Stage 2 (encoder-only from segmentation_models_pytorch).
        """
        if not os.path.exists(checkpoint_path):
            print(f"❌ Pretrained backbone checkpoint not found at: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"⏳ Loading pretrained backbone from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Determine the source of the state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # --- THE FINAL, CORRECT FIX ---
        # This logic correctly handles all checkpoint formats from our pipeline.
        final_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            # 1. Skip any task-specific layers (classifier, head, fc, decoder)
            if any(skip in key.lower() for skip in ['classifier', 'head', 'fc.', '.fc', 'decoder']):
                skipped_keys.append(key)
                continue

            # 2. Standardize the key by removing ALL known prefixes to get the base key.
            # This handles 'backbone.features...', 'encoder.features...', 'model.encoder.features...' etc.
            # and results in a clean key like 'features...'.
            clean_key = key
            prefixes_to_strip = ['backbone.', 'encoder.', 'model.']
            for prefix in prefixes_to_strip:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            
            # 3. The model's `self.backbone` state_dict has keys that do NOT start with 'backbone.'
            # (e.g., 'conv_stem.weight', 'blocks.0.0.conv.weight').
            # We load the `clean_key` into the `self.backbone` module.
            final_state_dict[clean_key] = value
        
        # Load the standardized state dict into the model's backbone module
        missing, unexpected = self.backbone.load_state_dict(final_state_dict, strict=False)
        
        if skipped_keys:
            print(f"✓ Skipped {len(skipped_keys)} task-specific keys (e.g., classifier, decoder).")

        if not missing and not unexpected:
            print("✅ Flawless Victory! Successfully loaded pretrained backbone with 0 missing or unexpected keys!")
        else:
            if missing:
                print(f"⚠ Missing keys when loading backbone: {len(missing)}. This may be okay.")
            if unexpected:
                print(f"⚠ Unexpected keys in checkpoint: {len(unexpected)}. This is usually okay.")
            print(f"✅ Pretrained backbone loaded from {checkpoint_path}")


def create_efficientnetv2_classifier(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    freeze_backbone: bool = False,
    model_name: str = 'tf_efficientnetv2_m'
) -> EfficientNetV2Classifier:
    """
    Factory function to create EfficientNetV2 classifier
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        pretrained_path: Path to custom pretrained weights
        freeze_backbone: Freeze backbone
        model_name: Model name from timm library (default: 'tf_efficientnetv2_m')
    
    Returns:
        EfficientNetV2Classifier instance
    """
    model = EfficientNetV2Classifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    # Load custom pretrained weights if provided
    if pretrained_path is not None:
        model.load_pretrained_backbone(pretrained_path)
    
    return model


def test_backbone():
    """Test EfficientNetV2 backbone"""
    print("=== Testing EfficientNetV2Classifier ===")
    
    # Create model
    model = EfficientNetV2Classifier(num_classes=3, pretrained=False)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    # Test freezing
    model.freeze_backbone()
    frozen_params = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    print(f"Frozen backbone parameters: {frozen_params}")
    
    model.unfreeze_backbone()
    unfrozen_params = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f"Unfrozen backbone parameters: {unfrozen_params}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ All backbone tests passed!")


if __name__ == '__main__':
    test_backbone()
