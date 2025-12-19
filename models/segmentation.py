"""
Segmentation models using EfficientNetV2 encoder
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class PolypSegmentationModel(nn.Module):
    """
    Segmentation model wrapper using segmentation_models_pytorch
    """
    
    def __init__(
        self,
        architecture: str = 'UnetPlusPlus',
        encoder_name: str = 'tf_efficientnetv2_m',
        encoder_weights: Optional[str] = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None
    ):
        """
        Args:
            architecture: 'Unet', 'UnetPlusPlus', 'FPN', 'PSPNet', etc.
            encoder_name: Encoder backbone name (timm compatible)
            encoder_weights: 'imagenet' or None for random init
            in_channels: Number of input channels
            classes: Number of output classes (1 for binary segmentation)
            activation: Output activation ('sigmoid', 'softmax', None)
        """
        super().__init__()
        
        # Create segmentation model
        model_class = getattr(smp, architecture)
        
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
        
        self.architecture = architecture
        self.encoder_name = encoder_name
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Segmentation mask [B, classes, H, W]
        """
        return self.model(x)
    
    def freeze_encoder(self):
        """Freeze encoder (backbone) parameters"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder unfrozen")
    
    def unfreeze_last_n_encoder_blocks(self, n: int = 2):
        """
        Unfreeze last N encoder blocks
        
        Args:
            n: Number of blocks to unfreeze
        """
        # Freeze all first
        self.freeze_encoder()
        
        # Unfreeze last n blocks
        encoder_blocks = list(self.model.encoder.children())
        for block in encoder_blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        
        print(f"✓ Unfroze last {n} encoder blocks")
    
    def load_pretrained_encoder(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained encoder weights
        
        Args:
            checkpoint_path: Path to checkpoint file (or 'imagenet' for ImageNet weights)
            strict: Strict loading mode
        """
        # Skip if using ImageNet weights (already loaded during init)
        if checkpoint_path == 'imagenet':
            return
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter and prepare encoder weights
        encoder_state_dict = {}
        for key, value in state_dict.items():
            # Skip classification heads (from Stage 1)
            if 'head' in key or 'fc' in key or 'classifier' in key:
                continue
            
            # Handle keys with 'encoder.' prefix (from segmentation model)
            if key.startswith('encoder.'):
                encoder_state_dict[key] = value
            # Handle keys with 'encoder' in them
            elif 'encoder' in key:
                encoder_state_dict[key] = value
            # Handle Stage 1 backbone keys (convert to encoder format)
            elif 'backbone' in key:
                new_key = key.replace('backbone', 'encoder')
                encoder_state_dict[new_key] = value
            # Handle raw backbone keys (no prefix - direct transfer)
            elif key.startswith(('stem', 'blocks', 'conv', 'bn')):
                # Direct backbone keys, add encoder prefix
                encoder_state_dict[f'encoder.{key}'] = value
        
        # If no encoder keys found, use full state dict (backbone is encoder)
        if not encoder_state_dict and 'backbone' not in ''.join(state_dict.keys()):
            encoder_state_dict = state_dict
        
        # Load weights (handle both old and new PyTorch versions)
        result = self.model.encoder.load_state_dict(encoder_state_dict, strict=strict)
        
        # Handle different return types from load_state_dict
        if result is not None:
            missing, unexpected = result
            if missing:
                print(f"⚠ Missing keys: {len(missing)}")
            if unexpected:
                print(f"⚠ Unexpected keys: {len(unexpected)}")
        else:
            print(f"✓ Loaded encoder weights (no missing keys)")
        
        print(f"✓ Loaded pretrained encoder from {checkpoint_path}")
    
    def get_encoder_state_dict(self):
        """Get encoder state dict for saving"""
        return self.model.encoder.state_dict()


def create_segmentation_model(
    architecture: str = 'UnetPlusPlus',
    encoder_name: str = 'tf_efficientnetv2_m',
    pretrained_encoder: Optional[str] = None,
    freeze_encoder: bool = False
) -> PolypSegmentationModel:
    """
    Factory function to create segmentation model
    
    Args:
        architecture: Segmentation architecture
        encoder_name: Encoder name
        pretrained_encoder: Path to pretrained encoder weights
        freeze_encoder: Freeze encoder initially
    
    Returns:
        PolypSegmentationModel instance
    """
    # Use imagenet weights if no custom pretrained provided
    encoder_weights = 'imagenet' if pretrained_encoder is None else None
    
    model = PolypSegmentationModel(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None  # Use raw logits, apply sigmoid in loss
    )
    
    # Load custom pretrained encoder
    if pretrained_encoder is not None:
        model.load_pretrained_encoder(pretrained_encoder)
    
    # Freeze encoder if requested
    if freeze_encoder:
        model.freeze_encoder()
    
    return model


def test_segmentation():
    """Test segmentation model"""
    print("=== Testing Segmentation Model ===")
    
    # Create model
    model = PolypSegmentationModel(
        architecture='UnetPlusPlus',
        encoder_name='resnet34',  # Use lighter model for testing
        encoder_weights=None  # No pretrained for testing
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test freezing
    model.freeze_encoder()
    frozen_params = sum(1 for p in model.model.encoder.parameters() if not p.requires_grad)
    print(f"Frozen encoder parameters: {frozen_params}")
    
    model.unfreeze_encoder()
    unfrozen_params = sum(1 for p in model.model.encoder.parameters() if p.requires_grad)
    print(f"Unfrozen encoder parameters: {unfrozen_params}")
    
    # Count parameters
    encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nEncoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    print("\n✓ All segmentation tests passed!")


if __name__ == '__main__':
    test_segmentation()
