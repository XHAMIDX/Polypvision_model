"""
Stage 2: Segmentation Training
Stage 2A: Train decoder only (frozen encoder)
Stage 2B: Fine-tune entire network
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_segmentation_model
from datasets import create_segmentation_dataset
from utils import (
    get_segmentation_train_transform,
    get_segmentation_val_transform,
    get_loss_function,
    SegmentationMetrics,
    CheckpointManager,
    EarlyStopping,
    get_scheduler
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    metrics = SegmentationMetrics()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        metrics.update(torch.sigmoid(outputs), masks)
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    result = metrics.compute()
    result['loss'] = running_loss / len(train_loader)
    return result


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    metrics = SegmentationMetrics()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            metrics.update(torch.sigmoid(outputs), masks)
    
    result = metrics.compute()
    result['loss'] = running_loss / len(val_loader)
    return result


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    stage_config = config['stage2']
    is_stage_2a = args.stage == '2a'
    substage_config = stage_config['stage2a'] if is_stage_2a else stage_config['stage2b']
    
    torch.manual_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== Stage {args.stage.upper()}: Segmentation Training ===")
    print(f"Device: {device}")
    
    # Load dataset
    train_transform = get_segmentation_train_transform(stage_config['input_size'])
    val_transform = get_segmentation_val_transform(stage_config['input_size'])
    
    # Support custom folder names from config
    image_dir = stage_config.get('image_dir', 'images')
    mask_dir = stage_config.get('mask_dir', 'masks')
    
    print(f"Loading segmentation dataset:")
    print(f"  Images from: {image_dir}/")
    print(f"  Masks from:  {mask_dir}/")
    
    full_dataset = create_segmentation_dataset(
        root_dir=config['data']['dataset_b_path'],
        transform=None,
        image_dir=image_dir,
        mask_dir=mask_dir
    )

    num_samples = len(full_dataset)
    if num_samples < 2:
        raise ValueError("Segmentation dataset must contain at least two samples to perform train/val split.")

    val_split = stage_config.get('val_split', 0.2)
    val_size = max(1, int(num_samples * val_split))
    val_size = min(val_size, num_samples - 1)
    indices = torch.randperm(num_samples).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = create_segmentation_dataset(
        root_dir=config['data']['dataset_b_path'],
        transform=train_transform,
        image_dir=image_dir,
        mask_dir=mask_dir,
        subset_indices=train_indices
    )

    val_dataset = create_segmentation_dataset(
        root_dir=config['data']['dataset_b_path'],
        transform=val_transform,
        image_dir=image_dir,
        mask_dir=mask_dir,
        subset_indices=val_indices
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=substage_config['batch_size'],
                             shuffle=True, num_workers=config['num_workers'],
                             pin_memory=config.get('pin_memory', False))
    val_loader = DataLoader(val_dataset, batch_size=substage_config['batch_size'],
                           shuffle=False, num_workers=config['num_workers'],
                           pin_memory=config.get('pin_memory', False))
    
    # Check for resume checkpoint FIRST
    start_epoch = 1
    resume_checkpoint_key = f'resume_checkpoint_2{args.stage[-1]}'
    resume_checkpoint_path = stage_config.get(resume_checkpoint_key, None)
    has_resume_checkpoint = resume_checkpoint_path and os.path.isfile(resume_checkpoint_path)
    
    # Create model
    # Note: encoder_weights can be 'imagenet' or path to checkpoint
    encoder_weights_config = stage_config.get('encoder_weights', 'imagenet')
    pretrained_encoder_path = None
    
    # Only load Stage 1 encoder if NOT resuming from Stage 2 checkpoint
    if not has_resume_checkpoint:
        # Check if it's a file path (not 'imagenet' or 'noweights')
        if encoder_weights_config not in ['imagenet', 'noweights', None]:
            if os.path.isfile(encoder_weights_config):
                print(f"✓ Loading Stage 1 checkpoint: {encoder_weights_config}")
                pretrained_encoder_path = encoder_weights_config
                encoder_weights_config = None
            else:
                print(f"⚠ Warning: Stage 1 checkpoint ({encoder_weights_config}) not found")
                print(f"  Using ImageNet weights instead for {stage_config['encoder']}")
                encoder_weights_config = 'imagenet'
    else:
        # Use ImageNet weights, will be overwritten by resume checkpoint
        print(f"⚠ Resume checkpoint found, skipping Stage 1 encoder load")
        encoder_weights_config = 'imagenet'
    
    # Create segmentation model
    model = create_segmentation_model(
        architecture=stage_config['architecture'],
        encoder_name=stage_config['encoder'],
        pretrained_encoder=pretrained_encoder_path or encoder_weights_config,
        freeze_encoder=substage_config['freeze_encoder']
    )
    
    # For stage 2b, unfreeze last N blocks if specified
    if not is_stage_2a and substage_config.get('unfreeze_last_n_blocks'):
        model.unfreeze_last_n_encoder_blocks(substage_config['unfreeze_last_n_blocks'])
    
    model = model.to(device)
    
    # Handle resume checkpoint - LOAD AFTER model creation
    if has_resume_checkpoint:
        print(f"✓ Resuming from Stage 2{args.stage[-1].upper()} checkpoint: {resume_checkpoint_path}")
        # Use weights_only=False to support older checkpoints with numpy scalars
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            start_epoch = checkpoint.get('epoch', 1) + 1
        else:
            # Assume it's a raw state dict
            state_dict = checkpoint
            start_epoch = 1
        
        # The checkpoint may have been saved with or without 'model.' prefix
        # Try loading into model.model first (if checkpoint saved just self.model.state_dict)
        # If that fails, try loading into self (if checkpoint saved self.state_dict with model. prefix)
        try:
            model.model.load_state_dict(state_dict)
            print(f"  ✓ Loaded checkpoint into model.model (direct)")
        except RuntimeError:
            # Fallback: try loading full model
            try:
                model.load_state_dict(state_dict)
                print(f"  ✓ Loaded checkpoint into model (wrapped)")
            except RuntimeError as e:
                print(f"  ⚠ Both loading methods failed, trying key adjustment...")
                # Last resort: try adding 'model.' prefix
                if any(k.startswith('encoder.') for k in state_dict.keys()):
                    state_dict_fixed = {'model.' + k: v for k, v in state_dict.items()}
                    model.load_state_dict(state_dict_fixed)
                    print(f"  ✓ Loaded checkpoint (added model. prefix)")
                else:
                    raise e
        
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f"  Resuming from epoch {start_epoch}")
    
    # Loss and optimizer
    criterion = get_loss_function(substage_config['loss'], device)
    
    if is_stage_2a:
        optimizer = torch.optim.AdamW(model.parameters(), lr=substage_config['optimizer']['lr'],
                                     weight_decay=substage_config['optimizer']['weight_decay'])
    else:
        # Different LR for encoder and decoder
        encoder_params = list(model.model.encoder.parameters())
        decoder_params = list(model.model.decoder.parameters()) + list(model.model.segmentation_head.parameters())
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': substage_config['optimizer']['lr_backbone']},
            {'params': decoder_params, 'lr': substage_config['optimizer']['lr_head']}
        ], weight_decay=substage_config['optimizer']['weight_decay'])
    
    scheduler = get_scheduler(substage_config.get('scheduler', {'name': 'none'}), 
                             optimizer, substage_config['epochs']) if not is_stage_2a else None
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=f'checkpoints/stage2{args.stage[-1]}',
        metric_name=substage_config.get('checkpoint_metric', 'dice'),
        mode=substage_config.get('checkpoint_mode', 'max')
    )
    
    early_stopping = EarlyStopping(patience=substage_config.get('early_stopping_patience', 15),
                                  mode='max') if not is_stage_2a else None
    
    writer = SummaryWriter(log_dir=f'logs/stage2{args.stage[-1]}')
    
    # Training loop
    for epoch in range(start_epoch, substage_config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{substage_config['epochs']}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        
        checkpoint_manager.save_checkpoint(epoch, model, optimizer, scheduler, val_metrics)
        
        if scheduler:
            scheduler.step()
        
        if early_stopping and early_stopping(val_metrics['dice']):
            break
    
    # Save outputs
    if is_stage_2a:
        print("✓ Stage 2A complete")
    else:
        checkpoint_manager.save_backbone(model, stage_config['output_backbone'], encoder_only=True)
        torch.save(model.state_dict(), stage_config['output_model'])
        print(f"✓ Saved backbone to {stage_config['output_backbone']}")
        print(f"✓ Saved model to {stage_config['output_model']}")
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--stage', type=str, choices=['2a', '2b'], required=True)
    args = parser.parse_args()
    main(args)
