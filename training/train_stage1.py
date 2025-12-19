"""
Stage 1: Pretrain Backbone on Dataset A (Binary Classification)
Train EfficientNetV2-M on hyperplastic vs adenoma classification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_efficientnetv2_classifier
from datasets import create_classification_dataset
from utils import (
    get_classification_train_transform,
    get_classification_val_transform,
    create_weighted_sampler,
    get_loss_function,
    MetricsTracker,
    CheckpointManager,
    EarlyStopping,
    get_scheduler
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    metrics_tracker = MetricsTracker(num_classes=2, class_names=['hyperplastic', 'adenoma'])
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        metrics_tracker.update(preds, labels, probs)
        
        # Update progress bar
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    # Compute metrics
    metrics = metrics_tracker.compute()
    metrics['loss'] = running_loss / len(train_loader)
    
    return metrics


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    
    running_loss = 0.0
    metrics_tracker = MetricsTracker(num_classes=2, class_names=['hyperplastic', 'adenoma'])
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            metrics_tracker.update(preds, labels, probs)
    
    # Compute metrics
    metrics = metrics_tracker.compute()
    metrics['loss'] = running_loss / len(val_loader)
    
    return metrics, metrics_tracker


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    stage_config = config['stage1']
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n=== Loading Dataset A ===")
    train_transform = get_classification_train_transform(stage_config['input_size'])
    val_transform = get_classification_val_transform(stage_config['input_size'])
    
    # Load full dataset and create deterministic split
    full_dataset = create_classification_dataset(
        root_dir=config['data']['dataset_a_path'],
        transform=None,
        binary_mode=False
    )

    num_samples = len(full_dataset)
    if num_samples < 2:
        raise ValueError("Dataset A must contain at least two samples to perform train/val split.")

    val_split = stage_config.get('val_split', 0.2)
    val_size = max(1, int(num_samples * val_split))
    val_size = min(val_size, num_samples - 1)  # Ensure at least one training sample
    indices = torch.randperm(num_samples).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    class_names = full_dataset.class_names

    train_dataset = create_classification_dataset(
        root_dir=config['data']['dataset_a_path'],
        transform=train_transform,
        binary_mode=False,
        class_names=class_names,
        subset_indices=train_indices
    )

    val_dataset = create_classification_dataset(
        root_dir=config['data']['dataset_a_path'],
        transform=val_transform,
        binary_mode=False,
        class_names=class_names,
        subset_indices=val_indices
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders with weighted sampler
    if stage_config['sampler'] == 'WeightedRandomSampler':
        from torch.utils.data import WeightedRandomSampler as TorchWeightedSampler
        train_sampler = TorchWeightedSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=stage_config['batch_size'],
            sampler=train_sampler,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=stage_config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=stage_config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model_name = stage_config.get('model', 'tf_efficientnetv2_m')
    print(f"Model: {model_name}")
    model = create_efficientnetv2_classifier(
        num_classes=stage_config['num_classes'],
        pretrained=True,
        model_name=model_name
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load checkpoint if resuming
    start_epoch = 1
    if stage_config.get('resume_checkpoint'):
        resume_path = stage_config['resume_checkpoint']
        if os.path.exists(resume_path):
            print(f"\n=== Resuming from checkpoint: {resume_path} ===")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Handle both old format (raw state_dict) and new format (dict with metadata)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 1) + 1
                print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, starting at epoch {start_epoch}")
            else:
                # Old format: checkpoint is just the state_dict
                model.load_state_dict(checkpoint)
                print("Resumed from legacy checkpoint format")
        else:
            print(f"⚠ Resume checkpoint not found: {resume_path}")
            print("Starting from scratch (pretrained ImageNet weights)")
    else:
        print("Training from scratch (pretrained ImageNet weights)")
    
    # Loss function
    loss_config = stage_config['loss'].copy()
    if loss_config.get('use_class_weights', False):
        class_weights = train_dataset.get_class_weights()
        loss_config['class_weights'] = class_weights.tolist()
    
    criterion = get_loss_function(loss_config, device)
    print(f"Loss function: {loss_config['name']}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_config['optimizer']['lr'],
        weight_decay=stage_config['optimizer']['weight_decay']
    )
    
    # Scheduler
    scheduler = get_scheduler(stage_config['scheduler'], optimizer, stage_config['epochs'])
    
    # Checkpoint manager
    checkpoint_dir = os.path.dirname(stage_config['output_path'])
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        metric_name=stage_config['checkpoint_metric'],
        mode=stage_config['checkpoint_mode']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=stage_config['early_stopping_patience'],
        mode=stage_config['checkpoint_mode']
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir='logs/stage1')
    
    # Training loop
    print("\n=== Starting Training ===")
    for epoch in range(start_epoch, stage_config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{stage_config['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics, metrics_tracker = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Macro-F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"Macro-F1: {val_metrics['macro_f1']:.4f}, "
              f"ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
        
        # Log to TensorBoard
        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=val_metrics
        )
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        # Early stopping
        if early_stopping(val_metrics[stage_config['checkpoint_metric']]):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Print final report
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    metrics_tracker.print_report()
    
    # Save final backbone
    checkpoint_manager.save_backbone(model, stage_config['output_path'], encoder_only=False)
    
    writer.close()
    print(f"\n✓ Saved backbone to {stage_config['output_path']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: Pretrain Backbone')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args)
