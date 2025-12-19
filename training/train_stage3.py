"""Stage 3: Adenoma subtype classification using Stage 2 encoder."""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_efficientnetv2_classifier
from datasets import create_classification_dataset
from utils import (
    get_classification_train_transform,
    get_classification_val_transform,
    get_loss_function,
    MetricsTracker,
    CheckpointManager,
    EarlyStopping,
    get_scheduler,
    MixUpTransform
)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    class_names: List[str],
    mixup: Optional[MixUpTransform]
) -> dict:
    """Train model for one epoch."""

    model.train()
    running_loss = 0.0
    metrics_tracker = MetricsTracker(num_classes=len(class_names), class_names=class_names)

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if mixup is not None:
            mixed_images, labels_a, labels_b, lam = mixup(images, labels)
            lam = float(lam)
            outputs = model(mixed_images)
            loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        metrics_tracker.update(preds, labels, probs)

        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

    metrics = metrics_tracker.compute()
    metrics['loss'] = running_loss / len(loader)
    return metrics


def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    class_names: List[str]
) -> Tuple[dict, MetricsTracker]:
    """Validate model for one epoch."""

    model.eval()
    running_loss = 0.0
    metrics_tracker = MetricsTracker(num_classes=len(class_names), class_names=class_names)

    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            metrics_tracker.update(preds, labels, probs)

    metrics = metrics_tracker.compute()
    metrics['loss'] = running_loss / len(loader)
    return metrics, metrics_tracker


def main(args: argparse.Namespace) -> None:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    stage_config = config['stage3']

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Stage 3: Adenoma Subtype Classification ===")
    print(f"Device: {device}")

    class_names = stage_config.get('class_names', ['tubular', 'tubulovillous', 'villous'])
    if len(class_names) != stage_config['num_classes']:
        raise ValueError("Number of class_names must match num_classes in Stage 3 configuration.")

    dataset_path = stage_config.get('dataset_path') or config['data']['dataset_c_path']
    print(f"Dataset: {dataset_path}")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Stage 3 dataset directory not found: {dataset_path}")

    augmentation_cfg = stage_config.get('augmentation', {})
    train_transform = get_classification_train_transform(
        stage_config['input_size'],
        advanced=augmentation_cfg.get('advanced', False)
    )
    val_transform = get_classification_val_transform(stage_config['input_size'])

    # Build deterministic train/val split
    full_dataset = create_classification_dataset(
        root_dir=dataset_path,
        transform=None,
        binary_mode=False,
        class_names=class_names
    )

    num_samples = len(full_dataset)
    if num_samples < 2:
        raise ValueError("Stage 3 dataset must contain at least two samples to perform train/val split.")

    val_split = stage_config.get('val_split', 0.2)
    val_size = max(1, int(num_samples * val_split))
    val_size = min(val_size, num_samples - 1)
    indices = torch.randperm(num_samples).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = create_classification_dataset(
        root_dir=dataset_path,
        transform=train_transform,
        binary_mode=False,
        class_names=class_names,
        subset_indices=train_indices
    )

    val_dataset = create_classification_dataset(
        root_dir=dataset_path,
        transform=val_transform,
        binary_mode=False,
        class_names=class_names,
        subset_indices=val_indices
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # DataLoaders
    if stage_config.get('sampler') == 'WeightedRandomSampler':
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
            pin_memory=config.get('pin_memory', False)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=stage_config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=config.get('pin_memory', False)
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=stage_config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config.get('pin_memory', False)
    )

    # Model
    model_name = stage_config.get('model', config.get('stage1', {}).get('model', 'tf_efficientnetv2_m'))
    print(f"Model: {model_name}")
    model = create_efficientnetv2_classifier(
        num_classes=stage_config['num_classes'],
        pretrained=True,
        pretrained_path=stage_config.get('pretrained_backbone'),
        freeze_backbone=stage_config.get('freeze_backbone', False),
        model_name=model_name
    )

    if stage_config.get('unfreeze_last_n_blocks') is not None:
        model.unfreeze_last_n_blocks(stage_config['unfreeze_last_n_blocks'])

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    start_epoch = 1
    resume_checkpoint = stage_config.get('resume_checkpoint')
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            print(f"\n✓ Resuming Stage 3 from checkpoint: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
            else:
                model.load_state_dict(checkpoint)
            print(f"Resuming at epoch {start_epoch}")
        else:
            print(f"⚠ Resume checkpoint not found at {resume_checkpoint}, starting from scratch.")

    loss_config = stage_config['loss'].copy()
    if loss_config.get('use_class_weights', False):
        loss_config['class_weights'] = train_dataset.get_class_weights().tolist()

    criterion = get_loss_function(loss_config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_config['optimizer']['lr'],
        weight_decay=stage_config['optimizer']['weight_decay']
    )

    scheduler = get_scheduler(stage_config['scheduler'], optimizer, stage_config['epochs'])

    checkpoint_dir = stage_config.get('checkpoint_dir') or os.path.dirname(stage_config['output_path'])
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        metric_name=stage_config['checkpoint_metric'],
        mode=stage_config['checkpoint_mode']
    )

    early_stopping = EarlyStopping(
        patience=stage_config['early_stopping_patience'],
        mode=stage_config['checkpoint_mode']
    )

    writer = SummaryWriter(log_dir=stage_config.get('log_dir', 'logs/stage3'))

    mixup_transform = None
    if augmentation_cfg.get('use_mixup', False):
        mixup_transform = MixUpTransform(alpha=augmentation_cfg.get('mixup_alpha', 0.2))
        print(f"MixUp enabled (alpha={augmentation_cfg.get('mixup_alpha', 0.2)})")

    print("\n=== Starting Stage 3 Training ===")
    val_tracker: Optional[MetricsTracker] = None

    for epoch in range(start_epoch, stage_config['epochs'] + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{stage_config['epochs']}")
        print(f"{'=' * 60}")

        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            class_names,
            mixup_transform
        )

        val_metrics, val_tracker = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            class_names
        )

        print(
            f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['macro_f1']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}"
        )
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['macro_f1']:.4f}, "
            f"ROC-AUC(OVR): {val_metrics.get('roc_auc_ovr', 0.0):.4f}"
        )

        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=val_metrics
        )

        if scheduler is not None:
            scheduler.step()

        if early_stopping(val_metrics[stage_config['checkpoint_metric']]):
            print(f"\n⛔ Early stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 60)
    print("Stage 3 Training Complete")
    print("=" * 60)

    if val_tracker is None:
        _, val_tracker = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch=stage_config['epochs'],
            class_names=class_names
        )

    val_tracker.print_report()

    checkpoint_manager.save_backbone(model, stage_config['output_path'], encoder_only=False)
    print(f"\n✓ Saved Stage 3 classifier to {stage_config['output_path']}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 3: Adenoma subtype classification')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args)
