"""
Visualization utilities for debugging and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_metrics, val_metrics, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        train_metrics: Dict of {metric_name: [values_per_epoch]}
        val_metrics: Dict of {metric_name: [values_per_epoch]}
        save_path: Path to save figure
    """
    num_metrics = len(train_metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
        ax = axes[idx]
        val_values = val_metrics.get(metric_name, [])
        
        epochs = range(1, len(train_values) + 1)
        ax.plot(epochs, train_values, 'b-', label='Train', linewidth=2)
        if val_values:
            ax.plot(epochs, val_values, 'r-', label='Val', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} vs Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(images, true_labels, pred_labels, class_names, num_samples=8, save_path=None):
    """
    Visualize sample predictions
    
    Args:
        images: Tensor of images [B, C, H, W]
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Path to save figure
    """
    num_samples = min(num_samples, len(images))
    rows = 2
    cols = (num_samples + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Denormalize image
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        true_label = class_names[true_labels[idx]]
        pred_label = class_names[pred_labels[idx]]
        
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved predictions visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_segmentation(image, true_mask, pred_mask, save_path=None):
    """
    Visualize segmentation results
    
    Args:
        image: Input image [C, H, W] or [H, W, C]
        true_mask: Ground truth mask [H, W]
        pred_mask: Predicted mask [H, W]
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Prepare image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3:  # [C, H, W]
        image = image.transpose(1, 2, 0)
    
    # Denormalize if needed
    if image.max() <= 1.0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    # Red for true positives, blue for false positives, green for false negatives
    tp = (true_mask > 0.5) & (pred_mask > 0.5)
    fp = (true_mask <= 0.5) & (pred_mask > 0.5)
    fn = (true_mask > 0.5) & (pred_mask <= 0.5)
    
    overlay[tp] = overlay[tp] * 0.5 + np.array([0, 1, 0]) * 0.5  # Green
    overlay[fp] = overlay[fp] * 0.5 + np.array([1, 0, 0]) * 0.5  # Red
    overlay[fn] = overlay[fn] * 0.5 + np.array([0, 0, 1]) * 0.5  # Blue
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (TP=Green, FP=Red, FN=Blue)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved segmentation visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(labels, class_names, save_path=None):
    """
    Plot class distribution
    
    Args:
        labels: List of labels
        class_names: List of class names
        save_path: Path to save figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique)), counts, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved class distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Test visualizations
    print("Testing visualization utilities...")
    
    # Test confusion matrix
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    plot_confusion_matrix(y_true, y_pred, ['Class A', 'Class B', 'Class C'])
    
    # Test training curves
    train_metrics = {
        'loss': np.random.rand(50) * 0.5 + 0.2,
        'accuracy': np.random.rand(50) * 0.2 + 0.75
    }
    val_metrics = {
        'loss': np.random.rand(50) * 0.5 + 0.3,
        'accuracy': np.random.rand(50) * 0.2 + 0.70
    }
    plot_training_curves(train_metrics, val_metrics)
    
    # Test class distribution
    labels = np.random.choice([0, 1, 2], size=300, p=[0.6, 0.3, 0.1])
    plot_class_distribution(labels, ['Hyperplastic', 'Tubular', 'Villous'])
    
    print("✓ All visualization tests passed!")
