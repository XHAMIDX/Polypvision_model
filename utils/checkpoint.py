"""
Checkpoint management utilities
"""

import torch
import os
from typing import Dict, Any, Optional
from pathlib import Path


class CheckpointManager:
    """Manage model checkpoints with best model tracking"""
    
    def __init__(
        self, 
        checkpoint_dir: str, 
        metric_name: str = 'macro_f1',
        mode: str = 'max',
        save_best_only: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            metric_name: Metric to monitor for best model
            mode: 'max' or 'min' (maximize or minimize metric)
            save_best_only: Only save when metric improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.metric_name = metric_name
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
    
    def is_best(self, metric_value: float) -> bool:
        """Check if current metric is the best so far"""
        if self.mode == 'max':
            return metric_value > self.best_metric
        else:
            return metric_value < self.best_metric
    
    def save_checkpoint(
        self, 
        epoch: int, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> bool:
        """
        Save checkpoint
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler (optional)
            metrics: Dictionary of metrics
            **kwargs: Additional data to save
        
        Returns:
            True if checkpoint was saved (is best), False otherwise
        """
        metrics = metrics or {}
        current_metric = metrics.get(self.metric_name, 
                                     self.best_metric if self.mode == 'max' else -self.best_metric)
        
        # Check if this is the best model
        is_best = self.is_best(current_metric)
        
        if not self.save_best_only or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_metric': current_metric if is_best else self.best_metric,
                'best_epoch': epoch if is_best else self.best_epoch,
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Add any additional kwargs
            checkpoint.update(kwargs)
            
            # Save checkpoint
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                save_path = self.checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, save_path)
                print(f"✓ Saved best model (epoch {epoch}, {self.metric_name}={current_metric:.4f})")
            
            # Also save last checkpoint
            last_path = self.checkpoint_dir / 'last_model.pth'
            torch.save(checkpoint, last_path)
        
        return is_best
    
    def load_checkpoint(
        self, 
        model: torch.nn.Module, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Specific checkpoint path (if None, uses best or last)
            load_best: Load best model (if True) or last model (if False)
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            filename = 'best_model.pth' if load_best else 'last_model.pth'
            checkpoint_path = self.checkpoint_dir / filename

        checkpoint_path = str(checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            print(f"  Best {self.metric_name}: {checkpoint['best_metric']:.4f}")
        
        return checkpoint
    
    def save_backbone(
        self, 
        model: torch.nn.Module, 
        output_path: str,
        encoder_only: bool = True
    ):
        """
        Save only the backbone/encoder for transfer learning
        
        Args:
            model: Full model
            output_path: Path to save backbone weights
            encoder_only: Save only encoder weights (for segmentation models)
        """
        if encoder_only:
            encoder_module = getattr(model, 'encoder', None)
            if encoder_module is None and hasattr(model, 'model'):
                encoder_module = getattr(model.model, 'encoder', None)

            if encoder_module is not None and hasattr(encoder_module, 'state_dict'):
                state_dict = encoder_module.state_dict()
            else:
                # Fallback to full model state if encoder cannot be isolated
                state_dict = model.state_dict()
        else:
            # For classification models, save full state
            state_dict = model.state_dict()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(state_dict, output_path)
        print(f"✓ Saved backbone to {output_path}")


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self, 
        patience: int = 7, 
        mode: str = 'max',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs with no improvement to wait
            mode: 'max' or 'min' (maximize or minimize metric)
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, metric_value: float) -> bool:
        """
        Check if should stop training
        
        Args:
            metric_value: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'max':
            improved = metric_value > (self.best_metric + self.min_delta)
        else:
            improved = metric_value < (self.best_metric - self.min_delta)
        
        if improved:
            self.best_metric = metric_value
            self.counter = 0
            if self.verbose:
                print(f"✓ Metric improved to {metric_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠ No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"⛔ Early stopping triggered!")
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')
        self.counter = 0
        self.early_stop = False


def test_checkpoint():
    """Test checkpoint manager"""
    import tempfile
    
    print("=== Testing CheckpointManager ===")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy model
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=tmp_dir,
            metric_name='macro_f1',
            mode='max'
        )
        
        # Simulate training
        for epoch in range(5):
            metrics = {'macro_f1': 0.5 + epoch * 0.05}
            is_best = manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=metrics
            )
            print(f"Epoch {epoch}: F1={metrics['macro_f1']:.4f}, Best={is_best}")
        
        # Load best checkpoint
        checkpoint = manager.load_checkpoint(model, optimizer, load_best=True)
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
    
    print("\n=== Testing EarlyStopping ===")
    early_stop = EarlyStopping(patience=3, mode='max', verbose=True)
    
    metrics = [0.7, 0.75, 0.78, 0.77, 0.76, 0.75, 0.74]  # Decreasing after epoch 2
    for epoch, metric in enumerate(metrics):
        print(f"Epoch {epoch}: metric={metric:.4f}")
        if early_stop(metric):
            print(f"Would stop at epoch {epoch}")
            break
    
    print("\nAll checkpoint tests passed!")


if __name__ == '__main__':
    test_checkpoint()
