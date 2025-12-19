"""
Learning rate schedulers
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import math
from typing import List


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing with warmup
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate decay
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        power: float = 0.9,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer
            max_epochs: Total number of epochs
            power: Polynomial power
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


def get_scheduler(scheduler_config: dict, optimizer: torch.optim.Optimizer, max_epochs: int):
    """
    Factory function to create scheduler from config
    
    Args:
        scheduler_config: Scheduler configuration
        optimizer: Optimizer
        max_epochs: Total number of epochs
    
    Returns:
        Learning rate scheduler
    """
    scheduler_name = scheduler_config['name'].lower()
    
    if scheduler_name == 'cosinewarmup':
        warmup_ratio = scheduler_config.get('warmup_ratio', 0.05)
        warmup_epochs = int(max_epochs * warmup_ratio)
        min_lr = scheduler_config.get('min_lr', 1e-6)
        
        return CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            min_lr=min_lr
        )
    
    elif scheduler_name == 'cosineannealinglr' or scheduler_name == 'cosine':
        T_max = scheduler_config.get('T_max', max_epochs)
        eta_min = scheduler_config.get('eta_min', 1e-6)
        
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    
    elif scheduler_name == 'polynomial' or scheduler_name == 'poly':
        power = scheduler_config.get('power', 0.9)
        min_lr = scheduler_config.get('min_lr', 1e-6)
        
        return PolynomialLR(
            optimizer=optimizer,
            max_epochs=max_epochs,
            power=power,
            min_lr=min_lr
        )
    
    elif scheduler_name == 'onecyclelr' or scheduler_name == 'onecycle':
        max_lr = scheduler_config.get('max_lr', optimizer.defaults['lr'])
        
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=max_epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    
    elif scheduler_name == 'none' or scheduler_name is None:
        # No scheduler
        return None
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def test_schedulers():
    """Test learning rate schedulers"""
    import matplotlib.pyplot as plt
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    max_epochs = 100
    
    # Test different schedulers
    schedulers = {
        'CosineWarmup': CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=max_epochs),
        'CosineAnnealing': CosineAnnealingLR(optimizer, T_max=max_epochs),
        'Polynomial': PolynomialLR(optimizer, max_epochs=max_epochs, power=0.9),
    }
    
    # Track learning rates
    lrs = {name: [] for name in schedulers.keys()}
    
    for epoch in range(max_epochs):
        for name, scheduler in schedulers.items():
            # Get current LR
            lrs[name].append(optimizer.param_groups[0]['lr'])
            
            # Step scheduler
            scheduler.step()
            
            # Reset optimizer LR for next scheduler test
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
    
    # Plot learning rates
    plt.figure(figsize=(12, 6))
    for name, lr_values in lrs.items():
        plt.plot(lr_values, label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('scheduler_comparison.png', dpi=150)
    print("✓ Saved scheduler comparison plot to scheduler_comparison.png")
    
    print("\nAll scheduler tests passed!")


if __name__ == '__main__':
    test_schedulers()
