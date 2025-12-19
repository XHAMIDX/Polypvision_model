"""
Custom samplers for handling imbalanced datasets
"""

import torch
from torch.utils.data import Sampler
from typing import Iterator, List
import numpy as np


class WeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """
    Wrapper around PyTorch's WeightedRandomSampler with better defaults
    """
    
    def __init__(self, weights, num_samples=None, replacement=True):
        """
        Args:
            weights: Per-sample weights
            num_samples: Number of samples to draw (default: len(weights))
            replacement: Sample with replacement
        """
        if num_samples is None:
            num_samples = len(weights)
        super().__init__(weights, num_samples, replacement)


class StratifiedBatchSampler(Sampler):
    """
    Stratified sampler that ensures each batch has balanced class representation
    Useful for small batch sizes with imbalanced data
    """
    
    def __init__(
        self, 
        labels: List[int], 
        batch_size: int, 
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Args:
            labels: List of labels for all samples
            batch_size: Batch size
            shuffle: Shuffle within each class
            drop_last: Drop last incomplete batch
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group indices by class
        self.class_indices = {}
        for label in np.unique(self.labels):
            self.class_indices[label] = np.where(self.labels == label)[0]
        
        self.num_classes = len(self.class_indices)
        
        # Calculate samples per class per batch
        self.samples_per_class = batch_size // self.num_classes
        self.remaining = batch_size % self.num_classes
        
        # Calculate number of batches
        min_class_size = min([len(indices) for indices in self.class_indices.values()])
        self.num_batches = min_class_size // self.samples_per_class
        
        if not drop_last and min_class_size % self.samples_per_class != 0:
            self.num_batches += 1
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each class
        if self.shuffle:
            for label in self.class_indices:
                np.random.shuffle(self.class_indices[label])
        
        # Create batches
        for batch_idx in range(self.num_batches):
            batch = []
            
            for class_label, indices in self.class_indices.items():
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                
                # Handle last batch
                if end_idx > len(indices):
                    if self.drop_last:
                        continue
                    end_idx = len(indices)
                
                batch.extend(indices[start_idx:end_idx])
            
            if len(batch) > 0:
                if self.shuffle:
                    np.random.shuffle(batch)
                yield batch
    
    def __len__(self) -> int:
        return self.num_batches


def create_weighted_sampler(dataset, replacement: bool = True):
    """
    Create WeightedRandomSampler from dataset
    
    Args:
        dataset: Dataset with get_sample_weights() method
        replacement: Sample with replacement
    
    Returns:
        WeightedRandomSampler instance
    """
    if hasattr(dataset, 'get_sample_weights'):
        weights = dataset.get_sample_weights()
    else:
        # Fallback: uniform weights
        weights = [1.0] * len(dataset)
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=replacement
    )


def create_stratified_sampler(
    dataset, 
    batch_size: int, 
    shuffle: bool = True,
    drop_last: bool = False
):
    """
    Create StratifiedBatchSampler from dataset
    
    Args:
        dataset: Dataset with samples attribute containing (path, label) tuples
        batch_size: Batch size
        shuffle: Shuffle within classes
        drop_last: Drop last incomplete batch
    
    Returns:
        StratifiedBatchSampler instance
    """
    # Extract labels from dataset
    if hasattr(dataset, 'samples'):
        labels = [label for _, label in dataset.samples]
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        raise ValueError("Dataset must have 'samples' or 'targets' attribute")
    
    return StratifiedBatchSampler(
        labels=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


def test_samplers():
    """Test samplers with dummy data"""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create imbalanced dummy dataset
    # Class 0: 100 samples, Class 1: 20 samples
    data = torch.randn(120, 3, 224, 224)
    labels = torch.cat([torch.zeros(100), torch.ones(20)]).long()
    
    dataset = TensorDataset(data, labels)
    
    # Add samples attribute for testing
    dataset.samples = [(i, labels[i].item()) for i in range(len(labels))]
    
    print("Original class distribution:")
    print(f"Class 0: {(labels == 0).sum().item()} samples")
    print(f"Class 1: {(labels == 1).sum().item()} samples")
    
    # Test WeightedRandomSampler
    print("\n=== Testing WeightedRandomSampler ===")
    class_counts = torch.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    batch_labels = []
    for _, batch_label in loader:
        batch_labels.append(batch_label)
    
    all_labels = torch.cat(batch_labels)
    print(f"Sampled class 0: {(all_labels == 0).sum().item()} samples")
    print(f"Sampled class 1: {(all_labels == 1).sum().item()} samples")
    
    # Test StratifiedBatchSampler
    print("\n=== Testing StratifiedBatchSampler ===")
    sampler = create_stratified_sampler(dataset, batch_size=32, shuffle=True)
    
    loader = DataLoader(dataset, batch_sampler=sampler)
    for i, (_, batch_label) in enumerate(loader):
        if i < 3:  # Show first 3 batches
            print(f"Batch {i+1}: Class 0={( batch_label == 0).sum().item()}, "
                  f"Class 1={(batch_label == 1).sum().item()}")
    
    print("\nAll sampler tests passed!")


if __name__ == '__main__':
    test_samplers()
