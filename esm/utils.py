"""
Utility functions for ESM experiments.
"""

import torch
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
import subprocess


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return 'unknown'


def permute_labels(dataset: Dataset, seed: int) -> None:
    """
    Randomly permute labels of a dataset in-place.
    
    Args:
        dataset: PyTorch dataset with .targets attribute
        seed: Random seed for permutation
    """
    rng = np.random.default_rng(seed)
    n_samples = len(dataset)
    perm = rng.permutation(n_samples)
    
    # Apply permutation
    if hasattr(dataset, 'targets'):
        original_targets = dataset.targets.copy() if isinstance(dataset.targets, np.ndarray) else dataset.targets[:]
        for i in range(n_samples):
            dataset.targets[i] = original_targets[perm[i]]
    elif hasattr(dataset, 'labels'):
        original_labels = dataset.labels.copy() if isinstance(dataset.labels, np.ndarray) else dataset.labels[:]
        for i in range(n_samples):
            dataset.labels[i] = original_labels[perm[i]]
    else:
        raise AttributeError("Dataset must have 'targets' or 'labels' attribute")


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: Dict, filename: str) -> None:
    """Save training checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def create_lr_scheduler(optimizer: torch.optim.Optimizer,
                       total_steps: int,
                       warmup_steps: int = 0,
                       schedule: str = 'cosine') -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        schedule: Type of schedule ('cosine', 'linear', 'constant')
    
    Returns:
        Learning rate scheduler
    """
    if schedule == 'cosine':
        if warmup_steps > 0:
            # Cosine with warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    elif schedule == 'linear':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif schedule == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count