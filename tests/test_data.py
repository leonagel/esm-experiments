#!/usr/bin/env python3
"""
Data loading and label corruption tests.
"""

import sys
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from esm import permute_labels, set_seed


def test_cifar10_loading():
    """Test CIFAR-10 dataset loading."""
    print("\nTesting CIFAR-10 loading...")
    
    # Create a temporary directory for data
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load CIFAR-10 (download if needed)
        train_dataset = datasets.CIFAR10(
            root=tmpdir, 
            train=True, 
            download=True,
            transform=transforms.ToTensor()
        )
        
        test_dataset = datasets.CIFAR10(
            root=tmpdir,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Verify dataset sizes
        assert len(train_dataset) == 50000, f"Wrong train size: {len(train_dataset)}"
        assert len(test_dataset) == 10000, f"Wrong test size: {len(test_dataset)}"
        
        # Verify data shape
        image, label = train_dataset[0]
        assert image.shape == (3, 32, 32), f"Wrong image shape: {image.shape}"
        assert 0 <= label <= 9, f"Invalid label: {label}"
        
        # Verify label distribution
        train_labels = [train_dataset[i][1] for i in range(100)]
        label_counts = Counter(train_labels)
        assert len(label_counts) > 1, "All labels are the same!"
        
        print(f"  Train dataset: {len(train_dataset)} samples")
        print(f"  Test dataset: {len(test_dataset)} samples")
        print(f"  Image shape: {image.shape}")
        print(f"  Num classes: {len(set(train_dataset.targets))}")
        print("✓ CIFAR-10 loading test passed!")


def test_cifar100_loading():
    """Test CIFAR-100 dataset loading."""
    print("\nTesting CIFAR-100 loading...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load CIFAR-100
        train_dataset = datasets.CIFAR100(
            root=tmpdir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Verify dataset size
        assert len(train_dataset) == 50000, f"Wrong train size: {len(train_dataset)}"
        
        # Verify number of classes
        unique_labels = set(train_dataset.targets)
        assert len(unique_labels) == 100, f"Wrong number of classes: {len(unique_labels)}"
        
        # Verify label range
        assert min(train_dataset.targets) == 0
        assert max(train_dataset.targets) == 99
        
        print(f"  Train dataset: {len(train_dataset)} samples")
        print(f"  Num classes: {len(unique_labels)}")
        print("✓ CIFAR-100 loading test passed!")


def test_data_transforms():
    """Test data augmentation transforms."""
    print("\nTesting data transforms...")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dummy image
    from PIL import Image
    dummy_img = Image.new('RGB', (32, 32), color='red')
    
    # Apply transforms
    train_img = train_transform(dummy_img)
    test_img = test_transform(dummy_img)
    
    # Verify shapes
    assert train_img.shape == (3, 32, 32), f"Wrong train shape: {train_img.shape}"
    assert test_img.shape == (3, 32, 32), f"Wrong test shape: {test_img.shape}"
    
    # Verify normalization
    assert -1.5 < train_img.mean() < 1.5, "Transform normalization failed"
    
    # Test that random transforms produce different results
    train_img1 = train_transform(dummy_img)
    train_img2 = train_transform(dummy_img)
    
    # Due to RandomCrop and RandomHorizontalFlip, images should sometimes differ
    # Run multiple times to ensure randomness works
    different = False
    for _ in range(10):
        img1 = train_transform(dummy_img)
        img2 = train_transform(dummy_img)
        if not torch.allclose(img1, img2):
            different = True
            break
    
    assert different, "Random transforms not producing different results"
    
    print("  Train transforms: ✓")
    print("  Test transforms: ✓")
    print("  Randomness: ✓")
    print("✓ Transform test passed!")


def test_label_permutation():
    """Test label permutation for corruption."""
    print("\nTesting label permutation...")
    
    # Create a mock dataset
    class MockDataset:
        def __init__(self, n_samples=1000, n_classes=10):
            self.targets = list(range(n_classes)) * (n_samples // n_classes)
            self.targets = self.targets[:n_samples]
        
        def __len__(self):
            return len(self.targets)
    
    # Test with different seeds
    for seed in [42, 123, 999]:
        dataset = MockDataset(n_samples=1000, n_classes=10)
        original_targets = dataset.targets.copy()
        
        # Permute labels
        permute_labels(dataset, seed=seed)
        
        # Verify permutation properties
        assert len(dataset.targets) == len(original_targets), "Length changed"
        assert set(dataset.targets) == set(original_targets), "Label set changed"
        assert dataset.targets != original_targets, "Labels not permuted"
        
        # Count how many labels changed
        n_changed = sum(1 for i in range(len(dataset.targets)) 
                       if dataset.targets[i] != original_targets[i])
        
        # Most labels should change (>80%)
        change_ratio = n_changed / len(dataset.targets)
        assert change_ratio > 0.8, f"Too few labels changed: {change_ratio:.2%}"
        
        print(f"  Seed {seed}: {change_ratio:.1%} labels changed")
    
    # Test determinism
    dataset1 = MockDataset()
    dataset2 = MockDataset()
    
    permute_labels(dataset1, seed=42)
    permute_labels(dataset2, seed=42)
    
    assert dataset1.targets == dataset2.targets, "Permutation not deterministic"
    
    print("  Determinism: ✓")
    print("✓ Label permutation test passed!")


def test_dataloader_creation():
    """Test DataLoader creation with various settings."""
    print("\nTesting DataLoader creation...")
    
    # Create synthetic dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 3, 32, 32),
        torch.randint(0, 10, (100,))
    )
    
    # Test different batch sizes
    batch_sizes = [1, 16, 32, 100]
    
    for batch_size in batch_sizes:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )
        
        # Get one batch
        images, labels = next(iter(loader))
        
        # Verify batch size
        actual_batch_size = images.shape[0]
        expected_batch_size = min(batch_size, len(dataset))
        assert actual_batch_size == expected_batch_size, \
            f"Wrong batch size: {actual_batch_size} vs {expected_batch_size}"
        
        # Verify shapes
        assert images.shape[1:] == (3, 32, 32), f"Wrong image shape: {images.shape}"
        assert labels.shape == (actual_batch_size,), f"Wrong label shape: {labels.shape}"
    
    print(f"  Tested batch sizes: {batch_sizes}")
    print("✓ DataLoader creation test passed!")


def test_dataset_splitting():
    """Test splitting dataset into train/val sets."""
    print("\nTesting dataset splitting...")
    
    # Create dataset
    full_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 3, 32, 32),
        torch.randint(0, 10, (1000,))
    )
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Verify sizes
    assert len(train_dataset) == train_size
    assert len(val_dataset) == val_size
    assert len(train_dataset) + len(val_dataset) == len(full_dataset)
    
    # Verify no overlap (check a few samples)
    train_indices = set(train_dataset.indices[:10])
    val_indices = set(val_dataset.indices[:10])
    assert len(train_indices & val_indices) == 0, "Train/val overlap detected"
    
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")
    print("✓ Dataset splitting test passed!")


def test_class_distribution():
    """Test class distribution in datasets."""
    print("\nTesting class distribution...")
    
    # Create imbalanced dataset
    n_samples = 1000
    n_classes = 10
    
    # Create imbalanced labels (more of class 0)
    labels = []
    for c in range(n_classes):
        if c == 0:
            count = n_samples // 2  # 50% of data
        else:
            count = n_samples // (2 * (n_classes - 1))
        labels.extend([c] * count)
    
    labels = torch.tensor(labels[:n_samples])
    images = torch.randn(len(labels), 3, 32, 32)
    
    dataset = torch.utils.data.TensorDataset(images, labels)
    
    # Count class frequencies
    class_counts = Counter(labels.numpy())
    
    # Verify imbalance
    assert class_counts[0] > class_counts[1] * 3, "Class 0 not sufficiently imbalanced"
    
    # Test balanced sampling
    from torch.utils.data import WeightedRandomSampler
    
    # Compute weights
    class_weights = 1.0 / torch.tensor([class_counts[i] for i in range(n_classes)])
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    # Create loader with balanced sampling
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        sampler=sampler
    )
    
    # Check a few batches for better balance
    sampled_labels = []
    for i, (_, batch_labels) in enumerate(loader):
        sampled_labels.extend(batch_labels.tolist())
        if i >= 5:  # Check 5 batches
            break
    
    sampled_counts = Counter(sampled_labels)
    
    # Verify more balanced distribution
    max_count = max(sampled_counts.values())
    min_count = min(sampled_counts.values())
    assert max_count / min_count < 3, "Balanced sampling not working"
    
    print(f"  Original class 0 ratio: {class_counts[0]/n_samples:.2%}")
    print(f"  Sampled class 0 ratio: {sampled_counts[0]/len(sampled_labels):.2%}")
    print("✓ Class distribution test passed!")


def test_data_preprocessing():
    """Test data preprocessing consistency."""
    print("\nTesting data preprocessing...")
    
    # Test CIFAR normalization values
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    
    # Create normalize transforms
    normalize_cifar10 = transforms.Normalize(cifar10_mean, cifar10_std)
    normalize_cifar100 = transforms.Normalize(cifar100_mean, cifar100_std)
    
    # Test on synthetic data
    fake_image = torch.ones(3, 32, 32) * 0.5  # Gray image
    
    norm_10 = normalize_cifar10(fake_image)
    norm_100 = normalize_cifar100(fake_image)
    
    # Check that normalization produces different results
    assert not torch.allclose(norm_10, norm_100), "Different normalizations produced same result"
    
    # Test inverse normalization
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        return tensor * std + mean
    
    denorm_10 = denormalize(norm_10, cifar10_mean, cifar10_std)
    assert torch.allclose(denorm_10, fake_image, atol=1e-6), "Denormalization failed"
    
    print("  CIFAR-10 normalization: ✓")
    print("  CIFAR-100 normalization: ✓")
    print("  Inverse transform: ✓")
    print("✓ Data preprocessing test passed!")


def run_all_data_tests():
    """Run all data tests."""
    print("="*60)
    print("Running Data Tests")
    print("="*60)
    
    tests = [
        ("CIFAR-10 loading", test_cifar10_loading),
        ("CIFAR-100 loading", test_cifar100_loading),
        ("Data transforms", test_data_transforms),
        ("Label permutation", test_label_permutation),
        ("DataLoader creation", test_dataloader_creation),
        ("Dataset splitting", test_dataset_splitting),
        ("Class distribution", test_class_distribution),
        ("Data preprocessing", test_data_preprocessing)
    ]
    
    failed = []
    
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed.append(test_name)
    
    print("\n" + "="*60)
    if not failed:
        print("✓ All data tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test in failed:
            print(f"  - {test}")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_data_tests()
    sys.exit(0 if success else 1)