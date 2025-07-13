#!/usr/bin/env python3
"""
Download CIFAR-10 and CIFAR-100 datasets to a local data directory.
This ensures datasets are downloaded only once before training.
"""

import os
import argparse
from pathlib import Path
import torchvision.datasets as datasets


def download_datasets(data_dir: str = "./data"):
    """Download CIFAR-10 and CIFAR-100 datasets."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading datasets to: {data_path.absolute()}")
    
    # Download CIFAR-10
    print("\nDownloading CIFAR-10...")
    try:
        datasets.CIFAR10(root=data_dir, train=True, download=True)
        datasets.CIFAR10(root=data_dir, train=False, download=True)
        print("✓ CIFAR-10 downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading CIFAR-10: {e}")
        return False
    
    # Download CIFAR-100
    print("\nDownloading CIFAR-100...")
    try:
        datasets.CIFAR100(root=data_dir, train=True, download=True)
        datasets.CIFAR100(root=data_dir, train=False, download=True)
        print("✓ CIFAR-100 downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading CIFAR-100: {e}")
        return False
    
    print("\n✓ All datasets downloaded successfully!")
    print(f"Data directory size: {get_dir_size(data_path):.1f} MB")
    return True


def get_dir_size(path: Path) -> float:
    """Get directory size in MB."""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024 * 1024)


def check_datasets(data_dir: str = "./data") -> dict:
    """Check which datasets are already downloaded."""
    status = {
        'cifar10': False,
        'cifar100': False
    }
    
    data_path = Path(data_dir)
    if not data_path.exists():
        return status
    
    # Check for CIFAR-10
    cifar10_path = data_path / "cifar-10-batches-py"
    if cifar10_path.exists():
        status['cifar10'] = True
    
    # Check for CIFAR-100
    cifar100_path = data_path / "cifar-100-python"
    if cifar100_path.exists():
        status['cifar100'] = True
    
    return status


def main():
    parser = argparse.ArgumentParser(description="Download CIFAR datasets")
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to save datasets (default: ./data)')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check if datasets exist, do not download')
    args = parser.parse_args()
    
    if args.check_only:
        status = check_datasets(args.data_dir)
        print(f"Dataset status in {args.data_dir}:")
        print(f"  CIFAR-10:  {'✓ Downloaded' if status['cifar10'] else '✗ Not found'}")
        print(f"  CIFAR-100: {'✓ Downloaded' if status['cifar100'] else '✗ Not found'}")
        return 0 if all(status.values()) else 1
    
    # Check existing datasets
    status = check_datasets(args.data_dir)
    if all(status.values()):
        print(f"All datasets already exist in {args.data_dir}")
        return 0
    
    # Download missing datasets
    success = download_datasets(args.data_dir)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())