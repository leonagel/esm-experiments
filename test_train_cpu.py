#!/usr/bin/env python3
"""
CPU test for train.py to ensure basic functionality works.
"""

import os
import tempfile
import shutil
from pathlib import Path
import yaml
import subprocess
import sys
import torch
import torch.multiprocessing as mp


def create_test_config(config_path, temp_dir):
    """Create a minimal test configuration for CPU training."""
    config = {
        'model': {
            'name': 'resnet18',
            'num_classes': 10
        },
        'dataset': {
            'name': 'cifar10',
            'root': './data',  # Will be overridden if data exists there
            'download': True,
            'label_corruption': False,
            'corruption_seed': None
        },
        'training': {
            'batch_size': 32,  # Small batch size for CPU
            'epochs': 1,  # Just one epoch for testing
            'optimizer': {
                'type': 'sgd',
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4
            },
            'scheduler': {
                'type': 'cosine',
                'warmup_steps': 0
            }
        },
        'curvature': {
            'checkpoints': [10],  # Early checkpoint for testing
            'top_k': 5,  # Small number for CPU
            'lanczos_max_iter': 10,  # Fewer iterations for testing
            'batch_size': 16
        },
        'distributed': {
            'devices': 1,  # Single device for CPU
            'backend': 'gloo'  # CPU-compatible backend
        },
        'experiment': {
            'seeds': [42],  # Single seed for testing
            'name': 'test_cpu_run',
            'save_dir': str(Path(temp_dir) / 'results'),
            'log_wandb': False
        },
        'compute': {
            'precision': 'float32',
            'compile_model': False  # Disable compilation for CPU test
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config


def test_basic_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import timm
        from omegaconf import OmegaConf
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_single_process_cpu():
    """Test train.py in single-process CPU mode."""
    print("\nTesting single-process CPU training...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test config
        config_path = Path(temp_dir) / 'test_config.yaml'
        config = create_test_config(config_path, temp_dir)
        
        # Set environment variables for CPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
        env['OMP_NUM_THREADS'] = '1'  # Prevent CPU overload
        
        # Run training with single process
        cmd = [
            sys.executable,
            'train.py',
            '--config', str(config_path)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✓ Single-process CPU training completed successfully")
                
                # Check if results were saved
                results_dir = Path(temp_dir) / 'results'
                result_files = list(results_dir.glob('*.pt'))
                if result_files:
                    print(f"✓ Found {len(result_files)} result files")
                else:
                    print("✗ No result files found")
                
                return True
            else:
                print(f"✗ Training failed with return code {result.returncode}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Training timed out")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False


def test_model_functions():
    """Test individual model and utility functions."""
    print("\nTesting model functions...")
    
    try:
        from esm import count_parameters, set_seed
        import torch.nn as nn
        
        # Test count_parameters
        model = nn.Linear(10, 5)
        n_params = count_parameters(model)
        expected = 10 * 5 + 5  # weights + bias
        if n_params == expected:
            print(f"✓ count_parameters works correctly: {n_params} parameters")
        else:
            print(f"✗ count_parameters incorrect: got {n_params}, expected {expected}")
        
        # Test set_seed
        set_seed(42)
        x1 = torch.randn(5)
        set_seed(42)
        x2 = torch.randn(5)
        if torch.allclose(x1, x2):
            print("✓ set_seed works correctly")
        else:
            print("✗ set_seed not working properly")
        
        return True
    except Exception as e:
        print(f"✗ Error testing model functions: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from train import get_transform
        from torchvision import datasets
        
        # Test transform creation
        transform = get_transform('cifar10', is_train=True)
        print("✓ Transform creation successful")
        
        # Use the standard data directory if it exists, otherwise use temp
        data_dir = './data' if Path('./data').exists() else None
        
        if data_dir and (Path(data_dir) / 'cifar-10-batches-py').exists():
            # Use pre-downloaded data
            print(f"  Loading CIFAR-10 from {data_dir}...")
            train_dataset = datasets.CIFAR10(
                root=data_dir, train=True, download=False, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root=data_dir, train=False, download=False, transform=transform
            )
        else:
            # Download to temp directory if needed
            with tempfile.TemporaryDirectory() as temp_dir:
                print("  Downloading CIFAR-10 dataset...")
                train_dataset = datasets.CIFAR10(
                    root=temp_dir, train=True, download=True, transform=transform
                )
                test_dataset = datasets.CIFAR10(
                    root=temp_dir, train=False, download=True, transform=transform
                )
        
        if len(train_dataset) == 50000 and len(test_dataset) == 10000:
            print("✓ Dataset loading successful")
            print(f"  Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        else:
            print(f"✗ Unexpected dataset sizes")
        
        return True
    except Exception as e:
        print(f"✗ Error testing data loading: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("CPU Test Suite for train.py")
    print("=" * 60)
    
    # Check if train.py exists
    if not Path('train.py').exists():
        print("Error: train.py not found in current directory")
        return 1
    
    # Run tests
    tests = [
        ("Import Test", test_basic_imports),
        ("Model Functions Test", test_model_functions),
        ("Data Loading Test", test_data_loading),
        ("Single-Process CPU Training Test", test_single_process_cpu),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"Running: {test_name}")
        print(f"{'=' * 40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    exit(main())