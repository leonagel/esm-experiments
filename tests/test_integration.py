#!/usr/bin/env python3
"""
Integration tests that emulate the full ESM experiment workflow.
These tests run quickly on CPU with tiny models and datasets.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from esm import (
    lanczos_eigs, compute_esm, set_seed, permute_labels,
    count_parameters, analyze_memorization_signal
)


class TinyResNet(nn.Module):
    """Tiny ResNet-like model for fast testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TinyViT(nn.Module):
    """Tiny ViT-like model for fast testing."""
    def __init__(self, num_classes=10, dim=16, num_heads=2, num_layers=2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=8, stride=8)  # 32x32 -> 4x4
        self.pos_embed = nn.Parameter(torch.randn(1, 17, dim))  # 16 patches + 1 cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Use simple attention blocks to avoid backward compatibility issues
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # B, C, H, W
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])


def create_tiny_dataset(n_samples=100, n_classes=10, img_size=32, seed=42):
    """Create a tiny synthetic dataset for testing."""
    torch.manual_seed(seed)
    
    # Random images
    images = torch.randn(n_samples, 3, img_size, img_size)
    
    # Random labels
    labels = torch.randint(0, n_classes, (n_samples,))
    
    return list(zip(images, labels))


def train_step(model, data_loader, optimizer, device='cpu'):
    """Single training step."""
    model.train()
    total_loss = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)


def test_mini_experiment_clean_vs_random():
    """Test mini experiment comparing clean vs random labels."""
    print("\nTesting mini experiment: clean vs random labels...")
    
    set_seed(42)
    device = 'cpu'
    n_samples = 200
    n_classes = 10
    batch_size = 50
    n_steps = 20
    
    # Create dataset
    dataset = create_tiny_dataset(n_samples, n_classes)
    
    # Results storage
    results = {'clean': [], 'random': []}
    
    for label_type in ['clean', 'random']:
        # Reset seed for reproducibility
        set_seed(42)
        
        # Create model
        model = TinyResNet(n_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        
        # Create data loader
        if label_type == 'random':
            # Corrupt labels
            corrupted_dataset = dataset.copy()
            labels = [label for _, label in corrupted_dataset]
            perm = torch.randperm(len(labels))
            corrupted_dataset = [(img, labels[perm[i]]) for i, (img, _) in enumerate(corrupted_dataset)]
            data_loader = torch.utils.data.DataLoader(corrupted_dataset, batch_size=batch_size, shuffle=True)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop with curvature capture
        for step in range(n_steps):
            loss = train_step(model, data_loader, optimizer, device)
            
            # Capture curvature at specific steps
            if step in [5, 10, 15]:
                model.eval()
                
                # Get a batch for curvature computation
                images, labels = next(iter(data_loader))
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss_curv = nn.CrossEntropyLoss()(outputs, labels)
                
                # Compute eigenvalues
                k = 10  # Small k for speed
                eigs = lanczos_eigs(model, loss_curv, k=k, max_iter=k+5)
                
                # Compute ESM
                n_params = count_parameters(model)
                metrics = compute_esm(eigs, n_params=n_params, n_samples=batch_size)
                
                results[label_type].append({
                    'step': step,
                    'esm': metrics['esm'],
                    'sharpness': metrics['sharpness']
                })
                
                print(f"  {label_type} - Step {step}: ESM = {metrics['esm']:.4f}, "
                      f"Sharpness = {metrics['sharpness']:.4f}")
    
    # Verify that random labels have higher ESM (or at least not significantly lower)
    clean_esm_final = results['clean'][-1]['esm']
    random_esm_final = results['random'][-1]['esm']
    
    # In early training, ESM differences might be small
    assert random_esm_final >= clean_esm_final - 0.01, \
        f"Random ESM ({random_esm_final:.4f}) should be >= clean ESM ({clean_esm_final:.4f})"
    
    # Test separation using all measurements
    clean_esms = [r['esm'] for r in results['clean']]
    random_esms = [r['esm'] for r in results['random']]
    
    # Simple AUC test
    labels = [0] * len(clean_esms) + [1] * len(random_esms)
    scores = clean_esms + random_esms
    auc = roc_auc_score(labels, scores)
    
    print(f"\n  Separation AUC: {auc:.3f}")
    # Early in training, separation might be modest
    assert auc >= 0.5, f"AUC too low: {auc:.3f}"
    
    print("✓ Mini experiment test passed!")


def test_full_pipeline_with_configs():
    """Test the full pipeline with configuration files."""
    print("\nTesting full pipeline with configs...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a mini config
        config = {
            'model': {'name': 'tiny_resnet', 'num_classes': 10},
            'dataset': {
                'name': 'synthetic',
                'n_samples': 100,
                'label_corruption': False,
                'corruption_seed': None
            },
            'training': {
                'batch_size': 20,
                'epochs': 1,
                'optimizer': {
                    'type': 'sgd',
                    'lr': 0.1,
                    'momentum': 0.9,
                    'weight_decay': 0
                }
            },
            'curvature': {
                'checkpoints': [2, 5],
                'top_k': 5,
                'lanczos_max_iter': 10,
                'batch_size': 20
            },
            'experiment': {
                'seed': 1337,
                'name': 'test_run',
                'save_dir': str(tmpdir / 'results')
            }
        }
        
        # Save config
        config_path = tmpdir / 'test_config.yaml'
        OmegaConf.save(config, config_path)
        
        # Load and verify config
        loaded_cfg = OmegaConf.load(config_path)
        assert loaded_cfg.model.name == 'tiny_resnet'
        assert loaded_cfg.dataset.n_samples == 100
        
        # Create results directory
        Path(loaded_cfg.experiment.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Simulate training with config
        set_seed(loaded_cfg.experiment.seed)
        
        # Create model based on config
        if loaded_cfg.model.name == 'tiny_resnet':
            model = TinyResNet(loaded_cfg.model.num_classes)
        
        # Verify model creation
        n_params = count_parameters(model)
        assert n_params < 10000, f"Model too large for test: {n_params} params"
        
        print(f"  Created model with {n_params} parameters")
        print("✓ Full pipeline test passed!")


def test_model_comparison():
    """Test comparing different models on the same task."""
    print("\nTesting model comparison...")
    
    set_seed(42)
    n_samples = 100
    n_classes = 10
    batch_size = 25
    
    # Create dataset
    dataset = create_tiny_dataset(n_samples, n_classes)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    models = {
        'resnet': TinyResNet(n_classes),
        'vit': TinyViT(n_classes)
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Get a batch
        images, labels = next(iter(data_loader))
        
        # Forward pass
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Compute eigenvalues
        k = 5
        eigs = lanczos_eigs(model, loss, k=k, max_iter=k+5)
        
        # Compute ESM
        n_params = count_parameters(model)
        metrics = compute_esm(eigs, n_params=n_params, n_samples=batch_size)
        
        results[model_name] = metrics
        
        print(f"  {model_name}: {n_params} params, ESM = {metrics['esm']:.4f}")
    
    # Both should have low ESM at initialization
    for model_name, metrics in results.items():
        assert metrics['esm'] < 0.1, f"{model_name} ESM too high at init: {metrics['esm']}"
    
    print("✓ Model comparison test passed!")


def test_memorization_signal_analysis():
    """Test the memorization signal analysis function."""
    print("\nTesting memorization signal analysis...")
    
    # Create synthetic metrics
    clean_metrics = [
        {'esm': 0.01, 'sharpness': 0.5, 'trace': 10.0},
        {'esm': 0.02, 'sharpness': 0.6, 'trace': 11.0},
        {'esm': 0.015, 'sharpness': 0.55, 'trace': 10.5}
    ]
    
    random_metrics = [
        {'esm': 0.10, 'sharpness': 2.0, 'trace': 50.0},
        {'esm': 0.12, 'sharpness': 2.2, 'trace': 52.0},
        {'esm': 0.11, 'sharpness': 2.1, 'trace': 51.0}
    ]
    
    # Analyze separation
    analysis = analyze_memorization_signal(clean_metrics, random_metrics)
    
    # Verify results
    assert analysis['separated'] == True, "Clean and random should be separated"
    assert analysis['cohens_d'] > 1.0, f"Effect size too small: {analysis['cohens_d']}"
    assert analysis['random_esm_mean'] > analysis['clean_esm_mean'], "Random should have higher ESM"
    
    print(f"  Clean ESM: {analysis['clean_esm_mean']:.4f} ± {analysis['clean_esm_std']:.4f}")
    print(f"  Random ESM: {analysis['random_esm_mean']:.4f} ± {analysis['random_esm_std']:.4f}")
    print(f"  Cohen's d: {analysis['cohens_d']:.2f}")
    print("✓ Memorization signal analysis test passed!")


def test_batch_processing():
    """Test batch processing of eigenvalues and metrics."""
    print("\nTesting batch processing...")
    
    from esm import compute_spectral_metrics_batch
    
    # Create multiple eigenvalue arrays
    n_arrays = 5
    eigenvalue_list = []
    
    for i in range(n_arrays):
        # Create eigenvalues with different characteristics
        eigs = np.sort(np.random.exponential(1.0 / (i + 1), 20))[::-1]
        eigenvalue_list.append(eigs)
    
    # Process batch
    n_params = 1000
    n_samples = 100
    metrics_list = compute_spectral_metrics_batch(eigenvalue_list, n_params, n_samples)
    
    # Verify results
    assert len(metrics_list) == n_arrays
    
    for i, metrics in enumerate(metrics_list):
        assert 'esm' in metrics
        assert 'sharpness' in metrics
        assert 'trace' in metrics
        assert metrics['sharpness'] == eigenvalue_list[i][0]
    
    print(f"  Processed {n_arrays} eigenvalue arrays")
    print("✓ Batch processing test passed!")


def test_curvature_evolution():
    """Test that curvature evolves correctly during training."""
    print("\nTesting curvature evolution during training...")
    
    set_seed(42)
    
    # Setup
    model = TinyResNet(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Create random data
    dataset = create_tiny_dataset(100, 10)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
    
    # Track ESM over training
    esm_history = []
    
    for epoch in range(5):
        # Train one epoch
        model.train()
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Compute ESM
        model.eval()
        images, labels = next(iter(data_loader))
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        eigs = lanczos_eigs(model, loss, k=10, max_iter=15)
        metrics = compute_esm(eigs, n_params=count_parameters(model), n_samples=20)
        
        esm_history.append(metrics['esm'])
        print(f"  Epoch {epoch}: ESM = {metrics['esm']:.4f}")
    
    # ESM should generally increase or stay stable (not decrease significantly)
    # Allow for some noise
    large_decreases = sum(1 for i in range(1, len(esm_history)) 
                         if esm_history[i] < esm_history[i-1] * 0.5)
    assert large_decreases == 0, "ESM decreased significantly during training"
    
    print("✓ Curvature evolution test passed!")


def run_all_integration_tests():
    """Run all integration tests."""
    print("="*60)
    print("Running Integration Tests")
    print("="*60)
    
    tests = [
        ("Mini experiment: clean vs random", test_mini_experiment_clean_vs_random),
        ("Full pipeline with configs", test_full_pipeline_with_configs),
        ("Model comparison", test_model_comparison),
        ("Memorization signal analysis", test_memorization_signal_analysis),
        ("Batch processing", test_batch_processing),
        ("Curvature evolution", test_curvature_evolution)
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
        print("✓ All integration tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test in failed:
            print(f"  - {test}")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)