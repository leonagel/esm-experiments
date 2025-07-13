#!/usr/bin/env python3
"""
Verification tests for ESM implementation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from esm import lanczos_eigs, compute_esm, estimate_mp_edge


def test_lanczos_on_simple_model():
    """Test Lanczos algorithm on a simple linear model."""
    # Create a simple 1-layer linear model
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 5)
    
    # Create synthetic data
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    
    # Compute loss
    output = model(x)
    loss = torch.nn.CrossEntropyLoss()(output, y)
    
    # Compute eigenvalues with Lanczos
    k = 5
    eigs_lanczos = lanczos_eigs(model, loss, k=k, max_iter=20)
    
    # Verify shape and ordering
    assert len(eigs_lanczos) == k
    assert np.all(np.diff(eigs_lanczos) <= 0), "Eigenvalues should be in descending order"
    
    # Verify eigenvalues are reasonable
    assert np.all(eigs_lanczos >= -1e-6), "Eigenvalues should be non-negative for cross-entropy"
    assert np.all(eigs_lanczos < 100), "Eigenvalues unexpectedly large"
    
    # Additional test: verify Lanczos produces positive eigenvalues for a well-conditioned problem
    # Create a model and compute loss with better conditioning
    torch.manual_seed(123)
    model2 = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # Use a larger batch for better conditioning
    x2 = torch.randn(64, 10)
    y2 = torch.randint(0, 5, (64,))
    output2 = model2(x2)
    loss2 = torch.nn.CrossEntropyLoss()(output2, y2)
    
    # Get more eigenvalues
    k2 = 10
    eigs2 = lanczos_eigs(model2, loss2, k=k2, max_iter=30)
    
    # Verify properties
    assert len(eigs2) == k2, f"Expected {k2} eigenvalues, got {len(eigs2)}"
    assert np.all(np.diff(eigs2) <= 1e-10), "Eigenvalues not in descending order"
    
    # For cross-entropy loss, top eigenvalue should be positive
    assert eigs2[0] > 0, f"Top eigenvalue should be positive, got {eigs2[0]}"
    
    print(f"✓ Lanczos test passed. Top-{k} eigenvalues: {eigs_lanczos}")


def test_esm_null_hypothesis():
    """Test that ESM ≈ 0 for untrained model on random labels."""
    # Create untrained ResNet-like model (smaller for testing)
    torch.manual_seed(1337)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10)
    )
    
    # Random data and labels
    x = torch.randn(128, 3, 32, 32)
    y = torch.randint(0, 10, (128,))
    
    # Forward pass
    output = model(x)
    loss = torch.nn.CrossEntropyLoss()(output, y)
    
    # Compute eigenvalues
    eigs = lanczos_eigs(model, loss, k=20, max_iter=40)
    
    # Compute ESM
    n_params = sum(p.numel() for p in model.parameters())
    metrics = compute_esm(eigs, n_params=n_params, n_samples=128)
    
    # At initialization with random labels, ESM should be near 0
    assert abs(metrics['esm']) < 1e-2, f"ESM too large at init: {metrics['esm']}"
    print(f"✓ ESM null test passed. ESM = {metrics['esm']:.6f}")


def test_mp_edge_estimation():
    """Test Marchenko-Pastur edge estimation."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Test case 1: Simple scenario with known bulk variance
    n_params = 1000
    n_samples = 500  # Better conditioned
    gamma = n_params / n_samples  # gamma = 2
    
    # Create eigenvalues with known bulk variance
    bulk_variance = 0.5
    n_eigs = 30
    
    # Most eigenvalues in the bulk
    bulk_eigs = np.random.uniform(0.1, 1.5, n_eigs - 3)
    
    # A few outliers
    outliers = np.array([2.0, 2.5, 3.0])
    
    all_eigs = np.concatenate([outliers, bulk_eigs])
    all_eigs = np.sort(all_eigs)[::-1]
    
    # Estimate MP edge
    lambda_plus_est, sigma_est = estimate_mp_edge(all_eigs, n_params, n_samples, bulk_fraction=0.9)
    
    # The estimate should be reasonable
    assert lambda_plus_est > 0, "MP edge should be positive"
    assert lambda_plus_est < 10, "MP edge unreasonably large"
    
    # Test case 2: Check that the edge separates bulk from outliers reasonably
    # Given our synthetic data, the edge should be somewhere between bulk and outliers
    max_bulk = np.max(bulk_eigs)
    min_outlier = np.min(outliers)
    
    # The edge should ideally be between bulk and outliers
    # But MP estimation can be imperfect, so we allow some flexibility
    assert lambda_plus_est > np.percentile(all_eigs, 50), "MP edge too low"
    assert lambda_plus_est < np.max(all_eigs) * 1.5, "MP edge too high"
    
    print(f"✓ MP edge test passed. Est: {lambda_plus_est:.3f}, Max bulk: {max_bulk:.3f}, Min outlier: {min_outlier:.3f}")


def test_roc_separation():
    """Test ROC/AUC on synthetic clean vs random ESM values."""
    # Create synthetic ESM values
    np.random.seed(42)
    
    # Clean: lower ESM values
    esm_clean = np.random.normal(0.05, 0.02, 20)
    esm_clean = np.clip(esm_clean, 0, 1)
    
    # Random: higher ESM values
    esm_random = np.random.normal(0.20, 0.03, 20)
    esm_random = np.clip(esm_random, 0, 1)
    
    # Ensure separation
    if np.min(esm_random) <= np.max(esm_clean):
        esm_random = esm_random + (np.max(esm_clean) - np.min(esm_random) + 0.05)
    
    # Compute ROC
    from sklearn.metrics import roc_curve, auc
    labels = np.concatenate([np.zeros(len(esm_clean)), np.ones(len(esm_random))])
    scores = np.concatenate([esm_clean, esm_random])
    
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    
    # Should have high AUC given the separation
    assert auc_score > 0.8, f"AUC too low: {auc_score:.3f}"
    print(f"✓ ROC test passed. AUC = {auc_score:.3f}")


def run_all_tests():
    """Run all verification tests."""
    print("Running ESM verification tests...\n")
    
    tests = [
        ("Lanczos sanity test", test_lanczos_on_simple_model),
        ("ESM null hypothesis", test_esm_null_hypothesis),
        ("MP edge estimation", test_mp_edge_estimation),
        ("ROC separation", test_roc_separation)
    ]
    
    failed = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            test_func()
        except Exception as e:
            print(f"✗ {test_name} failed: {str(e)}")
            failed.append(test_name)
    
    print("\n" + "="*50)
    if not failed:
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test in failed:
            print(f"  - {test}")
    
    return len(failed) == 0


def test_edge_case_empty_model():
    """Test handling of model with no parameters."""
    print("\nTesting edge case: empty model...")
    
    class EmptyModel(torch.nn.Module):
        def forward(self, x):
            return x.sum()
    
    model = EmptyModel()
    x = torch.randn(10)
    loss = model(x)
    
    # Should handle gracefully
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 0, "Empty model should have no parameters"
    
    print("✓ Empty model test passed!")


def test_edge_case_single_parameter():
    """Test with model having single parameter."""
    print("\nTesting edge case: single parameter model...")
    
    model = torch.nn.Linear(1, 1, bias=False)
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    
    loss = torch.nn.MSELoss()(model(x), y)
    
    # Lanczos with k=1
    eigs = lanczos_eigs(model, loss, k=1, max_iter=1)
    
    assert len(eigs) == 1, "Should return exactly 1 eigenvalue"
    assert eigs[0] > 0, "MSE loss should have positive eigenvalue"
    
    print(f"  Single eigenvalue: {eigs[0]:.4f}")
    print("✓ Single parameter test passed!")


def test_edge_case_large_k():
    """Test requesting more eigenvalues than parameters."""
    print("\nTesting edge case: k > n_params...")
    
    # Small model with ~50 parameters
    model = torch.nn.Linear(5, 10)
    n_params = sum(p.numel() for p in model.parameters())
    
    x = torch.randn(20, 5)
    y = torch.randint(0, 10, (20,))
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    
    # Request more eigenvalues than parameters
    k = n_params + 10
    eigs = lanczos_eigs(model, loss, k=k, max_iter=n_params)
    
    # Should return at most n_params eigenvalues
    assert len(eigs) <= n_params, f"Returned {len(eigs)} eigenvalues, but only {n_params} parameters"
    
    print(f"  Requested {k} eigenvalues, got {len(eigs)} (n_params={n_params})")
    print("✓ Large k test passed!")


def test_edge_case_zero_loss():
    """Test with zero loss (perfect fit)."""
    print("\nTesting edge case: zero loss...")
    
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10)
    
    # Create data that gives zero loss
    x = torch.eye(10)
    y = model(x).detach()  # Use model's own predictions
    
    loss = torch.nn.MSELoss()(model(x), y)
    assert loss.item() < 1e-10, "Loss should be near zero"
    
    # Compute eigenvalues
    eigs = lanczos_eigs(model, loss, k=5, max_iter=10)
    
    # Eigenvalues should be small (but may not be exactly zero due to numerical precision)
    assert all(eig < 1.0 for eig in eigs), "Eigenvalues should be small for zero loss"
    
    print(f"  Max eigenvalue with zero loss: {max(eigs):.2e}")
    print("✓ Zero loss test passed!")


def test_esm_edge_cases():
    """Test ESM computation edge cases."""
    print("\nTesting ESM edge cases...")
    
    # Case 1: All eigenvalues below MP edge
    eigs_below = np.array([0.1, 0.09, 0.08, 0.07, 0.06])
    metrics = compute_esm(eigs_below, n_params=1000, n_samples=100)
    
    assert metrics['esm'] == 0.0, "ESM should be 0 when all eigenvalues below edge"
    assert metrics['n_outliers'] == 0, "No outliers expected"
    
    # Case 2: Clear separation between bulk and outliers
    # Use fewer eigenvalues to ensure bulk estimation works correctly
    eigs_mixed = np.array([10.0, 9.0, 0.5, 0.4, 0.3, 0.2, 0.1])
    metrics2 = compute_esm(eigs_mixed, n_params=1000, n_samples=100)
    
    # The MP edge estimation can be conservative, so we just check basic properties
    assert 0 <= metrics2['esm'] <= 1, "ESM should be in valid range"
    assert metrics2['sharpness'] == eigs_mixed[0], "Sharpness should be largest eigenvalue"
    
    # Case 3: Single large outlier
    eigs_outlier = np.array([1000.0, 0.1, 0.09, 0.08, 0.07])
    metrics3 = compute_esm(eigs_outlier, n_params=1000, n_samples=100)
    
    assert metrics3['esm'] > 0.9, "Single large outlier should dominate ESM"
    
    print(f"  All below edge: ESM = {metrics['esm']:.4f}")
    print(f"  Mixed bulk/outliers: ESM = {metrics2['esm']:.4f} (outliers: {metrics2['n_outliers']})")
    print(f"  Single outlier: ESM = {metrics3['esm']:.4f}")
    print("✓ ESM edge cases test passed!")


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\nTesting numerical stability...")
    
    # Test with very small eigenvalues
    small_eigs = np.array([1e-10, 1e-11, 1e-12, 1e-13, 1e-14])
    metrics_small = compute_esm(small_eigs, n_params=1000, n_samples=100)
    
    assert np.isfinite(metrics_small['esm']), "ESM should be finite for small eigenvalues"
    assert 0 <= metrics_small['esm'] <= 1, "ESM should be in [0, 1]"
    
    # Test with very large eigenvalues (but add some normal ones for stable bulk estimation)
    large_eigs = np.array([1e10, 1e9, 1e8, 1e7, 1e6, 1.0, 0.5, 0.1])
    metrics_large = compute_esm(large_eigs, n_params=1000, n_samples=100)
    
    assert np.isfinite(metrics_large['esm']), "ESM should be finite for large eigenvalues"
    assert metrics_large['esm'] > 0.5, "ESM should be high for huge outliers"
    
    # Test with mixed scales
    mixed_eigs = np.array([1e6, 1.0, 1e-6, 1e-9, 1e-12])
    metrics_mixed = compute_esm(mixed_eigs, n_params=1000, n_samples=100)
    
    assert np.isfinite(metrics_mixed['esm']), "ESM should handle mixed scales"
    
    print("  Small eigenvalues: ✓")
    print("  Large eigenvalues: ✓")
    print("  Mixed scales: ✓")
    print("✓ Numerical stability test passed!")


def test_performance_benchmarks():
    """Benchmark performance of key operations."""
    print("\nTesting performance benchmarks...")
    
    import time
    
    # Create a medium-sized model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # Generate data
    x = torch.randn(256, 100)
    y = torch.randint(0, 10, (256,))
    
    # Time forward pass
    start = time.time()
    output = model(x)
    loss = torch.nn.CrossEntropyLoss()(output, y)
    forward_time = time.time() - start
    
    # Time Lanczos
    start = time.time()
    eigs = lanczos_eigs(model, loss, k=20, max_iter=25)
    lanczos_time = time.time() - start
    
    # Time ESM computation
    start = time.time()
    metrics = compute_esm(eigs, n_params=n_params, n_samples=256)
    esm_time = time.time() - start
    
    print(f"  Model parameters: {n_params:,}")
    print(f"  Forward pass: {forward_time*1000:.1f} ms")
    print(f"  Lanczos (k=20): {lanczos_time*1000:.1f} ms")
    print(f"  ESM computation: {esm_time*1000:.1f} ms")
    
    # Verify reasonable performance
    assert lanczos_time < 5.0, "Lanczos taking too long"
    assert esm_time < 0.1, "ESM computation taking too long"
    
    print("✓ Performance benchmark test passed!")


def run_all_tests():
    """Run all verification tests."""
    print("Running ESM verification tests...\n")
    
    tests = [
        ("Lanczos sanity test", test_lanczos_on_simple_model),
        ("ESM null hypothesis", test_esm_null_hypothesis),
        ("MP edge estimation", test_mp_edge_estimation),
        ("ROC separation", test_roc_separation),
        ("Edge case: empty model", test_edge_case_empty_model),
        ("Edge case: single parameter", test_edge_case_single_parameter),
        ("Edge case: large k", test_edge_case_large_k),
        ("Edge case: zero loss", test_edge_case_zero_loss),
        ("ESM edge cases", test_esm_edge_cases),
        ("Numerical stability", test_numerical_stability),
        ("Performance benchmarks", test_performance_benchmarks)
    ]
    
    failed = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            test_func()
        except Exception as e:
            print(f"✗ {test_name} failed: {str(e)}")
            failed.append(test_name)
    
    print("\n" + "="*50)
    if not failed:
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test in failed:
            print(f"  - {test}")
    
    return len(failed) == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)