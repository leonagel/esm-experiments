#!/usr/bin/env python3
"""
Analysis pipeline tests for ROC curves, plotting, and result aggregation.
"""

import sys
from pathlib import Path
import tempfile
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from esm import analyze_memorization_signal


def test_roc_auc_computation():
    """Test ROC curve and AUC computation."""
    print("\nTesting ROC/AUC computation...")
    
    # Create synthetic predictions
    np.random.seed(42)
    
    # Perfect separation case
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    assert auc_score == 1.0, f"Perfect separation should have AUC=1.0, got {auc_score}"
    
    # Random case
    y_random = np.random.randint(0, 2, 100)
    scores_random = np.random.rand(100)
    
    auc_random = roc_auc_score(y_random, scores_random)
    assert 0.3 < auc_random < 0.7, f"Random AUC should be ~0.5, got {auc_random}"
    
    # Inverted case (worse than random)
    y_inverted = np.array([1, 1, 1, 0, 0, 0])
    scores_inverted = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    auc_inverted = roc_auc_score(y_inverted, scores_inverted)
    assert auc_inverted < 0.5, f"Inverted predictions should have AUC<0.5, got {auc_inverted}"
    
    print(f"  Perfect separation AUC: {auc_score}")
    print(f"  Random AUC: {auc_random:.3f}")
    print(f"  Inverted AUC: {auc_inverted:.3f}")
    print("✓ ROC/AUC computation test passed!")


def test_result_aggregation():
    """Test aggregation of experimental results."""
    print("\nTesting result aggregation...")
    
    # Create synthetic results
    results = []
    
    # Add results for 2 architectures, 2 datasets, 3 steps
    for arch in ['resnet', 'vit']:
        for dataset in ['cifar10', 'cifar100']:
            for step in [100, 500, 2000]:
                for condition in ['clean', 'rand']:
                    results.append({
                        'arch': arch,
                        'dataset': dataset,
                        'step': step,
                        'label_condition': condition,
                        'esm': np.random.rand() * 0.1 if condition == 'clean' else np.random.rand() * 0.3 + 0.2,
                        'sharpness': np.random.rand() * 10,
                        'trace': np.random.rand() * 100
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Test aggregations
    # 1. Group by architecture
    arch_means = df.groupby('arch')['esm'].mean()
    assert len(arch_means) == 2
    assert all(0 <= val <= 1 for val in arch_means)
    
    # 2. Group by condition
    condition_means = df.groupby('label_condition')['esm'].mean()
    assert condition_means['rand'] > condition_means['clean'], \
        "Random should have higher ESM on average"
    
    # 3. Pivot table
    pivot = df.pivot_table(
        values='esm',
        index=['arch', 'dataset'],
        columns=['label_condition', 'step'],
        aggfunc='mean'
    )
    
    assert pivot.shape == (4, 6)  # 4 arch-dataset combos, 6 condition-step combos
    
    print(f"  Total results: {len(df)}")
    print(f"  Architectures: {df['arch'].unique()}")
    print(f"  Clean vs Random ESM: {condition_means['clean']:.3f} vs {condition_means['rand']:.3f}")
    print("✓ Result aggregation test passed!")


def test_plotting_functions():
    """Test plotting functions (without display)."""
    print("\nTesting plotting functions...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: ROC curve plot
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Curved ROC
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_path = Path(tmpdir) / 'roc_test.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert roc_path.exists(), "ROC plot not saved"
        assert roc_path.stat().st_size > 1000, "ROC plot too small"
        
        # Test 2: Eigenvalue distribution plot
        eigenvalues = np.exp(-np.arange(30) * 0.1)
        
        plt.figure(figsize=(8, 6))
        plt.semilogy(eigenvalues, 'b-', linewidth=2, label='Eigenvalues')
        plt.axhline(y=0.1, color='r', linestyle='--', label='MP edge')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        eig_path = Path(tmpdir) / 'eigenvalues_test.png'
        plt.savefig(eig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert eig_path.exists(), "Eigenvalue plot not saved"
        
        # Test 3: Multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        for i, ax in enumerate(axes.flat):
            x = np.linspace(0, 10, 100)
            y = np.sin(x + i)
            ax.plot(x, y)
            ax.set_title(f'Panel {i+1}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        multi_path = Path(tmpdir) / 'multi_panel_test.png'
        plt.savefig(multi_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert multi_path.exists(), "Multi-panel plot not saved"
        
        print("  ROC curve plot: ✓")
        print("  Eigenvalue plot: ✓")
        print("  Multi-panel plot: ✓")
        print("✓ Plotting test passed!")


def test_csv_output():
    """Test CSV output generation."""
    print("\nTesting CSV output...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        data = {
            'Architecture': ['ResNet', 'ResNet', 'ViT', 'ViT'],
            'Dataset': ['CIFAR-10', 'CIFAR-100', 'CIFAR-10', 'CIFAR-100'],
            'AUC@100': [0.85, 0.82, 0.88, 0.84],
            'AUC@500': [0.92, 0.90, 0.94, 0.91],
            'AUC@2000': [0.95, 0.93, 0.96, 0.94]
        }
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = Path(tmpdir) / 'results_summary.csv'
        df.to_csv(csv_path, index=False)
        
        # Verify file
        assert csv_path.exists(), "CSV not saved"
        
        # Read back and verify
        df_loaded = pd.read_csv(csv_path)
        assert len(df_loaded) == len(df), "Row count mismatch"
        assert list(df_loaded.columns) == list(df.columns), "Column mismatch"
        assert df_loaded['AUC@500'].mean() > 0.9, "Data corrupted"
        
        print(f"  Saved {len(df)} rows to CSV")
        print(f"  Columns: {list(df.columns)}")
        print("✓ CSV output test passed!")


def test_statistical_analysis():
    """Test statistical analysis functions."""
    print("\nTesting statistical analysis...")
    
    # Test Cohen's d calculation
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(2, 1, 100)  # Shifted by 2 SDs
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Cohen's d
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    cohens_d = (mean2 - mean1) / pooled_std
    
    # Should be approximately 2.0
    assert 1.5 < cohens_d < 2.5, f"Cohen's d calculation wrong: {cohens_d}"
    
    # Test confidence intervals
    from scipy import stats
    
    ci_low, ci_high = stats.t.interval(
        0.95, len(group1)-1, 
        loc=mean1, 
        scale=std1/np.sqrt(len(group1))
    )
    
    # Mean should be within CI
    assert ci_low < mean1 < ci_high, "CI calculation wrong"
    
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print("✓ Statistical analysis test passed!")


def test_metric_visualization():
    """Test metric visualization over time."""
    print("\nTesting metric visualization...")
    
    # Create time series data
    steps = np.array([0, 100, 200, 500, 1000, 2000])
    esm_clean = 0.01 + 0.02 * np.log1p(steps / 100) + np.random.normal(0, 0.005, len(steps))
    esm_random = 0.05 + 0.15 * np.log1p(steps / 100) + np.random.normal(0, 0.01, len(steps))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        plt.figure(figsize=(8, 6))
        
        plt.plot(steps, esm_clean, 'b-o', label='Clean', linewidth=2, markersize=8)
        plt.plot(steps, esm_random, 'r-s', label='Random', linewidth=2, markersize=8)
        
        plt.xlabel('Training Step')
        plt.ylabel('ESM')
        plt.title('ESM Evolution During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('symlog')  # Symmetric log scale for 0
        
        # Add shaded regions
        plt.fill_between(steps, esm_clean - 0.01, esm_clean + 0.01, 
                        alpha=0.2, color='blue')
        plt.fill_between(steps, esm_random - 0.02, esm_random + 0.02,
                        alpha=0.2, color='red')
        
        evolution_path = Path(tmpdir) / 'esm_evolution.png'
        plt.savefig(evolution_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert evolution_path.exists(), "Evolution plot not saved"
        
        # Verify separation increases over time
        separation = esm_random - esm_clean
        assert separation[-1] > separation[0], "Separation should increase"
        
        print(f"  Initial separation: {separation[1]:.3f}")
        print(f"  Final separation: {separation[-1]:.3f}")
        print("✓ Metric visualization test passed!")


def test_result_filtering():
    """Test filtering and querying results."""
    print("\nTesting result filtering...")
    
    # Create DataFrame with results
    n_samples = 100
    df = pd.DataFrame({
        'arch': np.random.choice(['resnet', 'vit'], n_samples),
        'dataset': np.random.choice(['cifar10', 'cifar100'], n_samples),
        'step': np.random.choice([100, 500, 2000], n_samples),
        'condition': np.random.choice(['clean', 'rand'], n_samples),
        'esm': np.random.rand(n_samples),
        'sharpness': np.random.rand(n_samples) * 10
    })
    
    # Test various filters
    # 1. Single condition
    resnet_only = df[df['arch'] == 'resnet']
    assert len(resnet_only) < len(df)
    assert all(resnet_only['arch'] == 'resnet')
    
    # 2. Multiple conditions
    specific = df[(df['arch'] == 'vit') & (df['dataset'] == 'cifar10') & (df['step'] == 500)]
    assert all(specific['arch'] == 'vit')
    assert all(specific['dataset'] == 'cifar10')
    assert all(specific['step'] == 500)
    
    # 3. Query method
    queried = df.query("esm > 0.5 and sharpness < 5")
    assert all(queried['esm'] > 0.5)
    assert all(queried['sharpness'] < 5)
    
    # 4. Sorting
    sorted_df = df.sort_values(['arch', 'step', 'esm'])
    assert sorted_df['arch'].iloc[0] <= sorted_df['arch'].iloc[-1]
    
    print(f"  Total samples: {len(df)}")
    print(f"  ResNet samples: {len(resnet_only)}")
    print(f"  Filtered samples: {len(specific)}")
    print(f"  High ESM samples: {len(queried)}")
    print("✓ Result filtering test passed!")


def test_summary_statistics():
    """Test computation of summary statistics."""
    print("\nTesting summary statistics...")
    
    # Create results for multiple runs
    results = []
    
    for run in range(3):
        for step in [100, 500, 2000]:
            results.append({
                'run': run,
                'step': step,
                'esm_clean': np.random.normal(0.05, 0.01),
                'esm_random': np.random.normal(0.20, 0.02),
                'auc': np.random.normal(0.9, 0.05)
            })
    
    df = pd.DataFrame(results)
    
    # Compute statistics
    summary = df.groupby('step').agg({
        'esm_clean': ['mean', 'std', 'min', 'max'],
        'esm_random': ['mean', 'std', 'min', 'max'],
        'auc': ['mean', 'std', 'min', 'max']
    })
    
    # Verify structure
    assert summary.shape == (3, 12)  # 3 steps, 12 statistics
    
    # Access multi-level columns
    clean_mean = summary[('esm_clean', 'mean')]
    assert len(clean_mean) == 3
    assert all(0 < val < 0.1 for val in clean_mean)
    
    # Compute separation statistics
    df['separation'] = df['esm_random'] - df['esm_clean']
    sep_stats = df.groupby('step')['separation'].describe()
    
    assert all(sep_stats['mean'] > 0), "Random should have higher ESM"
    
    print("  Summary statistics computed:")
    print(f"  Steps: {sorted(df['step'].unique())}")
    print(f"  Mean separation: {df['separation'].mean():.3f}")
    print("✓ Summary statistics test passed!")


def run_all_analysis_tests():
    """Run all analysis tests."""
    print("="*60)
    print("Running Analysis Tests")
    print("="*60)
    
    tests = [
        ("ROC/AUC computation", test_roc_auc_computation),
        ("Result aggregation", test_result_aggregation),
        ("Plotting functions", test_plotting_functions),
        ("CSV output", test_csv_output),
        ("Statistical analysis", test_statistical_analysis),
        ("Metric visualization", test_metric_visualization),
        ("Result filtering", test_result_filtering),
        ("Summary statistics", test_summary_statistics)
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
        print("✓ All analysis tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test in failed:
            print(f"  - {test}")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_analysis_tests()
    sys.exit(0 if success else 1)