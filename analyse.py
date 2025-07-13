#!/usr/bin/env python3
"""
Analysis script for ESM experiments - compute ROC curves and AUC scores with statistical aggregation.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import torch
from typing import Dict, List, Tuple
from scipy import stats


def load_curvature_results(results_dir: Path) -> pd.DataFrame:
    """Load all curvature results into a dataframe."""
    data = []
    
    for file_path in results_dir.glob("*_curvature.pt"):
        # Parse filename - now includes seed
        parts = file_path.stem.split('_')
        if len(parts) >= 5:  # e.g., resnet_cifar10_clean_seed1337_step500_curvature
            arch = parts[0]
            dataset = parts[1]
            label_condition = parts[2]
            seed = int(parts[3].replace('seed', ''))
            step = int(parts[4].replace('step', ''))
            
            # Load results
            results = torch.load(file_path, map_location='cpu')
            
            # Extract metrics
            metrics = results['metrics']
            data.append({
                'arch': arch,
                'dataset': dataset,
                'label_condition': label_condition,
                'seed': seed,
                'step': step,
                'esm': metrics['esm'],
                'sharpness': metrics['sharpness'],
                'trace': metrics['trace'],
                'lambda_plus': metrics['lambda_plus'],
                'n_outliers': metrics['n_outliers'],
                'eigenvalues': results['eigenvalues'],
                'n_params': results['n_params']
            })
    
    df = pd.DataFrame(data)
    return df


def compute_roc_auc_with_stats(df: pd.DataFrame, arch: str, dataset: str, step: int) -> Dict:
    """Compute ROC curve and AUC with statistics across seeds."""
    # Get unique seeds
    seeds = df['seed'].unique()
    auc_scores = []
    all_fprs = []
    all_tprs = []
    
    for seed in seeds:
        # Filter data for this seed
        mask_clean = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                     (df['label_condition'] == 'clean') & (df['step'] == step) & \
                     (df['seed'] == seed)
        mask_rand = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                    (df['label_condition'] == 'rand') & (df['step'] == step) & \
                    (df['seed'] == seed)
        
        # Get ESM values
        esm_clean = df[mask_clean]['esm'].values
        esm_rand = df[mask_rand]['esm'].values
        
        if len(esm_clean) == 0 or len(esm_rand) == 0:
            continue
        
        # Create labels (0 for clean, 1 for random)
        labels = np.concatenate([np.zeros(len(esm_clean)), np.ones(len(esm_rand))])
        scores = np.concatenate([esm_clean, esm_rand])
        
        # Compute ROC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        
        auc_scores.append(auc_score)
        all_fprs.append(fpr)
        all_tprs.append(tpr)
    
    if len(auc_scores) == 0:
        return None
    
    # Compute statistics
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    # Compute mean ROC curve by interpolation
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for fpr, tpr in zip(all_fprs, all_tprs):
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    
    mean_tpr /= len(all_fprs)
    
    # Compute confidence interval
    ci_lower = mean_auc - 1.96 * std_auc / np.sqrt(len(auc_scores))
    ci_upper = mean_auc + 1.96 * std_auc / np.sqrt(len(auc_scores))
    
    return {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'auc_scores': auc_scores,
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr
    }


def plot_roc_curves_with_stats(df: pd.DataFrame, save_dir: Path):
    """Plot ROC curves with confidence bands for all configurations."""
    architectures = df['arch'].unique()
    datasets = df['dataset'].unique()
    steps = sorted(df['step'].unique())
    
    for arch in architectures:
        for dataset in datasets:
            fig, axes = plt.subplots(1, len(steps), figsize=(5*len(steps), 5))
            if len(steps) == 1:
                axes = [axes]
            
            for i, step in enumerate(steps):
                results = compute_roc_auc_with_stats(df, arch, dataset, step)
                
                if results is not None:
                    # Plot mean ROC curve
                    axes[i].plot(results['mean_fpr'], results['mean_tpr'], 
                               label=f'Mean AUC = {results["mean_auc"]:.3f} ± {results["std_auc"]:.3f}', 
                               linewidth=2)
                    
                    # Add individual seed ROCs in lighter color
                    seeds = df['seed'].unique()
                    for seed in seeds:
                        mask_clean = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                                   (df['label_condition'] == 'clean') & (df['step'] == step) & \
                                   (df['seed'] == seed)
                        mask_rand = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                                  (df['label_condition'] == 'rand') & (df['step'] == step) & \
                                  (df['seed'] == seed)
                        
                        if mask_clean.any() and mask_rand.any():
                            esm_clean = df[mask_clean]['esm'].values
                            esm_rand = df[mask_rand]['esm'].values
                            labels = np.concatenate([np.zeros(len(esm_clean)), np.ones(len(esm_rand))])
                            scores = np.concatenate([esm_clean, esm_rand])
                            fpr, tpr, _ = roc_curve(labels, scores)
                            axes[i].plot(fpr, tpr, alpha=0.3, linewidth=1)
                    
                    axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    axes[i].set_xlabel('False Positive Rate')
                    axes[i].set_ylabel('True Positive Rate')
                    axes[i].set_title(f'{arch} {dataset} - Step {step}')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                else:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{arch} {dataset} - Step {step}')
            
            plt.tight_layout()
            save_path = save_dir / f'roc_{arch}_{dataset}_with_stats.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved ROC curves to {save_path}")


def plot_eigenvalue_distributions_by_seed(df: pd.DataFrame, save_dir: Path):
    """Plot eigenvalue distributions for clean vs random labels, averaged across seeds."""
    architectures = df['arch'].unique()
    datasets = df['dataset'].unique()
    
    for arch in architectures:
        for dataset in datasets:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            steps = sorted(df['step'].unique())[:6]  # Max 6 subplots
            
            for i, step in enumerate(steps):
                ax = axes[i]
                
                # Collect eigenvalues across seeds
                clean_eigs_list = []
                rand_eigs_list = []
                
                for seed in df['seed'].unique():
                    mask_clean = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                               (df['label_condition'] == 'clean') & (df['step'] == step) & \
                               (df['seed'] == seed)
                    mask_rand = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                              (df['label_condition'] == 'rand') & (df['step'] == step) & \
                              (df['seed'] == seed)
                    
                    if mask_clean.any():
                        clean_eigs_list.append(df[mask_clean].iloc[0]['eigenvalues'])
                    if mask_rand.any():
                        rand_eigs_list.append(df[mask_rand].iloc[0]['eigenvalues'])
                
                if clean_eigs_list and rand_eigs_list:
                    # Compute mean and std
                    clean_eigs_mean = np.mean(clean_eigs_list, axis=0)
                    clean_eigs_std = np.std(clean_eigs_list, axis=0)
                    rand_eigs_mean = np.mean(rand_eigs_list, axis=0)
                    rand_eigs_std = np.std(rand_eigs_list, axis=0)
                    
                    # Plot with error bands
                    x = np.arange(len(clean_eigs_mean))
                    ax.semilogy(x, clean_eigs_mean, 'b-', label='Clean', linewidth=2)
                    ax.fill_between(x, clean_eigs_mean - clean_eigs_std, 
                                   clean_eigs_mean + clean_eigs_std, 
                                   alpha=0.3, color='b')
                    
                    ax.semilogy(x, rand_eigs_mean, 'r-', label='Random', linewidth=2)
                    ax.fill_between(x, rand_eigs_mean - rand_eigs_std, 
                                   rand_eigs_mean + rand_eigs_std, 
                                   alpha=0.3, color='r')
                    
                    # Add MP edges
                    lambda_plus_clean = df[(df['arch'] == arch) & (df['dataset'] == dataset) & 
                                         (df['label_condition'] == 'clean') & (df['step'] == step)].iloc[0]['lambda_plus']
                    lambda_plus_rand = df[(df['arch'] == arch) & (df['dataset'] == dataset) & 
                                        (df['label_condition'] == 'rand') & (df['step'] == step)].iloc[0]['lambda_plus']
                    
                    ax.axhline(y=lambda_plus_clean, color='b', linestyle='--', alpha=0.5, label='MP edge (clean)')
                    ax.axhline(y=lambda_plus_rand, color='r', linestyle='--', alpha=0.5, label='MP edge (rand)')
                    
                    ax.set_xlabel('Eigenvalue index')
                    ax.set_ylabel('Eigenvalue')
                    ax.set_title(f'Step {step}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(len(steps), len(axes)):
                fig.delaxes(axes[i])
            
            plt.suptitle(f'{arch} on {dataset.upper()} - Eigenvalue Distributions (Mean ± Std)')
            plt.tight_layout()
            save_path = save_dir / f'eigenvalues_{arch}_{dataset}_with_stats.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved eigenvalue distributions to {save_path}")


def create_summary_table_with_stats(df: pd.DataFrame, save_dir: Path):
    """Create summary table with AUC scores and confidence intervals."""
    summary_data = []
    
    architectures = sorted(df['arch'].unique())
    datasets = sorted(df['dataset'].unique())
    steps = sorted(df['step'].unique())
    
    for arch in architectures:
        for dataset in datasets:
            row = {'Architecture': arch, 'Dataset': dataset}
            
            for step in steps:
                results = compute_roc_auc_with_stats(df, arch, dataset, step)
                if results is not None:
                    row[f'AUC@{step}'] = f'{results["mean_auc"]:.3f} ± {results["std_auc"]:.3f}'
                    row[f'CI@{step}'] = f'[{results["ci_lower"]:.3f}, {results["ci_upper"]:.3f}]'
                else:
                    row[f'AUC@{step}'] = 'N/A'
                    row[f'CI@{step}'] = 'N/A'
            
            # Add ESM statistics across seeds
            mask_clean = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                        (df['label_condition'] == 'clean') & (df['step'] == 500)
            mask_rand = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                       (df['label_condition'] == 'rand') & (df['step'] == 500)
            
            if mask_clean.any() and mask_rand.any():
                esm_clean = df[mask_clean]['esm'].values
                esm_rand = df[mask_rand]['esm'].values
                
                row['ESM_clean@500'] = f'{np.mean(esm_clean):.4f} ± {np.std(esm_clean):.4f}'
                row['ESM_rand@500'] = f'{np.mean(esm_rand):.4f} ± {np.std(esm_rand):.4f}'
            
            summary_data.append(row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = save_dir / 'experiment1_summary_with_stats.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")
    
    # Display
    print("\nSummary Table with Statistics:")
    print(summary_df.to_string(index=False))
    
    return summary_df


def plot_esm_evolution_with_error_bars(df: pd.DataFrame, save_dir: Path):
    """Plot ESM evolution over training steps with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors and markers
    colors = {'resnet': 'blue', 'vit': 'orange'}
    markers = {'clean': 'o', 'rand': 's'}
    
    for i, dataset in enumerate(['cifar10', 'cifar100']):
        ax = axes[i]
        
        for arch in df['arch'].unique():
            for condition in ['clean', 'rand']:
                # Group by step and compute statistics
                mask = (df['arch'] == arch) & (df['dataset'] == dataset) & \
                      (df['label_condition'] == condition)
                
                if mask.any():
                    grouped = df[mask].groupby('step')['esm']
                    means = grouped.mean().sort_index()
                    stds = grouped.std().sort_index()
                    counts = grouped.count().sort_index()
                    
                    # Standard error
                    sems = stds / np.sqrt(counts)
                    
                    label = f'{arch}-{condition}'
                    ax.errorbar(means.index, means.values, yerr=sems.values,
                               color=colors.get(arch, 'gray'),
                               marker=markers.get(condition, 'o'),
                               markersize=8, linewidth=2,
                               label=label, capsize=5)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('ESM')
        ax.set_title(f'{dataset.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    save_path = save_dir / 'esm_evolution_with_error.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ESM evolution plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ESM experiment results with statistical aggregation')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./results/analysis',
                        help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    # Setup
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("Loading curvature results...")
    df = load_curvature_results(results_dir)
    
    if df.empty:
        print("No results found! Make sure to run experiments first.")
        return
    
    print(f"Loaded {len(df)} curvature measurements")
    print(f"Architectures: {df['arch'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Steps: {sorted(df['step'].unique())}")
    
    # Generate plots and analysis
    print("\nGenerating ROC curves with statistics...")
    plot_roc_curves_with_stats(df, output_dir)
    
    print("\nPlotting eigenvalue distributions with statistics...")
    plot_eigenvalue_distributions_by_seed(df, output_dir)
    
    print("\nPlotting ESM evolution with error bars...")
    plot_esm_evolution_with_error_bars(df, output_dir)
    
    print("\nCreating summary table with statistics...")
    summary_df = create_summary_table_with_stats(df, output_dir)
    
    # Check if we achieve the target (AUC >= 0.9 at step 500)
    print("\nChecking experiment success criteria:")
    success = True
    for _, row in summary_df.iterrows():
        auc_500 = row.get('AUC@500', 'N/A')
        if auc_500 != 'N/A':
            # Extract mean AUC value
            mean_auc = float(auc_500.split(' ± ')[0])
            ci_500 = row.get('CI@500', 'N/A')
            status = "✓" if mean_auc >= 0.9 else "✗"
            print(f"{status} {row['Architecture']} on {row['Dataset']}: AUC@500 = {auc_500}, CI = {ci_500}")
            if mean_auc < 0.9:
                success = False
    
    if success:
        print("\n✓ All experiments achieved mean AUC >= 0.9 at step 500!")
    else:
        print("\n✗ Some experiments did not achieve target AUC.")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()