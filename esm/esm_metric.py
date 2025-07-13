"""
Early Spectral Memorization (ESM) metric computation.
"""

import numpy as np
from typing import Dict, List, Optional
from .mp_fit import estimate_mp_edge


def compute_esm(eigenvalues: np.ndarray,
                n_params: int,
                n_samples: int,
                bulk_fraction: float = 0.9) -> Dict[str, float]:
    """
    Compute ESM metric and related spectral quantities.
    
    Args:
        eigenvalues: Top-k eigenvalues (sorted descending)
        n_params: Total number of model parameters
        n_samples: Effective sample size (batch size)
        bulk_fraction: Fraction of eigenvalues for bulk estimation
    
    Returns:
        Dictionary with ESM, sharpness, trace, and MP edge
    """
    # Estimate MP edge
    lambda_plus, sigma_squared = estimate_mp_edge(
        eigenvalues, n_params, n_samples, bulk_fraction
    )
    
    # Compute excess spectral mass
    excess_eigs = eigenvalues[eigenvalues > lambda_plus]
    excess_mass = np.sum(excess_eigs - lambda_plus) if len(excess_eigs) > 0 else 0.0
    
    # Trace normalization
    total_mass = np.sum(eigenvalues)
    esm = excess_mass / total_mass if total_mass > 0 else 0.0
    
    # Sharpness (largest eigenvalue)
    sharpness = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
    
    # Trace estimation (scale up if partial spectrum)
    k = len(eigenvalues)
    trace_estimate = total_mass * (n_params / k) if k < n_params else total_mass
    
    return {
        'esm': esm,
        'sharpness': sharpness,
        'trace': trace_estimate,
        'lambda_plus': lambda_plus,
        'sigma_squared': sigma_squared,
        'n_outliers': len(excess_eigs)
    }


def compute_spectral_metrics_batch(eigenvalue_list: List[np.ndarray],
                                  n_params: int,
                                  n_samples: int) -> List[Dict[str, float]]:
    """
    Compute ESM metrics for a batch of eigenvalue arrays.
    
    Args:
        eigenvalue_list: List of eigenvalue arrays
        n_params: Total number of model parameters
        n_samples: Effective sample size
    
    Returns:
        List of metric dictionaries
    """
    return [
        compute_esm(eigs, n_params, n_samples)
        for eigs in eigenvalue_list
    ]


def analyze_memorization_signal(clean_metrics: List[Dict[str, float]],
                               random_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Analyze separation between clean and random label training.
    
    Args:
        clean_metrics: ESM metrics from clean label runs
        random_metrics: ESM metrics from random label runs
    
    Returns:
        Dictionary with separation statistics
    """
    clean_esm = [m['esm'] for m in clean_metrics]
    random_esm = [m['esm'] for m in random_metrics]
    
    # Compute separation statistics
    clean_mean = np.mean(clean_esm)
    random_mean = np.mean(random_esm)
    clean_std = np.std(clean_esm)
    random_std = np.std(random_esm)
    
    # Cohen's d effect size
    pooled_std = np.sqrt((clean_std**2 + random_std**2) / 2)
    cohens_d = (random_mean - clean_mean) / pooled_std if pooled_std > 0 else 0.0
    
    # Simple overlap test
    clean_max = np.max(clean_esm)
    random_min = np.min(random_esm)
    separated = random_min > clean_max
    
    return {
        'clean_esm_mean': clean_mean,
        'clean_esm_std': clean_std,
        'random_esm_mean': random_mean,
        'random_esm_std': random_std,
        'cohens_d': cohens_d,
        'separated': separated,
        'separation_gap': random_min - clean_max if separated else 0.0
    }