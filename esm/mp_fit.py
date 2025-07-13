"""
Marchenko-Pastur edge estimation for bulk-edge separation.
"""

import numpy as np
from typing import Tuple, Optional


def estimate_mp_edge(eigenvalues: np.ndarray, 
                     n_params: int,
                     n_samples: int,
                     bulk_fraction: float = 0.9) -> Tuple[float, float]:
    """
    Estimate Marchenko-Pastur edge for bulk-edge separation.
    
    Args:
        eigenvalues: Array of eigenvalues (sorted descending)
        n_params: Total number of model parameters
        n_samples: Effective sample size (typically batch size)
        bulk_fraction: Fraction of eigenvalues to consider as bulk
    
    Returns:
        (lambda_plus, sigma_squared): MP edge and bulk variance estimate
    """
    # Remove top outliers for bulk variance estimation
    n_bulk = int(len(eigenvalues) * bulk_fraction)
    bulk_eigs = eigenvalues[-n_bulk:] if n_bulk > 0 else eigenvalues
    
    # Estimate bulk variance
    sigma_squared = np.mean(bulk_eigs)
    
    # Aspect ratio
    gamma = n_params / n_samples
    
    # Marchenko-Pastur edge
    lambda_plus = sigma_squared * (1 + np.sqrt(gamma))**2
    
    return lambda_plus, sigma_squared


def mp_pdf(x: np.ndarray, gamma: float, sigma_squared: float) -> np.ndarray:
    """
    Marchenko-Pastur probability density function.
    
    Args:
        x: Points at which to evaluate PDF
        gamma: Aspect ratio (p/n)
        sigma_squared: Variance parameter
    
    Returns:
        PDF values at x
    """
    # MP support
    lambda_minus = sigma_squared * (1 - np.sqrt(gamma))**2
    lambda_plus = sigma_squared * (1 + np.sqrt(gamma))**2
    
    # PDF
    pdf = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    
    pdf[mask] = (1 / (2 * np.pi * sigma_squared * gamma)) * \
                np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / x[mask]
    
    return pdf


def compute_effective_rank(eigenvalues: np.ndarray, 
                          lambda_plus: float,
                          total_params: Optional[int] = None) -> float:
    """
    Compute effective rank based on eigenvalues above MP edge.
    
    Args:
        eigenvalues: Array of eigenvalues
        lambda_plus: Marchenko-Pastur edge
        total_params: Total number of parameters (for scaling)
    
    Returns:
        Effective rank estimate
    """
    # Count eigenvalues above MP edge
    n_above = np.sum(eigenvalues > lambda_plus)
    
    if total_params is not None and len(eigenvalues) < total_params:
        # Scale up if we only have partial spectrum
        scaling = total_params / len(eigenvalues)
        n_above = int(n_above * scaling)
    
    return n_above