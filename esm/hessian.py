"""
Hessian computation utilities using Hessian-vector products and stochastic Lanczos.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    """Flatten list of parameter tensors into a single vector."""
    return torch.cat([p.view(-1) for p in params])


def unflatten_params(vec: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Unflatten vector into list of tensors with given shapes."""
    params = []
    offset = 0
    for shape in shapes:
        numel = np.prod(shape)
        params.append(vec[offset:offset + numel].view(shape))
        offset += numel
    return params


def hvp(loss: torch.Tensor, params: List[torch.Tensor], v: List[torch.Tensor], retain_graph: bool = True) -> List[torch.Tensor]:
    """
    Compute Hessian-vector product using the Pearlmutter trick.
    
    Args:
        loss: Scalar loss tensor with grad graph
        params: List of model parameters
        v: List of vectors to multiply with Hessian
        retain_graph: Whether to retain computation graph (set False for last iteration)
    
    Returns:
        Hessian-vector product as list of tensors
    """
    # First compute gradients
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    
    # Handle None gradients
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    
    # Compute grad-vector dot product - ensure dtype compatibility
    grad_dot_v = sum(torch.sum(g * vec.to(g.dtype)) for g, vec in zip(grads, v))
    
    # Second derivative to get Hv
    hvp_result = torch.autograd.grad(grad_dot_v, params, retain_graph=retain_graph, allow_unused=True)
    
    # Handle None results
    hvp_result = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_result, params)]
    
    return hvp_result


def lanczos_eigs(model: torch.nn.Module, 
                 loss: torch.Tensor, 
                 k: int = 30, 
                 max_iter: int = 80,
                 tol: float = 1e-6,
                 use_double: bool = True) -> np.ndarray:
    """
    Compute top-k eigenvalues using stochastic Lanczos algorithm.
    Memory-efficient implementation that only keeps 2 vectors on GPU.
    
    Args:
        model: Neural network model
        loss: Loss tensor with computation graph
        k: Number of top eigenvalues to compute
        max_iter: Maximum Lanczos iterations
        tol: Convergence tolerance
        use_double: Whether to use float64 for better numerical precision
    
    Returns:
        Array of top-k eigenvalues in descending order
    """
    params = [p for p in model.parameters() if p.requires_grad]
    param_shapes = [p.shape for p in params]
    total_params = sum(p.numel() for p in params)
    
    # Determine dtype
    dtype = torch.float64 if use_double else torch.float32
    device = params[0].device
    
    # Initialize random Rademacher vector (more efficient than Gaussian)
    v_flat = torch.randint(0, 2, (total_params,), device=device).to(dtype) * 2 - 1
    v_flat = v_flat / torch.norm(v_flat)
    
    # Convert to list format for HVP
    v_list = unflatten_params(v_flat, param_shapes)
    
    # Lanczos iteration - only store current and previous vectors
    alpha = []
    beta = [0.0]
    
    # For reorthogonalization, we need to store Q matrix on CPU
    Q = torch.zeros((total_params, min(max_iter, k + 10)), dtype=torch.float64)  # Always use float64 on CPU
    Q[:, 0] = v_flat.cpu().to(torch.float64)
    
    v_prev_list = None
    
    n_iter = min(max_iter, k + 10)
    for i in range(n_iter):  # Only need k + small buffer iterations
        # Compute Hv (retain graph except for last iteration)
        retain = (i < n_iter - 1)
        Hv = hvp(loss, params, v_list, retain_graph=retain)
        Hv_flat = flatten_params(Hv).to(dtype)
        
        # Compute alpha_i = v_i^T H v_i
        alpha_i = torch.dot(v_flat, Hv_flat).item()
        alpha.append(alpha_i)
        
        # Compute w = Hv - alpha_i * v_i - beta_i * v_{i-1}
        w_flat = Hv_flat - alpha_i * v_flat
        if i > 0 and v_prev_list is not None:
            v_prev_flat = flatten_params(v_prev_list).to(dtype)
            w_flat = w_flat - beta[i] * v_prev_flat
        
        # Reorthogonalization against all previous vectors (done on GPU for efficiency)
        if i < total_params - 1:
            # Move necessary part of Q to GPU for orthogonalization
            Q_gpu = Q[:, :i+1].to(device).to(dtype)
            coeffs = Q_gpu.T @ w_flat
            w_flat = w_flat - Q_gpu @ coeffs
        
        # Compute beta_{i+1} = ||w||
        beta_next = torch.norm(w_flat).item()
        beta.append(beta_next)
        
        # Check convergence
        if beta_next < tol or i >= k + 5:  # Early stop if we have enough
            break
        
        # Update vectors
        v_prev_list = v_list
        v_flat = w_flat / beta_next
        v_list = unflatten_params(v_flat, param_shapes)
        
        # Store new vector in Q (on CPU)
        if i + 1 < Q.shape[1]:
            Q[:, i + 1] = v_flat.cpu().to(torch.float64)
    
    # Build tridiagonal matrix
    n = len(alpha)
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = alpha[i]
        if i < n - 1:
            T[i, i + 1] = beta[i + 1]
            T[i + 1, i] = beta[i + 1]
    
    # Compute eigenvalues of tridiagonal matrix
    eigvals = np.linalg.eigvalsh(T)
    
    # Return top-k eigenvalues in descending order
    eigvals = np.sort(eigvals)[::-1]
    return eigvals[:k]


def compute_trace_hutchinson(model: torch.nn.Module,
                           loss: torch.Tensor,
                           n_samples: int = 10) -> float:
    """
    Estimate trace of Hessian using Hutchinson's estimator.
    
    Args:
        model: Neural network model
        loss: Loss tensor
        n_samples: Number of random vectors for estimation
    
    Returns:
        Estimated trace
    """
    params = [p for p in model.parameters() if p.requires_grad]
    param_shapes = [p.shape for p in params]
    
    trace_est = 0.0
    
    for _ in range(n_samples):
        # Random Rademacher vector (+1/-1 with equal probability)
        z_list = []
        for shape in param_shapes:
            z = torch.randint(0, 2, shape, device=params[0].device, dtype=params[0].dtype) * 2 - 1
            z_list.append(z)
        
        # Compute z^T H z
        Hz = hvp(loss, params, z_list, retain_graph=True)
        trace_est += sum(torch.sum(z * hz) for z, hz in zip(z_list, Hz)).item()
    
    return trace_est / n_samples