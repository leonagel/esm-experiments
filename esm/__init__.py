"""
ESM (Early Spectral Memorization) analysis library.
"""

from .hessian import lanczos_eigs, hvp, compute_trace_hutchinson
from .mp_fit import estimate_mp_edge, compute_effective_rank
from .esm_metric import compute_esm, compute_spectral_metrics_batch, analyze_memorization_signal
from .utils import (
    set_seed, get_git_hash, permute_labels, count_parameters,
    save_checkpoint, load_checkpoint, create_lr_scheduler, AverageMeter
)

__all__ = [
    'lanczos_eigs', 'hvp', 'compute_trace_hutchinson',
    'estimate_mp_edge', 'compute_effective_rank',
    'compute_esm', 'compute_spectral_metrics_batch', 'analyze_memorization_signal',
    'set_seed', 'get_git_hash', 'permute_labels', 'count_parameters',
    'save_checkpoint', 'load_checkpoint', 'create_lr_scheduler', 'AverageMeter'
]