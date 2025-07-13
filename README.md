# ESM Experiment 1: Early Memorisation Detection Across Architectures

This repository implements Experiment 1 from the ESM (Early Spectral Memorization) paper, demonstrating that a single theory-derived threshold separates "clean" and "100% random-label" training runs on different architectures after ≤2000 gradient steps.

## Overview

The experiment compares ESM metrics between:
- **Models**: ResNet-18 and ViT-Small/16
- **Datasets**: CIFAR-10 and CIFAR-100
- **Label conditions**: Clean labels vs 100% randomly permuted labels
- **Checkpoints**: Curvature captured at steps 100, 500, and 2000

## Requirements

- 4 × NVIDIA A800 80GB GPUs (or similar)
- CUDA 12.2
- Python 3.10
- PyTorch 2.2.1

## Installation

### Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate esm-exp
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Installing PyTorch with CUDA

If PyTorch with CUDA is not properly installed:

```bash
pip install torch==2.2.1+cu122 torchvision==0.17.1+cu122 --index-url https://download.pytorch.org/whl/cu122
```

## Running Verification Tests

Before running the full experiment, verify the implementation:

```bash
python test_esm.py
```

All tests should pass with output like:
```
✓ Lanczos test passed. Top-5 eigenvalues: [...]
✓ ESM null test passed. ESM = 0.000123
✓ MP edge test passed. True: 4.000, Est: 3.987
✓ ROC test passed. AUC = 0.982
✓ All tests passed!
```

## Reproducing Experiment 1

### 1. Run All Experiments

The experiment consists of 6 training runs. Run them in parallel on your 4-GPU system:

```bash
# ResNet on CIFAR-10 (GPUs 0-1)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/resnet_cifar10_clean.yaml &

# ResNet on CIFAR-10 with random labels (GPUs 2-3)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/resnet_cifar10_rand.yaml &

# Wait for completion, then run next batch
wait

# ResNet on CIFAR-100 (GPUs 0-1)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/resnet_cifar100_clean.yaml &

# ResNet on CIFAR-100 with random labels (GPUs 2-3)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/resnet_cifar100_rand.yaml &

wait

# ViT on CIFAR-10 (GPUs 0-1)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/vit_cifar10_clean.yaml &

# ViT on CIFAR-10 with random labels (GPUs 2-3)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/vit_cifar10_rand.yaml &
```

### 2. Monitor Progress

You can monitor GPU usage and training progress:

```bash
# Watch GPU utilization
nvidia-smi -l 1

# Check training logs
tail -f results/*_curvature.pt
```

### 3. Analyze Results

After all experiments complete (approximately 6 hours), run the analysis:

```bash
python analyse.py --results_dir ./results --output_dir ./results/analysis
```

This will generate:
- ROC curves for each architecture/dataset/step combination
- Eigenvalue distribution plots
- ESM evolution plots
- Summary CSV with AUC scores

### 4. Expected Results

The experiment is successful if:
- AUC ≥ 0.9 at step 500 for all architecture/dataset combinations
- Clear separation between clean and random label ESM values
- Random label runs show higher ESM than clean label runs

Example output:
```
✓ resnet on cifar10: AUC@500 = 0.943
✓ resnet on cifar100: AUC@500 = 0.921
✓ vit on cifar10: AUC@500 = 0.958
✓ All experiments achieved AUC >= 0.9 at step 500!
```

## Running a Single Experiment

To run just one configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 train.py --config configs/resnet_cifar10_clean.yaml
```

## Customizing Experiments

Edit the YAML configuration files in `configs/` to modify:
- Batch size
- Learning rates
- Curvature checkpoints
- Number of top eigenvalues to compute

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in the config files:
```yaml
training:
  batch_size: 256  # Instead of 512
```

### Slow Curvature Computation

Reduce Lanczos iterations or number of eigenvalues:
```yaml
curvature:
  top_k: 20  # Instead of 30
  lanczos_max_iter: 40  # Instead of 80
```

### Determinism Issues

The code uses deterministic algorithms by default. If you encounter issues:
```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

## File Structure

```
.
├── configs/          # Experiment configurations
├── esm/             # Core ESM library
│   ├── hessian.py   # Lanczos algorithm
│   ├── mp_fit.py    # Marchenko-Pastur fitting
│   └── esm_metric.py # ESM computation
├── train.py         # Training script
├── analyse.py       # Analysis script
├── test_esm.py      # Verification tests
└── results/         # Output directory
    └── analysis/    # Plots and summaries
```

## Citation

If you use this code, please cite:

```bibtex
@article{esm2024,
  title={Early Spectral Memorization in Neural Networks},
  author={...},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.