# ESM Experiment Tests

Comprehensive test suite for the ESM (Early Spectral Memorization) experiment implementation.

## Test Categories

### 1. **test_esm.py** - Core ESM Tests
- Lanczos algorithm verification
- ESM metric computation
- Marchenko-Pastur edge estimation
- Edge cases and numerical stability
- Performance benchmarks

### 2. **test_integration.py** - Integration Tests
- Mini end-to-end experiments
- Clean vs random label comparison
- Model comparison (ResNet vs ViT)
- Curvature evolution during training
- Full pipeline with configs

### 3. **test_models.py** - Model Tests
- ResNet-18 creation and properties
- ViT model creation
- Forward pass verification
- Gradient computation
- Train/eval mode switching
- Parameter groups

### 4. **test_data.py** - Data Tests
- CIFAR-10/100 loading
- Data transforms and augmentation
- Label permutation for corruption
- DataLoader creation
- Dataset splitting
- Class distribution

### 5. **test_training.py** - Training Tests
- Optimizer creation (SGD, AdamW)
- Learning rate schedulers
- Checkpoint save/load
- Gradient accumulation
- Training loop components
- Deterministic training

### 6. **test_analysis.py** - Analysis Tests
- ROC curve and AUC computation
- Result aggregation
- Plotting functions
- Statistical analysis
- Metric visualization

## Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Individual Test Suites
```bash
# Core ESM tests
python tests/test_esm.py

# Integration tests
python tests/test_integration.py

# Model tests
python tests/test_models.py

# Data tests
python tests/test_data.py

# Training tests
python tests/test_training.py

# Analysis tests
python tests/test_analysis.py
```

## Test Requirements

- All tests run on CPU (no GPU required)
- Tests use small models and datasets for fast execution
- Total runtime: ~2-3 minutes for all tests
- Dependencies: Same as main experiment (see requirements.txt)

## Test Design Principles

1. **Fast Execution**: Use tiny models (<10K params) and small datasets
2. **CPU-Only**: No CUDA required, all tests run on laptop
3. **Deterministic**: Fixed seeds for reproducibility
4. **Comprehensive**: Cover all major components and edge cases
5. **Realistic**: Simulate actual experiment workflow

## Continuous Integration

These tests are designed to be CI-friendly:
- Exit code 0 on success, 1 on failure
- Clear output with test names and results
- Performance benchmarks to catch regressions
- No external dependencies beyond PyTorch ecosystem