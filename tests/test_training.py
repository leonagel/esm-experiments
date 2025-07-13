#!/usr/bin/env python3
"""
Training component tests including optimizers, schedulers, and checkpointing.
"""

import sys
import os
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from esm import (
    create_lr_scheduler, save_checkpoint, load_checkpoint,
    AverageMeter, set_seed, count_parameters
)


def test_optimizer_creation():
    """Test optimizer creation with different settings."""
    print("\nTesting optimizer creation...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Test SGD
    sgd_opt = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    assert sgd_opt.param_groups[0]['lr'] == 0.1
    assert sgd_opt.param_groups[0]['momentum'] == 0.9
    assert sgd_opt.param_groups[0]['weight_decay'] == 5e-4
    
    # Test AdamW
    adamw_opt = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        betas=(0.9, 0.95),
        weight_decay=1e-2
    )
    
    assert adamw_opt.param_groups[0]['lr'] == 5e-4
    assert adamw_opt.param_groups[0]['betas'] == (0.9, 0.95)
    assert adamw_opt.param_groups[0]['weight_decay'] == 1e-2
    
    print("  SGD optimizer: ✓")
    print("  AdamW optimizer: ✓")
    print("✓ Optimizer creation test passed!")


def test_lr_schedulers():
    """Test learning rate schedulers."""
    print("\nTesting learning rate schedulers...")
    
    model = nn.Linear(10, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Test cosine scheduler
    total_steps = 100
    scheduler_cosine = create_lr_scheduler(
        optimizer, total_steps, warmup_steps=0, schedule='cosine'
    )
    
    # Track LR over steps
    lrs_cosine = []
    for step in range(total_steps):
        lrs_cosine.append(optimizer.param_groups[0]['lr'])
        scheduler_cosine.step()
    
    # Verify cosine decay
    assert lrs_cosine[0] == 0.1, "Initial LR wrong"
    assert lrs_cosine[-1] < 0.01, "Final LR not decayed enough"
    assert all(lrs_cosine[i] >= lrs_cosine[i+1] for i in range(len(lrs_cosine)-1)), \
        "LR not monotonically decreasing"
    
    # Test with warmup
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler_warmup = create_lr_scheduler(
        optimizer, total_steps, warmup_steps=10, schedule='cosine'
    )
    
    lrs_warmup = []
    for step in range(20):
        lrs_warmup.append(optimizer.param_groups[0]['lr'])
        scheduler_warmup.step()
    
    # Verify warmup
    assert lrs_warmup[0] < lrs_warmup[9], "Warmup not increasing"
    assert lrs_warmup[10] >= lrs_warmup[11], "Not decreasing after warmup"
    
    print("  Cosine scheduler: ✓")
    print("  Warmup: ✓")
    print("✓ LR scheduler test passed!")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\nTesting checkpoint save/load...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model and optimizer
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few steps
        for _ in range(5):
            x = torch.randn(4, 10)
            y = torch.randn(4, 10)
            
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            optimizer.step()
        
        # Save checkpoint
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        state = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'custom_data': {'test': 123}
        }
        save_checkpoint(state, checkpoint_path)
        
        # Verify file exists
        assert os.path.exists(checkpoint_path), "Checkpoint not saved"
        
        # Create new model and optimizer
        model2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        
        # Load checkpoint
        loaded_state = load_checkpoint(checkpoint_path, model2, optimizer2)
        
        # Verify loading
        assert loaded_state['epoch'] == 5
        assert loaded_state['custom_data']['test'] == 123
        
        # Verify model weights are restored
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Model weights not restored"
        
        # Verify optimizer state is restored (check keys match)
        opt1_state = optimizer.state_dict()
        opt2_state = optimizer2.state_dict()
        assert set(opt1_state.keys()) == set(opt2_state.keys()), "Optimizer state keys mismatch"
        
        print("  Save checkpoint: ✓")
        print("  Load checkpoint: ✓")
        print("  State restoration: ✓")
        print("✓ Checkpoint test passed!")


def test_average_meter():
    """Test AverageMeter utility."""
    print("\nTesting AverageMeter...")
    
    meter = AverageMeter()
    
    # Test initial state
    assert meter.val == 0
    assert meter.avg == 0
    assert meter.sum == 0
    assert meter.count == 0
    
    # Add values
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for val in values:
        meter.update(val)
    
    # Check results
    assert meter.val == 5.0, "Last value wrong"
    assert meter.count == 5, "Count wrong"
    assert meter.sum == sum(values), "Sum wrong"
    assert abs(meter.avg - np.mean(values)) < 1e-6, "Average wrong"
    
    # Test batch update
    meter.reset()
    meter.update(10.0, n=5)  # 5 samples with value 10
    
    assert meter.count == 5
    assert meter.sum == 50.0
    assert meter.avg == 10.0
    
    # Test reset
    meter.reset()
    assert meter.count == 0
    assert meter.avg == 0
    
    print("  Single updates: ✓")
    print("  Batch updates: ✓")
    print("  Reset: ✓")
    print("✓ AverageMeter test passed!")


def test_gradient_accumulation():
    """Test gradient accumulation pattern."""
    print("\nTesting gradient accumulation...")
    
    model = nn.Linear(10, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Save initial weights
    initial_weight = model.weight.clone()
    
    # Method 1: Single large batch
    torch.manual_seed(42)
    x_large = torch.randn(40, 10)
    y_large = torch.randn(40, 10)
    
    optimizer.zero_grad()
    loss1 = nn.MSELoss()(model(x_large), y_large)
    loss1.backward()
    optimizer.step()
    
    weight_after_large = model.weight.clone()
    
    # Reset model
    model.weight.data = initial_weight.clone()
    optimizer.zero_grad()
    
    # Method 2: Gradient accumulation
    torch.manual_seed(42)
    accumulation_steps = 4
    
    for i in range(accumulation_steps):
        x_small = x_large[i*10:(i+1)*10]
        y_small = y_large[i*10:(i+1)*10]
        
        loss2 = nn.MSELoss()(model(x_small), y_small)
        loss2 = loss2 / accumulation_steps  # Scale loss
        loss2.backward()
    
    optimizer.step()
    weight_after_accum = model.weight.clone()
    
    # Results should be very similar (allow slightly more tolerance for numerical precision)
    assert torch.allclose(weight_after_large, weight_after_accum, atol=1e-4), \
        "Gradient accumulation produces different results"
    
    print("  Single batch update: ✓")
    print("  Accumulated updates: ✓")
    print("  Equivalence: ✓")
    print("✓ Gradient accumulation test passed!")


def test_training_loop_components():
    """Test components of a training loop."""
    print("\nTesting training loop components...")
    
    # Setup
    model = nn.Linear(10, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy data
    train_data = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randint(0, 2, (100,))
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10)
    
    # Training metrics
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    # One epoch
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        assert model.weight.grad is not None, "No gradients"
        assert not torch.isnan(model.weight.grad).any(), "NaN gradients"
        
        # Update
        optimizer.step()
        
        # Metrics
        loss_meter.update(loss.item(), inputs.size(0))
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Verify metrics
    assert loss_meter.count == len(train_data), "Not all samples processed"
    accuracy = 100. * correct / total
    assert 0 <= accuracy <= 100, f"Invalid accuracy: {accuracy}"
    
    print(f"  Average loss: {loss_meter.avg:.4f}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print("✓ Training loop test passed!")


def test_mixed_precision_compatibility():
    """Test mixed precision training compatibility."""
    print("\nTesting mixed precision compatibility...")
    
    if not torch.cuda.is_available():
        print("  Skipping (CUDA not available)")
        return
    
    model = nn.Linear(10, 10).cuda()
    optimizer = optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    
    # Training step with AMP
    x = torch.randn(4, 10).cuda()
    y = torch.randn(4, 10).cuda()
    
    with torch.cuda.amp.autocast():
        output = model(x)
        loss = nn.MSELoss()(output, y)
    
    # Scale loss and backward
    scaler.scale(loss).backward()
    
    # Check for scaled gradients
    assert model.weight.grad is not None
    
    # Unscale and step
    scaler.step(optimizer)
    scaler.update()
    
    print("  Autocast forward: ✓")
    print("  Scaled backward: ✓")
    print("  Optimizer step: ✓")
    print("✓ Mixed precision test passed!")


def test_deterministic_training():
    """Test training determinism with fixed seeds."""
    print("\nTesting deterministic training...")
    
    def train_model(seed):
        set_seed(seed)
        
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # Fixed data with seed
        torch.manual_seed(seed)
        x = torch.randn(20, 10)
        y = torch.randn(20, 10)
        
        # Train for a few steps
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        return losses, model.weight.clone()
    
    # Train twice with same seed
    losses1, weights1 = train_model(42)
    losses2, weights2 = train_model(42)
    
    # Should be identical
    assert losses1 == losses2, "Losses not deterministic"
    assert torch.allclose(weights1, weights2), "Weights not deterministic"
    
    # Train with different seed
    losses3, weights3 = train_model(123)
    
    # Should be different
    assert losses1 != losses3, "Different seeds produced same results"
    assert not torch.allclose(weights1, weights3), "Different seeds produced same weights"
    
    print("  Same seed reproducibility: ✓")
    print("  Different seed variation: ✓")
    print("✓ Deterministic training test passed!")


def test_nan_inf_detection():
    """Test detection of NaN/Inf in training."""
    print("\nTesting NaN/Inf detection...")
    
    model = nn.Linear(10, 10)
    
    # Create data that might cause issues
    x = torch.randn(4, 10)
    x[0, 0] = float('inf')  # Add infinity
    
    # Forward pass
    try:
        output = model(x)
        
        # Check for inf in output
        has_inf = torch.isinf(output).any()
        assert has_inf or torch.isnan(output).any(), \
            "Model should propagate inf/nan"
        
    except RuntimeError:
        # Some operations might raise errors with inf
        pass
    
    # Test gradient checking
    def check_gradients(model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    return False, f"NaN in {name} gradients"
                if torch.isinf(param.grad).any():
                    return False, f"Inf in {name} gradients"
        return True, "All gradients valid"
    
    # Normal case
    model.zero_grad()
    x_normal = torch.randn(4, 10)
    y_normal = torch.randn(4, 10)
    loss = nn.MSELoss()(model(x_normal), y_normal)
    loss.backward()
    
    is_valid, msg = check_gradients(model)
    assert is_valid, msg
    
    print("  Inf propagation: ✓")
    print("  Gradient validation: ✓")
    print("✓ NaN/Inf detection test passed!")


def run_all_training_tests():
    """Run all training tests."""
    print("="*60)
    print("Running Training Tests")
    print("="*60)
    
    tests = [
        ("Optimizer creation", test_optimizer_creation),
        ("LR schedulers", test_lr_schedulers),
        ("Checkpoint save/load", test_checkpoint_save_load),
        ("AverageMeter", test_average_meter),
        ("Gradient accumulation", test_gradient_accumulation),
        ("Training loop components", test_training_loop_components),
        ("Mixed precision compatibility", test_mixed_precision_compatibility),
        ("Deterministic training", test_deterministic_training),
        ("NaN/Inf detection", test_nan_inf_detection)
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
        print("✓ All training tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test in failed:
            print(f"  - {test}")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_training_tests()
    sys.exit(0 if success else 1)