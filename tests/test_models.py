#!/usr/bin/env python3
"""
Model-specific tests for ResNet and ViT architectures.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import timm

sys.path.append(str(Path(__file__).parent.parent))

from esm import count_parameters


def test_resnet18_creation():
    """Test ResNet-18 model creation and properties."""
    print("\nTesting ResNet-18 creation...")
    
    # Test CIFAR-10
    model_10 = models.resnet18(num_classes=10)
    n_params_10 = count_parameters(model_10)
    
    # Test CIFAR-100
    model_100 = models.resnet18(num_classes=100)
    n_params_100 = count_parameters(model_100)
    
    # Verify parameter counts
    assert n_params_10 > 11_000_000, f"ResNet-18 too small: {n_params_10}"
    assert n_params_10 < 12_000_000, f"ResNet-18 too large: {n_params_10}"
    
    # The difference should be 90 * 512 (fc layer)
    param_diff = n_params_100 - n_params_10
    expected_diff = 90 * 512
    assert abs(param_diff - expected_diff) < 100, f"Unexpected param diff: {param_diff}"
    
    print(f"  ResNet-18 (10 classes): {n_params_10:,} parameters")
    print(f"  ResNet-18 (100 classes): {n_params_100:,} parameters")
    print("✓ ResNet-18 creation test passed!")


def test_vit_creation():
    """Test ViT model creation and properties."""
    print("\nTesting ViT creation...")
    
    # Test ViT-Small/16 for CIFAR
    model = timm.create_model('vit_small_patch16_224', num_classes=10, img_size=32)
    n_params = count_parameters(model)
    
    # ViT-Small should have around 22M parameters
    assert n_params > 20_000_000, f"ViT-Small too small: {n_params}"
    assert n_params < 25_000_000, f"ViT-Small too large: {n_params}"
    
    # Test with different number of classes
    model_100 = timm.create_model('vit_small_patch16_224', num_classes=100, img_size=32)
    n_params_100 = count_parameters(model_100)
    
    # The difference should be in the head
    param_diff = n_params_100 - n_params
    assert param_diff > 0, "More classes should mean more parameters"
    
    print(f"  ViT-Small/16 (10 classes): {n_params:,} parameters")
    print(f"  ViT-Small/16 (100 classes): {n_params_100:,} parameters")
    print("✓ ViT creation test passed!")


def test_model_forward_pass():
    """Test forward pass for both models."""
    print("\nTesting model forward passes...")
    
    batch_size = 4
    img_size = 32
    
    # Create input
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Test ResNet
    resnet = models.resnet18(num_classes=10)
    resnet.eval()
    with torch.no_grad():
        out_resnet = resnet(x)
    
    assert out_resnet.shape == (batch_size, 10), f"Wrong ResNet output shape: {out_resnet.shape}"
    
    # Test ViT
    vit = timm.create_model('vit_small_patch16_224', num_classes=10, img_size=32)
    vit.eval()
    with torch.no_grad():
        out_vit = vit(x)
    
    assert out_vit.shape == (batch_size, 10), f"Wrong ViT output shape: {out_vit.shape}"
    
    print("  ResNet forward pass: ✓")
    print("  ViT forward pass: ✓")
    print("✓ Forward pass test passed!")


def test_model_gradients():
    """Test that models produce valid gradients."""
    print("\nTesting model gradients...")
    
    torch.manual_seed(42)
    batch_size = 8
    
    # Create data
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))
    
    models_to_test = {
        'ResNet': models.resnet18(num_classes=10),
        'ViT': timm.create_model('vit_small_patch16_224', num_classes=10, img_size=32)
    }
    
    for name, model in models_to_test.items():
        # Forward pass
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = False
        all_finite = True
        
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                if not torch.isfinite(param.grad).all():
                    all_finite = False
                    break
        
        assert has_grad, f"{name} has no gradients"
        assert all_finite, f"{name} has non-finite gradients"
        
        # Clear gradients
        model.zero_grad()
        
        print(f"  {name} gradients: ✓")
    
    print("✓ Gradient test passed!")


def test_model_modes():
    """Test train/eval mode switching."""
    print("\nTesting model train/eval modes...")
    
    models_to_test = {
        'ResNet': models.resnet18(num_classes=10),
        'ViT': timm.create_model('vit_small_patch16_224', num_classes=10, img_size=32)
    }
    
    x = torch.randn(2, 3, 32, 32)
    
    for name, model in models_to_test.items():
        # Test train mode
        model.train()
        assert model.training, f"{name} not in training mode"
        
        # Test eval mode
        model.eval()
        assert not model.training, f"{name} not in eval mode"
        
        # Test that batch norm behaves differently
        if name == 'ResNet':
            # ResNet has BatchNorm
            model.train()
            out_train = model(x)
            
            model.eval()
            with torch.no_grad():
                out_eval1 = model(x)
                out_eval2 = model(x)
            
            # In eval mode, outputs should be identical
            assert torch.allclose(out_eval1, out_eval2), "Eval outputs not deterministic"
        
        print(f"  {name} mode switching: ✓")
    
    print("✓ Model mode test passed!")


def test_model_initialization():
    """Test model initialization schemes."""
    print("\nTesting model initialization...")
    
    # Create models
    resnet = models.resnet18(num_classes=10)
    
    # Check conv layer initialization
    conv_layers = [m for m in resnet.modules() if isinstance(m, nn.Conv2d)]
    assert len(conv_layers) > 0, "No conv layers found"
    
    for conv in conv_layers:
        # Weights should have reasonable variance
        weight_std = conv.weight.std().item()
        assert 0.01 < weight_std < 1.0, f"Conv weight std out of range: {weight_std}"
        
        # Weights should be roughly centered
        weight_mean = conv.weight.mean().item()
        assert abs(weight_mean) < 0.1, f"Conv weight mean too large: {weight_mean}"
    
    # Check BatchNorm initialization
    bn_layers = [m for m in resnet.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_layers) > 0, "No BatchNorm layers found"
    
    for bn in bn_layers:
        # Weight should be close to 1
        assert torch.allclose(bn.weight, torch.ones_like(bn.weight), atol=0.1)
        # Bias should be close to 0
        assert torch.allclose(bn.bias, torch.zeros_like(bn.bias), atol=0.1)
    
    print("  Conv initialization: ✓")
    print("  BatchNorm initialization: ✓")
    print("✓ Initialization test passed!")


def test_model_device_transfer():
    """Test moving models between devices."""
    print("\nTesting model device transfer...")
    
    # Create model
    model = models.resnet18(num_classes=10)
    
    # Initially on CPU
    assert next(model.parameters()).device.type == 'cpu'
    
    # Test moving to same device (should work)
    model_cpu = model.to('cpu')
    assert model_cpu is model  # Should return same object
    
    # If CUDA available, test GPU transfer
    if torch.cuda.is_available():
        model_gpu = model.to('cuda')
        assert next(model_gpu.parameters()).device.type == 'cuda'
        
        # Move back to CPU
        model_cpu2 = model_gpu.to('cpu')
        assert next(model_cpu2.parameters()).device.type == 'cpu'
        
        print("  CPU ↔ GPU transfer: ✓")
    else:
        print("  CPU operations: ✓ (CUDA not available)")
    
    print("✓ Device transfer test passed!")


def test_parameter_groups():
    """Test creating parameter groups for different learning rates."""
    print("\nTesting parameter groups...")
    
    model = models.resnet18(num_classes=10)
    
    # Separate parameters by type
    conv_params = []
    bn_params = []
    fc_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'conv' in name:
            conv_params.append(param)
        elif 'bn' in name:
            bn_params.append(param)
        elif 'fc' in name:
            fc_params.append(param)
        else:
            other_params.append(param)
    
    # Verify we found parameters of each type
    assert len(conv_params) > 0, "No conv parameters found"
    assert len(bn_params) > 0, "No bn parameters found"
    assert len(fc_params) > 0, "No fc parameters found"
    
    # Create parameter groups
    param_groups = [
        {'params': conv_params, 'lr': 0.1},
        {'params': bn_params, 'lr': 0.01},
        {'params': fc_params, 'lr': 0.2}
    ]
    
    # Test with optimizer
    optimizer = torch.optim.SGD(param_groups)
    
    # Verify learning rates
    assert optimizer.param_groups[0]['lr'] == 0.1
    assert optimizer.param_groups[1]['lr'] == 0.01
    assert optimizer.param_groups[2]['lr'] == 0.2
    
    print(f"  Conv params: {len(conv_params)}")
    print(f"  BN params: {len(bn_params)}")
    print(f"  FC params: {len(fc_params)}")
    print("✓ Parameter groups test passed!")


def run_all_model_tests():
    """Run all model tests."""
    print("="*60)
    print("Running Model Tests")
    print("="*60)
    
    tests = [
        ("ResNet-18 creation", test_resnet18_creation),
        ("ViT creation", test_vit_creation),
        ("Model forward pass", test_model_forward_pass),
        ("Model gradients", test_model_gradients),
        ("Model train/eval modes", test_model_modes),
        ("Model initialization", test_model_initialization),
        ("Model device transfer", test_model_device_transfer),
        ("Parameter groups", test_parameter_groups)
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
        print("✓ All model tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test in failed:
            print(f"  - {test}")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_model_tests()
    sys.exit(0 if success else 1)