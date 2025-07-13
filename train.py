#!/usr/bin/env python3
"""
Distributed training script with curvature capture for ESM experiments.
"""

import os
import time
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import timm

from esm import (
    lanczos_eigs, compute_esm, set_seed, get_git_hash,
    permute_labels, count_parameters, save_checkpoint,
    create_lr_scheduler, AverageMeter
)


def get_transform(dataset_name: str, is_train: bool = True):
    """Get data transformations for CIFAR datasets."""
    if dataset_name in ['cifar10', 'cifar100']:
        mean = (0.4914, 0.4822, 0.4465) if dataset_name == 'cifar10' else (0.5071, 0.4867, 0.4408)
        std = (0.2023, 0.1994, 0.2010) if dataset_name == 'cifar10' else (0.2675, 0.2565, 0.2761)
        
        if is_train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transform


def get_dataset(cfg):
    """Load dataset with optional label corruption."""
    transform_train = get_transform(cfg.dataset.name, is_train=True)
    transform_test = get_transform(cfg.dataset.name, is_train=False)
    
    # Check ./data first, then fall back to config path
    data_root = cfg.dataset.root
    if Path('./data').exists():
        if cfg.dataset.name == 'cifar10' and (Path('./data') / 'cifar-10-batches-py').exists():
            data_root = './data'
            if data_root != cfg.dataset.root:
                print(f"Using CIFAR-10 data from ./data instead of {cfg.dataset.root}")
        elif cfg.dataset.name == 'cifar100' and (Path('./data') / 'cifar-100-python').exists():
            data_root = './data'
            if data_root != cfg.dataset.root:
                print(f"Using CIFAR-100 data from ./data instead of {cfg.dataset.root}")
    
    if cfg.dataset.name == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=cfg.dataset.download, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=cfg.dataset.download, transform=transform_test
        )
    elif cfg.dataset.name == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=data_root, train=True, download=cfg.dataset.download, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=data_root, train=False, download=cfg.dataset.download, transform=transform_test
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    # Apply label corruption if specified
    if cfg.dataset.label_corruption:
        permute_labels(train_dataset, cfg.dataset.corruption_seed)
    
    return train_dataset, test_dataset


def get_model(cfg):
    """Create model based on configuration."""
    if cfg.model.name == 'resnet18':
        model = torchvision.models.resnet18(num_classes=cfg.model.num_classes)
        # He initialization
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    elif cfg.model.name == 'vit_small_patch16_224':
        model = timm.create_model(
            'vit_small_patch16_224',
            num_classes=cfg.model.num_classes,
            img_size=cfg.model.img_size
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    
    return model


def get_optimizer(model, cfg):
    """Create optimizer based on configuration."""
    if cfg.training.optimizer.type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            momentum=cfg.training.optimizer.momentum,
            weight_decay=cfg.training.optimizer.weight_decay
        )
    elif cfg.training.optimizer.type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            betas=cfg.training.optimizer.betas,
            weight_decay=cfg.training.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer.type}")
    
    return optimizer


def capture_curvature(model, train_loader, device, cfg, global_step, rank, seed):
    """Capture curvature information at specified checkpoint."""
    model.eval()
    
    # Only compute on rank 0 to avoid duplicate work
    if rank == 0:
        # Disable torch.compile for curvature computation if model is compiled
        is_compiled = hasattr(model, '_orig_mod')
        if is_compiled:
            original_model = model._orig_mod
        else:
            original_model = model
        
        # Get a random batch for curvature computation
        dataloader = DataLoader(
            train_loader.dataset,
            batch_size=cfg.curvature.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Get one batch without augmentation
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            break
        
        # Forward pass (use original model if compiled)
        with torch.enable_grad():
            outputs = original_model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Compute top-k eigenvalues
        print(f"Computing curvature at step {global_step}...")
        start_time = time.time()
        
        eigenvalues = lanczos_eigs(
            original_model, loss,
            k=cfg.curvature.top_k,
            max_iter=cfg.curvature.lanczos_max_iter,
            use_double=True  # Use float64 for better numerical precision
        )
        
        # Compute ESM metrics
        n_params = count_parameters(original_model)
        metrics = compute_esm(
            eigenvalues,
            n_params=n_params,
            n_samples=cfg.curvature.batch_size
        )
        
        elapsed = time.time() - start_time
        print(f"Curvature computation took {elapsed:.2f}s")
        
        # Save results
        results = {
            'step': global_step,
            'eigenvalues': eigenvalues,
            'metrics': metrics,
            'n_params': n_params,
            'elapsed_time': elapsed,
            'git_hash': get_git_hash()
        }
        
        # Save to file
        save_path = Path(cfg.experiment.save_dir) / f"{cfg.experiment.name}_seed{seed}_step{global_step}_curvature.pt"
        torch.save(results, save_path)
        print(f"Saved curvature to {save_path}")
    else:
        # Other ranks wait
        metrics = None
    
    # Synchronize all ranks
    if dist.is_initialized():
        dist.barrier()
    
    model.train()
    return metrics


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, cfg, global_step, rank, seed):
    """Train for one epoch with curvature capture."""
    model.train()
    losses = AverageMeter()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Check if we need to capture curvature
        if global_step in cfg.curvature.checkpoints:
            capture_curvature(model, train_loader, device, cfg, global_step, rank, seed)
        
        # Forward pass
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        global_step += 1
        
        # Log progress
        if batch_idx % 50 == 0 and rank == 0:
            print(f"Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] "
                  f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return global_step


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def main(rank, world_size, cfg, seed):
    """Main training function for each process."""
    # Setup distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(cfg.distributed.backend, rank=rank, world_size=world_size)
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    
    # Set seed
    set_seed(seed + rank)
    
    # Load data
    train_dataset, test_dataset = get_dataset(cfg)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size // world_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    model = get_model(cfg).to(device)
    if cfg.compute.compile_model and torch.cuda.is_available():
        model = torch.compile(model)
    
    # Wrap with DDP - different args for CPU vs GPU
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, cfg)
    total_steps = len(train_loader) * cfg.training.epochs
    scheduler = create_lr_scheduler(
        optimizer, total_steps,
        warmup_steps=cfg.training.scheduler.warmup_steps,
        schedule=cfg.training.scheduler.type
    )
    
    # Training loop
    global_step = 0
    
    if rank == 0:
        print(f"Starting training: {cfg.experiment.name} (seed={seed})")
        print(f"Model parameters: {count_parameters(model):,}")
        print(f"Dataset: {cfg.dataset.name}, Corrupted: {cfg.dataset.label_corruption}")
    
    for epoch in range(cfg.training.epochs):
        train_sampler.set_epoch(epoch)
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, cfg, global_step, rank, seed
        )
        
        # Evaluate
        if rank == 0:
            accuracy = evaluate(model, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy = {accuracy:.2f}%")
    
    # Final checkpoint
    if rank == 0:
        final_state = {
            'epoch': cfg.training.epochs,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': OmegaConf.to_container(cfg),
            'git_hash': get_git_hash()
        }
        save_path = Path(cfg.experiment.save_dir) / f"{cfg.experiment.name}_seed{seed}_final.pt"
        save_checkpoint(final_state, str(save_path))
    
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Create save directory
    Path(cfg.experiment.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Launch distributed training for each seed
    world_size = cfg.distributed.devices
    
    # Run training for each seed
    for seed in cfg.experiment.seeds:
        print(f"\n{'='*50}")
        print(f"Starting run with seed {seed}")
        print(f"{'='*50}\n")
        
        torch.multiprocessing.spawn(main, args=(world_size, cfg, seed), nprocs=world_size)
        
        # Clean up between runs
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()