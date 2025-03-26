"""
Finetuning script for vision backbones on ImageNet with DDP support.
Run with:
    OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 examples/finetune.py --local_rank 0 
"""

import argparse
import json
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from helper import (
    build_backbone, init_distributed_mode, is_main_process, get_rank, 
    save_on_master, average_metrics
)

def get_parser():
    parser = argparse.ArgumentParser(description="Finetune vision backbones on ImageNet with DDP support")
    # Base arguments
    parser.add_argument("--output_dir", type=str, default='output', help="Directory to save checkpoints")
    parser.add_argument("--data_dir", type=str, default="/datasets01/imagenet_full_size/061417", help="Directory with training data")
    parser.add_argument("--val_dir", type=str, default=None, help="Directory with validation data, defaults to data_dir if not specified")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14", help="Model architecture to use")
    parser.add_argument("--model_path", type=str, default="", help="Path to pre-trained model weights")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes for classification")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=90, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Total batch size across all GPUs")
    parser.add_argument("--lr", type=float, default=0.1, help="Base learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["step", "cosine"], help="LR scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Data preprocessing
    parser.add_argument("--resize_size", type=int, default=256, help="Resize images to this size")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size for training/validation")
    parser.add_argument("--color_jitter", type=float, default=0.3, help="Color jitter strength")
    
    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--master_port", type=int, default=-1, help="Master port for DDP")
    parser.add_argument("--debug_slurm", type=bool, default=False, help="Debug SLURM setup")
    
    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    
    return parser

def main(params):
    # Set random seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)

    # Initialize distributed training
    init_distributed_mode(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    if is_main_process():
        os.makedirs(params.output_dir, exist_ok=True)
        with open(os.path.join(params.output_dir, "args.json"), "w") as f:
            json.dump(vars(params), f, indent=2)

    # Set up data preprocessing
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(params.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=params.color_jitter,
            contrast=params.color_jitter,
            saturation=params.color_jitter,
            hue=params.color_jitter/2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(params.resize_size),
        transforms.CenterCrop(params.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets and dataloaders
    val_dir = params.val_dir if params.val_dir else os.path.join(params.data_dir, "val")
    
    train_dataset = ImageFolder(os.path.join(params.data_dir, "train"), transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=params.world_size, 
        rank=params.global_rank, 
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=params.world_size, 
        rank=params.global_rank, 
        shuffle=False
    )
    
    # Scale down per-GPU batch size based on world size
    per_device_batch_size = params.batch_size // params.world_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        sampler=train_sampler,
        num_workers=10,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        sampler=val_sampler,
        num_workers=10,
        pin_memory=True,
        drop_last=False
    )
    
    # Build model
    print(f"[Rank {get_rank()}] Building model: {params.model_name}")
    backbone = build_backbone(path=params.model_path if params.model_path else None, name=params.model_name)
    
    # Add classification head
    # Create dummy input to determine feature dimension
    dummy_input = torch.rand(1, 3, params.crop_size, params.crop_size)
    with torch.no_grad():
        features = backbone(dummy_input)
    feature_dim = features.shape[1]
    print(f"[Rank {get_rank()}] Feature dimension: {feature_dim}")
    
    # Create classifier
    classifier = nn.Linear(feature_dim, params.num_classes).to(device)
    
    # Move model to device and wrap with DDP
    backbone = backbone.to(device)
    backbone = DDP(backbone, device_ids=[params.local_rank], output_device=params.local_rank, find_unused_parameters=True)
    
    classifier = classifier.to(device)
    classifier = DDP(classifier, device_ids=[params.local_rank], output_device=params.local_rank)
    
    # Define optimization
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.SGD(
        parameters, 
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.wd
    )
    
    # Learning rate scheduler
    if params.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, params.epochs - params.warmup_epochs
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if params.resume:
        if os.path.isfile(params.resume):
            print(f"[Rank {get_rank()}] Loading checkpoint '{params.resume}'")
            checkpoint = torch.load(params.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('best_acc', 0.0)
            backbone.module.load_state_dict(checkpoint['backbone'])
            classifier.module.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print(f"[Rank {get_rank()}] Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"[Rank {get_rank()}] No checkpoint found at '{params.resume}'")

    # Training loop
    for epoch in range(start_epoch, params.epochs):
        train_sampler.set_epoch(epoch)
        
        # Apply warmup learning rate
        if epoch < params.warmup_epochs:
            warmup_factor = (epoch + 1) / params.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = params.lr * warmup_factor
        
        # Train for one epoch
        train_loss, train_acc1 = train_one_epoch(
            backbone, classifier, train_loader, optimizer, epoch, device, params
        )
        
        # Update learning rate
        if epoch >= params.warmup_epochs:
            lr_scheduler.step()
        
        # Evaluate on validation set
        val_loss, val_acc1 = validate(
            backbone, classifier, val_loader, device, params
        )
        
        # Log metrics
        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch}/{params.epochs-1}, LR: {current_lr:.6f}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc1:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc@1: {val_acc1:.2f}%")
            
            # Log in format that can be parsed by nbutils
            metrics = {
                'train_loss': train_loss,
                'train_acc1': train_acc1,
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'lr': current_lr,
                'epoch': epoch
            }
            print(f"__log__:rawlogs: {metrics}")

        # Save checkpoint
        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)
        
        if (epoch + 1) % params.save_freq == 0 or is_best or epoch == params.epochs - 1:
            if is_main_process():
                save_dict = {
                    'epoch': epoch + 1,
                    'backbone': backbone.module.state_dict(),
                    'classifier': classifier.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'best_acc': best_acc
                }
                
                save_on_master(
                    save_dict,
                    os.path.join(params.output_dir, f'checkpoint_{epoch+1:03d}.pth')
                )
                
                if is_best:
                    save_on_master(
                        save_dict,
                        os.path.join(params.output_dir, 'checkpoint_best.pth')
                    )
        
        # Sync all processes before starting next epoch
        dist.barrier()
    
    if is_main_process():
        print(f"Training finished. Best accuracy: {best_acc:.2f}%")

def train_one_epoch(backbone, classifier, dataloader, optimizer, epoch, device, params):
    backbone.train()
    classifier.train()
    
    loss_meter, acc1_meter = 0.0, 0.0
    total_samples = 0
    
    start_time = time.time()
    
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = torch.tensor([t for t in targets], dtype=torch.long).to(device, non_blocking=True)
        
        # Zero gradients for every batch
        optimizer.zero_grad()
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda'):
            features = backbone(images)
            logits = classifier(features)
            loss = F.cross_entropy(logits, targets)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        acc1 = compute_accuracy(logits, targets, topk=(1,))[0]
        
        # Update meters
        batch_size = images.size(0)
        loss_meter += loss.item() * batch_size
        acc1_meter += acc1.item() * batch_size
        total_samples += batch_size
        
        if i % 20 == 0 and is_main_process():
            print(f"Train Epoch: {epoch} [{i}/{len(dataloader)}]\t"
                  f"Loss: {loss.item():.4f}\tAcc@1: {acc1.item():.2f}%\t"
                  f"Time: {time.time() - start_time:.2f}s")
            start_time = time.time()
    
    # Average metrics across all processes
    metrics = {"loss": loss_meter / total_samples, "acc1": acc1_meter / total_samples}
    averaged_metrics = average_metrics(metrics, count=total_samples)
    
    return averaged_metrics["loss"], averaged_metrics["acc1"]

def validate(backbone, classifier, dataloader, device, params):
    backbone.eval()
    classifier.eval()
    
    loss_meter, acc1_meter = 0.0, 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = torch.tensor([t for t in targets], dtype=torch.long).to(device, non_blocking=True)
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda'):
                features = backbone(images)
                logits = classifier(features)
                loss = F.cross_entropy(logits, targets)
            
            # Compute accuracy
            acc1 = compute_accuracy(logits, targets, topk=(1,))[0]
            
            # Update meters
            batch_size = images.size(0)
            loss_meter += loss.item() * batch_size
            acc1_meter += acc1.item() * batch_size
            total_samples += batch_size
    
    # Average metrics across all processes
    metrics = {"loss": loss_meter / total_samples, "acc1": acc1_meter / total_samples}
    averaged_metrics = average_metrics(metrics, count=total_samples)
    
    return averaged_metrics["loss"], averaged_metrics["acc1"]

def compute_accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    params = get_parser().parse_args()
    print("__log__:{}".format(json.dumps(vars(params))))
    main(params)
