#!/usr/bin/env python3
"""
MAFT Training Script - IMPROVED

New features:
- LR scheduling (warmup + cosine annealing)
- Early stopping with patience
- Data augmentation support
- Better regression loss (L1+MSE)
- Progress bars with tqdm
- Comprehensive logging

Run with:
  python train.py --use_synthetic --device mps
  python train.py --config configs/cpu_test_config.yaml --use_synthetic --device cpu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

from losses.consistency import symmetric_kl_multi


def schedule_dropout(epoch, total_epochs, p_min=0.05, p_max=0.35):
    """Schedule modality dropout rate from p_min to p_max over training"""
    alpha = min(1.0, epoch / max(1, int(0.6 * total_epochs)))
    return p_min + (p_max - p_min) * alpha


def compute_loss(outputs, batch, lambdas):
    """
    IMPROVED: Compute multi-task loss with better regression loss.
    
    Args:
        outputs: Model output dictionary
        batch: Batch dictionary with targets
        lambdas: Loss weight dictionary
    
    Returns:
        total_loss, parts dictionary
    """
    # Classification loss
    cls_loss = F.cross_entropy(outputs["logits"], batch["classification_targets"])
    
    # IMPROVED: Combined L1+MSE for better regression learning
    if "regression_targets" in batch:
        reg_pred = outputs["reg"].squeeze(-1)
        reg_target = batch["regression_targets"]
        
        # Combine L1 (robust to outliers) + MSE (smooth gradients)
        l1_loss = F.l1_loss(reg_pred, reg_target)
        mse_loss = F.mse_loss(reg_pred, reg_target)
        reg_loss = 0.5 * l1_loss + 0.5 * mse_loss
    else:
        reg_loss = torch.tensor(0.0, device=cls_loss.device)
    
    # Consistency loss (only if at least two modalities present)
    logits_list = []
    if outputs["logits_text"].numel() > 0:
        logits_list.append(outputs["logits_text"])
    if outputs["logits_audio"].numel() > 0:
        logits_list.append(outputs["logits_audio"])
    if outputs["logits_visual"].numel() > 0:
        logits_list.append(outputs["logits_visual"])
    
    # Use lower temperature for stronger consistency signal
    cons_loss = symmetric_kl_multi(logits_list, temperature=1.5)
    
    # Total weighted loss
    total = (
        lambdas["cls"] * cls_loss + 
        lambdas["reg"] * reg_loss + 
        lambdas["cons"] * cons_loss
    )
    
    return total, {
        "classification_loss": cls_loss,
        "regression_loss": reg_loss,
        "consistency_loss": cons_loss,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs, 
                    lambdas, augmentation=None, grad_clip=1.0):
    """Train for one epoch with progress bar and augmentation."""
    model.train()
    model.moddrop.p = schedule_dropout(epoch, total_epochs)
    
    logs = dict(cls=0.0, reg=0.0, cons=0.0, tot=0.0, n=0)
    
    # Progress bar
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs}', leave=False)
    
    for batch in pbar:
        # Move batch to device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        # Apply augmentation if provided
        if augmentation is not None:
            batch = augmentation(batch, training=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(batch)
            loss, parts = compute_loss(outputs, batch, lambdas)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        logs["tot"] += loss.item()
        logs["cls"] += parts["classification_loss"].item()
        logs["reg"] += parts["regression_loss"].item()
        logs["cons"] += parts["consistency_loss"].item()
        logs["n"] += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{parts["classification_loss"].item():.4f}'
        })
    
    # Average losses
    for k in list(logs.keys()):
        if k != "n":
            logs[k] = logs[k] / max(logs["n"], 1)
    
    return logs


def validate(model, loader, device, lambdas):
    """Validate model with progress bar."""
    model.eval()
    logs = dict(cls=0.0, reg=0.0, cons=0.0, tot=0.0, n=0)
    
    pbar = tqdm(loader, desc='Validating', leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            outputs = model(batch)
            loss, parts = compute_loss(outputs, batch, lambdas)
            
            logs["tot"] += loss.item()
            logs["cls"] += parts["classification_loss"].item()
            logs["reg"] += parts["regression_loss"].item()
            logs["cons"] += parts["consistency_loss"].item()
            logs["n"] += 1
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    # Average losses
    for k in list(logs.keys()):
        if k != "n":
            logs[k] = logs[k] / max(logs["n"], 1)
    
    return logs


def create_lr_scheduler(optimizer, config):
    """Create learning rate scheduler with warmup + cosine annealing."""
    warmup_epochs = config['training'].get('warmup_epochs', 3)
    total_epochs = config['training']['num_epochs']
    min_lr_factor = config['training'].get('min_lr_factor', 0.01)
    base_lr = config['training']['lr']
    
    # Warmup: linearly increase LR from 10% to 100%
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Main: cosine annealing to min_lr
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=base_lr * min_lr_factor
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler


def main():
    """Main training function with all improvements."""
    parser = argparse.ArgumentParser(description='Train MAFT model (IMPROVED)')
    parser.add_argument('--config', type=str, default='configs/cpu_test_config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda/mps)')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic data instead of real data')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("MAFT TRAINING")
    print("="*70)
    print(f"Config: {args.config}")
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Create model
    from models.maft import MAFT
    model = MAFT(
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        audio_input_dim=config['model']['audio_input_dim'],
        visual_input_dim=config['model']['visual_input_dim'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        modality_dropout_rate=config['model']['modality_dropout_rate'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    
    # Create data loaders
    if args.use_synthetic or config['dataset']['name'] == 'synthetic':
        print("\n📦 Using synthetic data")
        from validation_system.synthetic_data import get_synthetic_loaders
        train_loader, val_loader = get_synthetic_loaders(
            batch_size=config['training']['batch_size'],
            num_train_batches=50,
            num_val_batches=10,
            seq_len_text=config['dataset']['max_length'],
            seq_len_audio=config['dataset'].get('audio_max_length', 200),
            seq_len_visual=config['dataset'].get('visual_max_length', 200),
        )
        print(f"Train: {len(train_loader)} batches")
        print(f"Val: {len(val_loader)} batches")
        
    elif config['dataset']['name'] == 'mosei':
        print(f"\n📦 Loading CMU-MOSEI benchmark dataset")
        from mosei_dataloader import get_mosei_loaders
        
        train_loader, val_loader, test_loader = get_mosei_loaders(
            data_dir=config['dataset']['data_dir'],
            batch_size=config['training']['batch_size'],
            max_text_len=config['dataset']['max_length'],
            max_audio_len=config['dataset'].get('audio_max_length', 500),
            max_visual_len=config['dataset'].get('visual_max_length', 500)
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Valid batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
    else:
        print(f"\n📦 Loading {config['dataset']['name']} dataset")
        print("⚠️  Dataset type not recognized")
        print("   Supported: 'synthetic', 'mosei'")
        print("   Use --use_synthetic flag for testing")
        return
    
    # Load augmentation if enabled
    augmentation = None
    if config['training'].get('use_augmentation', False):
        print("🎨 Using data augmentation")
        try:
            from utils.augmentation import get_augmentation_from_config
            augmentation = get_augmentation_from_config(config)
        except ImportError:
            print("⚠️  Augmentation module not found, continuing without it")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create LR scheduler
    scheduler = create_lr_scheduler(optimizer, config)
    
    # Create scaler for mixed precision
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Loss weights
    lambdas = {
        'cls': config['training']['classification_weight'],
        'reg': config['training']['regression_weight'],
        'cons': config['training']['consistency_weight']
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training'].get('early_stopping_patience', 7)
    
    # Training loop
    print(f"\n🏃 Training for {config['training']['num_epochs']} epochs...")
    if patience > 0:
        print(f"Early stopping patience: {patience}")
    
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_logs = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, config['training']['num_epochs'], lambdas,
            augmentation, config['training']['max_grad_norm']
        )
        
        # Validate
        val_logs = validate(model, val_loader, device, lambdas)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"  Train: {train_logs['tot']:.4f} (cls={train_logs['cls']:.4f}, reg={train_logs['reg']:.4f}, cons={train_logs['cons']:.4f})")
        print(f"  Val:   {val_logs['tot']:.4f} (cls={val_logs['cls']:.4f}, reg={val_logs['reg']:.4f}, cons={val_logs['cons']:.4f})")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model & early stopping
        if val_logs['tot'] < best_val_loss:
            best_val_loss = val_logs['tot']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }
            
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            
            print(f"  ✅ New best model! (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience > 0:
                print(f"  ⚠️  No improvement ({patience_counter}/{patience})")
            
            # Early stopping
            if patience > 0 and patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save periodic checkpoints
        save_every = config['training'].get('save_every_n_epochs', None)
        if save_every and (epoch + 1) % save_every == 0:
            periodic_checkpoint = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, periodic_checkpoint)
            print(f"  💾 Saved periodic checkpoint: epoch {epoch+1}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("="*70)


if __name__ == '__main__':
    main()