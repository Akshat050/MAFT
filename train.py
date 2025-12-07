#!/usr/bin/env python3
"""
Complete MAFT Training Script

Run with:
  python train_complete.py --use_synthetic  # Use synthetic data (no download needed)
  python train_complete.py --config configs/mosei_config.yaml  # Use real data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import yaml
import argparse
from pathlib import Path

from losses.consistency import symmetric_kl_multi


def schedule_dropout(epoch, total_epochs, p_min=0.05, p_max=0.35):
    """Schedule modality dropout rate from p_min to p_max over training"""
    alpha = min(1.0, epoch / max(1, int(0.6 * total_epochs)))
    return p_min + (p_max - p_min) * alpha


def compute_loss(outputs, batch, lambdas):
    """
    Compute multi-task loss with classification, regression, and consistency.
    
    Args:
        outputs: Model output dictionary
        batch: Batch dictionary with targets
        lambdas: Loss weight dictionary with keys: reg, cons
    
    Returns:
        total_loss: Weighted sum of all losses
        parts: Dictionary of individual loss components
    """
    # Main task losses
    cls_loss = F.cross_entropy(outputs["logits"], batch["classification_targets"])
    
    reg_loss = (
        F.l1_loss(outputs["reg"].squeeze(-1), batch["regression_targets"])
        if "regression_targets" in batch
        else torch.tensor(0.0, device=cls_loss.device)
    )
    
    # Consistency loss (only if at least two modalities present)
    logits_list = []
    if outputs["logits_text"].numel() > 0:
        logits_list.append(outputs["logits_text"])
    if outputs["logits_audio"].numel() > 0:
        logits_list.append(outputs["logits_audio"])
    if outputs["logits_visual"].numel() > 0:
        logits_list.append(outputs["logits_visual"])
    
    cons_loss = symmetric_kl_multi(logits_list, temperature=2.0)
    
    # Total weighted loss
    total = cls_loss + lambdas["reg"] * reg_loss + lambdas["cons"] * cons_loss
    
    return total, {
        "classification_loss": cls_loss,
        "regression_loss": reg_loss,
        "consistency_loss": cons_loss,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs, lambdas, grad_clip=1.0):
    """Train for one epoch with mixed precision and gradient clipping."""
    model.train()
    model.moddrop.p = schedule_dropout(epoch, total_epochs)
    
    logs = dict(cls=0.0, reg=0.0, cons=0.0, tot=0.0, n=0)
    
    for batch in loader:
        # Move batch to device
        for k in batch:
            batch[k] = batch[k].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(batch)
            loss, parts = compute_loss(outputs, batch, lambdas)
        
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
    
    # Average losses
    for k in list(logs.keys()):
        if k != "n":
            logs[k] = logs[k] / max(logs["n"], 1)
    
    return logs


def validate(model, loader, device, lambdas):
    """Validate model on validation set."""
    model.eval()
    logs = dict(cls=0.0, reg=0.0, cons=0.0, tot=0.0, n=0)
    
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            
            outputs = model(batch)
            loss, parts = compute_loss(outputs, batch, lambdas)
            
            logs["tot"] += loss.item()
            logs["cls"] += parts["classification_loss"].item()
            logs["reg"] += parts["regression_loss"].item()
            logs["cons"] += parts["consistency_loss"].item()
            logs["n"] += 1
    
    # Average losses
    for k in list(logs.keys()):
        if k != "n":
            logs[k] = logs[k] / max(logs["n"], 1)
    
    return logs


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MAFT model')
    parser.add_argument('--config', type=str, default='configs/cpu_test_config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    else:
        print(f"\n📦 Loading {config['dataset']['name']} dataset")
        print("⚠️  Real dataset loading not yet implemented - use --use_synthetic")
        return
    
    print(f"Train: {len(train_loader)} batches")
    print(f"Val: {len(val_loader)} batches")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scaler for mixed precision
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Loss weights
    lambdas = {
        'reg': config['training']['regression_weight'],
        'cons': config['training']['consistency_weight']
    }
    
    # Training loop
    print(f"\n🏃 Training for {config['training']['num_epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_logs = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, config['training']['num_epochs'], lambdas
        )
        
        # Validate
        val_logs = validate(model, val_loader, device, lambdas)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"  Train Loss: {train_logs['tot']:.4f} (cls={train_logs['cls']:.4f}, reg={train_logs['reg']:.4f}, cons={train_logs['cons']:.4f})")
        print(f"  Val Loss:   {val_logs['tot']:.4f} (cls={val_logs['cls']:.4f}, reg={val_logs['reg']:.4f}, cons={val_logs['cons']:.4f})")
        
        # Save best model
        if val_logs['tot'] < best_val_loss:
            best_val_loss = val_logs['tot']
            print(f"  ✅ New best model! (val_loss={best_val_loss:.4f})")
            
            # Save checkpoint
            output_dir = Path(config['training']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, output_dir / 'best_model.pth')
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("="*70)


if __name__ == '__main__':
    main()