#!/usr/bin/env python3
"""
MAFT Training Script

This script implements the training loop for the Multimodal Attention Fusion Transformer
with multi-task learning, mixed precision training, and comprehensive logging.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import json
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.maft import MAFT, MAFTLoss
from utils.data_utils import MOSEIDataset, InterviewDataset, create_dataloader
from utils.metrics import MultimodalMetrics, plot_training_curves
from transformers import BertTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: torch.device) -> MAFT:
    """Create and initialize the MAFT model."""
    model = MAFT(
        text_model_name=config['model']['text_model_name'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        audio_input_dim=config['model']['audio_input_dim'],
        visual_input_dim=config['model']['visual_input_dim'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        modality_dropout_rate=config['model']['modality_dropout_rate'],
        freeze_bert=config['model']['freeze_bert']
    )
    
    model = model.to(device)
    return model


def create_optimizer(model: MAFT, config: dict) -> optim.Optimizer:
    """Create optimizer with different learning rates for BERT and other parameters."""
    bert_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'text_encoder.bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if bert_params:
        param_groups.append({'params': bert_params, 'lr': float(config['training']['bert_lr'])})
    if other_params:
        param_groups.append({'params': other_params, 'lr': float(config['training']['lr'])})

    if not param_groups:
        raise ValueError("No parameters found for optimizer!")

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=float(config['training']['weight_decay'])
    )

    return optimizer


def create_scheduler(optimizer: optim.Optimizer, num_training_steps: int, config: dict):
    """Create learning rate scheduler with warmup."""
    warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler


class ModalityDropoutLogger:
    """Log modality dropout statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.modality_drops = {'text': 0, 'audio': 0, 'visual': 0}
        self.total_batches = 0
    
    def log_drop(self, modality: str):
        self.modality_drops[modality] += 1
        self.total_batches += 1
    
    def get_stats(self) -> dict:
        if self.total_batches == 0:
            return {mod: 0.0 for mod in self.modality_drops}
        
        return {
            mod: count / self.total_batches 
            for mod, count in self.modality_drops.items()
        }


def train_epoch(model: MAFT, train_loader, optimizer: optim.Optimizer, scheduler,
                criterion: MAFTLoss, scaler: GradScaler, device: torch.device,
                epoch: int, config: dict, modality_logger: ModalityDropoutLogger,
                tb_writer, global_step: int) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    metrics = MultimodalMetrics()
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    print(f"Starting epoch {epoch+1} with {len(train_loader)} batches")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Only pass the model the inputs it expects (not the targets)
        model_inputs = {k: v for k, v in batch.items() if k not in ['classification_targets', 'regression_targets']}
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(**model_inputs)
            loss_dict = criterion(
                outputs['classification_logits'],
                outputs['regression_output'],
                batch['classification_targets'],
                batch['regression_targets']
            )
            loss = loss_dict['total_loss']
        
        # Log modality dropout if it occurred
        if hasattr(model, 'modality_dropout') and model.modality_dropout.training:
            # Check if any modality was dropped (this would need to be implemented in the model)
            pass
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        total_cls_loss += loss_dict['classification_loss'].item()
        total_reg_loss += loss_dict['regression_loss'].item()
        
        metrics.update(
            outputs['classification_logits'],
            outputs['regression_output'],
            batch['classification_targets'],
            batch['regression_targets']
        )
        
        # Log to tensorboard
        if global_step % config['training'].get('log_every', 10) == 0:
            tb_writer.add_scalar('Loss/Train_Total', loss.item(), global_step)
            tb_writer.add_scalar('Loss/Train_Classification', loss_dict['classification_loss'].item(), global_step)
            tb_writer.add_scalar('Loss/Train_Regression', loss_dict['regression_loss'].item(), global_step)
            tb_writer.add_scalar('LR/Learning_Rate', scheduler.get_last_lr()[0], global_step)
        
        global_step += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'LR': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    # Compute epoch metrics
    epoch_metrics = metrics.compute_all_metrics()
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)
    
    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'metrics': epoch_metrics,
        'global_step': global_step
    }


def validate_epoch(model: MAFT, val_loader, criterion: MAFTLoss, device: torch.device) -> dict:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    metrics = MultimodalMetrics()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Only pass the model the inputs it expects (not the targets)
            model_inputs = {k: v for k, v in batch.items() if k not in ['classification_targets', 'regression_targets']}
            
            # Forward pass
            outputs = model(**model_inputs)
            loss_dict = criterion(
                outputs['classification_logits'],
                outputs['regression_output'],
                batch['classification_targets'],
                batch['regression_targets']
            )
            loss = loss_dict['total_loss']
            
            # Update metrics
            total_loss += loss.item()
            total_cls_loss += loss_dict['classification_loss'].item()
            total_reg_loss += loss_dict['regression_loss'].item()
            
            metrics.update(
                outputs['classification_logits'],
                outputs['regression_output'],
                batch['classification_targets'],
                batch['regression_targets']
            )
    
    # Compute validation metrics
    val_metrics = metrics.compute_all_metrics()
    avg_loss = total_loss / len(val_loader)
    avg_cls_loss = total_cls_loss / len(val_loader)
    avg_reg_loss = total_reg_loss / len(val_loader)
    
    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'metrics': val_metrics
    }


def save_checkpoint(model: MAFT, optimizer: optim.Optimizer, scheduler, epoch: int,
                   train_metrics: dict, val_metrics: dict, config: dict, save_path: str,
                   seed: int, modality_stats: dict = None):
    """Save model checkpoint with enhanced metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': config,
        'seed': seed,
        'modality_stats': modality_stats,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved to {save_path}")


def extract_attention_maps(model: MAFT, batch: dict, device: torch.device) -> dict:
    """Extract attention maps for visualization."""
    model.eval()
    with torch.no_grad():
        # This would need to be implemented in the model to return attention weights
        # For now, return placeholder
        batch_size = batch['input_ids'].size(0)
        seq_len = batch['input_ids'].size(1) + batch['audio_features'].size(1) + batch['visual_features'].size(1)
        
        attention_maps = {
            'attention_weights': torch.ones(batch_size, 12, seq_len, seq_len),  # 12 heads
            'text_tokens': ['[CLS]'] + ['token'] * (batch['input_ids'].size(1) - 1),
            'audio_tokens': ['audio'] * batch['audio_features'].size(1),
            'visual_tokens': ['visual'] * batch['visual_features'].size(1)
        }
    
    return attention_maps


def main():
    parser = argparse.ArgumentParser(description='Train MAFT model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--save_attention', action='store_true', help='Save attention maps')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Debug prints for config values
    print("DEBUG: bert_lr =", config['training']['bert_lr'], type(config['training']['bert_lr']))
    print("DEBUG: lr =", config['training']['lr'], type(config['training']['lr']))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Initialize logging
    if args.wandb:
        wandb.init(
            project="maft",
            config=config,
            name=f"maft_{config['dataset']['name']}_{args.seed}",
            tags=[f"seed_{args.seed}", config['dataset']['name']]
        )
    
    # Create output directory
    output_dir = Path(config['training']['output_dir']) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration and seed
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(output_dir / 'seed.txt', 'w') as f:
        f.write(str(args.seed))
    
    # Setup tensorboard
    tb_writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # Create datasets
    if config['dataset']['name'] == 'mosei':
        train_dataset = MOSEIDataset(
            config['dataset']['data_path'],
            tokenizer,
            split='train',
            max_length=config['dataset']['max_length'],
            audio_max_length=config['dataset']['audio_max_length'],
            visual_max_length=config['dataset']['visual_max_length']
        )
        val_dataset = MOSEIDataset(
            config['dataset']['data_path'],
            tokenizer,
            split='val',
            max_length=config['dataset']['max_length'],
            audio_max_length=config['dataset']['audio_max_length'],
            visual_max_length=config['dataset']['visual_max_length']
        )
    elif config['dataset']['name'] == 'interview':
        train_dataset = InterviewDataset(
            config['dataset']['data_path'],
            tokenizer,
            split='train',
            max_length=config['dataset']['max_length'],
            audio_max_length=config['dataset']['audio_max_length'],
            visual_max_length=config['dataset']['visual_max_length']
        )
        val_dataset = InterviewDataset(
            config['dataset']['data_path'],
            tokenizer,
            split='val',
            max_length=config['dataset']['max_length'],
            audio_max_length=config['dataset']['audio_max_length'],
            visual_max_length=config['dataset']['visual_max_length']
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']['name']}")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    model = create_model(config, device)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Debug prints for config values
    print("bert_lr:", config['training']['bert_lr'], type(config['training']['bert_lr']))
    print("lr:", config['training']['lr'], type(config['training']['lr']))
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    
    # Debug prints for optimizer parameter groups
    print("Optimizer parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}: lr={group['lr']}, num_params={len(group['params'])}, type(lr)={type(group['lr'])}")
    
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = create_scheduler(optimizer, num_training_steps, config)
    
    # Create loss function
    criterion = MAFTLoss(
        classification_weight=config['training']['classification_weight'],
        regression_weight=config['training']['regression_weight']
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Create modality dropout logger
    modality_logger = ModalityDropoutLogger()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    global_step = 0
    
    print(f"\n🎯 Starting training for {config['training']['num_epochs']} epochs...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎲 Seed: {args.seed}")
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, 
            device, epoch, config, modality_logger, tb_writer, global_step
        )
        global_step = train_results['global_step']
        
        # Validate
        val_results = validate_epoch(model, val_loader, criterion, device)
        
        # Log metrics
        train_losses.append(train_results['loss'])
        val_losses.append(val_results['loss'])
        
        if train_results['metrics']['classification']:
            train_accuracies.append(train_results['metrics']['classification']['accuracy'])
            val_accuracies.append(val_results['metrics']['classification']['accuracy'])
        
        # Log to tensorboard
        tb_writer.add_scalar('Loss/Train', train_results['loss'], epoch)
        tb_writer.add_scalar('Loss/Val', val_results['loss'], epoch)
        tb_writer.add_scalar('Loss/Train_Classification', train_results['cls_loss'], epoch)
        tb_writer.add_scalar('Loss/Train_Regression', train_results['reg_loss'], epoch)
        tb_writer.add_scalar('Loss/Val_Classification', val_results['cls_loss'], epoch)
        tb_writer.add_scalar('Loss/Val_Regression', val_results['reg_loss'], epoch)
        
        if train_results['metrics']['classification']:
            tb_writer.add_scalar('Accuracy/Train', train_results['metrics']['classification']['accuracy'], epoch)
            tb_writer.add_scalar('Accuracy/Val', val_results['metrics']['classification']['accuracy'], epoch)
            tb_writer.add_scalar('F1/Train', train_results['metrics']['classification']['f1_score'], epoch)
            tb_writer.add_scalar('F1/Val', val_results['metrics']['classification']['f1_score'], epoch)
        
        if train_results['metrics']['regression']:
            tb_writer.add_scalar('MSE/Train', train_results['metrics']['regression']['mse'], epoch)
            tb_writer.add_scalar('MSE/Val', val_results['metrics']['regression']['mse'], epoch)
            tb_writer.add_scalar('Correlation/Train', train_results['metrics']['regression']['correlation'], epoch)
            tb_writer.add_scalar('Correlation/Val', val_results['metrics']['regression']['correlation'], epoch)
        
        # Log modality dropout stats
        modality_stats = modality_logger.get_stats()
        for modality, rate in modality_stats.items():
            tb_writer.add_scalar(f'ModalityDropout/{modality}', rate, epoch)
        
        # Print epoch results
        print(f"\n📈 Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"  Train Loss: {train_results['loss']:.4f}")
        print(f"  Val Loss: {val_results['loss']:.4f}")
        
        if train_results['metrics']['classification']:
            print(f"  Train Acc: {train_results['metrics']['classification']['accuracy']:.4f}")
            print(f"  Val Acc: {val_results['metrics']['classification']['accuracy']:.4f}")
        
        # Log to wandb
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_results['loss'],
                'val_loss': val_results['loss'],
                'train_accuracy': train_results['metrics']['classification']['accuracy'] if train_results['metrics']['classification'] else 0,
                'val_accuracy': val_results['metrics']['classification']['accuracy'] if val_results['metrics']['classification'] else 0,
                'learning_rate': scheduler.get_last_lr()[0],
                **{f'modality_dropout_{mod}': rate for mod, rate in modality_stats.items()}
            })
        
        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_results, val_results, config,
                output_dir / 'best_model.pth',
                args.seed, modality_stats
            )
        
        # Save checkpoint every few epochs
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_results, val_results, config,
                output_dir / f'checkpoint_epoch_{epoch+1}.pth',
                args.seed, modality_stats
            )
        
        # Save attention maps if requested
        if args.save_attention and epoch % 5 == 0:  # Save every 5 epochs
            try:
                # Get a sample batch
                sample_batch = next(iter(val_loader))
                sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}
                
                attention_maps = extract_attention_maps(model, sample_batch, device)
                
                # Save attention maps
                attention_file = output_dir / f'attention_maps_epoch_{epoch+1}.pt'
                torch.save(attention_maps, attention_file)
                print(f"  📊 Attention maps saved to {attention_file}")
                
            except Exception as e:
                print(f"  ⚠️  Could not save attention maps: {e}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, config['training']['num_epochs'] - 1,
        train_results, val_results, config,
        output_dir / 'final_model.pth',
        args.seed, modality_stats
    )
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        metric_name="Accuracy",
        save_path=output_dir / 'training_curves.png'
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'final_modality_stats': modality_stats
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print final results
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final modality dropout rates: {modality_stats}")
    
    # Close tensorboard writer
    tb_writer.close()
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 