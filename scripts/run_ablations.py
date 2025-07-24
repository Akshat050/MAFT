#!/usr/bin/env python3
"""
MAFT Ablation Study Script

This script runs comprehensive ablation studies to evaluate different model configurations
and prove the effectiveness of the unified fusion approach.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.maft import MAFT, MAFTLoss
from utils.data_utils import MOSEIDataset, InterviewDataset, create_dataloader
from utils.metrics import MultimodalMetrics
from utils.visualization import plot_training_curves
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch.optim as optim


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_ablation_configs(base_config: dict) -> list:
    """Create different ablation configurations."""
    ablation_configs = []
    
    # 1. Modality ablation: test each modality individually
    modalities = ['text', 'audio', 'visual']
    for modality in modalities:
        config = base_config.copy()
        config['ablation'] = {'type': 'modality', 'dropped_modality': modality}
        config['experiment_name'] = f"without_{modality}"
        ablation_configs.append(config)
    
    # 2. Fusion ablation: test different fusion strategies
    fusion_configs = [
        {'type': 'early_fusion', 'fusion_layer': 0},
        {'type': 'mid_fusion', 'fusion_layer': 6},
        {'type': 'late_fusion', 'fusion_layer': 12}
    ]
    
    for fusion_config in fusion_configs:
        config = base_config.copy()
        config['ablation'] = fusion_config
        config['experiment_name'] = f"{fusion_config['type']}"
        ablation_configs.append(config)
    
    # 3. Architecture ablation: test different model sizes
    architecture_configs = [
        {'hidden_dim': 384, 'num_heads': 6, 'num_layers': 1},
        {'hidden_dim': 768, 'num_heads': 12, 'num_layers': 1},  # Base
        {'hidden_dim': 768, 'num_heads': 12, 'num_layers': 2},
        {'hidden_dim': 1024, 'num_heads': 16, 'num_layers': 1}
    ]
    
    for arch_config in architecture_configs:
        config = base_config.copy()
        config['model'].update(arch_config)
        config['ablation'] = {'type': 'architecture', 'config': arch_config}
        config['experiment_name'] = f"arch_{arch_config['hidden_dim']}_{arch_config['num_layers']}"
        ablation_configs.append(config)
    
    # 4. Training ablation: test different loss weights
    loss_configs = [
        {'classification_weight': 0.3, 'regression_weight': 0.7},
        {'classification_weight': 0.5, 'regression_weight': 0.5},  # Base
        {'classification_weight': 0.7, 'regression_weight': 0.3}
    ]
    
    for loss_config in loss_configs:
        config = base_config.copy()
        config['training'].update(loss_config)
        config['ablation'] = {'type': 'loss_weights', 'config': loss_config}
        config['experiment_name'] = f"loss_{loss_config['classification_weight']}_{loss_config['regression_weight']}"
        ablation_configs.append(config)
    
    return ablation_configs


def train_ablation_model(config: dict, device: torch.device, seed: int) -> dict:
    """Train a single ablation model."""
    set_seed(seed)
    
    # Create model
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
    ).to(device)
    
    # Create datasets
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model_name'])
    
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
    else:
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
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * config['training']['warmup_ratio']),
        num_training_steps=num_training_steps
    )
    
    # Create loss function
    criterion = MAFTLoss(
        classification_weight=config['training']['classification_weight'],
        regression_weight=config['training']['regression_weight']
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss_dict = criterion(
                outputs['classification_logits'],
                outputs['regression_output'],
                batch['classification_targets'],
                batch['regression_targets']
            )
            loss = loss_dict['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_metrics = MultimodalMetrics()
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(**batch)
                loss_dict = criterion(
                    outputs['classification_logits'],
                    outputs['regression_output'],
                    batch['classification_targets'],
                    batch['regression_targets']
                )
                loss = loss_dict['total_loss']
                
                val_loss += loss.item()
                val_metrics.update(
                    outputs['classification_logits'],
                    outputs['regression_output'],
                    batch['classification_targets'],
                    batch['regression_targets']
                )
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.compute_all_metrics()
    
    return {
        'best_val_loss': best_val_loss,
        'best_metrics': best_metrics,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def run_ablation_study(base_config_path: str, output_dir: str, num_seeds: int = 5):
    """Run comprehensive ablation study."""
    print("🔬 Starting comprehensive ablation study...")
    
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create ablation configurations
    ablation_configs = create_ablation_configs(base_config)
    
    print(f"📋 Running {len(ablation_configs)} ablation experiments...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = {}
    
    # Run each ablation configuration
    for config in tqdm(ablation_configs, desc="Ablation experiments"):
        experiment_name = config['experiment_name']
        print(f"\n🧪 Running experiment: {experiment_name}")
        
        experiment_results = []
        
        # Run with multiple seeds
        for seed in range(num_seeds):
            print(f"  Seed {seed + 1}/{num_seeds}")
            
            try:
                result = train_ablation_model(config, device, seed)
                experiment_results.append(result)
            except Exception as e:
                print(f"  ❌ Error in seed {seed}: {e}")
                continue
        
        if experiment_results:
            # Aggregate results across seeds
            val_losses = [r['best_val_loss'] for r in experiment_results]
            accuracies = [r['best_metrics']['classification']['accuracy'] 
                         for r in experiment_results if r['best_metrics']['classification']]
            
            aggregated_result = {
                'config': config,
                'val_loss_mean': np.mean(val_losses),
                'val_loss_std': np.std(val_losses),
                'accuracy_mean': np.mean(accuracies) if accuracies else 0,
                'accuracy_std': np.std(accuracies) if accuracies else 0,
                'individual_results': experiment_results
            }
            
            all_results[experiment_name] = aggregated_result
    
    # Save results
    with open(output_path / 'ablation_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for exp_name, result in all_results.items():
            json_results[exp_name] = {
                'val_loss_mean': result['val_loss_mean'],
                'val_loss_std': result['val_loss_std'],
                'accuracy_mean': result['accuracy_mean'],
                'accuracy_std': result['accuracy_std'],
                'ablation_type': result['config']['ablation']['type']
            }
        
        json.dump(json_results, f, indent=2)
    
    # Create summary table
    summary_data = []
    for exp_name, result in all_results.items():
        summary_data.append({
            'Experiment': exp_name,
            'Ablation Type': result['config']['ablation']['type'],
            'Val Loss (mean±std)': f"{result['val_loss_mean']:.4f}±{result['val_loss_std']:.4f}",
            'Accuracy (mean±std)': f"{result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / 'ablation_summary.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print(f"\n✅ Ablation study completed! Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run MAFT ablation studies')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base configuration file')
    parser.add_argument('--output_dir', type=str, default='experiments/ablations',
                       help='Output directory for results')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of seeds to run per experiment')
    args = parser.parse_args()
    
    run_ablation_study(args.config, args.output_dir, args.num_seeds)


if __name__ == '__main__':
    main() 