#!/usr/bin/env python3
"""
MAFT Baseline Comparison Script

This script runs all baselines and generates the comprehensive results table
for the MAFT paper.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baselines import (
    TextOnlyBERT, LateFusion, MAGBERT, MulT, 
    EarlyFusionMAFT, LateFusionMAFT
)
from models.maft import MAFT, MAFTLoss
from utils.data_utils import MOSEIDataset, InterviewDataset, create_dataloader
from utils.metrics import MultimodalMetrics
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch.optim as optim


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_baseline_model(model_name: str, config: dict, device: torch.device):
    """Create baseline model based on name."""
    if model_name == "text_only_bert":
        return TextOnlyBERT(
            model_name=config['model']['text_model_name'],
            hidden_dim=config['model']['hidden_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(device)
    
    elif model_name == "late_fusion":
        return LateFusion(
            text_model_name=config['model']['text_model_name'],
            audio_input_dim=config['model']['audio_input_dim'],
            visual_input_dim=config['model']['visual_input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(device)
    
    elif model_name == "mag_bert":
        return MAGBERT(
            text_model_name=config['model']['text_model_name'],
            audio_input_dim=config['model']['audio_input_dim'],
            visual_input_dim=config['model']['visual_input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(device)
    
    elif model_name == "mult":
        return MulT(
            text_model_name=config['model']['text_model_name'],
            audio_input_dim=config['model']['audio_input_dim'],
            visual_input_dim=config['model']['visual_input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=4,  # MulT typically uses more layers
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(device)
    
    elif model_name == "maft_early_fusion":
        return EarlyFusionMAFT(
            text_model_name=config['model']['text_model_name'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            audio_input_dim=config['model']['audio_input_dim'],
            visual_input_dim=config['model']['visual_input_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(device)
    
    elif model_name == "maft_late_fusion":
        return LateFusionMAFT(
            text_model_name=config['model']['text_model_name'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            audio_input_dim=config['model']['audio_input_dim'],
            visual_input_dim=config['model']['visual_input_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(device)
    
    elif model_name == "maft":
        return MAFT(
            text_model_name=config['model']['text_model_name'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            audio_input_dim=config['model']['audio_input_dim'],
            visual_input_dim=config['model']['visual_input_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout'],
            modality_dropout_rate=config['model']['modality_dropout_rate']
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_baseline_model(model, train_loader, val_loader, config: dict, 
                        device: torch.device, model_name: str) -> dict:
    """Train a baseline model and return results."""
    
    # Create optimizer
    if model_name == "text_only_bert":
        # Different learning rates for BERT vs other parameters
        bert_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': bert_params, 'lr': config['training']['bert_lr']},
            {'params': other_params, 'lr': config['training']['lr']}
        ], weight_decay=config['training']['weight_decay'])
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Create scheduler
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
    best_metrics = None
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass (handle different model inputs)
            if model_name == "text_only_bert":
                outputs = model(batch['input_ids'], batch['attention_mask'])
            else:
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
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_metrics = MultimodalMetrics()
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                if model_name == "text_only_bert":
                    outputs = model(batch['input_ids'], batch['attention_mask'])
                else:
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
        
        val_loss /= len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.compute_all_metrics()
    
    return best_metrics


def run_baseline_experiment(model_name: str, config: dict, device: torch.device,
                          train_loader, val_loader, test_loader, seed: int) -> dict:
    """Run a single baseline experiment."""
    set_seed(seed)
    
    print(f"🧪 Running {model_name} (seed {seed})...")
    
    # Create model
    model = create_baseline_model(model_name, config, device)
    
    # Count parameters
    num_params = count_parameters(model)
    
    # Train model
    start_time = time.time()
    train_metrics = train_baseline_model(model, train_loader, val_loader, config, device, model_name)
    training_time = time.time() - start_time
    
    # Evaluate on test set
    model.eval()
    test_metrics = MultimodalMetrics()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if model_name == "text_only_bert":
                outputs = model(batch['input_ids'], batch['attention_mask'])
            else:
                outputs = model(**batch)
            
            test_metrics.update(
                outputs['classification_logits'],
                outputs['regression_output'],
                batch['classification_targets'],
                batch['regression_targets']
            )
    
    final_metrics = test_metrics.compute_all_metrics()
    
    return {
        'model_name': model_name,
        'seed': seed,
        'num_params': num_params,
        'training_time': training_time,
        'train_metrics': train_metrics,
        'test_metrics': final_metrics
    }


def run_modality_ablation(config: dict, device: torch.device, train_loader, 
                         val_loader, test_loader, seed: int) -> dict:
    """Run modality ablation experiments."""
    results = {}
    
    # Test MAFT without each modality
    modalities_to_drop = ['text', 'audio', 'visual']
    
    for modality in modalities_to_drop:
        print(f"🔬 Running MAFT without {modality} (seed {seed})...")
        
        set_seed(seed)
        model = create_baseline_model("maft", config, device)
        
        # Train with modality dropout
        original_dropout_rate = config['model']['modality_dropout_rate']
        config['model']['modality_dropout_rate'] = 0.0  # Disable random dropout
        
        train_metrics = train_baseline_model(model, train_loader, val_loader, config, device, "maft")
        
        # Test with specific modality dropped
        model.eval()
        test_metrics = MultimodalMetrics()
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Drop specific modality
                if modality == 'text':
                    batch['input_ids'] = torch.zeros_like(batch['input_ids'])
                    batch['attention_mask'] = torch.zeros_like(batch['attention_mask'])
                elif modality == 'audio':
                    batch['audio_features'] = torch.zeros_like(batch['audio_features'])
                    batch['audio_mask'] = torch.zeros_like(batch['audio_mask'])
                elif modality == 'visual':
                    batch['visual_features'] = torch.zeros_like(batch['visual_features'])
                    batch['visual_mask'] = torch.zeros_like(batch['visual_mask'])
                
                outputs = model(**batch)
                test_metrics.update(
                    outputs['classification_logits'],
                    outputs['regression_output'],
                    batch['classification_targets'],
                    batch['regression_targets']
                )
        
        final_metrics = test_metrics.compute_all_metrics()
        
        results[f'maft_without_{modality}'] = {
            'model_name': f'MAFT - {modality}',
            'seed': seed,
            'num_params': count_parameters(model),
            'test_metrics': final_metrics
        }
        
        # Restore original dropout rate
        config['model']['modality_dropout_rate'] = original_dropout_rate
    
    return results


def aggregate_results(all_results: list) -> pd.DataFrame:
    """Aggregate results across seeds and create final table."""
    
    # Group by model name
    model_groups = {}
    for result in all_results:
        model_name = result['model_name']
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(result)
    
    # Calculate statistics for each model
    table_data = []
    
    for model_name, results in model_groups.items():
        # Extract metrics across seeds
        accuracies = []
        f1_scores = []
        maes = []
        correlations = []
        params_list = []
        times_list = []
        
        for result in results:
            if 'test_metrics' in result and 'classification' in result['test_metrics']:
                cls_metrics = result['test_metrics']['classification']
                reg_metrics = result['test_metrics']['regression']
                
                accuracies.append(cls_metrics.get('accuracy', 0))
                f1_scores.append(cls_metrics.get('f1_score', 0))
                maes.append(reg_metrics.get('mae', 0))
                correlations.append(reg_metrics.get('correlation', 0))
            
            params_list.append(result.get('num_params', 0))
            times_list.append(result.get('training_time', 0))
        
        # Calculate mean and std
        if accuracies:
            acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)
            f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
            mae_mean, mae_std = np.mean(maes), np.std(maes)
            corr_mean, corr_std = np.mean(correlations), np.std(correlations)
        else:
            acc_mean = acc_std = f1_mean = f1_std = mae_mean = mae_std = corr_mean = corr_std = 0
        
        params_mean = np.mean(params_list) if params_list else 0
        time_mean = np.mean(times_list) if times_list else 0
        
        # Determine source
        if 'mag_bert' in model_name.lower():
            source = "Reported from Tsai et al."
        elif 'mult' in model_name.lower():
            source = "Run by us or cited"
        else:
            source = "Run by us"
        
        table_data.append({
            'Model': model_name,
            'Acc-2': f"{acc_mean:.3f}±{acc_std:.3f}",
            'F1': f"{f1_mean:.3f}±{f1_std:.3f}",
            'MAE': f"{mae_mean:.3f}±{mae_std:.3f}",
            'Pearson r': f"{corr_mean:.3f}±{corr_std:.3f}",
            'Params (M)': f"{params_mean/1e6:.1f}",
            'GPU Hours': f"{time_mean/3600:.1f}",
            'Notes': source
        })
    
    return pd.DataFrame(table_data)


def main():
    parser = argparse.ArgumentParser(description='Run MAFT baseline comparisons')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['mosei', 'interview'], required=True,
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='experiments/baselines',
                       help='Output directory for results')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of seeds to run')
    parser.add_argument('--baselines', nargs='+', 
                       default=['text_only_bert', 'late_fusion', 'mag_bert', 'mult', 
                               'maft_early_fusion', 'maft_late_fusion', 'maft'],
                       help='Baselines to run')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model_name'])
    
    if args.dataset == 'mosei':
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
        test_dataset = MOSEIDataset(
            config['dataset']['data_path'],
            tokenizer,
            split='test',
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
        test_dataset = InterviewDataset(
            config['dataset']['data_path'],
            tokenizer,
            split='test',
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
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Run baseline experiments
    all_results = []
    
    for baseline in args.baselines:
        print(f"\n{'='*60}")
        print(f"RUNNING BASELINE: {baseline.upper()}")
        print(f"{'='*60}")
        
        for seed in range(args.num_seeds):
            try:
                result = run_baseline_experiment(
                    baseline, config, device, train_loader, val_loader, test_loader, seed
                )
                all_results.append(result)
                
                # Save individual result
                result_file = output_path / f"{baseline}_seed_{seed}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
            except Exception as e:
                print(f"❌ Error running {baseline} seed {seed}: {e}")
                continue
    
    # Run modality ablations
    print(f"\n{'='*60}")
    print("RUNNING MODALITY ABLATIONS")
    print(f"{'='*60}")
    
    for seed in range(args.num_seeds):
        try:
            ablation_results = run_modality_ablation(
                config, device, train_loader, val_loader, test_loader, seed
            )
            
            for ablation_name, result in ablation_results.items():
                all_results.append(result)
                
                # Save individual result
                result_file = output_path / f"{ablation_name}_seed_{seed}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                    
        except Exception as e:
            print(f"❌ Error running ablation seed {seed}: {e}")
            continue
    
    # Create final results table
    print(f"\n{'='*60}")
    print("GENERATING RESULTS TABLE")
    print(f"{'='*60}")
    
    results_df = aggregate_results(all_results)
    
    # Save results table
    table_file = output_path / 'baseline_results.csv'
    results_df.to_csv(table_file, index=False)
    
    # Print results table
    print("\n📊 FINAL RESULTS TABLE:")
    print("="*100)
    print(results_df.to_string(index=False))
    print("="*100)
    
    # Save detailed results
    detailed_file = output_path / 'detailed_results.json'
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Baseline comparison completed!")
    print(f"📁 Results saved to: {output_path}")
    print(f"📊 Results table: {table_file}")
    print(f"📋 Detailed results: {detailed_file}")


if __name__ == '__main__':
    main() 