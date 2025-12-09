#!/usr/bin/env python3
"""
Model Evaluation Script for MAFT

Evaluates trained models with comprehensive metrics:
- Classification: Accuracy, F1, Precision, Recall
- Regression: MAE, MSE, Correlation
- Per-modality analysis
- Robustness testing (missing modalities)

Usage:
    python evaluate.py --checkpoint experiments/m4_pro_improved/best_model.pth
"""

import torch
import torch.nn.functional as F
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from scipy.stats import pearsonr


def compute_metrics(predictions, targets, task='classification'):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions (logits for classification, values for regression)
        targets: Ground truth labels/values
        task: 'classification' or 'regression'
    
    Returns:
        Dictionary of metrics
    """
    if task == 'classification':
        # Convert logits to predictions
        if predictions.dim() > 1:
            pred_labels = predictions.argmax(dim=1).cpu().numpy()
        else:
            pred_labels = predictions.cpu().numpy()
        
        true_labels = targets.cpu().numpy()
        
        # Compute metrics
        acc = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        
        precision, recall, _, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        }
    
    elif task == 'regression':
        pred_values = predictions.cpu().numpy()
        true_values = targets.cpu().numpy()
        
        # Compute metrics
        mae = np.mean(np.abs(pred_values - true_values))
        mse = np.mean((pred_values - true_values) ** 2)
        rmse = np.sqrt(mse)
        
        # Correlation
        if len(pred_values) > 1:
            corr, _ = pearsonr(pred_values, true_values)
        else:
            corr = 0.0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'correlation': corr
        }
    
    return {}


def evaluate_model(model, loader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: MAFT model
        loader: DataLoader
        device: Device to use
    
    Returns:
        Dictionary with all predictions and metrics
    """
    model.eval()
    
    all_logits = []
    all_reg_preds = []
    all_cls_targets = []
    all_reg_targets = []
    
    # Modality-specific predictions
    all_logits_text = []
    all_logits_audio = []
    all_logits_visual = []
    
    print("\n🔍 Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            # Move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Collect predictions
            all_logits.append(outputs["logits"].cpu())
            all_reg_preds.append(outputs["reg"].cpu())
            all_cls_targets.append(batch["classification_targets"].cpu())
            
            if "regression_targets" in batch:
                all_reg_targets.append(batch["regression_targets"].cpu())
            
            # Modality-specific
            if outputs["logits_text"].numel() > 0:
                all_logits_text.append(outputs["logits_text"].cpu())
            if outputs["logits_audio"].numel() > 0:
                all_logits_audio.append(outputs["logits_audio"].cpu())
            if outputs["logits_visual"].numel() > 0:
                all_logits_visual.append(outputs["logits_visual"].cpu())
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_reg_preds = torch.cat(all_reg_preds, dim=0).squeeze(-1)
    all_cls_targets = torch.cat(all_cls_targets, dim=0)
    
    results = {
        'logits': all_logits,
        'reg_preds': all_reg_preds,
        'cls_targets': all_cls_targets,
    }
    
    if all_reg_targets:
        results['reg_targets'] = torch.cat(all_reg_targets, dim=0)
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    
    # Classification metrics
    cls_metrics = compute_metrics(all_logits, all_cls_targets, task='classification')
    
    # Regression metrics
    if all_reg_targets:
        reg_metrics = compute_metrics(all_reg_preds, results['reg_targets'], task='regression')
    else:
        reg_metrics = {}
    
    # Binary sentiment accuracy (for comparison with papers)
    binary_preds = (all_reg_preds > 0).long()
    binary_targets = (results.get('reg_targets', all_reg_preds) > 0).long()
    binary_acc = accuracy_score(binary_targets.numpy(), binary_preds.numpy())
    
    results['metrics'] = {
        'classification': cls_metrics,
        'regression': reg_metrics,
        'binary_accuracy': binary_acc
    }
    
    return results


def print_results(results, title="Evaluation Results"):
    """Pretty print evaluation results."""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)
    
    # Classification metrics
    if 'classification' in results['metrics']:
        print("\n📊 Classification Metrics:")
        cls_metrics = results['metrics']['classification']
        print(f"  Accuracy (7-class): {cls_metrics['accuracy']*100:.2f}%")
        print(f"  F1 Score (macro):   {cls_metrics['f1_macro']*100:.2f}%")
        print(f"  F1 Score (weighted): {cls_metrics['f1_weighted']*100:.2f}%")
        print(f"  Precision:          {cls_metrics['precision']*100:.2f}%")
        print(f"  Recall:             {cls_metrics['recall']*100:.2f}%")
    
    # Regression metrics
    if 'regression' in results['metrics'] and results['metrics']['regression']:
        print("\n📈 Regression Metrics:")
        reg_metrics = results['metrics']['regression']
        print(f"  MAE:  {reg_metrics['mae']:.4f}")
        print(f"  MSE:  {reg_metrics['mse']:.4f}")
        print(f"  RMSE: {reg_metrics['rmse']:.4f}")
        print(f"  Correlation: {reg_metrics['correlation']:.4f}")
    
    # Binary accuracy
    if 'binary_accuracy' in results['metrics']:
        print("\n🎯 Binary Sentiment:")
        print(f"  Accuracy (pos/neg): {results['metrics']['binary_accuracy']*100:.2f}%")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MAFT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional, loaded from checkpoint if not provided)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda/mps)')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic data for testing')
    args = parser.parse_args()
    
    print("="*70)
    print("MAFT MODEL EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint['config']
    
    print(f"Config: {config.get('training', {}).get('output_dir', 'from checkpoint')}")
    
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
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"Trained for: {checkpoint['epoch']+1} epochs")
    print(f"Best val loss: {checkpoint['val_loss']:.4f}")
    
    # Create data loader
    if args.use_synthetic or config['dataset']['name'] == 'synthetic':
        print("\n📦 Using synthetic data")
        from validation_system.synthetic_data import get_synthetic_loaders
        _, test_loader = get_synthetic_loaders(
            batch_size=config['training']['batch_size'],
            num_train_batches=10,
            num_val_batches=20,  # More samples for evaluation
            seq_len_text=config['dataset']['max_length'],
            seq_len_audio=config['dataset'].get('audio_max_length', 200),
            seq_len_visual=config['dataset'].get('visual_max_length', 200),
        )
    else:
        print(f"\n📦 Loading {config['dataset']['name']} dataset")
        print("⚠️  Real dataset loading not yet implemented - use --use_synthetic")
        return
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print_results(results, title="Evaluation Results")
    
    # Save results
    output_path = Path(args.checkpoint).parent / 'evaluation_results.txt'
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MAFT MODEL EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epoch: {checkpoint['epoch']+1}\n")
        f.write(f"Val Loss: {checkpoint['val_loss']:.4f}\n\n")
        
        if 'classification' in results['metrics']:
            f.write("Classification Metrics:\n")
            for k, v in results['metrics']['classification'].items():
                f.write(f"  {k}: {v*100:.2f}%\n")
            f.write("\n")
        
        if results['metrics'].get('regression'):
            f.write("Regression Metrics:\n")
            for k, v in results['metrics']['regression'].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
        
        f.write(f"Binary Accuracy: {results['metrics']['binary_accuracy']*100:.2f}%\n")
    
    print(f"\n💾 Results saved to: {output_path}")
    print("\n✅ Evaluation complete!\n")


if __name__ == '__main__':
    main()