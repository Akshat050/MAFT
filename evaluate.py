#!/usr/bin/env python3
"""
MAFT Evaluation Script

This script evaluates trained MAFT models and runs ablation studies.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import json
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.maft import MAFT
from utils.data_utils import MOSEIDataset, InterviewDataset, create_dataloader
from utils.metrics import MultimodalMetrics, compute_modality_ablation_metrics
from utils.visualization import plot_confusion_matrix, plot_regression_scatter
from transformers import BertTokenizer


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configuration
    config = checkpoint['config']
    
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
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, config


def evaluate_model(model: MAFT, dataloader, device: torch.device, 
                  output_dir: Path = None, save_errors: bool = True) -> dict:
    """Evaluate model on dataset with error analysis."""
    model.eval()
    metrics = MultimodalMetrics()
    
    # Error analysis storage
    misclassified_samples = []
    regression_outliers = []
    
    print("🔍 Evaluating model...")
    
    # Profile inference time
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Time inference
            start_time = time.time()
            outputs = model(**batch)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Update metrics
            metrics.update(
                outputs['classification_logits'],
                outputs['regression_output'],
                batch['classification_targets'],
                batch['regression_targets']
            )
            
            # Error analysis
            if save_errors:
                cls_preds = torch.argmax(outputs['classification_logits'], dim=1)
                reg_preds = outputs['regression_output'].squeeze(-1)
                
                for i in range(len(cls_preds)):
                    # Check for misclassified samples
                    if cls_preds[i] != batch['classification_targets'][i]:
                        misclassified_samples.append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'true_label': batch['classification_targets'][i].item(),
                            'predicted_label': cls_preds[i].item(),
                            'confidence': torch.softmax(outputs['classification_logits'][i], dim=0).max().item()
                        })
                    
                    # Check for regression outliers (large errors)
                    reg_error = abs(reg_preds[i] - batch['regression_targets'][i])
                    if reg_error > 1.0:  # Threshold for outlier
                        regression_outliers.append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'true_value': batch['regression_targets'][i].item(),
                            'predicted_value': reg_preds[i].item(),
                            'error': reg_error.item()
                        })
    
    # Compute metrics
    all_metrics = metrics.compute_all_metrics()
    
    # Add inference statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    all_metrics['inference'] = {
        'avg_time_per_sample': avg_inference_time,
        'std_time_per_sample': std_inference_time,
        'samples_per_second': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    }
    
    # Print results
    metrics.print_metrics()
    print(f"\n⏱️  Inference Performance:")
    print(f"  Average time per sample: {avg_inference_time:.4f} ± {std_inference_time:.4f} seconds")
    print(f"  Samples per second: {all_metrics['inference']['samples_per_second']:.2f}")
    
    # Save error analysis
    if output_dir and save_errors:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save misclassified samples
        if misclassified_samples:
            misclassified_df = pd.DataFrame(misclassified_samples)
            misclassified_df.to_csv(output_dir / 'misclassified_samples.csv', index=False)
            print(f"📊 Misclassified samples saved: {len(misclassified_samples)} samples")
        
        # Save regression outliers
        if regression_outliers:
            outliers_df = pd.DataFrame(regression_outliers)
            outliers_df.to_csv(output_dir / 'regression_outliers.csv', index=False)
            print(f"📊 Regression outliers saved: {len(regression_outliers)} samples")
        
        # Save metrics to JSON
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {}
            for task, task_metrics in all_metrics.items():
                json_metrics[task] = {}
                for metric_name, metric_value in task_metrics.items():
                    if isinstance(metric_value, np.ndarray):
                        json_metrics[task][metric_name] = metric_value.tolist()
                    else:
                        json_metrics[task][metric_name] = metric_value
            
            json.dump(json_metrics, f, indent=2)
        
        # Plot confusion matrix if available
        if all_metrics['classification'] and 'confusion_matrix' in all_metrics['classification']:
            cm = all_metrics['classification']['confusion_matrix']
            plot_confusion_matrix(
                cm,
                class_names=['Negative', 'Positive'],
                title="Classification Confusion Matrix",
                save_path=output_dir / 'confusion_matrix.png'
            )
        
        # Plot regression scatter if available
        if all_metrics['regression']:
            y_true = np.array(metrics.regression_targets)
            y_pred = np.array(metrics.regression_predictions)
            plot_regression_scatter(
                y_true, y_pred,
                title="Regression Predictions",
                save_path=output_dir / 'regression_scatter.png'
            )
    
    return all_metrics


def run_ablation_study(model: MAFT, dataloader, device: torch.device, 
                      output_dir: Path = None) -> dict:
    """Run ablation study by dropping different modalities."""
    print("\n🔬 Running ablation study...")
    
    ablation_results = compute_modality_ablation_metrics(
        model, dataloader, device, modalities_to_drop=['text', 'audio', 'visual']
    )
    
    # Print ablation results
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    
    for ablation_name, results in ablation_results.items():
        print(f"\n{ablation_name.upper()}:")
        
        if results['classification']:
            cls_metrics = results['classification']
            print(f"  Classification Accuracy: {cls_metrics['accuracy']:.4f}")
            print(f"  Classification F1: {cls_metrics['f1_score']:.4f}")
        
        if results['regression']:
            reg_metrics = results['regression']
            print(f"  Regression MSE: {reg_metrics['mse']:.4f}")
            print(f"  Regression Correlation: {reg_metrics['correlation']:.4f}")
    
    # Save ablation results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        json_results = {}
        for ablation_name, results in ablation_results.items():
            json_results[ablation_name] = {}
            for task, task_metrics in results.items():
                json_results[ablation_name][task] = {}
                for metric_name, metric_value in task_metrics.items():
                    if isinstance(metric_value, np.ndarray):
                        json_results[ablation_name][task][metric_name] = metric_value.tolist()
                    else:
                        json_results[ablation_name][task][metric_name] = metric_value
        
        with open(output_dir / 'ablation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
    
    return ablation_results


def profile_compute_cost(model: MAFT, dataloader, device: torch.device, 
                        num_batches: int = 10) -> dict:
    """Profile compute cost and memory usage."""
    print(f"\n💻 Profiling compute cost...")
    
    model.eval()
    
    # Memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    
    # Time tracking
    forward_times = []
    memory_usage = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Profiling")):
            if batch_idx >= num_batches:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Measure forward pass time
            start_time = time.time()
            outputs = model(**batch)
            forward_time = time.time() - start_time
            forward_times.append(forward_time)
            
            # Measure memory usage
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                memory_usage.append(current_memory - initial_memory)
    
    # Calculate statistics
    avg_forward_time = np.mean(forward_times)
    std_forward_time = np.std(forward_times)
    
    if torch.cuda.is_available():
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
    else:
        avg_memory = max_memory = 0
    
    # Estimate FLOPs (rough approximation)
    batch_size = next(iter(dataloader))['input_ids'].size(0)
    seq_len = next(iter(dataloader))['input_ids'].size(1)
    hidden_dim = model.fusion_transformer.hidden_dim
    
    # Rough FLOPs estimation for transformer
    flops_per_sample = 4 * hidden_dim * hidden_dim * seq_len * seq_len  # Self-attention
    flops_per_sample += 4 * hidden_dim * hidden_dim * seq_len  # Feed-forward
    total_flops = flops_per_sample * batch_size
    
    compute_stats = {
        'avg_forward_time': avg_forward_time,
        'std_forward_time': std_forward_time,
        'samples_per_second': batch_size / avg_forward_time if avg_forward_time > 0 else 0,
        'avg_memory_usage_mb': avg_memory / (1024 * 1024) if avg_memory > 0 else 0,
        'max_memory_usage_mb': max_memory / (1024 * 1024) if max_memory > 0 else 0,
        'estimated_flops': total_flops,
        'estimated_gflops': total_flops / 1e9
    }
    
    print(f"⏱️  Compute Profile:")
    print(f"  Average forward time: {avg_forward_time:.4f} ± {std_forward_time:.4f} seconds")
    print(f"  Samples per second: {compute_stats['samples_per_second']:.2f}")
    print(f"  Average memory usage: {compute_stats['avg_memory_usage_mb']:.1f} MB")
    print(f"  Max memory usage: {compute_stats['max_memory_usage_mb']:.1f} MB")
    print(f"  Estimated GFLOPS: {compute_stats['estimated_gflops']:.2f}")
    
    return compute_stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate MAFT model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['mosei', 'interview'], 
                       required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset (if different from config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study')
    parser.add_argument('--profile', action='store_true',
                       help='Profile compute cost')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--save_errors', action='store_true',
                       help='Save misclassified samples and outliers')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"experiments/evaluation/{args.dataset}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # Create dataset
    data_path = args.data_path or config['dataset']['data_path']
    
    if args.dataset == 'mosei':
        test_dataset = MOSEIDataset(
            data_path,
            tokenizer,
            split='test',
            max_length=config['dataset']['max_length'],
            audio_max_length=config['dataset']['audio_max_length'],
            visual_max_length=config['dataset']['visual_max_length']
        )
    elif args.dataset == 'interview':
        test_dataset = InterviewDataset(
            data_path,
            tokenizer,
            split='test',
            max_length=config['dataset']['max_length'],
            audio_max_length=config['dataset']['audio_max_length'],
            visual_max_length=config['dataset']['visual_max_length']
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create dataloader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"📊 Test dataset size: {len(test_dataset)}")
    
    # Evaluate model
    evaluation_results = evaluate_model(
        model, test_loader, device, output_dir, args.save_errors
    )
    
    # Profile compute cost
    if args.profile:
        compute_stats = profile_compute_cost(model, test_loader, device)
        
        # Save compute profile
        with open(output_dir / 'compute_profile.json', 'w') as f:
            json.dump(compute_stats, f, indent=2)
    
    # Run ablation study
    if args.ablation:
        ablation_results = run_ablation_study(model, test_loader, device, output_dir)
    
    print(f"\n✅ Evaluation completed! Results saved to: {output_dir}")


if __name__ == '__main__':
    main() 