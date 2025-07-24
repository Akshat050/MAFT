#!/usr/bin/env python3
"""
MAFT Efficiency Analysis Script

This script provides comprehensive efficiency analysis including:
- Parameter count comparison
- Training and inference speed
- Memory usage analysis
- Computational complexity analysis
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import json
import time
import psutil
import GPUtil
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.maft import MAFT
from models.baselines import (
    TextOnlyBERT, LateFusion, MAGBERT, MulT, 
    EarlyFusionMAFT, LateFusionMAFT
)
from utils.data_utils import MOSEIDataset, InterviewDataset, create_dataloader
from transformers import BertTokenizer


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameters_millions': total_params / 1e6
    }


def measure_memory_usage(model: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    """Measure memory usage of model."""
    if device.type == 'cuda':
        # GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Allocate model to GPU
        model = model.to(device)
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        peak = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        
        return {
            'gpu_allocated_gb': allocated,
            'gpu_reserved_gb': reserved,
            'gpu_peak_gb': peak
        }
    else:
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'cpu_memory_mb': memory_info.rss / 1024**2,
            'cpu_memory_gb': memory_info.rss / 1024**3
        }


def measure_inference_speed(model: torch.nn.Module, dataloader: any, 
                          device: torch.device, num_batches: int = 50) -> Dict[str, float]:
    """Measure inference speed of model."""
    model.eval()
    
    inference_times = []
    samples_per_second = []
    
    print(f"⏱️  Measuring inference speed for {num_batches} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Inference speed test")):
            if i >= num_batches:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Measure inference time
            start_time = time.time()
            outputs = model(**batch)
            end_time = time.time()
            
            inference_time = end_time - start_time
            batch_size = batch['input_ids'].size(0)
            
            inference_times.append(inference_time)
            samples_per_second.append(batch_size / inference_time)
    
    return {
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'std_inference_time_ms': np.std(inference_times) * 1000,
        'avg_samples_per_second': np.mean(samples_per_second),
        'std_samples_per_second': np.std(samples_per_second),
        'min_inference_time_ms': np.min(inference_times) * 1000,
        'max_inference_time_ms': np.max(inference_times) * 1000
    }


def measure_training_speed(model: torch.nn.Module, dataloader: any, 
                         device: torch.device, num_batches: int = 20) -> Dict[str, float]:
    """Measure training speed of model."""
    model.train()
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    training_times = []
    gradient_computation_times = []
    
    print(f"🏃 Measuring training speed for {num_batches} batches...")
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training speed test")):
        if i >= num_batches:
            break
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Measure training time
        start_time = time.time()
        
        optimizer.zero_grad()
        outputs = model(**batch)
        
        # Compute loss
        loss = criterion(outputs['classification_logits'], batch['classification_targets'])
        
        # Measure gradient computation time
        grad_start = time.time()
        loss.backward()
        grad_end = time.time()
        
        optimizer.step()
        end_time = time.time()
        
        training_time = end_time - start_time
        grad_time = grad_end - grad_start
        
        training_times.append(training_time)
        gradient_computation_times.append(grad_time)
    
    return {
        'avg_training_time_ms': np.mean(training_times) * 1000,
        'std_training_time_ms': np.std(training_times) * 1000,
        'avg_gradient_time_ms': np.mean(gradient_computation_times) * 1000,
        'std_gradient_time_ms': np.std(gradient_computation_times) * 1000,
        'avg_samples_per_second': len(dataloader.dataset) / (np.mean(training_times) * num_batches)
    }


def analyze_computational_complexity(model: torch.nn.Module, dataloader: any,
                                   device: torch.device) -> Dict[str, any]:
    """Analyze computational complexity of model."""
    
    # Get model architecture info
    model_info = {
        'num_layers': getattr(model, 'num_layers', 1),
        'num_heads': getattr(model, 'num_heads', 12),
        'hidden_dim': getattr(model, 'hidden_dim', 768),
        'model_type': model.__class__.__name__
    }
    
    # Analyze sequence length impact
    sequence_lengths = [50, 100, 200, 400, 800]
    inference_times = []
    
    print("🔬 Analyzing computational complexity...")
    
    for seq_len in sequence_lengths:
        # Create dummy batch with specific sequence length
        batch_size = 8
        dummy_batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            'attention_mask': torch.ones(batch_size, seq_len).to(device),
            'audio_features': torch.randn(batch_size, seq_len, 74).to(device),
            'audio_mask': torch.ones(batch_size, seq_len).to(device),
            'visual_features': torch.randn(batch_size, seq_len, 35).to(device),
            'visual_mask': torch.ones(batch_size, seq_len).to(device),
            'classification_targets': torch.randint(0, 2, (batch_size,)).to(device),
            'regression_targets': torch.randn(batch_size, 1).to(device)
        }
        
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = model(**dummy_batch)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            inference_times.append(inference_time)
    
    # Calculate complexity metrics
    complexity_analysis = {
        'sequence_length_impact': dict(zip(sequence_lengths, inference_times)),
        'quadratic_scaling': np.polyfit(sequence_lengths, inference_times, 2)[0],
        'linear_scaling': np.polyfit(sequence_lengths, inference_times, 1)[0]
    }
    
    return {**model_info, **complexity_analysis}


def compare_model_efficiency(models: Dict[str, torch.nn.Module], dataloader: any,
                           device: torch.device) -> Dict[str, Dict[str, any]]:
    """Compare efficiency across different models."""
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📊 Analyzing efficiency of {model_name}...")
        
        # Count parameters
        param_info = count_parameters(model)
        
        # Measure memory usage
        memory_info = measure_memory_usage(model, device)
        
        # Measure inference speed
        inference_info = measure_inference_speed(model, dataloader, device, num_batches=20)
        
        # Measure training speed
        training_info = measure_training_speed(model, dataloader, device, num_batches=10)
        
        # Analyze computational complexity
        complexity_info = analyze_computational_complexity(model, dataloader, device)
        
        results[model_name] = {
            'parameters': param_info,
            'memory': memory_info,
            'inference': inference_info,
            'training': training_info,
            'complexity': complexity_info
        }
    
    return results


def create_efficiency_report(results: Dict[str, Dict[str, any]], output_dir: Path):
    """Create comprehensive efficiency report."""
    
    print("📋 Creating efficiency report...")
    
    # Create comparison tables
    comparison_data = []
    
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Parameters (M)': result['parameters']['parameters_millions'],
            'GPU Memory (GB)': result['memory'].get('gpu_allocated_gb', 0),
            'Inference Time (ms)': result['inference']['avg_inference_time_ms'],
            'Samples/sec': result['inference']['avg_samples_per_second'],
            'Training Time (ms)': result['training']['avg_training_time_ms']
        })
    
    # Create efficiency comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Parameter count comparison
    models = [data['Model'] for data in comparison_data]
    params = [data['Parameters (M)'] for data in comparison_data]
    
    axes[0, 0].bar(models, params, alpha=0.7, color='skyblue')
    axes[0, 0].set_ylabel('Parameters (Millions)')
    axes[0, 0].set_title('Model Size Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Memory usage comparison
    memory = [data['GPU Memory (GB)'] for data in comparison_data]
    
    axes[0, 1].bar(models, memory, alpha=0.7, color='lightcoral')
    axes[0, 1].set_ylabel('GPU Memory (GB)')
    axes[0, 1].set_title('Memory Usage Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Inference speed comparison
    inference_times = [data['Inference Time (ms)'] for data in comparison_data]
    
    axes[1, 0].bar(models, inference_times, alpha=0.7, color='lightgreen')
    axes[1, 0].set_ylabel('Inference Time (ms)')
    axes[1, 0].set_title('Inference Speed Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Training speed comparison
    training_times = [data['Training Time (ms)'] for data in comparison_data]
    
    axes[1, 1].bar(models, training_times, alpha=0.7, color='gold')
    axes[1, 1].set_ylabel('Training Time (ms)')
    axes[1, 1].set_title('Training Speed Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create complexity analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sequence length vs inference time
    for model_name, result in results.items():
        complexity = result['complexity']
        seq_lengths = list(complexity['sequence_length_impact'].keys())
        inference_times = list(complexity['sequence_length_impact'].values())
        
        axes[0].plot(seq_lengths, inference_times, marker='o', label=model_name, linewidth=2)
    
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Inference Time (ms)')
    axes[0].set_title('Computational Complexity Analysis')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Efficiency vs accuracy trade-off (placeholder)
    efficiency_scores = []
    for data in comparison_data:
        # Simple efficiency score: samples/sec / parameters
        efficiency = data['Samples/sec'] / data['Parameters (M)']
        efficiency_scores.append(efficiency)
    
    axes[1].bar(models, efficiency_scores, alpha=0.7, color='purple')
    axes[1].set_ylabel('Efficiency Score (samples/sec/M params)')
    axes[1].set_title('Efficiency Score Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    with open(output_dir / 'efficiency_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save comparison table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'efficiency_comparison.csv', index=False)
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Analyze MAFT efficiency')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['mosei', 'interview'], required=True,
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='efficiency_analysis',
                       help='Output directory for analysis results')
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
    
    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # Create dataset
    if args.dataset == 'mosei':
        dataset = MOSEIDataset(
            data_dir=config['data']['data_dir'],
            split='test',
            max_length=config['data']['text_max_length'],
            audio_max_length=config['data']['audio_max_length'],
            visual_max_length=config['data']['visual_max_length']
        )
    else:
        dataset = InterviewDataset(
            data_dir=config['data']['data_dir'],
            split='test',
            max_length=config['data']['text_max_length'],
            audio_max_length=config['data']['audio_max_length'],
            visual_max_length=config['data']['visual_max_length']
        )
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Create models for comparison
    models = {
        'Text-only BERT': TextOnlyBERT(
            model_name=config['model']['text_model_name'],
            hidden_dim=config['model']['hidden_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ),
        'Late Fusion': LateFusion(
            text_model_name=config['model']['text_model_name'],
            hidden_dim=config['model']['hidden_dim'],
            audio_input_dim=config['model']['audio_input_dim'],
            visual_input_dim=config['model']['visual_input_dim'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ),
        'MAFT (ours)': MAFT(
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
    }
    
    # Compare model efficiency
    results = compare_model_efficiency(models, dataloader, device)
    
    # Create efficiency report
    comparison_df = create_efficiency_report(results, output_path)
    
    # Print summary
    print(f"\n📊 Efficiency Analysis Summary:")
    print(f"   Models analyzed: {len(models)}")
    print(f"   Results saved to: {output_path}")
    
    print(f"\n🏆 Efficiency Rankings:")
    
    # Most efficient model
    efficiency_scores = []
    for data in comparison_df.to_dict('records'):
        efficiency = data['Samples/sec'] / data['Parameters (M)']
        efficiency_scores.append((data['Model'], efficiency))
    
    efficiency_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(efficiency_scores):
        print(f"   {i+1}. {model}: {score:.2f} samples/sec/M params")
    
    print(f"\n💡 Key Insights:")
    
    # Parameter efficiency
    maft_params = comparison_df[comparison_df['Model'] == 'MAFT (ours)']['Parameters (M)'].iloc[0]
    bert_params = comparison_df[comparison_df['Model'] == 'Text-only BERT']['Parameters (M)'].iloc[0]
    param_reduction = ((bert_params - maft_params) / bert_params) * 100
    
    print(f"   • MAFT uses {param_reduction:.1f}% fewer parameters than BERT")
    
    # Speed comparison
    maft_speed = comparison_df[comparison_df['Model'] == 'MAFT (ours)']['Samples/sec'].iloc[0]
    bert_speed = comparison_df[comparison_df['Model'] == 'Text-only BERT']['Samples/sec'].iloc[0]
    speed_ratio = maft_speed / bert_speed
    
    print(f"   • MAFT inference speed: {speed_ratio:.2f}x relative to BERT")


if __name__ == '__main__':
    main() 