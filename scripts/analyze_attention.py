#!/usr/bin/env python3
"""
MAFT Attention Analysis Script

This script provides comprehensive analysis of attention patterns in MAFT,
including cross-modal interactions, modality importance, and temporal patterns.
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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.maft import MAFT
from utils.data_utils import MOSEIDataset, InterviewDataset, create_dataloader
from utils.visualization import (
    plot_attention_heatmap, plot_modality_importance, 
    plot_temporal_attention, plot_feature_importance
)
from transformers import BertTokenizer


def load_model_and_data(checkpoint_path: str, config: dict, dataset_name: str, 
                       device: torch.device) -> Tuple[MAFT, any, any]:
    """Load trained model and data."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with attention extraction enabled
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
        freeze_bert=config['model']['freeze_bert'],
        return_attention=True
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # Create dataset
    if dataset_name == 'mosei':
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
    
    return model, dataloader, tokenizer


def analyze_cross_modal_attention(attention_weights: torch.Tensor, 
                                text_tokens: List[str],
                                audio_len: int, visual_len: int) -> Dict[str, float]:
    """Analyze cross-modal attention patterns."""
    
    batch_size, num_heads, total_len, _ = attention_weights.shape
    text_len = len(text_tokens)
    
    # Define modality boundaries
    cls_len = 1
    text_start, text_end = cls_len, cls_len + text_len
    audio_start, audio_end = text_end, text_end + audio_len
    visual_start, visual_end = audio_end, audio_end + visual_len
    
    # Calculate cross-modal attention scores
    cross_modal_scores = {
        'text_to_audio': 0.0,
        'text_to_visual': 0.0,
        'audio_to_text': 0.0,
        'audio_to_visual': 0.0,
        'visual_to_text': 0.0,
        'visual_to_audio': 0.0
    }
    
    # Average across batch and heads
    avg_attention = attention_weights.mean(dim=0).mean(dim=0)  # [total_len, total_len]
    
    # Text to other modalities
    if text_len > 0 and audio_len > 0:
        cross_modal_scores['text_to_audio'] = avg_attention[text_start:text_end, audio_start:audio_end].mean().item()
    if text_len > 0 and visual_len > 0:
        cross_modal_scores['text_to_visual'] = avg_attention[text_start:text_end, visual_start:visual_end].mean().item()
    
    # Audio to other modalities
    if audio_len > 0 and text_len > 0:
        cross_modal_scores['audio_to_text'] = avg_attention[audio_start:audio_end, text_start:text_end].mean().item()
    if audio_len > 0 and visual_len > 0:
        cross_modal_scores['audio_to_visual'] = avg_attention[audio_start:audio_end, visual_start:visual_end].mean().item()
    
    # Visual to other modalities
    if visual_len > 0 and text_len > 0:
        cross_modal_scores['visual_to_text'] = avg_attention[visual_start:visual_end, text_start:text_end].mean().item()
    if visual_len > 0 and audio_len > 0:
        cross_modal_scores['visual_to_audio'] = avg_attention[visual_start:visual_end, audio_start:audio_end].mean().item()
    
    return cross_modal_scores


def analyze_modality_importance(attention_weights: torch.Tensor,
                              text_len: int, audio_len: int, visual_len: int) -> Dict[str, float]:
    """Analyze the importance of each modality based on attention weights."""
    
    batch_size, num_heads, total_len, _ = attention_weights.shape
    
    # Define modality boundaries
    cls_len = 1
    text_start, text_end = cls_len, cls_len + text_len
    audio_start, audio_end = text_end, text_end + audio_len
    visual_start, visual_end = audio_end, audio_end + visual_len
    
    # Calculate attention to each modality from all other tokens
    modality_importance = {}
    
    # Average across batch and heads
    avg_attention = attention_weights.mean(dim=0).mean(dim=0)  # [total_len, total_len]
    
    # Attention to text modality
    if text_len > 0:
        text_attention = avg_attention[:, text_start:text_end].mean().item()
        modality_importance['text'] = text_attention
    
    # Attention to audio modality
    if audio_len > 0:
        audio_attention = avg_attention[:, audio_start:audio_end].mean().item()
        modality_importance['audio'] = audio_attention
    
    # Attention to visual modality
    if visual_len > 0:
        visual_attention = avg_attention[:, visual_start:visual_end].mean().item()
        modality_importance['visual'] = visual_attention
    
    return modality_importance


def analyze_attention_head_specialization(attention_weights: torch.Tensor,
                                        text_len: int, audio_len: int, visual_len: int) -> Dict[str, List[float]]:
    """Analyze how different attention heads specialize in different modalities."""
    
    batch_size, num_heads, total_len, _ = attention_weights.shape
    
    # Define modality boundaries
    cls_len = 1
    text_start, text_end = cls_len, cls_len + text_len
    audio_start, audio_end = text_end, text_end + audio_len
    visual_start, visual_end = audio_end, audio_end + visual_len
    
    # Calculate modality preference for each head
    head_specialization = {
        'text_preference': [],
        'audio_preference': [],
        'visual_preference': []
    }
    
    # Average across batch
    avg_attention = attention_weights.mean(dim=0)  # [num_heads, total_len, total_len]
    
    for head_idx in range(num_heads):
        head_attention = avg_attention[head_idx]
        
        # Calculate preference for each modality
        if text_len > 0:
            text_pref = head_attention[:, text_start:text_end].mean().item()
            head_specialization['text_preference'].append(text_pref)
        
        if audio_len > 0:
            audio_pref = head_attention[:, audio_start:audio_end].mean().item()
            head_specialization['audio_preference'].append(audio_pref)
        
        if visual_len > 0:
            visual_pref = head_attention[:, visual_start:visual_end].mean().item()
            head_specialization['visual_preference'].append(visual_pref)
    
    return head_specialization


def create_attention_analysis_report(model: MAFT, dataloader: any, tokenizer: BertTokenizer,
                                   device: torch.device, output_dir: Path, 
                                   num_samples: int = 100) -> Dict[str, any]:
    """Create comprehensive attention analysis report."""
    
    print(f"🔍 Analyzing attention patterns for {num_samples} samples...")
    
    all_cross_modal_scores = []
    all_modality_importance = []
    all_head_specialization = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing attention"):
            if sample_count >= num_samples:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with attention extraction
            outputs = model(**batch)
            attention_weights = outputs['attention_weights']
            
            # Get text tokens
            text_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][0].cpu().numpy())
            text_tokens = [token for token in text_tokens if token != '[PAD]']
            
            # Get sequence lengths
            text_len = len(text_tokens)
            audio_len = batch['audio_features'].size(1)
            visual_len = batch['visual_features'].size(1)
            
            # Analyze attention patterns
            cross_modal_scores = analyze_cross_modal_attention(
                attention_weights, text_tokens, audio_len, visual_len
            )
            all_cross_modal_scores.append(cross_modal_scores)
            
            modality_importance = analyze_modality_importance(
                attention_weights, text_len, audio_len, visual_len
            )
            all_modality_importance.append(modality_importance)
            
            head_specialization = analyze_attention_head_specialization(
                attention_weights, text_len, audio_len, visual_len
            )
            all_head_specialization.append(head_specialization)
            
            sample_count += 1
    
    # Aggregate results
    report = {
        'cross_modal_attention': {},
        'modality_importance': {},
        'head_specialization': {},
        'summary_statistics': {}
    }
    
    # Average cross-modal attention scores
    for key in all_cross_modal_scores[0].keys():
        values = [score[key] for score in all_cross_modal_scores]
        report['cross_modal_attention'][key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Average modality importance
    for key in all_modality_importance[0].keys():
        values = [importance[key] for importance in all_modality_importance]
        report['modality_importance'][key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Average head specialization
    for key in all_head_specialization[0].keys():
        all_values = []
        for spec in all_head_specialization:
            all_values.extend(spec[key])
        report['head_specialization'][key] = {
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values)
        }
    
    # Summary statistics
    report['summary_statistics'] = {
        'total_samples_analyzed': sample_count,
        'average_text_length': np.mean([len(score) for score in all_cross_modal_scores]),
        'attention_patterns_identified': len(report['cross_modal_attention'])
    }
    
    return report


def visualize_attention_analysis(report: Dict[str, any], output_dir: Path):
    """Create visualizations for attention analysis."""
    
    print("📊 Creating attention visualizations...")
    
    # 1. Cross-modal attention heatmap
    cross_modal_data = report['cross_modal_attention']
    modalities = ['text', 'audio', 'visual']
    
    # Create cross-modal attention matrix
    attention_matrix = np.zeros((3, 3))
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i != j:
                key = f'{mod1}_to_{mod2}'
                if key in cross_modal_data:
                    attention_matrix[i, j] = cross_modal_data[key]['mean']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, annot=True, fmt='.3f', 
                xticklabels=modalities, yticklabels=modalities,
                cmap='Blues', cbar_kws={'label': 'Attention Weight'})
    plt.title('Cross-Modal Attention Patterns')
    plt.xlabel('Target Modality')
    plt.ylabel('Source Modality')
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_modal_attention.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Modality importance bar plot
    modality_data = report['modality_importance']
    modalities = list(modality_data.keys())
    means = [modality_data[mod]['mean'] for mod in modalities]
    stds = [modality_data[mod]['std'] for mod in modalities]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modalities, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Average Attention Weight')
    plt.title('Modality Importance')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'modality_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Head specialization plot
    head_data = report['head_specialization']
    modalities = ['text', 'audio', 'visual']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, modality in enumerate(modalities):
        key = f'{modality}_preference'
        if key in head_data:
            means = [head_data[key]['mean']] * 12  # 12 heads
            stds = [head_data[key]['std']] * 12
            
            axes[i].bar(range(1, 13), means, yerr=stds, capsize=5, alpha=0.7)
            axes[i].set_xlabel('Attention Head')
            axes[i].set_ylabel('Preference Score')
            axes[i].set_title(f'{modality.title()} Preference by Head')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'head_specialization.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze MAFT attention patterns')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['mosei', 'interview'], required=True,
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='attention_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to analyze')
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
    
    # Load model and data
    model, dataloader, tokenizer = load_model_and_data(
        args.checkpoint, config, args.dataset, device
    )
    
    # Create attention analysis report
    report = create_attention_analysis_report(
        model, dataloader, tokenizer, device, output_path, args.num_samples
    )
    
    # Save report
    with open(output_path / 'attention_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create visualizations
    visualize_attention_analysis(report, output_path)
    
    # Print summary
    print(f"\n📋 Attention Analysis Summary:")
    print(f"   Samples analyzed: {report['summary_statistics']['total_samples_analyzed']}")
    print(f"   Cross-modal patterns: {len(report['cross_modal_attention'])}")
    
    print(f"\n🎯 Key Findings:")
    
    # Most important modality
    modality_importance = report['modality_importance']
    if modality_importance:
        most_important = max(modality_importance.items(), 
                           key=lambda x: x[1]['mean'])
        print(f"   Most important modality: {most_important[0]} "
              f"({most_important[1]['mean']:.3f}±{most_important[1]['std']:.3f})")
    
    # Strongest cross-modal interaction
    cross_modal = report['cross_modal_attention']
    if cross_modal:
        strongest = max(cross_modal.items(), key=lambda x: x[1]['mean'])
        print(f"   Strongest cross-modal interaction: {strongest[0]} "
              f"({strongest[1]['mean']:.3f}±{strongest[1]['std']:.3f})")
    
    print(f"\n📁 Results saved to: {output_path}")


if __name__ == '__main__':
    main() 