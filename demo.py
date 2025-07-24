#!/usr/bin/env python3
"""
MAFT Demonstration Script

This script demonstrates the complete MAFT system with a simple example.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.maft import MAFT
from utils.metrics import MultimodalMetrics
from transformers import BertTokenizer


def create_demo_data():
    """Create demo data for testing."""
    # Mock input data
    batch_size = 2
    text_len = 10
    audio_len = 50
    visual_len = 50
    
    # Text data
    input_ids = torch.randint(0, 1000, (batch_size, text_len))
    attention_mask = torch.ones(batch_size, text_len)
    
    # Audio data (COVAREP features)
    audio_features = torch.randn(batch_size, audio_len, 74)
    audio_mask = torch.ones(batch_size, audio_len)
    audio_lengths = torch.tensor([audio_len, audio_len])
    
    # Visual data (FACET features)
    visual_features = torch.randn(batch_size, visual_len, 35)
    visual_mask = torch.ones(batch_size, visual_len)
    visual_lengths = torch.tensor([visual_len, visual_len])
    
    # Labels
    classification_targets = torch.randint(0, 2, (batch_size,))
    regression_targets = torch.rand(batch_size) * 6 - 3  # Range: [-3, 3]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'audio_features': audio_features,
        'audio_mask': audio_mask,
        'audio_lengths': audio_lengths,
        'visual_features': visual_features,
        'visual_mask': visual_mask,
        'visual_lengths': visual_lengths,
        'classification_targets': classification_targets,
        'regression_targets': regression_targets
    }


def demo_model_forward():
    """Demonstrate model forward pass."""
    print("🚀 MAFT Model Demonstration")
    print("="*50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MAFT(
        text_model_name="bert-base-uncased",
        hidden_dim=768,
        num_heads=12,
        num_layers=1,
        audio_input_dim=74,
        visual_input_dim=35,
        num_classes=2,
        dropout=0.1,
        modality_dropout_rate=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create demo data
    batch = create_demo_data()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Forward pass
    print("\n📊 Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
    
    # Print outputs
    print(f"Classification logits shape: {outputs['classification_logits'].shape}")
    print(f"Regression output shape: {outputs['regression_output'].shape}")
    print(f"CLS features shape: {outputs['cls_features'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    # Show predictions
    cls_preds = torch.argmax(outputs['classification_logits'], dim=1)
    reg_preds = outputs['regression_output'].squeeze(-1)
    
    print(f"\n📈 Predictions:")
    for i in range(len(cls_preds)):
        print(f"  Sample {i+1}:")
        print(f"    Classification: {cls_preds[i].item()} (target: {batch['classification_targets'][i].item()})")
        print(f"    Regression: {reg_preds[i].item():.3f} (target: {batch['regression_targets'][i].item():.3f})")
    
    return model, outputs, batch


def demo_training_step():
    """Demonstrate a single training step."""
    print("\n🎯 Training Step Demonstration")
    print("="*50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and data
    model, outputs, batch = demo_model_forward()
    
    # Create loss function
    from models.maft import MAFTLoss
    criterion = MAFTLoss(classification_weight=0.5, regression_weight=0.5)
    
    # Compute loss
    loss_dict = criterion(
        outputs['classification_logits'],
        outputs['regression_output'],
        batch['classification_targets'],
        batch['regression_targets']
    )
    
    print(f"\n💡 Loss computation:")
    print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Classification loss: {loss_dict['classification_loss'].item():.4f}")
    print(f"  Regression loss: {loss_dict['regression_loss'].item():.4f}")
    
    # Simulate backward pass
    print(f"\n🔄 Simulating backward pass...")
    loss_dict['total_loss'].backward()
    
    # Check gradients
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"  Total gradient norm: {total_grad_norm:.4f}")


def demo_metrics():
    """Demonstrate metrics computation."""
    print("\n📊 Metrics Demonstration")
    print("="*50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and get outputs
    model, outputs, batch = demo_model_forward()
    
    # Create metrics
    metrics = MultimodalMetrics()
    
    # Update metrics
    metrics.update(
        outputs['classification_logits'],
        outputs['regression_output'],
        batch['classification_targets'],
        batch['regression_targets']
    )
    
    # Compute and display metrics
    all_metrics = metrics.compute_all_metrics()
    metrics.print_metrics()


def demo_modality_ablation():
    """Demonstrate modality ablation."""
    print("\n🔬 Modality Ablation Demonstration")
    print("="*50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = MAFT(
        text_model_name="bert-base-uncased",
        hidden_dim=768,
        num_heads=12,
        num_layers=1,
        audio_input_dim=74,
        visual_input_dim=35,
        num_classes=2,
        dropout=0.1,
        modality_dropout_rate=0.1
    ).to(device)
    
    # Create demo data
    batch = create_demo_data()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Test different modality combinations
    modalities = ['text', 'audio', 'visual']
    
    for modality in modalities:
        print(f"\n🧪 Testing without {modality} modality:")
        
        # Create modified batch
        modified_batch = batch.copy()
        
        if modality == 'text':
            modified_batch['input_ids'] = torch.zeros_like(batch['input_ids'])
            modified_batch['attention_mask'] = torch.zeros_like(batch['attention_mask'])
        elif modality == 'audio':
            modified_batch['audio_features'] = torch.zeros_like(batch['audio_features'])
            modified_batch['audio_mask'] = torch.zeros_like(batch['audio_mask'])
        elif modality == 'visual':
            modified_batch['visual_features'] = torch.zeros_like(batch['visual_features'])
            modified_batch['visual_mask'] = torch.zeros_like(batch['visual_mask'])
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**modified_batch)
        
        # Show predictions
        cls_preds = torch.argmax(outputs['classification_logits'], dim=1)
        reg_preds = outputs['regression_output'].squeeze(-1)
        
        print(f"  Classification predictions: {cls_preds.tolist()}")
        print(f"  Regression predictions: {reg_preds.tolist()}")


def main():
    """Run all demonstrations."""
    print("🎉 Welcome to MAFT (Multimodal Attention Fusion Transformer) Demo!")
    print("This demo showcases the key features of the MAFT system.\n")
    
    try:
        # Model forward pass
        demo_model_forward()
        
        # Training step
        demo_training_step()
        
        # Metrics computation
        demo_metrics()
        
        # Modality ablation
        demo_modality_ablation()
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("="*60)
        print("\n📝 Next steps:")
        print("1. Prepare your data using scripts/prepare_mosei.py or scripts/prepare_interview.py")
        print("2. Train the model: python train.py --config configs/mosei_config.yaml")
        print("3. Evaluate the model: python evaluate.py --checkpoint path/to/checkpoint --dataset mosei")
        print("4. Run ablation studies: python scripts/run_ablations.py --config configs/mosei_config.yaml")
        print("5. Run comprehensive experiments: python scripts/run_experiments.py --config configs/mosei_config.yaml --dataset mosei")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your installation and dependencies.")


if __name__ == '__main__':
    import os
    main() 