#!/usr/bin/env python3
"""
Diagnostic script to identify the source of NaN losses in MOSEI training.
Updated with correct MAFT parameter names.
"""

import torch
import numpy as np
import yaml

print("="*70)
print("DIAGNOSING NaN ISSUE IN MOSEI TRAINING")
print("="*70)

# Load config
config_path = 'configs/mosei_benchmark_config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"Config structure: {list(config.keys())}")

# Extract model and training parameters
model_config = config.get('model', {})
training_config = config.get('training', {})

# Load model with correct parameter names
from models.maft import MAFT

# MAFT uses: hidden_dim (not d_model)
model = MAFT(
    num_classes=model_config['num_classes'],
    hidden_dim=model_config['hidden_dim'],
    num_heads=model_config['num_heads'],
    num_layers=model_config['num_layers'],
    dropout=model_config['dropout']
)
print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Load data
from mosei_dataloader import get_mosei_loaders
train_loader, valid_loader, test_loader = get_mosei_loaders(
    data_dir=config['dataset']['data_dir'],
    batch_size=training_config['batch_size'],
    num_workers=0
)
print(f"✅ Data loaded: {len(train_loader)} train batches")

# Get first batch
print("\n" + "="*70)
print("STEP 1: INSPECTING BATCH STRUCTURE")
print("="*70)
batch = next(iter(train_loader))

print(f"\nBatch keys: {list(batch.keys())}")
print("\nDetailed inspection:")
for key in batch.keys():
    if isinstance(batch[key], torch.Tensor):
        tensor = batch[key]
        print(f"\n{key}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        if tensor.numel() > 0:
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  Has NaN: {has_nan}")
            print(f"  Has Inf: {has_inf}")
            
            if has_nan:
                nan_count = torch.isnan(tensor).sum().item()
                print(f"  ⚠️  WARNING: {nan_count} NaN values found!")
            if has_inf:
                inf_count = torch.isinf(tensor).sum().item()
                print(f"  ⚠️  WARNING: {inf_count} Inf values found!")

# Check if augmentation is being used
print(f"\nAugmentation enabled: {training_config.get('use_augmentation', False)}")

# Test forward pass
print("\n" + "="*70)
print("STEP 2: TESTING FORWARD PASS")
print("="*70)
try:
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    
    print("✅ Forward pass successful!")
    print(f"\nOutput inspection:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Logits range: [{outputs['logits'].min().item():.4f}, {outputs['logits'].max().item():.4f}]")
    print(f"  Logits has NaN: {torch.isnan(outputs['logits']).any().item()}")
    print(f"  Logits has Inf: {torch.isinf(outputs['logits']).any().item()}")
    
    print(f"\n  Regression shape: {outputs['regression'].shape}")
    print(f"  Regression range: [{outputs['regression'].min().item():.4f}, {outputs['regression'].max().item():.4f}]")
    print(f"  Regression has NaN: {torch.isnan(outputs['regression']).any().item()}")
    print(f"  Regression has Inf: {torch.isinf(outputs['regression']).any().item()}")
    
    if torch.isnan(outputs['logits']).any() or torch.isnan(outputs['regression']).any():
        print("\n⚠️  NaN detected in forward pass outputs!")
        
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nCannot proceed with loss computation test.")
    exit(1)

# Test loss computation
print("\n" + "="*70)
print("STEP 3: TESTING LOSS COMPUTATION")
print("="*70)
try:
    model.train()
    outputs = model(batch)
    
    # Import compute_loss from train.py
    from train import compute_loss
    
    loss, parts = compute_loss(
        outputs, batch, 
        lambda_cls=training_config['classification_weight'], 
        lambda_reg=training_config['regression_weight'], 
        lambda_cons=training_config['consistency_weight']
    )
    
    print(f"\nLoss components:")
    print(f"  Classification loss: {parts['classification_loss'].item():.6f}")
    print(f"  Regression loss: {parts['regression_loss'].item():.6f}")
    print(f"  Consistency loss: {parts['consistency_loss'].item():.6f}")
    print(f"  Total loss: {loss.item():.6f}")
    
    # Check each component for NaN
    nan_components = []
    if torch.isnan(parts['classification_loss']):
        nan_components.append("Classification")
    if torch.isnan(parts['regression_loss']):
        nan_components.append("Regression")
    if torch.isnan(parts['consistency_loss']):
        nan_components.append("Consistency")
    
    if nan_components:
        print(f"\n❌ NaN detected in: {', '.join(nan_components)}")
        print("\nLikely causes:")
        if "Consistency" in nan_components:
            print("  - Consistency loss may have division by zero")
            print("  - Check if modality dropout is removing all samples")
        if "Classification" in nan_components:
            print("  - Classification targets may be invalid")
            print("  - Check target range and class indices")
        if "Regression" in nan_components:
            print("  - Regression targets may be unbounded")
            print("  - Check if targets contain extreme values")
    else:
        print("\n✅ All loss components are valid!")
        print("   NaN issue may be intermittent or related to specific batches.")
        
except Exception as e:
    print(f"❌ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\nRecommended actions:")
print("1. If NaN found in input data → Add input validation in data loader")
print("2. If NaN in forward pass → Check model initialization and text projection layer")
print("3. If NaN in consistency loss → Add epsilon to avoid division by zero")
print("4. If NaN in regression loss → Clamp regression targets to [-3, 3]")
print("5. Try disabling augmentation: set use_augmentation: false in config")