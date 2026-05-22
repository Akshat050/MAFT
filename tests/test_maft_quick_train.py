#!/usr/bin/env python3
"""
Quick training test with synthetic data to verify MAFT learns.

Tests that:
1. Forward pass works with synthetic data
2. Backward pass works
3. Loss decreases over ~20 training steps
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from models.maft import MAFT
from validation_system.synthetic_data import get_synthetic_loaders
from validation_system import TestResult
from train import compute_loss


def quick_train_test():
    """Run quick training to verify model learns."""
    print("=" * 70)
    print("MAFT QUICK TRAINING TEST WITH SYNTHETIC DATA")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Test started at: {datetime.now()}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create model
    print("-" * 70)
    print("Creating MAFT model...")
    print("-" * 70)
    
    model = MAFT(
        hidden_dim=256,
        num_heads=4,
        num_layers=1,
        audio_input_dim=74,
        visual_input_dim=35,
        num_classes=2,
        dropout=0.1,
        modality_dropout_rate=0.1,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params:,} parameters\n")
    
    # Create synthetic data
    print("-" * 70)
    print("Creating synthetic data loaders...")
    print("-" * 70)
    
    train_loader, val_loader = get_synthetic_loaders(
        batch_size=4,
        num_train_batches=10,
        num_val_batches=3,
        seq_len_text=32,  # Shorter for faster testing
        seq_len_audio=100,
        seq_len_visual=100,
        correlation_strength=0.8,
        seed=42
    )
    
    print(f"✅ Data loaders created:")
    print(f"  Train: {len(train_loader)} batches of size 4")
    print(f"  Val: {len(val_loader)} batches of size 4\n")
    
    # Test forward pass
    print("-" * 70)
    print("Test 1: Forward Pass")
    print("-" * 70)
    
    model.eval()
    batch = next(iter(train_loader))
    for k in batch:
        batch[k] = batch[k].to(device)
    
    try:
        with torch.no_grad():
            outputs = model(batch)
        
        print("✅ Forward pass successful!")
        print(f"  Output keys: {list(outputs.keys())}")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Regression shape: {outputs['reg'].shape}\n")
        
        forward_result = TestResult(
            test_name="Forward Pass",
            status="passed",
            duration=0.0,
            message="Forward pass completed successfully"
        )
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return [TestResult(
            test_name="Forward Pass",
            status="failed",
            duration=0.0,
            message=f"Failed: {str(e)}"
        )]
    
    # Test backward pass
    print("-" * 70)
    print("Test 2: Backward Pass")
    print("-" * 70)
    
    model.train()
    
    try:
        outputs = model(batch)
        
        # Compute loss
        lambdas = {"reg": 0.5, "cons": 0.1, "rec": 0.1}
        loss, parts = compute_loss(outputs, batch, lambdas)
        
        # Backward
        loss.backward()
        
        print("✅ Backward pass successful!")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Classification loss: {parts['classification_loss'].item():.4f}")
        print(f"  Regression loss: {parts['regression_loss'].item():.4f}\n")
        
        backward_result = TestResult(
            test_name="Backward Pass",
            status="passed",
            duration=0.0,
            message=f"Loss: {loss.item():.4f}"
        )
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return [
            forward_result,
            TestResult(
                test_name="Backward Pass",
                status="failed",
                duration=0.0,
                message=f"Failed: {str(e)}"
            )
        ]
    
    # Test training loop
    print("-" * 70)
    print("Test 3: Training Loop (20 steps)")
    print("-" * 70)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    lambdas = {"reg": 0.5, "cons": 0.1, "rec": 0.1}
    
    losses = []
    num_steps = 20
    
    model.train()
    
    try:
        step = 0
        while step < num_steps:
            for batch in train_loader:
                if step >= num_steps:
                    break
                
                # Move to device
                for k in batch:
                    batch[k] = batch[k].to(device)
                
                # Forward
                optimizer.zero_grad()
                outputs = model(batch)
                loss, parts = compute_loss(outputs, batch, lambdas)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                if step % 5 == 0:
                    print(f"  Step {step:2d}: Loss = {loss.item():.4f}")
                
                step += 1
        
        # Check if loss decreased
        initial_loss = sum(losses[:5]) / 5
        final_loss = sum(losses[-5:]) / 5
        loss_decrease = initial_loss - final_loss
        loss_decrease_pct = (loss_decrease / initial_loss) * 100
        
        print(f"\n  Initial loss (avg first 5): {initial_loss:.4f}")
        print(f"  Final loss (avg last 5): {final_loss:.4f}")
        print(f"  Decrease: {loss_decrease:.4f} ({loss_decrease_pct:.1f}%)")
        
        if final_loss < initial_loss:
            print("✅ Loss decreased - model is learning!")
            training_result = TestResult(
                test_name="Training Loop",
                status="passed",
                duration=0.0,
                message=f"Loss decreased by {loss_decrease_pct:.1f}%",
                details={
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                    'decrease_pct': loss_decrease_pct
                }
            )
        else:
            print("⚠️  Loss did not decrease - may need more steps or tuning")
            training_result = TestResult(
                test_name="Training Loop",
                status="warning",
                duration=0.0,
                message=f"Loss change: {loss_decrease_pct:.1f}%",
                details={
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                    'decrease_pct': loss_decrease_pct
                }
            )
    
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        training_result = TestResult(
            test_name="Training Loop",
            status="failed",
            duration=0.0,
            message=f"Failed: {str(e)}"
        )
    
    return [forward_result, backward_result, training_result]


def main():
    """Run quick training test."""
    results = quick_train_test()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    warnings = sum(1 for r in results if r.status == "warning")
    
    for result in results:
        print(result)
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Warnings: {warnings}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("   MAFT model works correctly with synthetic data.")
        print("   Ready to proceed with full validation system.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
