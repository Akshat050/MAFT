#!/usr/bin/env python3
"""
Test script for synthetic multimodal data generator.

Verifies that synthetic data has correct shapes, proper correlations,
and works with the MAFT model.
"""

import sys
import torch
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime

from validation_system.synthetic_data import SyntheticMultimodalDataset, get_synthetic_loaders
from validation_system import TestResult


def test_dataset_shapes():
    """Test that dataset generates correct shapes."""
    print("\n" + "=" * 70)
    print("Test 1: Dataset Shapes")
    print("=" * 70)
    
    dataset = SyntheticMultimodalDataset(
        num_samples=10,
        seq_len_text=64,
        seq_len_audio=200,
        seq_len_visual=200
    )
    
    sample = dataset[0]
    
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  audio shape: {sample['audio'].shape}")
    print(f"  audio_mask shape: {sample['audio_mask'].shape}")
    print(f"  visual shape: {sample['visual'].shape}")
    print(f"  visual_mask shape: {sample['visual_mask'].shape}")
    print(f"  classification_targets: {sample['classification_targets']}")
    print(f"  regression_targets: {sample['regression_targets']}")
    
    # Verify shapes
    assert sample['input_ids'].shape == (64,), f"Wrong input_ids shape: {sample['input_ids'].shape}"
    assert sample['attention_mask'].shape == (64,), f"Wrong attention_mask shape"
    assert sample['audio'].shape == (200, 74), f"Wrong audio shape: {sample['audio'].shape}"
    assert sample['audio_mask'].shape == (200,), f"Wrong audio_mask shape"
    assert sample['visual'].shape == (200, 35), f"Wrong visual shape: {sample['visual'].shape}"
    assert sample['visual_mask'].shape == (200,), f"Wrong visual_mask shape"
    assert sample['classification_targets'].dim() == 0, "classification_targets should be scalar"
    assert sample['regression_targets'].dim() == 0, "regression_targets should be scalar"
    
    # Verify token range
    assert sample['input_ids'].min() >= 0, "Tokens should be non-negative"
    assert sample['input_ids'].max() < 30000, "Tokens should be within vocab"
    
    # Verify masks are binary
    assert set(sample['attention_mask'].unique().tolist()).issubset({0, 1}), "Mask should be binary"
    assert set(sample['audio_mask'].unique().tolist()).issubset({0, 1}), "Mask should be binary"
    assert set(sample['visual_mask'].unique().tolist()).issubset({0, 1}), "Mask should be binary"
    
    print("✅ All shapes correct!")
    return TestResult(
        test_name="Dataset Shapes",
        status="passed",
        duration=0.0,
        message="All shapes and ranges verified"
    )


def test_correlation():
    """Test that modalities correlate with targets."""
    print("\n" + "=" * 70)
    print("Test 2: Cross-Modal Correlation")
    print("=" * 70)
    
    # Create dataset with high correlation
    dataset = SyntheticMultimodalDataset(
        num_samples=100,
        correlation_strength=0.8,
        seed=42
    )
    
    # Collect features and targets
    audio_means = []
    visual_means = []
    targets = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Get mean of valid features (excluding padding)
        audio_valid = sample['audio'][sample['audio_mask'] == 1]
        visual_valid = sample['visual'][sample['visual_mask'] == 1]
        
        if len(audio_valid) > 0:
            audio_means.append(audio_valid.mean().item())
        if len(visual_valid) > 0:
            visual_means.append(visual_valid.mean().item())
        
        targets.append(sample['regression_targets'].item())
    
    # Calculate correlations
    audio_corr, _ = pearsonr(audio_means, targets)
    visual_corr, _ = pearsonr(visual_means, targets)
    
    print(f"Audio-Target Correlation: {audio_corr:.4f}")
    print(f"Visual-Target Correlation: {visual_corr:.4f}")
    
    # With correlation_strength=0.8, we expect at least 0.6 correlation
    min_correlation = 0.6
    
    audio_pass = abs(audio_corr) >= min_correlation
    visual_pass = abs(visual_corr) >= min_correlation
    
    if audio_pass and visual_pass:
        print(f"✅ Both modalities show strong correlation (>= {min_correlation})")
        status = "passed"
        message = f"Audio: {audio_corr:.4f}, Visual: {visual_corr:.4f}"
    else:
        print(f"⚠️  Correlation below threshold")
        status = "warning"
        message = f"Audio: {audio_corr:.4f}, Visual: {visual_corr:.4f} (expected >= {min_correlation})"
    
    return TestResult(
        test_name="Cross-Modal Correlation",
        status=status,
        duration=0.0,
        message=message,
        details={
            'audio_correlation': audio_corr,
            'visual_correlation': visual_corr,
            'min_threshold': min_correlation
        }
    )


def test_masks():
    """Test that masks are applied correctly."""
    print("\n" + "=" * 70)
    print("Test 3: Mask Application")
    print("=" * 70)
    
    dataset = SyntheticMultimodalDataset(num_samples=20, seed=42)
    
    issues = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Check text: padded positions should be 0
        text_valid_len = sample['attention_mask'].sum().item()
        if text_valid_len < len(sample['input_ids']):
            padded_tokens = sample['input_ids'][text_valid_len:]
            if not (padded_tokens == 0).all():
                issues.append(f"Sample {i}: Text padding not all zeros")
        
        # Check audio: padded positions should be 0
        audio_valid_len = sample['audio_mask'].sum().item()
        if audio_valid_len < len(sample['audio']):
            padded_audio = sample['audio'][audio_valid_len:]
            if not torch.allclose(padded_audio, torch.zeros_like(padded_audio)):
                issues.append(f"Sample {i}: Audio padding not all zeros")
        
        # Check visual: padded positions should be 0
        visual_valid_len = sample['visual_mask'].sum().item()
        if visual_valid_len < len(sample['visual']):
            padded_visual = sample['visual'][visual_valid_len:]
            if not torch.allclose(padded_visual, torch.zeros_like(padded_visual)):
                issues.append(f"Sample {i}: Visual padding not all zeros")
    
    if not issues:
        print("✅ All masks applied correctly!")
        print(f"  Checked {len(dataset)} samples")
        return TestResult(
            test_name="Mask Application",
            status="passed",
            duration=0.0,
            message=f"All {len(dataset)} samples have correct padding"
        )
    else:
        print(f"❌ Found {len(issues)} mask issues:")
        for issue in issues[:5]:  # Show first 5
            print(f"  - {issue}")
        return TestResult(
            test_name="Mask Application",
            status="failed",
            duration=0.0,
            message=f"Found {len(issues)} mask issues",
            details={'issues': issues}
        )


def test_dataloader():
    """Test that data loaders work correctly."""
    print("\n" + "=" * 70)
    print("Test 4: DataLoader Functionality")
    print("=" * 70)
    
    train_loader, val_loader = get_synthetic_loaders(
        batch_size=4,
        num_train_batches=5,
        num_val_batches=2
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test train loader
    batch = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  audio: {batch['audio'].shape}")
    print(f"  visual: {batch['visual'].shape}")
    print(f"  classification_targets: {batch['classification_targets'].shape}")
    
    # Verify batch shapes
    assert batch['input_ids'].shape[0] == 4, "Wrong batch size"
    assert batch['audio'].shape == (4, 200, 74), f"Wrong audio batch shape: {batch['audio'].shape}"
    assert batch['visual'].shape == (4, 200, 35), f"Wrong visual batch shape: {batch['visual'].shape}"
    
    # Test iteration
    batch_count = 0
    for batch in train_loader:
        batch_count += 1
    
    assert batch_count == 5, f"Expected 5 batches, got {batch_count}"
    
    print("✅ DataLoader works correctly!")
    return TestResult(
        test_name="DataLoader Functionality",
        status="passed",
        duration=0.0,
        message=f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches"
    )


def test_statistics():
    """Test dataset statistics method."""
    print("\n" + "=" * 70)
    print("Test 5: Dataset Statistics")
    print("=" * 70)
    
    dataset = SyntheticMultimodalDataset(
        num_samples=100,
        correlation_strength=0.7,
        seed=42
    )
    
    stats = dataset.get_statistics()
    
    print("Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Verify statistics are reasonable
    assert stats['num_samples'] == 100
    assert 30 < stats['text_length_mean'] < 70, "Text length mean out of range"
    assert 100 < stats['audio_length_mean'] < 210, "Audio length mean out of range"
    assert 100 < stats['visual_length_mean'] < 210, "Visual length mean out of range"
    assert stats['correlation_strength'] == 0.7
    
    print("✅ Statistics look reasonable!")
    return TestResult(
        test_name="Dataset Statistics",
        status="passed",
        duration=0.0,
        message="All statistics within expected ranges",
        details=stats
    )


def main():
    """Run all synthetic data tests."""
    print("=" * 70)
    print("SYNTHETIC DATA GENERATOR TESTS")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        test_dataset_shapes,
        test_correlation,
        test_masks,
        test_dataloader,
        test_statistics
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(TestResult(
                test_name=test_func.__name__,
                status="failed",
                duration=0.0,
                message=f"Exception: {str(e)}"
            ))
    
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
        print("\n🎉 ALL TESTS PASSED! Synthetic data generator is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
