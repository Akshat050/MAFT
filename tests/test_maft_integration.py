#!/usr/bin/env python3
"""
Quick integration test to verify validation system works with MAFT model.
"""

import sys
import torch
import time
from datetime import datetime

# Import validation system
from validation_system import TestResult, ValidationReport, ResourceStats, ModelResults

# Import MAFT components
from models.maft import MAFT
from tests.synthetic_loader import make_batch


def test_maft_with_validation_system():
    """Test MAFT model and wrap results in validation system classes."""
    print("=" * 70)
    print("MAFT + VALIDATION SYSTEM INTEGRATION TEST")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Test started at: {datetime.now()}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    test_results = []
    
    # Test 1: Model Creation
    print("-" * 70)
    print("Test 1: Model Creation")
    print("-" * 70)
    start_time = time.time()
    
    try:
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
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        duration = time.time() - start_time
        
        result = TestResult(
            test_name="Model Creation",
            status="passed",
            duration=duration,
            message=f"Model created successfully with {total_params:,} parameters",
            details={
                "total_params": total_params,
                "trainable_params": trainable_params,
                "device": str(device)
            }
        )
        test_results.append(result)
        print(result)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        duration = time.time() - start_time
        result = TestResult(
            test_name="Model Creation",
            status="failed",
            duration=duration,
            message=f"Model creation failed: {str(e)}",
            details={"error": str(e)}
        )
        test_results.append(result)
        print(result)
        return test_results
    
    # Test 2: Forward Pass
    print("\n" + "-" * 70)
    print("Test 2: Forward Pass")
    print("-" * 70)
    start_time = time.time()
    
    try:
        batch = make_batch(B=2, Lt=8, La=12, Lv=10, Da=74, Dv=35, C=2, device=str(device))
        
        with torch.no_grad():
            outputs = model(batch)
        
        duration = time.time() - start_time
        
        # Verify outputs
        assert "logits" in outputs, "Missing 'logits' in outputs"
        assert "reg" in outputs, "Missing 'reg' in outputs"
        assert outputs["logits"].shape == (2, 2), f"Wrong logits shape: {outputs['logits'].shape}"
        assert outputs["reg"].shape == (2, 1), f"Wrong reg shape: {outputs['reg'].shape}"
        
        result = TestResult(
            test_name="Forward Pass",
            status="passed",
            duration=duration,
            message="Forward pass completed successfully",
            details={
                "batch_size": 2,
                "logits_shape": str(outputs["logits"].shape),
                "reg_shape": str(outputs["reg"].shape),
                "output_keys": list(outputs.keys())
            }
        )
        test_results.append(result)
        print(result)
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Regression shape: {outputs['reg'].shape}")
        
    except Exception as e:
        duration = time.time() - start_time
        result = TestResult(
            test_name="Forward Pass",
            status="failed",
            duration=duration,
            message=f"Forward pass failed: {str(e)}",
            details={"error": str(e)}
        )
        test_results.append(result)
        print(result)
        import traceback
        traceback.print_exc()
    
    # Test 3: Backward Pass
    print("\n" + "-" * 70)
    print("Test 3: Backward Pass")
    print("-" * 70)
    start_time = time.time()
    
    try:
        model.train()
        batch = make_batch(B=2, Lt=8, La=12, Lv=10, Da=74, Dv=35, C=2, device=str(device))
        
        outputs = model(batch)
        
        # Simple loss computation
        loss = torch.nn.functional.cross_entropy(
            outputs["logits"], 
            batch["classification_targets"]
        )
        
        loss.backward()
        
        # Check gradients
        has_gradients = all(
            p.grad is not None 
            for p in model.parameters() 
            if p.requires_grad
        )
        
        duration = time.time() - start_time
        
        if has_gradients:
            result = TestResult(
                test_name="Backward Pass",
                status="passed",
                duration=duration,
                message="Backward pass completed, all parameters have gradients",
                details={
                    "loss_value": loss.item(),
                    "has_gradients": has_gradients
                }
            )
        else:
            result = TestResult(
                test_name="Backward Pass",
                status="warning",
                duration=duration,
                message="Backward pass completed but some parameters missing gradients",
                details={
                    "loss_value": loss.item(),
                    "has_gradients": has_gradients
                }
            )
        
        test_results.append(result)
        print(result)
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  All parameters have gradients: {has_gradients}")
        
    except Exception as e:
        duration = time.time() - start_time
        result = TestResult(
            test_name="Backward Pass",
            status="failed",
            duration=duration,
            message=f"Backward pass failed: {str(e)}",
            details={"error": str(e)}
        )
        test_results.append(result)
        print(result)
        import traceback
        traceback.print_exc()
    
    return test_results


def main():
    """Run integration test and generate report."""
    test_results = test_maft_with_validation_system()
    
    # Create resource stats (simplified for now)
    import psutil
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    resource_stats = ResourceStats(
        cpu_cores=psutil.cpu_count(),
        cpu_usage_avg=psutil.cpu_percent(interval=0.1),
        cpu_usage_peak=psutil.cpu_percent(interval=0.1),
        memory_total_gb=memory.total / (1024**3),
        memory_available_gb=memory.available / (1024**3),
        memory_peak_usage_gb=memory.used / (1024**3),
        disk_total_gb=disk.total / (1024**3),
        disk_free_gb=disk.free / (1024**3),
        gpu_available=torch.cuda.is_available(),
        gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else None
    )
    
    # Count passed/failed tests
    passed = sum(1 for r in test_results if r.status == "passed")
    failed = sum(1 for r in test_results if r.status == "failed")
    warnings = sum(1 for r in test_results if r.status == "warning")
    total_duration = sum(r.duration for r in test_results)
    
    # Determine overall status
    if failed > 0:
        overall_status = "failed"
        deployment_ready = False
    elif warnings > 0:
        overall_status = "warning"
        deployment_ready = True
    else:
        overall_status = "passed"
        deployment_ready = True
    
    # Create recommendations
    recommendations = []
    if deployment_ready:
        recommendations.append("✅ Basic MAFT functionality verified")
        recommendations.append("✅ Model can perform forward and backward passes")
        recommendations.append("Ready to proceed with full validation suite")
    else:
        recommendations.append("❌ Fix failed tests before proceeding")
        recommendations.append("Check error messages above for details")
    
    # Create validation report
    report = ValidationReport(
        overall_status=overall_status,
        total_tests=len(test_results),
        passed_tests=passed,
        failed_tests=failed,
        warnings=warnings,
        total_duration=total_duration,
        test_results=test_results,
        resource_stats=resource_stats,
        deployment_ready=deployment_ready,
        recommendations=recommendations
    )
    
    # Print report
    print("\n" + str(report))
    
    return 0 if deployment_ready else 1


if __name__ == "__main__":
    sys.exit(main())
