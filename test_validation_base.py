#!/usr/bin/env python3
"""
Test script to verify the validation system base classes work correctly.
"""

import sys
import time
from datetime import datetime

# Test imports
try:
    from validation_system import (
        TestResult,
        ValidationReport,
        ResourceStats,
        ModelResults,
        format_duration,
        format_bytes,
        setup_logging
    )
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_test_result():
    """Test TestResult class."""
    print("\n" + "=" * 60)
    print("Testing TestResult class")
    print("=" * 60)
    
    # Create a test result
    result = TestResult(
        test_name="Sample Test",
        status="passed",
        duration=1.234,
        message="Test completed successfully",
        details={"accuracy": 0.95, "loss": 0.05}
    )
    
    print(f"Created TestResult: {result}")
    print(f"  Test name: {result.test_name}")
    print(f"  Status: {result.status}")
    print(f"  Duration: {result.duration}s")
    print(f"  Message: {result.message}")
    print(f"  Details: {result.details}")
    print(f"  Timestamp: {result.timestamp}")
    
    assert result.test_name == "Sample Test"
    assert result.status == "passed"
    assert result.duration == 1.234
    print("✅ TestResult works correctly!")
    return True


def test_resource_stats():
    """Test ResourceStats class."""
    print("\n" + "=" * 60)
    print("Testing ResourceStats class")
    print("=" * 60)
    
    # Create resource stats
    stats = ResourceStats(
        cpu_cores=8,
        cpu_usage_avg=45.5,
        cpu_usage_peak=78.2,
        memory_total_gb=16.0,
        memory_available_gb=8.5,
        memory_peak_usage_gb=12.3,
        disk_total_gb=500.0,
        disk_free_gb=250.0,
        gpu_available=False
    )
    
    print(f"Created ResourceStats:\n{stats}")
    
    assert stats.cpu_cores == 8
    assert stats.memory_total_gb == 16.0
    assert not stats.gpu_available
    print("✅ ResourceStats works correctly!")
    return True


def test_model_results():
    """Test ModelResults class."""
    print("\n" + "=" * 60)
    print("Testing ModelResults class")
    print("=" * 60)
    
    # Create model results
    results = ModelResults(
        model_name="MAFT",
        accuracy=0.856,
        f1_score=0.854,
        mae=0.598,
        pearson_r=0.823,
        training_time=120.5,
        num_parameters=85000000,
        memory_usage_mb=2048.0
    )
    
    print(f"Created ModelResults:\n{results}")
    
    assert results.model_name == "MAFT"
    assert results.accuracy == 0.856
    assert results.num_parameters == 85000000
    print("✅ ModelResults works correctly!")
    return True


def test_validation_report():
    """Test ValidationReport class."""
    print("\n" + "=" * 60)
    print("Testing ValidationReport class")
    print("=" * 60)
    
    # Create some test results
    test_results = [
        TestResult("Test 1", "passed", 1.0, "Success"),
        TestResult("Test 2", "passed", 2.0, "Success"),
        TestResult("Test 3", "failed", 0.5, "Failed"),
    ]
    
    # Create resource stats
    stats = ResourceStats(
        cpu_cores=4,
        cpu_usage_avg=50.0,
        cpu_usage_peak=80.0,
        memory_total_gb=8.0,
        memory_available_gb=4.0,
        memory_peak_usage_gb=6.0,
        disk_total_gb=256.0,
        disk_free_gb=128.0
    )
    
    # Create validation report
    report = ValidationReport(
        overall_status="warning",
        total_tests=3,
        passed_tests=2,
        failed_tests=1,
        warnings=0,
        total_duration=3.5,
        test_results=test_results,
        resource_stats=stats,
        deployment_ready=False,
        recommendations=[
            "Fix failed test before deployment",
            "Consider increasing memory for cloud instance"
        ]
    )
    
    print(f"Created ValidationReport:\n{report}")
    
    assert report.total_tests == 3
    assert report.passed_tests == 2
    assert report.failed_tests == 1
    assert not report.deployment_ready
    assert len(report.recommendations) == 2
    print("✅ ValidationReport works correctly!")
    return True


def test_utility_functions():
    """Test utility functions."""
    print("\n" + "=" * 60)
    print("Testing utility functions")
    print("=" * 60)
    
    # Test format_duration
    print("\nTesting format_duration:")
    print(f"  0.5s -> {format_duration(0.5)}")
    print(f"  5.0s -> {format_duration(5.0)}")
    print(f"  65.0s -> {format_duration(65.0)}")
    print(f"  3665.0s -> {format_duration(3665.0)}")
    
    assert "ms" in format_duration(0.5)
    assert "s" in format_duration(5.0)
    assert "m" in format_duration(65.0)
    assert "h" in format_duration(3665.0)
    
    # Test format_bytes
    print("\nTesting format_bytes:")
    print(f"  512 bytes -> {format_bytes(512)}")
    print(f"  1024 bytes -> {format_bytes(1024)}")
    print(f"  1048576 bytes -> {format_bytes(1048576)}")
    print(f"  1073741824 bytes -> {format_bytes(1073741824)}")
    
    assert "B" in format_bytes(512)
    assert "KB" in format_bytes(1024)
    assert "MB" in format_bytes(1048576)
    assert "GB" in format_bytes(1073741824)
    
    # Test setup_logging
    print("\nTesting setup_logging:")
    logger = setup_logging(verbose=False)
    logger.info("Test log message")
    print("  Logger created successfully")
    
    print("✅ Utility functions work correctly!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("VALIDATION SYSTEM BASE CLASSES TEST")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        ("TestResult", test_test_result),
        ("ResourceStats", test_resource_stats),
        ("ModelResults", test_model_results),
        ("ValidationReport", test_validation_report),
        ("Utility Functions", test_utility_functions),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Base classes are working correctly.")
        print("   You can now proceed with implementing the rest of the validation system.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
