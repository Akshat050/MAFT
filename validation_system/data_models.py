"""
Data models for validation system results and reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class TestResult:
    """Result from a single test."""
    test_name: str
    status: str  # "passed", "failed", "warning"
    duration: float  # seconds
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        status_icon = {
            "passed": "✅",
            "failed": "❌",
            "warning": "⚠️"
        }.get(self.status, "❓")
        
        return f"{status_icon} {self.test_name} ({self.duration:.2f}s): {self.message}"


@dataclass
class ResourceStats:
    """System resource statistics."""
    cpu_cores: int
    cpu_usage_avg: float
    cpu_usage_peak: float
    memory_total_gb: float
    memory_available_gb: float
    memory_peak_usage_gb: float
    disk_total_gb: float
    disk_free_gb: float
    gpu_available: bool = False
    gpu_memory_gb: Optional[float] = None
    
    def __str__(self) -> str:
        lines = [
            f"CPU: {self.cpu_cores} cores, {self.cpu_usage_avg:.1f}% avg, {self.cpu_usage_peak:.1f}% peak",
            f"Memory: {self.memory_total_gb:.1f} GB total, {self.memory_available_gb:.1f} GB available, {self.memory_peak_usage_gb:.1f} GB peak",
            f"Disk: {self.disk_total_gb:.1f} GB total, {self.disk_free_gb:.1f} GB free",
        ]
        if self.gpu_available:
            lines.append(f"GPU: Available, {self.gpu_memory_gb:.1f} GB memory")
        else:
            lines.append("GPU: Not available")
        return "\n".join(lines)


@dataclass
class ModelResults:
    """Results from training/evaluating a model."""
    model_name: str
    accuracy: float
    f1_score: float
    mae: float
    pearson_r: float
    training_time: float
    num_parameters: int
    memory_usage_mb: float
    
    def __str__(self) -> str:
        return (
            f"{self.model_name}:\n"
            f"  Accuracy: {self.accuracy:.4f}, F1: {self.f1_score:.4f}\n"
            f"  MAE: {self.mae:.4f}, Pearson r: {self.pearson_r:.4f}\n"
            f"  Training time: {self.training_time:.2f}s\n"
            f"  Parameters: {self.num_parameters:,}\n"
            f"  Memory: {self.memory_usage_mb:.1f} MB"
        )


@dataclass
class ValidationReport:
    """Complete validation report."""
    overall_status: str  # "passed", "failed", "warning"
    total_tests: int
    passed_tests: int
    failed_tests: int
    warnings: int
    total_duration: float
    test_results: List[TestResult]
    resource_stats: ResourceStats
    deployment_ready: bool
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> str:
        """Get a summary string of the validation report."""
        status_icon = {
            "passed": "✅",
            "failed": "❌",
            "warning": "⚠️"
        }.get(self.overall_status, "❓")
        
        lines = [
            "=" * 70,
            "VALIDATION SUMMARY",
            "=" * 70,
            f"Overall Status: {status_icon} {self.overall_status.upper()}",
            f"Tests: {self.passed_tests}/{self.total_tests} passed",
            f"Failed: {self.failed_tests}, Warnings: {self.warnings}",
            f"Total Duration: {self.total_duration:.2f}s",
            f"Deployment Ready: {'Yes' if self.deployment_ready else 'No'}",
            "",
            "Resource Usage:",
            str(self.resource_stats),
        ]
        
        if self.recommendations:
            lines.extend([
                "",
                "Recommendations:",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.get_summary()
