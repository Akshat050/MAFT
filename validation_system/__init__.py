"""
MAFT Local Validation System

A comprehensive testing framework for validating the MAFT model
before cloud deployment.
"""

from .data_models import TestResult, ValidationReport, ResourceStats, ModelResults
from .utils import format_duration, format_bytes, setup_logging

__version__ = "1.0.0"
__all__ = [
    "TestResult",
    "ValidationReport", 
    "ResourceStats",
    "ModelResults",
    "format_duration",
    "format_bytes",
    "setup_logging",
]
