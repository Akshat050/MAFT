"""
Utility functions for the validation system.
"""

import time
import logging
from typing import Callable, Any
from functools import wraps
from datetime import datetime


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_bytes(bytes_value: float) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('validation_system')
    return logger


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Store duration in result if it's a TestResult
        if hasattr(result, 'duration'):
            result.duration = duration
        
        return result
    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def aggregate_results(results: list, key: str) -> dict:
    """Aggregate results by extracting a specific key."""
    values = []
    for result in results:
        if hasattr(result, key):
            values.append(getattr(result, key))
        elif isinstance(result, dict) and key in result:
            values.append(result[key])
    
    if not values:
        return {}
    
    return {
        'mean': sum(values) / len(values),
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str, width: int = 70) -> None:
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)
