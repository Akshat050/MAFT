# Contributing to MAFT

Thank you for your interest in contributing to MAFT (Multimodal Attention Fusion Transformer)! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MAFT.git
   cd MAFT
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- Git

### Installation

1. **Install dependencies:**
   ```bash
   pip install torch transformers psutil scipy numpy
   ```

2. **Install development dependencies:**
   ```bash
   pip install flake8 black isort pytest
   ```

3. **Verify installation:**
   ```bash
   python -c "from models.maft import MAFT; print('✅ MAFT import successful')"
   ```

## Running Tests

Before submitting any changes, ensure all tests pass:

### Validation Tests (Required)

```bash
# Run all validation tests
python test_validation_base.py      # 5/5 tests should pass
python test_synthetic_data.py       # 5/5 tests should pass
python test_maft_integration.py     # 3/3 tests should pass
python test_maft_quick_train.py     # 3/3 tests should pass
```

### Quick Validation

For rapid testing during development:
```bash
# Quick training test (< 1 minute)
python test_maft_quick_train.py
```

### Test Requirements

- All existing tests must continue to pass
- New features should include appropriate tests
- Tests should be fast (< 5 minutes total)
- Use synthetic data for testing when possible

## Code Style

### Python Style Guidelines

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### Code Formatting

We use automated formatting tools:

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Check style with flake8
flake8 .
```

### Example Code Style

```python
def generate_synthetic_data(
    num_samples: int = 100,
    correlation_strength: float = 0.7,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic multimodal data for testing.
    
    Args:
        num_samples: Number of samples to generate
        correlation_strength: Cross-modal correlation (0-1)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing synthetic data tensors
    """
    torch.manual_seed(seed)
    # Implementation here...
    return data_dict
```

## Submitting Changes

### Before Submitting

1. **Run all tests:**
   ```bash
   python test_validation_base.py
   python test_synthetic_data.py
   python test_maft_integration.py
   python test_maft_quick_train.py
   ```

2. **Check code style:**
   ```bash
   flake8 .
   black --check .
   isort --check-only .
   ```

3. **Update documentation** if needed

### Pull Request Process

1. **Create a pull request** from your feature branch to `main`
2. **Fill out the PR template** completely
3. **Ensure CI tests pass** (GitHub Actions will run automatically)
4. **Respond to review feedback** promptly
5. **Squash commits** if requested

### Commit Message Format

Use clear, descriptive commit messages:

```
feat: Add synthetic data correlation mechanism
fix: Resolve quality estimator mask handling
docs: Update validation system documentation
test: Add component test for fusion module
refactor: Simplify audio encoder implementation
```

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements

## Reporting Issues

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Run validation tests** to isolate the problem
3. **Gather system information** (OS, Python version, etc.)

### Issue Types

Use the appropriate issue template:

- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting new features
- **Validation Issue**: For validation system problems

### Good Issue Reports Include

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Error messages (full output)
- System information
- Validation test results

## Development Guidelines

### Adding New Features

1. **Start with tests** - Write tests first when possible
2. **Use synthetic data** - For testing and validation
3. **Document thoroughly** - Add docstrings and comments
4. **Keep it simple** - Prefer simple, readable solutions
5. **Validate thoroughly** - Ensure all tests pass

### Modifying Existing Code

1. **Understand the impact** - Run all tests
2. **Maintain backward compatibility** when possible
3. **Update tests** if behavior changes
4. **Update documentation** if interfaces change

### Working with the Validation System

When adding new components:

1. **Add to validation system** - Create appropriate tests
2. **Use existing patterns** - Follow established test patterns
3. **Test with synthetic data** - Ensure it works with generated data
4. **Document usage** - Add examples and documentation

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Submit PRs for feedback

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Project documentation

Thank you for contributing to MAFT! 🚀