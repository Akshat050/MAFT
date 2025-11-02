---
name: Validation Issue
about: Report issues with the validation system
title: '[VALIDATION] '
labels: validation
assignees: ''

---

**Validation Component**
Which part of the validation system is having issues?
- [ ] Synthetic data generation
- [ ] Component tests
- [ ] Pipeline tests
- [ ] Resource monitoring
- [ ] Report generation
- [ ] Other: ___________

**Test Results**
Please run all validation tests and report results:

```bash
# Base classes test
python test_validation_base.py
# Result: ___/5 passed

# Synthetic data test  
python test_synthetic_data.py
# Result: ___/5 passed

# Integration test
python test_maft_integration.py
# Result: ___/3 passed

# Quick training test
python test_maft_quick_train.py
# Result: ___/3 passed
```

**Error Details**
```
Paste the full error output here
```

**Configuration**
```python
# If using custom parameters, paste them here
dataset = SyntheticMultimodalDataset(
    num_samples=100,
    correlation_strength=0.7,
    # ... other parameters
)
```

**Expected vs Actual**
- **Expected:** What should happen
- **Actual:** What actually happens

**System Information**
- OS: [e.g. Windows 11, Ubuntu 20.04]
- Python: [e.g. 3.11.9]
- PyTorch: [e.g. 2.9.0+cpu]
- Memory: [e.g. 16GB]
- CPU: [e.g. 8 cores]

**Additional Context**
Any other information that might help diagnose the issue.