# MAFT Project Structure (Cleaned)

**Date:** October 25, 2025  
**Status:** Cleaned and organized for local validation

## Directory Structure

```
MAFT/
├── .kiro/                          # Kiro IDE specs and settings
│   └── specs/
│       └── local-validation-system/
│           ├── requirements.md     # Validation requirements
│           ├── design.md          # Validation design
│           └── tasks.md           # Implementation tasks
│
├── configs/                        # Configuration files
│   ├── cpu_test_config.yaml       # CPU testing config
│   ├── interview_config.yaml      # Interview dataset config
│   └── mosei_config.yaml          # MOSEI dataset config
│
├── losses/                         # Loss functions
│   └── consistency.py             # Consistency loss (symmetric KL)
│
├── models/                         # Model implementations
│   ├── encoders.py                # Text/Audio/Visual encoders
│   ├── fusion.py                  # Fusion transformer
│   ├── maft.py                    # Main MAFT model
│   └── quality.py                 # Quality estimator
│
├── scripts/                        # Utility scripts
│   ├── analyze_attention.py       # Attention analysis
│   ├── efficiency_analysis.py     # Efficiency benchmarking
│   ├── generate_results_table.py  # Results table generation
│   ├── prepare_interview.py       # Interview data prep
│   ├── prepare_mosei.py           # MOSEI data prep
│   ├── run_ablations.py           # Ablation studies
│   ├── run_baselines.py           # Baseline comparisons
│   └── run_experiments.py         # Experiment runner
│
├── tests/                          # Test utilities
│   ├── __init__.py
│   ├── synthetic_loader.py        # Simple synthetic data (legacy)
│   └── test_forward.py            # Basic forward pass test
│
├── validation_system/              # NEW: Validation framework
│   ├── __init__.py
│   ├── data_models.py             # Test result data structures
│   ├── synthetic_data.py          # Synthetic data generator
│   └── utils.py                   # Utility functions
│
├── evaluate.py                     # Model evaluation script
├── train.py                        # Training script
│
├── test_validation_base.py         # NEW: Base classes test
├── test_synthetic_data.py          # NEW: Synthetic data test
├── test_maft_integration.py        # NEW: Integration test
├── test_maft_quick_train.py        # NEW: Quick training test
│
├── LICENSE                         # MIT License
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── VALIDATION_STATUS.md            # NEW: Validation progress
└── TASK2_SUMMARY.md               # NEW: Task 2 summary
```

## Core Components

### Models (`models/`)
- **encoders.py**: Text (embedding-based), Audio (BiLSTM), Visual (BiLSTM) encoders
- **fusion.py**: Fusion transformer with cross-modal attention
- **maft.py**: Main MAFT model integrating all components
- **quality.py**: Per-token quality estimation

### Validation System (`validation_system/`)
- **data_models.py**: TestResult, ValidationReport, ResourceStats, ModelResults
- **synthetic_data.py**: SyntheticMultimodalDataset with controllable correlations
- **utils.py**: Timing, formatting, logging utilities

### Training & Evaluation
- **train.py**: Main training script with loss computation
- **evaluate.py**: Model evaluation with metrics

### Test Scripts
- **test_validation_base.py**: Tests validation system base classes
- **test_synthetic_data.py**: Tests synthetic data generator
- **test_maft_integration.py**: Tests MAFT with validation system
- **test_maft_quick_train.py**: Quick training validation (20 steps)

### Configuration Files (`configs/`)
- **cpu_test_config.yaml**: Optimized for CPU testing
- **mosei_config.yaml**: CMU-MOSEI dataset configuration
- **interview_config.yaml**: Interview dataset configuration

## Removed Files (Cleanup)

### GCP/Cloud Related
- ❌ `deploy_maft_gcp.py`
- ❌ `deploy_maft_gcp_auto.py`
- ❌ `run_maft_gcp.py`
- ❌ `setup_gcp.py`
- ❌ `GCP_DEPLOYMENT.md`
- ❌ `GCP_QUICK_START.md`
- ❌ `scripts/deploy_gcp.py`
- ❌ `scripts/gcp_full_setup.py`
- ❌ `scripts/gcp_startup.sh`
- ❌ `scripts/quick_deploy.sh`
- ❌ `scripts/download_and_upload_dataset.py`

### Docker Related
- ❌ `Dockerfile`
- ❌ `docker-compose.yml`

### Demo Files
- ❌ `demo.py`
- ❌ `demo_for_professor.py`
- ❌ `professor_demo.py`
- ❌ `quick_demo.py`
- ❌ `impressive_demo.py`

### Old Test Files
- ❌ `test_maft.py`
- ❌ `test_maft_minimal.py`
- ❌ `test_maft_comparison.py`
- ❌ `run_cpu_tests.py`
- ❌ `train_sanity.py`
- ❌ `CPU_TEST_README.md`

### Documentation
- ❌ `CONTRIBUTING.md`
- ❌ `PAPER_CHECKLIST.md`
- ❌ `REPRODUCIBILITY.md` (info moved to README)

## What's Left (Essential Files Only)

### Core Implementation ✅
- MAFT model and components
- Training and evaluation scripts
- Loss functions

### Validation System ✅
- Synthetic data generator
- Test framework
- Validation utilities

### Configuration ✅
- Dataset configs
- Training configs

### Scripts ✅
- Data preparation
- Analysis tools
- Experiment runners

### Documentation ✅
- README.md (main documentation)
- LICENSE
- VALIDATION_STATUS.md (progress tracking)
- TASK2_SUMMARY.md (task documentation)

## Quick Start

### Run Validation Tests
```bash
# Test base classes
python test_validation_base.py

# Test synthetic data
python test_synthetic_data.py

# Test MAFT integration
python test_maft_integration.py

# Quick training test
python test_maft_quick_train.py
```

### Use Synthetic Data
```python
from validation_system.synthetic_data import get_synthetic_loaders

train_loader, val_loader = get_synthetic_loaders(
    batch_size=8,
    num_train_batches=10
)
```

### Train MAFT
```bash
# With real data (when available)
python train.py --config configs/mosei_config.yaml

# With synthetic data (for testing)
python test_maft_quick_train.py
```

## Dependencies

Install required packages:
```bash
pip install torch transformers psutil scipy numpy
```

## Next Steps

1. ✅ **Task 1 Complete**: Base classes and utilities
2. ✅ **Task 2 Complete**: Synthetic data generator
3. ⏳ **Task 3**: Component test suite
4. ⏳ **Task 4**: Pipeline test suite
5. ⏳ **Task 11**: Validation orchestrator

## Notes

- All GCP/cloud deployment code removed
- All demo files removed
- Old test files replaced with new validation system
- Project now focused on local validation before cloud deployment
- Clean, minimal structure for development and testing
