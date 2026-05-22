# MAFT

A PyTorch implementation of a multimodal transformer for sentiment analysis on text, audio, and visual inputs. This is **ongoing empirical work**, not a completed paper. The repository contains the model, training pipeline, and a validation framework I use to test changes before committing real-data runs.

## Status

- **Architecture:** implemented and exercised end-to-end on synthetic data
- **Validation framework:** synthetic multimodal dataset with controllable cross-modal correlation; all forward/backward/learning-dynamics tests pass
- **Real-data training (CMU-MOSEI):** partial runs completed with GloVe/COVAREP/FACET features; full benchmark runs paused pending access to GPU compute
- **Benchmark numbers vs. published baselines:** not yet established. Earlier versions of this README quoted target numbers from the literature as if achieved here. They were not. I have removed those claims.

## What this project is trying to study

Rather than chase state-of-the-art on a saturated benchmark, I am using this codebase to study questions that bear on the *reliability* of multimodal transformers:

- How does **scheduled modality dropout** affect robustness when one modality is corrupted or absent at inference time?
- Does the **symmetric-KL consistency loss** between per-modality auxiliary heads actually align modality representations, or does it function as a generic regularizer?
- Can the model be trained to **defer or refuse** when modalities disagree, rather than confidently fuse contradictory signals?
- How well do conclusions drawn on **synthetic data with controllable correlations** transfer to real CMU-MOSEI behavior?

These are the questions I'd like to answer once I have stable benchmark runs. The roadmap in `ROADMAP.md` outlines specific experiments.

## Architecture

MAFT processes text, audio, and visual sequences through modality-specific encoders, then fuses them with a single shared transformer.

```text
text  -> embedding + linear projection --\
audio -> linear + BiLSTM -----------------+-> concat + modality embeddings
visual -> linear + BiLSTM ----------------/       |
                                                  v
                                      [bottleneck tokens] + [tokens]
                                                  |
                                                  v
                                      shared TransformerEncoder
                                                  |
                               +------------------+------------------+
                               v                  v                  v
                         classification       regression        per-modality
                         + temperature        head              auxiliary heads
                                                              (consistency loss)
```

Key design choices:

- **Modality-aware embeddings** added to each token so the shared transformer can distinguish text/audio/visual positions.
- **Bottleneck tokens** that attend to all modalities and produce a gated, pooled representation used by the prediction heads. Conceptually related to the bottlenecks in MBT (Nagrani et al., 2021).
- **Scheduled modality dropout** that anneals from a low rate at the start of training to a higher rate later, randomly zeroing entire modalities for some samples to encourage robust representations.
- **Symmetric-KL consistency loss** between per-modality auxiliary classifier heads, encouraging modality views to agree on the prediction.
- **Quality-estimator masks** that distinguish valid tokens from padding throughout the model.

## Repository Layout

```text
models/
  encoders.py       text / audio (BiLSTM) / visual (BiLSTM) encoders
  fusion.py         shared transformer with modality embeddings + bottlenecks
  maft.py           top-level model wiring encoders + fusion + heads
losses/
  consistency.py    symmetric-KL multi-view consistency loss
validation_system/
  data_models.py    TestResult / ValidationReport dataclasses
  synthetic_data.py controllable synthetic multimodal dataset
  utils.py          formatting, timing, logging helpers
configs/            YAML configs for CPU, M-series, interview, and MOSEI runs
scripts/            data preparation, analysis, baseline, and ablation scripts
scripts/dev/        local diagnostics and one-off development utilities
train.py            training loop with AMP, LR schedule, early stopping
evaluate.py         evaluation with classification + regression metrics
mosei_dataloader.py CMU-MOSEI loader
tests/              forward-pass, architecture, synthetic-data, and training tests
```

## Setup

Tested on Python 3.11 with PyTorch 2.x.

```bash
pip install -r requirements.txt
```

For CMU-MOSEI data preparation you additionally need the CMU Multimodal SDK; see `prepare_mosei_benchmark.py` and `scripts/prepare_mosei.py` for usage.

## Running

Synthetic-data smoke tests, no real data needed:

```bash
pytest tests/test_validation_base.py
pytest tests/test_synthetic_data.py
pytest tests/test_maft_quick_train.py
```

These should complete quickly on CPU and confirm the model trains and the loss decreases.

CMU-MOSEI training, requiring prepared data in `data/mosei/`:

```bash
python train.py --config configs/mosei_benchmark_config.yaml --device cuda
```

On Apple Silicon:

```bash
python train.py --config configs/mosei_benchmark_config.yaml --device mps
```

Run a synthetic training smoke test:

```bash
python train.py --config configs/cpu_test_config.yaml --use_synthetic --device cpu
```

## Evaluation and Analysis

Evaluate a checkpoint:

```bash
python evaluate.py --checkpoint outputs/mosei_benchmark/best_model.pth
```

Run baseline comparisons:

```bash
python scripts/run_baselines.py --config configs/mosei_benchmark_config.yaml --dataset mosei --num_seeds 5
```

Run ablations:

```bash
python scripts/run_ablations.py --config configs/mosei_benchmark_config.yaml --num_seeds 5
```

Analyze attention:

```bash
python scripts/analyze_attention.py \
  --checkpoint outputs/mosei_benchmark/best_model.pth \
  --config configs/mosei_benchmark_config.yaml \
  --dataset mosei
```

Generate result tables:

```bash
python scripts/generate_results_table.py --dataset mosei
```

## What works

- Forward and backward passes are stable on synthetic and on cleaned MOSEI batches
- Loss decreases consistently on the synthetic task
- Mixed-precision training, warmup + cosine LR schedule, and early stopping are wired
- The synthetic-data generator produces features with measurably tunable cross-modal correlation

## Known Issues and Limitations

- **No completed full benchmark run on CMU-MOSEI.** I do not currently have a number to report against published baselines on this dataset, and I do not want to publish synthetic-data numbers as if they were real-data numbers.
- **No baseline reproductions.** Comparisons to MulT, MISA, Self-MM, MMIM, etc. require running those baselines in the same training environment. I have not done that here.
- **Compute-limited.** Training has been done on Apple Silicon (MPS) and CPU. Some configurations are tuned downward for that setup.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the experiments I would like to run once compute is available, ordered by what I think is most informative.

## License

MIT.

## Acknowledgements

This work draws on the existing multimodal sentiment analysis literature, including MulT (Tsai et al., 2019), MISA (Hazarika et al., 2020), Self-MM (Yu et al., 2021), MMIM (Han et al., 2021), and the Multimodal Bottleneck Transformer (Nagrani et al., 2021). I have not yet reproduced these baselines in this codebase.
