# MAFT: Multimodal Attention Fusion Transformer

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

```
text  ─► embedding + linear projection ──┐
audio ─► linear + BiLSTM ────────────────┼─► concat + modality embeddings
visual ─► linear + BiLSTM ───────────────┘    │
                                              ▼
                                    [bottleneck tokens] + [tokens]
                                              │
                                              ▼
                              shared TransformerEncoder
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                         classification   regression    per-modality
                         + temperature     head         auxiliary heads
                                                      (used in consistency loss)
```

Key design choices:
- **Modality-aware embeddings** added to each token so the shared transformer can distinguish text/audio/visual positions.
- **Bottleneck tokens** that attend to all modalities and produce a gated, pooled representation used by the prediction heads. Conceptually related to the bottlenecks in MBT (Nagrani et al., 2021).
- **Scheduled modality dropout** that anneals from a low rate at the start of training to a higher rate later, randomly zeroing entire modalities for some samples to encourage robust representations.
- **Symmetric-KL consistency loss** between per-modality auxiliary classifier heads, encouraging modality views to agree on the prediction.
- **Quality-estimator masks** that distinguish valid tokens from padding throughout the model (one of the bugs that took longest to debug — see Known Issues below).

## Repository layout

```
models/
  encoders.py       text / audio (BiLSTM) / visual (BiLSTM) encoders
  fusion.py         shared transformer with modality embeddings + bottlenecks
  maft.py           top-level model wiring encoders + fusion + heads
  quality.py        per-token quality / mask handling
losses/
  consistency.py    symmetric-KL multi-view consistency loss
validation_system/
  data_models.py    TestResult / ValidationReport dataclasses
  synthetic_data.py controllable synthetic multimodal dataset
  utils.py          formatting, timing, logging helpers
configs/            YAML configs for CPU/M-series test runs and MOSEI
train.py            training loop with AMP, LR schedule, early stopping
evaluate.py         evaluation with classification + regression metrics
mosei_dataloader.py CMU-MOSEI loader (handles Inf/NaN in raw COVAREP features)
tests/              forward-pass + architecture smoke tests
```

## Setup

Tested on Python 3.11 with PyTorch 2.x.

```bash
pip install torch torchvision torchaudio
pip install transformers numpy scipy scikit-learn pyyaml tqdm psutil
```

For CMU-MOSEI data preparation you additionally need the CMU Multimodal SDK; see `prepare_mosei_benchmark.py` for usage.

## Running

**Synthetic-data smoke test (fastest, no real data needed):**

```bash
python test_validation_base.py     # validation framework primitives
python test_synthetic_data.py      # synthetic dataset correctness
python test_maft_quick_train.py    # ~20-step training run on synthetic data
```

These should complete in under a minute on CPU and confirm the model trains and the loss decreases.

**CMU-MOSEI training (requires prepared data in `data/mosei/`):**

```bash
python train.py --config configs/mosei_benchmark_config.yaml --device cuda
```

On Apple Silicon, the MPS backend works after a one-line fix to the transformer encoder (`enable_nested_tensor=False`), which is already applied in `models/fusion.py`.

## What works

- Forward and backward passes are stable on synthetic and on cleaned MOSEI batches
- Loss decreases consistently on the synthetic task (the validation harness checks this automatically)
- Mixed-precision training, warmup + cosine LR schedule, and early stopping are wired and tested
- The synthetic-data generator produces features with measurably tunable cross-modal correlation, so the model's response to varying correlation strength can be studied independently of MOSEI-specific noise

## Known issues and limitations

- **No completed full benchmark run on CMU-MOSEI.** I do not currently have a number to report against published baselines on this dataset, and I do not want to publish synthetic-data numbers as if they were real-data numbers. Sample efficiency on smaller MOSEI subsets has been promising but is not a substitute for a clean full run.
- **No baseline reproductions.** Comparisons to MulT, MISA, Self-MM, MMIM, etc. require running those baselines in the same training environment. I have not done that here. The earlier comparison tables in this README were not from runs I performed and have been removed.
- **Compute-limited.** Training has been done on Apple Silicon (MPS) and CPU. Some configurations (large hidden dim, batch size > 4) are not stable in this setup, so the codebase has been tuned downward in places.
- **No published paper or preprint.** The previous BibTeX entry in this README claimed a publication that does not exist. I have removed it.

## Failure modes I have fixed during development

Listing these because they are part of what I have actually learned from this project:

- **NaN / Inf in raw COVAREP audio features** propagated through standardization and then through the loss. Fixed by `np.nan_to_num` on raw features before fitting the `StandardScaler`, and again after scaling.
- **MPS backend incompatibility with `nn.TransformerEncoder`'s nested-tensor fast path.** Resolved by passing `enable_nested_tensor=False` at encoder construction.
- **Mask-convention bug in the quality estimator.** The module assumed `True == valid` while the rest of the codebase used `True == padding`. Silently produced wrong masks until I noticed attention weights collapsing in a specific run.
- **`pack_padded_sequence` instabilities in the BiLSTM encoders** under variable-length audio batches. Replaced with a plain LSTM forward pass plus an explicit pad mask; small loss in efficiency, large gain in reliability.
- **Label extraction from the CMU SDK** silently producing a single class for all samples on early runs because I was indexing the wrong column of `All Labels`. Added a sanity check that fails preprocessing if `len(set(labels)) == 1`.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the experiments I would like to run once compute is available, ordered by what I think is most informative.

## License

MIT.

## Acknowledgements

This work draws on the existing multimodal sentiment analysis literature, including MulT (Tsai et al., 2019), MISA (Hazarika et al., 2020), Self-MM (Yu et al., 2021), MMIM (Han et al., 2021), and the Multimodal Bottleneck Transformer (Nagrani et al., 2021). I have not yet reproduced these baselines in this codebase.
