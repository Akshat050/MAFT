# MAFT Research Roadmap

This document outlines the experiments I would run on MAFT once compute is available, ordered roughly by how informative I expect each to be. The framing is closer to "reliability and robustness study" than to "benchmark chase": multimodal sentiment is a saturated leaderboard, and the more useful questions are about how these models fail.

Each item below states the question, experiment, metric, and what a positive, negative, or null result would mean.

## Phase 1: Establish a Credible Baseline

Before any of the more interesting questions can be answered, MAFT needs one clean, reproducible training run on CMU-MOSEI with all results logged and seeds tracked.

### 1.1 Single-Seed Reference Run

**Question:** What does MAFT actually achieve on CMU-MOSEI under controlled conditions?

**Experiment:** One full training run on CMU-MOSEI with `configs/mosei_benchmark_config.yaml`, fixed seed, AMP enabled, and early stopping. Log training loss decomposition, validation loss, learning rate, modality dropout rate over time, and final test-set metrics.

**Expected result:** A defensible single-seed number. The goal is to have a real number to compare future ablations against, not to claim SOTA prematurely.

### 1.2 Five-Seed Run for Variance

**Question:** Is the single-seed number stable, or is MAFT high-variance under different initializations?

**Experiment:** Repeat 1.1 with seeds 42, 43, 44, 45, and 46. Report mean and standard deviation for each metric.

**Why it matters:** Knowing variance is a precondition for trusting any ablation result.

## Phase 2: Reliability Under Modality Corruption

This is the part of the project that connects most directly to robustness and honest model behavior.

### 2.1 Modality-Drop Evaluation

**Question:** How much does each modality contribute? Does the model degrade gracefully when one is missing?

**Experiment:** Take the trained model from Phase 1 and evaluate on the test set with each modality independently zeroed out at inference. Compare to full-modality performance.

**Interpretation:**

- If all three drops cause similar degradation, the model is genuinely integrating modalities.
- If dropping text catastrophically hurts while audio and visual barely matter, the model is essentially a text classifier with multimodal regularization.
- If dropping any single modality drops performance below a unimodal baseline, fusion is brittle.

### 2.2 Modality-Noise Injection

**Question:** Does the model degrade gracefully under noisy modalities, or does noise in one channel poison the prediction?

**Experiment:** At inference, add Gaussian noise of increasing standard deviation to one modality at a time. Plot accuracy against noise level for each modality.

**Why it matters:** Real-world multimodal inputs are noisy. A model that produces confident-but-wrong predictions under one corrupted modality has a reliability problem.

### 2.3 Modality Disagreement

**Question:** When modalities provide conflicting signals, what does the model do?

**Experiment:** Construct test cases where text and audio/visual signals disagree by swapping modalities across samples. Measure which modality the model sides with, whether confidence drops, and whether consistency loss correlates with disagreement.

**Useful outcome:** Lower confidence on disagreement cases. A confident prediction despite disagreement would be a failure mode worth documenting.

## Phase 3: Component Ablations

These quantify the value of MAFT's specific design choices.

### 3.1 Modality Dropout Schedule

**Question:** Does scheduled annealing of modality dropout matter, or is a constant rate equally good?

**Experiment:** Compare constant dropout rates against the current scheduled rate. Measure clean-test accuracy and corruption robustness.

### 3.2 Consistency Loss

**Question:** Does symmetric-KL consistency loss between per-modality heads actually do anything?

**Experiment:** Train with multiple values of `consistency_weight`. Measure clean accuracy, corruption robustness, and agreement between per-modality predictions.

**Interpretation:**

- If it improves clean accuracy but not robustness, it may be acting as a generic regularizer.
- If it improves robustness specifically, it is doing what it was designed to do.
- If it helps neither, it should be removed.

### 3.3 Bottleneck Token Count

**Question:** Does the number of bottleneck tokens matter, and is there a sweet spot?

**Experiment:** Train with several bottleneck-token counts and compare clean accuracy, parameter count, and training time.

### 3.4 Fusion Depth

**Question:** Does adding more fusion transformer layers help, or does most of the useful work happen early?

**Experiment:** Compare shallow and deeper fusion stacks under the same benchmark settings.

## Phase 4: Calibration and Abstention

This phase is directly relevant to making multimodal predictions more honest under uncertainty.

### 4.1 Calibration Analysis

**Question:** Is the model's confidence calibrated against actual accuracy?

**Experiment:** Generate reliability diagrams and expected calibration error on the test set and on corrupted-modality conditions. Compare with and without temperature scaling.

### 4.2 Selective Prediction

**Question:** Can the model usefully abstain on uncertain inputs?

**Experiment:** Use max softmax probability as a confidence score and plot accuracy against coverage.

### 4.3 Disagreement-Aware Abstention

**Question:** Is modality disagreement a better abstention signal than overall confidence?

**Experiment:** Compute per-modality predictions from auxiliary heads, define a disagreement score, and compare selective-prediction curves using confidence, disagreement, and both together.

## Phase 5: Synthetic-to-Real Transfer

### 5.1 Synthetic vs. Real Conclusions

**Question:** Do trends observed on the controllable synthetic dataset reproduce on real CMU-MOSEI?

**Experiment:** Run a small set of ablations on both synthetic data and MOSEI. Look for cases where conclusions differ.

**Why it matters:** If synthetic conclusions do not transfer, the validation framework is misleading and needs to be redesigned.

## Repository Hygiene

- Keep `configs/mosei_benchmark_config.yaml` as the canonical MOSEI benchmark config.
- Keep developer-only diagnostics in `scripts/dev/`.
- Avoid committing generated summaries, local backups, private IDE metadata, cloud state, or infrastructure secrets.
- Review dependencies and remove packages that are no longer needed after infrastructure cleanup.

## Out of Scope

- **Beating SOTA on CMU-MOSEI.** This is not the main goal of the project.
- **End-to-end audio/visual feature learning.** Replacing hand-engineered COVAREP and FACET features would be a separate project.
- **New datasets.** The priority is one credible CMU-MOSEI study rather than multiple incomplete dataset studies.

## What I Would Publish

If Phase 1, Phase 2, and most of Phase 4 produce clean results, there may be a workshop-paper-sized contribution in studying how a typical multimodal sentiment transformer fails under modality corruption and whether internal modality-disagreement signals can support honest abstention.
