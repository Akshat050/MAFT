# MAFT Research Roadmap

This document outlines the experiments I would run on MAFT once compute is available, ordered roughly by how informative I expect each to be. The framing is closer to "reliability and robustness study" than to "benchmark chase" — multimodal sentiment is a saturated leaderboard, and the more useful questions are about *how* these models fail, not whether they can be pushed another 0.2% on MOSEI.

Each item below states: the question, the experiment, the metric, and what a positive / negative / null result would mean.

---

## Phase 1: Establish a credible baseline

Before any of the more interesting questions can be answered, MAFT needs one clean, reproducible training run on CMU-MOSEI with all results logged and seeds tracked.

### 1.1 Single-seed reference run

**Question:** What does MAFT actually achieve on CMU-MOSEI under controlled conditions?

**Experiment:** One full training run on CMU-MOSEI with `mosei_benchmark_config.yaml`, fixed seed, AMP enabled, early stopping with patience 7. Log: training loss decomposition (cls / reg / cons), validation loss, learning rate, modality dropout rate over time, and final test-set metrics (7-class accuracy, binary accuracy, MAE, Pearson r).

**Expected result:** A defensible single-seed number. Probably below current published SOTA, possibly meaningfully so. That is fine — the goal of this run is to have a real number to compare future ablations against, not to beat MMIM.

**Why this is the prerequisite for everything else:** Without it, every later result is uninterpretable.

### 1.2 Five-seed run for variance

**Question:** Is the single-seed number stable, or is MAFT high-variance under different initializations?

**Experiment:** Repeat 1.1 with seeds 42, 43, 44, 45, 46. Report mean ± std for each metric.

**Why it matters:** Many published multimodal results are reported with implausibly low variance. Knowing my own variance is a precondition for trusting any ablation result.

---

## Phase 2: Reliability under modality corruption

This is the part of the project that connects most directly to AI safety questions about robustness and honest model behavior.

### 2.1 Modality-drop evaluation

**Question:** How much does each modality contribute? Does the model degrade gracefully when one is missing?

**Experiment:** Take the trained model from Phase 1 and evaluate on the test set with each modality independently zeroed out at inference. Compare to full-modality performance.

**What different results mean:**
- *All three drops cause similar degradation* → model is genuinely integrating modalities
- *Dropping text catastrophically hurts, dropping audio/visual barely matters* → model is essentially a text classifier with multimodal regularization
- *Dropping any single modality drops performance below a unimodal baseline* → fusion is brittle and creates unhealthy dependence on co-occurrence

This is a cheap experiment and one of the most informative.

### 2.2 Modality-noise injection

**Question:** Does the model degrade gracefully under noisy modalities, or does noise in one channel poison the prediction?

**Experiment:** At inference, add Gaussian noise of increasing standard deviation (0.0, 0.25, 0.5, 1.0, 2.0 × feature std) to one modality at a time. Plot accuracy vs. noise level for each modality.

**Why it matters:** Real-world multimodal inputs are noisy. A model that produces confident-but-wrong predictions under one corrupted modality is an honesty failure, not a calibration failure.

### 2.3 Modality disagreement

**Question:** When modalities provide *conflicting* signals, what does the model do?

**Experiment:** Construct test cases where text says "positive" and audio/visual say "negative" (or vice versa). Easiest construction: take a positive sample and swap in text from a negative sample (or vice versa). Measure (a) which modality the model sides with, (b) whether prediction confidence drops, (c) whether the consistency loss correlates with disagreement.

**What I'd hope to see:** Lower confidence on disagreement cases. **What I'd actually expect to see:** Confidently siding with whichever modality has the strongest signal, with little uncertainty calibration. That gap would itself be a useful finding.

---

## Phase 3: Component ablations

These quantify the value of MAFT's specific design choices.

### 3.1 Modality dropout schedule

**Question:** Does the *scheduled* annealing of modality dropout matter, or is a constant rate equally good?

**Experiment:** Three runs holding everything else fixed: dropout rate constant at 0.05, constant at 0.2, and scheduled from 0.05 → 0.35 (the current default). Compare both clean-test accuracy and accuracy under the 2.1/2.2 corruption conditions.

**Why I expect this to matter:** Schedule is intended to let the model learn fusion first, then learn robustness. If a constant rate works just as well under corruption, that intuition was wrong, and scheduling is just complexity for its own sake.

### 3.2 Consistency loss

**Question:** Does the symmetric-KL consistency loss between per-modality heads actually do anything?

**Experiment:** Train three configurations: λ_cons = 0.0, 0.1 (current default), and 0.5. Measure (a) clean accuracy, (b) corruption-robustness from 2.1, and (c) alignment between per-modality predictions on held-out data (with and without consistency loss).

**What different outcomes mean:**
- *λ_cons > 0 improves clean accuracy but not robustness* → it's acting as a generic regularizer
- *λ_cons > 0 improves robustness specifically* → it's doing what it's designed to do
- *λ_cons > 0 doesn't help either* → it's load-bearing for nothing and should be removed

A null result here is just as publishable as a positive one, and a lot more honest than the current literature norm.

### 3.3 Bottleneck token count

**Question:** Does the number of bottleneck tokens matter, and is there a sweet spot?

**Experiment:** Train with `num_bottlenecks` ∈ {1, 4, 8, 16, 32}. Compare clean accuracy, parameter count, and training time.

**Hypothesis:** Performance plateaus quickly (probably by 4-8 tokens) and bottleneck count is mostly a knob with minor effect.

### 3.4 Fusion depth

**Question:** Does adding more layers to the fusion transformer help, or does the bulk of the work happen in the first one or two?

**Experiment:** `num_layers` ∈ {1, 2, 4}. Same metrics as 3.3.

---

## Phase 4: Calibration and abstention

This phase is the most directly safety-relevant and is genuinely under-studied in multimodal sentiment work.

### 4.1 Calibration analysis

**Question:** Is the model's confidence (max softmax probability) calibrated against actual accuracy?

**Experiment:** Standard reliability diagrams + expected calibration error (ECE) on the test set, plus on the corrupted conditions from 2.1/2.2. Compare with and without the temperature-scaling parameter `log_temp` that's already in the model head.

### 4.2 Selective prediction

**Question:** Can the model usefully *abstain* on uncertain inputs?

**Experiment:** Use max softmax probability as a confidence score. Plot accuracy vs. coverage (fraction of test samples the model "answers"). A useful model should be able to reach high accuracy by abstaining on its hardest 20-30% of inputs.

**Why this matters:** Honest abstention is a primitive AI safety capability. A model that knows when it doesn't know is much more useful in deployment than a slightly higher-accuracy one that doesn't.

### 4.3 Disagreement-aware abstention

**Question:** Is *modality disagreement* a better abstention signal than overall confidence?

**Experiment:** Compute per-modality predictions from the auxiliary heads. Define a disagreement score (e.g., KL between modality predictions). Compare selective-prediction curves using (a) max softmax confidence, (b) disagreement score, (c) combined.

**What I'd hope:** Disagreement is a useful complementary signal that catches errors confidence misses. This would justify keeping the auxiliary heads at inference and not just at training time.

---

## Phase 5: Transfer of synthetic-data conclusions to real data

A meta-question about the validation framework itself.

### 5.1 Synthetic vs. real conclusions

**Question:** Do trends observed on the controllable synthetic dataset (e.g., effect of correlation_strength on robustness) reproduce on real CMU-MOSEI?

**Experiment:** Run a small set of the above ablations on both synthetic data and on MOSEI. Look for cases where the conclusion differs.

**Why it matters:** The synthetic-data validation framework is core to how I develop this codebase. If conclusions don't transfer, the framework is misleading me — and that itself is a useful methodological finding.

---

## Out of scope (deliberately)

These would be interesting but are beyond what I think I can do well in the near term:

- **Beating SOTA on CMU-MOSEI.** Saturated benchmark, and 0.5% improvements aren't where I should be spending compute.
- **End-to-end audio/visual feature learning.** COVAREP and FACET are hand-engineered; replacing them with learned features (e.g., wav2vec2, CLIP visual) would be a different project.
- **New datasets.** The codebase is structured to accept new datasets, but I'd rather have one credible CMU-MOSEI study than two half-baked ones across datasets.

---

## What I would publish

If Phase 1 + Phase 2 + most of Phase 4 produce clean, honest results — even if MAFT doesn't beat published baselines — I think there is a workshop-paper-sized contribution in: *"How does a typical multimodal sentiment transformer fail under modality corruption, and can its own internal modality-disagreement signal be used for honest abstention?"*

That framing requires no SOTA, plays directly to robustness and honesty questions that AI safety researchers care about, and is achievable with modest compute.
