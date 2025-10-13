---

marp: true
theme: default
class: invert
paginate: true

style: |
  section {
    font-size: 180%;
  }
  pre {
    vertical-align: text-top;
    font-size: 60%;
    line-height: 1.0;
  }
  .columns {
    display: flex;
    gap: 1em;
  }
  .col {
    flex: 1;
  }
---

# Random Forest × Item Response Theory

- Trees become respondents, images become items.
- Response matrix records per-tree correctness on held-out examples.
- Goal: explain RF behavior via IRT ability & difficulty signals.

---

# Item Response Theory Primer

<div class="columns">
  <div class="col">

**Key Terms**

- Ability (θ): latent skill of a respondent (tree).
- Difficulty (δ): latent hardness of an item (image).
- Discrimination (a): slope/steepness capturing how sharply success changes near δ.
- Guessing (c): lower asymptote (3PL); optional for multi-choice exams.

  </div>
  <div class="col">

**Models**

- 1PL/Rasch: P(correct) = sigmoid(theta − delta); single slope for every item.
- 2PL: P = sigmoid(a * (theta − delta)) — each item learns its own discrimination "a".
- 3PL: adds a guessing floor c: P = c + (1 − c) * sigmoid(a * (theta − delta)).
- Outputs give interpretable maps of strong trees, ambiguous items, and high-informative regions.

  </div>
</div>

---

# Random Forest Refresher

<div class="columns">
  <div class="col">

**Ensemble Mechanics**

- Bagging of decision trees (bootstrap + feature subsampling).
- Each tree votes; majority (or probability average) yields prediction.
- Strength from low-bias trees + reduced variance via averaging.
- Key levers: number of trees, max depth, feature subsampling, leaf size.

  </div>
  <div class="col">

**Signals We Track**

- Per-tree accuracy → ability (θ) comparison with IRT.
- Margins = p(true) − max p(other) highlight high/low confidence.
- Entropy over class probabilities measures tree disagreement.
- Feature importances & permutation scores remain our global view.

  </div>
</div>

---

# Pipeline Recap

<div class="columns">
  <div class="col">

**Data Prep (done)**

- Stratified CIFAR-10 subset: 10k train / 2k val / 2k test.
- Resize 64×64, normalize, PCA → 128-D embeddings.
- Cached artifacts in `data/cifar10_subset.npz` + `data/cifar10_embeddings.npz`.

  </div>

  <div class="col">

**Modeling Status**

- RF (200 trees) trained; metrics + importances saved.
- Response matrix `(200 × 2000)` persisted for IRT.
- 1PL Rasch fit (SVI, 600 epochs) complete.

  </div>
</div>

---

# Dataset Overview

| Dataset | Train | Val | Test | Feature Pipeline | Notes |
|---|---|---|---|---|---|
| CIFAR-10 subset | 10,000 | 2,000 | 2,000 | 64×64 RGB → PCA-128 or MobileNet-V3 (960-D) | Shared splits across Study I & II |
| MNIST mini | 4,000 | 800 | 800 | 28×28 grayscale → raw pixels (no PCA) | Sanity check for clean handwriting |

- All studies reuse cached artifacts under `data/` for reproducibility.
- CIFAR runs differ only in embedding backbone (PCA vs MobileNet); labels & splits stay fixed.
- MNIST mini-study mirrors the workflow to confirm signals transfer to simpler data.

---

# Study I: CIFAR-10 + PCA-128 Embeddings

- Baseline vision setup: 64×64 resize + PCA to 128 dims.
- 200-tree Random Forest trained on embeddings; response matrix size 200 × 2000.
- Use this run to introduce IRT diagnostics and identify weak spots.

---

# Study I Setup: CIFAR-10 + PCA-128

<div class="columns">
  <div class="col">
    <ul>
      <li>CIFAR-10 subset (10k / 2k / 2k) with stratified sampling and fixed seed.</li>
      <li>Preprocess: resize 64×64, normalize, PCA → 128-D embeddings (`data/cifar10_embeddings.npz`).</li>
      <li>Response matrix shape 200 × 2000 with mean tree accuracy 0.176.</li>
      <li>Artifacts: metrics, margins, entropy, and IRT outputs stored under `data/` & `figures/` root.</li>
    </ul>
  </div>
  <div class="col">
    <img src="figures/datasets/study1_cifar_samples.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Study I sample grid — stratified CIFAR-10 slices</p>
  </div>
</div>

---

# Study I Performance (PCA-128)

| Metric | Value |
|---|---|
| Test / Val / OOB acc | 0.4305 / 0.4145 / 0.3730 |
| Per-class range | 0.225 (cat) → 0.595 (ship) |
| Mean tree accuracy | 0.1759 |
| Mean margin / entropy | −0.0028 / 2.1503 |
| δ ↔ margin (Pearson) | −0.8286 |
| δ ↔ entropy (Pearson) | 0.6782 |

- Baseline ensemble underperforms due to weak PCA features yet preserves δ alignment.
- Low mean margin + high entropy indicate broad tree disagreement → fertile ground for IRT.
- Artifacts: metrics (`data/rf_metrics.json`), confusion (`data/rf_confusion.npy`), importances, permutations.

---

# Study I Confusion Matrix

<div class="columns">
  <div class="col">
    <img src="figures/rf_confusion_matrix.png" style="width:95%; border:1px solid #ccc;" />
  </div>
  <div class="col">

**Reading the matrix**

- High off-diagonal mass for cat ↔ dog, bird ↔ airplane, horse ↔ deer.
- Ships and trucks maintain >80% normalized diagonal despite shared structure.
- Hotspots align with IRT δ spikes (slides that follow), signalling data curation targets.

  </div>
</div>

---

# Study I Diagnostics: Ability Profiles

<div class="columns">
  <div class="col">
    <img src="figures/ability_vs_accuracy.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Ability (θ) vs tree accuracy — Spearman ≈ 0.99</p>
  </div>
  <div class="col">
    <img src="figures/wright_map.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Wright map: θ cluster near −11; δ stretches to 14</p>
  </div>
</div>

- Trees with θ above −10 outperform peers by ~3 pp even with PCA features.
- Long tail of low-ability trees (< −11.5) drags ensemble accuracy; pruning candidates.
- Wright map shows limited θ spread versus broad δ tail → feature quality bottleneck.

---

# Study I Diagnostics: δ vs Error Rate

<div class="columns">
  <div class="col">
    <img src="figures/difficulty_vs_error.png" style="width:95%; border:1px solid #ccc;" />
  </div>
  <div class="col">

- δ > 10 corresponds to averaged tree error >80%, mostly ambiguous animals.
- Items with δ < 0 are “free points” — nearly every tree agrees.
- Pearson ≈ 0.95, Spearman ≈ 0.94. Difficulty doubles as an error heat-map.

  </div>
</div>

---

# Study I Diagnostics: δ vs RF Signals

<div class="columns">
  <div class="col">
    <img src="figures/difficulty_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%;">PCA run: δ vs margin (Pearson −0.83)</p>
  </div>
  <div class="col">
    <img src="figures/difficulty_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%;">PCA run: δ vs entropy (Pearson 0.68)</p>
  </div>
</div>

- Hard items cluster bottom-right (low margin, high entropy) → ripe for relabeling or augmentation.
- Opposite corner contains “easy wins” with positive margin and low entropy.
- Study II mirrors these plots with MobileNet features, pushing |corr| above 0.80.

---

# Study I Evidence: Hard vs Easy Examples

<div class="columns">
  <div class="col">

![Hardest items](figures/hardest_items_test.png)

  </div>
  <div class="col">

![Easiest items](figures/easiest_items_test.png)

  </div>
</div>

- Hardest items skew toward ambiguous airplane/ship silhouettes and cluttered cat/dog scenes.
- Notice recurring mislabeled-looking ships (δ ≈ 14) flagged for manual review.
- Easy set dominated by deterministic cues (red fire trucks, high-contrast ships) → low δ and entropy.

---

# Study II: CIFAR-10 + MobileNet Embeddings

- Swap PCA features for MobileNet-V3 (960-D) while keeping tree count and splits constant.
- Measure how richer features alter RF metrics, margins/entropy, and IRT parameter spreads.
- Use as a reality check before expanding to new datasets.

---

# Study II Setup: CIFAR-10 + MobileNet-V3

<div class="columns">
  <div class="col">
    <ul>
      <li>Reuse CIFAR-10 subset splits from Study I to isolate feature effects.</li>
      <li>Extract 960-D embeddings from pretrained MobileNet-V3 Small (`data/cifar10_mobilenet_embeddings.npz`).</li>
      <li>Response matrix shape 200 × 2000 with mean tree accuracy 0.482.</li>
      <li>Dedicated artifacts: `data/mobilenet/*`, plots in `figures/mobilenet/`.</li>
    </ul>
  </div>
  <div class="col">
    <img src="figures/datasets/study2_cifar_samples.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Study II sample grid — same splits, MobileNet embeddings</p>
  </div>
</div>

---

# Study II Performance (MobileNet-V3)

| Metric | Value |
|---|---|
| Test / Val / OOB acc | 0.8090 / 0.8135 / 0.7967 |
| Per-class range | 0.68 (cat) → 0.915 (ship) |
| Mean tree accuracy | 0.4817 |
| Mean margin / entropy | 0.2806 / 1.4663 |
| δ ↔ margin (Pearson) | −0.8825 |
| δ ↔ entropy (Pearson) | 0.8113 |

- Pretrained features boost accuracy by 37 pp while strengthening δ correlations.
- Higher margins + lower entropy show confidence gains except on stubborn animal classes.
- Artifacts live under `data/mobilenet/` (metrics, response matrix, signals, IRT outputs).

---

# Study II Diagnostics: δ vs RF Signals

<div class="columns">
  <div class="col">
    <img src="figures/mobilenet/mobilenet_difficulty_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%;">δ vs margin (Pearson −0.88)</p>
  </div>
  <div class="col">
    <img src="figures/mobilenet/mobilenet_difficulty_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%;">δ vs entropy (Pearson 0.81)</p>
  </div>
</div>

- MobileNet compresses the easy cluster (high margin, low entropy) while isolating true hard cases.
- Higher correlation magnitudes indicate better alignment between δ and RF uncertainty signals.
- Cat/dog confusions persist despite stronger features → candidates for targeted curation.

---

# Study II Diagnostics: Ability Profiles

<div class="columns">
  <div class="col">
    <img src="figures/mobilenet/ability_vs_accuracy.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Ability (θ) vs tree accuracy — Pearson 0.983</p>
  </div>
  <div class="col">
    <img src="figures/mobilenet/wright_map.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Wright map: θ variance shrinks to 0.25</p>
  </div>
</div>

- θ mean −0.21 ± 0.25: trees cluster closer together than PCA baseline (σ 0.55 → 0.25).
- Ability remains tightly coupled to per-tree accuracy; even weakest trees clear 40%.
- Shared axis shows overlap where confident trees meet easy airplane/ship items.
- Ability compression signals that feature quality, not tree diversity, now limits performance.

---

# Study II Diagnostics: δ vs Error Rate

<div class="columns">
  <div class="col">
    <img src="figures/mobilenet/difficulty_vs_error.png" style="width:95%; border:1px solid #ccc;" />
  </div>
  <div class="col">

- Pearson 0.922: δ remains strongly aligned with mean tree error despite higher accuracy ceiling.
- Hardest items (δ > 8) persist from PCA run — mostly cat/dog overlaps and ambiguous aircraft.
- Easy zone (δ < −3) expands, showing MobileNet features unlock more “free points.”

  </div>
</div>

---


# Study III: MNIST Mini-Study

- Lightweight handwriting dataset to validate RF × IRT beyond CIFAR-10.
- Serves as control: simpler classes, higher accuracy, clearer δ separation.
- Highlights how pipeline behaves when ambiguity is rare but still detectable.

---

# Study III Setup: MNIST Mini-Study

<div class="columns">
  <div class="col">
    <ul>
      <li>Split 4k / 800 / 800 digits with stratified sampling and fixed seed.</li>
      <li>Minimal preprocessing: 28×28 grayscale flattened; no augmentation.</li>
      <li>Random Forest (200 trees) trained on raw pixels; response matrix shape 200 × 800.</li>
      <li>Artifacts stored under `data/mnist/` with plots in `figures/mnist/`.</li>
    </ul>
  </div>
  <div class="col">
    <img src="figures/datasets/mnist_samples.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Study III sample grid — curated MNIST mini split</p>
  </div>
</div>

---

# Study III Performance (MNIST)

| Metric | Value |
|---|---|
| Train / Val / Test | 4000 / 800 / 800 |
| RF test / val / OOB | 0.9475 / 0.9413 / 0.9140 |
| Mean margin / entropy | 0.5546 / 1.0351 |
| δ ↔ margin (Pearson) | −0.950 |
| δ ↔ entropy (Pearson) | 0.958 |
| θ mean ± σ | 4.23 ± 0.44 |
| δ mean ± σ | −1.75 ± 8.19 |

- Ambiguous digits (e.g., brushed 5 vs 6) spike δ toward ±20; trees vote confidently elsewhere.
- Reinforces link between low entropy, high margin, and low δ on clean handwriting data.
- Provides a “sanity benchmark” to validate the RF × IRT pipeline outside CIFAR.

---

# Study III Diagnostics: δ vs RF Signals

<div class="columns">
  <div class="col">
    <img src="figures/mnist/mnist_difficulty_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%;">δ vs margin (Pearson −0.95)</p>
  </div>
  <div class="col">
    <img src="figures/mnist/mnist_difficulty_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%;">δ vs entropy (Pearson 0.96)</p>
  </div>
</div>

- Clean digits show near-perfect alignment between IRT difficulty and RF uncertainty signals.
- Low scatter indicates only a handful of items drive ensemble uncertainty.
- Marks δ > 12 digits for manual audit (stroke collisions, 3 vs 5, 4 vs 9).

---

# Study III Diagnostics: Ability Profiles

<div class="columns">
  <div class="col">
    <img src="figures/mnist/ability_vs_accuracy.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Ability (θ) vs tree accuracy — Pearson 0.995</p>
  </div>
  <div class="col">
    <img src="figures/mnist/wright_map.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Wright map: θ mean 4.23 ± 0.44; δ mean −1.75 ± 8.19</p>

- θ mean 4.23 ± 0.44: trees quickly separate easy digits, reflecting high consensus.
- δ mean −1.75 ± 8.19 with heavy tails on ambiguous strokes.
- Shared scale shows abundant overlap → most items are easy wins with a few hard spikes.
- Provides contrast against CIFAR studies where ability mass sat below zero.

  </div>
</div>

---

# Study III Diagnostics: δ vs Error Rate

<div class="columns">
  <div class="col">
    <img src="figures/mnist/difficulty_vs_error.png" style="width:95%; border:1px solid #ccc;" />
  </div>
  <div class="col">

- Pearson 0.962: δ spikes pinpoint the rare ambiguous digits despite overall high accuracy.
- Outliers (δ > 12) correspond to stroke-collided 3/5/8 and 4/9 pairs flagged for curation.
- Long negative tail shows the majority of digits are trivial for the ensemble.

  </div>
</div>

---

# Cross-Study Snapshot

| Study | Feature Backbone | Test Acc | δ ↔ margin (Pearson) | δ ↔ entropy (Pearson) | θ σ | δ σ |
|---|---|---|---|---|---|---|
| Study I: CIFAR + PCA-128 | PCA-128 | 0.4305 | −0.8286 | 0.6782 | 0.55 | 4.10 |
| Study II: CIFAR + MobileNet | MobileNet-V3 (960-D) | 0.8090 | −0.8825 | 0.8113 | 0.25 | 4.67 |
| Study III: MNIST Mini | Raw pixels | 0.9475 | −0.950 | 0.958 | 0.44 | 8.19 |

- Feature backbone drives both accuracy gains and δ alignment strength.
- θ variance collapses with MobileNet (0.25) indicating tree consistency; MNIST keeps moderate spread despite high accuracy.
- MNIST δ σ expands to 8.19, highlighting rare but extreme digit ambiguities versus CIFAR’s visual noise.

---

# Tabular RF Reference (Breast Cancer)

| Metric | Value |
|---|---|
| Samples / features | 569 / 30 |
| RF test accuracy | 0.971 |
| RF OOB accuracy | 0.965 |
| ROC AUC | 0.997 |
| Top features | worst concave points, worst perimeter, worst radius |

- Provides a well-behaved baseline: high accuracy, small δ expected when we run IRT.
- Feature rankings highlight geometric shape descriptors → contrast to image pipelines.
- Script: `python scripts/run_rf_tabular_example.py --dataset breast_cancer`

---

# IRT Fit (Study I Baseline, 1PL 600 epochs)

- Optimizer: Adam lr=0.05, SVI Trace_ELBO, seed=7.
- Final loss: **1.50M** (down from 165M at init).
- Tree ability (θ): mean −11.14, σ 0.55, range [−12.79, −9.68].
- Item difficulty (δ): mean 5.90, σ 4.10, range [−10.74, 14.26].
- Correlations — ability ↔ tree accuracy **0.999**, difficulty ↔ item error **0.950**.
- Cross-check: embedding & MNIST tables confirm these correlations persist across datasets.

Diagnostic JSON: `data/irt_summary.json`, extremes in `data/irt_extremes.json`.

---

# Edge Cases Across Datasets

- **CIFAR-10 (PCA):** δ tail contains grayscale ships + occluded pets. Margin < −0.2, entropy > 2.2.
- **CIFAR-10 (MobileNet):** Outliers shrink but persist for cat/dog overlap; δ still > 8 despite cleaner features.
- **MNIST:** High δ digits stem from stroke noise (e.g., 9 vs 4). Entropy jumps above 1.9 only for these cases.
- **Breast Cancer:** Expect δ near zero (not yet fitted) — contrast will highlight how clean tabular data behaves.
- Actionable: focus audits on items with δ > 8 + low margins; they recur across embeddings.

---

# Class Difficulty vs RF Error

<div class="columns">
  <div class="col">

- Cats, horses, dogs exhibit δ ≈ 7–8 with RF error ≥ 0.60, marking priority classes for curation.
- Ships and airplanes remain easiest: δ ≈ 4 with RF error ≤ 0.46.
- Aligning δ with RF error spotlights where ensemble uncertainty mirrors misclassification hotspots.
  </div>

  <div class="col">
![Class summary](figures/class_difficulty_vs_error.png)
  </div>
</div>

---

# Training Loss & Distributions

<div class="columns">
  <div class="col">

<img src="figures/irt_training_loss.png" style="width:95%; border:1px solid #ccc;" />

  </div>
  <div class="col">

<img src="figures/ability_hist.png" style="width:95%; border:1px solid #ccc;" />
</div>
  <div class="col">

<img src="figures/difficulty_hist.png" style="width:95%; border:1px solid #ccc; margin-top:0.75em;" />

  </div>
</div>

---

# Emerging Insights

- Top 10 trees achieve 19–20% accuracy; lowest performers drop below 15%.
- Hardest items (δ > 13) align with CIFAR-10 ships/airplanes confusions.
- Easiest items (δ < −9.5) mostly belong to truck/ship classes with distinctive features.
- MobileNet run yields θ variance halved and pushes accuracy to 81%, confirming sensitivity to embeddings.
- MobileNet difficulty plots confirm tighter δ alignment (|corr| ≥ 0.81) and isolate stubborn animal confusions.
- MNIST pipeline shows how clean data collapses entropy while keeping δ informative for rare ambiguities.
- Parallel study sections simplify comparisons — identical tables & diagnostics spotlight backbone effects.
- Loss curve still descending: consider more epochs or lower lr for finer convergence.

Extremes listed in `data/irt_extremes.json` for manual inspection.

---

# Next Steps

- Drop confusion matrix + new montages into the storytelling deck.
- Promote notebook automation to emit the tables used here (embedding, MNIST, tabular).
- Run planned 2PL/3PL experiments (see `reports/discrimination_analysis_plan.md`) to get discrimination.
- Compare tree ability with structural traits (depth, leaves) for richer diagnostics.
- Extend edge-case audit: inspect δ>8 + margin<0 items across embeddings and datasets.
- Hook the new diagnostics script into the notebook so plots regenerate alongside tables.
