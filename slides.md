---

marp: true
theme: default
paginate: true

style: |
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

# Data & Embeddings Snapshot

- PCA-128 embeddings, mean 0.0 ± 0.06 per feature.
- `scripts/data_pipeline.py` CLI caches subsets & embeddings.
- Ready for alternative feature backbones (MobileNet/ResNet) if needed.

---

# Random Forest Results

<div class="columns">
  <div class="col">

**Key Metrics**

- Test accuracy **43.1%**, validation **41.5%**, OOB **37.3%**.
- Per-class accuracy span: 22.5% (cat) → 59.5% (ship).
- Response matrix mean tree accuracy: **17.6%**.

  </div>

  <div class="col">

**Artifacts**

- Metrics: `data/rf_metrics.json`
- Confusion: `data/rf_confusion.npy`
- Importances: `data/rf_feature_importances.npy`
- Permutation: `data/rf_permutation_importance.csv`

  </div>
</div>

---

# Embedding Comparison

- **Baseline (PCA‑128):** Accuracy 43.1%, margin mean ≈ −0.003, δ↔margin Pearson −0.83.
- **MobileNet-V3 (960-D):** Accuracy 80.9%, margin mean ≈ 0.281, δ↔margin Pearson −0.88.
- Entropy drops from 2.15 → 1.47 and ability spread tightens (σ: 0.55 → 0.25).
- Difficulty spread widens slightly (σ: 4.10 → 4.67) indicating richer separation of hard/easy items.
- Full table: `reports/embedding_comparison.md`.

---

# Confusion Matrix View

![Normalized confusion matrix](figures/rf_confusion_matrix.png)

- Notable confusions: cat ↔ dog, bird ↔ airplane, horse ↔ deer.
- Strong separability for ship/truck classes despite noise in others.
- Guides which classes to prioritize for feature or data augmentation.

---

# IRT Fit (1PL, 600 epochs)

- Optimizer: Adam lr=0.05, SVI Trace_ELBO, seed=7.
- Final loss: **1.50M** (down from 165M at init).
- Tree ability (θ): mean −11.14, σ 0.55, range [−12.79, −9.68].
- Item difficulty (δ): mean 5.90, σ 4.10, range [−10.74, 14.26].
- Correlations — ability ↔ tree accuracy **0.999**, difficulty ↔ item error **0.950**.

Diagnostic JSON: `data/irt_summary.json`, extremes in `data/irt_extremes.json`.

---

# Diagnostics: Ability vs Accuracy

![Tree ability vs accuracy](figures/ability_vs_accuracy.png)

Mean tree accuracy ≈17.6%; higher ability trees cluster near 20% correct.

---

# Diagnostics: Difficulty vs Error Rate

![Item difficulty vs error](figures/difficulty_vs_error.png)

High-difficulty items correlate strongly with tree error (ρ ≈ 0.95).

---

# Difficulty vs RF Signals

<div class="columns">
  <div class="col">

![δ vs margin](figures/difficulty_vs_margin.png)

  </div>
  <div class="col">

![δ vs entropy](figures/difficulty_vs_entropy.png)

  </div>
</div>

- Pearson(δ, margin) = **−0.83**, Spearman = **−0.79** → harder items drive negative margins.
- Pearson(δ, entropy) = **0.68**, Spearman = **0.55** → higher difficulty aligns with uncertain trees.

---

# Wright Map Snapshot

![Wright map](figures/wright_map.png)

- Tree abilities cluster tightly (θ ≈ −11), suggesting limited diversity among respondents.
- Item difficulties span a wide range (δ ∈ [−10.7, 14.3]); tail heavy on very hard items.
- Shared axis highlights sparse overlap where strong trees meet easy items.

---

# Hardest vs Easiest Test Examples

<div class="columns">
  <div class="col">

![Hardest items](figures/hardest_items_test.png)

  </div>
  <div class="col">

![Easiest items](figures/easiest_items_test.png)

  </div>
</div>

- Hardest items skew toward ambiguous airplane/ship silhouettes and cluttered animal scenes.
- Easiest items show high-saturation trucks/ships with distinctive backgrounds aiding tree votes.

---

# Class Difficulty vs RF Error

![Class summary](figures/class_difficulty_vs_error.png)

- Cats, horses, dogs exhibit δ ≈ 7–8 with RF error ≥ 0.60, marking priority classes for curation.
- Ships and airplanes remain easiest: δ ≈ 4 with RF error ≤ 0.46.
- Aligning δ with RF error spotlights where ensemble uncertainty mirrors misclassification hotspots.

---

# Training Loss & Distributions

<div class="columns">
  <div class="col">

![SVI loss over epochs](figures/irt_training_loss.png)

  </div>
  <div class="col">

![Ability histogram](figures/ability_hist.png)

![Difficulty histogram](figures/difficulty_hist.png)

  </div>
</div>

---

# Emerging Insights

- Top 10 trees achieve 19–20% accuracy; lowest performers drop below 15%.
- Hardest items (δ > 13) align with CIFAR-10 ships/airplanes confusions.
- Easiest items (δ < −9.5) mostly belong to truck/ship classes with distinctive features.
- Loss curve still descending: consider more epochs or lower lr for finer convergence.

Extremes listed in `data/irt_extremes.json` for manual inspection.

---

# Next Steps

- Drop confusion matrix + new montages into the storytelling deck.
- Enhance notebook to emit report-ready tables and scripted plots.
- Explore alternate embeddings (e.g., MobileNet) to test θ/δ stability.
- Compare tree ability with structural traits (depth, leaves) for richer diagnostics.
- Experiment with 2PL to relate discrimination to RF confidence dynamics.
