# Random Forest × Item Response Theory (IRT)

**Goal:** Build a small, reproducible study that links *Random Forest (RF)* behavior to *Item Response Theory (IRT)*. We’ll treat **trees as respondents** and **examples as items** to see whether IRT surfaces difficulty/ability signals that complement RF feature importance. Designed to run on **one CPU/GPU** (or none) and to produce a **slides.md** deck for a short talk.

---

## TL;DR (for Copilot / iterative prompting)

Use this README as a **recursive prompt**. Each commit:

1. update this file with the next micro‑goal, 2) run code, 3) paste outputs/plots back here, 4) iterate. Keep changes small.

**Deliverables**

* `slides.md` — talk slides (Marp/Reveal compatible)
* `notebooks/rf_irt.ipynb` — single notebook to run end‑to‑end (data → RF → IRT → plots)
* `src/` — optional python modules if the notebook grows
* `data/` — auto‑downloaded datasets / cached features

---

## Concept

* RF gives **feature importance** and per‑example **margins** but mixes signals.
* IRT (e.g., 1PL/Rasch or 2PL) estimates:

  * **Ability** of a respondent ((\theta)) — here, each **tree’s skill**.
  * **Difficulty** of an item ((\delta)) — here, each **example’s hardness**.
  * (2PL adds **discrimination** (a) — slope/steepness.)
* Mapping:

  * Respondent = **decision tree** in the forest
  * Item = **dataset example** (image)
  * Response = **1 if the tree classifies the example correctly, else 0**
* Hypothesis: IRT identifies **hard examples** (high (\delta)) and **strong trees** (high (\theta)). These can explain **where** permutation/Gini importance succeeds/fails, and guide **data curation** (e.g., mislabeled or ambiguous items).

---

## Dataset & Features

**Default:** CIFAR‑10 (50k train / 10k test). To keep it tractable on CPU:

* Use a **subset** (e.g., 10k train / 2k test) by stratified sampling.
* Extract **compact embeddings**:

  * Option A (fast, CPU‑friendly): resize to 64×64, **PCA** down to 128 dims.
  * Option B (better semantics): **pretrained MobileNetV3‑Small** or **ResNet18** features (frozen), then PCA to 128.
* Save embeddings to `data/cifar10_embeddings.npz` for reuse.

> If images are too heavy: swap to **SVHN** or **Fashion‑MNIST** with the same pipeline.

---

## Pipeline Overview

1. **Load data** → split train/val/test; make embeddings.
2. **Train RandomForestClassifier** on embeddings.
3. **Collect per‑tree predictions** on the test set → binary matrix (R) of shape *(n_trees × n_items)*.
4. **Fit IRT** (1PL first; optionally 2PL) on (R) to estimate (\theta) (tree ability) and (\delta) (item difficulty).
5. **Analyze/compare**:

   * RF **feature importances** (Gini + permutation) vs. IRT **item difficulty** (note: different granularity; we also compute **per‑example margin** from RF to compare directly with (\delta)).
   * Correlate **item difficulty** with **RF margin**, **confidence**, and **error rate by class**.
   * Visualize **Wright Map** (ability vs. difficulty) and overlay class labels.
6. **Report**: plots + short text into `slides.md`.

---

## Milestones

### Completed
- Repo scaffolding (`notebooks/`, `src/`, `data/`, `figures/`) with reproducible `requirements.txt`.
- CIFAR-10 embeddings (PCA + MobileNet) and MNIST control run cached under `data/`.
- Random Forest training, signal extraction, and confusion diagnostics stored for every study.
- Response matrix builds and 1PL (`Rasch`) IRT fits with ability/difficulty exports and plots.
- Comparative analysis pipeline: Wright maps, δ↔margin/entropy correlations, class difficulty summaries, qualitative hardest/easiest grids.
- Deck + reports: `slides.md`, `reports/embedding_comparison.md`, `reports/mnist_summary.md`, and supporting figures all synced to the latest runs.

### In Flight / Next
- Extend `scripts/fit_irt.py` to toggle 2PL/3PL fits and capture discrimination parameters (see `reports/discrimination_analysis_plan.md`).
- Correlate discrimination (a) with tree structure (depth, leaves, OOB accuracy) via `scripts/analyze_rf_irt_correlations.py` updates.
- Automate the notebook export so plots/tables land in `reports/` and `slides.md` without manual copy-paste.
- Probe parameter stability with smaller forests (50/100 trees) before scaling the discrimination study.

---

- ## Current Status (PCA vs MobileNet runs)

- **Data prep:** Stratified CIFAR-10 subset (train 10k / val 2k / test 2k). PCA embeddings cached in `data/cifar10_embeddings.npz` (128-D) and MobileNet-V3 features cached in `data/cifar10_mobilenet_embeddings.npz` (960-D).
- **Random Forest (PCA):** Overall test accuracy **0.4305**, validation **0.4145**, OOB **0.3730**. Per-class stats logged in `data/rf_metrics.json`; confusion matrix serialized to `data/rf_confusion.npy`.
- **Random Forest (MobileNet):** Overall test accuracy **0.8090**, validation **0.8135**, OOB **0.7967** with per-class accuracies 0.68–0.92 (`data/mobilenet/rf_metrics.json`).
- **Response matrices:** PCA run stored at `data/response_matrix.npz` (mean tree accuracy **0.1759**); MobileNet run stored at `data/mobilenet/response_matrix.npz` (mean tree accuracy **0.4817**).
- **IRT fits:** PCA summary in `data/irt_summary.json` (θ mean **−11.14 ± 0.55**, δ mean **5.90 ± 4.10**); MobileNet summary in `data/mobilenet/irt_summary.json` (θ mean **−0.21 ± 0.25**, δ mean **0.07 ± 4.67**). Both trained for 600 epochs via `scripts/fit_irt.py`.
- **Diagnostics:** Shared plots for PCA (`figures/*.png`) and MobileNet (`figures/mobilenet/*.png`) cover histograms, loss curves, Wright maps, and δ vs signal scatterplots.
- **RF signals:** PCA margins average **−0.0028**, entropy **2.15**; MobileNet margins **0.2806**, entropy **1.47** (JSON summaries in `data/rf_signal_summary.json` and `data/mobilenet/rf_signal_summary.json`).
- **Cross-model correlations:** PCA δ↔margin Pearson **−0.83**, δ↔entropy **0.68** vs. MobileNet δ↔margin **−0.88**, δ↔entropy **0.81** (`data/rf_irt_correlations.json`, `data/mobilenet/rf_irt_correlations.json`).
- **Qualitative inspection:** CIFAR-10 hardest/easiest montages (PCA) plus new MobileNet Wright map for improved alignment between ability and accuracy.
- **Reports:** `reports/embedding_comparison.md` captures a side-by-side metric table; notebook exports still provide class summaries and run metadata (`reports/class_difficulty_summary.md`, `reports/rf_irt_summary.json`).
- **Discrimination roadmap:** See `reports/discrimination_analysis_plan.md` for the 2PL/ability-depth follow-up design.
- **MNIST variant:** `scripts/build_mnist_embeddings.py` + `scripts/train_random_forest.py` yield a compact MNIST run (test acc 94.8%) with artifacts under `data/mnist/`; summary logged in `reports/mnist_summary.md`.
- **Tabular reference:** `scripts/run_rf_tabular_example.py` catalogs classic datasets (e.g., breast cancer) with summaries written to `reports/rf_*_summary.json` (overview in `reports/rf_examples.md`).

Run the IRT stage end-to-end:

```bash
source .venv/bin/activate
python scripts/fit_irt.py --epochs 600 --learning-rate 0.05 --verbose --log-every 100
python scripts/compute_rf_signals.py
python scripts/analyze_rf_irt_correlations.py
python scripts/plot_wright_map.py
python scripts/visualize_difficulty_extremes.py --split test --count 10
python scripts/class_difficulty_summary.py
python scripts/plot_confusion_matrix.py --normalize
```


## Notebook Skeleton (`notebooks/rf_irt.ipynb`)

**Sections**

- Background on Item Response Theory (IRT)
- Background on Random Forests (RF)
- Experimental Procedure
    1. Setup & imports
    2. Data download/sampling
    3. Embedding pipeline (choose A/B)
    4. Train RF; metrics; importances
    5. Build response matrix; save
    6. IRT fit (1PL); diagnostics
    7. Correlations & plots
    8. Slide export (write markdown to `slides.md`)

**Key imports**

```python
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
# torchvision + torch if using CNN features
# from py_irt import irt  # or `import pyirt` as fallback
```

---

## Slide Template (`slides.md`)

Use Marp/Reveal style. Start with minimal bullets; paste in plots as they’re produced.

```markdown
---
marp: true
class: invert
---
# RF × IRT: Difficulty & Ability Signals

- Trees = respondents; examples = items
- Response = 1 if tree predicts correctly
- 1PL IRT on response matrix R

---
# Data & Embeddings
- Dataset: CIFAR‑10 (subset)
- Embedding: PCA‑128 (or MobileNetV3→PCA)

---
# Random Forest Results
- Acc: …; Per‑class Acc: …
- Importances (Gini vs Permutation)

---
# IRT Results (1PL)
- Distribution of θ (trees) and δ (items)
- Wright Map

---
# Comparison
- corr(δ, RF margin) = …
- Hard items (top‑10) with thumbnails

---
# Takeaways
- IRT exposes hard items & strong trees
- Complements RF importances
- Next: 2PL, active data curation
```

---

## Slide Deck Outline (v2)

1. **Orientation**
  - Title, motivation, RF & IRT primers, pipeline recap.
  - Global dataset overview (CIFAR-10 subset + MNIST mini-study) with split tables.
2. **Study I — CIFAR-10 + PCA-128** (baseline)
  - Setup slide: dataset slice, embedding recipe, caching paths.
  - RF performance slide: accuracy table, per-class range, tree/margin stats.
  - IRT diagnostics slide: θ vs accuracy, δ vs RF signals (margin/entropy), δ vs error.
  - Evidence slide: Wright map, hardest/easiest montage, key edge cases.
3. **Study II — CIFAR-10 + MobileNet-V3** (transfer features)
  - Setup slide mirroring Study I (data reuse, embedding dimensionality, storage).
  - RF performance slide with identical table schema for easy comparison.
  - IRT diagnostics slide mirroring Study I (δ vs signals, θ spread, Wright map excerpt if available).
  - Evidence slide: highlight persistent edge cases, note improvements vs PCA.
4. **Study III — MNIST Mini-Study**
  - Setup slide (sampling, preprocessing, embedding/feature choices).
  - RF performance slide (same metric table schema).
  - IRT diagnostics slide (δ vs signals, ability spread, confusion snippets if available).
  - Evidence slide: ambiguous digit gallery, link to artifacts.
5. **Cross-Study Synthesis**
  - Edge-case comparison across datasets.
  - Class-level difficulty vs error overlay.
  - Training loss, histograms, and actionable next steps.

> Maintain uniform headings and table formats so each study tells the same story: setup → RF metrics → IRT diagnostics → qualitative evidence → takeaways.

---

## Installation

`requirements.txt` (suggested)

```
numpy
pandas
scikit-learn
matplotlib
seaborn
torch
torchvision
py-irt  # or pyirt if unavailable
marp-cli  # optional for slide rendering
```

> If `py-irt` fails to install, try `pyirt`, or switch to a simple **Rasch** implementation (there are minimal Python examples) for binary responses.

---

## Evaluation & Reporting

* Primary: overall/cls accuracy; **CIDEr not applicable** here (classification task).
* New: **Wright Map**, **δ vs margin** correlation, **hard examples** inspection.
* Log **runtime** and **hardware** used; report **SUs (GPU hours)** if any.

---

## Stretch Ideas

* Use **leaf path length** or **node depth** as a proxy for item difficulty; compare with (\delta).
* Train RFs with different **feature subsets** to test stability of IRT parameters.
* Active selection: prioritize items with high (\delta) for **label audit**.

---

## Repro Notes

* Set seeds (`np.random.seed`, `torch.manual_seed`).
* Cache embeddings to avoid recompute.
* Keep subsets fixed across runs for fair comparison.

---

## Next Edit Cycle

Focus shifts to discrimination analysis: implement 2PL fits, extend correlation tooling, and feed condensed takeaways back into `slides.md` and the reports set. Notebook automation should follow once the new experiments stabilize.
