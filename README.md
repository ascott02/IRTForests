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
- 2PL discrimination pipeline (`scripts/fit_irt.py --model 2pl`) with new artifacts (`data/irt_parameters_2pl.npz`, `data/irt_summary_2pl.json`, `data/rf_irt_correlations_2pl.json`, `figures/2pl_*`, `figures/discrimination_hist.png`).
- Deck + reports: `slides.md`, `reports/embedding_comparison.md`, `reports/mnist_summary.md`, and supporting figures all synced to the latest runs.
- 2PL fits refreshed for **all** studies (PCA, MobileNet, MNIST) with dataset-specific summaries (`data/*/irt_summary_2pl.json`) and updated scatter plots (`figures/*_2pl_*`).
- 3PL pilot on CIFAR + MobileNet (1k epochs @ 0.01 LR) capturing per-item guessing priors; outputs stored under `data/mobilenet/irt_*_3pl.*`.
- Tree-attribute export + correlation tooling (`scripts/train_random_forest.py --tree-attributes`, `scripts/analyze_tree_attribute_correlations.py`) producing per-tree CSV/JSON reports and figure sets (`figures/*_tree_*.png`).
- Item/person fit-statistics pipeline (`scripts/compute_fit_statistics.py`) writing `person_fit_stats*.csv` and `item_fit_stats*.csv` for every available study/model.

### In Flight / Next
- QA refresh on figures: inject dataset + experiment identifiers into every plot title before publishing.
- Benchmark longer IRT sweeps (≥2000 epochs for 1PL/2PL/3PL) on top of the 2000-tree forests to check convergence margins.
- Integrate item/person fit statistics into `slides.md` (flagging >1.2 underfit or <0.8 overfit cases) and decide how to surface them alongside Wright maps.
- Evaluate whether 3PL adds lift beyond 2PL on MNIST and PCA baselines (or conclude it's unnecessary).
- Stress-test discrimination stability with smaller forests (50/100 trees) and alternate seeds.
- Automate the notebook export so plots/tables land in `reports/` and `slides.md` without manual copy-paste.
- Extend tree-level analysis with additional structural descriptors (path length, feature usage) and link to pruning heuristics.

---

- ## Current Status (PCA vs MobileNet runs)

- **Data prep:** Stratified CIFAR-10 subset (train 10k / val 2k / test 2k). PCA embeddings cached in `data/cifar10_embeddings.npz` (128-D) and MobileNet-V3 features cached in `data/cifar10_mobilenet_embeddings.npz` (960-D).
- **Random Forest (PCA, 2000 trees):** Test **0.468**, validation **0.470**, OOB **0.442**. Per-class stats live in `data/rf_metrics.json`; confusion matrix at `data/rf_confusion.npy`.
- **Random Forest (MobileNet, 2000 trees):** Test **0.819**, validation **0.820**, OOB **0.812** with per-class accuracy range 0.70–0.93 (`data/mobilenet/rf_metrics.json`).
- **Random Forest (MNIST, 2000 trees):** Test **0.954**, validation **0.944**, OOB **0.939** (`data/mnist/rf_metrics.json`).
- **Response matrices:** PCA → `data/response_matrix.npz` (mean tree accuracy **0.176**); MobileNet → `data/mobilenet/response_matrix.npz` (**0.479**); MNIST → `data/mnist/response_matrix.npz` (**0.678**).
- **IRT fits (2PL):**
  - PCA: ability mean **−4.23 ± 0.15**, slope mean **0.35 ± 0.10** (`data/irt_summary_2pl.json`).
  - MobileNet: ability mean **−0.46 ± 0.23**, slope mean **0.27 ± 0.15** (`data/mobilenet/irt_summary_2pl.json`).
  - MNIST: ability mean **3.04 ± 0.29**, slope mean **0.33 ± 0.25** (`data/mnist/irt_summary_2pl.json`).
- **IRT fits (3PL pilot):** MobileNet guess mean **0.35 ± 0.16**, ability-to-accuracy corr **0.977** (`data/mobilenet/irt_summary_3pl.json`).
- **Discrimination correlations:** Updated scatter plots + JSON summaries in `data/*/rf_irt_correlations_2pl.json` (see `figures/*_2pl_*.png`). PCA slope ↔ margin Pearson **−0.83**; MobileNet slope ↔ margin **−0.32**; MNIST slope ↔ margin **0.85** with entropy signs flipped as expected on the near-perfect run.
- **Tree attributes:** Per-tree stats exported as `data/*/tree_attributes_with_signals.csv`; notable trends include MobileNet leaf count vs θ (Pearson **−0.73**) and OOB accuracy vs θ (Pearson **0.70**).
- **Qualitative inspection:** CIFAR-10 hardest/easiest montages (PCA) plus MobileNet + MNIST overlays remain at `figures/*/hardest_*`.
- **Reports:** `reports/embedding_comparison.md` captures cross-study tables; discrimination + tree attribute narrative lives in `reports/discrimination_analysis_plan.md`.
- **Discrimination roadmap:** Updated progress + follow-ups captured in `reports/discrimination_analysis_plan.md`.
- **Tabular reference:** `scripts/run_rf_tabular_example.py` catalogs classic datasets (e.g., breast cancer) with summaries written to `reports/rf_*_summary.json` (overview in `reports/rf_examples.md`).

-Rebuild everything (RF training → signals → IRT → figures) with one command:

```bash
.venv/bin/python scripts/regenerate_all.py
```

Use `--help` for toggles such as `--skip-training` or `--skip-plots` when iterating quickly.


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
