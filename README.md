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

## Tasks (bite‑sized)

**T0 – Repo Setup**

* [x] Create folders: `notebooks/`, `src/`, `data/`, `figures/`.
* [x] Add `environment.yml` or `requirements.txt`.

**T1 – Data & Embeddings**

* [x] Download CIFAR‑10 via `torchvision`.
* [x] Build embeddings (PCA‑128) cached to `data/cifar10_embeddings.npz` (shape `(12000, 128)`).

**T2 – Random Forest**

* [x] Train `RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1)`.
* [x] Save: accuracy, per‑class accuracy, confusion matrix (`data/rf_metrics.json`, `data/rf_confusion.npy`).
* [x] Save: `feature_importances_` (Gini) and permutation importance (`data/rf_feature_importances.npy`, `data/rf_permutation_importance.csv`).
* [x] Compute per‑example **margin** = p(correct class) − max p(other classes) and per‑item entropy (`scripts/compute_rf_signals.py` → `data/rf_margins.npy`, `data/rf_entropy.npy`).

**T3 – Build Response Matrix R**

* [x] Collect each tree’s predicted label on **test**. `R` stored in `data/response_matrix.npz` (shape `(200, 2000)`).
* [x] Persist `R` (e.g., `np.save('data/response_matrix.npy', R)`).

**T4 – IRT Fit**

* [x] Fit **1PL (Rasch)** using `py-irt` via `scripts/fit_irt.py` (SVI, 600 epochs).
* [x] Extract (\theta) for trees, (\delta) for items (`data/irt_parameters.npz`).
* [x] Validate convergence; plot histograms of (\theta), (\delta) (`figures/ability_hist.png`, `figures/difficulty_hist.png`) and monitor loss (`figures/irt_training_loss.png`).

**T5 – Comparative Analysis**

* [x] Plot **Wright Map**: tree abilities vs item difficulties on shared axis (`figures/wright_map.png`).
* [x] Correlate item difficulty (\delta) with RF **margin** and **entropy**; report Pearson/Spearman (`scripts/analyze_rf_irt_correlations.py` → `data/rf_irt_correlations.json`, scatter plots in `figures/`).
* [ ] Identify top‑10 **hard items** (high (\delta)); visualize and inspect.
* [ ] Class‑wise view: average (\delta) per class vs RF error per class.

**T6 – Slides**

* [ ] Autogenerate `slides.md` sections (see template below) with key plots and tables.

**T7 – Polish**

* [ ] Re‑run with different `n_estimators` (e.g., 50/100/300) to see stability of (\theta), (\delta).
* [ ] Optional: try **2PL** and compare discrimination (a) with RF **tree depth** or **leaf count**.

---

## Current Status (600-epoch 1PL run)

- **Data prep:** Stratified CIFAR-10 subset (train 10k / val 2k / test 2k). PCA embeddings cached in `data/cifar10_embeddings.npz` with 128 features per example.
- **Random Forest:** Overall test accuracy **0.4305**, validation **0.4145**, OOB **0.3730**. Per-class stats logged in `data/rf_metrics.json`; confusion matrix serialized to `data/rf_confusion.npy`.
- **Response matrix:** `data/response_matrix.npz` stores a `(200, 2000)` binary matrix (trees × items) with mean accuracy **0.1759** per tree (see `data/response_summary.json`).
- **IRT fit:** `scripts/fit_irt.py` (SVI, 600 epochs, lr=0.05) yields tree ability mean **−11.14 ± 0.55** and item difficulty mean **5.90 ± 4.10**. Correlations: ability ↔ tree accuracy **0.999**, difficulty ↔ item error **0.950** (`data/irt_summary.json`).
- **Diagnostics:** Parameter histograms (`figures/ability_hist.png`, `figures/difficulty_hist.png`), ability vs. accuracy scatter (`figures/ability_vs_accuracy.png`), difficulty vs. error scatter (`figures/difficulty_vs_error.png`), SVI loss curve (`figures/irt_training_loss.png`). Extremes captured in `data/irt_extremes.json`.
- **RF signals:** Margins average **−0.0028 ± 0.10**, entropy average **2.15 ± 0.13** (`data/rf_signal_summary.json`).
- **Cross-model correlations:** δ vs margin Pearson **−0.83**, δ vs entropy Pearson **0.68** with similar Spearman trends (`data/rf_irt_correlations.json`; plots `figures/difficulty_vs_margin.png`, `figures/difficulty_vs_entropy.png`).
- **Wright map:** Combined θ/δ histogram overlay stored at `figures/wright_map.png` for slide inclusion.

Run the IRT stage end-to-end:

```bash
source .venv/bin/activate
python scripts/fit_irt.py --epochs 600 --learning-rate 0.05 --verbose --log-every 100
python scripts/compute_rf_signals.py
python scripts/analyze_rf_irt_correlations.py
python scripts/plot_wright_map.py
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

## Next Edit Cycle (after RF signal analysis)

Completed this round: RF margins/entropy derived, δ correlations quantified, and Wright map generated for storytelling.

Upcoming priorities:

* [ ] Visualize top-10 hardest/easiest items (thumbnails) and integrate qualitative insights (T5).
* [ ] Extend correlations to include RF confidence/entropy vs. per-class error summaries (T5).
* [ ] Embed new analyses into `notebooks/rf_irt.ipynb` for a reproducible single-run experience (T6).
* [ ] Update `slides.md` with confusion matrix, correlation tables, and item examples once assets are ready (T6).
* [ ] Explore alternative embeddings or RF configurations to test robustness of θ/δ patterns (T7).
