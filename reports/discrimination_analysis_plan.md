# Discrimination Analysis Plan

## Context Snapshot
- Response matrices already exist for all three studies:
  - `data/response_matrix.npz` (CIFAR + PCA),
  - `data/mobilenet/response_matrix.npz` (CIFAR + MobileNet),
  - `data/mnist/response_matrix.npz` (MNIST control).
- Qualitative montages now accompany each study in the deck (`figures/*/hardest_*`, `figures/*/easiest_*`), making it straightforward to cross-reference high-δ items once discrimination scores are available.
- Slides close with a “Next Steps” prompt that already calls out the 2PL/3PL extension; this plan should feed directly into that follow-up narrative.

## Objectives
- Extend the RF × IRT study beyond the Rasch/1PL setting to capture item discrimination (\(a\)) and, optionally, guessing (\(c\)).
- Diagnose how tree-level characteristics (depth, leaf count, OOB accuracy) relate to estimated discrimination values.
- Evaluate whether high-discrimination items align with high RF entropy or misclassification clusters across *all* studies, informing data curation and dataset comparisons.

## Progress Update (Oct 2025)
- `scripts/fit_irt.py` now toggles between 1PL/2PL/3PL; the CIFAR + PCA study has an 800-epoch 2PL run cached as `data/irt_parameters_2pl.npz` / `data/irt_summary_2pl.json`.
- Fresh discrimination diagnostics ship alongside the run: `data/rf_irt_correlations_2pl.json`, `figures/2pl_*` scatter plots, and `figures/discrimination_hist.png`.
- Correlation CLI accepts multiple parameters (`--parameters difficulty discrimination`) so δ and \(a\) plots/json summaries stay in sync with new fits.

## Proposed Experiments
1. **2PL Fits on Existing Studies** *(baseline CIFAR + PCA done; replicate for remaining studies)*
   - Inputs: the three response matrices listed above; reuse cached splits to keep comparisons apples-to-apples.
   - Method: run `python scripts/fit_irt.py --model 2pl --response-matrix <path> --epochs 800 --learning-rate 0.02` and export \(a\), \(b\), \(\theta\).
   - Outputs per study: `irt_parameters_2pl.npz`, `irt_summary_2pl.json`, discrimination histograms, \(a\) vs margin/entropy scatter plots.
2. **3PL Pilot (Optional)**
   - Attempt a 3PL fit on the CIFAR + MobileNet run to test whether a guessing parameter \(c\) yields additional insight.
   - Record convergence behaviour and determine whether the added complexity justifies the runtime for other studies.
3. **Model Stability Checks**
   - Train reduced forests (50 and 100 trees) on PCA embeddings to see how respondent count affects \(a\) estimates.
   - Repeat if time permits for MobileNet to confirm robustness under stronger features.
4. **Tree Attribute Correlation**
   - Extract depth, leaf count, and OOB accuracy per tree using scikit-learn estimators.
   - Correlate these attributes with \(\theta\) and \(a\); produce scatter plots and Spearman/Pearson summaries.
   - Prioritize deltas between PCA and MobileNet to highlight how feature richness alters discrimination dynamics.
5. **Item Cluster Analysis**
   - Bucket items into high/medium/low discrimination tiers; overlay with existing qualitative grids for each study.
   - Cross-tab \(a\) with RF entropy, margin, and class labels. Flag items where high discrimination co-occurs with persistent misclassification.
6. **Slide + Report Integration**
   - Summarize discrimination findings in one new slide per study (or a combined cross-study slide) and add a synthesis bullet to the existing “Next Steps” section.
   - Extend `reports/embedding_comparison.md` with a discrimination table to keep notebooks and slides in sync.

## Implementation Notes
- `py-irt` exposes 2PL via `TwoParamLogistic`; confirm support for batching the larger MobileNet matrix. If blocked, pivot to `pyirt` or a lightweight Pyro implementation.
- `scripts/fit_irt.py --model {1pl,2pl,3pl}` now drives all fits; 3PL currently supports only `--priors vague` and returns \(c\) via Beta posteriors.
- Expect slower convergence than Rasch—start with 800 epochs, lower learning rate (0.02), and monitor ELBO.
- Cache intermediate diagnostics (`data/*/irt_training_loss_2pl.npy`) to avoid reruns when tweaking hyperparameters.
- Reuse the new qualitative figures by overlaying discrimination tiers in captions rather than generating fresh grids from scratch.
- Ensure reproducibility: fix seeds, log optimizer settings, hardware, and runtime per study in `reports/rf_irt_summary.json` (extend schema if needed).

## Deliverables
- `scripts/fit_irt.py`: parameterized to toggle 1PL/2PL/3PL and output per-study artifacts.
- `scripts/analyze_rf_irt_correlations.py`: extended to handle \(a\) correlations and generate comparison plots.
- Figures: discrimination histograms, \(a\) vs entropy/margin scatter, tree attribute vs discrimination scatter, item-tier heatmaps.
- `scripts/plot_additional_diagnostics.py`: drops `discrimination_hist.png` whenever slope estimates exist.
- Documentation: updated `reports/embedding_comparison.md` and new slide content covering discrimination insights.
- Optional notebook cell(s) to run the entire discrimination suite end-to-end for reproducibility.

## Open Questions
- Do we need a partial-credit or nominal response model for multi-class nuances?
- Should we treat classes separately to inspect discrimination per class group?
- How sensitive are discrimination estimates to label noise versus genuine ambiguity?
- Does MNIST’s near-perfect accuracy compress \(a\) enough that 3PL becomes unnecessary for “easy” datasets?
