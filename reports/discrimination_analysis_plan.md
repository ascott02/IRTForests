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
- All three studies (PCA, MobileNet, MNIST) now ship 800-epoch 2PL fits with refreshed artifacts under `data/*/irt_parameters_2pl.npz` / `data/*/irt_summary_2pl.json` and scatter plots in `figures/*_2pl_*.png`.
- A 1k-epoch 3PL pilot on CIFAR + MobileNet (lr 0.01) converged with guess mean ≈0.25 and θ↔accuracy Pearson 0.98 (`data/mobilenet/irt_summary_3pl.json`).
- `scripts/train_random_forest.py` exports per-tree depth/leaf/OOB stats (`data/*/tree_attributes_with_signals.csv`), while `scripts/analyze_tree_attribute_correlations.py` highlights trends such as MobileNet leaves ↔ θ Pearson −0.78 and OOB ↔ θ Pearson 0.75.
- `scripts/analyze_rf_irt_correlations.py` supports multi-parameter sweeps so δ and \(a\) scatter plots/JSON summaries stay aligned across studies.

## Proposed Experiments
1. **2PL Fits on Existing Studies** ✅ *(Oct 2025)*
   - Each study now has 2PL exports (`data/*/irt_parameters_2pl.npz`, `data/*/irt_summary_2pl.json`, `figures/*_2pl_*`).
   - Follow-up: roll slope stats into `reports/embedding_comparison.md` for quick lookup.
2. **3PL Pilot (Optional)** ✅ *(Oct 2025 — MobileNet complete)*
   - Result: stable convergence with guessing prior mean ≈0.25 and wider slope spread (σ≈0.078).
   - Next decision: test 3PL on PCA/MNIST or document why 2PL suffices for those regimes.
3. **Model Stability Checks**
   - Train reduced forests (50 and 100 trees) on PCA embeddings to see how respondent count affects \(a\) estimates.
   - Repeat if time permits for MobileNet to confirm robustness under stronger features.
4. **Tree Attribute Correlation** ✅ *(Oct 2025)*
   - Delivered via `scripts/analyze_tree_attribute_correlations.py`; outputs sit in `data/*/tree_attribute_correlations*.json` and `figures/*_tree_*.png`.
   - Key takeaways: MobileNet OOB accuracy ↔ θ Pearson 0.75; PCA leaf count ↔ θ Pearson −0.20; MNIST leaf count ↔ θ Pearson −0.47.
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
- `scripts/analyze_tree_attribute_correlations.py`: merges per-tree stats with IRT outputs and emits JSON + figure artifacts.
- Documentation: updated `reports/embedding_comparison.md` and new slide content covering discrimination insights.
- Optional notebook cell(s) to run the entire discrimination suite end-to-end for reproducibility.

## Open Questions
- Do we need a partial-credit or nominal response model for multi-class nuances?
- Should we treat classes separately to inspect discrimination per class group?
- How sensitive are discrimination estimates to label noise versus genuine ambiguity?
- Does MNIST’s near-perfect accuracy compress \(a\) enough that 3PL becomes unnecessary for “easy” datasets?
