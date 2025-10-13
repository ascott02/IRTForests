# Discrimination Analysis Plan

## Objectives
- Extend the RF × IRT study beyond the Rasch/1PL setting to capture item discrimination (\(a\)) and, optionally, guessing (\(c\)).
- Diagnose how tree-level characteristics (depth, leaf count, OOB accuracy) relate to estimated discrimination values.
- Evaluate whether high-discrimination items align with high RF entropy or misclassification clusters, informing data curation.

## Proposed Experiments
1. **2PL Fit on Existing CFAIR-10 Runs**
   - Input: existing response matrices (`data/response_matrix.npz`, `data/mobilenet/response_matrix.npz`).
   - Method: switch `py-irt` model to 2PL (or alternative library) and record \(a\), \(b\) (difficulty), \(\theta\).
   - Output artifacts: `irt_parameters_2pl.npz`, `irt_summary_2pl.json`, comparison plots (discrimination histograms, \(a\) vs entropy).
2. **Model Stability Checks**
   - Run shorter forests (e.g., 50/100 trees) to confirm discrimination estimates remain stable with fewer respondents.
   - Compare discrimination statistics across PCA vs MobileNet runs.
3. **Tree Attribute Correlation**
   - Extract depth/leaf count per tree from scikit-learn estimators.
   - Correlate with \(\theta\) and 2PL discrimination parameters.
   - Visualize via scatter plots and grouped summaries.
4. **Item Cluster Analysis**
   - Segment items by discrimination (high/medium/low) and inspect original images for qualitative patterns.
   - Cross-tab discrimination with RF entropy, margin, and class labels.

## Implementation Notes
- `py-irt` exposes 2PL via `TwoParamLogistic` (verify API, otherwise adapt or use `pyirt`).
- Increase epochs or adjust learning rate if 2PL loss converges slowly.
- Ensure reproducibility: log seeds, optimizer settings, and runtime per experiment.

## Deliverables
- Updated scripts to fit 2PL and export diagnostics.
- New figures: discrimination histogram, \(a\) vs entropy/margin scatter, tree attribute vs discrimination plots.
- Slide updates summarizing discrimination findings and their implications for RF analysis.

## Open Questions
- Do we need a partial-credit or nominal response model for multi-class nuances?
- Should we treat classes separately to inspect discrimination per class group?
- How sensitive are discrimination estimates to label noise versus genuine ambiguity?
