# Standard Random Forest Examples

## Breast Cancer Wisconsin (Diagnostic)
- Dataset size: 569 samples, 30 features (binary classification).
- Train/test split: 70/30, stratified, random_state=7.
- RandomForest (200 trees) accuracy: **0.971**, ROC AUC: **0.997**, OOB accuracy: **0.965**.
- Top features: worst concave points, worst perimeter, worst radius, mean concave points, worst area.
- Full JSON summary: `reports/rf_breast_cancer_summary.json`.

## Usage
```bash
source .venv/bin/activate
python scripts/run_rf_tabular_example.py --dataset breast_cancer --output reports/rf_breast_cancer_summary.json
python scripts/run_rf_tabular_example.py --dataset wine --output reports/rf_wine_summary.json
```

The script prints metrics to stdout and saves them to the specified JSON for inclusion in reports or comparisons.
