# Embedding Comparison

PCA reflects the original 128-D projection, MobileNet uses a pretrained MobileNetV3-Large backbone (960-D).

| Metric | PCA | MobileNet |
|---|---|---|
| Overall accuracy | 0.4305 | 0.8090 |
| Validation accuracy | 0.4145 | 0.8135 |
| OOB accuracy | 0.3730 | 0.7967 |
| Margin mean | -0.0028 | 0.2806 |
| Entropy mean | 2.1503 | 1.4663 |
| Pearson δ↔margin | -0.8286 | -0.8825 |
| Pearson δ↔entropy | 0.6782 | 0.8113 |
| Ability σ | 0.5473 | 0.2549 |
| Difficulty σ | 4.1029 | 4.6663 |
