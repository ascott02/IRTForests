# Embedding Comparison

PCA reflects the original 128-D projection, MobileNet uses a pretrained MobileNetV3-Large backbone (960-D).

| Metric | PCA | MobileNet |
|---|---|---|
| Overall accuracy | 0.4335 | 0.8090 |
| Validation accuracy | 0.4235 | 0.8135 |
| OOB accuracy | 0.3630 | 0.7967 |
| Margin mean | -0.0028 | 0.2806 |
| Entropy mean | 2.1503 | 1.4663 |
| Pearson δ↔margin (2PL) | -0.198 | -0.718 |
| Pearson δ↔entropy (2PL) | 0.165 | 0.632 |
| Ability σ (2PL) | 0.155 | 0.338 |
| Difficulty σ (2PL) | 0.070 | 0.104 |

## Discrimination Snapshot (2PL)

| Study | Mean a | σ(a) | Pearson a↔margin | Pearson a↔entropy |
|---|---|---|---|---|
| CIFAR + PCA | 0.286 | 0.081 | -0.734 | 0.596 |
| CIFAR + MobileNet | 0.167 | 0.052 | -0.828 | 0.672 |
| MNIST Mini | 0.240 | 0.164 | +0.893 | -0.958 |

MNIST flips correlation signs because almost every item is easy—high discrimination now marks the rare ambiguous strokes.
