# MNIST RF × IRT Snapshot

| Metric | Value |
|---|---|
| Train / Val / Test sizes | 4000 / 800 / 800 |
| RandomForest test accuracy | 0.9475 |
| RandomForest validation accuracy | 0.9413 |
| RandomForest OOB accuracy | 0.9140 |
| Mean RF margin | 0.5546 |
| Mean RF entropy | 1.0351 |
| δ ↔ margin (Pearson / Spearman) | −0.950 / −0.979 |
| δ ↔ entropy (Pearson / Spearman) | 0.958 / 0.968 |
| Ability mean ± σ | 4.23 ± 0.44 |
| Difficulty mean ± σ | −1.75 ± 8.19 |

Artifacts:
- Embeddings: `data/mnist_embeddings.npz`
- RF artifacts: `data/mnist/` (metrics, confusion, response matrix, signals)
- IRT outputs: `data/mnist/irt_parameters.npz`, `data/mnist/irt_summary.json`
- Figures: `figures/mnist/*.png`
