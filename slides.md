---

marp: true
theme: default
paginate: true

style: |
  pre {
    vertical-align: text-top;
    font-size: 60%;
    line-height: 1.0;
  }
  .columns {
    display: flex;
    gap: 1em;
  }
  .col {
    flex: 1;
  }
---

# Random Forest × Item Response Theory

- Trees become respondents, images become items.
- Response matrix records per-tree correctness on held-out examples.
- Goal: explain RF behavior via IRT ability & difficulty signals.

---

# Pipeline at a Glance

<div class="columns">
  <div class="col">

**Data Prep**

- Stratified CIFAR-10 subset (train/val/test).
- Resize 64×64, normalize, cache tensors.
- PCA → 128 dims for RF-friendly embeddings.

  </div>

  <div class="col">

**Modeling Plan**

- Fit RandomForestClassifier (200 trees).
- Collect per-tree predictions → response matrix.
- Run 1PL Rasch model → θ (trees) & δ (items).

  </div>
</div>

---

# Roadmap Before Execution

- Confirm virtual env and cached embeddings.
- Flesh out RF training + diagnostics in notebook.
- Implement IRT fit & Wright map visual.
- Push key metrics/plots back into these slides.

---

# Metrics & Visuals Inbound

<div class="columns">
  <div class="col">

**Planned Figures**

- RF accuracy table + confusion matrix heatmap.
- Wright map overlaying θ (trees) vs δ (items).
- Correlation scatter: δ vs RF margin.

  </div>

  <div class="col">

**Data Drops**

- `rf_metrics.json` (acc, OOB, per-class stats).
- `response_matrix.npz` (trees × items correctness).
- `irt_parameters.npz` (θ, δ, discrimination).

  </div>
</div>

---

# Story Beats to Capture

- Which CIFAR-10 classes drive hardest items?
- Do high-ability trees share structural traits?
- How do RF margins align with IRT δ outliers?
- Next experiments: vary n_estimators, explore 2PL.
