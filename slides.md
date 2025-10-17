---

marp: true
theme: default
paginate: true
class: invert
math: katex
style: |
  section {
    font-size: 140%;
    width: 1280px;
    height: 720px;
    padding-top: 40px;
    padding-bottom: 40px;
    padding-left: 100px;
    padding-right: 100px;
  }
  ul {
    line-height: 1.2;
  }
  li {
    margin-bottom: 0em;
  }
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
  img {
    border-radius: .2em;
  }
footer: ATS &copy; 2025
---

# IRTForests 

<p style="font-size:80%; margin-top:2.5em;">Andrew T. Scott, Fall 2025</p>
<p style="font-size:70%;"><a href="https://github.com/ascott02/IRTForests">github.com/ascott02/IRTForests</a></p>

---

# Item Response Theory + Random Forests

- Trees become respondents, images become items.
- Response matrix records per-tree correctness on held-out examples.
- Goal: explain RF behavior via IRT ability & difficulty signals and vice versa.

---

# Agenda

- Background: IRT Background, RF Background
- Pipeline: Datasets, embeddings, and response matrices powering the studies.
- Case Studies: Baseline CIFAR (PCA), CIFAR (MobileNet), and MNIST.
- Cross-study comparison, takeaways, and next steps.

---

# Why Item Response Theory (IRT)?

**Because performance != ability — but they’re related.**


- Classical Test Theory (CTT) tells us *how someone did on this test.*

- IRT models *how someone would perform on any set of items that measure the same underlying ability*.

- IRT doesn’t replace CTT, it generalizes it with **portable, interpretable measurements** of capability.

<center>

| CTT | IRT |
|-----------------------|----------------------|
| Measures perf. on specific test | Estimates underlying ability |
| Test = sample of items | Items = samples from a calibrated continuum |
| Precision assumed constant | Precision varies with ability |
| Great for grading | Great for understanding and interpretability |

</center>

> A joint calibration framework where ability and difficulty are inferred together, each defined only in relation to the other. It’s less like grading individuals and more like synchronizing clocks — each calibrated against the ensemble.



---


# Item Response Theory Building Blocks

  <div class="col">

## **Core Terms**

- Ability (θ): respondent skill; higher → higher success odds.
- Difficulty (δ): item hardness; higher → harder even for strong respondents.
- Discrimination (𝑎): slope near δ (2PL).
- Guessing (𝑐): floor for multiple-choice exams (rare here) (3PL).

  </div>
  <div class="col">

## **Translated to Tree Ensemble Analogy**

- Respondents → decision trees on a shared test set.
- Items → images; responses are binary (tree correct?).
- Response matrix $R_{ij} \in \{0,1\}$ feeds variational IRT.
- Outputs: posteriors over θᵢ, δⱼ, and information curves.

  </div>

---

# Rasch (1PL) Model in One Picture

<div class="columns">
  <div class="col">

$$\Pr(R_{ij}=1 \mid \theta_i, \delta_j) = \frac{1}{1 + e^{- (\theta_i - \delta_j)}}$$

- Single global slope keeps parameters on a shared logit scale.
- θ − δ = 0 ⇒ 50% success; shifts left/right change odds.
- Fisher information peaks where curves are steepest.
- See <a href="https://ascott02.github.io/irt.html">IRT ICC Visualizer</a> for 2PL, 3PL, 4PL

  </div>
  <div class="col" style="text-align:center;">
  <center>
    <img width="85%" src="figures/irt/rasch_curve.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%;">1PL Item Characteristic Curves (ICC)</p>
    </center>
  </div>
</div>

---

# IRT Output

- **Ability histograms** flag low-skill trees worth pruning.
- **Difficulty ladders** highlight mislabeled or ambiguous items.
- **Wright maps** overlay θ and δ to expose coverage gaps.
- **Information curves** reveal where ensemble confidence is fragile.
- Together they explain *who* struggles and *why*, beyond RF metrics.


---

# Random Forests — Many Noisy Trees, One Stable Voice

A **Random Forest** grows lots of trees on **bootstrapped (OOB)** samples  
and **random subsets of features** at each split.

This randomness:
- decorrelates trees → lowers variance,  
- lets each tree explore a different view of the data.

Final prediction = **majority vote** (classification) or **mean** (regression).  

**Margins:** how much more the correct class wins over the runner-up → a measure of confidence.  
**Entropy:** how uncertain the ensemble is — low entropy = decisive forest, high entropy = disagreement.

- Trees are fragile storytellers; forests are resilient crowds.  
- Margins and entropy tell you how loudly, and how harmoniously, that crowd speaks.
- Combining both with δ surfaces mislabeled or OOD items and tracks curation gains.

---

# Random Forest Margins — How Confident Is the Crowd?

$$ \text{margin}(x_i) =
P_{\text{correct}}(x_i)
- \max_{j \neq \text{true}} P_j(x_i) $$

The **margin** measures how far ahead the correct class is
over its nearest competitor.

- **High margin:** trees vote strongly for the right class → confident.  
- **Low or negative margin:** trees disagree or favor another class → uncertain.  

> Think of it as the *vote gap* in an election — the wider the gap, the clearer the win.

---

# Ensemble Entropy — How Much Do Trees Disagree?

$$ H(x_i) = - \sum_j P_j(x_i) \log_2 P_j(x_i) $$

The **entropy** measures how dispersed the votes are across classes.

- **Low entropy:** trees nearly unanimous → decisive prediction.  
- **High entropy:** votes spread out → uncertainty or class confusion.  

> Within trees, entropy drives splits (purity).  
> Across trees, entropy reveals disagreement — the forest’s collective uncertainty.

---

# GenAI in the Loop Scientific Experimentation

- Recursive Prompting, (similar to <a href="https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents">Context Engineering</a>)
- Start from focused `README.md` spec outlining goals, datasets, and diagnostics.
- Automate CLI runs to iterate every experiment end-to-end.
- Promoted results/figures/tables into this deck!
- Commit, push, rinse, and repeat...
- <a href="https://github.com/ascott02/IRTForests">github.com/ascott02/IRTForests</a>

> Plastic tubes and pots and pans
> Bits and pieces and the magic from the hand - Oingo Boingo, "Weird Science" 1985

---


# Pipeline Overview

<div class="columns">
  <div class="col">

**Data Preperation for 3 Experiements**

1. Stratified CIFAR-10 subset: 10k / 2k / 2k splits. Resize 64×64, normalize, PCA → 128-D embeddings.
2. Stratified CIFAR-10 subset: 10k / 2k / 2k splits. Resize 64×64, normalize, MobileNet → 960-D embeddings.
3. MNIST mini: 4k / 800 / 800 digits, normalized 28×28 grayscale. Raw pixels.

**Random Foreset Training**

- RF (2000 trees) trained for every study; metrics and importances saved.
- Response matrices saved: CIFAR `(2000 × 2000)` for PCA & MobileNet, MNIST `(2000 × 800)`.

**IRT Analsysis**
- 1PL Rasch (SVI, 600 epochs) complete for CIFAR+PCA, CIFAR+MobileNet, and MNIST.
- 2PL (SVI, 800 epochs) complete for CIFAR+PCA, CIFAR+MobileNet, and MNIST.
- 3PL (SVI, 1000 epochs) CIFAR MobileNet only.

  </div>
</div>

---

# Datasets Overview

| Dataset | Train | Val | Test | Feature Pipeline | Notes |
|---|---|---|---|---|---|
| CIFAR-10 subset | 10,000 | 2,000 | 2,000 | PCA-128 / MobileNet-V3 (960-D) | Shared splits Study I & II |
| MNIST mini | 4,000 | 800 | 800 | 28×28 grayscale → raw pixels (no PCA) | Control for clean handwriting |

- CIFAR runs differ only by embeddings; labels and splits stay fixed.
- MNIST mirrors the workflow to confirm signals on cleaner data.

---


# Study I Setup: CIFAR-10 + PCA-128

<div class="columns">
  <div class="col">
    <ul>
  <li>Establish the PCA baseline and its uncertainty signals.</li>
  <li>Use IRT to pinpoint weak trees and hard items that motivate stronger features.</li>
  <li>Fixed stratified CIFAR-10 split (10k/2k/2k).</li>
  <li>Resize 64×64, normalize, PCA → 128-D embeddings.</li>
  <li>Train and test 2000 trees.
  <li>Response matrix 2000 × 2000 with mean tree accuracy.</li>
  <li>Artifacts: Metrics, Margins, Entropy, IRT outputs.</li>
    </ul>
  </div>

  <div class="col">
  <center>

<img width="85%" src="figures/datasets/study1_cifar_samples.png" style="width:100%; border:1px solid #ccc;" />
<p style="font-size:85%; text-align:center;">Study I sample grid — stratified CIFAR-10 slices</p>
    </center>
  </div>
</div>

---

# Study I Performance (PCA-128)

<center>

| Metric | Value |
|---|---|
| Test / Val / OOB acc | 0.468 / 0.470 / 0.442 |
| Per-class range | 0.260 (bird) → 0.635 (ship) |
| Mean tree accuracy | 0.1763 |
| Mean margin / entropy | 0.0058 / 2.1723 |
| δ negatively correlates with margin (Pearson) | −0.815 |
| δ positively correlates with entropy (Pearson) | 0.687 |

</center>

- Baseline ensemble still underperforms due to weak PCA features yet preserves δ alignment.
- Margins hover near zero (mean ≈0.006) and entropy stays high (2.17), signalling broad disagreement—prime for IRT.

---

# Study I Confusion Matrix

<div class="columns">
  <div class="col">
    <img src="figures/rf_confusion_matrix.png" style="width:95%; border:1px solid #ccc;" />
  </div>
  <div class="col">

- Off-diagonal spikes (cat vs dog, bird vs airplane, horse vs deer) mirror high-δ items.
- Ships/trucks stay >80% on-diagonal; the highlighted hotspots mark curation targets.

  </div>
</div>

---

# Study I Diagnostics: Ability Profiles

<div class="columns">
  <div class="col">
  <center>
    <img width="85%" src="figures/ability_vs_accuracy.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Ability (θ) vs tree accuracy — Spearman ≈ 0.99</p>
    </center>
  </div>
  <div class="col">
  <center>
  <img width="84%" src="figures/wright_map.png" style="width:95%; border:1px solid #ccc;" />
  <p style="font-size:85%; text-align:center;">Wright map: θ around −4; δ spans roughly [−0.5, 0.6]</p>
    </center>
  </div>
</div>

- θ spans roughly −4.7 to −3.7; a +0.2 shift in ability still separates stronger trees by ~3 pp.
- δ clusters near zero but stretches past ±0.5, flagging the ambiguous animal images against a compressed ability band.

---

# Study I Diagnostics: δ vs Error Rate

<div class="columns">
  <div class="col">
    <img src="figures/difficulty_vs_error.png" style="width:95%; border:1px solid #ccc;" />
  </div>
  <div class="col">

- δ > 0.4 maps to >80% tree error—mostly ambiguous animals—while δ < −0.3 becomes “free points.”
- Pearson ≈ 0.87, Spearman ≈ 0.86: difficulty doubles as an error heat-map.

  </div>
</div>

---

# Study I Diagnostics: δ vs RF Signals

<div class="columns">
  <div class="col">
  <center>
  <img src="figures/difficulty_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%;">PCA run: δ vs margin (Pearson −0.82)</p>
    </center>
  </div>
  <div class="col">
  <center>
  <img src="figures/difficulty_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%;">PCA run: δ vs entropy (Pearson 0.69)</p>
    </center>
  </div>
</div>

- Hard items cluster bottom-right (low margin, high entropy); opposite corner houses easy wins.
- Study II mirrors the trend with even stronger correlations.

---

# Study I Evidence: Hard vs Easy Examples

<div class="columns">
  <div class="col">

![Hardest items](figures/hardest_items_test.png)

  </div>
  <div class="col">

![Easiest items](figures/easiest_items_test.png)

  </div>
</div>

- Hardest items skew toward ambiguous airplane/ship silhouettes and cluttered cat/dog scenes.
- Easy set is dominated by high-contrast cues (e.g., red fire trucks), yielding low δ and entropy.

---

# Study I Takeaways

- Weak PCA features create long tails in both ability (θ) and difficulty (δ), exposing erratic trees.
- Margin and entropy correlate with δ, but clusters of high-difficulty animals persist across diagnostics.
- Visual inspection confirms mislabeled or low-signal items driving high δ, motivating feature upgrades.

---

# Study I Fit Checks & Edge Cases

<div class="columns">
  <div class="col">

**Fit diagnostics**

| Metric | Value |
|---|---|
| Item infit μ / p95 | 0.18 / 0.35 |
| Item outfit μ / p95 | 0.18 / 0.34 |
| Tree infit μ / p95 | 0.35 / 0.48 |
| Tree outfit μ / p95 | 0.18 / 0.19 |

- MSQs well below 1 show tree responses are steadier than a pure Rasch prior; |z| never exceeds 0.05.

  </div>
  <div class="col">

**Edge cases worth a look**

- `#118` bird → deer votes (δ ≈ 13.4, margin ≈ −0.05, entropy ≈ 2.28).
- `#1734` truck → cat/frog split (δ ≈ 13.2, margin ≈ −0.09, entropy ≈ 2.27).
- `#1602` horse → dog/horse tie (δ ≈ 13.2, margin ≈ −0.11, entropy ≈ 2.22).

- Each item sits below 9% tree accuracy—prime targets for relabeling or curated augmentations.

<center>
    <img width="60%" src="figures/study1_edge_cases.png" style="width:100%; border:1px solid #ccc; margin-top:0.8em;" />
    <p style="font-size:75%; text-align:center;">Study I edge cases · IDs 118, 1734, 1602</p>
</center>

  </div>
</div>

---

# Section II · Feature-Rich CIFAR (MobileNet)

- Hold the splits fixed to isolate feature gains.
- Test whether richer embeddings tighten θ spread and retain δ alignment.

---

# Study II: CIFAR-10 + MobileNet Embeddings

- Swap PCA features for MobileNet-V3 (960-D) while keeping tree count and splits constant.
- Compare RF metrics, uncertainty signals, and IRT parameters against the baseline.

---

# Study II Setup: CIFAR-10 + MobileNet-V3

<div class="columns">
  <div class="col">
    <ul>
  <li>Reuse Study I splits to isolate feature effects.</li>
  <li>Extract 960-D MobileNet-V3 Small embeddings (`data/cifar10_mobilenet_embeddings.npz`).</li>
  <li>Response matrix 2000 × 2000 with mean tree accuracy 0.479.</li>
  <li>Artifacts live under `data/mobilenet/*` and `figures/mobilenet/`.</li>
    </ul>
  </div>
  <div class="col">
    <img src="figures/datasets/study2_cifar_samples.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Study II sample grid — same splits, MobileNet embeddings</p>
  </div>
</div>

---

# Study II Performance (MobileNet-V3)

| Metric | Value |
|---|---|
| Test / Val / OOB acc | 0.819 / 0.820 / 0.812 |
| Per-class range | 0.695 (bird) → 0.925 (ship) |
| Mean tree accuracy | 0.4792 |
| Mean margin / entropy | 0.2806 / 1.4929 |
| δ negatively correlates with margin (Pearson) | −0.950 |
| δ positively correlates with entropy (Pearson) | 0.881 |

- Pretrained features boost accuracy by 35 pp while strengthening δ correlations.
- Higher margins and lower entropy show confidence gains except on stubborn animal classes.
- Artifacts: metrics, response matrix, signals, and IRT outputs under `data/mobilenet/`.

---

# Study II Diagnostics: δ vs RF Signals

<div class="columns">
  <div class="col">
  <center>
    <img width="85%" src="figures/mobilenet/mobilenet_difficulty_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%;">δ vs margin (Pearson −0.95)</p>
  </center>
  </div>
  <div class="col">
  <center>
    <img width="85%" src="figures/mobilenet/mobilenet_difficulty_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%;">δ vs entropy (Pearson 0.88)</p>
  </center>
  </div>
</div>

- MobileNet compresses the easy cluster (high margin, low entropy) while isolating true hard cases.
- Larger |corr| values show tighter agreement between δ and RF uncertainty.
- Cat/dog confusions persist, marking curation targets.

---

# Study II Diagnostics: Ability Profiles

<div class="columns">
  <div class="col">
  <center>
    <img width="85%" src="figures/mobilenet/ability_vs_accuracy.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%; text-align:center;">Ability (θ) vs tree accuracy — Pearson 0.96</p>
  </center>
  </div>
  <div class="col">
  <center>
    <img width="84%%" src="figures/mobilenet/wright_map.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%; text-align:center;">Wright map: θ ≈ −0.46 ± 0.23; δ spans ±2.1</p>
  </center>
  </div>
</div>

- θ mean −0.46 ± 0.23 keeps the ensemble tightly banded while still ranking trees cleanly.
- Ability remains tied to per-tree accuracy, so feature quality—rather than tree diversity—now caps gains.

---

# Study II Diagnostics: δ vs Error Rate

<div class="columns">
  <div class="col">
    <img src="figures/mobilenet/difficulty_vs_error.png" style="width:95%; border:1px solid #ccc;" />
  </div>
  <div class="col">

- Pearson 0.99 keeps δ aligned with mean tree error even at the higher accuracy ceiling.
- Hardest items (δ > 1.5) persist—mostly cat/dog overlaps and ambiguous aircraft—while the easy zone (δ < −1) expands.

  </div>
</div>

---

# Study II Evidence: Hard vs Easy Examples

<div class="columns">
  <div class="col">

![MobileNet hardest items](figures/mobilenet/hardest_items_test.png)

  </div>
  <div class="col">

![MobileNet easiest items](figures/mobilenet/easiest_items_test.png)

  </div>
</div>

- MobileNet tightens easy clusters yet the same cat/dog outliers survive with δ > 1.5.
- Easy wins sharpen into high-contrast ships and trucks, showing how feature upgrades cleanly separate low-δ items.

---

# Study II Takeaways

- MobileNet embeddings add 35 pp of accuracy while maintaining a focused ability band (Std(θ) ≈ 0.23).
- δ stays aligned with RF uncertainty, isolating a smaller yet stubborn ambiguous cluster.
- Residual cat/dog confusion points to data curation as the next lever.

---

# Study II Fit Checks & Edge Cases

<div class="columns">
  <div class="col">

**Fit diagnostics**

| Metric | Value |
|---|---|
| Item infit μ / p95 | 0.27 / 0.37 |
| Item outfit μ / p95 | 0.27 / 0.37 |
| Tree infit μ / p95 | 0.29 / 0.31 |
| Tree outfit μ / p95 | 0.27 / 0.29 |

- Narrow MSQ spread (≤0.37) confirms MobileNet trees behave consistently; no misfit flags at |z| > 0.05.

  </div>
  <div class="col">

**Edge cases worth a look**

- `#1190` automobile → frog votes (δ ≈ 15.4, margin ≈ −0.22, entropy ≈ 1.85; top probs frog 0.28, deer 0.27).
- `#1196` bird → horse (δ ≈ 14.9, margin ≈ −0.38, entropy ≈ 1.31; horse 0.41, deer 0.41, bird 0.08).
- `#95` frog → bird (δ ≈ 14.8, margin ≈ −0.25, entropy ≈ 1.89; bird 0.32, deer 0.20, frog 0.17).

- These persistent outliers survive the feature upgrade—queue them for image/label review next.
<center>
    <img width="60%" src="figures/study2_edge_cases.png" style="width:100%; border:1px solid #ccc; margin-top:0.8em;" />
    <p style="font-size:75%; text-align:center;">Study II edge cases · IDs 1190, 1196, 95</p>
</center>

  </div>
</div>

---


# Section III · Control Study (MNIST)

- Probe the pipeline on a high-signal, low-noise dataset.
- Confirm that IRT still mirrors RF uncertainty when accuracy is near perfect.

---

# Study III: MNIST Mini-Study

- Lightweight handwriting dataset to validate RF × IRT beyond CIFAR-10.
- Acts as a control where ambiguity is rare yet still detectable.

---

# Study III Setup: MNIST Mini-Study

<div class="columns">
  <div class="col">
    <ul>
  <li>Split 4k / 800 / 800 digits with stratified sampling and a fixed seed.</li>
  <li>Flatten 28×28 grayscale digits; no augmentation.</li>
  <li>Train a 2000-tree RF on raw pixels; response matrix 2000 × 800.</li>
  <li>Artifacts land in `data/mnist/` with plots in `figures/mnist/`.</li>
    </ul>
  </div>
  <div class="col">
    <img src="figures/datasets/mnist_samples.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:85%; text-align:center;">Study III sample grid — curated MNIST mini split</p>
  </div>
</div>

---

# Study III Performance (MNIST)

| Metric | Value |
|---|---|
| Train / Val / Test | 4000 / 800 / 800 |
| RF test / val / OOB | 0.954 / 0.944 / 0.939 |
| Mean margin / entropy | 0.5644 / 1.0768 |
| δ negatively correlates with margin (Pearson) | −0.975 |
| δ positively correlates with entropy (Pearson) | 0.970 |
| θ mean ± std | 3.04 ± 0.29 |
| δ mean ± std | −0.13 ± 0.47 |

- Ambiguous digits (e.g., brushed 5 vs 6) still spike δ toward the positive tail; elsewhere the forest is decisive.
- Low entropy + high margin line up with low δ, giving a “sanity benchmark” beyond CIFAR.

---

# Study III Diagnostics: δ vs RF Signals

<div class="columns">
  <div class="col">
    <img src="figures/mnist/mnist_difficulty_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%;">δ vs margin (Pearson −0.97)</p>
  </div>
  <div class="col">
    <img src="figures/mnist/mnist_difficulty_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%;">δ vs entropy (Pearson 0.97)</p>
  </div>
</div>

- Clean digits show near-perfect alignment between δ and RF uncertainty.
- Only a handful of δ > 1.2 digits drive the residual uncertainty (stroke collisions like 3/5, 4/9).

---

# Study III Diagnostics: Ability Profiles

<div class="columns">
  <div class="col">
  <center>
    <img width="75%" src="figures/mnist/ability_vs_accuracy.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%; text-align:center;">Ability (θ) vs tree accuracy — Pearson 0.98</p>
  </center>
  </div>
  <div class="col">
  <center>
    <img width="74%" src="figures/mnist/wright_map.png" style="width:100%; border:1px solid #ccc;" />
  <p style="font-size:85%; text-align:center;">Wright map: θ mean 3.04 ± 0.29; δ mean −0.13 ± 0.47</p>
  </center>
  </div>
</div>

- θ mean 3.04 ± 0.29 shows strong consensus, while δ mean −0.13 ± 0.47 keeps a modest positive tail for ambiguous strokes.
- Shared scales expose plentiful easy wins with only a few sharp spikes—opposite of the CIFAR baseline.


---

# Study III Diagnostics: δ vs Error Rate

<div class="columns">
  <div class="col">
    <img src="figures/mnist/difficulty_vs_error.png" style="width:95%; border:1px solid #ccc;" />
  </div>
  <div class="col">

- Pearson 0.98 keeps δ tied to mean tree error despite the high accuracy ceiling.
- δ > 1.2 corresponds to stroke-collided 3/5/8 and 4/9 pairs; the long negative tail is trivial for the ensemble.

  </div>
</div>

---

# Study III Evidence: Hard vs Easy Digits

<div class="columns">
  <div class="col">

![MNIST hardest digits](figures/mnist/hardest_digits_test.png)

  </div>
  <div class="col">

![MNIST easiest digits](figures/mnist/easiest_digits_test.png)

  </div>
</div>

- Hardest digits show stroke collisions (3 vs 5, 4 vs 9) that push δ above 1 despite high margins elsewhere.
- Easy digits are crisp, centered strokes—useful anchors when explaining why δ plunges on most of the dataset.

---

# Study III Takeaways

- δ and RF uncertainty agree almost perfectly, while θ stays high yet still flags the rare ambiguous strokes.
- The control study confirms the RF × IRT pipeline holds outside noisy vision data.

---

# Study III Fit Checks & Edge Cases

<div class="columns">
  <div class="col">

**Fit diagnostics**

<center>

| Metric | Value |
|---|---|
| Item infit μ / p95 | 0.23 / 0.38 |
| Item outfit μ / p95 | 0.22 / 0.37 |
| Tree infit μ / p95 | 0.30 / 0.32 |
| Tree outfit μ / p95 | 0.22 / 0.25 |
</center>

- Rasch residuals stay tight (|z| < 0.07), confirming the control study’s consistency.

  </div>
  <div class="col">

**Edge cases worth a look**

- `#296` digit 0 → vote 7 (δ ≈ 17.6, margin ≈ −0.35, entropy ≈ 1.83; top probs 7=0.38, 9=0.18, 4=0.16).
- `#151` digit 9 → vote 6 (δ ≈ 17.3, margin ≈ −0.34, entropy ≈ 1.93; top probs 6=0.39, 5=0.12, 2=0.10).
- `#708` digit 4 → vote 3 (δ ≈ 16.3, margin ≈ −0.08, entropy ≈ 2.10; top probs 3=0.19, 4=0.18, 9=0.15).

- Archive these strokes for a “confusing digits” gallery or curation playbook.

<center>
    <img width="60%" src="figures/study3_edge_cases.png" style="width:100%; border:1px solid #ccc; margin-top:0.8em;" />
    <p style="font-size:75%; text-align:center;">Study III edge cases · IDs 296, 151, 708</p>
<center>

  </div>
</div>

---

# Section IV · Cross-Study & Diagnostics

- Compare backbones and datasets on a shared θ/δ scale.
- Surface recurring themes before the close.

---

# Cross-Study Snapshot

| Study | Feature Backbone | Test Acc | δ negatively correlates with margin (Pearson) | δ positively correlates with entropy (Pearson) | Std(θ) | Std(δ) |
|---|---|---|---|---|---|---|
| Study I: CIFAR + PCA-128 | PCA-128 | 0.468 | −0.815 | 0.687 | 0.154 | 0.150 |
| Study II: CIFAR + MobileNet | MobileNet-V3 (960-D) | 0.819 | −0.950 | 0.881 | 0.228 | 0.871 |
| Study III: MNIST Mini | Raw pixels | 0.954 | −0.975 | 0.970 | 0.289 | 0.472 |

- <small>*Std(θ) measures tree ability spread; Std(δ) measures item difficulty spread.*</small>

- Across studies δ remains negatively correlated with margin and positively correlated with entropy: PCA lands near −0.82, MobileNet tightens to −0.95, and MNIST saturates the scale at −0.98.
- θ spread remains compact (Std(θ) ≈ 0.15–0.29) even with 2000 trees; MobileNet widens slightly as headroom grows.
- Difficulty variance balloons on MobileNet (Std(δ) ≈ 0.87) while MNIST stays moderate, underscoring how rich features surface nuanced “hard” digits.

---

# Cross-Study Fit Snapshot

| Study | Item infit μ / p95 | Item outfit μ / p95 | Tree infit μ / p95 | Tree outfit μ / p95 |
|---|---|---|---|---|
| CIFAR + PCA | 0.18 / 0.35 | 0.18 / 0.34 | 0.35 / 0.48 | 0.18 / 0.19 |
| CIFAR + MobileNet | 0.27 / 0.37 | 0.27 / 0.37 | 0.29 / 0.31 | 0.27 / 0.29 |
| MNIST mini | 0.23 / 0.38 | 0.22 / 0.37 | 0.30 / 0.32 | 0.22 / 0.25 |

- All MSQs stay well below 1, indicating over-dispersed errors are rare and Rasch assumptions hold after 2000-tree scaling.
- MobileNet’s slight lift in item MSQ reflects richer feature diversity, while MNIST keeps both item and tree fits exceptionally tight.

---

# 2PL Discrimination (CIFAR + PCA)

- 800-epoch 2PL fit (lr 0.02) yields mean 𝑎 ≈ **0.35 ± 0.10** (range 0.07–0.71).
- 𝑎 correlates with margin at **−0.83** and with entropy at **+0.63**, aligning slope with RF uncertainty signals.
- Discrimination peaks on the low-margin, high-entropy animal items and steadily tapers for easier scenes, leaving high-margin images with softer slopes.

<div class="columns">
  <div class="col">

  <center>
    <img width="85%" src="figures/2pl_discrimination_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
  </center>
  </div>
  <div class="col">

  <center>
    <img width="85%" src="figures/2pl_discrimination_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
  </center>
  </div>
</div>

---

# 2PL Discrimination (CIFAR + MobileNet)

- Mean 𝑎 settles at **0.27 ± 0.15** with a modest tail (max ≈1.16).
- 𝑎 correlates with margin at **−0.32** and with entropy at **+0.10**, keeping residual cat/dog confusion in focus while the easy cluster sharpens.
- Discrimination concentrates in the tails: hard animal confusions and trivially easy scenes separate trees, while mid-uncertainty items contribute little.

<div class="columns">
  <div class="col">

  <center>
    <img width="85%" src="figures/mobilenet_2pl_discrimination_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
  </center>
  </div>
  <div class="col">

  <center>
    <img width="85%" src="figures/mobilenet_2pl_discrimination_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
  </center>
  </div>
</div>


---

# 2PL Discrimination (MNIST)

- Mean 𝑎 lifts to **0.33 ± 0.25**, so only a modest slice of digits remains truly separating despite the high accuracy ceiling.
- 𝑎 correlates with margin at **+0.89** while its correlation with entropy flips to **−0.96**—uncertainty vanishes outside the awkward strokes.
- Discrimination climbs with margin and falls with entropy: crisp, easy digits carry the steepest slopes while ambiguous stroke collisions stay much flatter.

<div class="columns">
  <div class="col">

  <center>
    <img width="85%" src="figures/mnist_2pl_discrimination_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
  </center>
  </div>
  <div class="col">

  <center>
    <img width="85%" src="figures/mnist_2pl_discrimination_vs_entropy.png" style="width:100%; border:1px solid #ccc;" />
  </center>
  </div>
</div>

---

# 3PL Pilot · MobileNet

- 1k-epoch 3PL run (lr 0.01) lands at guess mean **0.35 ± 0.16**.
- θ vs accuracy stays tight (Pearson **0.98**); slopes average **0.32 ± 0.08** with a broader tail.
- High guess mass piles onto the ambiguous animal scenes (low margin, high entropy), reinforcing the “guessing” narrative.

<div class="columns">
  <div class="col">
    <img src="figures/mobilenet_3pl_guess_hist.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:72%; text-align:center;">3PL MobileNet · Guess distribution</p>
  </div>
  <div class="col">
    <img src="figures/mobilenet_3pl_guess_vs_margin.png" style="width:100%; border:1px solid #ccc;" />
    <p style="font-size:72%; text-align:center;">3PL MobileNet · Guess vs Margin (colored by entropy)</p>
  </div>
</div>

---

# Tree Attribute Correlations

- MobileNet: Pearson corr. (leaf count, θ) **−0.78**; (OOB accuracy, θ) **+0.75**—shallow, accurate trees shine.
- PCA baseline: Pearson corr. (leaf count, θ) **−0.20**; (OOB accuracy, θ) **+0.28**; MNIST shows similar leaf penalties (−0.47).

<div class="columns">
  <div class="col">

  <center>
    <img width="85%" src="figures/mobilenet_tree_oob_accuracy_vs_theta.png" style="width:100%; border:1px solid #ccc;" />
  </center>
  </div>
  <div class="col">

  <center>
    <img width="85%" src="figures/pca_tree_n_leaves_vs_theta.png" style="width:100%; border:1px solid #ccc;" />
  </center>
  </div>
</div>


---

# Key Takeaways

- IRT and RF still move in lockstep: θ tracks per-tree accuracy, while δ and 𝑎 surface stubborn item pockets.
- MobileNet’s discrimination tail isolates animal confusions despite stronger features; MNIST flips signs because mistakes are rare.
- 3PL adds a modest guessing floor (~0.25) without upsetting θ–accuracy alignment.
- Tree attributes expose pruning cues: shallow, high-OOB trees consistently land higher θ.

---

# Next Steps

- Fold discrimination stats into `reports/embedding_comparison.md` & deck tables for quick grabs.
- Run stability sweeps (50/100 trees, alternate seeds) to quantify variance in 𝑎 and θ.
- Decide whether 3PL merits extension to PCA/MNIST or documenting as MobileNet-only.
- Finish item-tier overlays (high/medium/low 𝑎) and align them with the qualitative grids.

--- 

# Decision Trees — From Data to Splits

<div class="columns">
  <div class="col">

**Idea:** recursively split data to  increase *purity* of labels.  

Example:  
> “PetalLength < 2.5?” → all *Setosa* left, others right.

At each node:
- compute **impurity** (e.g., *entropy* or *Gini*):
  $$ H = -\sum_i p_i \log_2 p_i $$
- choose the split that **maximally reduces impurity** — i.e. makes groups more uniform.  

A single tree = a set of *if–then* rules that classify or predict.
  </div>

  <div class="col">

<div class="col">

<center>

| PetalLength | PetalWidth | Species |
|--------------|-------------|----------|
| 1.4 | 0.2 | Setosa |
| 4.7 | 1.4 | Versicolor |
| 5.5 | 2.0 | Virginica |

<br />
<br />
<img width="65%" src="iris.svg">

</center>
</div>

</div>


---

# Gini vs. Entropy — Two Lenses on Node Impurity

<div class="columns">
<div class="col">

**Entropy (Information Theory):**

$$ H = - \sum_i p_i \log_2 p_i $$

Measures **uncertainty** —  expected information (in bits) needed to classify a random sample.  *High when classes are evenly mixed.*
</div>

<div class="col">

**Gini Impurity (Probability of Misclassification):**

$$ G = 1 - \sum_i p_i^2 $$

Measures **chance of error** —  probability that two randomly drawn samples from the node  belong to different classes.
</div>
</div>


<center>

| Metric | Theoretical Lens | Interpretation | Typical Use |
|---------|------------------|----------------|--------------|
| **Entropy** | Information theory | “How surprised would I be?” | ID3, C4.5 trees |
| **Gini** | Probability theory | “How often would I be wrong?” | CART trees, scikit-learn default |
</center>

> Both peak when classes are perfectly mixed (p = 0.5).  
> Gini is slightly flatter — faster to compute, less sensitive to extremes.


