# The DriftWatch Research Gap — How to Explain It to a Professor

This file is a script you can read (or adapt) when someone asks: *"What is your
research contribution, and why does it matter?"* It is written to be spoken in
about 3 minutes, with the numbers you actually measured.

---

## 1. The 3-line pitch

> "DriftWatch studies whether the drift detectors people actually deploy — PSI,
> KS, and domain-classifier AUC — really predict accuracy loss when a model is
> moved to a new distribution. Across real US census shifts between states, we
> find these input-only scores are weak predictors of accuracy drop, while a
> simple model-aware signal (prediction shift) and, especially, label shift
> explain far more. We also show that a standard PSI alert threshold, applied
> literally, is essentially a coin-flip: it never misses, but it raises a false
> alarm on every benign shift."

---

## 2. Why it matters (for the professor)

**The problem:** MLOps tools ship drift alarms that practitioners trust to mean
"your model is degrading". But the premise — *drift ⇒ harm* — is rarely tested
against ground truth. Drift detection is a *probe*; accuracy loss is the *thing
you actually care about*. We test the link.

**Why it's non-obvious:** a model can drift in its inputs yet keep working
(drift without harm), and it can silently degrade with no input drift at all
(harm without drift) — e.g. when the relationship between inputs and labels
changes. So a first-order, data-backed statement about when drift scores are
informative is genuinely useful.

**Why census data is the right testbed:** model shifts between US states are
*natural*, real-world covariate and label shifts — not synthetic perturbations.
That makes the conclusions about methods, not about a toy.

---

## 3. The gap vs related work (the core claim)

**Where we sit.** The literature has two threads: (a) *label-free performance
estimators* (PAPE, ATC, DoC) that try to predict accuracy without labels, and
(b) *drift benchmarks* (TableShift, Failing Loudly) that show drift signals are
incomplete. We are a third and largely-missing piece: an **empirical study of
why common label-free drift scores fail** — i.e. we measure, on natural
cross-state and temporal census shifts, how much accuracy loss each score
actually captures, and whether the standard alert thresholds discriminate.

| Paper | What it did | What it does NOT do (our gap) |
|---|---|---|
| TableShift (NeurIPS 2023) | Benchmark linking the accuracy gap to label-distribution shift | Uses states as a benchmark, but does not decompose *how* input-only drift scores fail, nor audit thresholds |
| PAPE (NeurIPS 2025) | Label-free performance *estimator*, tested on 900+ census combos | The contrast method; does not study *why* drift scores fail or audit thresholds |
| ATC (Garg et al. 2022), DoC (Guillory et al. 2021) | Confidence-based accuracy estimators | We evaluate them *as accuracy estimators* (MAE), not just as rankers |
| Failing Loudly (NeurIPS 2019) | Compared shift detectors under *perturbations* | Synthetic perturbations, not natural shifts, not measured against accuracy |
| Podkopaev & Ramdas (ICLR 2022) | Sequential tests for "harmful vs benign" | Requires labels; theory-focused, not an empirical decomposition/threshold audit |
| Dong et al. (VLDB 2024) | Shows not all drift hurts; retrain on harmful drift only | Motivates our false-alarm question; does not measure score effectiveness |
| MMD (classical) | Two-sample kernel distance | Ours is (to our knowledge) the first to place MMD in this drift-vs-accuracy audit |

**Our claim, one sentence:**

> "PAPE and TableShift show that drift signals are insufficient and that label
> shift matters. We measure, on real cross-state and temporal census shifts,
> how much accuracy loss each common label-free score (PSI, KS, domain AUC, MMD)
> actually captures, evaluate the standard alert thresholds against true accuracy
> loss, and decompose the loss into covariate vs concept shift — across two
> independent tasks and two frontiers."

**How we differ, concretely.**
- We study *why scores drift scores fail* — not a new estimator, not a new
  benchmark. Our testbed is natural shifts (US states, and 2018→2019), not
  synthetic perturbations.
- We **audit the de-facto standard thresholds** (PSI 0.10 / 0.25) against true
  accuracy drop and report false-alarm/miss rates — a direct, testable check of
  a widely-used rule of thumb that prior work does not do.
- We report a **leave-one-source-out / leave-one-target-out** cross-validation so
  the predictive-importance numbers are honest when rows share a source model.
- We **cross-fit the importance-reweighting decomposition** and report it with
  bootstrap SEs, truncation-cap sensitivity, and an effective-sample-size (ESS)
  diagnostic — so the covariate-vs-concept split is not overstated.

That is a clean, defensible niche: **an empirical decomposition and an alert-
threshold audit**, not a claim to have built a better detector.

---

## 4. What you actually built (methods, one slide each)

- **Data:** Folktables ACS income task, 2018, per US state (≤30k rows/state,
  subsampled for equal weight). Train on a source state, evaluate on target states.
- **Models:** histogram gradient boosting and logistic regression (so results
  are not tied to one algorithm).
- **Drift scores (label-free):**
  - PSI per feature (numeric quantile buckets; categorical top-20 frequency);
  - KS on the three numeric features; and
  - domain-classifier AUC (can a model tell old vs new rows?).
- **Model-aware signals:**
  - `pred_shift` = |mean predicted probability on target − on reference| — label-free, uses the model's own scores;
  - `atc` (Average Threshold Confidence) and `doc` (Difference of Confidences) — confidence-based accuracy estimators (Garg et al. 2022; Guillory et al. 2021);
  - `mmd` (Maximum Mean Discrepancy) — a classical label-free two-sample test.
- **Benchmark design:** cross-state pairs (train on one state, test on others) **and** temporal pairs (train on 2018, test on 2019 within the same state). All 15 states run as sources (210 cross-state rows), not just 3.
- **Ground-truth outcome:** `acc_drop` = home accuracy − target accuracy.
- **Statistics:** Spearman correlations with 2,000-sample bootstrap 95% CIs;
  leave-one-out R² to decompose which signal explains accuracy drop; false-alarm
  and miss rates at alert thresholds; ROC AUC for ranking "harmful" pairs.

---

## 5. The results (your actual numbers)

**Numeric input-drift scores track accuracy loss moderately; model-aware pred_shift does not:**

| Signal | Spearman ρ (95% CI) | LOO R² |
|---|---|---|
| mean PSI | 0.24 (−0.07, 0.46) | **−0.05** |
| mean PSI (numeric only) | 0.33 (0.12, 0.49) | 0.14 |
| max PSI | 0.15 (−0.17, 0.41) | — |
| mean KS | 0.31 (0.12, 0.46) | 0.09 |
| domain AUC | 0.18 (−0.02, 0.36) | 0.01 |
| **pred_shift (model-aware)** | **0.09 (−0.11, 0.26)** | −0.01 |
| **MMD** | **0.44 (0.28, 0.56)** | 0.27 |

> These are the **full 210-row HGB run** (15 sources × 14 non-self targets,
> ACS 2018). The earlier 42-row pilot story (pred_shift best, ρ≈0.36) did NOT
> survive the full run and has been discarded.

Interpretation to say aloud: *"the numeric input scores (MMD, numeric PSI, KS)
have correlations with accuracy drop that exclude zero, but they are moderate,
not strong (ρ ≈ 0.3–0.44). The model-aware `pred_shift` does NOT track accuracy
loss here (ρ ≈ 0.09, CI includes 0)."*

**Label shift explains the most, but MMD is close:** leave-one-source-out R² for
label base-rate shift alone ≈ +0.31, input drift + label shift ≈ +0.33, and MMD
alone ≈ +0.24. So label shift is still the strongest single predictor, but MMD
is not "far below" it as the pilot suggested. All LOO numbers are now group-LOO
(leave-one-source-out and leave-one-target-out), so pairs sharing a source model
cannot leak into each other.

**The alert audit (the simplest, most striking result):**

PSI "alert" thresholds vs. "harmful" = accuracy drop ≥ 0.03 (66 harmful of 210):

- `mean_psi ≥ 0.10` → fires on all 210 pairs (66 tp + 144 fp): **false-alarm
  rate 1.0**, miss rate 0.0. Constant classifier.
- `mean_psi_numeric ≥ 0.10` → fires on **zero** pairs: miss rate 1.0. Constant
  classifier.
- `max_psi ≥ 0.25` → 66 tp + 144 fp: false-alarm rate 1.0.

So **the fixed standard thresholds do not discriminate here** — they are either
always-on or always-off. `mean_psi` is worse than numeric-only PSI, consistent
with high-cardinality categorical inflation (POBP, OCCP) — a hypothesis, still
unproven until tested directly. **We report a full threshold sweep (ROC/PR
curves) in the supplementary rather than relying on these two fixed thresholds.**

**The decomposition is preliminary.** On the pooled full run, concept_share
(+0.027) dwarfs covariate_share (−0.005), but — (a) 30 of 210 pairs have ESS
ratio < 0.2 (unreliable importance weights); excluding them lowers concept_share
to +0.021; (b) the split is dominated by a few low-ESS sources (AL, WV, ME),
shown in the per-source table; (c) the $50k cutoff is not state-adjusted, which
may inflate "concept shift". So we report the decomposition with the ESS
diagnostics, per-source, and across truncation caps 10/20/50 — we do not claim
"concept dominates" as a headline.

---

## 6. The "surprise" (memorable takeaway)

Two defensible surprises:
1. **The fixed PSI thresholds are uninformative as stated** — always-on or
   always-off, 0% miss but 100% false alarm. This is a direct, testable check of
   a widely-used rule of thumb.
2. **`pred_shift` — the model-aware, label-free signal — does not track accuracy
   in this setup**, while simple numeric input scores (MMD/KS/PSI-numeric) do
   moderately. The "fancier" signal fails where the simple ones partly work.

---

## 7. Limitations (be upfront)

- **Two tasks** (income, employment), tabular ACS data, and **two frontiers**
  (cross-state 2018; temporal 2018→2019). Still one data modality (tabular US
  census) — generalization to other modalities (image/text) is future work.
- **State pairs share the source model** → rows are not independent. We report
  group leave-one-source-out/leave-one-target-out R² instead of plain LOO; the
  bootstrap intervals remain approximate.
- **The $50k income cutoff is NOT state-adjusted** (Folktables uses a nominal
  cutoff). Across states with different cost-of-living/income distributions this
  can inflate the apparent "concept shift" (P(y|x) change). We use the
  importance-reweighting decomposition to separate this, and we report the
  covariate/concept split with ESS diagnostics — but the residual can still mix
  a real P(y|x) change with a cutoff artifact. See `PAPER.md` §Limitations for
  the argued position.
- **2018→2019 temporal cutoff is not inflation-adjusted** → an arithmetic
  upward drift in the label rate, so the temporal accuracy drop is an upper
  bound on the "real" temporal concept shift.
- Synthetic runs are for testing only, not research results.
- Categorical PSI is unstable on high-cardinality codes (`POBP`, `OCCP`), which
  is itself a finding (`mean_psi` underperforms numeric-only PSI) — a hypothesis,
  not yet a proven mechanism until tested directly.

---

## 8. Suggested title(s)

- *"When Drift Alerts Lie: Decomposing What Label-Free Drift Scores Capture in
  Real Model Shifts"*
- *"Drift Scores vs. Accuracy Loss: An Empirical Decomposition on US Census
  State Shifts"*
- *"Do Standard Drift Thresholds Work? An Audit on Real Cross-State Shifts"*

---

## 9. One-line answers to likely questions

- **"Did you deploy it?"** We simulated deployment with real labels (so we
  could measure the true accuracy effect) and built a working local FastAPI
  service with `/predict`, `/drift`, and a dashboard. In live production the
  monitor sees only input drift until labels arrive — which is exactly the gap
  under study.
- **"Is the novelty enough?"** We're not claiming a new detector. We claim a
  first empirical decomposition and threshold audit on natural shifts — a solid,
  honest contribution, not a breakthrough. We state that plainly.
- **"Why not just use PAPE?"** PAPE is the contrast method; our question is
  different (why do drift scores fail, and do the standard thresholds work),
  and we include the estimator comparison `pred_shift` as the practical output.
