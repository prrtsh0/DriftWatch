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

| Paper | What it did | What it does NOT do (the gap) |
|---|---|---|
| TableShift (NeurIPS 2023) | Benchmark linking the accuracy gap to label-distribution shift | Uses states as a benchmark; does not decompose *how* input-only drift scores fail |
| PAPE (NeurIPS 2025) | Label-free performance *estimator*, tested on 900+ census combos | The contrast method, not a study of *why* drift scores fail |
| Failing Loudly (NeurIPS 2019) | Compared shift detectors under *perturbations* | Synthetic perturbations, not natural shifts, not measured against accuracy |
| Podkopaev & Ramdas (ICLR 2022) | Sequential tests for "harmful vs benign" warning | Requires labels, theory-focused, not an empirical decomposition |
| Dong et al. (VLDB 2024) | Shows not all drift hurts; retrain on harmful drift only | Motivates our false-alarm question; does not measure score effectiveness |

**Our claim, one sentence:**

> "PAPE and TableShift show that drift signals are insufficient and that label
> shift matters; we are the first to measure, on real cross-state census shifts,
> how much accuracy loss each common label-free score (PSI, KS, domain AUC)
> actually captures, and to evaluate the standard alert thresholds against true
> accuracy loss."

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
- **Model-aware signal:** `pred_shift` = |mean predicted probability on target −
  on reference| — label-free, uses the model's own scores.
- **Ground-truth outcome:** `acc_drop` = home accuracy − target accuracy.
- **Statistics:** Spearman correlations with 2,000-sample bootstrap 95% CIs;
  leave-one-out R² to decompose which signal explains accuracy drop; false-alarm
  and miss rates at alert thresholds; ROC AUC for ranking "harmful" pairs.

---

## 5. The results (your actual numbers)

**Input-only drift scores are weak predictors:**

| Signal | Spearman ρ (95% CI) | LOO R² |
|---|---|---|
| mean PSI | 0.15 (−0.17, 0.45) | **−0.05** |
| mean PSI (numeric only) | 0.16 (−0.15, 0.45) | −0.07 |
| max PSI | −0.13 (−0.41, 0.18) | — |
| mean KS | 0.18 (−0.15, 0.49) | — |
| domain AUC | 0.20 (−0.11, 0.47) | **−0.05** |
| **pred_shift (model-aware)** | **0.36 (0.06, 0.60)** | — |

Interpretation to say aloud: *"no pure input score has a statistically reliable
positive correlation with accuracy drop — the confidence intervals all straddle
zero, and the leave-one-out R² is negative, which means they are worse than
just predicting the average drop."*

**Label shift explains more:** LOO R² for label shift alone ≈ +0.13, and for
input drift + label shift ≈ +0.15. So the label-rate change is the signal that
actually tracks the accuracy drop — consistent with TableShift's hypothesis,
now demonstrated on your own state pairs.

**The alert audit (the simplest, most striking result):**

PSI "alert" (0.10 or 0.25) vs. "harmful" = accuracy drop ≥ 0.03:

| | Alert fired | No alert |
|---|---|---|
| harmful (11 pairs) | `tp` = 11 | `fn` = 0 |
| not harmful (31 pairs) | `fp` = 31 | `tn` = 0 |

- Miss rate = 0% (it never misses a harmful shift)
- False-alarm rate = 100% (it fires on every benign shift)
- Precision = 26% (same as the base rate — the alert adds **zero** information)

Say: *"an alert that fires on 100% of benign shifts tells you nothing; the only
reason it never misses is that it always fires."*

**The model-aware estimator is the best free signal:** `pred_shift` has the
highest Spearman ρ (0.36) and the highest AUC for ranking harmful pairs (0.82
vs 0.67 for mean PSI). This is a concrete, easy-to-adopt improvement.

---

## 6. The "surprise" (memorable takeaway)

The biggest (and most publishable) surprise is the **threshold audit**: the
de-facto standard PSI thresholds are effectively uninformative as stated —
0% miss but 100% false alarm. This is a direct, testable, reproducible statement
that questions a widely-used rule of thumb. That is a stronger contribution than
"correlation exists".

---

## 7. Limitations (be upfront)

- One task (income), tabular data, one survey year.
- State pairs share the source model → rows are not independent; bootstrap
  intervals are approximate.
- Synthetic runs are for testing only, not research results.
- Categorical PSI is unstable on high-cardinality codes (`POBP`, `OCCP`),
  which is itself a finding to mention.

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
