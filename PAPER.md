# DriftWatch — Paper Draft

> Working title: **"When Drift Alerts Lie: Decomposing What Label-Free Drift
> Scores Capture in Real Model Shifts"**
>
> Status: DRAFT. Numbers marked **[EMP]** come from the employment-task run and
> are filled in when that run finishes. Income numbers are the verified 210-row
> HGB run (commit `1d07460`).

---

## Abstract (draft)

Deployed ML models drift. Practitioners often monitor **label-free drift scores**
(PSI, KS, domain-classifier AUC, MMD) and alert when they exceed thresholds,
because true labels are unavailable online. But how well do these scores
actually track accuracy loss, and do the standard alert thresholds work?

Using **natural** shifts from US Census data — across 15 states (cross-state)
and across years 2018→2019 (temporal), on two independent tasks (income,
employment) and two model families — we find: (1) numeric input scores track
accuracy loss **moderately** (Spearman ≈ 0.3–0.44; leave-one-source-out R² up
to ~0.24); (2) the model-aware, label-free `pred_shift` does **not** track
accuracy in this setup (ρ ≈ 0.09, CI includes 0); (3) the standard fixed PSI
thresholds are **uninformative as stated** — always-on or always-off, 0% miss
but 100% false alarm; and (4) an importance-reweighting decomposition, reported
with cross-fitting, truncation sensitivity, and effective-sample-size
diagnostics, attributes most of the accuracy drop to **label/concept shift** —
but the split is fragile for low-overlap pairs and is confounded by a
non-state-adjusted $50k threshold. We release code, data-loading, and full
results.

---

## 1. Introduction

- Drift monitoring is a rule of thumb in practice: watch input distributions,
  alert on threshold crossings. Yet "drift" ≠ "accuracy loss".
- We ask: *for real shifts (not synthetic perturbations), how much accuracy loss
  does each common label-free score capture, and do the standard thresholds
  discriminate?*
- Contributions:
  1. An **empirical audit** of PSI / KS / domain-AUC / MMD / pred_shift / ATC /
     DoC against true accuracy drop on natural cross-state and temporal census
     shifts, across two tasks and two models.
  2. A **threshold audit** with false-alarm / miss rates and a full threshold
     sweep (not just the two standard cutoffs).
  3. A **decomposition** of the accuracy drop into covariate vs concept shift,
     reported with the rigor the split needs (cross-fitting, truncation caps,
     bootstrap SEs, effective-sample-size filtering, per-source breakdown).

## 2. Related Work

- Label-free performance **estimators**: PAPE (NeurIPS 2025), ATC (Garg et al.
  2022), DoC (Guillory et al. 2021). We evaluate them (as rankers and as
  accuracy estimators) but our question is *why drift scores fail*, not a new
  estimator.
- Drift **benchmarks**: TableShift (NeurIPS 2023) links the accuracy gap to
  label shift; Failing Loudly (NeurIPS 2019) compares detectors under synthetic
  perturbations; Dong et al. (VLDB 2024) show not all drift hurts and retrain on
  harmful drift. We differ: natural shifts, measured against true accuracy, plus
  a threshold audit and a decomposition.
- Podkopaev & Ramdas (ICLR 2022): sequential tests for harmful-vs-benign, but
  requires labels and is theory-focused.

## 3. Setup

- **Data:** Folktables ACS. Task `income` (predict >$50k) and task `employment`
  (ESR). 15 US states as sources, all-pairs cross-state (210 rows per run) +
  temporal (same state, 2018→2019). Subsampled to 30k rows/state.
- **Models:** HistGradientBoosting and logistic regression.
- **Drift scores (label-free):** PSI (numeric quantile + categorical top-20),
  KS (numeric), domain-classifier AUC, MMD (RBF, numeric), pred_shift (model
  output distribution shift), ATC, DoC.
- **Ground truth:** `acc_drop` = home accuracy − target accuracy.
- **Statistics:** Spearman with cluster-bootstrap CIs; predictor importance via
  leave-one-out / leave-one-source-out / leave-one-target-out R²; harmful-AUC
  with CIs; threshold sweep; decomposition with ESS diagnostics.

## 4. Results (income — verified 210-row HGB run)

### 4.1 Numeric input scores track loss; pred_shift does not

| Signal | Spearman ρ (95% CI) | LOO R² | LOSO R² | LOTO R² |
|---|---|---|---|---|
| mean PSI | 0.24 (−0.07, 0.46) | −0.05 | — | — |
| mean PSI numeric | 0.33 (0.12, 0.49) | 0.14 | 0.087 | 0.085 |
| mean KS | 0.31 (0.12, 0.46) | 0.09 | 0.037 | 0.003 |
| domain AUC | 0.18 (−0.02, 0.36) | 0.01 | −0.065 | −0.020 |
| pred_shift | 0.09 (−0.11, 0.26) | −0.01 | −0.083 | −0.046 |
| **MMD** | **0.44 (0.28, 0.56)** | **0.27** | **0.235** | **0.233** |
| label shift | — | 0.361 | 0.314 | 0.326 |

### 4.2 Threshold audit

| Rule | TP | FP | FN | TN | False-alarm | Miss |
|---|---|---|---|---|---|---|
| mean_psi ≥ 0.10 | 66 | 144 | 0 | 0 | 1.00 | 0.00 |
| mean_psi_numeric ≥ 0.10 | 0 | 0 | 66 | 144 | — | 1.00 |
| max_psi ≥ 0.25 | 66 | 144 | 0 | 0 | 1.00 | 0.00 |

Full threshold sweep in supplementary; standard cutoffs are constant classifiers
here.

### 4.3 Decomposition (provisional)

| Quantity | Value |
|---|---|
| error_home | 0.187 |
| error_target | 0.209 |
| covariate_share | −0.005 (SE 0.006) |
| concept_share | +0.027 (SE 0.009) |
| concept_share, ESS ≥ 0.2 excluded (n=180) | +0.021 |
| median ESS ratio | 0.733 |
| share of pairs with ESS < 0.2 | 0.143 |

Per-source table shows AL/WV/ME (low ESS) drive the split. Interpret with the
$50k-cutoff caveat.

## 5. Results (employment — [EMP] pending)

- [EMP] Spearman table for employment.
- [EMP] threshold audit.
- [EMP] decomposition.

## 6. Discussion

- Why input scores partly work and pred_shift fails: numeric distribution
  distance tracks covariate shift which correlates with loss; pred_shift is a
  *model-output* shift that can be large with no accuracy effect (e.g. the model
  recalibrating without losing discriminative power).
- Thresholds: fixed thresholds fail because the right cutoff depends on the task,
  the model, and the feature cardinality. Practitioners need task-specific,
  calibrated thresholds or a different monitoring statistic.
- Decomposition: label/concept shift is the dominant driver, but the split is
  not stable for low-overlap pairs — report diagnostics.

## 7. Limitations (argued)

- $50k cutoff not state-adjusted → may inflate concept share. We argue:
  the decomposition's covariate term is still valid; the concept residual is an
  upper bound on true concept shift; we report per-source and low-ESS variants so
  the reader sees the sensitivity. A state-adjusted threshold (cost-of-living
  indexed) is future work.
- 2018→2019 cutoff not inflation-adjusted → temporal accuracy drop is an upper
  bound on temporal concept shift.
- One modality (tabular census), two tasks, two frontiers. Generalization to
  image/text is future work.
- Pairs share source model → group LOO reported.
- Categorical PSI inflation (POBP/OCCP) → `mean_psi` < `mean_psi_numeric`.

## 8. Reproducibility

- Fixed seeds; global RNG pinned; every run records its commit SHA. Code in
  `driftwatch/`; tests: 23 + second-task tests; results in `outputs/`.

## 9. Suggested venues

- Workshop / short paper: drift detection workshops at ICML/NeurIPS, or a
  reproducible-benchmark venue.
- Main-track candidate only after adding a non-tabular modality or a third task.
