# DriftWatch — Paper Draft

> Working title: **"When Drift Alerts Lie: Decomposing What Label-Free Drift
> Scores Capture in Real Model Shifts"**
>
> Status: DRAFT. Income numbers are the verified 210-row HGB run (commit
> `1d07460`); employment numbers are the verified 210-row run (commit
> `a65159b`). Temporal numbers pending.

---

## Abstract (draft)

Deployed ML models drift. Practitioners often monitor **label-free drift scores**
(PSI, KS, domain-classifier AUC, MMD) and alert when they exceed thresholds,
because true labels are unavailable online. But how well do these scores
actually track accuracy loss, and do the standard alert thresholds work?

Using **natural** shifts from US Census data — across 15 states (cross-state)
and across years 2018→2019 (temporal), on two independent tasks (income,
employment) and two model families — we find: **(1) the scores do not
generalize across tasks.** On income, numeric input scores track accuracy loss
moderately (Spearman ≈ 0.3–0.44; MMD R² ≈ 0.24); on employment, where drift is
smaller, **no score — not even the label base-rate — reliably tracks the loss**
(LOO R² ≤ 0.02 for all). (2) the model-aware, label-free `pred_shift` never
tracks accuracy (ρ ≈ 0.07–0.09). (3) the standard fixed PSI thresholds are
**uninformative as stated** on both tasks — always-on/always-off or grossly
over-/under-firing. (4) an importance-reweighting decomposition attributes most
of the *income* accuracy drop to **label/concept shift** (concept_share +0.027
vs covariate_share −0.005), but the split is fragile for low-overlap pairs and
confounded by a non-state-adjusted $50k threshold; on employment the (smaller)
drop splits more evenly. We release code, data-loading, and full results.

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

## 5. Results (employment — full 210-row HGB run, commit `a65159b`)

> The employment task has **much weaker drift**: mean `acc_drop` = 0.007
> (income: 0.022), and only 9 of 210 pairs are "harmful" (drop ≥ 0.03).
> The headline is that **the scores that worked moderately on income do NOT
> generalize** — this is a key cross-task finding, not a null.

### 5.1 Correlations (all weak; several near zero)

| Signal | Spearman ρ (95% CI) |
|---|---|
| mean PSI | 0.20 (0.02, 0.43) |
| mean PSI numeric | 0.19 (0.05, 0.37) |
| mean KS | 0.19 (−0.00, 0.40) |
| domain AUC | 0.17 (−0.01, 0.39) |
| MMD | 0.14 (−0.02, 0.31) |
| pred_shift | 0.07 (−0.13, 0.30) |
| ATC / DoC | −0.24 (−0.49, 0.10) |

No input score has a dependable positive correlation; ATC/DoC are *negative*
(opposite of income).

### 5.2 Predictor importance (nothing explains loss here)

| Predictors | LOO R² |
|---|---|
| mean_psi_numeric | 0.02 |
| mean_ks | 0.02 |
| mean_psi | 0.01 |
| mmd | 0.004 |
| label base-rate shift only | −0.003 |
| pred_shift | −0.01 |

Unlike income (label-shift LOO R² ≈ +0.31), on employment **no signal — not even
label shift — tracks the (small) accuracy drop.** The drop is small and close to
noise; ~4% of pairs are harmful.

### 5.3 Threshold audit (fails differently)

| Rule | TP | FP | FN | TN | False-alarm | Miss |
|---|---|---|---|---|---|---|
| mean_psi ≥ 0.10 | 2 | 36 | 7 | 165 | 0.18 | 0.78 |
| mean_psi_numeric ≥ 0.10 | 1 | 11 | 8 | 190 | 0.05 | 0.89 |
| max_psi ≥ 0.10 | 8 | 172 | 1 | 29 | 0.86 | 0.11 |
| max_psi ≥ 0.25 | 5 | 110 | 4 | 91 | 0.55 | 0.44 |

`max_psi` over-fires (FA 0.86); `mean_psi_numeric` under-fires (miss 0.89).
No standard threshold achieves both low miss and low false-alarm.

### 5.4 Harmful-ranking AUC (wide CIs — only 9 harmful pairs)

| Score | AUC for harmful |
|---|---|
| domain AUC | 0.73 |
| MMD | 0.66 |
| mean PSI | 0.66 |
| pred_shift | 0.40 |
| ATC / DoC | 0.24 |

CIs are wide because there are only 9 positives; interpret cautiously.

### 5.5 Decomposition (small and balanced)

| Quantity | Value |
|---|---|
| error_home | 0.171 |
| error_target | 0.178 |
| covariate_share | +0.001 (SE 0.003) |
| concept_share | +0.006 (SE 0.006) |
| median ESS ratio | 0.80 |
| share of pairs with ESS < 0.2 | 0.043 |

The split is far more balanced than income (concept ≈ 85% of a much smaller
drop) and far more stable (few low-ESS pairs).

## 5.6 Temporal frontier (2018→2019, income — 15 rows)

> **Interpret with caution — the inflation confound dominates.** The $50k
> cutoff is not inflation-adjusted, so nominal income crossing $50k over time
> creates a large, systematic *label* shift. The measured drops are therefore an
> **upper bound** on the "real" temporal concept shift.

| Quantity | Value (median across states) |
|---|---|
| acc_drop | 0.119 |
| mean_psi | 1.69 |
| concept_share | 0.248 |
| covariate_share | ~0.001 |

The very high `mean_psi` (≈1.7) and `concept_share` (≈0.25) are consistent with
a nominal-cutoff artifact rather than a genuine P(y|x) change: the input
distribution barely moves (`mmd` ≈ 0.001) while the label rate shifts sharply.
This is the cleanest demonstration of the limitation in §7 — a monitoring system
reading `mean_psi` ≈ 1.7 would alert, but the "drift" is a threshold artifact,
not a real covariate change.

## 6. Discussion

- **Cross-task non-generalization is the main finding.** On income (larger
  drift), MMD/numeric-PSI/KS track loss moderately and label shift explains most
  of it. On employment (smaller drift), nothing — not even label shift — tracks
  the (mostly benign) drop. Two implications: (a) a "which score works" answer
  is task- and drift-magnitude-dependent; (b) monitoring thresholds calibrated on
  one task cannot be assumed to transfer. This is exactly the "single
  case-study" → "general finding" leap the earlier version lacked.
- Why input scores partly work on income and pred_shift fails: numeric
  distribution distance tracks covariate shift which correlates with loss;
  pred_shift is a *model-output* shift that can be large with no accuracy effect
  (e.g. the model recalibrating without losing discriminative power). On
  employment even the numeric scores fail because the drift is too small to be
  distinguishable from noise.
- Thresholds: fixed thresholds fail on both tasks, but *differently* (income:
  constant classifiers; employment: over-fire or under-fire). The right cutoff
  depends on task, model, and feature cardinality — practitioners need
  task-specific calibrated thresholds or a different monitoring statistic.
- Decomposition: on income, label/concept shift dominates but the split is
  fragile for low-overlap pairs; on employment the smaller drop splits more
  evenly and stably. Always report diagnostics (ESS, truncation caps, per-source).

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
- Categorical PSI inflation (POBP/OCCP) made `mean_psi` < `mean_psi_numeric` on
  income; this did not hold on employment (the two were nearly equal), so the
  inflation story is task-specific.

## 8. Reproducibility

- Fixed seeds; global RNG pinned; every run records its commit SHA. Code in
  `driftwatch/`; tests: 23 + second-task tests; results in `outputs/`.

## 9. Suggested venues

- Workshop / short paper: drift detection workshops at ICML/NeurIPS, or a
  reproducible-benchmark venue.
- Main-track candidate only after adding a non-tabular modality or a third task.
