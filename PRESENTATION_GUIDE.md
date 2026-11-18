# DriftWatch — Everything You Need to Present

This guide walks the whole project from the smallest piece of code up to the
research story. Use it as your script. All numbers are the **verified real-data
runs** (ACS 2018, 15 US states, 210 cross-state pairs, HGB default; plus a 15-row
temporal 2018→2019 run and a second task).

---

## 0. The one-sentence story

> "On real US-census shifts, common label-free drift scores (PSI, KS, domain AUC,
> MMD) capture only part of the accuracy loss, the standard alert thresholds are
> effectively uninformative, and the pattern does not even generalize across
> tasks — so drift monitoring needs task-specific calibration, not fixed rules."

**Pitch for a professor:** we don't invent a new detector or estimator; we
*audit* the ones people already use, on *natural* shifts (not synthetic), and we
decompose *why* they fail.

---

## 1. The project folder (what each thing is)

```
driftwatch/                          # repo root (this is the project)
├── driftwatch/                      # Python package (the engine)
│   ├── data.py                      # loads ACS states; defines the task registry
│   ├── drift.py                     # label-free drift scores: PSI, KS, domain AUC
│   ├── model.py                     # two models: HGB + LogisticRegression
│   ├── baselines.py                 # accuracy-estimation baselines + decomposition
│   ├── experiment.py                # the benchmark loop (train source → test targets)
│   ├── analysis.py                  # turns results.csv into tables/plots
│   ├── monitor.py                   # the live DriftMonitor used by the API
│   └── __init__.py
├── api/
│   ├── app.py                       # FastAPI: /predict, /drift, /health
│   └── dashboard.py                 # HTML dashboard served by the API
├── tests/
│   ├── test_driftwatch.py           # PSI/KS/monitor/API tests
│   ├── test_baselines.py            # MMD/ATC/DoC/pred_shift/decomposition tests
│   ├── test_analysis.py             # analysis table tests
│   └── test_second_task.py          # employment-task tests
├── run_experiment.py                # CLI entry for the benchmark
├── run_analysis.py                  # CLI entry for the analysis
├── run_temporal.py                  # runs the 2018→2019 temporal benchmark
├── run_logreg.py                    # runs the logistic-regression variant
├── train_service_model.py           # trains the model the API serves
├── check_new_data.py                # checks a new CSV/state for drift offline
├── PAPER.md                         # manuscript draft (tables + prose)
├── RESEARCH_GAP.md                  # the "why this matters" script
├── PRESENTATION_GUIDE.md            # this file
├── HANDOFF.md / PROJECT_STRUCTURE.md
├── outputs/                         # [gitignored] result CSVs + analysis tables
├── data/                            # [gitignored] raw ACS files (2017/2018/2019)
└── artifacts/                       # [gitignored] trained model + reference sample
```

**Key point to say:** the package `driftwatch/` is the reusable engine; the
`run_*.py` scripts are thin CLI wrappers; `outputs/` is the evidence; `api/` is
a demo deployment of the same monitoring logic.

---

## 2. The data (data.py)

**Real data:** US Census American Community Survey (ACS) via the `folktables`
package. One US state = one dataset. Task `income` (folktables `ACSIncome`):
predict whether personal income > $50k.

- 10 features: `AGEP` (age), `COW`, `SCHL` (schooling), `MAR`, `OCCP` (occupation
  code), `POBP`, `RELP` (relationship), `WKHP` (hours/week), `SEX`, `RAC1P`.
- Numeric columns (used by KS/MMD): `AGEP`, `SCHL`, `WKHP`.
- Categorical: the rest. High-cardinality ones (`POBP`, `OCCP`) matter later.

**Task registry** — I added a second task, `employment` (folktables
`ACSEmployment`), target `ESR` (employed?), 16 features. This is what lets us
claim generality.

**Data loading details:**
- `load_state(state, year, n_max, synthetic, task)` returns `(X, y)`.
- Subsampled to `n_max = 30000` rows/state (equal weight per state).
- `train_test_split(..., test_size=0.3, random_state=0, stratify=y)` → 70% train,
  30% reference holdout.
- **2019 schema fix:** in 2019 ACS renamed `RELP` → `RELSHIPP`; `data.py` renames
  it back before folktables filters. This is why the temporal frontier works.

**Synthetic data** (`make_synthetic_state`): a fake "state" where inputs, label
rate, and the input-label link differ per state. Used ONLY for tests/CI, never
for research numbers.

---

## 3. The models (model.py)

`build_model(kind, task)`:
- `hgb` → `HistGradientBoostingClassifier(max_iter=200, random_state=0)` (tree-based).
- `logreg` → `Pipeline`: OneHotEncoder on categorical → StandardScaler on numeric
  → `LogisticRegression(max_iter=1000)`.

Two model families let us claim results aren't tied to one algorithm.

---

## 4. The drift scores (drift.py) — the label-free signals

All of these use **no true labels** (that's the deployed-monitoring setting).

| Score | What it measures | How |
|---|---|---|
| **PSI** (Population Stability Index) | per-feature shift | bucket ref vs new, `Σ (q−p)·ln(q/p)`; numeric = 10 quantile buckets from ref; categorical = top-20 frequency, rare grouped as "other" |
| **KS** | distribution shift, numeric only | `scipy.stats.ks_2samp` statistic on `AGEP/SCHL/WKHP` |
| **domain AUC** | "can a model tell old from new rows?" | train HGB to classify ref(0) vs new(1), report 3-fold CV `roc_auc` |
| **MMD** (baselines.py) | kernel distance between distributions | Gaussian RBF on numeric features, median-bandwidth heuristic |
| **pred_shift** | model-output shift | `\|mean p(new) − mean p(ref)\|` |
| **ATC** / **DoC** | confidence-based accuracy estimators | mean confidence of the k=100 most-confident examples, ref vs new |

`drift_summary()` collapses these per pair into: `mean_psi`, `max_psi`,
`mean_psi_numeric` (numeric only — avoids high-cardinality inflation),
`mean_ks`, `domain_auc`.

**Say this:** PSI/KS/domain-AUC/MMD look only at *inputs* (or at ref-vs-new
separability). None of them can see a change in the label rate or in the
input-label relationship — that's the whole point of the study.

---

## 5. The benchmark pipeline (experiment.py)

For each source state `s` (15 states):
1. Load `s` (2018), split into train + reference holdout.
2. Train model on the train split. Record **home accuracy** on the reference.
3. For each target state `t` ≠ `s` (14 targets):
   - Load `t`. Score with the source-trained model.
   - Record **target accuracy** and `acc_drop = ref_acc − target_acc`.
   - Compute label shift `|mean(y_t) − mean(y_tr)|` (uses labels — explanatory only).
   - Compute all the drift scores (PSI/KS/domain AUC/MMD/pred_shift/ATC/DoC).
   - Compute the **error decomposition** (baselines.py, §6).
4. Append one row per (source,target) pair.

**Result:** `outputs/results.csv` = **210 rows** (15 × 14). Columns:
`source, target, model, task, ref_acc, target_acc, acc_drop, label_shift,
pred_shift, atc, doc, mmd, error_home, error_target, error_cov_only,
covariate_share, concept_share, norm_weights_*, ... , mean_psi, max_psi,
mean_ks, mean_psi_numeric, domain_auc`.

**Same pipeline** produces:
- `results_employment.csv` (task=employment),
- `results_temporal.csv` (train 2018 → test 2019 on the *same* state; 15 rows),
- `results_logreg.csv` (logreg model).

---

## 6. The error decomposition (baselines.py) — the "why"

This is the most methodologically-important function: `decompose_error_shifts`.

Idea: split the error increase into
- **Covariate shift** (P(x) changed) vs
- **Concept shift** (P(y|x) changed — the label/relationship change).

Method (importance reweighting):
1. Fit a **domain classifier** (logistic) to tell ref(0) from new(1) rows,
   **cross-fitted** (5-fold, out-of-fold predictions) so no row is scored by a
   model trained on it.
2. Convert to weights `w(x) = P(new|x)/P(ref|x)` (odss ratio, normalized to mean 1).
3. Truncate weights to `[1/cap, cap]` for `cap ∈ {10, 20, 50}` (bias–variance
   tradeoff; report all three).
4. `error_home` = model error on reference.
   `error_target` = model error on new data.
   `error_cov_only` = ref error reweighted by `w` (the error we'd expect if only
   P(x) changed).
5. `covariate_share = error_cov_only − error_home`
   `concept_share   = error_target − error_cov_only` (residual).
6. Report **bootstrap SEs** (n_boot=200) and **effective sample size** ratio
   `ESS/n = (Σw)²/Σw²` — a low ESS means the weights are unstable and the split
   is unreliable. Filter out `ESS < 0.2` pairs in a sensitivity row.

**Caveat to say:** this is *not* a clean causal estimate — `concept_share` also
absorbs interaction/approximation error. Always present it WITH the ESS
diagnostic and the per-source breakdown.

---

## 7. The analysis (analysis.py) — every evaluation metric

Input: a results CSV. Outputs (each → a CSV in `outputs/`):

| Table | Metric | Meaning |
|---|---|---|
| `correlations.csv` | **Spearman ρ** with **cluster bootstrap 95% CI** | rank correlation of each score with `acc_drop`; resamples at *source cluster* level so pairs sharing a source model aren't falsely independent |
| `predictor_importance.csv` | **R²** (LOO, LOSO, LOTO) | linear-model predictive power of each signal for `acc_drop`. LOO = leave-one-pair-out; LOSO = leave-one-source-out; LOTO = leave-one-target-out (group LOO is honest) |
| `shift_decomposition.csv` | covariate vs concept shares + SEs + ESS | the §6 decomposition, averaged over pairs |
| `shift_decomposition_by_source.csv` | per-source shares | is the split driven by a few states? |
| `per_pair_cap_drift.csv` | cap sensitivity | which pairs' `concept_share` changes most across caps 10/20/50 |
| `alerts.csv` | **false-alarm & miss rates** | treat `score ≥ threshold` as an alert; compare to `harmful = acc_drop ≥ 0.03`; report tp/fp/fn/tn, false-alarm rate fp/(fp+tn), miss rate fn/(fn+tp) |
| `threshold_sweep.csv` | full sweep | 40 thresholds per score, with precision/recall (shows whether ANY threshold works) |
| `harmful_auc.csv` | **harmful-ranking AUC** + cluster-bootstrap CI | how well each score ranks harmful pairs (acc_drop ≥ 0.03) |

**Metric definitions to be able to state:**
- **Spearman ρ**: rank correlation; 1 = perfect monotone match, 0 = none.
- **Cluster bootstrap CI**: resample whole source-blocks, recompute ρ, take 2.5/97.5
  percentiles → interval that accounts for non-independence.
- **R² (LOO/LOSO/LOTO)**: 1 − SS_res/SS_tot for a linear model predicting
  `acc_drop` from the score; negative ⇒ worse than predicting the mean.
- **False-alarm rate** = fraction of *non-harmful* pairs that trigger an alert.
- **Miss rate** = fraction of *harmful* pairs that DON'T trigger an alert.
- **Harmful-ranking AUC** = ROC AUC of the score as a classifier for
  `harmful` (acc_drop ≥ 0.03).

---

## 8. The numbers — income (the headline verified run)

### 8.1 Does drift track accuracy loss? (correlations)
| Score | Spearman ρ (95% CI) |
|---|---|
| MMD | **0.44 (0.28, 0.56)** |
| mean PSI numeric | 0.33 (0.12, 0.49) |
| mean KS | 0.31 (0.12, 0.46) |
| mean PSI | 0.24 (−0.07, 0.46) |
| domain AUC | 0.18 (−0.02, 0.36) |
| max PSI | 0.15 (−0.17, 0.41) |
| **pred_shift** | **0.09 (−0.11, 0.26)** |
| ATC / DoC | 0.23 (−0.08, 0.49) |

→ Numeric input scores track loss *moderately*; the model-aware `pred_shift`
does **not** (CI includes 0).

### 8.2 What explains the drop? (predictor_importance)
| Predictors | LOO R² | LOSO R² | LOTO R² |
|---|---|---|---|
| label base-rate shift only | **0.36** | 0.31 | 0.33 |
| input drift + label shift | 0.40 | 0.33 | 0.36 |
| MMD only | 0.27 | 0.24 | 0.23 |
| mean PSI numeric | 0.14 | 0.09 | 0.08 |
| mean KS | 0.09 | 0.04 | 0.00 |
| pred_shift | −0.01 | −0.08 | −0.05 |

→ **Label shift is the strongest single predictor.** MMD is second and not far
off. `pred_shift` adds nothing.

### 8.3 Decomposition (shift_decomposition)
- error at home = **0.187**, error on target = **0.209** (drop ≈ 0.022).
- **covariate_share = −0.005 (SE 0.006)**
- **concept_share = +0.027 (SE 0.009)**
- median ESS ratio = 0.73; 14.3% of pairs have ESS < 0.2.
- Excluding low-ESS pairs (n=180): concept_share drops to +0.021.
- Across caps 10/20/50 concept_share varies 0.026–0.028 (stable-ish).

→ Concept shift dominates, but the split is fragile for low-overlap pairs.

### 8.4 The alarm thresholds (alerts) — the striking bit
"harmful" = acc_drop ≥ 0.03 → **66 of 210 pairs harmful.**

| Rule | TP | FP | FN | TN | False-alarm | Miss |
|---|---|---|---|---|---|---|
| mean_psi ≥ 0.10 | 66 | 144 | 0 | 0 | **1.00** | 0.00 |
| mean_psi_numeric ≥ 0.10 | 0 | 0 | 66 | 144 | 0.00 | **1.00** |
| max_psi ≥ 0.10 | 66 | 144 | 0 | 0 | **1.00** | 0.00 |
| max_psi ≥ 0.25 | 66 | 144 | 0 | 0 | **1.00** | 0.00 |

→ **The standard thresholds are constant classifiers**: `mean_psi`/`max_psi`
fire on every pair (0 miss, 100% false alarm); `mean_psi_numeric` never fires
(100% miss). No discriminating power. The full sweep confirms no threshold works
well here.

### 8.5 Harmful-ranking AUC
MMD **0.75** (CI 0.70–0.79), mean PSI numeric 0.73, mean KS 0.72, domain AUC
0.61, pred_shift 0.60, max PSI 0.59.

---

## 9. The numbers — employment (the cross-task check)

Task = predict employment (16 features). **Key result: it does NOT replicate.**

- Mean acc_drop = **0.007** (much smaller than income's 0.022); only 9 of 210
  pairs are "harmful".
- Correlations: MMD 0.14, mean PSI numeric 0.19, KS 0.19, domain AUC 0.17,
  pred_shift 0.07, **ATC/DoC −0.24** (negative!).
- Predictor importance: **every** signal has LOO R² ≈ 0 or negative — including
  label shift (−0.003). Nothing explains the (tiny) drop.
- Decomposition: concept_share = +0.006 (SE 0.006), covariate_share = +0.001;
  median ESS 0.80; only 4.3% low-ESS pairs. Stable — but the drop is small.
- Alerts: `max_psi ≥ 0.10` → false-alarm 0.86, miss 0.11; `mean_psi_numeric ≥
  0.10` → miss 0.89. Thresholds fail again, differently.
- Harmful-ranking AUC (wide CIs — only 9 positives): domain AUC 0.73, MMD 0.66,
  pred_shift 0.40, ATC/DoC 0.24.

**The takeaway:** "which score works" depends on the task and the drift
magnitude. A threshold tuned on income would be useless on employment. This is
exactly the general-finding leap the project needed.

---

## 10. The numbers — temporal frontier (2018→2019, income)

Train on a state in 2018, test on the same state in 2019. 15 rows.

- `acc_drop` median ≈ **0.12** (range 0.03–0.24) — large.
- `mean_psi` ≈ **1.69** (huge) but `mmd` ≈ 0.001 (inputs barely move).
- `concept_share` ≈ **0.25** — the decomposition attributes the drop to concept shift.

**Interpretation (critical):** the $50k cutoff is **not inflation-adjusted**.
Over a year, more nominal incomes cross $50k → a large *label* shift with almost
no input movement. So the temporal drop is an **upper bound** on true temporal
concept shift — it's mostly a threshold artifact. This is the cleanest real
demonstration that "monitoring saw big PSI" ≠ "the inputs really drifted."

---

## 11. The live monitor + API

`monitor.py` — `DriftMonitor(reference, warn=0.1, alert=0.25)`:
- `check(batch)` computes PSI vs the stored reference, takes `max_psi`,
  sets `status = alert if max_psi ≥ 0.25 else warn if ≥ 0.1 else ok`.
- Returns `n_rows`, `status`, `mean_psi`, `max_psi`, `mean_ks`,
  `top_drifting_features` (top-3 by PSI), `domain_auc` (only when batch ≥ 500),
  and the rule string.

`api/app.py` — FastAPI (uses `artifacts/model.joblib` + `reference.csv`):
- `GET /health` → model loaded? features.
- `POST /predict` → `probability_income_over_50k`, `prediction` (threshold 0.5).
- `POST /drift` → the monitor's check output.
- `GET /` → dashboard (api/dashboard.py) reading monitoring history.

`check_new_data.py` — offline CLI: `python check_new_data.py my_rows.csv` or
`--state NY`; prints the monitor report + the share predicted >$50k.

**The honest caveat to always say:** `/drift` and the monitor see only *inputs*.
They cannot see accuracy loss until labels arrive — which is precisely the gap
the whole project studies.

---

## 12. How to run everything (for the demo)

```powershell
.venv\Scripts\Activate.ps1
python -m pytest -q                 # all tests pass (ND + second-task)
python run_experiment.py --synthetic --n-max 8000   # fast smoke test
python run_experiment.py            # real: 210 rows → outputs/results.csv
python run_analysis.py              # tables/plots → outputs/
python train_service_model.py
python -m uvicorn api.app:app --reload   # open http://127.0.0.1:8000/docs
python check_new_data.py --state NY --synthetic
```

---

## 13. The 90-second spoken script

1. **Problem:** in production you can't see labels, so you monitor input-drift
   scores and alert on thresholds. Does that actually track accuracy? Do the
   standard thresholds work?
2. **Testbed:** natural US-census shifts — 15 states, 210 cross-state pairs, plus
   a temporal 2018→2019 frontier, plus a second task (employment), plus two models.
3. **Finding 1:** on income, numeric drift scores (MMD, PSI-numeric, KS) track
   accuracy loss *moderately* (ρ ≈ 0.3–0.44); the model-aware `pred_shift` does
   not.
4. **Finding 2:** label shift explains the most (LOO R² ≈ 0.36); the error
   decomposition says concept shift dominates covariate shift — but it's fragile
   for low-overlap pairs, so we report it with ESS diagnostics.
5. **Finding 3:** the standard PSI thresholds are effectively useless — always-on
   or always-off, 0% miss but 100% false alarm.
6. **Finding 4:** it does NOT generalize to employment — nothing (not even label
   shift) tracks the small drop there. So thresholds are task-specific.
7. **Finding 5 (temporal):** 2018→2019 shows a huge PSI but almost no input
   movement — the non-inflation-adjusted $50k cutoff creates a label artifact.
   A live demo of "drift ≠ accuracy."
8. **Limitations:** one modality (tabular census), pairs share a source model
   (hence cluster/group LOO), $50k cutoff not state-adjusted.
9. **Positioning:** we differ from PAPE (an estimator), TableShift (a benchmark),
   and Failing Loudly (synthetic perturbations) — we audit *why* drift scores fail
   and whether the thresholds work, on natural shifts.

---

## 14. One-line answers to likely questions

- **"Did you build a new detector?"** No. We audit existing ones and the
  thresholds practitioners actually use.
- **"Isn't this just a correlation?"** We go further: predictor importance with
  honest group-LOO, a covariate-vs-concept decomposition with diagnostics, and a
  threshold audit.
- **"Why two tasks?"** To check generality. The result is that it does NOT fully
  generalize — which is itself the key finding.
- **"Why is pred_shift failing?"** It measures a *model-output* shift, which can
  be large with no accuracy effect (e.g. recalibration without losing
  discriminative power).
- **"Is the $50k cutoff a problem?"** Yes, and we say so. It inflates temporal
  concept shift; we report per-source and low-ESS variants so the reader sees the
  sensitivity.
