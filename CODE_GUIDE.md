# DriftWatch — Code Guide (compact, for another LLM)

A single self-contained file describing the **entire codebase** — modules, key
functions, signatures, and how they connect. No data. Read this to understand or
modify the repo without inspecting every file.

Repo root: `driftwatch/`. Package: `driftwatch/` (the inner folder).

---

## 1. Layout

```
api/app.py              FastAPI service (health / predict / drift / dashboard / homepage)
api/dashboard.py        Server-rendered HTML dashboard + shared site chrome
driftwatch/data.py      Real (Folktables ACS) + synthetic data loading; task registry
driftwatch/drift.py     Label-free drift scores: PSI, KS, domain-AUC
driftwatch/model.py     Two models: HistGradientBoosting, LogisticRegression
driftwatch/experiment.py  Benchmark: train on source state, test on many targets
driftwatch/baselines.py   Accuracy-estimation baselines: pred_shift, ATC, DoC, MMD, decomposition
driftwatch/analysis.py    Turn results.csv into tables + charts (paper)
driftwatch/monitor.py     DriftMonitor class used by the API
run_experiment.py / run_analysis.py / run_temporal.py / run_full_hgb.py / run_logreg.py
train_service_model.py     Train the served model + write artifacts/
check_new_data.py          Offline drift check for a new CSV or census state
tests/                     pytest suite (7 test files)
artifacts/                 model.joblib, reference.csv, metrics.json, monitor_history.jsonl
outputs*/                  results CSVs + charts (generated)
```

---

## 2. Data loading — `driftwatch/data.py`

- `FEATURES` (income, 10): AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P
- `NUMERIC`: AGEP, SCHL, WKHP · `CATEGORICAL`: the rest.
- Employment task: `EMPLOYMENT_FEATURES` (16), `EMPLOYMENT_NUMERIC` = AGE/SCHL.
- `TASKS` dict: per-task `features/numeric/categorical/y_name/synthetic`.
- `task_spec(task) -> dict` (raises on unknown).
- `load_state(state, year="2018", n_max=30000, seed=0, synthetic=False, root_dir="data") -> (X, y)`
  - synthetic → `make_synthetic_state(state, n, seed)` (fake states with slight covariate/label/concept shifts).
  - real → Folktables `ACSDataSource` + `ACSIncome`/`ACSEmployment`.
  - **Recode note:** `RELP` → `RELSHIPP` rename in 2019+ (ACS schema change).
    Verified: 2018 codes 0–9 vs 2019 codes 20–29 — **not** comparable; must be
    dropped for temporal analysis.

## 3. Drift scores — `driftwatch/drift.py`

- `psi_feature(ref, cur, categorical=False, bins=10, top_k=20)`: PSI for one column
  (0 = identical; bigger = more different). Numeric → quantile buckets; categorical → top-k + "other".
- `psi_table(ref, cur)`: Series of PSI per feature.
- `ks_table(ref, cur)`: KS statistic per numeric feature (`ks_2samp`).
- `domain_auc(ref, cur, n=4000, seed=0)`: train HGB to tell old vs new rows, report CV AUC.
- `drift_summary(ref, cur, with_domain=True)` → `{mean_psi, max_psi, mean_ks,
  mean_psi_numeric, domain_auc}`.

## 4. Models — `driftwatch/model.py`

- `build_model(kind="hgb")` → HistGradientBoostingClassifier(max_iter=200) or
  LogisticRegression pipeline (OneHotEncoder for categorical + StandardScaler for numeric).

## 5. Benchmark — `driftwatch/experiment.py`

- `run_experiment(sources, targets, model_kind="hgb", year="2018", n_max=30000,
  synthetic=False, out_csv="outputs/results.csv")`
  - For each source: train/test split (30% ref), fit model, `ref_acc` on ref.
  - For each target: predict, `target_acc`, `acc_drop = ref_acc - target_acc`.
  - Rows include `label_shift` (uses labels, for explanation only), `pred_shift`
    (model-aware), and `**drift_summary(...)` (label-free scores).

## 6. Baselines + decomposition — `driftwatch/baselines.py`

- `pred_shift(model, ref, cur, task)` → |mean prob target − mean prob ref|.
- `atc(model, ref, cur, k=100)` → Average Threshold Confidence shift.
- `doc(model, ref, cur, k=100)` → Difference of Confidences (top-k).
- `mmd(ref, cur, gamma=None, task)` → Gaussian RBF MMD on numeric features.
- `_density_ratio(ref, cur)` → importance-reweighting weights (dict keyed by cap 10/20/50).
- `decompose_error_shifts(model, ref, y_ref, cur, y_cur)` → separates the error
  **increase** into covariate vs concept:
  - `error_home`, `error_target`, `error_cov_only` (reweighted error),
  - `covariate_share = error_cov_only - error_home`,
  - `concept_share = error_target - error_cov_only` (residual — not a clean causal estimate),
  - plus `norm_weights_ess_ratio` (diagnostic).
  - Rules: weights computed on **full data** (no subsample); cross-fit model to
    avoid overfit weights; reported at cap 10/20/50.

## 7. Analysis — `driftwatch/analysis.py`

- `spearman_ci(df, col, target="acc_drop", n_boot=2000)` → Spearman ρ + bootstrap 95% CI.
- `correlations(df)` → table of ρ per drift score.
- `decomposition(df)` → leave-one-out R² for feature sets (input drift only / label shift only / both).
- `alert_eval(df, col, threshold, harmful_drop=0.03)` → TP/FP/FN/TN + false-alarm & miss rates.
- `harmful_auc(df, harmful_drop)` → ROC AUC for flagging harmful drops.
- `make_plots(df, outdir)` → scatter charts (drift score vs acc_drop).
- `run_analysis(csv, outdir, harmful_drop)` → writes tables + chart.

## 8. Live monitor — `driftwatch/monitor.py`

- `DriftMonitor(reference, warn=0.1, alert=0.25, min_domain_n=500)`
- `.check(batch)` → `{n_rows, status(ok/warn/alert), mean_psi, max_psi, mean_ks,
  top_drifting_features, domain_auc, rule}`. Status = alert if max_psi ≥ 0.25,
  warn if ≥ 0.10, else ok.

## 9. API — `api/app.py`

- `lifespan` loads model + reference + metrics; **seeds demo history** if
  `monitor_history.jsonl` is empty (3 rows: ok/warn/alert, tagged `source:"demo"`).
- Routes:
  - `GET /` → landing homepage (HTML, `homepage()`).
  - `GET /health` → `{status, model_loaded, features}`.
  - `POST /predict` → `{probability_income_over_50k, prediction}` (all 10 features required).
  - `POST /drift` → monitor check + `pred_shift`, then appends to history + sets `state["latest"]`.
  - `GET /dashboard` → HTML dashboard (`build_dashboard(latest, history)`).
  - `GET /drift-data` → JSON `{latest, history}`.

## 10. Dashboard — `api/dashboard.py`

- `load_history(limit=200)` reads `monitor_history.jsonl` (newest last).
- `append_history(record)` appends one record (adds timestamp).
- `_status_chart(history)` / `_feature_psi_chart(phi)` → matplotlib PNGs, base64-embedded.
- `build_dashboard(latest, history)` → full HTML page.
- `homepage()` → landing page.
- `_shell(title, active, inner, page_css)` → shared site header/nav + footer wrapper.
- Shared CSS constants `_SITE_CSS` / `_DASHBOARD_CSS` use **single braces**.

## 11. Training + check — `train_service_model.py`, `check_new_data.py`

- `train_service_model.py`: `ART = Path(os.getenv("DRIFTWATCH_ARTIFACTS", "artifacts"))`;
  writes `model.joblib`, `reference.csv`, `metrics.json` (with confusion matrix + `ref_prob_mean`).
- `check_new_data.py`: loads `artifacts/reference.csv` + model, then
  `DriftMonitor(reference).check(new)` and prints a JSON report incl. `share_predicted_over_50k`.

## 12. Docker / Render

- `Dockerfile`: `python:3.11-slim`, sets `ENV DRIFTWATCH_ARTIFACTS=artifacts`,
  defaults `TRAIN_FLAGS="--synthetic --n-max 20000"`, builds artifacts during build.
- `render.yaml`: Blueprint, web service, `/health` check, auto-deploy.
- Note: `artifacts/` and `outputs*/` are gitignored → Docker regenerates them at build.

---

## 13. Testing

- `tests/test_driftwatch.py` — PSI zero/large/categorical-known-value, KS, monitor status, API e2e.
- `tests/test_baselines.py` — MMD zero/positive, pred_shift≥0, ATC/DoC bounded,
  decomposition keys/reconciliation/no-change-zero, density-ratio clipping,
  covariate-only known-answer (concept_share≈0) and concept-only known-answer (covariate_share≈0).
- `tests/test_analysis.py` — analysis helpers.
- Run: `python -m pytest -q`.

---

## 14. Key research findings (for context)

- Income (210 state-pairs): MMD/numeric-PSI/KS track loss moderately; **label shift
  is the strongest predictor** (LOO R² 0.36); standard thresholds are constant classifiers.
- Employment: low power (only 9 harmful pairs) → reframed as underpowered cross-task check.
- Temporal 2018→2019: apparent drop is **entirely the RELP recode artifact**
  (removing RELP: acc_drop 0.123 → 0.006).
- Constraints: do not edit `RESEARCH_GAP.md` (user rule); pair rows share source model → use group-LOO.
