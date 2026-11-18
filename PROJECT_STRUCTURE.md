# DriftWatch — Complete Project Structure (Handoff for a New LLM/Developer)

This document explains the entire project from scratch. Read this first; read the
source files second. No prior context is assumed.

---

## 0. What is DriftWatch? (one-liner)

A Python system that (a) **monitors a deployed ML model for data drift** using
label-free scores, and (b) is a **research study** asking whether those drift
scores actually predict real accuracy loss, using US Census data.

**Two faces of the same repo:**
1. **Research pipeline** — train a model on one state, test it on others, and
   measure whether drift scores correlate with the real accuracy drop.
2. **Live service** — a FastAPI app that serves predictions and runs the drift
   monitor on new batches, with a dashboard.

---

## 1. Repository layout

The git repo root is **`driftwatch\driftwatch\`** (note: there is a parent
folder also named `driftwatch` — always work inside the nested one).

```
driftwatch\driftwatch\          <- GIT ROOT
├── .gitignore                  # ignores data/, artifacts/, outputs*/, .venv/, caches
├── README.md                   # setup + run order + what each file does
├── DEPLOYMENT.md               # production deployment + continuous drift management guide
├── RESEARCH_GAP.md             # how to explain the research gap to a professor
├── PROJECT_STRUCTURE.md        # THIS FILE
├── requirements.txt            # python dependencies
├── Dockerfile                  # container build for the FastAPI service
│
├── driftwatch\                 # the core Python package
│   ├── __init__.py
│   ├── data.py                 # data loading (real Folktables census + synthetic fallback)
│   ├── drift.py                # label-free drift scores: PSI, KS, domain classifier AUC
│   ├── model.py                # two models: HistGradientBoosting + LogisticRegression
│   ├── experiment.py           # the benchmark: train source, test targets, record drift+drop
│   ├── analysis.py             # turns results into paper tables/charts
│   └── monitor.py              # DriftMonitor class used by the live API
│
├── api\
│   ├── __init__.py
│   ├── app.py                  # FastAPI service: /health /predict /drift /dashboard
│   └── dashboard.py            # server-side HTML dashboard rendering (matplotlib -> base64)
│
├── tests\
│   └── test_driftwatch.py      # 7 unit tests
│
├── artifacts\                  # GENERATED, git-ignored (model + reference + history)
│   ├── model.joblib            # trained model (joblib dump)
│   ├── reference.csv           # reference sample the monitor compares against
│   ├── metrics.json             # model metadata + confusion matrix
│   └── monitor_history.jsonl    # appended drift-check records (powers dashboard)
│
├── outputs\                    # GENERATED, git-ignored (research results + charts)
│   ├── results.csv              # 42 rows: source/target pairs with drift scores + accuracy drop
│   ├── results_logreg.csv       # same, for logistic regression model
│   ├── correlations.csv         # Spearman + bootstrap CI per drift score
│   ├── predictor_importance.csv # leave-one-out R² (non-causal; see docs)
│   ├── shift_decomposition.csv  # covariate vs concept shift via reweighting
│   ├── alerts.csv               # false-alarm / miss rate audit
│   ├── harmful_auc.csv          # ROC AUC for ranking harmful pairs
│   └── drift_vs_accuracy_drop.png
│
├── data\                       # GENERATED, git-ignored (downloaded census parquet files)
├── run_experiment.py           # CLI entry point for the benchmark
├── run_analysis.py             # CLI entry point for analysis
├── train_service_model.py      # trains the model the API serves
├── check_new_data.py           # CLI drift check of new data vs the deployed reference
├── demo_simulate_drift.py      # simulates drift history for the dashboard
├── conftest.py                 # pytest config (empty)
└── .venv\                     # python virtualenv (local, not committed)
```

---

## 2. Data (data.py)

Real data = **US Census ACS income task** via the `folktables` package.
One US state = one dataset.

- Target `y` = `PINCP`: income > $50k → 1, else 0.
- Default year 2018, up to 30,000 rows per state (subsampled for equal weight).
- `load_state(state, year, n_max, synthetic, root_dir)` returns `(X, y)`.
  - `synthetic=True` → `make_synthetic_state()` generates fake data for tests
    (no internet). Inputs, label rate and input-label link differ a little per
    state, so even tests exercise covariate/label/concept shift.

**10 features:**
- Numeric (3): `AGEP` (age), `SCHL` (schooling), `WKHP` (hours/week)
- Categorical (7): `COW`, `MAR`, `OCCP`, `POBP`, `RELP`, `SEX`, `RAC1P`

---

## 3. Drift scores (drift.py) — the "label-free" core

All three never use labels, so they work at inference time.

| Score | What it measures | Implementation |
|---|---|---|
| **PSI** (Population Stability Index) | How much one feature's distribution moved | Per-feature. Numeric: 10 buckets from reference quantiles. Categorical: top-20 frequency, rare grouped as "other". `sum((q-p)*ln(q/p))` with floor `EPS=1e-4` |
| **KS** | Biggest gap between two cumulative curves | `scipy.stats.ks_2samp` on the 3 numeric features |
| **Domain AUC** | Can a model tell old rows from new rows? | Trains HistGradientBoosting to classify origin (old=0/new=1), reports cross-val ROC AUC |

Helpers:
- `psi_table(ref, cur)` → Series of PSI per feature
- `ks_table(ref, cur)` → Series of KS per numeric feature
- `drift_summary(ref, cur)` → dict: `mean_psi`, `max_psi`, `mean_ks`,
  `mean_psi_numeric`, `domain_auc`

---

## 4. Models (model.py)

`build_model(kind)`:
- `"hgb"` → `HistGradientBoostingClassifier(max_iter=200)`
- `"logreg"` → Pipeline: OneHotEncoder (min_frequency=20, handle_unknown="ignore")
  on categoricals + StandardScaler on numerics → `LogisticRegression`

Two models are used so conclusions don't depend on one algorithm.

---

## 5. The research benchmark (experiment.py)

`run_experiment(sources, targets, model_kind, year, n_max, synthetic, out_csv)`:

For each source state:
1. Load it, 70/30 stratified split (seed 0).
2. Train model on 70%, compute **home accuracy** on the 30% (`ref_acc`).
3. For every *other* target state:
   - Predict, compute accuracy there (`target_acc`) → `acc_drop = ref_acc - target_acc`.
   - Record drift scores between reference sample and target (PSI/KS/domain AUC).
   - Record `label_shift` = |target label rate − train label rate| (uses labels — *explains*, not a drift score).
   - Record `pred_shift` = |mean predicted prob on target − mean on ref| (**label-free, model-aware**).

Writes `outputs/results.csv` (one row per source/target pair).

Key idea: this is a **simulated deployment** — we have the true labels for
target states, so we can measure the *actual* accuracy drop and compare it with
what the drift scores predict. A real deployment can't do this (new data has no
labels yet).

---

## 6. Analysis (analysis.py)

`run_analysis(csv, outdir, harmful_drop=0.03)` produces:

- **`correlations.csv`** — Spearman ρ between each drift score and `acc_drop`,
  with a **2000-sample bootstrap 95% CI** (robust, handles outliers).
- **`predictor_importance.csv`** — **leave-one-out R²**: how well each recorded
  signal predicts `acc_drop` (input-only vs label base-rate vs model-aware).
  Negative R² = worse than predicting the mean. **Non-causal**: it measures
  predictive importance, not a causal separation of shift types.
- **`shift_decomposition.csv`** — the **causal, methodologically-grounded**
  counterpart: importance-reweighting decomposition of the error increase into
  **covariate_share** (the part explained by a change in P(x) alone) and
  **concept_share** (the residual = change in P(y|x) + interaction). This is what
  directly addresses "concept shift is never measured".
- **`alerts.csv`** — treats "alert when score ≥ threshold" as a classifier vs
  "harmful = acc_drop ≥ 0.03". Reports `tp/fp/fn/tn`, **false_alarm_rate**, **miss_rate**.
- **`harmful_auc.csv`** — ROC AUC for ranking harmful vs harmless pairs by each score.
- **Chart** — scatter of each score vs accuracy drop.
- **`alerts` threshold choices**: mean_psi@0.10, mean_psi_numeric@0.10,
  max_psi@0.10, max_psi@0.25.

---

## 7. The live monitor (monitor.py)

`DriftMonitor(reference, warn=0.1, alert=0.25, min_domain_n=500)`:
- `check(batch)` → dict:
  - `n_rows`, `status` (`ok`/`warn`/`alert` based on `max_psi` thresholds)
  - `mean_psi`, `max_psi`, `mean_ks`
  - `feature_psi` (full per-feature PSI dict)
  - `top_drifting_features` (top-3 list)
  - `domain_auc` (only if batch ≥ 500 rows, else None)
  - `rule` (human-readable threshold explanation)

---

## 8. The FastAPI service (api/app.py)

Lifespan loads `artifacts/model.joblib` and builds `DriftMonitor` from
`artifacts/reference.csv` (path from env `DRIFTWATCH_ARTIFACTS`, default `artifacts`).

| Endpoint | Method | Behavior |
|---|---|---|
| `/health` | GET | `{status, model_loaded, features}` — readiness probe |
| `/predict` | POST | body `{"records":[{...10 features...}]}` → probabilities + binary prediction |
| `/drift` | POST | body = batch; runs monitor; **appends result to history**; stores as `state["latest"]` |
| `/dashboard` | GET | HTML dashboard (server-rendered charts) |
| `/drift-data` | GET | JSON `{latest, history}` for a custom frontend |

Validation: `/predict` and `/drift` require all 10 feature columns; `FEATURES`
list defines them. Missing columns → 422.

---

## 9. The dashboard (api/dashboard.py)

Self-contained HTML page. Charts drawn with matplotlib, embedded as **base64 PNGs**
(no internet/CDN, no frontend build).

- `load_history()` reads `artifacts/monitor_history.jsonl` (newest last).
- `append_history(record)` appends a check record with a timestamp.
- `build_dashboard(latest, history)` renders:
  - **Status cards**: status, n_rows, max/mean PSI, mean KS, domain AUC, pred_shift
  - **Status-history chart**: max PSI per check, with warn (0.10) / alert (0.25) lines
  - **Per-feature PSI chart**: horizontal bars with thresholds, colored by status
  - **Top-drifting-features table**

Note: cards + feature chart need a `latest` check (i.e., a `POST /drift` in the
current server session). The history chart always shows from the file.

---

## 10. The research gap (the "why this matters" story)

**Question:** across real US census shifts between states, which kinds of shift
(inputs, labels, input-label relationship) explain accuracy loss, and how much of
it do common drift scores (PSI, KS, domain AUC) capture?

**Confirmed results (real data, from `outputs/`):**

| Signal | Spearman ρ (95% CI) | LOO R² | Harmful AUC |
|---|---|---|---|
| mean PSI | 0.150 (−0.172, 0.446) | −0.047 | 0.669 |
| mean PSI (numeric) | 0.159 (−0.153, 0.452) | −0.065 | 0.616 |
| max PSI | −0.129 (−0.408, 0.178) | — | 0.472 |
| mean KS | 0.181 (−0.146, 0.489) | — | 0.569 |
| domain AUC | 0.203 (−0.110, 0.467) | −0.053 | 0.613 |
| **pred_shift** | **0.365 (0.065, 0.601)** | — | **0.815** |
| label shift | — | **+0.128** | — |
| input drift + label shift | — | **+0.147** | — |

**Headline findings:**
1. **Input-only scores are weak predictors.** All CIs straddle zero; LOO R² is
   negative (worse than guessing the average drop).
2. **Label shift explains more.** LOO R² ≈ +0.13 alone, +0.15 combined. Confirms
   the TableShift hypothesis on our own state pairs.
3. **PSI alert thresholds are effectively uninformative.** At 0.10 or 0.25:
   **0% miss but 100% false alarm** (fires on all 31 benign pairs; precision 26%
   = base rate, so no added information).
4. **The model-aware estimator is the best free signal.** `pred_shift` has the
   highest correlation (0.365) and highest harmful-ranking AUC (0.815).

**The gap (vs related work):**
- TableShift (NeurIPS 2023): benchmark linking gap to label shift — but does not
  decompose *how* input-only scores fail.
- PAPE (NeurIPS 2025): label-free *estimator* on 900+ census combos — the contrast
  method, not a study of *why* drift scores fail.
- Failing Loudly (NeurIPS 2019): compared detectors under *perturbations* — not
  natural shifts, not tied to accuracy.
- Podkopaev & Ramdas (ICLR 2022): sequential harmful-vs-benign tests — needs labels.
- Dong et al. (VLDB 2024): retrain on harmful drift only — motivates our false-alarm audit.

**Our claim:** *"PAPE and TableShift show drift signals are insufficient and label
shift matters; we measure, on real cross-state census shifts, how much accuracy
loss each common label-free score captures and whether the standard alert
thresholds work — they largely don't."*

---

## 11. How to run everything (from the git root)

Environment is `driftwatch\driftwatch\.venv\`. Activate, then:

| Task | Command |
|---|---|
| Tests | `python -m pytest -q` (7 should pass) |
| Populate dashboard history (demo) | `python demo_simulate_drift.py` |
| Live drift check CLI | `python check_new_data.py --state TX --synthetic` |
| Train the served model | `python train_service_model.py` |
| Start the API | `uvicorn api.app:app --reload --host 127.0.0.1 --port 8000` |
| Synthetic benchmark (fast, no net) | `python run_experiment.py --synthetic --n-max 8000` |
| Real census benchmark | `python run_experiment.py` |
| Analysis (tables + chart) | `python run_analysis.py` |
| View dashboard | open `http://127.0.0.1:8000/dashboard` |
| Swagger docs | open `http://127.0.0.1:8000/docs` |

---

## 12. Deployment / continuous drift management (summary)

Full detail in `DEPLOYMENT.md`. The essential loop:
1. DefaultCapture production rows at `/predict` (logging not yet implemented — open improvement).
2. Run scheduled `/drift` checks.
3. Alert on `warn`/`alert` (Slack/PagerDuty).
4. Verify harm with delayed labels.
5. Retrain (`train_service_model.py`), atomically swap `artifacts/`, bump `metrics.json`.

Honest limits: live monitoring sees *input drift only*; detecting *harm* requires
labels that arrive later — exactly the gap the paper studies.

---

## 13. Common gotchas

- **Wrong folder:** always `cd` into `driftwatch\driftwatch\` (nested). Running
  from the parent causes "No module named 'api'".
- **Root URL 404:** there is no route at `/`. Use `/dashboard`, `/docs`, `/health`.
- **Port in use:** close other uvicorn or use `--port 8010`.
- **`data/`, `artifacts/`, `outputs*/` are git-ignored** — never committed.
- **Default `python` may lack `folktables`** — use the venv (`\.venv\Scripts\python.exe`).
- **PSI on high-cardinality categoricals** (`POBP`, `OCCP`) can explode — watch
  `mean_psi_numeric` too.

---

## 14. Where to add new things

- New drift score → add to `drift.py` + `drift_summary`, propagate through
  `experiment.py` and `analysis.py`.
- New model → `model.py` `build_model`, then pass `--model` to experiments.
- New API endpoint → `api/app.py`.
- New dashboard viz → `api/dashboard.py` (matplotlib → base64).
- New tests → `tests/test_driftwatch.py`.
