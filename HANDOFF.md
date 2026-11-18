# DriftWatch — Complete Handoff for a New LLM

This document is a full, self-contained briefing on the DriftWatch project: what it is,
what has been built, what the current state is, what is still in flight, and what a
continuation must do next. Read this before touching any code.

---

## 1. What DriftWatch is

A **Python research + monitoring system** for deployed ML models. Two halves:

1. **A research benchmark** — does *label-free* drift detect *real accuracy loss*?
   It trains a model on one US state (Census/ACS income task) and evaluates it on
   many other states, measuring both drift scores (no labels) and the true accuracy
   drop. The research question: which kinds of shift (input drift, label shift,
   concept drift) explain accuracy loss, and how much of it do common drift scores
   capture?
2. **A live monitoring API** — FastAPI service that serves a trained model and
   checks new batches for drift against a stored reference sample.

Project root: `C:\Users\Pritesh\Downloads\driftwatch\driftwatch`
Git remote: `https://github.com/prrtsh0/DriftWatch.git` (branch `main`)

---

## 2. Folder layout

```
driftwatch/                     # the actual python project (git root)
├── driftwatch/                 # core package
│   ├── data.py                 # load one US state (folktables ACS income) or synthetic
│   ├── drift.py                # label-free drift scores: PSI, KS, domain-AUC
│   ├── model.py                # HistGradientBoosting + logistic regression
│   ├── experiment.py           # run_experiment() / run_temporal_experiment()
│   ├── analysis.py             # tables + charts: correlations, decomposition, alerts
│   ├── monitor.py              # DriftMonitor used by the API (warn PSI .1 / alert .25)
│   ├── baselines.py            # accuracy-estimation baselines: pred_shift, ATC, DoC, MMD,
│   │                           #   plus the importance-reweighting decomposition
│   └── __init__.py
├── api/app.py                  # FastAPI: /health, /predict, /drift
├── tests/                      # pytest suite (currently 18 tests)
├── artifacts/                  # trained model.joblib, reference.csv, metrics.json
├── outputs/                    # experiment results + analysis tables/charts
├── run_experiment.py           # CLI entry for the benchmark
├── run_analysis.py             # CLI entry that turns results.csv into tables/charts
├── train_service_model.py      # trains the model the API serves + saves reference
├── check_new_data.py           # drift-check a new CSV / census state vs deployed reference
├── tmp_run_full_hgb.py         # runner for the full 15-source HGB experiment (explicit config)
├── requirements.txt, Dockerfile, README.md, PROJECT_STRUCTURE.md, RESEARCH_GAP.md
```

---

## 3. The research design

- **Task:** US Census income (Folktables `ACSIncome`): predict income > $50k.
  Features (10): AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P.
- **Source/target:** train on a source state, test on target states.
- **Drift scores (label-free):** mean/max PSI, mean PSI (numeric), mean KS,
  domain-classifier AUC, pred_shift. None use true labels.
- **Ground truth:** `acc_drop = ref_acc - target_acc` computed on real labels.
- **Hypothesis:** *label shift* drives the accuracy gap, so input-only drift scores
  will miss it.
- **Related work to cite/differ from:** PAPE (NeurIPS 2025), TableShift (NeurIPS
  2023), Failing Loudly (NeurIPS 2019), Garg et al. ATC (2022), Guillory et al.
  DoC (2021).

---

## 4. The decomposition (the most-refined piece)

`baselines.py::decompose_error_shifts` separates the error **increase** into:

- `error_home` — model error on the reference holdout.
- `error_target` — model error on the new (target) data.
- `error_cov_only` — error on the reference, **reweighted** so the covariate
  distribution matches the target (importance reweighting). This is the error we
  would expect if only P(x) changed and P(y|x) were fixed.
- `covariate_share = error_cov_only - error_home` — part explained by input shift.
- `concept_share = error_target - error_cov_only` — residual: P(y|x) change +
  interaction + approximation error. Read as "the part not explained by covariate
  shift alone", NOT a clean causal estimate.

### Key design decisions (hard-won, do not regress)
1. **Cross-fitted weights.** The domain classifier (logistic) is fit with k-fold
   out-of-fold predictions over the FULL ref+target data, so no row is scored by a
   model trained on it. This prevents overconfident / overfit weights.
2. **Full-data reconciliation.** All errors AND weights are computed on the full
   data (NOT a subsample). This is critical: `covariate_share + concept_share`
   must equal `acc_drop` **exactly**. There is a regression test for this. (An
   earlier version subsampled to 2000 rows and the shares did NOT reconcile with
   full-data acc_drop — that was a real bug the reviewer caught.)
3. **Multi-cap truncation reporting.** Weights are normalized to mean 1 and clipped
   to `[1/cap, cap] × mean`. The decomposition is reported at caps 10, 20, 50
   (`concept_share_c10`, `concept_share` default 20, `concept_share_c50`) plus a
   "max drift across caps" summary. If the concept_share moves a lot across caps,
   the number is not solid.
4. **Bootstrap SE.** `covariate_share_se` and `concept_share_se` are reported
   (200 resamples, seeded). This quantifies sampling noise (~1.3–1.5pp on the
   pilot).
5. **ESS / clip fraction diagnostics.** `norm_weights_ess_ratio`, 
   `norm_weights_max`, `norm_weights_clip_fraction` are reported per pair. Low ESS
   (< ~0.2) means weights are unstable and concept_share is dominated by
   reweighting noise.
6. **Reproducibility.** All stochastic internals use fixed local seeds
   (`StratifiedKFold(random_state=seed)`, `np.random.default_rng(seed)`). The
   global RNG is also pinned (`np.random.seed(0)`) at run entry for defense.
   Verified: results are identical across different global RNG states.

---

## 5. What has been verified (tests — 18 passing)

- PSI zero for identical data; large for shifted; categorical known-value (0.1696).
- KS detects shift; monitor status transitions ok→warn→alert.
- API end-to-end (health, predict, drift, 422 on missing columns) via TestClient.
- Baselines run on categorical features; MMD zero/positive; pred_shift non-negative;
  ATC/DoC bounded.
- **Decomposition:** consistent keys; error_target matches measured; shares sum
  exactly to total error increase; no-change → zero shares; covariate-only known
  answer → concept_share ≈ 0; full-data reconciliation on data > 2000 rows.
- Density-ratio weights clipped at imposed bounds; tighter cap clips more.

Run with: `.venv\Scripts\python.exe -m pytest -q` (18 passed).

---

## 6. Full run (IN FLIGHT — most important current state)

A **full 15-source HGB experiment** is running in the background right now
(PID 33564), launched via `tmp_run_full_hgb.py` with explicit flags:

```
source/t = all 15 states (CA TX NY FL WA IL OH MI MN MS AL WV ME UT MA)
model = hgb, year = 2018, n_max = 30000, synthetic = False
out = outputs/results.csv, metadata -> outputs/run_meta_full_hgb.json
```

- **Pilot timing:** 12 rows (3 sources × 5 targets) took ~28s at n_max 8000
  *before* the full-data cross-fit change; ~53s *after*. The full run therefore
  takes roughly **30–45 min** (210 rows, n_max 30000). Plan for the high end.
- The run records the commit hash next to results (metadata file).

**Do not modify code while this run is in flight** — that would mix pipeline
versions across the 210 rows. Wait for it to finish.

---

## 7. What must be done AFTER the full run finishes

1. **Verify the output**: exactly 210 rows, no NaN in score columns, no source or
   target state missing.
2. Run `python run_analysis.py` (reads `outputs/results.csv` by default — pass
   `--csv` explicitly if analyzing any other run).
3. **Paste to the user** (they will tell you which claims the tables support):
   - correlation table
   - harmful-AUC table
   - shift_decomposition rows (with ESS ratio, clip fraction, SE columns)
4. Then run the other jobs, each writing to its OWN file (already safe by default):
   - `python run_experiment.py --model logreg` → `outputs/results_logreg.csv`
   - `python run_experiment.py --temporal` → `outputs/results_temporal.csv`
   - Re-run `run_analysis.py` for each with `--csv <file> --outdir <dir>`.
5. **The user writes the claims from the tables themselves.** Do NOT rewrite
   `RESEARCH_GAP.md` or call synthetic numbers "great". The user explicitly wants
   claims written from the actual result tables.

---

## 8. Current background processes

- `PID 33564` — full HGB experiment (running).
- `PID 31532` — uvicorn API on `http://127.0.0.1:8000` (running, fresh, serves the
  retrained real model — verified `/health`: model_loaded=true).
- (Old PID 24072 had failed to bind port 8000 because it was in use; that was the
  harmless "2 steps, 1 failed" — now resolved, fresh server runs.)

To stop a background process: use `stop_background` with the PID.

---

## 9. Environment / how to run

Windows, PowerShell. From the project root:
```powershell
.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pytest -q          # 18 tests
.venv\Scripts\python.exe run_experiment.py --synthetic --n-max 8000   # fast sanity
.venv\Scripts\python.exe run_experiment.py                             # real census
.venv\Scripts\python.exe run_analysis.py
.venv\Scripts\python.exe train_service_model.py
.venv\Scripts\python.exe -m uvicorn api.app:app --reload   # then open :8000/docs
```
`run_experiment.py` has safe per-mode output defaults (results.csv for HGB,
results_logreg.csv for logreg, results_temporal.csv for temporal) so runs cannot
overwrite each other.

---

## 10. Communication guardrails (from the user)

- The user is a rigorous reviewer. They check numbers against claims. Do not
  over-claim from small samples (e.g. "concept shift dominates" from 12 pilot rows
  was correctly called out as too strong).
- Prefer showing the actual numbers/tables over asserting conclusions.
- Respect explicit constraints: do not touch RESEARCH_GAP.md; do not hand work
  back as instructions; do the work yourself with the tools.
- State clearly when something is a guess/estimate vs verified.
