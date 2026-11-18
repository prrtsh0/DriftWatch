# DriftWatch

A monitoring system for deployed ML models, plus a study of whether label-free drift scores predict real accuracy loss.

**Research question:** across real US census shifts between states, which kinds of shift (inputs, labels, input-label relationship) explain accuracy loss, and how much of it do common drift scores (PSI, KS, domain-classifier AUC) capture?

## Setup (Windows PowerShell, from this folder)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q                 # all tests should pass
```

## Run order
1. `python run_experiment.py --synthetic --n-max 8000`   (no internet; checks the code works. NOT research results)
2. `python run_experiment.py`   (real census data; first run downloads each state from census.gov, takes a while)
   Try also `--model logreg`, and `--out outputs/results_logreg.csv`.
3. `python run_analysis.py`   (tables + chart in `outputs/`)
4. `python train_service_model.py`  then  `python -m uvicorn api.app:app --reload`  and open http://127.0.0.1:8000/docs
   (add `--synthetic` to step 4's first command if you have no data yet)

## What each file does
| File | Job |
|---|---|
| driftwatch/data.py | loads one US state (Folktables income task), or synthetic data for tests |
| driftwatch/drift.py | PSI, KS, domain-classifier AUC: drift scores that never use labels |
| driftwatch/model.py | the two models: gradient boosting and logistic regression |
| driftwatch/experiment.py | trains on a source state, tests on many target states, records drift scores AND true accuracy drop |
| driftwatch/analysis.py | correlations with bootstrap intervals, decomposition, false-alarm/miss rates, chart |
| driftwatch/monitor.py | the live drift checker used by the API (warn at PSI 0.1, alert at 0.25) |
| api/app.py | FastAPI service: /predict and /drift |
| tests/ | unit tests including a PSI hand-calculation |

## Live API + dashboard

The monitoring logic ships as a FastAPI service. Start it locally:

```powershell
python train_service_model.py        # (optionally --synthetic) writes artifacts/
python -m uvicorn api.app:app --reload
# open http://127.0.0.1:8000/docs  (Swagger UI) or /dashboard
```

Endpoints:
| Route | Purpose |
|---|---|
| `GET /health` | service status, whether a model is loaded, feature list |
| `POST /predict` | probability of income > $50k + binary prediction for a batch |
| `POST /drift` | compare a new batch against the stored reference: PSI/KS/domain-AUC, status `ok/warn/alert` |
| `GET /dashboard` | self-contained HTML dashboard (server-rendered charts, no CDN) |
| `GET /drift-data` | JSON for a custom frontend |

Example request (`/predict` — all 10 features required):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"AGEP":45,"COW":5,"SCHL":18,"MAR":2,"OCCP":2200,"POBP":6,"RELP":0,"WKHP":40,"SEX":1,"RAC1P":1}]}'
```

```bash
curl -X POST http://127.0.0.1:8000/drift \
  -H "Content-Type: application/json" \
  -d '{"records":[{...10 features...},{...}]}'
```

Offline check of a new CSV or a census state: `python check_new_data.py my_rows.csv` or `python check_new_data.py --state NY`.

> **Important limit:** the monitor and `/drift` see only *inputs*. They cannot
> report accuracy until true labels arrive — which is exactly the gap the study
> investigates.

## Deployment (Render)

The **API + dashboard** are deployable as a demo service; the research/experiment
pipeline stays local (it is a batch job, not a web service).

- `render.yaml` — Render Blueprint (web service, Docker, `/health` check, auto-deploy).
- `Dockerfile` — builds a self-contained image: it **trains a model during the
  build** (synthetic by default, since `artifacts/` is gitignored) so the app
  starts with a model ready.

Deploy steps:
1. Push this repo to GitHub.
2. Render → **New → Blueprint → connect the repo** (Render reads `render.yaml`).
3. After the build finishes, open `https://<service>.onrender.com/health`.

To serve a real income model instead of the synthetic demo, build with
`--build-arg TRAIN_FLAGS="--state CA"` (downloads ACS 2018 CA during build).

> Caveats: free-tier instances sleep after ~15 min idle (cold start ~30–60s) —
> fine for a demo, not for continuous monitoring; endpoints are unauthenticated,
> so this is for demo/education only.

## You must be able to explain (before presenting)
- Drift vs accuracy drop: why one can happen without the other.
- PSI by hand: the 50/50 vs 70/30 example gives about 0.17.
- KS: biggest gap between two cumulative curves. Domain AUC: can a model tell old rows from new rows?
- Label shift vs input drift: input scores cannot see a change in the label rate or the input-label link.
- Your own results table: what correlates, what does not, and one surprise.

## Limits (put these in the paper)
- One task (income prediction), tabular data, one survey year unless you add more.
- State pairs share a source model, so rows are not fully independent; bootstrap intervals are approximate.
- Synthetic results from this repo are for testing only.
- Related work to cite and differ from: PAPE (NeurIPS 2025), TableShift (NeurIPS 2023), Failing Loudly (NeurIPS 2019).
