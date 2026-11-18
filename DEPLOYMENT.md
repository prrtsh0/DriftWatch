# DriftWatch — Deployment & Live Drift Management

This guide covers taking the DriftWatch research code and running it as a real,
continuously-monitored model service. It assumes you already have a trained model
(`artifacts/model.joblib`, `artifacts/reference.csv`) produced by
`python train_service_model.py`.

---

## 1. What you are deploying

The service is a FastAPI app that does three things:

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness/readiness probe (model loaded? which features?) |
| `POST /predict` | Model inference for one or many records |
| `POST /drift`  | Runs the label-free drift check on a batch vs the stored reference |
| `GET /dashboard` | HTML dashboard (charts rendered server-side, no JS build) |
| `GET /drift-data` | JSON of latest check + history (for a custom frontend) |

The drift monitor (`driftwatch/monitor.py`) computes PSI per feature, KS on
numeric features, and a domain-classifier AUC, returning a status of
`ok` / `warn` / `alert`. Every `POST /drift` is appended to
`artifacts/monitor_history.jsonl`, which powers the dashboard.

---

## 2. Local run (for development / demo)

```bash
# from the repo root (driftwatch/)
python -m venv .venv
.venv/Scripts/activate            # Windows
# .venv/bin/activate              # macOS/Linux
pip install -r requirements.txt

# train the model (uses real census CA data; downloads on first run)
python train_service_model.py

# start the API
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

Then open:
- Swagger UI: http://127.0.0.1:8000/docs
- Dashboard:    http://127.0.0.1:8000/dashboard

---

## 3. Containerised deployment (Docker)

The included `Dockerfile` builds the service. It expects `artifacts/` to exist
before building (train on your machine first).

```bash
# build
docker build -t driftwatch .

# run
docker run --rm -p 8000:8000 driftwatch
```

Note: `artifacts/` is git-ignored, so the model itself is not in the repo.
In real CI you would either (a) build the image after training in the pipeline
and pass artifacts via volumes/secrets, or (b) keep the model in a private
artifact store and mount it at runtime.

---

## 4. Production deployment (recommended options)

### Option A — single VM / VPS (cheapest, good for a research demo)
- Run the container behind a reverse proxy (nginx / Caddy / Traefik) that
  terminates TLS and forwards `/` to `localhost:8000`.
- Add a systemd unit (Linux) or a container orchestration tool to restart on crash.
- Expose only `443` (HTTPS) publicly; keep `8000` internal.

### Option B — cloud managed (best for "real" delivery)
- **Compute:** one small instance (e.g. AWS EC2 / GCP Compute Engine / a
  container service like AWS Fargate or Azure Container Apps).
- **TLS:** a managed HTTPS endpoint (ALB/Ingress/App Gateway).
- **Secrets:** model artifacts in a private object store (S3/GCS) or mounted volume.
- **Observability:** forward logs (uvicorn) to your logging stack; map
  `/health` into the platform's health check.

---

## 5. Continuous drift management (the "manage drift continuously" part)

A single `/drift` call is only the *detection* step. To actually manage drift
you need these pieces:

### 5.1 Capture production data
Currently `/predict` does **not** log the rows it serves. Add a small logging
step so you can build the "current batch" to test. Two clean options:
- Append every payload to a daily parquet/JSONL file (e.g. `logs/predict_YYYYMMDD.jsonl`).
- Or, if you already have a data warehouse, log the timestamp + feature vector there.

This also matters for your paper: a monitor that never sees production rows
can't tell you anything.

### 5.2 Scheduled drift checks
Run `DriftMonitor.check()` on a schedule (e.g. every hour / daily):

```bash
# cron: 0 * * * *  cd /app && python scheduled_check.py >> drift.log 2>&1
```

Where `scheduled_check.py` loads the latest captured batch, runs the monitor,
and either writes `/ JSON payload or alerts.

### 5.3 Alerting
Turn the `status` field into notifications:
- `warn` → Slack/Discord/email (low priority).
- `alert` → PagerDuty/Opsgenie (high priority).
- Include the top drifting features and `domain_auc` in the alert text so an
  engineer can act without opening a dashboard.

### 5.4 Dashboards
`/dashboard` already renders the history. For a hosted view, either:
- Serve `/dashboard` behind your proxy (no extra code), or
- Point Grafana at `/drift-data` (JSON) and build panels there.

### 5.5 Handling a confirmed alert (retraining loop)
Because drift scores are **label-free**, an alert only says "the inputs moved".
Before retraining you must decide whether the movement was *harmful*:
1. Collect labels for the flagged batch (delayed labels / manual review).
2. Measure the true accuracy drop (`acc_drop`).
3. If harmful → retrain on the new data, re-run `train_service_model.py`,
   atomically swap `artifacts/` (new model + new reference), and restart.
4. Update `metrics.json` with the new model version so it is auditable.

This "alert → verify with labels → retrain" loop is exactly the gap your paper
studies: drift detection is only the first step.

---

## 6. Operational checklist (production readiness)

- [ ] `/health` wired into the orchestrator/load-balancer health check.
- [ ] TLS enabled on the public endpoint.
- [ ] Request logging for `/predict` is enabled (for future drift batches).
- [ ] Scheduled drift check + alert route configured.
- [ ] Monitoring history file is persisted somewhere durable (volume/object store)
      so a restart does not lose it.
- [ ] Model artifacts are versioned and the deployed version is recorded.
- [ ] Set `DRIFTWATCH_ARTIFACTS` env var to the artifacts path in the container.

---

## 7. Honest limits to state (also in the paper)

- **No labels at inference time.** `/drift` measures input drift only. It cannot
  see accuracy, label shift, or concept drift until labels arrive.
- **PSI on high-cardinality categoricals** (`POBP`, `OCCP`) can explode; the
  monitor reports `mean_psi_numeric` too — keep an eye on the numeric subset.
- **Threshold values** (0.10 warn / 0.25 alert) are standard rules of thumb and
  are exactly what your paper evaluates — they are not validated for all systems.
