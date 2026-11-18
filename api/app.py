"""FastAPI service: /predict returns model predictions, /drift checks a batch for drift."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from driftwatch.data import FEATURES
from driftwatch.monitor import DriftMonitor
from api.dashboard import append_history, build_dashboard, load_history

import numpy as np

ART = Path(os.getenv("DRIFTWATCH_ARTIFACTS", "artifacts"))
state = {}


@asynccontextmanager
async def lifespan(app):
    if (ART / "model.joblib").exists():
        state["model"] = joblib.load(ART / "model.joblib")
        state["monitor"] = DriftMonitor(pd.read_csv(ART / "reference.csv"))
        if (ART / "metrics.json").exists():
            import json
            with open(ART / "metrics.json") as fh:
                state["ref_prob_mean"] = json.load(fh).get("ref_prob_mean")
    yield


app = FastAPI(title="DriftWatch", lifespan=lifespan)


class Batch(BaseModel):
    records: list[dict[str, float]]


def _frame(batch: Batch) -> pd.DataFrame:
    if "model" not in state:
        raise HTTPException(503, "No model loaded. Run train_service_model.py first.")
    df = pd.DataFrame(batch.records)
    missing = [f for f in FEATURES if f not in df.columns]
    if df.empty or missing:
        raise HTTPException(422, f"records must be non-empty and include {FEATURES} (missing: {missing})")
    return df[FEATURES]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "features": FEATURES}


@app.post("/predict")
def predict(batch: Batch):
    prob = state["model"].predict_proba(_frame(batch))[:, 1]
    return {"probability_income_over_50k": prob.round(4).tolist(), "prediction": (prob >= 0.5).astype(int).tolist()}


@app.post("/drift")
def drift(batch: Batch):
    df = _frame(batch)
    result = state["monitor"].check(df)
    # pred_shift: the model-aware signal (the study's best predictor). Requires
    # the reference probability mean saved by train_service_model.py.
    if "model" in state and state.get("ref_prob_mean") is not None:
        batch_prob_mean = float(state["model"].predict_proba(df)[:, 1].mean())
        result["pred_shift"] = abs(batch_prob_mean - state["ref_prob_mean"])
    else:
        result["pred_shift"] = None
    append_history(result)          # persist so the dashboard can show history
    state["latest"] = result
    return result


@app.get("/dashboard")
def dashboard():
    """Self-contained HTML dashboard (charts rendered server-side)."""
    history = load_history()
    latest = state.get("latest")
    from fastapi.responses import HTMLResponse
    return HTMLResponse(build_dashboard(latest, history))


@app.get("/drift-data")
def drift_data():
    """JSON for a frontend: latest check + history."""
    return {"latest": state.get("latest"), "history": load_history()}
