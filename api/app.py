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

ART = Path(os.getenv("DRIFTWATCH_ARTIFACTS", "artifacts"))
state = {}


@asynccontextmanager
async def lifespan(app):
    if (ART / "model.joblib").exists():
        state["model"] = joblib.load(ART / "model.joblib")
        state["monitor"] = DriftMonitor(pd.read_csv(ART / "reference.csv"))
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
    return state["monitor"].check(_frame(batch))
