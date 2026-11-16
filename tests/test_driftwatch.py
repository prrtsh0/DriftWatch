import importlib

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from driftwatch.data import load_state
from driftwatch.drift import ks_table, psi_feature
from driftwatch.model import build_model
from driftwatch.monitor import DriftMonitor


def test_psi_zero_for_same_data():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi_feature(x, x) < 1e-6


def test_psi_large_for_shifted_data():
    rng = np.random.default_rng(0)
    assert psi_feature(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)) > 0.25


def test_psi_categorical_known_value():
    ref = pd.Series([0] * 50 + [1] * 50)
    cur = pd.Series([0] * 70 + [1] * 30)
    assert abs(psi_feature(ref, cur, categorical=True) - 0.1696) < 0.001   # the worked example


def test_ks_detects_shift():
    X, _ = load_state("CA", n_max=3000, synthetic=True)
    X2 = X.copy()
    X2["AGEP"] += 15
    assert ks_table(X, X2)["AGEP"] > 0.3


def test_monitor_status_changes_with_drift():
    X, _ = load_state("CA", n_max=4000, synthetic=True)
    m = DriftMonitor(X.iloc[:2000])
    assert m.check(X.iloc[2000:])["status"] == "ok"
    shifted = X.iloc[2000:].copy()
    shifted["AGEP"] += 25
    assert m.check(shifted)["status"] == "alert"


def test_logreg_model_trains():
    X, y = load_state("CA", n_max=3000, synthetic=True)
    assert build_model("logreg").fit(X, y).predict(X).shape[0] == 3000


def test_api_end_to_end(tmp_path, monkeypatch):
    X, y = load_state("CA", n_max=4000, synthetic=True)
    joblib.dump(build_model("hgb").fit(X, y), tmp_path / "model.joblib")
    X.sample(1000, random_state=0).to_csv(tmp_path / "reference.csv", index=False)
    monkeypatch.setenv("DRIFTWATCH_ARTIFACTS", str(tmp_path))
    import api.app as app_module
    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        assert c.get("/health").json()["model_loaded"] is True
        assert len(c.post("/predict", json={"records": X.head(5).to_dict("records")}).json()["prediction"]) == 5
        assert c.post("/drift", json={"records": X.head(600).to_dict("records")}).json()["status"] in {"ok", "warn", "alert"}
        assert c.post("/predict", json={"records": [{"AGEP": 30}]}).status_code == 422
