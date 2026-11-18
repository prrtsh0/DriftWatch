import numpy as np
import pandas as pd

from driftwatch.baselines import (atc, compute_baselines, decompose_error_shifts,
                                  doc, mmd, pred_shift)
from driftwatch.data import FEATURES, load_state, make_synthetic_state
from driftwatch.model import build_model


def _model_ref_cur(n=2000):
    X, y = load_state("CA", n_max=n, synthetic=True)
    X2, _ = load_state("TX", n_max=n, synthetic=True)
    model = build_model("logreg").fit(X, y)
    return model, X, X2


def test_mmd_zero_for_identical_data():
    _, X, _ = _model_ref_cur()
    assert mmd(X, X) < 1e-3


def test_mmd_positive_for_shifted_data():
    model, X, X2 = _model_ref_cur()
    assert mmd(X, X2) > 1e-6


def test_pred_shift_non_negative():
    model, X, X2 = _model_ref_cur()
    assert pred_shift(model, X, X2) >= 0


def test_atc_doc_bounded():
    model, X, X2 = _model_ref_cur()
    assert 0 <= atc(model, X, X2) <= 1
    assert 0 <= doc(model, X, X2) <= 1


def test_compute_baselines_keys():
    model, X, X2 = _model_ref_cur()
    res = compute_baselines(model, X, X2)
    assert set(res) >= {"pred_shift", "atc", "doc", "mmd"}
    assert all(np.isfinite(v) for v in res.values())


def test_baselines_run_on_categorical_features():
    # ensure FEATURES columns exist and baselines don't crash on the full frame
    model, X, X2 = _model_ref_cur()
    assert set(X.columns) >= set(FEATURES)
    compute_baselines(model, X, X2)


def test_decompose_error_shifts_consistent():
    model, X, X2 = _model_ref_cur()
    y = (model.predict(X) >= 0.5).astype(int)
    y2 = (model.predict(X2) >= 0.5).astype(int)
    res = decompose_error_shifts(model, X, y, X2, y2)
    assert set(res) >= {"error_home", "error_target", "error_cov_only",
                        "covariate_share", "concept_share"}
    assert all(np.isfinite(v) for v in res.values())
    # error_target should be close to the measured (1 - acc) on target
    assert abs(res["error_target"] - (y2 != (model.predict_proba(X2)[:, 1] >= 0.5)).mean()) < 1e-9
    # covariate_share + concept_share should reconstruct the total error increase
    total = res["error_target"] - res["error_home"]
    assert abs((res["covariate_share"] + res["concept_share"]) - total) < 1e-6


def test_decompose_no_change_zero_shares():
    model, X, _ = _model_ref_cur()
    y = (model.predict(X) >= 0.5).astype(int)
    res = decompose_error_shifts(model, X, y, X.copy(), y.copy())
    assert abs(res["error_home"] - res["error_target"]) < 1e-6
    assert abs(res["covariate_share"]) < 1e-3
    assert abs(res["concept_share"]) < 1e-3
