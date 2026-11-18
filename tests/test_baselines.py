import numpy as np
import pandas as pd

from driftwatch.baselines import atc, compute_baselines, doc, mmd, pred_shift
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
