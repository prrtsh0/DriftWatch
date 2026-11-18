import numpy as np
import pandas as pd

from driftwatch.analysis import (harmful_auc, per_pair_cap_drift,
                                 predictor_importance, shift_decomposition,
                                 shift_decomposition_by_source)


def _fake_df(n=20):
    rng = np.random.default_rng(0)
    sources = [f"S{i % 5}" for i in range(n)]
    targets = [f"T{i % 6}" for i in range(n)]
    acc_drop = rng.uniform(-0.02, 0.12, n)
    return pd.DataFrame({
        "source": sources, "target": targets,
        "acc_drop": acc_drop,
        "mean_psi": rng.uniform(0, 0.3, n),
        "mean_psi_numeric": rng.uniform(0, 0.2, n),
        "max_psi": rng.uniform(0, 0.5, n),
        "mean_ks": rng.uniform(0, 0.4, n),
        "domain_auc": rng.uniform(0.5, 0.9, n),
        "pred_shift": rng.uniform(0, 0.2, n),
        "atc": rng.uniform(0, 1, n),
        "doc": rng.uniform(0, 1, n),
        "mmd": rng.uniform(0, 0.1, n),
        "label_shift": rng.uniform(0, 0.3, n),
        "error_home": rng.uniform(0.1, 0.3, n),
        "error_target": rng.uniform(0.1, 0.4, n),
        "error_cov_only": rng.uniform(0.1, 0.3, n),
        "covariate_share": rng.uniform(-0.02, 0.03, n),
        "concept_share": rng.uniform(-0.02, 0.10, n),
        "covariate_share_se": rng.uniform(0.003, 0.01, n),
        "concept_share_se": rng.uniform(0.003, 0.01, n),
        "norm_weights_ess_ratio": rng.uniform(0.1, 1.0, n),
        "norm_weights_max": rng.uniform(1, 20, n),
        "norm_weights_clip_fraction": rng.uniform(0, 0.2, n),
        "concept_share_c10": rng.uniform(-0.02, 0.10, n),
        "concept_share_c50": rng.uniform(-0.02, 0.10, n),
    })


def test_predictor_importance_has_group_loo_and_mmd_ks():
    df = _fake_df()
    out = predictor_importance(df)
    assert "r2_loso" in out.columns and "r2_loto" in out.columns
    names = set(out["predictors"])
    assert any("mmd" in s for s in names)
    assert any("mean_ks" in s for s in names)
    assert out["loo_r2"].notna().all()


def test_shift_decomposition_includes_lowess_exclusion():
    df = _fake_df()
    out = shift_decomposition(df)
    comps = set(out["component"])
    assert any("low-ESS" in c for c in comps)


def test_shift_decomposition_by_source_has_one_row_per_source():
    df = _fake_df()
    out = shift_decomposition_by_source(df)
    assert out["source"].nunique() == len(out)
    assert set(out.columns) >= {"source", "concept_share", "covariate_share"}


def test_per_pair_cap_drift_returns_sorted():
    df = _fake_df()
    out = per_pair_cap_drift(df)
    assert "cap_drift" in out.columns
    # sorted descending by cap_drift
    assert (out["cap_drift"].diff().dropna() <= 0).all()


def test_harmful_auc_has_ci_columns():
    df = _fake_df()
    out = harmful_auc(df, harmful_drop=0.03)
    assert "ci_low" in out.columns and "ci_high" in out.columns
    assert out["auc_for_harmful"].between(0, 1).all()
