"""Turn results.csv into the paper's tables and charts."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

DRIFT_COLS = ["mean_psi", "mean_psi_numeric", "max_psi", "mean_ks", "domain_auc",
              "pred_shift", "atc", "doc", "mmd"]


def spearman_ci(df, col, target="acc_drop", n_boot=2000, seed=0, cluster=None):
    """Rank correlation between a drift score and accuracy drop, with a bootstrap 95% interval.

    If `cluster` is a column name (e.g. "source"), the bootstrap resamples at the
    cluster level (a cluster bootstrap) instead of individual rows. This gives
    honest intervals when rows are not independent (pairs share a source model).
    """
    rho = spearmanr(df[col], df[target])[0]
    rng = np.random.default_rng(seed)
    vals = []
    if cluster is None:
        n = len(df)
        for _ in range(n_boot):
            s = df.iloc[rng.integers(0, n, n)]
            r = spearmanr(s[col], s[target])[0]
            if not np.isnan(r):
                vals.append(r)
    else:
        # Precompute integer row positions per cluster once, then resample by
        # concatenating positions. This avoids re-masking + re-concat'ing
        # dataframes on every bootstrap iteration (which was the bottleneck).
        groups = df[cluster].to_numpy()
        clusters = np.unique(groups)
        pos_by_cluster = [np.flatnonzero(groups == c) for c in clusters]
        k = len(clusters)
        for _ in range(n_boot):
            chosen = rng.integers(0, k, size=k)
            sel = np.concatenate([pos_by_cluster[c] for c in chosen])
            s = df.iloc[sel]
            r = spearmanr(s[col], s[target])[0]
            if not np.isnan(r):
                vals.append(r)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"score": col, "spearman": rho, "ci_low": lo, "ci_high": hi}


def correlations(df, cluster="source"):
    """Column-wise Spearman + bootstrap CI. Uses a cluster bootstrap on `cluster`
    (the source state) so the intervals account for rows sharing a source model.
    """
    rows = [spearman_ci(df, c, cluster=cluster) for c in DRIFT_COLS]
    return pd.DataFrame(rows)


def predictor_importance(df, group_cols=("source", "target")):
    """Predictor importance for accuracy drop.

    Reports three leave-one-out R^2 variants per predictor set:
      - loo_r2        : row-level LOO over individual pairs.
      - r2_loso       : leave-one-*source*-out (all pairs sharing a source model
                         are held out together, so the model never sees a pair
                         from the same source as the test rows).
      - r2_loto       : leave-one-*target*-out (likewise for target state).
    Group LOO is honest when pairs share a source/target; row-level LOO leaks
    because the same source model contributes many rows.

    NOTE: These are predictive-importance numbers, not causal effects. For an
    actual covariate-vs-concept separation, see `shift_decomposition`.

    MMD and KS are included (they are the strongest drift signals elsewhere), so
    "drift scores are far below label shift" can actually be tested against them.
    """
    sets = {"input drift only (mean_psi)": ["mean_psi"],
            "input drift only (mean_psi_numeric)": ["mean_psi_numeric"],
            "input drift only (domain_auc)": ["domain_auc"],
            "input drift only (mean_ks)": ["mean_ks"],
            "input drift only (mmd)": ["mmd"],
            "label base-rate shift only": ["label_shift"],
            "input drift + label base-rate shift": ["mean_psi", "label_shift"],
            "model-aware only (pred_shift)": ["pred_shift"]}

    def _loo_r2(cols, groups=None):
        X, y = df[cols].values, df["acc_drop"].values
        if groups is None:
            pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
        else:
            pred = np.empty(len(df))
            for g in df[groups].unique():
                test = df[groups].values == g
                tr = ~test
                lin = LinearRegression().fit(X[tr], y[tr])
                pred[test] = lin.predict(X[test])
        return r2_score(y, pred)

    rows = []
    for name, cols in sets.items():
        row = {"predictors": name, "loo_r2": _loo_r2(cols)}
        for g in group_cols:
            key = "r2_loso" if g == "source" else "r2_loto"
            row[key] = _loo_r2(cols, groups=g)
        rows.append(row)
    return pd.DataFrame(rows)


def shift_decomposition(df):
    """Covariate-vs-concept decomposition of the error increase using importance
    reweighting (computed per-pair in experiment.py).

    - covariate_share = mean(error_cov_only - error_home): the part of the
      average error increase attributable to a change in P(x) alone.
    - concept_share   = mean(error_target - error_cov_only): the residual, i.e.
      how much of the increase is NOT explained by covariate shift (change in
      P(y|x), plus any interaction/estimation error from the reweighting).
    This is the honest, methodologically-grounded counterpart to the (non-causal)
    predictor_importance table.
    """
    if "error_cov_only" not in df.columns:
        return pd.DataFrame({"note": ["reweighting decomposition not present in results"]})
    rows = [
        {"component": "mean error at home (reference)", "value": float(df["error_home"].mean())},
        {"component": "mean error on target", "value": float(df["error_target"].mean())},
        {"component": "mean error under covariate shift only (reweighted ref)", "value": float(df["error_cov_only"].mean())},
        {"component": "covariate_share (error_cov_only - error_home)", "value": float(df["covariate_share"].mean())},
        {"component": "concept_share (error_target - error_cov_only)", "value": float(df["concept_share"].mean())},
    ]
    # bootstrap SE on the shares (sampling noise of the decomposition)
    if "covariate_share_se" in df.columns:
        rows += [
            {"component": "covariate_share mean bootstrap SE", "value": float(df["covariate_share_se"].mean())},
            {"component": "concept_share mean bootstrap SE", "value": float(df["concept_share_se"].mean())},
        ]
    # reliability indicators for the reweighting (low ESS => concept_share noisy)
    if "norm_weights_ess_ratio" in df.columns:
        rows += [
            {"component": "norm_weights_ess_ratio (median across pairs)", "value": float(df["norm_weights_ess_ratio"].median())},
            {"component": "norm_weights_max (median across pairs)", "value": float(df["norm_weights_max"].median())},
            {"component": "norm_weights_clip_fraction (median across pairs)", "value": float(df["norm_weights_clip_fraction"].median())},
            {"component": "share of pairs with ESS ratio < 0.2", "value": float((df["norm_weights_ess_ratio"] < 0.2).mean())},
        ]
    # Truncation-bias check: the decomposition at the two alternative caps (10, 50)
    # vs the default (20). If concept_share moves a lot across caps, it's not solid.
    if "concept_share_c10" in df.columns and "concept_share_c50" in df.columns:
        rows += [
            {"component": "concept_share at cap 10 (mean)", "value": float(df["concept_share_c10"].mean())},
            {"component": "concept_share at cap 20 (mean, default)", "value": float(df["concept_share"].mean())},
            {"component": "concept_share at cap 50 (mean)", "value": float(df["concept_share_c50"].mean())},
            {"component": "max drift in concept_share across caps 10/20/50", "value": float(
                np.ptp(df[["concept_share_c10", "concept_share", "concept_share_c50"]].mean()))},
        ]
    # Low-ESS robustness: report the decomposition excluding pairs where the
    # reweighting is unreliable (ESS ratio < 0.2). If concept_share changes a lot
    # when those are dropped, the headline number is driven by unstable weights.
    if "norm_weights_ess_ratio" in df.columns:
        good = df[df["norm_weights_ess_ratio"] >= 0.2]
        if len(good) < len(df) and len(good) > 0:
            rows += [
                {"component": "concept_share excluding low-ESS pairs (ESS>=0.2)", "value": float(good["concept_share"].mean())},
                {"component": "n pairs with ESS >= 0.2", "value": float(len(good))},
            ]
    return pd.DataFrame(rows)


def shift_decomposition_by_source(df):
    """Decomposition reported per source state (one row per source), so we can
    see whether the concept/covariate split is consistent across sources or is
    driven by a few of them."""
    return pd.DataFrame({
        "source": df.groupby("source")["concept_share"].mean().index,
        "concept_share": df.groupby("source")["concept_share"].mean().values,
        "covariate_share": df.groupby("source")["covariate_share"].mean().values,
        "mean_acc_drop": df.groupby("source")["acc_drop"].mean().values,
        "median_ess_ratio": df.groupby("source")["norm_weights_ess_ratio"].median().values,
    })


def per_pair_cap_drift(df):
    """Per-pair drift in concept_share across caps 10/20/50. The pooled mean can
    hide that only a few high-clip pairs are actually sensitive to the cap. This
    reports the rows where the cap matters most, so those pairs can be inspected
    directly (the reviewer asked to show CA->WV, CA->AL explicitly)."""
    import numpy as _np
    cols = [c for c in ("concept_share_c10", "concept_share", "concept_share_c50")
            if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    d = df[["source", "target"] + cols].copy()
    d["cap_drift"] = d[cols].apply(_np.ptp, axis=1)
    d["clip_fraction"] = df["norm_weights_clip_fraction"]
    d["ess_ratio"] = df["norm_weights_ess_ratio"]
    return d.sort_values("cap_drift", ascending=False).reset_index(drop=True)


def alert_eval(df, col, threshold, harmful_drop=0.03):
    """If we alert when score >= threshold, how often is the alert wrong or missing?"""
    alert, harmful = df[col] >= threshold, df["acc_drop"] >= harmful_drop
    tp, fp = int((alert & harmful).sum()), int((alert & ~harmful).sum())
    fn, tn = int((~alert & harmful).sum()), int((~alert & ~harmful).sum())
    return {"score": col, "threshold": threshold, "harmful_if_drop>=": harmful_drop,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "false_alarm_rate": fp / max(fp + tn, 1), "miss_rate": fn / max(fn + tp, 1)}


def threshold_sweep(df, col, harmful_drop=0.03, thresholds=None):
    """Audit a drift score across many alert thresholds. For each threshold, treat
    `score >= threshold` as an alert and evaluate it against harmful pairs
    (acc_drop >= harmful_drop). Returns a table with tp/fp/fn/tn, precision,
    recall, miss rate, and false-alarm rate. This replaces the 2-threshold check
    with a full curve, so we can see whether ANY threshold discriminates.
    """
    if thresholds is None:
        thresholds = np.linspace(df[col].min(), df[col].max(), 40)
    harmful = (df["acc_drop"] >= harmful_drop).astype(bool)
    rows = []
    for t in thresholds:
        alert = df[col].astype(float) >= t
        tp = int((alert & harmful).sum()); fp = int((alert & ~harmful).sum())
        fn = int((~alert & harmful).sum()); tn = int((~alert & ~harmful).sum())
        rows.append({
            "score": col, "threshold": round(float(t), 4), "harmful_if_drop>=": harmful_drop,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "false_alarm_rate": fp / max(fp + tn, 1), "miss_rate": fn / max(fn + tp, 1),
            "n_alerts": tp + fp, "n_harmful": int(harmful.sum()),
        })
    return pd.DataFrame(rows)


def alert_sweep_all(df, harmful_drop=0.03):
    """Full threshold sweep for the label-free scores + the model-aware signal."""
    cols = [c for c in DRIFT_COLS if c in df.columns]
    return pd.concat([threshold_sweep(df, c, harmful_drop) for c in cols], ignore_index=True)


def _cluster_auc_ci(df, col, harmful, cluster, n_boot=2000, seed=0):
    """Cluster-bootstrap 95% CI for harmful-AUC, resampling the cluster column
    (source) so pairs sharing a source model are kept together."""
    rng = np.random.default_rng(seed)
    clusters = df[cluster].unique()
    aucs = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        mask = np.zeros(len(df), dtype=bool)
        for c in chosen:
            mask |= (df[cluster].values == c)
        s = df.iloc[mask]
        if s[col].nunique() < 2 or harmful.loc[mask].nunique() < 2:
            continue
        aucs.append(roc_auc_score(harmful.loc[mask], s[col]))
    if len(aucs) < 2:
        return np.nan, np.nan
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def harmful_auc(df, harmful_drop=0.03):
    harmful = (df["acc_drop"] >= harmful_drop).astype(int)
    if harmful.nunique() < 2:
        return pd.DataFrame()
    rows = []
    for c in DRIFT_COLS:
        if c not in df.columns:
            continue
        lo, hi = _cluster_auc_ci(df, c, harmful, cluster="source")
        rows.append({"score": c, "auc_for_harmful": roc_auc_score(harmful, df[c]),
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def make_plots(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    cols = ["mean_psi", "max_psi", "domain_auc", "label_shift"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, c in zip(axes, cols):
        ax.scatter(df[c], df["acc_drop"], s=18, alpha=0.7)
        ax.set_xlabel(c)
        ax.set_ylabel("accuracy drop")
        ax.set_title(f"Spearman = {spearmanr(df[c], df['acc_drop'])[0]:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drift_vs_accuracy_drop.png"), dpi=150)
    plt.close(fig)


def run_analysis(csv="outputs/results.csv", outdir="outputs", harmful_drop=0.03):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv)
    tables = {}
    # Compute and persist each table independently so a slow component never
    # blocks the others (and partial results survive an interruption).
    def _store(name, t):
        tables[name] = t
        t.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
        print(f"\n== {name} ==\n{t.round(3).to_string(index=False)}", flush=True)

    _store("correlations", correlations(df))
    _store("predictor_importance", predictor_importance(df))
    _store("shift_decomposition", shift_decomposition(df))
    _store("shift_decomposition_by_source", shift_decomposition_by_source(df))
    _store("per_pair_cap_drift", per_pair_cap_drift(df))
    _store("alerts", pd.DataFrame([alert_eval(df, "mean_psi", 0.1, harmful_drop),
                                    alert_eval(df, "mean_psi_numeric", 0.1, harmful_drop),
                                    alert_eval(df, "max_psi", 0.1, harmful_drop),
                                    alert_eval(df, "max_psi", 0.25, harmful_drop)]))
    _store("threshold_sweep", alert_sweep_all(df, harmful_drop))
    _store("harmful_auc", harmful_auc(df, harmful_drop))
    make_plots(df, outdir)
    return tables
