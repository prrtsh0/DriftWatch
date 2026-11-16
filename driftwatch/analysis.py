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

DRIFT_COLS = ["mean_psi", "mean_psi_numeric", "max_psi", "mean_ks", "domain_auc", "pred_shift"]


def spearman_ci(df, col, target="acc_drop", n_boot=2000, seed=0):
    """Rank correlation between a drift score and accuracy drop, with a bootstrap 95% interval."""
    rho = spearmanr(df[col], df[target])[0]
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        r = spearmanr(s[col], s[target])[0]
        if not np.isnan(r):
            vals.append(r)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"score": col, "spearman": rho, "ci_low": lo, "ci_high": hi}


def correlations(df):
    return pd.DataFrame([spearman_ci(df, c) for c in DRIFT_COLS])


def decomposition(df):
    """How much of the accuracy drop does each kind of information explain? (leave-one-out R^2)"""
    sets = {"input drift only (mean_psi)": ["mean_psi"],
            "input drift only (mean_psi_numeric)": ["mean_psi_numeric"],
            "input drift only (domain_auc)": ["domain_auc"],
            "label shift only": ["label_shift"],
            "input drift + label shift": ["mean_psi", "label_shift"]}
    rows = []
    for name, cols in sets.items():
        X, y = df[cols].values, df["acc_drop"].values
        pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
        rows.append({"features": name, "loo_r2": r2_score(y, pred)})
    return pd.DataFrame(rows)


def alert_eval(df, col, threshold, harmful_drop=0.03):
    """If we alert when score >= threshold, how often is the alert wrong or missing?"""
    alert, harmful = df[col] >= threshold, df["acc_drop"] >= harmful_drop
    tp, fp = int((alert & harmful).sum()), int((alert & ~harmful).sum())
    fn, tn = int((~alert & harmful).sum()), int((~alert & ~harmful).sum())
    return {"score": col, "threshold": threshold, "harmful_if_drop>=": harmful_drop,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "false_alarm_rate": fp / max(fp + tn, 1), "miss_rate": fn / max(fn + tp, 1)}


def harmful_auc(df, harmful_drop=0.03):
    harmful = (df["acc_drop"] >= harmful_drop).astype(int)
    if harmful.nunique() < 2:
        return pd.DataFrame()
    return pd.DataFrame([{"score": c, "auc_for_harmful": roc_auc_score(harmful, df[c])} for c in DRIFT_COLS])


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
    df = pd.read_csv(csv)
    tables = {"correlations": correlations(df), "decomposition": decomposition(df),
              "alerts": pd.DataFrame([alert_eval(df, "mean_psi", 0.1, harmful_drop),
                                      alert_eval(df, "mean_psi_numeric", 0.1, harmful_drop),
                                      alert_eval(df, "max_psi", 0.1, harmful_drop),
                                      alert_eval(df, "max_psi", 0.25, harmful_drop)]),
              "harmful_auc": harmful_auc(df, harmful_drop)}
    for name, t in tables.items():
        t.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
        print(f"\n== {name} ==\n{t.round(3).to_string(index=False)}")
    make_plots(df, outdir)
    return tables
