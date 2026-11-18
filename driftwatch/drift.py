"""Label-free drift scores: compare a reference sample with new data. No true labels needed."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from .data import CATEGORICAL, FEATURES, NUMERIC, task_spec

EPS = 1e-4  # avoids log(0) when a bucket is empty


def _psi(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p / p.sum(), EPS, None)   # share of reference rows per bucket
    q = np.clip(q / q.sum(), EPS, None)   # share of new rows per bucket
    return float(np.sum((q - p) * np.log(q / p)))


def psi_feature(ref, cur, categorical: bool = False, bins: int = 10, top_k: int = 20) -> float:
    """Population Stability Index for one column. 0 = identical; bigger = more different."""
    ref = pd.Series(ref).reset_index(drop=True)
    cur = pd.Series(cur).reset_index(drop=True)
    edges = None if categorical else np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if categorical or len(edges) < 3:              # category frequencies (rare ones grouped as "other")
        top = ref.value_counts().index[:top_k]
        r, c = ref.where(ref.isin(top), -999), cur.where(cur.isin(top), -999)
        cats = sorted(set(r.unique()) | set(c.unique()))
        p = r.value_counts().reindex(cats, fill_value=0).values.astype(float)
        q = c.value_counts().reindex(cats, fill_value=0).values.astype(float)
    else:                                          # numeric: buckets from reference quantiles
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0].astype(float)
        q = np.histogram(cur, edges)[0].astype(float)
    return _psi(p, q)


def psi_table(ref: pd.DataFrame, cur: pd.DataFrame, task: str = "income") -> pd.Series:
    spec = task_spec(task)
    return pd.Series({f: psi_feature(ref[f], cur[f], categorical=f in spec["categorical"])
                      for f in spec["features"]})


def ks_table(ref: pd.DataFrame, cur: pd.DataFrame, task: str = "income") -> pd.Series:
    """KS statistic = biggest gap between the two cumulative curves (numeric columns only)."""
    spec = task_spec(task)
    return pd.Series({f: ks_2samp(ref[f], cur[f]).statistic for f in spec["numeric"]})


def domain_auc(ref: pd.DataFrame, cur: pd.DataFrame, n: int = 4000, seed: int = 0,
               task: str = "income") -> float:
    """Train a model to tell old rows from new rows. AUC 0.5 = cannot tell; 1.0 = easily told."""
    spec = task_spec(task)
    r = ref.sample(min(n, len(ref)), random_state=seed)
    c = cur.sample(min(n, len(cur)), random_state=seed)
    X = pd.concat([r, c])[spec["features"]]
    y = np.r_[np.zeros(len(r)), np.ones(len(c))]
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean())


def drift_summary(ref: pd.DataFrame, cur: pd.DataFrame, with_domain: bool = True,
                  task: str = "income") -> dict:
    spec = task_spec(task)
    psi, ks = psi_table(ref, cur, task=task), ks_table(ref, cur, task=task)
    out = {"mean_psi": psi.mean(), "max_psi": psi.max(), "mean_ks": ks.mean(),
           "mean_psi_numeric": psi[spec["numeric"]].mean()}   # numeric only (no high-cardinality codes)
    out["domain_auc"] = domain_auc(ref, cur, task=task) if with_domain else np.nan
    return out
