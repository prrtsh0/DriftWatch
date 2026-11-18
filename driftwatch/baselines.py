"""Accuracy-estimation baselines for the drift study.

These estimate the *label-free* performance of the model on new data, so they
can be compared with the drift scores. We include:

- pred_shift  : |mean predicted prob on target - mean on ref| (simplest baseline)
- ATC         : Average Threshold Confidence (Garg et al., 2022) — uses the
                 confidence of the most-confident examples as a score shift signal
- DoC         : Difference of Confidences (Guillory et al., 2021) — mean of the
                 top-k model confidences
- MMD         : Maximum Mean Discrepancy on the numeric features (a classical
                 two-sample test / kernel distance between ref and target)

Truth: the reference sample and the target batch, plus the trained model.
"""
import numpy as np
import pandas as pd

from .data import NUMERIC, FEATURES


def pred_shift(model, ref: pd.DataFrame, cur: pd.DataFrame) -> float:
    """Mean predicted-probability shift between reference and target."""
    ref_p = model.predict_proba(ref[FEATURES][:2000])[:, 1].mean()
    cur_p = model.predict_proba(cur[FEATURES])[:, 1].mean()
    return float(abs(ref_p - cur_p))


def _prob(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def atc(model, ref: pd.DataFrame, cur: pd.DataFrame, k: int = 100) -> float:
    """Average Threshold Confidence: mean confidence of the k most-confident target
    examples, compared with the same quantity on the reference. A drop in this
    quantity indicates the model is less confident on new data -> likely harm.
    """
    ref_conf = np.sort(_prob(model, ref))[-k:]
    cur_conf = np.sort(_prob(model, cur))[-k:]
    return float(abs(ref_conf.mean() - cur_conf.mean()))


def doc(model, ref: pd.DataFrame, cur: pd.DataFrame, k: int = 100) -> float:
    """Difference of Confidences (top-k mean confidence)."""
    ref_top = _prob(model, ref)
    cur_top = _prob(model, cur)
    return float(abs(np.sort(ref_top)[-k:].mean() - np.sort(cur_top)[-k:].mean()))


def mmd(ref: pd.DataFrame, cur: pd.DataFrame, gamma: float = 1.0) -> float:
    """Gaussian RBF Maximum Mean Discrepancy on the numeric features.

    A simple, classical two-sample test that is label-free. Higher = distributions
    differ more. Uses a heuristic median-based bandwidth if gamma is None.
    """
    r = ref[NUMERIC].values.astype(float)
    c = cur[NUMERIC].values.astype(float)
    n, m = len(r), len(c)
    # deterministic strided subsample (identical input -> identical subsample)
    if n > 1500:
        idx = np.linspace(0, n - 1, 1500).astype(int)
        r = r[idx]
    if m > 1500:
        idx = np.linspace(0, m - 1, 1500).astype(int)
        c = c[idx]
    n, m = len(r), len(c)

    def _sq_dist(a, b):
        aa = (a ** 2).sum(axis=1, keepdims=True)
        bb = (b ** 2).sum(axis=1, keepdims=True)
        return aa + bb.T - 2 * a @ b.T

    if gamma is None:
        # median pairwise distance heuristic
        sample = np.vstack([r[:200], c[:200]])
        d = _sq_dist(sample, sample)
        gamma = 1.0 / (np.median(d[d > 0]) + 1e-9)

    kxx = np.exp(-gamma * _sq_dist(r, r))
    kyy = np.exp(-gamma * _sq_dist(c, c))
    kxy = np.exp(-gamma * _sq_dist(r, c))
    return float(
        kxx.sum() / (n * n) + kyy.sum() / (m * m) - 2 * kxy.sum() / (n * m)
    )


def compute_baselines(model, ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """Return all baseline signals as a dict for one target pair."""
    return {
        "pred_shift": pred_shift(model, ref, cur),
        "atc": atc(model, ref, cur),
        "doc": doc(model, ref, cur),
        "mmd": mmd(ref, cur),
    }
