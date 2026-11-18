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


def _density_ratio(ref, cur, n=2000, seed=0):
    """Estimate w(x) = P_cur(x) / P_ref(x) with a domain classifier (logistic on
    primary features). w = p(origin=cur) / p(origin=ref). Used for reweighting.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    r = ref.sample(min(n, len(ref)), random_state=seed).copy()
    c = cur.sample(min(n, len(cur)), random_state=seed).copy()
    X = pd.concat([r[FEATURES], c[FEATURES]])
    y = np.r_[np.zeros(len(r)), np.ones(len(c))]
    sc = StandardScaler().fit(X[FEATURES])
    Xs = sc.transform(X[FEATURES])
    clf = LogisticRegression(max_iter=1000).fit(Xs, y)
    p = clf.predict_proba(sc.transform(ref[FEATURES]))[:, 1]
    w = p / np.clip(1 - p, 1e-6, None)   # odds ratio ≈ P_cur/P_ref
    return np.clip(w / w.mean(), 1e-3, 1e3)


def decompose_error_shifts(model, ref, y_ref, cur, y_cur):
    """Decompose the accuracy/error *increase* into a covariate-shift part and a
    concept-shift part using importance reweighting (a standard technique for
    separating these under covariate shift).

    - error_home      = error of the model on the reference (holdout) sample.
    - error_target     = error on the new (target) data.
    - error_cov_only   = error on the reference, reweighted so the covariate
                         distribution matches the target. This is the error we
                         *would* expect if only P(x) changed and P(y|x) were fixed.
    - covariate_share  = error_cov_only - error_home   (part explained by P(x) change)
    - concept_share    = error_target - error_cov_only (residual = P(y|x) change
                         + interaction/approximation error)
    - norm_weights     = a summary of how reasonable the importance weights were
                         (max weight, and effective sample size ratio ESS/n).
    NOTE: concept_share is an upper-bound-ish estimate; it also captures any
    departure from the importance-weighting assumptions (non-overlap, estimation
    error), so it should be read as 'the part of the drop not explained by
    covariate shift alone', not a clean causal estimate. Report `norm_weights`
    alongside it: a very low effective-sample-size ratio (e.g. < 0.2) means the
    weights are unstable and concept_share is dominated by reweighting noise.
    """
    pred_ref = model.predict(ref[FEATURES]) >= 0.5
    pred_cur = model.predict(cur[FEATURES]) >= 0.5
    err_home = float((pred_ref != np.asarray(y_ref)).mean())
    err_target = float((pred_cur != np.asarray(y_cur)).mean())
    w = _density_ratio(ref, cur)
    err_cov_only = float(np.average(pred_ref != np.asarray(y_ref), weights=w))
    ess = (w.sum() ** 2) / (w ** 2).sum()
    return {
        "error_home": err_home, "error_target": err_target,
        "error_cov_only": err_cov_only,
        "covariate_share": err_cov_only - err_home,
        "concept_share": err_target - err_cov_only,
        "norm_weights_max": float(w.max()),
        "norm_weights_ess_ratio": float(ess / len(w)),
    }
