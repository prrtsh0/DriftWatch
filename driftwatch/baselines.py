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


def _density_ratio(ref, cur, n=2000, seed=0, max_mult=None, cv=5):
    """Estimate w(x) = P_cur(x) / P_ref(x) with a domain classifier (logistic on
    primary features). w = p(origin=cur) / p(origin=ref). Used for reweighting.

    The classifier is **cross-fitted**: we k-fold the reference rows and, for
    each fold, train on (reference rows outside the fold) + target rows, then
    predict on the held-out reference fold. This yields out-of-fold probabilities
    for EVERY reference row without the classifier having seen that row, so the
    weights do not reflect in-sample overfitting.

    Weights are normalized to mean 1 and truncated to [1/max_mult, max_mult]
    times the mean.

    Returns dict keyed by cap -> (w, clip_fraction) where w has length = len(ref).
    When max_mult is None it reports caps 10, 20, 50 so the reader can see how
    sensitive the decomposition is to truncation (bias-variance tradeoff).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    r = ref.sample(min(n, len(ref)), random_state=seed).copy()   # reference rows
    c = cur.sample(min(n, len(cur)), random_state=seed).copy()   # target rows
    sc = StandardScaler().fit(r[FEATURES])
    p = np.zeros(len(r))
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(r, r["SEX"] if "SEX" in r else np.zeros(len(r))):
        tr = r.iloc[tr_idx]
        va = r.iloc[va_idx]
        Xtr = pd.concat([tr[FEATURES], c[FEATURES]])
        ytr = np.r_[np.zeros(len(tr)), np.ones(len(c))]
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        p[va_idx] = clf.predict_proba(sc.transform(va[FEATURES]))[:, 1]

    raw = p / np.clip(1 - p, 1e-6, None)   # odds ratio ≈ P_cur/P_ref
    raw = raw / raw.mean()                  # normalize so mean = 1

    caps = (10.0, 20.0, 50.0) if max_mult is None else (float(max_mult),)
    out = {}
    for cap in caps:
        lo, hi = 1.0 / cap, cap
        clipped = np.clip(raw, lo, hi)
        clip_fraction = float((clipped != raw).mean())
        out[cap] = (clipped, clip_fraction)
    return out


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
    # Subsample ref/cur consistently so weights and errors are computed on the
    # SAME rows (the density-ratio estimator internally subsamples to n).
    n = 2000
    rng = np.random.default_rng(0)
    if len(ref) > n:
        idx = rng.choice(len(ref), n, replace=False)
        ref = ref.iloc[idx].copy()
        y_ref = np.asarray(y_ref)[idx]
    if len(cur) > n:
        idx = rng.choice(len(cur), n, replace=False)
        cur = cur.iloc[idx].copy()
        y_cur = np.asarray(y_cur)[idx]

    pred_ref = model.predict(ref[FEATURES]) >= 0.5
    pred_cur = model.predict(cur[FEATURES]) >= 0.5
    err_home = float((pred_ref != np.asarray(y_ref)).mean())
    err_target = float((pred_cur != np.asarray(y_cur)).mean())

    # Decompose at several truncation caps to expose truncation bias: if the
    # covariate/concept split changes a lot across caps, the number is not solid.
    ratio = _density_ratio(ref, cur)
    out = {
        "error_home": err_home, "error_target": err_target,
    }
    # primary (default) cap = 20
    for cap, (w, clip_fraction) in ratio.items():
        err_cov_only = float(np.average(pred_ref != np.asarray(y_ref), weights=w))
        ess = (w.sum() ** 2) / (w ** 2).sum()
        tag = "" if cap == 20.0 else f"_c{int(cap)}"
        out[f"covariate_share{tag}"] = err_cov_only - err_home
        out[f"concept_share{tag}"] = err_target - err_cov_only
        out[f"error_cov_only{tag}"] = err_cov_only
        out[f"norm_weights_max{tag}"] = float(w.max())
        out[f"norm_weights_ess_ratio{tag}"] = float(ess / len(w))
        out[f"norm_weights_clip_fraction{tag}"] = clip_fraction
    return out
