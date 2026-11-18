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

from .data import NUMERIC, FEATURES, task_spec


def pred_shift(model, ref: pd.DataFrame, cur: pd.DataFrame, task: str = "income") -> float:
    """Mean predicted-probability shift between reference and target."""
    spec = task_spec(task)
    f = spec["features"]
    ref_p = model.predict_proba(ref[f][:2000])[:, 1].mean()
    cur_p = model.predict_proba(cur[f])[:, 1].mean()
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


def mmd(ref: pd.DataFrame, cur: pd.DataFrame, gamma: float = 1.0,
        task: str = "income") -> float:
    """Gaussian RBF Maximum Mean Discrepancy on the numeric features.

    A simple, classical two-sample test that is label-free. Higher = distributions
    differ more. Uses a heuristic median-based bandwidth if gamma is None.
    """
    spec = task_spec(task)
    r = ref[spec["numeric"]].values.astype(float)
    c = cur[spec["numeric"]].values.astype(float)
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


def compute_baselines(model, ref: pd.DataFrame, cur: pd.DataFrame,
                      task: str = "income") -> dict:
    """Return all baseline signals as a dict for one target pair."""
    return {
        "pred_shift": pred_shift(model, ref, cur, task=task),
        "atc": atc(model, ref, cur),
        "doc": doc(model, ref, cur),
        "mmd": mmd(ref, cur, task=task),
    }


def _density_ratio(ref, cur, max_mult=None, cv=5, seed=0, task: str = "income"):
    """Estimate w(x) = P_cur(x) / P_ref(x) with a domain classifier (logistic on
    primary features). w = p(origin=cur) / p(origin=ref). Used for reweighting.

    The classifier is **cross-fitted over the FULL reference and target**: we
    k-fold the concatenated (ref+target) rows and predict out-of-fold
    P(origin=cur|x) for every row, so no row is scored by a model trained on it.
    This yields a weight for EVERY reference row, so the decomposition can be
    computed on the same full data as acc_drop and will reconcile with it.

    Weights are normalized to mean 1 and truncated to [1/max_mult, max_mult]
    times the mean.

    Returns dict keyed by cap -> (w, clip_fraction) where w has length = len(ref).
    When max_mult is None it reports caps 10, 20, 50 so the reader can see how
    sensitive the decomposition is to truncation (bias-variance tradeoff).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    spec = task_spec(task)
    X = pd.concat([ref[spec["features"]], cur[spec["features"]]])
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=1000)
    # out-of-fold P(origin=cur | x) for every concatenated row
    p_all = cross_val_predict(
        clf, Xs, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        method="predict_proba")[:, 1]
    p = p_all[: len(ref)]                     # reference rows only
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


def _share_bootstrap_se(pred_ref, y_ref, pred_cur, y_cur, w, n_boot=200, seed=0):
    """Bootstrap standard error for covariate_share and concept_share.

    Resamples the (prediction, label, weight) rows of the reference and target
    subsamples (with replacement), recomputes the two shares each time, and
    returns their standard deviations. This quantifies the sampling noise in the
    decomposition (the reviewer asked for a CI/SE; SE is cheaper and adequate).
    """
    rng = np.random.default_rng(seed)
    n_ref, n_cur = len(w), len(pred_cur)
    cov_shares, concept_shares = [], []
    for _ in range(n_boot):
        ir = rng.integers(0, n_ref, n_ref)
        ic = rng.integers(0, n_cur, n_cur)
        err_home = float((pred_ref[ir] != y_ref[ir]).mean())
        err_target = float((pred_cur[ic] != y_cur[ic]).mean())
        err_cov_only = float(np.average(pred_ref[ir] != y_ref[ir], weights=w[ir]))
        cov_shares.append(err_cov_only - err_home)
        concept_shares.append(err_target - err_cov_only)
    return float(np.std(cov_shares)), float(np.std(concept_shares))


def decompose_error_shifts(model, ref, y_ref, cur, y_cur, task: str = "income"):
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
    # Everything computed on the FULL data: the errors use all rows (so the
    # decomposition reconciles exactly with acc_drop in experiment.py), and the
    # density-ratio weights are cross-fitted over all rows too.
    spec = task_spec(task)
    pred_ref = model.predict(ref[spec["features"]]) >= 0.5
    pred_cur = model.predict(cur[spec["features"]]) >= 0.5
    err_home = float((pred_ref != np.asarray(y_ref)).mean())
    err_target = float((pred_cur != np.asarray(y_cur)).mean())

    # Decompose at several truncation caps to expose truncation bias: if the
    # covariate/concept split changes a lot across caps, the number is not solid.
    ratio = _density_ratio(ref, cur, task=task)
    out = {
        "error_home": err_home, "error_target": err_target,
    }
    # primary (default) cap = 20
    for cap, (w, clip_fraction) in ratio.items():
        err_cov_only = float(np.average(pred_ref != np.asarray(y_ref), weights=w))
        ess = (w.sum() ** 2) / (w ** 2).sum()
        tag = "" if cap == 20.0 else f"_c{int(cap)}"
        cov_share = err_cov_only - err_home
        concept_share = err_target - err_cov_only
        out[f"covariate_share{tag}"] = cov_share
        out[f"concept_share{tag}"] = concept_share
        out[f"error_cov_only{tag}"] = err_cov_only
        out[f"norm_weights_max{tag}"] = float(w.max())
        out[f"norm_weights_ess_ratio{tag}"] = float(ess / len(w))
        out[f"norm_weights_clip_fraction{tag}"] = clip_fraction
        if tag == "":  # bootstrap SE on the headline shares (cap 20)
            se = _share_bootstrap_se(pred_ref, np.asarray(y_ref), pred_cur,
                                     np.asarray(y_cur), w, n_boot=200, seed=0)
            out["covariate_share_se"] = se[0]
            out["concept_share_se"] = se[1]
    return out
