"""The benchmark: train on a source state, test on many target states, record drift and accuracy drop."""
import os

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .data import load_state
from .drift import drift_summary
from .model import build_model
from .baselines import compute_baselines, decompose_error_shifts


def run_temporal_experiment(states, model_kind="hgb", year="2018", year2="2019",
                            n_max=30000, synthetic=False, out_csv="outputs/results_temporal.csv",
                            task="income"):
    """Train on a state in {year}, test on the SAME state in {year2} (temporal
    shift — the most common real-world deployment case). No label leakage: the
    reference is the year-year holdout; the target is next year's rows.

    KNOWN CONFOUND: the $50k income cutoff is NOT inflation-adjusted between years
    (Folktables uses a nominal cutoff). This adds a small, systematic label-shift
    component to the temporal results (more people cross $50k nominal over time).
    Treat the temporal accuracy drop as an upper bound on the "real" temporal
    concept shift, and state this in the write-up.
    """
    rows = []
    for state in states:
        if synthetic:
            # synthetic year doesn't exist; reuse but note it's not a true temporal test
            X_ref, y_ref = load_state(state, year="2018", n_max=n_max, synthetic=True, task=task)
            Xt, yt = load_state(state, year="2018", n_max=n_max, synthetic=True,
                                 seed=1, task=task)   # different seed => different draw
        else:
            X_ref, y_ref = load_state(state, year=year, n_max=n_max, task=task)
            Xt, yt = load_state(state, year=year2, n_max=n_max, task=task)
        X_tr, X_val, y_tr, y_val = train_test_split(X_ref, y_ref, test_size=0.3,
                                                    random_state=0, stratify=y_ref)
        model = build_model(model_kind, task=task).fit(X_tr, y_tr)
        ref_prob = model.predict_proba(X_val)[:, 1]
        ref_acc = accuracy_score(y_val, ref_prob >= 0.5)
        prob = model.predict_proba(Xt)[:, 1]
        acc = accuracy_score(yt, prob >= 0.5)
        rows.append({
            "source": state, "target": f"{state}_{year2}", "model": model_kind, "task": task,
            "ref_acc": ref_acc, "target_acc": acc, "acc_drop": ref_acc - acc,
            "label_shift": abs(yt.mean() - y_tr.mean()),
            "pred_shift": abs(prob.mean() - ref_prob.mean()),
            **compute_baselines(model, X_val, Xt, task=task),
            **decompose_error_shifts(model, X_val, y_val, Xt, yt, task=task),
            **drift_summary(X_val, Xt, task=task),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"saved {len(df)} temporal rows to {out_csv}")
    return df


def run_experiment(sources, targets, model_kind="hgb", year="2018", n_max=30000,
                   synthetic=False, out_csv="outputs/results.csv", task="income"):
    cache = {}

    def get(state):
        if state not in cache:
            print(f"  loading {state} ...")
            cache[state] = load_state(state, year=year, n_max=n_max, synthetic=synthetic, task=task)
        return cache[state]

    rows = []
    for src in sources:
        X, y = get(src)
        X_tr, X_ref, y_tr, y_ref = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        model = build_model(model_kind, task=task).fit(X_tr, y_tr)
        ref_prob = model.predict_proba(X_ref)[:, 1]
        ref_acc = accuracy_score(y_ref, ref_prob >= 0.5)          # accuracy at "home"
        print(f"source {src} ({model_kind}): home accuracy {ref_acc:.3f}")
        for tgt in targets:
            if tgt == src:
                continue
            Xt, yt = get(tgt)
            prob = model.predict_proba(Xt)[:, 1]
            acc = accuracy_score(yt, prob >= 0.5)                 # accuracy in the new state
            rows.append({
                "source": src, "target": tgt, "model": model_kind, "task": task,
                "ref_acc": ref_acc, "target_acc": acc, "acc_drop": ref_acc - acc,
                "label_shift": abs(yt.mean() - y_tr.mean()),      # uses labels: explains, not a drift score
                "pred_shift": abs(prob.mean() - ref_prob.mean()), # label-free, model-aware
                **compute_baselines(model, X_ref, Xt, task=task),  # ATC, DoC, MMD, pred_shift
                **decompose_error_shifts(model, X_ref, y_ref, Xt, yt, task=task),  # covariate vs concept share
                **drift_summary(X_ref, Xt, task=task),            # label-free drift scores
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"saved {len(df)} rows to {out_csv}")
    return df
