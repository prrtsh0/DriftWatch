"""The benchmark: train on a source state, test on many target states, record drift and accuracy drop."""
import os

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .data import load_state
from .drift import drift_summary
from .model import build_model


def run_experiment(sources, targets, model_kind="hgb", year="2018", n_max=30000,
                   synthetic=False, out_csv="outputs/results.csv"):
    cache = {}

    def get(state):
        if state not in cache:
            print(f"  loading {state} ...")
            cache[state] = load_state(state, year=year, n_max=n_max, synthetic=synthetic)
        return cache[state]

    rows = []
    for src in sources:
        X, y = get(src)
        X_tr, X_ref, y_tr, y_ref = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        model = build_model(model_kind).fit(X_tr, y_tr)
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
                "source": src, "target": tgt, "model": model_kind,
                "ref_acc": ref_acc, "target_acc": acc, "acc_drop": ref_acc - acc,
                "label_shift": abs(yt.mean() - y_tr.mean()),      # uses labels: explains, not a drift score
                "pred_shift": abs(prob.mean() - ref_prob.mean()), # label-free, model-aware
                **drift_summary(X_ref, Xt),                       # label-free drift scores
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"saved {len(df)} rows to {out_csv}")
    return df
