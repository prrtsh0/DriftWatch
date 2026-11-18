"""Simulate a sequence of drift checks so the dashboard has history to show.

Runs the deployed monitor against a few synthetic target-state batches (with
increasing shift), appends each result to the monitor history, and prints a
short summary. After running, open the dashboard at http://127.0.0.1:8000/dashboard

Usage:  python demo_simulate_drift.py
"""
import sys

import joblib
import pandas as pd

from driftwatch.data import load_state
from driftwatch.monitor import DriftMonitor
from api.dashboard import append_history, load_history


def main():
    reference = pd.read_csv("artifacts/reference.csv")
    monitor = DriftMonitor(reference)

    # "New data" batches with increasing shift vs the CA reference: WV (mild),
    # MS (moderate), then a strongly shifted batch (shifted age/schooling).
    states = ["WV", "MS"]
    batches = []
    for st in states:
        X, _ = load_state(st, n_max=6000, synthetic=True)
        batches.append((st, X.head(3000)))

    # Strongly perturbed batch (worst case).
    X, _ = load_state("AL", n_max=6000, synthetic=True)
    strong = X.head(3000).copy()
    strong["AGEP"] = strong["AGEP"] + 30
    strong["SCHL"] = strong["SCHL"] + 8
    batches.append(("AL+shift", strong))

    for name, batch in batches:
        result = monitor.check(batch)
        result["batch"] = name
        append_history(result)
        print(f"[{name:9s}] status={result['status']:5s} max_psi={result['max_psi']:.3f} "
              f"mean_psi={result['mean_psi']:.3f} domain_auc={result.get('domain_auc')}")

    hist = load_history()
    print(f"\nAppended {len(hist)} history record(s) to artifacts/monitor_history.jsonl")
    print("Start the API (python -m uvicorn api.app:app --reload) then open "
          "http://127.0.0.1:8000/dashboard")


if __name__ == "__main__":
    sys.exit(main())
