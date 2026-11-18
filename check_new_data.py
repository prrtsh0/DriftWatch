"""Check NEW data for drift against the reference sample of the deployed model.

Run train_service_model.py once first (it saves the model and the reference sample).

Usage:
  python check_new_data.py my_new_rows.csv     # a CSV with the 10 feature columns
  python check_new_data.py --state NY          # use another state's census data as "new data"
"""
import argparse
import json
import sys

import joblib
import pandas as pd

from driftwatch.data import FEATURES, load_state
from driftwatch.monitor import DriftMonitor

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", help="CSV file with new rows")
ap.add_argument("--state", help="two-letter US state, e.g. NY")
ap.add_argument("--year", default="2018")
ap.add_argument("--n-max", type=int, default=5000)
ap.add_argument("--synthetic", action="store_true", help="test mode, no download")
a = ap.parse_args()

try:
    reference = pd.read_csv("artifacts/reference.csv")
    model = joblib.load("artifacts/model.joblib")
except FileNotFoundError:
    sys.exit("Run first:  python train_service_model.py   (it creates the artifacts folder)")

if a.csv:
    new = pd.read_csv(a.csv)
elif a.state:
    new, _ = load_state(a.state, a.year, a.n_max, synthetic=a.synthetic)
else:
    ap.error("give a CSV file or --state")

missing = [f for f in FEATURES if f not in new.columns]
if missing:
    sys.exit(f"Your data is missing these columns: {missing}\nNeeded: {FEATURES}")
new = new[FEATURES]

report = DriftMonitor(reference).check(new)
report["share_predicted_over_50k"] = float((model.predict_proba(new)[:, 1] >= 0.5).mean())
try:
    import json as _json
    with open("artifacts/metrics.json") as _fh:
        _ref_prob_mean = _json.load(_fh).get("ref_prob_mean")
    if _ref_prob_mean is not None:
        report["pred_shift"] = abs(float((model.predict_proba(new)[:, 1].mean()) - _ref_prob_mean))
except Exception:
    report["pred_shift"] = None
print(json.dumps(report, indent=2))
print("\nNote: this shows how much the inputs CHANGED. It cannot show accuracy, because new data has no true answers yet.")
