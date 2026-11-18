"""Run the full 15-source logreg benchmark at the current commit, recording metadata."""
import subprocess
import sys
import time

ALL_STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "OH", "MI", "MN",
              "MS", "AL", "WV", "ME", "UT", "MA"]

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".").decode().strip()
t0 = time.time()
rc = subprocess.call(
    [sys.executable, "run_experiment.py",
     "--sources", *ALL_STATES, "--targets", *ALL_STATES,
     "--model", "logreg", "--year", "2018", "--n-max", "30000",
     "--out", "outputs/results_logreg.csv"])
elapsed = round(time.time() - t0, 1)
import json
meta = {"run": "full_logreg_15sources", "commit": commit, "model": "logreg",
        "sources": ALL_STATES, "targets": ALL_STATES, "year": "2018",
        "n_max": 30000, "synthetic": False, "elapsed_sec": elapsed, "exit_code": rc}
with open("outputs/run_meta_full_logreg.json", "w") as f:
    json.dump(meta, f, indent=2)
print("RUN_META", json.dumps(meta))
sys.exit(rc)
