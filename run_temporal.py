"""Run the temporal benchmark (train 2018, test 2019, same states) at current commit."""
import json
import subprocess
import sys
import time

ALL_STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "OH", "MI", "MN",
              "MS", "AL", "WV", "ME", "UT", "MA"]

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".").decode().strip()
t0 = time.time()
rc = subprocess.call(
    [sys.executable, "run_experiment.py",
     "--sources", *ALL_STATES, "--model", "hgb",
     "--year", "2018", "--year2", "2019", "--n-max", "30000",
     "--temporal", "--out", "outputs/results_temporal.csv"])
elapsed = round(time.time() - t0, 1)
meta = {"run": "temporal_hgb_2018to2019", "commit": commit, "model": "hgb",
        "sources": ALL_STATES, "year": "2018", "year2": "2019",
        "n_max": 30000, "synthetic": False, "elapsed_sec": elapsed, "exit_code": rc}
with open("outputs/run_meta_full_temporal.json", "w") as f:
    json.dump(meta, f, indent=2)
print("RUN_META", json.dumps(meta))
sys.exit(rc)
