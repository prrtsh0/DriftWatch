"""Run the temporal benchmark (train 2018, test 2019, same states) at current commit."""
import argparse
import json
import subprocess
import sys
import time

ALL_STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "OH", "MI", "MN",
              "MS", "AL", "WV", "ME", "UT", "MA"]

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="income", choices=["income", "employment"])
a = ap.parse_args()

task_suffix = f"_{a.task}" if a.task != "income" else ""
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".").decode().strip()
t0 = time.time()
rc = subprocess.call(
    [sys.executable, "run_experiment.py",
     "--sources", *ALL_STATES, "--model", "hgb",
     "--year", "2018", "--year2", "2019", "--n-max", "30000",
     "--temporal", "--task", a.task, "--out", f"outputs/results_temporal{task_suffix}.csv"])
elapsed = round(time.time() - t0, 1)
meta = {"run": f"temporal_hgb_2018to2019_{a.task}", "commit": commit, "model": "hgb",
        "task": a.task, "sources": ALL_STATES, "year": "2018", "year2": "2019",
        "n_max": 30000, "synthetic": False, "elapsed_sec": elapsed, "exit_code": rc}
with open(f"outputs/run_meta_full_temporal{task_suffix}.json", "w") as f:
    json.dump(meta, f, indent=2)
print("RUN_META", json.dumps(meta))
sys.exit(rc)
