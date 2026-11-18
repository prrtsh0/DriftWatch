"""Run the full 15-source HGB benchmark with an explicit, reproducible config."""
import json
import subprocess
import sys
import time

import numpy as np

ALL_15_SOURCES = ["CA", "TX", "NY", "FL", "WA", "IL", "OH", "MI", "MN",
                  "MS", "AL", "WV", "ME", "UT", "MA"]

# Reproducibility: pin the global RNG *and* all local seeds are fixed in the
# code (StratifiedKFold(random_state=seed), default_rng(seed)). This global seed
# is defense-in-depth so any unseeded global RNG draw is also deterministic.
np.random.seed(0)

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".").decode().strip()
t0 = time.time()
rc = subprocess.call(
    [sys.executable, "run_experiment.py",
     "--sources", *ALL_15_SOURCES,
     "--targets", *ALL_15_SOURCES,
     "--model", "hgb",
     "--year", "2018",
     "--n-max", "30000",
     "--out", "outputs/results.csv"],
)
elapsed = round(time.time() - t0, 1)
meta = {
    "run": "full_hgb_15sources",
    "commit": commit,
    "model": "hgb",
    "sources": ALL_15_SOURCES,
    "targets": ALL_15_SOURCES,
    "year": "2018", "n_max": 30000,
    "synthetic": False,
    "elapsed_sec": elapsed,
    "exit_code": rc,
}
with open("outputs/run_meta_full_hgb.json", "w") as f:
    json.dump(meta, f, indent=2)
print("RUN_META", json.dumps(meta))
sys.exit(rc)
