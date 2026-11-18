"""Run the benchmark. Real data: python run_experiment.py   Test without internet: add --synthetic"""
import argparse
import numpy as np
from driftwatch.experiment import run_experiment, run_temporal_experiment

# Reproducibility: pin the global RNG. The stochastic internals also use fixed
# local seeds (random_state=seed / default_rng(seed)), so this is defensive.
np.random.seed(0)

ap = argparse.ArgumentParser()
ALL_STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "OH", "MI", "MN", "MS", "AL", "WV", "ME", "UT", "MA"]
ap.add_argument("--sources", nargs="+", default=ALL_STATES)
ap.add_argument("--targets", nargs="+", default=ALL_STATES)
ap.add_argument("--model", default="hgb", choices=["hgb", "logreg"])
ap.add_argument("--task", default="income", choices=["income", "employment"])
ap.add_argument("--year", default="2018")
ap.add_argument("--year2", default="2019", help="for --temporal: test year")
ap.add_argument("--n-max", type=int, default=30000)
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--temporal", action="store_true", help="train on year, test on year2 (same states)")
ap.add_argument("--out", default=None,
                help="output CSV. Default is safe per run type and never overwrites "
                     "another run's file: results.csv (HGB), results_logreg.csv "
                     "(logreg), results_temporal.csv (temporal).")
a = ap.parse_args()
# Safe per-mode defaults so runs cannot silently overwrite each other.
if a.out is None:
    task_suffix = f"_{a.task}" if a.task != "income" else ""
    if a.temporal:
        a.out = f"outputs/results_temporal{task_suffix}.csv"
    elif a.model == "logreg":
        a.out = f"outputs/results_logreg{task_suffix}.csv"
    else:
        a.out = f"outputs/results{task_suffix}.csv"
if a.temporal:
    run_temporal_experiment(a.sources, a.model, a.year, a.year2, a.n_max,
                            a.synthetic, a.out, task=a.task)
else:
    run_experiment(a.sources, a.targets, a.model, a.year, a.n_max, a.synthetic, a.out,
                   task=a.task)
