"""Run the benchmark. Real data: python run_experiment.py   Test without internet: add --synthetic"""
import argparse
from driftwatch.experiment import run_experiment, run_temporal_experiment

ap = argparse.ArgumentParser()
ALL_STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "OH", "MI", "MN", "MS", "AL", "WV", "ME", "UT", "MA"]
ap.add_argument("--sources", nargs="+", default=ALL_STATES)
ap.add_argument("--targets", nargs="+", default=ALL_STATES)
ap.add_argument("--model", default="hgb", choices=["hgb", "logreg"])
ap.add_argument("--year", default="2018")
ap.add_argument("--year2", default="2019", help="for --temporal: test year")
ap.add_argument("--n-max", type=int, default=30000)
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--temporal", action="store_true", help="train on year, test on year2 (same states)")
ap.add_argument("--out", default="outputs/results.csv")
a = ap.parse_args()
if a.temporal:
    run_temporal_experiment(a.sources, a.model, a.year, a.year2, a.n_max,
                            a.synthetic, a.out)
else:
    run_experiment(a.sources, a.targets, a.model, a.year, a.n_max, a.synthetic, a.out)
