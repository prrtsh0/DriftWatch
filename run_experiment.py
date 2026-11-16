"""Run the benchmark. Real data: python run_experiment.py   Test without internet: add --synthetic"""
import argparse
from driftwatch.experiment import run_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--sources", nargs="+", default=["CA", "TX", "NY"])
ap.add_argument("--targets", nargs="+", default=["CA", "TX", "NY", "FL", "WA", "IL", "OH", "MI", "MN", "MS", "AL", "WV", "ME", "UT", "MA"])
ap.add_argument("--model", default="hgb", choices=["hgb", "logreg"])
ap.add_argument("--year", default="2018"); ap.add_argument("--n-max", type=int, default=30000)
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--out", default="outputs/results.csv")
a = ap.parse_args()
run_experiment(a.sources, a.targets, a.model, a.year, a.n_max, a.synthetic, a.out)
