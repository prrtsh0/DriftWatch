import argparse
from driftwatch.analysis import run_analysis
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="outputs/results.csv"); ap.add_argument("--outdir", default="outputs")
ap.add_argument("--harmful-drop", type=float, default=0.03)
a = ap.parse_args()
run_analysis(a.csv, a.outdir, a.harmful_drop)
