import numpy as np

from driftwatch.data import load_state, task_spec
from driftwatch.model import build_model
from driftwatch.experiment import run_experiment


def test_employment_task_spec():
    spec = task_spec("employment")
    assert "features" in spec and "numeric" in spec
    assert "AGEP" in spec["features"]
    assert "RELP" in spec["features"]  # the renamed ACS relationship variable
    assert "PINCP" not in spec["features"]  # different domain, different features


def test_employment_synthetic_loads():
    X, y = load_state("CA", n_max=2000, synthetic=True, task="employment")
    spec = task_spec("employment")
    assert X.shape[1] == len(spec["features"])
    assert y.name == "ESR"
    # positive rate should be reasonable (not degenerate all-zeros)
    assert 0.05 < y.mean() < 0.95


def test_employment_model_trains_synthetic():
    X, y = load_state("CA", n_max=2000, synthetic=True, task="employment")
    m = build_model("hgb", task="employment").fit(X, y)
    assert m.predict(X).shape[0] == len(X)


def test_employment_experiment_runs_synthetic(tmp_path):
    out = str(tmp_path / "emp_results.csv")
    df = run_experiment(["CA", "TX"], ["CA", "TX", "NY"], model_kind="hgb",
                        n_max=1500, synthetic=True, out_csv=out, task="employment")
    assert len(df) == 2 * 2  # CA->TX, CA->NY; TX->CA, TX->NY
    assert (df["task"] == "employment").all()
    assert "concept_share" in df.columns and "mmd" in df.columns
    assert df["concept_share"].notna().all()
