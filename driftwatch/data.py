"""Data loading.

Real data  : US Census (ACS) tasks via the `folktables` package. One state = one dataset.
Fallback   : a synthetic generator (only for testing the code, NOT for research results).

Tasks (one per frontier/domain):
  - "income"      : ACSIncome — predict income > $50k (10 features).
  - "employment"  : ACSEmployment — predict employment status (16 features).
                     A second, independent domain to test generality.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
NUMERIC = ["AGEP", "SCHL", "WKHP"]                       # age, schooling level, hours/week
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]  # codes where order has no meaning

EMPLOYMENT_FEATURES = ["AGEP", "SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG",
                       "MIL", "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "SEX", "RAC1P"]
EMPLOYMENT_NUMERIC = ["AGEP", "SCHL"]                    # age, schooling level
EMPLOYMENT_CATEGORICAL = [c for c in EMPLOYMENT_FEATURES if c not in EMPLOYMENT_NUMERIC]

TASKS = {
    "income": {
        "features": FEATURES,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "y_name": "PINCP",
        "synthetic": None,  # set below
    },
    "employment": {
        "features": EMPLOYMENT_FEATURES,
        "numeric": EMPLOYMENT_NUMERIC,
        "categorical": EMPLOYMENT_CATEGORICAL,
        "y_name": "ESR",
        "synthetic": None,
    },
}


def task_spec(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task} (available: {list(TASKS)})")
    return TASKS[task]


def _state_seed(state: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(state.upper()))


def make_synthetic_state(state: str, n: int = 20000, seed: int = 0, task: str = "income"):
    """Fake 'state' whose inputs, label rate and input-label link differ a little per state."""
    srng = np.random.default_rng(_state_seed(state))
    age_shift, edu_shift = srng.normal(0, 4), srng.normal(0, 1.5)      # covariate shift
    intercept, coef_shift = srng.normal(0, 0.5), srng.normal(0, 0.15)  # label / concept shift
    rng = np.random.default_rng(seed + _state_seed(state))
    df = pd.DataFrame({
        "AGEP": np.clip(rng.normal(40 + age_shift, 13, n), 17, 90).round(),
        "COW": rng.integers(1, 9, n).astype(float),
        "SCHL": np.clip(rng.normal(18 + edu_shift, 3, n), 1, 24).round(),
        "MAR": rng.integers(1, 6, n).astype(float),
        "OCCP": rng.integers(10, 9830, n).astype(float),
        "POBP": rng.integers(1, 60, n).astype(float),
        "RELP": rng.integers(0, 17, n).astype(float),
        "WKHP": np.clip(rng.normal(38, 12, n), 1, 99).round(),
        "SEX": rng.integers(1, 3, n).astype(float),
        "RAC1P": rng.integers(1, 10, n).astype(float),
        "DIS": rng.integers(1, 3, n).astype(float),
        "ESP": rng.integers(1, 3, n).astype(float),
        "CIT": rng.integers(1, 6, n).astype(float),
        "MIG": rng.integers(1, 4, n).astype(float),
        "MIL": rng.integers(1, 5, n).astype(float),
        "ANC": rng.integers(1, 10, n).astype(float),
        "NATIVITY": rng.integers(1, 3, n).astype(float),
        "DEAR": rng.integers(1, 3, n).astype(float),
        "DEYE": rng.integers(1, 3, n).astype(float),
        "DREM": rng.integers(1, 3, n).astype(float),
    })
    spec = task_spec(task)
    df = df[spec["features"]]
    if task == "income":
        z = (-3.2 + 0.03 * (df.AGEP - 40) + (0.25 + coef_shift) * (df.SCHL - 16)
             + 0.03 * df.WKHP + 0.4 * (df.SEX == 1) + intercept)
        y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-z))).astype(int), name="PINCP")
    else:  # employment
        z = (-0.2 + 0.04 * (df.AGEP - 40) + (0.2 + coef_shift) * (df.SCHL - 16)
             + 0.3 * (df.SEX == 1) - 0.8 * (df.DIS == 1) + intercept)
        y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-z))).astype(int), name="ESR")
    return df, y


def load_state(state: str, year: str = "2018", n_max: int = 30000, seed: int = 0,
               synthetic: bool = False, root_dir: str = "data", task: str = "income"):
    """Return (X, y) for one US state, for the given task.

    - income: y = 1 if income > $50k.
    - employment: y = 1 if employed (ESR == 1).
    """
    spec = task_spec(task)
    if synthetic:
        return make_synthetic_state(state, n_max, seed, task=task)
    from folktables import ACSDataSource, ACSIncome, ACSEmployment  # imported here so tests need no download
    problem = {"income": ACSIncome, "employment": ACSEmployment}[task]
    source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person", root_dir=root_dir)
    acs = source.get_data(states=[state], download=True)  # first call downloads from census.gov
    # The ACS person-level schema changed: RELP (relationship) was renamed to
    # RELSHIPP in 2019+. Folktables hardcodes "RELP" as a feature, so rename the
    # column back before it filters, so the same features work for both years.
    if "RELP" not in acs.columns and "RELSHIPP" in acs.columns:
        acs = acs.rename(columns={"RELSHIPP": "RELP"})
    X, y, _ = problem.df_to_pandas(acs)
    y = y[spec["y_name"]].astype(int)
    X = X[spec["features"]].astype(float)
    if len(X) > n_max:                                    # subsample so every state has equal weight
        idx = X.sample(n_max, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    return X.reset_index(drop=True), y.reset_index(drop=True)
