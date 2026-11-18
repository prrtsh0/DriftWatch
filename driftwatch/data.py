"""Data loading.

Real data  : US Census (ACS) income task via the `folktables` package. One state = one dataset.
Fallback   : a synthetic generator (only for testing the code, NOT for research results).
"""
import numpy as np
import pandas as pd

FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
NUMERIC = ["AGEP", "SCHL", "WKHP"]                       # age, schooling level, hours/week
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]  # codes where order has no meaning


def _state_seed(state: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(state.upper()))


def make_synthetic_state(state: str, n: int = 20000, seed: int = 0):
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
    })[FEATURES]
    z = (-3.2 + 0.03 * (df.AGEP - 40) + (0.25 + coef_shift) * (df.SCHL - 16)
         + 0.03 * df.WKHP + 0.4 * (df.SEX == 1) + intercept)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-z))).astype(int), name="PINCP")
    return df, y


def load_state(state: str, year: str = "2018", n_max: int = 30000, seed: int = 0,
               synthetic: bool = False, root_dir: str = "data"):
    """Return (X, y) for one US state. y = 1 if income > $50k."""
    if synthetic:
        return make_synthetic_state(state, n_max, seed)
    from folktables import ACSDataSource, ACSIncome       # imported here so tests need no download
    source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person", root_dir=root_dir)
    acs = source.get_data(states=[state], download=True)  # first call downloads from census.gov
    # The ACS person-level schema changed: RELP (relationship) was renamed to
    # RELSHIPP in 2019+. Folktables' ACSIncome hardcodes "RELP" as a feature, so
    # rename the column back before it filters, so the same FEATURES work for
    # both frontiers.
    if "RELP" not in acs.columns and "RELSHIPP" in acs.columns:
        acs = acs.rename(columns={"RELSHIPP": "RELP"})
    X, y, _ = ACSIncome.df_to_pandas(acs)
    y = y["PINCP"].astype(int)
    X = X[FEATURES].astype(float)
    if len(X) > n_max:                                    # subsample so every state has equal weight
        idx = X.sample(n_max, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    return X.reset_index(drop=True), y.reset_index(drop=True)
