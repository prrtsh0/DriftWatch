"""Models: one tree-based (HistGradientBoosting) and one simple linear (logistic regression)."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, task_spec


def build_model(kind: str = "hgb", task: str = "income"):
    spec = task_spec(task)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)
    if kind == "logreg":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), spec["categorical"]),
            ("num", StandardScaler(), spec["numeric"])])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    raise ValueError(f"unknown model kind: {kind}")
