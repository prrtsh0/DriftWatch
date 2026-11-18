"""Monitoring component used by the API: compare a new batch with the stored reference sample."""
import pandas as pd

from .data import FEATURES
from .drift import domain_auc, ks_table, psi_table


class DriftMonitor:
    def __init__(self, reference: pd.DataFrame, warn: float = 0.1, alert: float = 0.25, min_domain_n: int = 500):
        self.reference = reference[FEATURES].reset_index(drop=True)
        self.warn, self.alert, self.min_domain_n = warn, alert, min_domain_n

    def check(self, batch: pd.DataFrame) -> dict:
        batch = batch[FEATURES].reset_index(drop=True)
        psi, ks = psi_table(self.reference, batch), ks_table(self.reference, batch)
        worst = float(psi.max())
        status = "alert" if worst >= self.alert else "warn" if worst >= self.warn else "ok"
        return {
            "n_rows": len(batch), "status": status,
            "mean_psi": float(psi.mean()), "max_psi": worst, "mean_ks": float(ks.mean()),
            "feature_psi": {f: float(v) for f, v in psi.items()},
            "top_drifting_features": [{"feature": f, "psi": float(v)} for f, v in psi.nlargest(3).items()],
            "domain_auc": domain_auc(self.reference, batch) if len(batch) >= self.min_domain_n else None,
            "rule": f"status is warn at max_psi>={self.warn}, alert at >={self.alert}",
        }
