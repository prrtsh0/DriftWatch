import pandas as pd

ALL = ["CA", "TX", "NY", "FL", "WA", "IL", "OH", "MI", "MN",
       "MS", "AL", "WV", "ME", "UT", "MA"]
d = pd.read_csv("outputs/results.csv")
print("n_rows", len(d))
print("n_unique_sources", d.source.nunique())
print("n_unique_targets", d.target.nunique())
score_cols = ["mean_psi", "max_psi", "mean_ks", "domain_auc", "acc_drop",
              "covariate_share", "concept_share",
              "norm_weights_ess_ratio", "concept_share_se"]
print("any_nan_scores", d[score_cols].isna().any().any())
print("missing_sources", sorted(set(ALL) - set(d.source)))
print("missing_targets", sorted(set(ALL) - set(d.target)))
print()
print(d[["source", "target", "acc_drop", "covariate_share", "concept_share",
         "norm_weights_ess_ratio", "norm_weights_clip_fraction",
         "concept_share_se"]].round(3).head(15).to_string(index=False))
