"""Train the model the API serves and save it with a reference sample for drift checks."""
import argparse, json, os
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from driftwatch.data import load_state
from driftwatch.model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="CA"); ap.add_argument("--model", default="hgb")
ap.add_argument("--year", default="2018"); ap.add_argument("--n-max", type=int, default=30000)
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

# Honor the same env var the API uses, so build-time output and runtime reads
# always agree regardless of the working directory.
ART = Path(os.getenv("DRIFTWATCH_ARTIFACTS", "artifacts"))

X, y = load_state(a.state, a.year, a.n_max, synthetic=a.synthetic)
X_tr, X_ref, y_tr, y_ref = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = build_model(a.model).fit(X_tr, y_tr)
y_pred = model.predict(X_ref)
acc = accuracy_score(y_ref, y_pred)
tn, fp, fn, tp = confusion_matrix(y_ref, y_pred).ravel()
ART.mkdir(parents=True, exist_ok=True)
joblib.dump(model, ART / "model.joblib")
X_ref.sample(min(5000, len(X_ref)), random_state=0).to_csv(ART / "reference.csv", index=False)
ref_prob = model.predict_proba(X_ref)[:, 1]
ref_prob_mean = float(ref_prob.mean())
metrics = {"state": a.state, "model": a.model, "home_accuracy": acc,
           "ref_prob_mean": ref_prob_mean,
           "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}
json.dump(metrics, open(ART / "metrics.json", "w"))
print(f"saved model trained on {a.state} to {ART}; home accuracy {acc:.3f}")
print("\nModel 2x2 confusion matrix on the held-out 30% (rows = true, cols = predicted):")
print(f"  truly over-$50k -> predicted over / under : {tp:5d} / {fn:5d}")
print(f"  truly under-$50k -> predicted over / under : {fp:5d} / {tn:5d}")
print(f"  accuracy = (TP+TN)/(all) = ({tp}+{tn})/{tp+fp+fn+tn} = {acc:.3f}")
