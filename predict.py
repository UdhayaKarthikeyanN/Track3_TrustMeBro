"""Inference for the SECOM wafer fault model.

Python:
    from predict import predict, predict_proba
    out = predict(test_df)            # DataFrame: prob_fail, pred (+ timestamp if present)
    p = predict_proba(test_df)        # numpy array of fail probabilities

CLI:
    python predict.py test.csv -o predictions.csv

Use `pred` (0/1, tuned threshold) for Balanced Accuracy and `prob_fail` for PR-AUC.
The input should have the train.csv columns. `target`, extra columns and column order
are ignored; a missing sensor column is treated as all-NaN, and a missing or
unreadable `timestamp` makes the time model fall back to the overall failure rate.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "secom_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"

_cache = {}


def load_model(path=MODEL_PATH):
    """The fitted model; its predict() already applies the tuned threshold."""
    path = Path(path)
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]


def _feature_columns():
    return json.loads(META_PATH.read_text())["feature_columns"]


def _prepare(df):
    # Keep exactly the training columns; absent sensors become NaN and are imputed,
    # an absent timestamp makes the time model fall back to the overall failure rate.
    out = df.reindex(columns=_feature_columns())
    sensors = [c for c in out.columns if c != "timestamp"]
    out[sensors] = out[sensors].astype(float)
    return out


def predict_proba(df, model_path=MODEL_PATH):
    """Probability that each wafer fails (class 1)."""
    return load_model(model_path).predict_proba(_prepare(df))[:, 1]


def predict(df, model_path=MODEL_PATH):
    """Fail probability and 0/1 prediction (1 = defective) for each wafer."""
    model = load_model(model_path)
    proba = model.predict_proba(_prepare(df))[:, 1]
    out = pd.DataFrame({"prob_fail": proba, "pred": (proba >= model.threshold).astype(int)},
                       index=df.index)
    if "timestamp" in df.columns:
        out.insert(0, "timestamp", df["timestamp"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="input CSV with the train.csv sensor columns")
    ap.add_argument("-o", "--output", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = predict(df)
    out.to_csv(args.output, index=False)
    print(f"{len(out)} wafers, {int(out.pred.sum())} predicted defective -> {args.output}")
    if "target" in df.columns:
        from sklearn.metrics import average_precision_score, balanced_accuracy_score
        y = df["target"].astype(int)
        print(f"Balanced Accuracy {balanced_accuracy_score(y, out.pred):.4f}  "
              f"PR-AUC {average_precision_score(y, out.prob_fail):.4f}")


if __name__ == "__main__":
    main()
