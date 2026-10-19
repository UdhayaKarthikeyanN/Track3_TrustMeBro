"""Train the final SECOM wafer model and save it.

Steps:
1. Repeated stratified 5-fold CV (5 repeats) of the final recipe -> local scores.
2. Decision threshold: training failure rate ("prior", default) or the Youden point
   of the out-of-fold scores ("youden").
3. Refit on all of train.csv, bake the threshold in with FixedThresholdClassifier
   and save models/secom_model.joblib (+ models/model_meta.json).

Usage: python train.py [--blend lightgbm|catboost|xgboost|none] [--time-weight 0.4] [--detrend-hours 72] [--threshold prior|youden] [--repeats 5]
"""
import argparse
import gzip
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from joblib.externals import cloudpickle
import pandas as pd
import sklearn

import secom
from secom.data import TIMESTAMP, load_train, usable_sensor_columns
from secom.evaluate import repeated_oof, summarize_oof, youden_threshold
from secom.pipeline import BOOSTERS, SEED, SchemaGuard, build_model, with_threshold

MODEL_PATH = ROOT / "models" / "secom_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
OOF_PATH = ROOT / "reports" / "oof_predictions.csv"

# Chosen from run_experiments.py (see reports/leaderboard_summary.csv).
DEFAULT_BLEND = "lightgbm"
DEFAULT_TIME_WEIGHT = 0.4
DEFAULT_DETREND_HOURS = 72.0


def library_versions():
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}
    for lib in BOOSTERS:
        try:
            versions[lib] = __import__(lib).__version__
        except ImportError:
            pass
    return versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default=DEFAULT_BLEND, choices=["none", *BOOSTERS])
    ap.add_argument("--time-weight", type=float, default=DEFAULT_TIME_WEIGHT,
                    help="weight of the temporal failure-rate model in the logit blend (0 = sensors only)")
    ap.add_argument("--detrend-hours", type=float, default=DEFAULT_DETREND_HOURS,
                    help="add a forest on sensors relative to wafers made within +/- this many hours (0 = off)")
    ap.add_argument("--no-detrend-z", dest="detrend_scaled", action="store_false",
                    help="leave out the forest on robust local z-scores")
    ap.add_argument("--threshold", default="prior", choices=["prior", "youden"])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    blend = None if args.blend == "none" else args.blend

    X, y = load_train()
    columns = usable_sensor_columns(X)
    print(f"train rows {len(y)}, failures {y.sum()} ({y.mean():.3%}), sensors used {len(columns)}/{X.shape[1] - 1}")
    members = ["RandomForest(2000 trees, unweighted)"]
    if args.detrend_hours:
        members.append(f"RandomForest on sensors detrended over +/-{args.detrend_hours:g} h")
        if args.detrend_scaled:
            members.append(f"RandomForest on robust local z-scores over +/-{args.detrend_hours:g} h")
    if blend:
        members.append(f"calibrated {blend}")
    description = " + ".join(members) + (f" (equal-weight soft vote)" if len(members) > 1 else "")
    if args.time_weight:
        description = f"logit blend: {1 - args.time_weight:.2f} x [{description}] + {args.time_weight:.2f} x temporal failure-rate model"
    print(f"model: {description}")

    t0 = time.time()
    oof, folds = repeated_oof(lambda Xtr: build_model(usable_sensor_columns(Xtr), blend, args.time_weight,
                                                          detrend_hours=args.detrend_hours,
                                                          detrend_scaled=args.detrend_scaled), X, y,
                              n_repeats=args.repeats)
    cv = summarize_oof(y, oof, folds)
    print(f"repeated 5-fold CV x{args.repeats} ({time.time() - t0:.0f}s):")
    for col in cv:
        print(f"  {col:10s} {cv[col].mean():.3f} +/- {cv[col].std():.3f}")

    prior = float(y.mean())
    youden = float(np.median([youden_threshold(y, p) for p in oof]))
    threshold = prior if args.threshold == "prior" else youden
    print(f"threshold ({args.threshold}): {threshold:.4f}  [prior {prior:.4f}, youden {youden:.4f}]")

    # SchemaGuard lets the saved model accept any column order, missing columns or a NumPy array.
    schema = [c for c in X.columns if c != "target"]
    model = with_threshold(SchemaGuard(build_model(columns, blend, args.time_weight, detrend_hours=args.detrend_hours,
                                              detrend_scaled=args.detrend_scaled), schema), threshold).fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    # Pickle this package's classes by value so the file loads with plain joblib.load,
    # without src/ on the path.
    cloudpickle.register_pickle_by_value(secom)
    with gzip.open(MODEL_PATH, "wb", compresslevel=6) as f:  # joblib.load detects gzip
        cloudpickle.dump(model, f)

    OOF_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame({TIMESTAMP: X[TIMESTAMP], "target": y, "oof_prob_fail": oof.mean(0)}).to_csv(OOF_PATH, index=False)

    meta = {
        "model": description,
        "time_weight": args.time_weight,
        "detrend_hours": args.detrend_hours,
        "detrend_scaled": args.detrend_scaled,
        "threshold": threshold,
        "threshold_strategy": args.threshold,
        "threshold_candidates": {"prior": prior, "youden": youden},
        "feature_columns": columns + ([TIMESTAMP] if args.time_weight else []),
        "n_train": int(len(y)),
        "n_failures": int(y.sum()),
        "cv": {"scheme": f"repeated stratified 5-fold x{args.repeats}",
               **{col: {"mean": float(cv[col].mean()), "std": float(cv[col].std())} for col in cv}},
        "oof_flagged_frac": float(np.mean(oof >= threshold)),
        "seed": SEED,
        "versions": library_versions(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"saved {MODEL_PATH.relative_to(ROOT)} and {META_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
