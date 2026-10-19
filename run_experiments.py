"""Model selection experiments (slow, ~20 min on 16 cores).

1. Simulated leaderboard: 20 random stratified 80/20 splits of train.csv, each test
   part about the size of the hidden test. Compares the random forest alone, 50/50
   blends of the forest and a calibrated booster, and the sensor model logit-blended
   with the temporal failure-rate model (both-direction and past-only windows), all
   thresholded at the training failure rate.
2. Ablations: repeated 5-fold CV (3 repeats) of preprocessing / model variants.

Writes reports/leaderboard_rounds.csv, reports/leaderboard_summary.csv and
reports/ablations.csv. Usage: python run_experiments.py [--skip-ablations]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from secom.data import load_train, parse_timestamp, usable_sensor_columns
from secom.evaluate import (leaderboard_rounds, leaderboard_scores, leaderboard_summary,
                            repeated_oof, simulate_leaderboard, summarize_oof)
from secom.pipeline import (BOOSTERS, _logit, _select, build_model, calibrated_booster, detrended_forest,
                            random_forest, time_model)

REPORTS = ROOT / "reports"
TIME_WEIGHT = 0.4


def logit_blend(sensor_probs, time_probs, w=TIME_WEIGHT):
    """Same arithmetic as pipeline.LogitBlend, applied to cached probabilities."""
    return [1 / (1 + np.exp(-((1 - w) * _logit(s) + w * _logit(t)))) for s, t in zip(sensor_probs, time_probs)]


def run_leaderboard(X, y):
    rounds = leaderboard_rounds(y)
    builders = {"rf": lambda Xtr: random_forest(usable_sensor_columns(Xtr))}
    for b in BOOSTERS:
        builders[f"cal_{b}"] = lambda Xtr, b=b: calibrated_booster(usable_sensor_columns(Xtr), b)
    builders["rf_detrended"] = lambda Xtr: detrended_forest(usable_sensor_columns(Xtr))
    builders["rf_detrended_z"] = lambda Xtr: detrended_forest(usable_sensor_columns(Xtr), scale=True)
    builders["time_only"] = lambda Xtr: time_model()
    builders["time_only_past"] = lambda Xtr: time_model(past_only=True)
    t0 = time.time()
    probs = simulate_leaderboard(builders, X, y, rounds)
    print(f"leaderboard fits: {time.time() - t0:.0f}s")
    # A VotingClassifier(soft) blend is exactly the average of its members' probabilities.
    for b in BOOSTERS:
        probs[f"rf+{b}"] = [(r + g) / 2 for r, g in zip(probs["rf"], probs[f"cal_{b}"])]
    probs["rf+all_boosters"] = [np.mean([probs["rf"][i], *(probs[f"cal_{b}"][i] for b in BOOSTERS)], 0)
                                for i in range(len(rounds))]
    probs["rf+rf_detrended+lightgbm"] = [(a + b + c) / 3 for a, b, c in zip(probs["rf"], probs["rf_detrended"], probs["cal_lightgbm"])]
    probs["rf+rf_detrended+rf_detrended_z+lightgbm"] = [(a + b + c + d) / 4 for a, b, c, d in zip(
        probs["rf"], probs["rf_detrended"], probs["rf_detrended_z"], probs["cal_lightgbm"])]
    probs["FINAL: rf+rf_detrended+rf_detrended_z+lightgbm + time"] = logit_blend(
        probs["rf+rf_detrended+rf_detrended_z+lightgbm"], probs["time_only"])
    probs["v2: rf+rf_detrended+lightgbm + time"] = logit_blend(probs["rf+rf_detrended+lightgbm"], probs["time_only"])
    probs["v1: rf+lightgbm + time"] = logit_blend(probs["rf+lightgbm"], probs["time_only"])
    probs["rf+lightgbm + time (past-only)"] = logit_blend(probs["rf+lightgbm"], probs["time_only_past"])
    scores = leaderboard_scores(y, rounds, probs)
    scores.to_csv(REPORTS / "leaderboard_rounds.csv", index=False)
    summary = leaderboard_summary(scores)
    summary.to_csv(REPORTS / "leaderboard_summary.csv")
    print(summary.round(3).to_string())
    # Paired comparison against the forest alone (same splits, so differences are less noisy).
    base = scores[scores.model == "rf"].set_index("round")
    for name in summary.index.drop("rf"):
        other = scores[scores.model == name].set_index("round")
        d_pr = other.pr_auc - base.pr_auc
        d_ba = other.balanced_accuracy - base.balanced_accuracy
        print(f"{name:32s} vs rf: dPR {d_pr.mean():+.3f} (se {d_pr.std() / np.sqrt(len(d_pr)):.3f})  "
              f"dBA {d_ba.mean():+.3f} (se {d_ba.std() / np.sqrt(len(d_ba)):.3f})  "
              f"PR wins {int((d_pr > 0).sum())}/{len(d_pr)}")


def run_ablations(X, y):
    days = (parse_timestamp(X) - parse_timestamp(X).min()).dt.total_seconds() / 86400
    X_time = X.assign(days_since_start=days)
    rf = dict(n_estimators=500)  # 500 trees for speed; 500 vs 2000 scored the same.

    def lin(cols, clf):
        return Pipeline([("impute", _select(cols, SimpleImputer(strategy="median"))),
                         ("drop_constant", VarianceThreshold()), ("scale", StandardScaler()), ("clf", clf)])

    variants = {
        "rf (final recipe)": (X, lambda Xtr: random_forest(usable_sensor_columns(Xtr), **rf)),
        "rf class_weight=balanced": (X, lambda Xtr: random_forest(usable_sensor_columns(Xtr), class_weight="balanced_subsample", **rf)),
        "rf min_samples_leaf=1": (X, lambda Xtr: random_forest(usable_sensor_columns(Xtr), min_samples_leaf=1, **rf)),
        "rf max_missing=0.2": (X, lambda Xtr: random_forest(usable_sensor_columns(Xtr, 0.2), **rf)),
        "rf + SelectKBest(50)": (X, lambda Xtr: Pipeline([*random_forest(usable_sensor_columns(Xtr), **rf).steps[:2],
                                                         ("kbest", SelectKBest(f_classif, k=50)),
                                                         random_forest([], **rf).steps[2]])),
        "rf + missing indicators": (X, lambda Xtr: Pipeline([("impute", _select(usable_sensor_columns(Xtr), SimpleImputer(strategy="median", add_indicator=True))),
                                                            *random_forest([], **rf).steps[1:]])),
        "rf + days_since_start": (X_time, lambda Xtr: random_forest(usable_sensor_columns(Xtr) + ["days_since_start"], **rf)),
        "extra trees": (X, lambda Xtr: Pipeline([*random_forest(usable_sensor_columns(Xtr)).steps[:2],
                                                ("et", ExtraTreesClassifier(1000, min_samples_leaf=3, n_jobs=-1, random_state=42))])),
        "hist gradient boosting": (X, lambda Xtr: Pipeline([("select", _select(usable_sensor_columns(Xtr))),
                                                           ("hgb", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.02, max_iter=400, min_samples_leaf=20, max_features=0.3, l2_regularization=1.0, random_state=42))])),
        "logistic regression (L2)": (X, lambda Xtr: lin(usable_sensor_columns(Xtr), LogisticRegression(C=0.001, max_iter=3000))),
        # Full-size models from here on: the submitted sensor model, the time model, and their blends.
        "sensors: rf + lightgbm": (X, lambda Xtr: build_model(usable_sensor_columns(Xtr), "lightgbm")),
        "time model only": (X, lambda Xtr: time_model()),
        "time model only (past-only)": (X, lambda Xtr: time_model(past_only=True)),
        "v1: sensors + time (w=0.4)": (X, lambda Xtr: build_model(usable_sensor_columns(Xtr), "lightgbm", TIME_WEIGHT)),
        "FINAL: sensors + detrended rf + time": (X, lambda Xtr: build_model(usable_sensor_columns(Xtr), "lightgbm", TIME_WEIGHT,
                                                                             detrend_hours=72.0)),
        "sensors + time, past-only (w=0.4)": (X, lambda Xtr: build_model(usable_sensor_columns(Xtr), "lightgbm", TIME_WEIGHT, past_only=True)),
    }
    rows = []
    for name, (Xv, build) in variants.items():
        t0 = time.time()
        oof, folds = repeated_oof(build, Xv, y, n_repeats=3)
        s = summarize_oof(y, oof, folds)
        rows.append({"variant": name, **{f"{c}_mean": s[c].mean() for c in s}, **{f"{c}_std": s[c].std() for c in s}})
        print(f"{name:28s} PR {s.pr_auc.mean():.3f}  ROC {s.roc_auc.mean():.3f}  "
              f"BA(prior) {s.ba_prior.mean():.3f}  BA(youden) {s.ba_youden.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(REPORTS / "ablations.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--skip-leaderboard", action="store_true")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    X, y = load_train()
    if not args.skip_leaderboard:
        print("== simulated leaderboard (20 x 80/20) ==")
        run_leaderboard(X, y)
    if not args.skip_ablations:
        print("\n== ablations (repeated 5-fold CV, 3 repeats) ==")
        run_ablations(X, y)
