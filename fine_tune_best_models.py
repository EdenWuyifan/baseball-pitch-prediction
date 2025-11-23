#!/usr/bin/env python3
"""
Fine-tune the best-performing LightGBM and ExtraTrees models surfaced by FLAML
and export a submission CSV.

Compared to train_sota_classifiers.py, this script provides direct control over
the estimators so we can run lightweight hyper-parameter searches and inspect
diagnostics (class balance, validation accuracy, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from ml_features import FeaturePipeline, INV_PITCH_MAP, PITCH_MAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LightGBM/ExtraTrees baselines and make predictions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train/test CSVs.")
    parser.add_argument("--output", type=Path, default=Path("ml_finetuned_submission.csv"), help="Submission CSV path.")
    parser.add_argument("--random-state", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--pitch-iter", type=int, default=20, help="Randomized search iterations for pitch model.")
    parser.add_argument("--zone-iter", type=int, default=30, help="Randomized search iterations for zone model.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds for evaluation/tuning.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallelism for scikit-learn searches (default: use all available cores).",
    )
    return parser.parse_args()


def _maybe_tune_model(
    estimator,
    search_space: Dict[str, list],
    *,
    n_iter: int,
    cv: StratifiedKFold,
    scoring: str,
    n_jobs: int,
    random_state: int,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[Any, float]:
    """Optionally run RandomizedSearchCV and return (fitted_estimator, cv_score)."""
    if n_iter <= 0:
        estimator.fit(X, y)
        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        return estimator, float(np.mean(scores))

    search = RandomizedSearchCV(
        estimator,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X, y)
    best_estimator = search.best_estimator_
    # Ensure the refitted model sees the entire dataset.
    best_estimator.fit(X, y)
    return best_estimator, float(search.best_score_)


def build_pitch_estimator(random_state: int, n_jobs: int) -> LGBMClassifier:
    # Seeded with the FLAML best config.
    return LGBMClassifier(
        objective="binary",
        n_estimators=141,
        learning_rate=0.04824748268727149,
        num_leaves=139,
        colsample_bytree=0.5261441571042451,
        max_bin=511,
        min_child_samples=8,
        reg_alpha=0.0028969208338993344,
        reg_lambda=0.024463247502165594,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def pitch_search_space() -> Dict[str, list]:
    return {
        "num_leaves": [100, 120, 140, 160, 180],
        "learning_rate": [0.02, 0.035, 0.05, 0.065, 0.08],
        "n_estimators": [120, 150, 180, 210, 260],
        "max_bin": [383, 511, 767],
        "min_child_samples": [4, 8, 12, 24],
        "colsample_bytree": [0.45, 0.55, 0.65, 0.75],
        "reg_alpha": [0.0, 0.0025, 0.005, 0.01],
        "reg_lambda": [0.0, 0.02, 0.04, 0.08],
    }


def build_zone_estimator(random_state: int, n_jobs: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=458,
        criterion="entropy",
        max_features=1.0,
        max_leaf_nodes=18344,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def zone_search_space() -> Dict[str, list]:
    return {
        "n_estimators": [350, 450, 550, 650],
        "max_depth": [None, 32, 48],
        "max_leaf_nodes": [12000, 18344, 24000],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.6, 0.8, 1.0],
        "criterion": ["entropy", "gini"],
    }


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train_ground_truth.csv"
    test_path = args.data_dir / "test_features.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train/test CSVs in the provided data directory.")

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    pipeline = FeaturePipeline()
    pipeline.fit(train_df, test_df)
    train_features = pipeline.transform(train_df)
    test_features = pipeline.transform(test_df)

    pitch_labels = train_df["pitch_class"].str.lower().str.strip().map(PITCH_MAP)
    zone_labels = train_df["zone"]

    pitch_mask = pitch_labels.notna()
    zone_mask = zone_labels.notna()

    if not pitch_mask.all():
        print(f"Dropping {len(pitch_labels) - pitch_mask.sum()} rows without pitch labels.")
    if not zone_mask.all():
        print(f"Dropping {len(zone_labels) - zone_mask.sum()} rows without zone labels.")

    pitch_features = train_features.loc[pitch_mask].reset_index(drop=True)
    pitch_labels = pitch_labels.loc[pitch_mask].astype(int).reset_index(drop=True)

    zone_features = train_features.loc[zone_mask].reset_index(drop=True)
    zone_labels = zone_labels.loc[zone_mask].round().astype(int).reset_index(drop=True)

    pitch_cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    zone_cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    print("\nTuning LightGBM pitch classifier...")
    pitch_estimator = build_pitch_estimator(args.random_state, args.n_jobs)
    pitch_model, pitch_score = _maybe_tune_model(
        pitch_estimator,
        pitch_search_space(),
        n_iter=args.pitch_iter,
        cv=pitch_cv,
        scoring="accuracy",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        X=pitch_features,
        y=pitch_labels,
    )
    print(f"Pitch CV accuracy≈{pitch_score:.4f}")

    print("\nTuning ExtraTrees zone classifier...")
    zone_estimator = build_zone_estimator(args.random_state, args.n_jobs)
    zone_model, zone_score = _maybe_tune_model(
        zone_estimator,
        zone_search_space(),
        n_iter=args.zone_iter,
        cv=zone_cv,
        scoring="accuracy",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        X=zone_features,
        y=zone_labels,
    )
    print(f"Zone CV accuracy≈{zone_score:.4f}")

    print("\nRefitting models on the full training set...")
    pitch_model.fit(pitch_features, pitch_labels)
    zone_model.fit(zone_features, zone_labels)

    pitch_preds = pitch_model.predict(test_features)
    zone_preds = zone_model.predict(test_features).astype(int)
    zone_preds = np.clip(zone_preds, 1, 14)

    pitch_output = [INV_PITCH_MAP[int(val)] for val in pitch_preds]
    submission = pd.DataFrame(
        {
            "file_name": test_df["file_name"],
            "pitch_class": pitch_output,
            "zone": zone_preds,
        }
    )

    pitch_counts = submission["pitch_class"].value_counts()
    zone_counts = submission["zone"].value_counts().sort_index()
    print("\nPrediction sanity checks:")
    print(f"  Pitch predictions: {pitch_counts.to_dict()}")
    print(f"  Zone predictions: {zone_counts.to_dict()}")

    submission.to_csv(args.output, index=False)
    print(f"\nSubmission written to {args.output} ({len(submission)} rows).")


if __name__ == "__main__":
    main()
