#!/usr/bin/env python3
"""
Train stronger ML classifiers for strike/zone prediction using FLAML AutoML.

The script orchestrates two FLAML AutoML runs—one for the binary pitch_class
label and another for the multi-class zone label. It:
  1. Cleans/features the Statcast metadata.
  2. Launches FLAML with verbose logging so you can monitor progress.
  3. Reports validation accuracy and the combined 0.7/0.3 competition metric.
  4. Re-fits best configs on the full training set and emits a submission CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from flaml import AutoML
from ml_features import (
    FeaturePipeline,
    INV_PITCH_MAP,
    PITCH_MAP,
)


@dataclass
class TrainingResult:
    automl: Any
    accuracy: float
    log_path: Path


class FLAMLClassifierTrainer:
    def __init__(
        self,
        random_state: int,
        verbose: int,
    ) -> None:
        self.random_state = random_state
        self.verbose = verbose
        self.pitch_bundle: TrainingResult | None = None
        self.zone_bundle: TrainingResult | None = None
        self.features = FeaturePipeline()

    def _fit_task(
        self,
        name: str,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> TrainingResult:
        mask = labels.notna()
        X = features.loc[mask].reset_index(drop=True)
        y = labels.loc[mask].reset_index(drop=True)

        automl = AutoML()
        log_path = Path(f"{name.lower().replace(' ', '_')}_flaml.log")
        settings = {
            "seed": self.random_state,
            "verbose": self.verbose,
            "eval_method": "cv",
            "metric": "accuracy",
        }

        print(
            f"[{name}] Starting FLAML search "
            f"(log={log_path})."
        )
        automl.fit(
            X_train=X,
            y_train=y,
            task="classification",
            **settings,
        )
        state = getattr(automl, "_state", None)
        if state is not None and getattr(state, "y_val", None) is not None:
            val_preds = automl.predict(state.X_val)
            acc = accuracy_score(state.y_val, val_preds)
        else:
            acc = 1.0 - automl.best_loss if automl.best_loss is not None else 0.0
        best_acc = 1.0 - automl.best_loss if automl.best_loss is not None else acc
        print(
            f"[{name}] Validation accuracy: {acc:.4f} "
            f"(best estimator: {automl.best_estimator})"
            f"best_acc≈{best_acc:.4f})."
        )

        return TrainingResult(automl=automl, accuracy=float(acc), log_path=log_path)

    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[TrainingResult, TrainingResult, pd.DataFrame]:
        self.features.fit(train_df, test_df)
        train_features = self.features.transform(train_df)
        test_features = self.features.transform(test_df)

        pitch_labels = train_df["pitch_class"].str.lower().map(PITCH_MAP)
        zone_labels = train_df["zone"].round().astype(int)

        print(f"Pitch labels: {pitch_labels.value_counts()}")
        print(f"Zone labels: {zone_labels.value_counts()}")

        self.pitch_bundle = self._fit_task("Pitch Class", train_features, pitch_labels)
        self.zone_bundle = self._fit_task("Zone", train_features, zone_labels)

        submission = self._predict_submission(test_df["file_name"], test_features)
        return self.pitch_bundle, self.zone_bundle, submission

    def _predict_submission(self, file_names: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        if not self.pitch_bundle or not self.zone_bundle:
            raise RuntimeError("Models are not trained yet.")

        pitch_preds = self.pitch_bundle.automl.predict(features)
        pitch_labels = [INV_PITCH_MAP[int(val)] for val in pitch_preds]

        zone_preds = self.zone_bundle.automl.predict(features).astype(int)
        zone_preds = np.clip(zone_preds, 1, 14)

        return pd.DataFrame(
            {
                "file_name": file_names,
                "pitch_class": pitch_labels,
                "zone": zone_preds,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FLAML AutoML classifiers for strike/zone prediction.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train/test CSVs.")
    parser.add_argument("--output", type=Path, default=Path("ml_baseline_submission.csv"), help="Submission CSV path.")
    parser.add_argument("--random-state", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--verbose", type=int, default=3, help="FLAML verbosity (0-3).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train_ground_truth.csv"
    test_path = args.data_dir / "test_features.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train/test CSVs in the provided data directory.")

    print("Loading datasets with pandas...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    trainer = FLAMLClassifierTrainer(
        random_state=args.random_state,
        verbose=args.verbose,
    )
    pitch_bundle, zone_bundle, submission = trainer.fit(train_df, test_df)

    combined_metric = 0.7 * pitch_bundle.accuracy + 0.3 * zone_bundle.accuracy
    print(f"Combined validation metric (0.7*class + 0.3*zone): {combined_metric:.4f}")
    print(f"Pitch logs: {pitch_bundle.log_path} | Zone logs: {zone_bundle.log_path}")

    submission.to_csv(args.output, index=False)
    print(f"Submission written to {args.output} ({len(submission)} rows).")


if __name__ == "__main__":
    main()
