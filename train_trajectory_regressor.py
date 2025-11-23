#!/usr/bin/env python3
"""
Train a regression model to predict plate_x and plate_z from:
  - Statcast-style metadata features, and
  - Simple features summarising the 2D ball trajectory in pixel space.

This script is intentionally self-contained and reuses the shared
FeaturePipeline from ml_features.py to avoid duplicating preprocessing.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from flaml import AutoML

from ml_features import FeaturePipeline, build_aligned_trajectory_features

try:  # optional dependency
    import shap  # type: ignore
except Exception:  # pragma: no cover - shap may be unavailable
    shap = None


LOG_DIR = Path("logs")


def train_flaml_regressor(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
) -> AutoML:
    """Train a single-target FLAML regressor with a compact configuration."""
    automl = AutoML()
    settings = {
        "task": "regression",
        "metric": "mae",
        "eval_method": "cv",
        "seed": seed,
        "time_budget": 360,  # seconds; adjust if you want a deeper search
        "verbose": 1,
    }
    print(f"Training FLAML regressor for {name} (time_budget={settings['time_budget']}s)...")
    automl.fit(X_train=X, y_train=y, **settings)
    best_loss = automl.best_loss if automl.best_loss is not None else float("nan")
    print(f"[{name}] Best CV MAE≈{best_loss:.3f} ft (estimator={automl.best_estimator})")
    return automl


def log_model_artifacts(
    name: str,
    automl: AutoML,
    feature_frame: pd.DataFrame,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metadata_path = LOG_DIR / f"{name}_regressor_{timestamp}.json"
    metadata: dict[str, object] = {
        "model_name": name,
        "timestamp": timestamp,
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "n_features": int(feature_frame.shape[1]),
        "n_samples": int(feature_frame.shape[0]),
    }

    if shap is None:
        warnings.warn("shap not installed; skipping SHAP plot generation.", RuntimeWarning)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return

    try:
        shap_summary: list[dict[str, float]] | None = None
        shap_plot_path: Path | None = None
        sample = feature_frame.sample(
            min(len(feature_frame), 512),
            random_state=0,
        )
        base_model = getattr(automl.model, "model", automl.model)
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer(sample)
        except Exception:
            explainer = shap.Explainer(base_model.predict, sample)
            shap_values = explainer(sample)

        feature_names = getattr(shap_values, "feature_names", feature_frame.columns)
        shap_values_array = np.asarray(shap_values.values)
        mean_abs = np.mean(np.abs(shap_values_array), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:20]
        shap_summary = [
            {
                "feature": str(feature_names[i]),
                "mean_abs_shap": float(mean_abs[i]),
            }
            for i in top_idx
            if mean_abs[i] > 0
        ]

        shap.plots.beeswarm(shap_values, show=False)
        shap_plot_path = LOG_DIR / f"{name}_regressor_shap_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=200)
        plt.close()
        metadata["shap_plot"] = str(shap_plot_path)
        metadata["shap_top_features"] = shap_summary or []
    except Exception as exc:  # pragma: no cover - visualization best-effort
        warnings.warn(f"Failed to save SHAP plot for {name}: {exc}", RuntimeWarning)
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a regression model to predict plate_x and plate_z from "
            "metadata + pixel-space trajectory summaries."
        )
    )
    parser.add_argument(
        "--train-meta",
        type=Path,
        default=Path("data/train_ground_truth.csv"),
        help="CSV with training metadata and ground-truth plate_x/plate_z.",
    )
    parser.add_argument(
        "--test-meta",
        type=Path,
        default=Path("data/test_features.csv"),
        help="CSV with test metadata (no labels).",
    )
    parser.add_argument(
        "--train-detections",
        type=Path,
        default=Path("data/train_object_detections.csv"),
        help="CSV with YOLO detections for the trimmed training videos.",
    )
    parser.add_argument(
        "--test-detections",
        type=Path,
        default=Path("data/test_object_detections.csv"),
        help="CSV with YOLO detections for the test videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectory_regression_predictions.csv"),
        help="Where to save test-time plate_x / plate_z predictions.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="If set, train the regressors on train data only and skip test predictions.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed for the regressor and train/val split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_meta.exists():
        raise FileNotFoundError(f"Training metadata not found: {args.train_meta}")
    if not args.train_detections.exists():
        raise FileNotFoundError(f"Training detections not found: {args.train_detections}")

    has_test = not args.no_test
    if has_test:
        if not args.test_meta.exists():
            raise FileNotFoundError(f"Test metadata not found: {args.test_meta}")
        if not args.test_detections.exists():
            raise FileNotFoundError(f"Test detections not found: {args.test_detections}")

    print("Loading metadata...")
    train_meta = pd.read_csv(args.train_meta)
    test_meta = pd.read_csv(args.test_meta) if has_test else None

    print("Loading detections and building trajectory features...")
    train_det = pd.read_csv(args.train_detections)
    test_det = pd.read_csv(args.test_detections) if has_test else None

    traj_train = build_aligned_trajectory_features(train_meta, train_det)
    train_meta = train_meta.copy()
    train_meta[traj_train.columns] = traj_train

    if has_test and test_meta is not None and test_det is not None:
        traj_test = build_aligned_trajectory_features(test_meta, test_det)
        test_meta = test_meta.copy()
        test_meta[traj_test.columns] = traj_test
    else:
        traj_test = None
        test_meta = test_meta if has_test else None

    print("Building metadata feature matrices...")
    feat_pipe = FeaturePipeline()
    if has_test and test_meta is not None:
        feat_pipe.fit(train_meta, test_meta)
    else:
        feat_pipe.fit(train_meta)

    X_train = feat_pipe.transform(train_meta, log_tag="train_features")
    if has_test and test_meta is not None and traj_test is not None:
        X_test = feat_pipe.transform(test_meta, log_tag="test_features")
    else:
        X_test = None

    y_train = train_meta[["plate_x", "plate_z"]].to_numpy(dtype=float)

    print(f"Training samples with trajectories: {X_train.shape[0]}")
    print(f"Feature dimensionality: {X_train.shape[1]}")
    # Train separate FLAML regressors for plate_x and plate_z.
    model_x = train_flaml_regressor("plate_x", X_train, y_train[:, 0], args.random_state)
    model_z = train_flaml_regressor("plate_z", X_train, y_train[:, 1], args.random_state)

    log_model_artifacts("plate_x", model_x, X_train)
    log_model_artifacts("plate_z", model_z, X_train)

    if has_test and X_test is not None and test_meta is not None:
        print("Predicting plate_x / plate_z for test set...")
        plate_x_pred = model_x.predict(X_test)
        plate_z_pred = model_z.predict(X_test)

        output_df = pd.DataFrame(
            {
                "file_name": test_meta["file_name"],
                "plate_x_pred": plate_x_pred,
                "plate_z_pred": plate_z_pred,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output, index=False)
        print(f"Saved test predictions to {args.output} ({len(output_df)} rows).")
    else:
        print("Skipping test predictions (no-test mode).")


if __name__ == "__main__":
    main()
