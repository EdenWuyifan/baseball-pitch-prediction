#!/usr/bin/env python3
"""
Convert plate_x / plate_z predictions into pitch_class and zone labels.

This script merges:
  1. trajectory_regression_predictions.csv  (plate_x_pred, plate_z_pred, optional strike_proba)
  2. data/test_features.csv                (strike-zone metadata)

and emits plate_based_submission.csv using the MLB strike-zone geometry
described in README.md. If strike probabilities are provided the script
can apply the hybrid override logic from the EDEN Kaggle notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PLATE_WIDTH_FT = 17 / 12
PLATE_HALF = PLATE_WIDTH_FT / 2.0


def map_coordinates_to_zone(
    plate_x: float,
    plate_z: float,
    sz_top: float,
    sz_bot: float,
) -> int:
    """Map coordinates to MLB strike/ball zones (1-14)."""
    height = max(sz_top - sz_bot, 1e-3)
    z1 = sz_bot + height / 3.0
    z2 = sz_bot + 2.0 * height / 3.0
    x1 = -PLATE_HALF / 3.0
    x2 = PLATE_HALF / 3.0

    if -PLATE_HALF <= plate_x <= PLATE_HALF and sz_bot <= plate_z <= sz_top:
        if plate_x < x1:
            col = 0
        elif plate_x < x2:
            col = 1
        else:
            col = 2

        if plate_z > z2:
            row = 0
        elif plate_z > z1:
            row = 1
        else:
            row = 2

        return row * 3 + col + 1

    mid_height = sz_bot + height / 2.0
    if plate_x < -PLATE_HALF:
        return 11 if plate_z > mid_height else 13
    if plate_x > PLATE_HALF:
        return 12 if plate_z > mid_height else 14
    if plate_z > sz_top:
        return 11 if plate_x < 0 else 12
    return 13 if plate_x < 0 else 14


def map_coordinates_to_zone_hybrid(
    plate_x: float,
    plate_z: float,
    sz_top: float,
    sz_bot: float,
    strike_proba: float,
    *,
    threshold_low: float,
    threshold_high: float,
) -> int:
    """Hybrid classifier override for edge cases."""
    base_zone = map_coordinates_to_zone(plate_x, plate_z, sz_top, sz_bot)
    is_strike_by_coord = base_zone <= 9
    sz_top = float(sz_top)
    sz_bot = float(sz_bot)
    height = max(sz_top - sz_bot, 1e-3)
    z1 = sz_bot + height / 3.0
    z2 = sz_bot + 2.0 * height / 3.0
    sz_mid = sz_bot + height / 2.0

    if strike_proba < threshold_low and is_strike_by_coord:
        if base_zone in (1, 4, 7):
            return 11 if plate_z > sz_mid else 13
        if base_zone in (3, 6, 9):
            return 12 if plate_z > sz_mid else 14
        if base_zone == 2:
            return 11 if plate_x < 0 else 12
        if base_zone == 8:
            return 13 if plate_x < 0 else 14
        return 11 if plate_z > sz_mid else 13

    if strike_proba > threshold_high and not is_strike_by_coord:
        if base_zone == 11:
            return 1 if plate_z > z2 else (4 if plate_z > z1 else 7)
        if base_zone == 12:
            return 3 if plate_z > z2 else (6 if plate_z > z1 else 9)
        if base_zone == 13:
            return 7 if plate_x < 0 else 9
        if base_zone == 14:
            return 7 if plate_x < 0 else 9

    return base_zone


def get_pitch_class(zone: int) -> str:
    return "strike" if zone <= 9 else "ball"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("trajectory_regression_predictions.csv"),
        help="CSV containing plate_x_pred and plate_z_pred.",
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=Path("data/test_features.csv"),
        help="Metadata CSV with strike-zone parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plate_based_submission.csv"),
        help="Path to save derived pitch_class/zone predictions.",
    )
    parser.add_argument(
        "--strike-proba-column",
        type=str,
        default="strike_proba",
        help="Column in predictions CSV containing strike probabilities.",
    )
    parser.add_argument(
        "--hybrid-low",
        type=float,
        default=0.35,
        help="Lower strike probability threshold for ball overrides.",
    )
    parser.add_argument(
        "--hybrid-high",
        type=float,
        default=0.65,
        help="Upper strike probability threshold for strike overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = pd.read_csv(args.predictions)
    meta = pd.read_csv(args.test_features)

    merged = meta.merge(preds, on="file_name", how="left", validate="one_to_one")
    missing = merged["plate_x_pred"].isna().sum()
    if missing:
        raise RuntimeError(f"Missing plate predictions for {missing} rows.")

    sz_top_mean = merged["sz_top"].mean()
    sz_bot_mean = merged["sz_bot"].mean()
    merged["sz_top"] = merged["sz_top"].fillna(sz_top_mean)
    merged["sz_bot"] = merged["sz_bot"].fillna(sz_bot_mean)

    strike_col = args.strike_proba_column
    if strike_col not in merged.columns:
        raise RuntimeError(
            f"Strike probability column '{strike_col}' not found in predictions."
        )
    merged[strike_col] = merged[strike_col].fillna(0.5)

    def compute_zone(row: pd.Series) -> int:
        strike_proba = float(row[strike_col])
        return map_coordinates_to_zone_hybrid(
            float(row["plate_x_pred"]),
            float(row["plate_z_pred"]),
            float(row["sz_top"]),
            float(row["sz_bot"]),
            strike_proba,
            threshold_low=args.hybrid_low,
            threshold_high=args.hybrid_high,
        )

    merged["zone"] = merged.apply(compute_zone, axis=1).astype(int)
    merged["pitch_class"] = merged["zone"].apply(get_pitch_class)

    submission = merged[["file_name", "pitch_class", "zone"]].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(
        f"Saved {len(submission)} derived predictions to {args.output}\n"
        f"Strike-rate preview:\n{submission['pitch_class'].value_counts(normalize=True)}"
    )


if __name__ == "__main__":
    main()
