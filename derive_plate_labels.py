#!/usr/bin/env python3
"""
Convert plate_x / plate_z predictions into pitch_class and zone labels.

This script merges:
  1. trajectory_regression_predictions.csv  (plate_x_pred, plate_z_pred)
  2. data/test_features.csv                (strike-zone metadata)

and emits plate_based_submission.csv using the MLB strike-zone geometry
described in README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PLATE_WIDTH_FT = 17 / 12
BALL_RADIUS_FT = 1.5 / 12
PLATE_HALF = PLATE_WIDTH_FT / 2.0


def derive_pitch_class(row: pd.Series) -> str:
    x = float(row["plate_x_pred"])
    z = float(row["plate_z_pred"])
    sz_top = float(row["sz_top"])
    sz_bot = float(row["sz_bot"])

    x_limit = PLATE_HALF + BALL_RADIUS_FT
    z_lower = sz_bot - BALL_RADIUS_FT
    z_upper = sz_top + BALL_RADIUS_FT
    is_strike = (abs(x) <= x_limit) and (z_lower <= z <= z_upper)
    return "strike" if is_strike else "ball"


def derive_zone(row: pd.Series) -> int:
    x = float(row["plate_x_pred"])
    z = float(row["plate_z_pred"])
    sz_top = float(row["sz_top"])
    sz_bot = float(row["sz_bot"])

    x_limit = PLATE_HALF + BALL_RADIUS_FT
    height = max(sz_top - sz_bot, 1e-3)
    low_cut = sz_bot + height / 3.0
    high_cut = sz_bot + 2.0 * height / 3.0

    if (abs(x) <= x_limit) and (sz_bot <= z <= sz_top):
        if z >= high_cut:
            row_idx = 0
        elif z >= low_cut:
            row_idx = 1
        else:
            row_idx = 2

        if x < -PLATE_HALF / 3.0:
            col_idx = 0
        elif x > PLATE_HALF / 3.0:
            col_idx = 2
        else:
            col_idx = 1

        return row_idx * 3 + col_idx + 1

    if z > sz_top:
        if x < -x_limit:
            return 11
        if x > x_limit:
            return 13
        return 12

    if z < sz_bot:
        return 14

    return 11 if x < 0 else 13


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = pd.read_csv(args.predictions)
    meta = pd.read_csv(args.test_features)

    merged = meta.merge(preds, on="file_name", how="left", validate="one_to_one")
    missing = merged["plate_x_pred"].isna().sum()
    if missing:
        raise RuntimeError(f"Missing plate predictions for {missing} rows.")

    merged["pitch_class"] = merged.apply(derive_pitch_class, axis=1)
    merged["zone"] = merged.apply(derive_zone, axis=1).astype(int)

    submission = merged[["file_name", "pitch_class", "zone"]].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(
        f"Saved {len(submission)} derived predictions to {args.output}\n"
        f"Strike-rate preview:\n{submission['pitch_class'].value_counts(normalize=True)}"
    )


if __name__ == "__main__":
    main()
