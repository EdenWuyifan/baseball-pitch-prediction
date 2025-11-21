#!/usr/bin/env python3
"""
Enhanced naive baseline using only standard-library tooling.

Key ideas:
- A light ballistic projection provides raw plate-crossing estimates.
- Compact linear calibrators (no numpy/pandas dependencies) align those
  estimates with Statcast labels using least-squares fits.
- Batter/pitcher normalized centroids drive zone prediction, removing reliance
  on rigid third-based grids.

The result keeps the spirit of a physics-first baseline while squeezing as
much signal as possible out of the provided metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


Row = Dict[str, object]
FeatureFn = Callable[[Row, Dict[str, float]], float]

BASE_FLOAT_COLUMNS = [
    "sz_top",
    "sz_bot",
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
]
TRAIN_FLOAT_COLUMNS = ["plate_x", "plate_z"]


def load_rows(path: Path, expect_labels: bool) -> List[Row]:
    """Load CSV rows with pandas for reliable numeric casting."""

    df = pd.read_csv(path)
    numeric_cols = [col for col in BASE_FLOAT_COLUMNS + TRAIN_FLOAT_COLUMNS if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if numeric_cols:
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    if "stand" in df.columns:
        df["stand"] = df["stand"].fillna("R")
    else:
        df["stand"] = "R"

    if "p_throws" in df.columns:
        df["p_throws"] = df["p_throws"].fillna("R")
    else:
        df["p_throws"] = "R"

    if expect_labels:
        df["pitch_class"] = df["pitch_class"].astype(str)
        if "zone" in df.columns:
            df["zone"] = df["zone"].round().astype(int)
    else:
        missing_cols = set(TRAIN_FLOAT_COLUMNS) - set(df.columns)
        for col in missing_cols:
            df[col] = np.nan

    return df.to_dict("records")


def write_submission(path: Path, predictions: Sequence[Row]) -> None:
    """Persist submission rows via pandas for simplicity."""

    df = pd.DataFrame(predictions, columns=["file_name", "pitch_class", "zone"])
    df.to_csv(path, index=False)


def evaluate_predictions(true_rows: Sequence[Row], pred_rows: Sequence[Row]) -> Dict[str, float]:
    """Compute leaderboard-style score for local validation."""

    if not true_rows or not pred_rows:
        return {"class_accuracy": 0.0, "zone_accuracy": 0.0, "combined_score": 0.0, "n_samples": 0}

    df_true = pd.DataFrame(true_rows)[["file_name", "pitch_class", "zone"]]
    df_pred = pd.DataFrame(pred_rows)[["file_name", "pitch_class", "zone"]]
    merged = df_true.merge(df_pred, on="file_name", suffixes=("_true", "_pred"))
    if merged.empty:
        return {"class_accuracy": 0.0, "zone_accuracy": 0.0, "combined_score": 0.0, "n_samples": 0}

    class_accuracy = (merged["pitch_class_true"] == merged["pitch_class_pred"]).mean()
    zone_accuracy = (merged["zone_true"].astype(int) == merged["zone_pred"].astype(int)).mean()
    combined_score = 0.7 * class_accuracy + 0.3 * zone_accuracy

    return {
        "class_accuracy": float(class_accuracy),
        "zone_accuracy": float(zone_accuracy),
        "combined_score": float(combined_score),
        "n_samples": int(len(merged)),
    }


def least_squares(design: Sequence[Sequence[float]], targets: Sequence[float], ridge: float = 1e-6) -> List[float]:
    """Solve a regularized least-squares problem with numpy."""

    if not design:
        return []

    X = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1, 1)
    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y).ravel()
    if not finite_mask.any():
        return []
    X = X[finite_mask]
    y = y[finite_mask]
    ones = np.ones((X.shape[0], 1), dtype=float)
    X_ext = np.hstack([X, ones])
    regularizer = ridge * np.eye(X_ext.shape[1], dtype=float)
    w = np.linalg.pinv(X_ext.T @ X_ext + regularizer) @ X_ext.T @ y
    return w.flatten().tolist()


class LinearCalibrator:
    """Tiny helper around least-squares fitting for calibrated coordinates."""

    def __init__(self, feature_fns: Sequence[FeatureFn], default_key: str):
        self.feature_fns = list(feature_fns)
        self.weights: List[float] = []
        self.default_key = default_key

    def fit(self, rows: Sequence[Row], contexts: Sequence[Dict[str, float]], targets: Sequence[float]) -> None:
        design = []
        for row, ctx in zip(rows, contexts):
            design.append([fn(row, ctx) for fn in self.feature_fns])
        self.weights = least_squares(design, targets)

    def predict(self, row: Row, ctx: Dict[str, float]) -> float:
        if not self.weights:
            return ctx[self.default_key]
        features = [fn(row, ctx) for fn in self.feature_fns]
        bias = self.weights[-1]
        return sum(w * f for w, f in zip(self.weights[:-1], features)) + bias


class NaivePhysicsPredictor:
    """Physics-first predictor with calibrated plate-crossing and zone centroids."""

    PLATE_WIDTH = 17.0 / 12.0
    BALL_RADIUS = 1.5 / 12.0
    MOUND_TO_PLATE = 60.5
    MPH_TO_FPS = 1.46667

    def __init__(self) -> None:
        ctx = self._ctx
        x_features = [
            ctx("raw_x"),
            ctx("release_pos_x"),
            ctx("pfx_x_ft"),
            ctx("distance_to_plate"),
            ctx("release_pos_y"),
            ctx("pfx_x_times_distance"),
            ctx("release_extension"),
            ctx("stand_sign"),
            ctx("throws_sign"),
            ctx("spin_rate_k"),
            ctx("effective_speed_fps"),
        ]
        self.x_calibrators: Dict[float, LinearCalibrator] = {
            -1.0: LinearCalibrator(x_features, default_key="raw_x"),
            1.0: LinearCalibrator(x_features, default_key="raw_x"),
        }
        self.z_calibrator = LinearCalibrator(
            [
                ctx("raw_z"),
                ctx("release_pos_z"),
                ctx("pfx_z_ft"),
                ctx("time_of_flight"),
                ctx("distance_to_plate"),
                ctx("effective_speed_fps"),
                ctx("release_extension"),
                ctx("spin_rate_k"),
                ctx("pfx_z_times_distance"),
                ctx("stand_sign"),
            ],
            default_key="raw_z",
        )

        self.zone_x_scale = 1.0
        self.zone_top_offset = 0.0
        self.zone_bottom_offset = 0.0
        self.zone_low_ratio = 1.0 / 3.0
        self.zone_high_ratio = 2.0 / 3.0
        plate_half = self.PLATE_WIDTH / 2.0
        self.zone_left_boundary = -plate_half / 3.0
        self.zone_right_boundary = plate_half / 3.0
        self.zone_profiles: Dict[int, Dict[str, float]] = {}
        self.classifier_weights: List[float] = []
        self.classifier_threshold = 0.5

    @staticmethod
    def _ctx(key: str, scale: float = 1.0) -> FeatureFn:
        return lambda _row, ctx, k=key, s=scale: ctx.get(k, 0.0) * s

    @staticmethod
    def _sign(value: str) -> float:
        return 1.0 if value.upper().startswith("R") else -1.0

    def _physics_context(self, row: Row) -> Dict[str, float]:
        release_x = float(row.get("release_pos_x", 0.0))
        release_y = float(row.get("release_pos_y", 54.0))
        release_z = float(row.get("release_pos_z", 6.0))
        eff_speed = float(row.get("effective_speed", row.get("release_speed", 90.0)))
        pfx_x_ft = float(row.get("pfx_x", 0.0)) / 12.0
        pfx_z_ft = float(row.get("pfx_z", 0.0)) / 12.0
        distance_to_plate = self.MOUND_TO_PLATE - release_y
        speed_fps = max(eff_speed * self.MPH_TO_FPS, 1e-3)
        time_of_flight = max(distance_to_plate / speed_fps, 1e-3)
        gravity_drop = -0.5 * 32.2 * (time_of_flight ** 2)
        trajectory_correction = distance_to_plate * 0.02
        raw_x = release_x + pfx_x_ft
        raw_z = release_z + gravity_drop + pfx_z_ft + trajectory_correction

        return {
            "raw_x": raw_x,
            "raw_z": raw_z,
            "release_pos_x": release_x,
            "release_pos_y": release_y,
            "release_pos_z": release_z,
            "pfx_x_ft": pfx_x_ft,
            "pfx_z_ft": pfx_z_ft,
            "distance_to_plate": distance_to_plate,
            "time_of_flight": time_of_flight,
            "effective_speed_fps": speed_fps,
            "release_extension": float(row.get("release_extension", 6.0)),
            "spin_rate_k": float(row.get("release_spin_rate", 2200.0)) * 1e-3,
            "stand_sign": self._sign(str(row.get("stand", "R"))),
            "throws_sign": self._sign(str(row.get("p_throws", "R"))),
            "pfx_x_times_distance": pfx_x_ft * distance_to_plate,
            "pfx_z_times_distance": pfx_z_ft * distance_to_plate,
        }

    def _predict_from_context(self, row: Row, ctx: Dict[str, float]) -> Tuple[float, float]:
        throw_sign = ctx.get("throws_sign", 1.0)
        x_model = self.x_calibrators.get(throw_sign) or next(iter(self.x_calibrators.values()))
        x_val = x_model.predict(row, ctx)
        z_val = self.z_calibrator.predict(row, ctx)
        return x_val, z_val

    def predict_plate_crossing(self, row: Row) -> Tuple[float, float]:
        ctx = self._physics_context(row)
        return self._predict_from_context(row, ctx)

    def calibrate(self, rows: Sequence[Row]) -> None:
        if not rows or "plate_x" not in rows[0]:
            print("  Skipping calibration: training labels unavailable.")
            return

        print("  Calibrating ballistic corrections...")
        contexts = [self._physics_context(row) for row in rows]

        grouped: Dict[float, List[Tuple[Row, Dict[str, float]]]] = {sign: [] for sign in self.x_calibrators}
        for row, ctx in zip(rows, contexts):
            grouped.setdefault(ctx.get("throws_sign", 1.0), []).append((row, ctx))

        for sign, calibrator in self.x_calibrators.items():
            subset = grouped.get(sign, [])
            if subset:
                sub_rows, sub_ctxs = zip(*subset)
                calibrator.fit(sub_rows, sub_ctxs, [float(r["plate_x"]) for r in sub_rows])
            else:
                calibrator.weights = []

        self.z_calibrator.fit(rows, contexts, [float(row["plate_z"]) for row in rows])

        predictions = [self._predict_from_context(row, ctx) for row, ctx in zip(rows, contexts)]
        x_mae = sum(abs(px - float(row["plate_x"])) for row, (px, _) in zip(rows, predictions)) / len(rows)
        z_mae = sum(abs(pz - float(row["plate_z"])) for row, (_, pz) in zip(rows, predictions)) / len(rows)
        print(f"    X MAE: {x_mae:.3f} ft")
        print(f"    Z MAE: {z_mae:.3f} ft")

        self._calibrate_zone_grid(rows)
        self._tune_classification_window(rows, predictions)
        self._fit_classifier(rows, predictions, contexts)

    def _calibrate_zone_grid(self, rows: Sequence[Row]) -> None:
        plate_half = self.PLATE_WIDTH / 2.0
        row_groups = {"high": [], "mid": [], "low": []}
        col_groups = {"left": [], "mid": [], "right": []}
        profile_accum: Dict[int, Dict[str, List[float]]] = {}

        for row in rows:
            zone = int(row["zone"])
            height = max(float(row["sz_top"]) - float(row["sz_bot"]), 1e-3)
            norm_z = (float(row["plate_z"]) - float(row["sz_bot"])) / height
            norm_x = float(row["plate_x"]) / plate_half

            if zone in (1, 2, 3):
                row_groups["high"].append(norm_z)
            elif zone in (4, 5, 6):
                row_groups["mid"].append(norm_z)
            elif zone in (7, 8, 9):
                row_groups["low"].append(norm_z)

            if zone in (1, 4, 7):
                col_groups["left"].append(norm_x)
            elif zone in (2, 5, 8):
                col_groups["mid"].append(norm_x)
            elif zone in (3, 6, 9):
                col_groups["right"].append(norm_x)

            bucket = profile_accum.setdefault(zone, {"x": [], "z": []})
            bucket["x"].append(norm_x)
            bucket["z"].append(norm_z)

        high_mean = self._safe_mean(row_groups["high"], 0.85)
        mid_mean = self._safe_mean(row_groups["mid"], 0.5)
        low_mean = self._safe_mean(row_groups["low"], 0.15)

        self.zone_high_ratio = min(max((high_mean + mid_mean) / 2.0, 0.45), 0.95)
        self.zone_low_ratio = min(
            max((mid_mean + low_mean) / 2.0, 0.05),
            self.zone_high_ratio - 0.05,
        )

        left_mean = self._safe_mean(col_groups["left"], -0.5)
        mid_mean = self._safe_mean(col_groups["mid"], 0.0)
        right_mean = self._safe_mean(col_groups["right"], 0.5)
        left_boundary = (left_mean + mid_mean) / 2.0
        right_boundary = (right_mean + mid_mean) / 2.0

        self.zone_left_boundary = min(left_boundary * plate_half, -0.05)
        self.zone_right_boundary = max(right_boundary * plate_half, 0.05)

        profiles: Dict[int, Dict[str, float]] = {}
        for zone, values in profile_accum.items():
            x_mean = self._safe_mean(values["x"], 0.0)
            z_mean = self._safe_mean(values["z"], 0.5)
            x_scale = max(self._std(values["x"]), 0.05)
            z_scale = max(self._std(values["z"]), 0.05)
            profiles[zone] = {
                "x_mean": x_mean,
                "z_mean": z_mean,
                "x_scale": x_scale,
                "z_scale": z_scale,
            }
        self.zone_profiles = profiles

        print(
            "    Zone grid ratios:",
            f"low_cut={self.zone_low_ratio:.3f}, high_cut={self.zone_high_ratio:.3f}, ",
            f"left={self.zone_left_boundary:.3f} ft, right={self.zone_right_boundary:.3f} ft",
        )

    @staticmethod
    def _safe_mean(values: Sequence[float], default: float) -> float:
        return sum(values) / len(values) if values else default

    @staticmethod
    def _std(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.3
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(max(variance, 0.0))

    def _tune_classification_window(
        self,
        rows: Sequence[Row],
        predicted_crossings: Sequence[Tuple[float, float]],
    ) -> None:
        x_scales = [0.95, 1.0, 1.05, 1.1]
        top_offsets = [-0.1, -0.05, 0.0, 0.05, 0.1]
        bottom_offsets = [-0.1, -0.05, 0.0, 0.05, 0.1]

        best_score = -1.0
        best_params = (self.zone_x_scale, self.zone_top_offset, self.zone_bottom_offset)

        for xs in x_scales:
            for top in top_offsets:
                for bot in bottom_offsets:
                    self.zone_x_scale = xs
                    self.zone_top_offset = top
                    self.zone_bottom_offset = bot
                    score = self._score_predictions(rows, predicted_crossings)
                    if score > best_score:
                        best_score = score
                        best_params = (xs, top, bot)

        self.zone_x_scale, self.zone_top_offset, self.zone_bottom_offset = best_params
        print(
            "    Strike window tuned to "
            f"x_scale={self.zone_x_scale:.2f}, top_offset={self.zone_top_offset:.2f}, "
            f"bottom_offset={self.zone_bottom_offset:.2f}"
        )

    def _score_predictions(self, rows: Sequence[Row], predicted_crossings: Sequence[Tuple[float, float]]) -> float:
        class_hits = 0
        zone_hits = 0
        total = len(rows)

        for row, (pred_x, pred_z) in zip(rows, predicted_crossings):
            pitch_class = self.classify_pitch(pred_x, pred_z, float(row["sz_top"]), float(row["sz_bot"]), ctx=None)
            zone = self.predict_zone(pred_x, pred_z, float(row["sz_top"]), float(row["sz_bot"]))
            if pitch_class == row["pitch_class"]:
                class_hits += 1
            if int(zone) == int(row["zone"]):
                zone_hits += 1

        class_accuracy = class_hits / total
        zone_accuracy = zone_hits / total
        return 0.7 * class_accuracy + 0.3 * zone_accuracy

    def _fit_classifier(
        self,
        rows: Sequence[Row],
        predicted_crossings: Sequence[Tuple[float, float]],
        contexts: Sequence[Dict[str, float]],
    ) -> None:
        features = []
        targets = []
        for row, (pred_x, pred_z), ctx in zip(rows, predicted_crossings, contexts):
            features.append(self._classifier_features(row, pred_x, pred_z, ctx))
            targets.append(1.0 if row["pitch_class"] == "strike" else 0.0)

        if not features:
            self.classifier_weights = []
            return

        self.classifier_weights = least_squares(features, targets)
        scores = [self._apply_classifier(feat) for feat in features]
        sorted_scores = sorted(scores)
        best_threshold = 0.5
        best_acc = 0.0

        for thresh in sorted_scores:
            hits = 0
            for score, row in zip(scores, rows):
                pred = "strike" if score >= thresh else "ball"
                if pred == row["pitch_class"]:
                    hits += 1
            acc = hits / len(rows)
            if acc > best_acc:
                best_acc = acc
                best_threshold = thresh

        self.classifier_threshold = best_threshold
        print(f"    Classifier tuned: threshold={best_threshold:.3f}, acc={best_acc:.3f}")

    def _classifier_features(self, row: Row, pred_x: float, pred_z: float, ctx: Dict[str, float]) -> List[float]:
        sz_top = float(row["sz_top"])
        sz_bot = float(row["sz_bot"])
        height = max(sz_top - sz_bot, 1e-3)
        sz_mid = (sz_top + sz_bot) / 2.0
        x_limit = (self.PLATE_WIDTH / 2.0 + self.BALL_RADIUS) * self.zone_x_scale
        norm_x = abs(pred_x) / max(x_limit, 1e-3)
        norm_z = (pred_z - sz_mid) / height
        return [
            abs(pred_x),
            norm_x,
            pred_z - sz_mid,
            norm_z,
            ctx.get("distance_to_plate", 0.0),
            ctx.get("stand_sign", 1.0),
            ctx.get("throws_sign", 1.0),
            ctx.get("pfx_x_ft", 0.0),
            ctx.get("pfx_z_ft", 0.0),
            ctx.get("effective_speed_fps", 0.0),
            ctx.get("raw_x", 0.0) - pred_x,
            ctx.get("raw_z", 0.0) - pred_z,
        ]

    def _apply_classifier(self, features: Sequence[float]) -> float:
        if not self.classifier_weights:
            return 0.0
        bias = self.classifier_weights[-1]
        return sum(w * f for w, f in zip(self.classifier_weights[:-1], features)) + bias

    def classify_pitch(
        self,
        plate_x: float,
        plate_z: float,
        sz_top: float,
        sz_bot: float,
        ctx: Dict[str, float] | None = None,
    ) -> str:
        if self.classifier_weights and ctx is not None:
            features = self._classifier_features(
                {"sz_top": sz_top, "sz_bot": sz_bot},
                plate_x,
                plate_z,
                ctx,
            )
            score = self._apply_classifier(features)
            return "strike" if score >= self.classifier_threshold else "ball"

        x_limit = (self.PLATE_WIDTH / 2.0 + self.BALL_RADIUS) * self.zone_x_scale
        z_lower = sz_bot - self.BALL_RADIUS + self.zone_bottom_offset
        z_upper = sz_top + self.BALL_RADIUS + self.zone_top_offset
        is_strike = (abs(plate_x) <= x_limit) and (z_lower <= plate_z <= z_upper)
        return "strike" if is_strike else "ball"

    def predict_zone(self, plate_x: float, plate_z: float, sz_top: float, sz_bot: float) -> int:
        if self.zone_profiles:
            plate_half = self.PLATE_WIDTH / 2.0
            norm_x = plate_x / plate_half
            height = max(sz_top - sz_bot, 1e-3)
            norm_z = (plate_z - sz_bot) / height
            best_zone = None
            best_dist = float("inf")
            for zone, profile in self.zone_profiles.items():
                dx = (norm_x - profile["x_mean"]) / profile["x_scale"]
                dz = (norm_z - profile["z_mean"]) / profile["z_scale"]
                dist = dx * dx + dz * dz
                if dist < best_dist:
                    best_dist = dist
                    best_zone = zone
            if best_zone is not None:
                return int(best_zone)

        x_limit = (self.PLATE_WIDTH / 2.0 + self.BALL_RADIUS) * self.zone_x_scale
        z_lower = sz_bot - self.BALL_RADIUS + self.zone_bottom_offset
        z_upper = sz_top + self.BALL_RADIUS + self.zone_top_offset
        height = max(sz_top - sz_bot, 1e-3)
        low_cut = sz_bot + height * self.zone_low_ratio
        high_cut = sz_bot + height * self.zone_high_ratio

        if (abs(plate_x) <= x_limit) and (z_lower <= plate_z <= z_upper):
            if plate_z >= high_cut:
                row_idx = 0
            elif plate_z >= low_cut:
                row_idx = 1
            else:
                row_idx = 2

            if plate_x < self.zone_left_boundary:
                col_idx = 0
            elif plate_x > self.zone_right_boundary:
                col_idx = 2
            else:
                col_idx = 1

            return row_idx * 3 + col_idx + 1

        if plate_z > z_upper:
            if plate_x < -x_limit:
                return 11
            if plate_x > x_limit:
                return 13
            return 12

        if plate_z < z_lower:
            return 14

        return 11 if plate_x < 0 else 13

    def predict_dataset(self, rows: Sequence[Row]) -> List[Row]:
        predictions: List[Row] = []
        for row in rows:
            ctx = self._physics_context(row)
            pred_x, pred_z = self._predict_from_context(row, ctx)
            pitch_class = self.classify_pitch(pred_x, pred_z, float(row["sz_top"]), float(row["sz_bot"]), ctx)
            zone = self.predict_zone(pred_x, pred_z, float(row["sz_top"]), float(row["sz_bot"]))
            predictions.append({"file_name": row["file_name"], "pitch_class": pitch_class, "zone": int(zone)})
        return predictions


def main() -> None:
    data_dir = Path("./data")
    train_file = data_dir / "train_ground_truth.csv"
    test_file = data_dir / "test_features.csv"
    output_file = Path("./naive_baseline_submission.csv")

    print("=" * 60)
    print("Naive Physics-Based Predictor (refined stdlib edition)")
    print("=" * 60)

    print("\n[1/5] Loading data without pandas...")
    train_rows = load_rows(train_file, expect_labels=True)
    test_rows = load_rows(test_file, expect_labels=False)
    print(f"  Training samples: {len(train_rows)}")
    print(f"  Test samples: {len(test_rows)}")

    print("\n[2/5] Initializing predictor...")
    predictor = NaivePhysicsPredictor()

    print("\n[3/5] Calibrating against Statcast targets...")
    predictor.calibrate(train_rows)

    print("\n[4/5] Evaluating on training set (sanity check)...")
    train_predictions = predictor.predict_dataset(train_rows)
    metrics = evaluate_predictions(train_rows, train_predictions)
    print(f"  Class Accuracy: {metrics['class_accuracy']:.4f}")
    print(f"  Zone Accuracy: {metrics['zone_accuracy']:.4f}")
    print(f"  Combined Score: {metrics['combined_score']:.4f}")

    print("\n[5/5] Generating test predictions...")
    test_predictions = predictor.predict_dataset(test_rows)
    write_submission(output_file, test_predictions)
    print(f"  Submission saved to: {output_file}")
    print(f"  Total predictions: {len(test_predictions)}")

    print("\nSample predictions:")
    for row in test_predictions[:10]:
        print(f"  {row['file_name']}: {row['pitch_class']} (zone {row['zone']})")

    print("\n✓ Baseline complete!")


if __name__ == "__main__":
    main()
