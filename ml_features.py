from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE_NUMERIC_FEATURES: list[str] = [
    "sz_top",
    "sz_bot",
    "sz_height",
    "sz_mid",
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "total_movement",
    "movement_angle",
    "release_dist",
    "release_height_ratio",
    "speed_squared",
    "spin_per_mph",
]

OBJECT_CLASSES: tuple[str, ...] = ("baseball", "homeplate", "rubber")
SEGMENT_FIELDS: list[str] = ["x_min", "y_min", "x_max", "y_max", "confidence"]

COUNT_FEATURES: list[str] = [f"{obj}_count" for obj in OBJECT_CLASSES]

BASEBALL_POSITION_FEATURES: list[str] = [
    "baseball_start_frame",
    "baseball_end_frame",
    "baseball_start_x",
    "baseball_start_y",
    "baseball_end_x",
    "baseball_end_y",
]

TRAJECTORY_STAT_FEATURES: list[str] = [
    "trajectory_frames",
    "trajectory_duration",
    "trajectory_coverage",
    "baseball_cx_mean",
    "baseball_cy_mean",
    "baseball_cx_std",
    "baseball_cy_std",
    "baseball_cx_min",
    "baseball_cx_max",
    "baseball_cy_min",
    "baseball_cy_max",
    "baseball_dx_total",
    "baseball_dy_total",
    "baseball_displacement",
]

VELOCITY_FEATURES: list[str] = [
    "velocity_x_mean",
    "velocity_y_mean",
    "velocity_x_std",
    "velocity_y_std",
    "speed_mean",
    "speed_max",
    "speed_min",
    "velocity_x_start",
    "velocity_y_start",
    "velocity_x_end",
    "velocity_y_end",
    "accel_x_mean",
    "accel_y_mean",
    "accel_x_std",
    "accel_y_std",
]

POLYNOMIAL_FEATURES: list[str] = [
    "poly_x_a",
    "poly_x_b",
    "poly_y_a",
    "poly_y_b",
    "poly_x_residual",
    "poly_y_residual",
    "poly_x_at_end",
    "poly_y_at_end",
    "poly_x_extrapolate",
    "poly_y_extrapolate",
]

BALL_SIZE_FEATURES: list[str] = [
    "ball_width_mean",
    "ball_width_start",
    "ball_width_end",
    "ball_width_change",
    "ball_width_ratio",
]

CONFIDENCE_FEATURES: list[str] = [
    "baseball_conf_mean",
    "baseball_conf_min",
    "baseball_conf_end",
]

NORMALISED_FEATURES: list[str] = [
    "baseball_start_x_norm",
    "baseball_start_y_norm",
    "baseball_end_x_norm",
    "baseball_end_y_norm",
    "baseball_cx_mean_norm",
    "baseball_cy_mean_norm",
    "poly_x_at_end_norm",
    "poly_y_at_end_norm",
]

ROTATED_FEATURES: list[str] = [
    "baseball_start_x_rot",
    "baseball_start_y_rot",
    "baseball_end_x_rot",
    "baseball_end_y_rot",
    "poly_x_rot",
    "poly_y_rot",
]

LEGACY_KINEMATIC_FEATURES: list[str] = [
    "baseball_vx_px",
    "baseball_vy_px",
    "baseball_trajectory_length_px",
    "baseball_trajectory_angle",
    "baseball_curvature_px",
]

BASEBALL_FEATURES: list[str] = (
    BASEBALL_POSITION_FEATURES
    + TRAJECTORY_STAT_FEATURES
    + VELOCITY_FEATURES
    + POLYNOMIAL_FEATURES
    + BALL_SIZE_FEATURES
    + CONFIDENCE_FEATURES
    + NORMALISED_FEATURES
    + ROTATED_FEATURES
    + LEGACY_KINEMATIC_FEATURES
)

HOMEPLATE_MEAN_FEATURES: list[str] = [
    f"homeplate_{field}_mean" for field in SEGMENT_FIELDS
]
RUBBER_MEAN_FEATURES: list[str] = [f"rubber_{field}_mean" for field in SEGMENT_FIELDS]

CALIBRATION_FEATURES: list[str] = [
    "homeplate_width_px",
    "rubber_width_px",
    "plate_rubber_width_ratio",
    "estimated_ft_per_px",
    "baseball_dx_ft",
    "baseball_dy_ft",
]

DETECTION_FEATURES: list[str] = (
    COUNT_FEATURES
    + BASEBALL_FEATURES
    + HOMEPLATE_MEAN_FEATURES
    + RUBBER_MEAN_FEATURES
    + CALIBRATION_FEATURES
)

INTERACTION_FEATURES: list[str] = ["traj_pfx_x_diff", "traj_pfx_z_diff"]

NUMERIC_FEATURES: list[str] = (
    BASE_NUMERIC_FEATURES + DETECTION_FEATURES + INTERACTION_FEATURES
)

CATEGORICAL_FEATURES: list[str] = ["stand", "p_throws", "same_side"]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LOG_DIR = Path("logs")

PITCH_MAP = {"ball": 0, "strike": 1}
INV_PITCH_MAP = {v: k for k, v in PITCH_MAP.items()}


@dataclass
class FeaturePipeline:
    """Reusable feature engineering helper shared by different training scripts."""

    numeric_medians: pd.Series | None = None
    category_maps: dict[str, dict[str, int]] | None = None
    category_unknown_index: dict[str, int] | None = None

    def fit(self, *dfs: pd.DataFrame) -> FeaturePipeline:
        """Learn numeric medians and categorical vocabularies from provided frames."""
        if not dfs:
            raise ValueError("FeaturePipeline.fit() requires at least one dataframe.")

        combined = pd.concat(dfs, axis=0, ignore_index=True, copy=False)
        for col in ALL_FEATURES:
            if col not in combined.columns:
                combined[col] = np.nan

        numeric_block = combined[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        medians = numeric_block.median()
        self.numeric_medians = medians.fillna(0.0)

        self.category_maps = {}
        self.category_unknown_index = {}

        for col in CATEGORICAL_FEATURES:
            normalized = (
                combined[col]
                .fillna("missing")
                .astype(str)
                .str.strip()
                .replace("", "missing")
            )
            unique_vals = sorted({val for val in normalized if val != "missing"})
            unique_vals.append("missing")
            mapping = {value: idx for idx, value in enumerate(unique_vals)}
            self.category_maps[col] = mapping
            self.category_unknown_index[col] = len(mapping)

        return self

    def transform(
        self, df: pd.DataFrame, *, log_tag: str | None = None
    ) -> pd.DataFrame:
        """Apply learned statistics to a new frame."""
        if self.numeric_medians is None or self.category_maps is None:
            raise RuntimeError(
                "FeaturePipeline must be fitted before calling transform()."
            )

        frame = df.copy()
        for col in ALL_FEATURES:
            if col not in frame.columns:
                frame[col] = np.nan

        frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )
        frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].fillna(self.numeric_medians)

        for col in CATEGORICAL_FEATURES:
            normalized = (
                frame[col]
                .fillna("missing")
                .astype(str)
                .str.strip()
                .replace("", "missing")
            )
            mapped = normalized.map(self.category_maps[col]).fillna(
                self.category_unknown_index[col]
            )
            frame[col] = mapped.astype(np.int16)

        result = frame[ALL_FEATURES].copy()
        self._log_snapshot(result, log_tag or "transform")
        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        *extra_dfs: pd.DataFrame,
        log_tag: str | None = None,
    ) -> pd.DataFrame:
        """Convenience helper for fitting on multiple frames before transforming the first."""
        return self.fit(df, *extra_dfs).transform(df, log_tag=log_tag)

    def _log_snapshot(self, frame: pd.DataFrame, tag: str) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_tag = tag.replace(" ", "_")
        path = LOG_DIR / f"ml_features_{safe_tag}_{timestamp}.csv"
        frame.head().to_csv(path, index=False)


def build_aligned_trajectory_features(
    meta_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    file_key: str = "file_name",
) -> pd.DataFrame:
    """
    Align raw object segmentation summaries (ball, homeplate, rubber) to the metadata frame.
    """
    if meta_df.empty:
        return pd.DataFrame(columns=DETECTION_FEATURES)

    summary = build_detection_feature_frame(detections_df, file_key=file_key)
    if summary.empty:
        aligned = pd.DataFrame(columns=DETECTION_FEATURES, index=meta_df.index)
    else:
        aligned = (
            summary.set_index(file_key)
            .reindex(meta_df[file_key])
            .reset_index(drop=True)
        )
        aligned = aligned.reindex(columns=DETECTION_FEATURES)

    return aligned.fillna(0.0)


def build_detection_feature_frame(
    detections: pd.DataFrame,
    file_key: str = "file_name",
) -> pd.DataFrame:
    """
    Summarise the original object segmentations (ball, homeplate, rubber) without conversions.
    """
    required_cols = {file_key, "class_name", "frame_index"} | set(SEGMENT_FIELDS)
    if detections.empty or not required_cols.issubset(detections.columns):
        return pd.DataFrame(columns=[file_key] + DETECTION_FEATURES)

    detections = detections[detections["class_name"].isin(OBJECT_CLASSES)].copy()
    if detections.empty:
        return pd.DataFrame(columns=[file_key] + DETECTION_FEATURES)

    grouped = detections.groupby(file_key, sort=False)
    rows: list[dict[str, float]] = []
    for file_name, group in grouped:
        row: dict[str, float | str] = {file_key: file_name}
        for obj in OBJECT_CLASSES:
            subset = group[group["class_name"] == obj]
            row[f"{obj}_count"] = float(len(subset))
            if obj == "baseball":
                if len(subset):
                    row.update(_summarise_baseball_track(subset))
            else:
                row.update(_mean_segment_fields(obj, subset))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in HOMEPLATE_MEAN_FEATURES + RUBBER_MEAN_FEATURES:
        if col not in df:
            df[col] = 0.0

    required_cols = set(
        BASEBALL_POSITION_FEATURES
        + ["baseball_cx_mean", "baseball_cy_mean", "poly_x_at_end", "poly_y_at_end"]
    )
    for col in required_cols:
        if col not in df:
            df[col] = 0.0

    homeplate_center_x = (df["homeplate_x_min_mean"] + df["homeplate_x_max_mean"]) / 2.0
    homeplate_center_y = (df["homeplate_y_min_mean"] + df["homeplate_y_max_mean"]) / 2.0
    df["baseball_start_x_norm"] = df["baseball_start_x"] - homeplate_center_x
    df["baseball_start_y_norm"] = df["baseball_start_y"] - homeplate_center_y
    df["baseball_end_x_norm"] = df["baseball_end_x"] - homeplate_center_x
    df["baseball_end_y_norm"] = df["baseball_end_y"] - homeplate_center_y
    df["baseball_cx_mean_norm"] = df["baseball_cx_mean"] - homeplate_center_x
    df["baseball_cy_mean_norm"] = df["baseball_cy_mean"] - homeplate_center_y
    df["poly_x_at_end_norm"] = df["poly_x_at_end"] - homeplate_center_x
    df["poly_y_at_end_norm"] = df["poly_y_at_end"] - homeplate_center_y

    rubber_center_x = (df["rubber_x_min_mean"] + df["rubber_x_max_mean"]) / 2.0
    rubber_center_y = (df["rubber_y_min_mean"] + df["rubber_y_max_mean"]) / 2.0
    vector_x = homeplate_center_x - rubber_center_x
    vector_y = homeplate_center_y - rubber_center_y
    norm = np.sqrt(vector_x**2 + vector_y**2)
    cos_theta = np.divide(vector_x, norm, out=np.ones_like(vector_x), where=norm > 1e-6)
    sin_theta = np.divide(
        vector_y, norm, out=np.zeros_like(vector_y), where=norm > 1e-6
    )

    def rotate(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
        x_rot = x * cos_theta + y * sin_theta
        y_rot = -x * sin_theta + y * cos_theta
        return x_rot, y_rot

    start_x_rot, start_y_rot = rotate(
        df["baseball_start_x_norm"], df["baseball_start_y_norm"]
    )
    end_x_rot, end_y_rot = rotate(df["baseball_end_x_norm"], df["baseball_end_y_norm"])
    df["baseball_start_x_rot"] = start_x_rot
    df["baseball_start_y_rot"] = start_y_rot
    df["baseball_end_x_rot"] = end_x_rot
    df["baseball_end_y_rot"] = end_y_rot
    poly_x_rot, poly_y_rot = rotate(df["poly_x_at_end_norm"], df["poly_y_at_end_norm"])
    df["poly_x_rot"] = poly_x_rot
    df["poly_y_rot"] = poly_y_rot

    plate_width_px = df["homeplate_x_max_mean"] - df["homeplate_x_min_mean"]
    rubber_width_px = df["rubber_x_max_mean"] - df["rubber_x_min_mean"]
    df["homeplate_width_px"] = plate_width_px
    df["rubber_width_px"] = rubber_width_px
    with np.errstate(divide="ignore", invalid="ignore"):
        df["plate_rubber_width_ratio"] = np.divide(
            plate_width_px,
            rubber_width_px,
            out=np.zeros_like(plate_width_px),
            where=rubber_width_px != 0,
        )

    estimated_ft_per_px = (17.0 / 12.0) / np.where(
        plate_width_px > 1e-3, plate_width_px, np.nan
    )
    estimated_ft_per_px = np.nan_to_num(
        estimated_ft_per_px, nan=0.0, posinf=0.0, neginf=0.0
    )
    df["estimated_ft_per_px"] = estimated_ft_per_px
    df["baseball_dx_ft"] = df["baseball_dx_total"] * estimated_ft_per_px
    df["baseball_dy_ft"] = df["baseball_dy_total"] * estimated_ft_per_px

    ordered_columns = [file_key] + DETECTION_FEATURES
    return df.reindex(columns=ordered_columns).fillna(0.0)


def _mean_segment_fields(obj: str, subset: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in SEGMENT_FIELDS:
        key = f"{obj}_{field}_mean"
        if len(subset):
            result[key] = float(subset[field].mean())
        else:
            result[key] = 0.0
    return result


def _summarise_baseball_track(subset: pd.DataFrame) -> dict[str, float]:
    ordered = subset.sort_values("frame_index")
    frames = ordered["frame_index"].to_numpy(dtype=float)
    cx = ((ordered["x_min"] + ordered["x_max"]) / 2.0).to_numpy(dtype=float)
    cy = ((ordered["y_min"] + ordered["y_max"]) / 2.0).to_numpy(dtype=float)
    widths = (ordered["x_max"] - ordered["x_min"]).to_numpy(dtype=float)
    conf = ordered["confidence"].to_numpy(dtype=float)

    summary: dict[str, float] = {
        "baseball_start_frame": frames[0],
        "baseball_end_frame": frames[-1],
        "baseball_start_x": cx[0],
        "baseball_start_y": cy[0],
        "baseball_end_x": cx[-1],
        "baseball_end_y": cy[-1],
        "trajectory_frames": float(len(frames)),
    }

    frame_span = max(frames[-1] - frames[0], 0.0)
    summary["trajectory_duration"] = frame_span + 1.0
    summary["trajectory_coverage"] = summary["trajectory_frames"] / max(
        1.0, summary["trajectory_duration"]
    )

    summary["baseball_cx_mean"] = float(np.mean(cx))
    summary["baseball_cy_mean"] = float(np.mean(cy))
    summary["baseball_cx_std"] = float(np.std(cx)) if len(cx) > 1 else 0.0
    summary["baseball_cy_std"] = float(np.std(cy)) if len(cy) > 1 else 0.0
    summary["baseball_cx_min"] = float(np.min(cx))
    summary["baseball_cx_max"] = float(np.max(cx))
    summary["baseball_cy_min"] = float(np.min(cy))
    summary["baseball_cy_max"] = float(np.max(cy))

    dx_total = cx[-1] - cx[0]
    dy_total = cy[-1] - cy[0]
    displacement = math.hypot(dx_total, dy_total)
    summary["baseball_dx_total"] = float(dx_total)
    summary["baseball_dy_total"] = float(dy_total)
    summary["baseball_displacement"] = float(displacement)

    if frame_span > 0:
        summary["baseball_vx_px"] = float(dx_total / frame_span)
        summary["baseball_vy_px"] = float(dy_total / frame_span)
    else:
        summary["baseball_vx_px"] = 0.0
        summary["baseball_vy_px"] = 0.0
    summary["baseball_trajectory_length_px"] = float(displacement)
    summary["baseball_trajectory_angle"] = (
        float(math.atan2(dy_total, dx_total)) if displacement > 0 else 0.0
    )

    if len(frames) > 2 and displacement > 0:
        avg_x = float(np.mean(cx))
        avg_y = float(np.mean(cy))
        mid_x = (cx[0] + cx[-1]) / 2.0
        mid_y = (cy[0] + cy[-1]) / 2.0
        dist = math.hypot(mid_x - avg_x, mid_y - avg_y)
        summary["baseball_curvature_px"] = float(dist / displacement)
    else:
        summary["baseball_curvature_px"] = 0.0

    if len(frames) >= 2:
        dt = np.diff(frames)
        dt[dt == 0] = 1.0
        vx = np.diff(cx) / dt
        vy = np.diff(cy) / dt
        speed = np.sqrt(vx**2 + vy**2)
        summary["velocity_x_mean"] = float(np.mean(vx))
        summary["velocity_y_mean"] = float(np.mean(vy))
        summary["velocity_x_std"] = float(np.std(vx)) if len(vx) > 1 else 0.0
        summary["velocity_y_std"] = float(np.std(vy)) if len(vy) > 1 else 0.0
        summary["speed_mean"] = float(np.mean(speed))
        summary["speed_max"] = float(np.max(speed))
        summary["speed_min"] = float(np.min(speed))

        n_samples = max(1, min(3, len(vx)))
        summary["velocity_x_start"] = float(np.mean(vx[:n_samples]))
        summary["velocity_y_start"] = float(np.mean(vy[:n_samples]))
        summary["velocity_x_end"] = float(np.mean(vx[-n_samples:]))
        summary["velocity_y_end"] = float(np.mean(vy[-n_samples:]))

        if len(vx) >= 2:
            ax = np.diff(vx)
            ay = np.diff(vy)
            summary["accel_x_mean"] = float(np.mean(ax))
            summary["accel_y_mean"] = float(np.mean(ay))
            summary["accel_x_std"] = float(np.std(ax)) if len(ax) > 1 else 0.0
            summary["accel_y_std"] = float(np.std(ay)) if len(ay) > 1 else 0.0
        else:
            summary["accel_x_mean"] = 0.0
            summary["accel_y_mean"] = 0.0
            summary["accel_x_std"] = 0.0
            summary["accel_y_std"] = 0.0
    else:
        summary["velocity_x_mean"] = 0.0
        summary["velocity_y_mean"] = 0.0
        summary["velocity_x_std"] = 0.0
        summary["velocity_y_std"] = 0.0
        summary["speed_mean"] = 0.0
        summary["speed_max"] = 0.0
        summary["speed_min"] = 0.0
        summary["velocity_x_start"] = 0.0
        summary["velocity_y_start"] = 0.0
        summary["velocity_x_end"] = 0.0
        summary["velocity_y_end"] = 0.0
        summary["accel_x_mean"] = 0.0
        summary["accel_y_mean"] = 0.0
        summary["accel_x_std"] = 0.0
        summary["accel_y_std"] = 0.0

    if len(frames) >= 3:
        try:
            t_norm = (frames - frames[0]) / max(1.0, frames[-1] - frames[0])
            poly_x = np.polyfit(t_norm, cx, 2)
            poly_y = np.polyfit(t_norm, cy, 2)
            x_pred = np.polyval(poly_x, t_norm)
            y_pred = np.polyval(poly_y, t_norm)
            summary["poly_x_a"] = float(poly_x[0])
            summary["poly_x_b"] = float(poly_x[1])
            summary["poly_y_a"] = float(poly_y[0])
            summary["poly_y_b"] = float(poly_y[1])
            summary["poly_x_residual"] = float(np.mean(np.abs(cx - x_pred)))
            summary["poly_y_residual"] = float(np.mean(np.abs(cy - y_pred)))
            summary["poly_x_at_end"] = float(np.polyval(poly_x, 1.0))
            summary["poly_y_at_end"] = float(np.polyval(poly_y, 1.0))
            summary["poly_x_extrapolate"] = float(np.polyval(poly_x, 1.1))
            summary["poly_y_extrapolate"] = float(np.polyval(poly_y, 1.1))
        except Exception:
            summary["poly_x_a"] = 0.0
            summary["poly_x_b"] = 0.0
            summary["poly_y_a"] = 0.0
            summary["poly_y_b"] = 0.0
            summary["poly_x_residual"] = 0.0
            summary["poly_y_residual"] = 0.0
            summary["poly_x_at_end"] = float(cx[-1])
            summary["poly_y_at_end"] = float(cy[-1])
            summary["poly_x_extrapolate"] = summary["poly_x_at_end"]
            summary["poly_y_extrapolate"] = summary["poly_y_at_end"]
    else:
        summary["poly_x_a"] = 0.0
        summary["poly_x_b"] = 0.0
        summary["poly_y_a"] = 0.0
        summary["poly_y_b"] = 0.0
        summary["poly_x_residual"] = 0.0
        summary["poly_y_residual"] = 0.0
        summary["poly_x_at_end"] = float(cx[-1])
        summary["poly_y_at_end"] = float(cy[-1])
        summary["poly_x_extrapolate"] = summary["poly_x_at_end"]
        summary["poly_y_extrapolate"] = summary["poly_y_at_end"]

    summary["ball_width_mean"] = float(np.mean(widths))
    summary["ball_width_start"] = float(widths[0])
    summary["ball_width_end"] = float(widths[-1])
    summary["ball_width_change"] = float(widths[-1] - widths[0])
    denom = widths[0] if abs(widths[0]) > 1e-3 else 1e-3
    summary["ball_width_ratio"] = float(widths[-1] / denom)

    summary["baseball_conf_mean"] = float(np.mean(conf))
    summary["baseball_conf_min"] = float(np.min(conf))
    summary["baseball_conf_end"] = float(conf[-1])

    return summary


def augment_statcast_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived Statcast-style features used by the ML pipeline."""
    if df.empty:
        return df.copy()

    result = df.copy()

    if {"sz_top", "sz_bot"}.issubset(result.columns):
        result["sz_height"] = result["sz_top"] - result["sz_bot"]
        result["sz_mid"] = result["sz_bot"] + result["sz_height"] / 2.0
    else:
        result["sz_height"] = result.get("sz_height", 0.0)
        result["sz_mid"] = result.get("sz_mid", 0.0)

    if {"pfx_x", "pfx_z"}.issubset(result.columns):
        result["total_movement"] = np.sqrt(result["pfx_x"] ** 2 + result["pfx_z"] ** 2)
        result["movement_angle"] = np.arctan2(result["pfx_z"], result["pfx_x"])
    else:
        result["total_movement"] = result.get("total_movement", 0.0)
        result["movement_angle"] = result.get("movement_angle", 0.0)

    if {"release_pos_x", "release_pos_y"}.issubset(result.columns):
        result["release_dist"] = np.sqrt(
            result["release_pos_x"] ** 2 + result["release_pos_y"] ** 2
        )
    else:
        result["release_dist"] = result.get("release_dist", 0.0)

    if {"release_pos_z", "sz_height"}.issubset(result.columns):
        denom = result["sz_height"].replace(0, np.nan)
        ratio = result["release_pos_z"] / denom
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["release_height_ratio"] = ratio
    else:
        result["release_height_ratio"] = result.get("release_height_ratio", 0.0)

    if "release_speed" in result.columns:
        result["speed_squared"] = result["release_speed"] ** 2
        if "release_spin_rate" in result.columns:
            spin_ratio = result["release_spin_rate"] / result["release_speed"].replace(
                0, np.nan
            )
            result["spin_per_mph"] = spin_ratio.replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        else:
            result["spin_per_mph"] = result.get("spin_per_mph", 0.0)
    else:
        result["speed_squared"] = result.get("speed_squared", 0.0)
        result["spin_per_mph"] = result.get("spin_per_mph", 0.0)

    if {"p_throws", "stand"}.issubset(result.columns):
        same_side = (
            result["p_throws"].fillna("").astype(str).str.upper()
            == result["stand"].fillna("").astype(str).str.upper()
        ).astype(int)
        result["same_side"] = same_side
    else:
        result["same_side"] = result.get("same_side", 0)

    return result


def add_statcast_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add blended Statcast + trajectory interaction features."""
    if df.empty:
        return df.copy()

    result = df.copy()

    if {"velocity_x_mean", "pfx_x"}.issubset(result.columns):
        result["traj_pfx_x_diff"] = (
            result["velocity_x_mean"] - result["pfx_x"]
        ).fillna(0.0)
    else:
        result["traj_pfx_x_diff"] = result.get("traj_pfx_x_diff", 0.0)

    if {"velocity_y_mean", "pfx_z"}.issubset(result.columns):
        result["traj_pfx_z_diff"] = (
            result["velocity_y_mean"] - result["pfx_z"]
        ).fillna(0.0)
    else:
        result["traj_pfx_z_diff"] = result.get("traj_pfx_z_diff", 0.0)

    return result
