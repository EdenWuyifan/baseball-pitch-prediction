from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_NUMERIC_FEATURES: list[str] = [
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

OBJECT_CLASSES: tuple[str, ...] = ("baseball", "homeplate", "rubber")
SEGMENT_FIELDS: list[str] = ["x_min", "y_min", "x_max", "y_max", "confidence"]

DETECTION_FEATURES: list[str] = []
for obj in OBJECT_CLASSES:
    DETECTION_FEATURES.append(f"{obj}_count")
    if obj == "baseball":
        DETECTION_FEATURES.extend(
            [
                "baseball_start_frame",
                "baseball_start_x",
                "baseball_start_y",
                "baseball_end_frame",
                "baseball_end_x",
                "baseball_end_y",
                "baseball_start_x_norm",
                "baseball_start_y_norm",
                "baseball_end_x_norm",
                "baseball_end_y_norm",
                "baseball_start_x_rot",
                "baseball_start_y_rot",
                "baseball_end_x_rot",
                "baseball_end_y_rot",
            ]
        )
    else:
        for field in SEGMENT_FIELDS:
            DETECTION_FEATURES.append(f"{obj}_{field}_mean")
DETECTION_FEATURES.extend(
    [
        "homeplate_width_px",
        "rubber_width_px",
        "plate_rubber_width_ratio",
        "estimated_ft_per_px",
        "baseball_dx_ft",
        "baseball_dy_ft",
    ]
)

NUMERIC_FEATURES: list[str] = BASE_NUMERIC_FEATURES + DETECTION_FEATURES

CATEGORICAL_FEATURES: list[str] = ["stand", "p_throws"]
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

    def transform(self, df: pd.DataFrame, *, log_tag: str | None = None) -> pd.DataFrame:
        """Apply learned statistics to a new frame."""
        if self.numeric_medians is None or self.category_maps is None:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform().")

        frame = df.copy()
        for col in ALL_FEATURES:
            if col not in frame.columns:
                frame[col] = np.nan

        frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
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
    required_cols = {file_key, "class_name"} | set(SEGMENT_FIELDS)
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
                    ordered = subset.sort_values("frame_index")
                    start = ordered.iloc[0]
                    end = ordered.iloc[-1]
                    row["baseball_start_frame"] = float(start["frame_index"])
                    row["baseball_end_frame"] = float(end["frame_index"])
                    row["baseball_start_x"] = float((start["x_min"] + start["x_max"]) / 2.0)
                    row["baseball_start_y"] = float((start["y_min"] + start["y_max"]) / 2.0)
                    row["baseball_end_x"] = float((end["x_min"] + end["x_max"]) / 2.0)
                    row["baseball_end_y"] = float((end["y_min"] + end["y_max"]) / 2.0)
                    # Normalized coordinates: subtract homeplate center (computed later, default 0).
                    row["baseball_start_x_norm"] = row["baseball_start_x"]
                    row["baseball_start_y_norm"] = row["baseball_start_y"]
                    row["baseball_end_x_norm"] = row["baseball_end_x"]
                    row["baseball_end_y_norm"] = row["baseball_end_y"]
                    row["baseball_start_x_rot"] = row["baseball_start_x"]
                    row["baseball_start_y_rot"] = row["baseball_start_y"]
                    row["baseball_end_x_rot"] = row["baseball_end_x"]
                    row["baseball_end_y_rot"] = row["baseball_end_y"]
                else:
                    row["baseball_start_frame"] = 0.0
                    row["baseball_end_frame"] = 0.0
                    row["baseball_start_x"] = 0.0
                    row["baseball_start_y"] = 0.0
                    row["baseball_end_x"] = 0.0
                    row["baseball_end_y"] = 0.0
                    row["baseball_start_x_norm"] = 0.0
                    row["baseball_start_y_norm"] = 0.0
                    row["baseball_end_x_norm"] = 0.0
                    row["baseball_end_y_norm"] = 0.0
                    row["baseball_start_x_rot"] = 0.0
                    row["baseball_start_y_rot"] = 0.0
                    row["baseball_end_x_rot"] = 0.0
                    row["baseball_end_y_rot"] = 0.0
            else:
                for field in SEGMENT_FIELDS:
                    key = f"{obj}_{field}_mean"
                    row[key] = float(subset[field].mean()) if len(subset) else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    homeplate_center_x = (df["homeplate_x_min_mean"] + df["homeplate_x_max_mean"]) / 2.0
    homeplate_center_y = (df["homeplate_y_min_mean"] + df["homeplate_y_max_mean"]) / 2.0
    df["baseball_start_x_norm"] = df["baseball_start_x"] - homeplate_center_x
    df["baseball_start_y_norm"] = df["baseball_start_y"] - homeplate_center_y
    df["baseball_end_x_norm"] = df["baseball_end_x"] - homeplate_center_x
    df["baseball_end_y_norm"] = df["baseball_end_y"] - homeplate_center_y

    rubber_center_x = (df["rubber_x_min_mean"] + df["rubber_x_max_mean"]) / 2.0
    rubber_center_y = (df["rubber_y_min_mean"] + df["rubber_y_max_mean"]) / 2.0
    vector_x = homeplate_center_x - rubber_center_x
    vector_y = homeplate_center_y - rubber_center_y
    norm = np.sqrt(vector_x ** 2 + vector_y ** 2)
    cos_theta = np.divide(vector_x, norm, out=np.ones_like(vector_x), where=norm > 1e-6)
    sin_theta = np.divide(vector_y, norm, out=np.zeros_like(vector_y), where=norm > 1e-6)

    def rotate(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
        x_rot = x * cos_theta + y * sin_theta
        y_rot = -x * sin_theta + y * cos_theta
        return x_rot, y_rot

    start_x_rot, start_y_rot = rotate(df["baseball_start_x_norm"], df["baseball_start_y_norm"])
    end_x_rot, end_y_rot = rotate(df["baseball_end_x_norm"], df["baseball_end_y_norm"])
    df["baseball_start_x_rot"] = start_x_rot
    df["baseball_start_y_rot"] = start_y_rot
    df["baseball_end_x_rot"] = end_x_rot
    df["baseball_end_y_rot"] = end_y_rot

    plate_width_px = df["homeplate_x_max_mean"] - df["homeplate_x_min_mean"]
    rubber_width_px = df["rubber_x_max_mean"] - df["rubber_x_min_mean"]
    df["homeplate_width_px"] = plate_width_px
    df["rubber_width_px"] = rubber_width_px
    with np.errstate(divide="ignore", invalid="ignore"):
        df["plate_rubber_width_ratio"] = np.divide(
            plate_width_px, rubber_width_px, out=np.zeros_like(plate_width_px), where=rubber_width_px != 0
        )

    estimated_ft_per_px = (17.0 / 12.0) / np.where(plate_width_px > 1e-3, plate_width_px, np.nan)
    estimated_ft_per_px = np.nan_to_num(estimated_ft_per_px, nan=0.0, posinf=0.0, neginf=0.0)
    df["estimated_ft_per_px"] = estimated_ft_per_px
    df["baseball_dx_ft"] = (df["baseball_end_x_norm"] - df["baseball_start_x_norm"]) * estimated_ft_per_px
    df["baseball_dy_ft"] = (df["baseball_end_y_norm"] - df["baseball_start_y_norm"]) * estimated_ft_per_px

    return df
