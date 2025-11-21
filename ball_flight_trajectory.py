from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


HOME_PLATE_WIDTH_FT = 17 / 12  # 17 inches
PITCHING_RUBBER_WIDTH_FT = 24 / 12  # 24 inches
PLATE_TO_RUBBER_FT = 60.5
DEFAULT_FT_PER_PX = 1 / 75  # coarse prior; replace with camera-specific calibration where possible


@dataclass
class LandmarkMeasurement:
    frame_index: int
    center: tuple[float, float]
    width_px: float
    height_px: float
    confidence: float


def to_landmark(row: pd.Series) -> LandmarkMeasurement:
    width_px = float(row["x_max"] - row["x_min"])
    height_px = float(row["y_max"] - row["y_min"])
    center = (
        float((row["x_min"] + row["x_max"]) / 2),
        float((row["y_min"] + row["y_max"]) / 2),
    )
    return LandmarkMeasurement(
        frame_index=int(row["frame_index"]),
        center=center,
        width_px=width_px,
        height_px=height_px,
        confidence=float(row["confidence"]),
    )


def pick_highest_confidence(
    frames: list[Optional[LandmarkMeasurement]], measurement: LandmarkMeasurement
) -> None:
    existing = frames[measurement.frame_index]
    if existing is None or existing.confidence < measurement.confidence:
        frames[measurement.frame_index] = measurement


def infer_ft_per_px(
    homeplate: Optional[LandmarkMeasurement],
    rubber: Optional[LandmarkMeasurement],
    fallback_scale: float,
) -> Optional[float]:
    scales: list[float] = []
    if homeplate is not None and homeplate.width_px > 0:
        scales.append(HOME_PLATE_WIDTH_FT / homeplate.width_px)
    if rubber is not None and rubber.width_px > 0:
        scales.append(PITCHING_RUBBER_WIDTH_FT / rubber.width_px)
    if scales:
        return float(np.median(scales))
    return fallback_scale


def project_to_field_space(
    baseball: LandmarkMeasurement,
    homeplate: Optional[LandmarkMeasurement],
    rubber: Optional[LandmarkMeasurement],
    ft_per_px: float,
) -> tuple[float, float]:
    if homeplate is not None:
        origin_px = homeplate.center
        z_offset_ft = 0.0
    elif rubber is not None:
        origin_px = rubber.center
        z_offset_ft = -PLATE_TO_RUBBER_FT
    else:
        raise ValueError("Need at least one static landmark to anchor projection.")

    u_delta = baseball.center[0] - origin_px[0]
    v_delta = origin_px[1] - baseball.center[1]

    lateral_ft = u_delta * ft_per_px
    depth_ft = z_offset_ft + v_delta * ft_per_px
    return lateral_ft, depth_ft


def iter_projected_positions(
    baseball_frames: list[Optional[LandmarkMeasurement]],
    homeplate_frames: list[Optional[LandmarkMeasurement]],
    rubber_frames: list[Optional[LandmarkMeasurement]],
    fallback_scale: float,
) -> Iterable[tuple[int, float, float]]:
    last_scale = fallback_scale
    for frame_idx, baseball in enumerate(baseball_frames):
        if baseball is None:
            continue

        homeplate = homeplate_frames[frame_idx]
        rubber = rubber_frames[frame_idx]

        scale = infer_ft_per_px(homeplate, rubber, last_scale)
        if scale is None:
            continue

        try:
            x_ft, z_ft = project_to_field_space(baseball, homeplate, rubber, scale)
        except ValueError:
            continue

        last_scale = scale
        yield frame_idx, x_ft, z_ft


def main() -> None:
    detections_path = Path("ball_detections_test.csv")
    df = pd.read_csv(detections_path)
    grouped = df.groupby("file_name")

    for file_name, group in grouped:
        n_frames = int(group["frame_index"].max()) + 1
        baseball_frames: list[Optional[LandmarkMeasurement]] = [None] * n_frames
        homeplate_frames: list[Optional[LandmarkMeasurement]] = [None] * n_frames
        rubber_frames: list[Optional[LandmarkMeasurement]] = [None] * n_frames

        for _, row in group.iterrows():
            measurement = to_landmark(row)
            if row["class_name"] == "baseball":
                pick_highest_confidence(baseball_frames, measurement)
            elif row["class_name"] == "homeplate":
                pick_highest_confidence(homeplate_frames, measurement)
            elif row["class_name"] == "rubber":
                pick_highest_confidence(rubber_frames, measurement)

        projected_positions = list(
            iter_projected_positions(
                baseball_frames,
                homeplate_frames,
                rubber_frames,
                fallback_scale=DEFAULT_FT_PER_PX,
            )
        )
        print(f"{file_name}: projected {len(projected_positions)} samples to feet")

