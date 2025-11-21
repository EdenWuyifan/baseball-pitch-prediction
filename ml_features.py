from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMERIC_FEATURES: list[str] = [
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

CATEGORICAL_FEATURES: list[str] = ["stand", "p_throws"]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

        return frame[ALL_FEATURES].copy()

    def fit_transform(self, df: pd.DataFrame, *extra_dfs: pd.DataFrame) -> pd.DataFrame:
        """Convenience helper for fitting on multiple frames before transforming the first."""
        return self.fit(df, *extra_dfs).transform(df)
