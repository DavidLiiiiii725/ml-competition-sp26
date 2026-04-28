from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def build_scaler(kind: str):
    kind = kind.lower()
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"unknown scaler: {kind}")


def scale_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    scaler = build_scaler(kind)
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    train_out = train_df.copy()
    val_out = val_df.copy()
    train_out[feature_cols] = train_scaled
    val_out[feature_cols] = val_scaled
    return train_out, val_out, scaler


def rolling_zscore(
    panel: pd.DataFrame,
    feature_cols: list[str],
    window: int = 60,
) -> pd.DataFrame:
    panel = panel.sort_values(["stock_code", "date"]).copy()
    for col in feature_cols:
        def _zscore(s: pd.Series) -> pd.Series:
            mean = s.rolling(window).mean()
            std = s.rolling(window).std().replace(0, np.nan)
            return (s - mean) / std
        panel[col] = panel.groupby("stock_code")[col].transform(_zscore)
    return panel
