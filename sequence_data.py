from __future__ import annotations

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def make_rnn_sequences(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    min_date: str | None = None,
    max_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = panel.dropna(subset=feature_cols + [target_col]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    df = df.sort_values(["stock_code", "date"])

    sequences = []
    targets = []
    dates = []
    codes = []
    for code, grp in df.groupby("stock_code"):
        values = grp[feature_cols].to_numpy()
        y = grp[target_col].to_numpy()
        date_arr = grp["date"].to_numpy()
        for idx in range(lookback - 1, len(grp)):
            seq = values[idx - lookback + 1: idx + 1]
            if np.isnan(seq).any() or np.isnan(y[idx]):
                continue
            sequences.append(seq)
            targets.append(y[idx])
            dates.append(date_arr[idx])
            codes.append(code)
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(dates),
        np.asarray(codes),
    )


def make_rnn_prediction_batch(
    panel: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    as_of: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    df = panel.copy()
    if as_of is None:
        as_of = df["date"].max()
    as_of = pd.Timestamp(as_of)
    df = df[df["date"] <= as_of].sort_values(["stock_code", "date"])

    sequences = []
    codes = []
    for code, grp in df.groupby("stock_code"):
        grp = grp.dropna(subset=feature_cols)
        if len(grp) < lookback:
            continue
        seq = grp[feature_cols].to_numpy()[-lookback:]
        if np.isnan(seq).any():
            continue
        sequences.append(seq)
        codes.append(code)
    return np.asarray(sequences, dtype=np.float32), np.asarray(codes)


def add_time_index(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[pd.Timestamp, int]]:
    panel = panel.copy()
    dates = np.sort(panel["date"].unique())
    mapping = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    panel["time_idx"] = panel["date"].map(mapping)
    return panel, mapping


def build_tft_datasets(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    max_encoder_length: int,
    max_prediction_length: int,
    cutoff_date: str | None = None,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame, dict[pd.Timestamp, int]]:
    df = panel.dropna(subset=feature_cols + [target_col]).copy()
    df, mapping = add_time_index(df)

    if cutoff_date is None:
        cutoff_idx = df["time_idx"].max() - max_prediction_length
    else:
        cutoff_idx = mapping[pd.Timestamp(cutoff_date)]

    training_data = df[df["time_idx"] <= cutoff_idx]

    training = TimeSeriesDataSet(
        training_data,
        time_idx="time_idx",
        target=target_col,
        group_ids=["stock_code"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=feature_cols + [target_col],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True,
    )
    return training, validation, df, mapping
