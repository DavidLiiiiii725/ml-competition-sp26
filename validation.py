from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import max_drawdown, rank_ic, sharpe_ratio
from portfolio import build_portfolio


def rolling_windows(
    dates: np.ndarray,
    train_days: int,
    val_days: int,
    step: int,
    embargo: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows = []
    dates = np.sort(pd.to_datetime(dates))
    total = len(dates)
    start_idx = 0
    while start_idx + train_days + embargo + val_days <= total:
        train_start = dates[start_idx]
        train_end = dates[start_idx + train_days - 1]
        val_start = dates[start_idx + train_days + embargo]
        val_end = dates[start_idx + train_days + embargo + val_days - 1]
        windows.append((train_start, train_end, val_start, val_end))
        start_idx += step
    return windows


def evaluate_predictions(
    pred_df: pd.DataFrame,
    target_col: str,
    top_k: int,
    min_turnover: float | None = None,
    max_vol: float | None = None,
    max_vol_quantile: float | None = None,
) -> dict:
    pred_df = pred_df.dropna(subset=["score", target_col])
    ic = rank_ic(
        pred_df[target_col].to_numpy(),
        pred_df["score"].to_numpy(),
        pred_df["date"].to_numpy(),
    )

    returns = []
    for date, group in pred_df.groupby("date"):
        scores = group.set_index("stock_code")["score"]
        weights = build_portfolio(
            scores,
            top_k=top_k,
            features=group,
            min_turnover=min_turnover,
            max_vol=max_vol,
            max_vol_quantile=max_vol_quantile,
        )
        realized = float((weights * group.set_index("stock_code")[target_col]).sum())
        returns.append(realized)

    returns_arr = np.asarray(returns, dtype=float)
    return {
        "rank_ic": ic,
        "mean_return": float(np.nanmean(returns_arr)) if returns_arr.size else float("nan"),
        "sharpe": sharpe_ratio(returns_arr),
        "max_drawdown": max_drawdown(returns_arr),
        "n_dates": int(len(returns_arr)),
    }
