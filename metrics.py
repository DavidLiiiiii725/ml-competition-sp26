from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252 / 5) -> float:
    returns = np.asarray(returns)
    if returns.size == 0:
        return float("nan")
    mean = np.nanmean(returns)
    std = np.nanstd(returns, ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(mean / std * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    returns = np.asarray(returns)
    if returns.size == 0:
        return float("nan")
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())
