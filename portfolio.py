from __future__ import annotations

import numpy as np
import pandas as pd

MIN_STOCKS = 30
MAX_WEIGHT = 0.10
MAX_WEIGHT_ITERATIONS = 50


def _filter_scores(
    scores: pd.Series,
    features: pd.DataFrame | None = None,
    min_turnover: float | None = None,
    max_vol: float | None = None,
    max_vol_quantile: float | None = None,
) -> pd.Series:
    if features is None:
        return scores
    if "stock_code" in features.columns:
        features = features.set_index("stock_code")
    aligned = features.reindex(scores.index)
    mask = pd.Series(True, index=scores.index)
    if min_turnover is not None and "turnover_ma_20d" in aligned.columns:
        mask &= aligned["turnover_ma_20d"] >= min_turnover
    if max_vol is not None and "vol_20d" in aligned.columns:
        mask &= aligned["vol_20d"] <= max_vol
    if max_vol_quantile is not None and "vol_20d" in aligned.columns:
        threshold = aligned["vol_20d"].quantile(max_vol_quantile)
        mask &= aligned["vol_20d"] <= threshold
    filtered = scores[mask]
    if len(filtered) < MIN_STOCKS:
        return scores
    return filtered


def build_portfolio(
    scores: pd.Series,
    top_k: int,
    min_stocks: int = MIN_STOCKS,
    max_weight: float = MAX_WEIGHT,
    features: pd.DataFrame | None = None,
    min_turnover: float | None = None,
    max_vol: float | None = None,
    max_vol_quantile: float | None = None,
) -> pd.Series:
    if top_k < min_stocks:
        raise ValueError(f"top_k must be >= {min_stocks}")
    filtered_scores = _filter_scores(
        scores,
        features=features,
        min_turnover=min_turnover,
        max_vol=max_vol,
        max_vol_quantile=max_vol_quantile,
    )
    chosen = filtered_scores.sort_values(ascending=False).head(top_k).copy()
    ranks = np.arange(len(chosen), 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(MAX_WEIGHT_ITERATIONS):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    if (w > 0).sum() < min_stocks:
        raise ValueError("too few names after weighting")
    if (w > max_weight + 1e-9).any():
        raise ValueError("weight cap violated")
    w = w / w.sum()
    return w
