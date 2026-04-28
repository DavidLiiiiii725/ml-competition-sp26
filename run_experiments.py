from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMN, build_features, training_frame
from modeling import (
    predict_linear,
    predict_rnn,
    predict_tft,
    train_linear_model,
    train_rnn_model,
    train_tft_model,
)
from preprocessing import rolling_zscore
from sequence_data import build_tft_datasets, make_rnn_sequences
from validation import evaluate_predictions, rolling_windows


def load_panel(prices_path: str) -> pd.DataFrame:
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    return build_features(prices)


def run_linear_cv(panel: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = training_frame(panel)
    windows = rolling_windows(
        df["date"].unique(),
        train_days=args.train_days,
        val_days=args.val_days,
        step=args.step,
        embargo=args.embargo,
    )

    results = []
    coef_rows = []
    for scaler in args.scalers:
        for train_start, train_end, val_start, val_end in windows:
            train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
            val_df = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
            model, scaler_obj, val_pred = train_linear_model(
                train_df,
                val_df,
                FEATURE_COLUMNS,
                TARGET_COLUMN,
                model_type=args.model,
                scaler_kind=scaler,
                alpha=args.alpha,
            )
            pred_df = val_df.assign(score=val_pred)
            metrics = evaluate_predictions(
                pred_df,
                TARGET_COLUMN,
                top_k=args.top_k,
                min_turnover=args.min_turnover,
                max_vol=args.max_vol,
                max_vol_quantile=args.max_vol_quantile,
            )
            metrics.update(
                {
                    "model": args.model,
                    "scaler": scaler,
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "val_start": val_start.date().isoformat(),
                    "val_end": val_end.date().isoformat(),
                }
            )
            results.append(metrics)
            coef_rows.append(
                pd.DataFrame(
                    {
                        "feature": FEATURE_COLUMNS,
                        "coef": model.coef_,
                        "scaler": scaler,
                        "train_end": train_end.date().isoformat(),
                    }
                )
            )

    coef_df = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame()
    return pd.DataFrame(results), coef_df


def run_rnn_cv(panel: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.stationary:
        panel = rolling_zscore(panel, FEATURE_COLUMNS, window=args.stationary_window)
    seq_x, seq_y, seq_dates, seq_codes = make_rnn_sequences(
        panel,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        lookback=args.lookback,
    )
    windows = rolling_windows(
        np.unique(seq_dates),
        train_days=args.train_days,
        val_days=args.val_days,
        step=args.step,
        embargo=args.embargo,
    )
    results = []
    for train_start, train_end, val_start, val_end in windows:
        train_mask = (seq_dates >= np.datetime64(train_start)) & (seq_dates <= np.datetime64(train_end))
        val_mask = (seq_dates >= np.datetime64(val_start)) & (seq_dates <= np.datetime64(val_end))
        model = train_rnn_model(
            seq_x[train_mask],
            seq_y[train_mask],
            seq_x[val_mask],
            seq_y[val_mask],
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
        )
        preds = predict_rnn(model, seq_x[val_mask], batch_size=args.batch_size)
        pred_df = pd.DataFrame(
            {
                "date": pd.to_datetime(seq_dates[val_mask]),
                "stock_code": seq_codes[val_mask],
                TARGET_COLUMN: seq_y[val_mask],
                "score": preds,
            }
        )
        metrics = evaluate_predictions(
            pred_df,
            TARGET_COLUMN,
            top_k=args.top_k,
            min_turnover=args.min_turnover,
            max_vol=args.max_vol,
            max_vol_quantile=args.max_vol_quantile,
        )
        metrics.update(
            {
                "model": "rnn",
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "val_start": val_start.date().isoformat(),
                "val_end": val_end.date().isoformat(),
            }
        )
        results.append(metrics)
    return pd.DataFrame(results)


def run_tft_cv(panel: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.stationary:
        panel = rolling_zscore(panel, FEATURE_COLUMNS, window=args.stationary_window)
    df = training_frame(panel)
    windows = rolling_windows(
        df["date"].unique(),
        train_days=args.train_days,
        val_days=args.val_days,
        step=args.step,
        embargo=args.embargo,
    )
    results = []
    for train_start, train_end, val_start, val_end in windows:
        training, validation, full_df, mapping = build_tft_datasets(
            df,
            FEATURE_COLUMNS,
            TARGET_COLUMN,
            max_encoder_length=args.lookback,
            max_prediction_length=1,
            cutoff_date=train_end.strftime("%Y-%m-%d"),
        )
        model = train_tft_model(
            training,
            validation,
            hidden_size=args.hidden_size,
            hidden_continuous_size=args.hidden_continuous_size,
            dropout=args.dropout,
            attention_head_size=args.attention_head_size,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            seed=args.seed,
        )
        preds, index = predict_tft(model, validation, batch_size=args.batch_size)
        reverse_mapping = {v: k for k, v in mapping.items()}
        pred_index = index.copy()
        pred_index["date"] = pred_index["time_idx"].map(reverse_mapping)
        pred_index["score"] = preds
        pred_df = pred_index.merge(
            full_df[["stock_code", "date", TARGET_COLUMN, "turnover_ma_20d", "vol_20d"]],
            on=["stock_code", "date"],
            how="left",
            suffixes=("", "_feature"),
        )
        for col in ["turnover_ma_20d", "vol_20d"]:
            feature_col = f"{col}_feature"
            if feature_col in pred_df.columns:
                if col in pred_df.columns:
                    pred_df[col] = pred_df[col].fillna(pred_df[feature_col])
                else:
                    pred_df[col] = pred_df[feature_col]
                pred_df = pred_df.drop(columns=[feature_col])
        pred_df = pred_df[(pred_df["date"] >= val_start) & (pred_df["date"] <= val_end)]
        metrics = evaluate_predictions(
            pred_df,
            TARGET_COLUMN,
            top_k=args.top_k,
            min_turnover=args.min_turnover,
            max_vol=args.max_vol,
            max_vol_quantile=args.max_vol_quantile,
        )
        metrics.update(
            {
                "model": "tft",
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "val_start": val_start.date().isoformat(),
                "val_end": val_end.date().isoformat(),
            }
        )
        results.append(metrics)
    return pd.DataFrame(results)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/prices.parquet")
    p.add_argument("--model", choices=["ridge", "lasso", "rnn", "tft"], default="ridge")
    p.add_argument("--scalers", default="standard")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--train-days", type=int, default=252)
    p.add_argument("--val-days", type=int, default=20)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--embargo", type=int, default=5)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-turnover", type=float, default=None)
    p.add_argument("--max-vol", type=float, default=None)
    p.add_argument("--max-vol-quantile", type=float, default=None)
    p.add_argument("--stationary", action="store_true")
    p.add_argument("--stationary-window", type=int, default=60)
    p.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--hidden-continuous-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--attention-head-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/metrics.csv")
    p.add_argument("--coef-out", default="outputs/coefficients.csv")
    args = p.parse_args()

    args.scalers = [s.strip() for s in args.scalers.split(",") if s.strip()]
    panel = load_panel(args.prices)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.model in {"ridge", "lasso"}:
        metrics_df, coef_df = run_linear_cv(panel, args)
        metrics_df.to_csv(out_path, index=False)
        if not coef_df.empty:
            coef_out = Path(args.coef_out)
            coef_out.parent.mkdir(parents=True, exist_ok=True)
            coef_df.to_csv(coef_out, index=False)
    elif args.model == "rnn":
        metrics_df = run_rnn_cv(panel, args)
        metrics_df.to_csv(out_path, index=False)
    else:
        metrics_df = run_tft_cv(panel, args)
        metrics_df.to_csv(out_path, index=False)

    if not metrics_df.empty:
        ic_summary = metrics_df.groupby("model")["rank_ic"].mean()
        print("Rank IC summary:")
        print(ic_summary)


if __name__ == "__main__":
    main()
