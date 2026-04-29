from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON, build_features, prediction_frame, training_frame
from modeling import predict_linear, predict_rnn, predict_tft, train_linear_model, train_rnn_model, train_tft_model
from portfolio import build_portfolio
from preprocessing import rolling_zscore
from sequence_data import build_tft_datasets, make_rnn_prediction_batch, make_rnn_sequences


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/prices.parquet")
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date")
    p.add_argument("--model", choices=["ridge", "lasso", "rnn", "tft"], default="ridge")
    p.add_argument("--scaler", default="standard")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--hidden-continuous-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--attention-head-size", type=int, default=4)
    p.add_argument("--stationary", action="store_true")
    p.add_argument("--stationary-window", type=int, default=60)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-turnover", type=float, default=None)
    p.add_argument("--max-vol", type=float, default=None)
    p.add_argument("--max-vol-quantile", type=float, default=None)
    p.add_argument("--out", default="submissions/model.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    panel = build_features(prices)
    if args.stationary:
        panel = rolling_zscore(panel, FEATURE_COLUMNS, window=args.stationary_window)

    as_of = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_df = training_frame(panel, max_date=train_cutoff)

    if args.model in {"ridge", "lasso"}:
        val_df = train_df.tail(1)
        model, scaler, _ = train_linear_model(
            train_df,
            val_df,
            FEATURE_COLUMNS,
            TARGET_COLUMN,
            model_type=args.model,
            scaler_kind=args.scaler,
            alpha=args.alpha,
        )
        pred_df = prediction_frame(panel, as_of=as_of)
        pred_df = pred_df.assign(score=predict_linear(model, scaler, pred_df, FEATURE_COLUMNS))
    elif args.model == "rnn":
        seq_x, seq_y, seq_dates, seq_codes = make_rnn_sequences(
            panel,
            FEATURE_COLUMNS,
            TARGET_COLUMN,
            lookback=args.lookback,
            max_date=train_cutoff.strftime("%Y-%m-%d"),
        )
        train_mask = seq_dates <= np.datetime64(train_cutoff)
        model = train_rnn_model(
            seq_x[train_mask],
            seq_y[train_mask],
            seq_x[train_mask],
            seq_y[train_mask],
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
        )
        pred_seq, pred_codes = make_rnn_prediction_batch(panel, FEATURE_COLUMNS, args.lookback, as_of=as_of)
        scores = predict_rnn(model, pred_seq, batch_size=args.batch_size)
        pred_df = pd.DataFrame({"stock_code": pred_codes, "score": scores})
    else:
        training, validation, full_df, mapping = build_tft_datasets(
            panel,
            FEATURE_COLUMNS,
            TARGET_COLUMN,
            max_encoder_length=args.lookback,
            max_prediction_length=1,
            cutoff_date=train_cutoff.strftime("%Y-%m-%d"),
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
        pred_df = pred_index[pred_index["date"] == as_of]

    if "date" in pred_df.columns:
        pred_df = pred_df[pred_df["date"] == as_of]
    if "stock_code" not in pred_df.columns:
        raise RuntimeError("prediction output missing stock_code")

    if args.model == "rnn":
        features = panel[panel["date"] == as_of]
        pred_df = pred_df.merge(features[["stock_code", "turnover_ma_20d", "vol_20d"]], on="stock_code", how="left")
    elif args.model == "tft":
        pred_df = pred_df.merge(
            full_df[["stock_code", "date", "turnover_ma_20d", "vol_20d"]],
            on=["stock_code", "date"],
            how="left",
        )

    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(
        scores,
        top_k=args.top_k,
        features=pred_df,
        min_turnover=args.min_turnover,
        max_vol=args.max_vol,
        max_vol_quantile=args.max_vol_quantile,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} names to {out_path}")


if __name__ == "__main__":
    main()
