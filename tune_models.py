from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMN, build_features, training_frame
from modeling import predict_rnn, predict_tft, train_rnn_model, train_tft_model
from preprocessing import rolling_zscore
from sequence_data import build_tft_datasets, make_rnn_sequences
from validation import evaluate_predictions, rolling_windows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/prices.parquet")
    p.add_argument("--model", choices=["rnn", "tft"], default="rnn")
    p.add_argument("--train-days", type=int, default=252)
    p.add_argument("--val-days", type=int, default=20)
    p.add_argument("--embargo", type=int, default=5)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--stationary", action="store_true")
    p.add_argument("--stationary-window", type=int, default=60)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/optuna_best.json")
    args = p.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    panel = build_features(prices)
    if args.stationary:
        panel = rolling_zscore(panel, FEATURE_COLUMNS, window=args.stationary_window)
    df = training_frame(panel)

    windows = rolling_windows(
        df["date"].unique(),
        train_days=args.train_days,
        val_days=args.val_days,
        step=args.val_days,
        embargo=args.embargo,
    )
    if not windows:
        raise RuntimeError("Not enough dates for tuning windows")
    train_start, train_end, val_start, val_end = windows[0]

    def objective(trial: optuna.Trial) -> float:
        if args.model == "rnn":
            seq_x, seq_y, seq_dates, seq_codes = make_rnn_sequences(
                df,
                FEATURE_COLUMNS,
                TARGET_COLUMN,
                lookback=args.lookback,
            )
            train_mask = (seq_dates >= train_start.to_datetime64()) & (seq_dates <= train_end.to_datetime64())
            val_mask = (seq_dates >= val_start.to_datetime64()) & (seq_dates <= val_end.to_datetime64())
            model = train_rnn_model(
                seq_x[train_mask],
                seq_y[train_mask],
                seq_x[val_mask],
                seq_y[val_mask],
                rnn_type=trial.suggest_categorical("rnn_type", ["gru", "lstm"]),
                hidden_size=trial.suggest_int("hidden_size", 16, 128, step=16),
                num_layers=trial.suggest_int("num_layers", 1, 3),
                dropout=trial.suggest_float("dropout", 0.0, 0.4),
                lr=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
                epochs=trial.suggest_int("epochs", 5, 20),
                seed=args.seed,
            )
            preds = predict_rnn(model, seq_x[val_mask], batch_size=256)
            pred_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(seq_dates[val_mask]),
                    "stock_code": seq_codes[val_mask],
                    TARGET_COLUMN: seq_y[val_mask],
                    "score": preds,
                }
            )
            metrics = evaluate_predictions(pred_df, TARGET_COLUMN, top_k=50)
            return metrics["rank_ic"]

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
            hidden_size=trial.suggest_int("hidden_size", 16, 64, step=8),
            hidden_continuous_size=trial.suggest_int("hidden_continuous_size", 8, 32, step=4),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            attention_head_size=trial.suggest_int("attention_head_size", 2, 8, step=2),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256]),
            max_epochs=trial.suggest_int("epochs", 5, 15),
            seed=args.seed,
        )
        preds, index = predict_tft(model, validation, batch_size=256)
        reverse_mapping = {v: k for k, v in mapping.items()}
        pred_index = index.copy()
        pred_index["date"] = pred_index["time_idx"].map(reverse_mapping)
        pred_index["score"] = preds
        pred_df = pred_index.merge(
            full_df[["stock_code", "date", TARGET_COLUMN]],
            on=["stock_code", "date"],
            how="left",
        )
        pred_df = pred_df[(pred_df["date"] >= val_start) & (pred_df["date"] <= val_end)]
        metrics = evaluate_predictions(pred_df, TARGET_COLUMN, top_k=50)
        return metrics["rank_ic"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(study.best_params, indent=2))
    print(study.best_params)


if __name__ == "__main__":
    main()
