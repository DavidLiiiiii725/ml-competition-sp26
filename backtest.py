from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import EVALUATION_WINDOWS
from score_submission import score_window


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("submission", help="path to submission CSV")
    p.add_argument("--prices", default="data/prices.parquet")
    p.add_argument("--index", default="data/index.parquet")
    args = p.parse_args()

    sub = pd.read_csv(args.submission, dtype={"stock_code": str})
    sub["stock_code"] = sub["stock_code"].str.zfill(6)
    weights = sub.set_index("stock_code")["weight"].astype(float)

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    total_excess = 0.0
    for name, (start, end) in EVALUATION_WINDOWS.items():
        result = score_window(weights, prices, index_df, pd.Timestamp(start), pd.Timestamp(end))
        total_excess += result["excess_return"]
        print(f"{name}: {result['start']} to {result['end']}")
        print(f"  portfolio return : {result['portfolio_return']*100:+.3f}%")
        print(f"  benchmark return : {result['benchmark_return']*100:+.3f}%")
        print(f"  excess return    : {result['excess_return']*100:+.3f}%")
    print(f"Total excess (sum of windows): {total_excess*100:+.3f}%")


if __name__ == "__main__":
    main()
