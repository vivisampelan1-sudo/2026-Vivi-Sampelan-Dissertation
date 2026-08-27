from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from asset_config import RAW_DIR, normalize_asset  # noqa: E402
from intraday_reopen_utils import (  # noqa: E402
    comex_week_reopen_utc,
    first_observation_at_or_after,
    last_observation_before,
)


RESULTS_DIR = DISS3_DIR / "05_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build separate intraday reopen events without touching daily outputs.")
    parser.add_argument("--asset", default="paxg", choices=["paxg", "xaut"])
    parser.add_argument(
        "--token-source",
        default="coingecko",
        choices=["coingecko", "yahoo"],
        help="Intraday token source used to build the event file.",
    )
    parser.add_argument(
        "--benchmark",
        default="gc_f_yahoo",
        choices=["xauusd_fx", "gc_f_yahoo"],
        help="Current benchmark source for the intraday prototype.",
    )
    return parser.parse_args()


def load_token(asset: str, token_source: str) -> pd.DataFrame:
    if token_source == "yahoo":
        path = RAW_DIR / f"{normalize_asset(asset)}_yahoo_60m.csv"
        price_col = "close"
    else:
        path = RAW_DIR / f"{normalize_asset(asset)}_coingecko_hourly.csv"
        price_col = "price"
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df.rename(columns={price_col: "token_price"})


def load_benchmark() -> pd.DataFrame:
    path = RAW_DIR / "xauusd_alphavantage_fx_60min.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df.rename(columns={"close": "benchmark_close", "open": "benchmark_open"})


def load_futures_benchmark() -> pd.DataFrame:
    path = RAW_DIR / "gc_f_yahoo_60m.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df.rename(columns={"close": "benchmark_close", "open": "benchmark_open"})


def observation_at_timestamp(frame: pd.DataFrame, ts_col: str, value_col: str, target_ts: pd.Timestamp) -> pd.Series | None:
    subset = frame.loc[frame[ts_col] == target_ts, [ts_col, value_col]].dropna()
    if subset.empty:
        return None
    return subset.sort_values(ts_col).iloc[-1]


def build_intraday_reopen_events(asset: str, benchmark_name: str, token_source: str) -> pd.DataFrame:
    token = load_token(asset, token_source)
    if benchmark_name == "gc_f_yahoo":
        benchmark = load_futures_benchmark()
    else:
        benchmark = load_benchmark()

    monday_dates = (
        benchmark["timestamp"]
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    monday_dates = monday_dates[monday_dates.dt.weekday == 0]

    rows = []
    for monday_date in monday_dates:
        friday_cutoff = monday_date - pd.Timedelta(days=3)
        common_friday_anchor_ts = friday_cutoff + pd.Timedelta(hours=20)
        reopen_ts = comex_week_reopen_utc(monday_date)

        friday_token = observation_at_timestamp(token, "timestamp", "token_price", common_friday_anchor_ts)
        pre_reopen_token = last_observation_before(token, "timestamp", "token_price", reopen_ts - pd.Timedelta(seconds=1))
        friday_benchmark = observation_at_timestamp(
            benchmark,
            "timestamp",
            "benchmark_close",
            common_friday_anchor_ts,
        )
        reopen_benchmark = first_observation_at_or_after(benchmark, "timestamp", "benchmark_open", reopen_ts)

        if any(item is None for item in [friday_token, pre_reopen_token, friday_benchmark, reopen_benchmark]):
            continue

        weekend_token_return = float(np.log(pre_reopen_token["token_price"]) - np.log(friday_token["token_price"]))
        reopen_gap_return = float(np.log(reopen_benchmark["benchmark_open"]) - np.log(friday_benchmark["benchmark_close"]))

        rows.append(
            {
                "monday_date": monday_date,
                "common_friday_anchor_ts": common_friday_anchor_ts,
                "friday_token_ts": friday_token["timestamp"],
                "pre_reopen_token_ts": pre_reopen_token["timestamp"],
                "friday_benchmark_ts": friday_benchmark["timestamp"],
                "reopen_benchmark_ts": reopen_benchmark["timestamp"],
                "weekend_token_return_pre_reopen": weekend_token_return,
                "benchmark_reopen_gap_return": reopen_gap_return,
                "direction_match": int(np.sign(weekend_token_return) == np.sign(reopen_gap_return)),
            }
        )

    return pd.DataFrame(rows).sort_values("monday_date").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    asset = normalize_asset(args.asset)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    events = build_intraday_reopen_events(asset, args.benchmark, args.token_source)
    out = RESULTS_DIR / f"{asset}_intraday_reopen_events_{args.token_source}_{args.benchmark}.csv"
    events.to_csv(out, index=False)
    print(f"Saved: {out}")
    if not events.empty:
        print(events.head().to_string(index=False))


if __name__ == "__main__":
    main()
