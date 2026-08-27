from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from asset_config import RAW_DIR, asset_name, asset_ticker, normalize_asset  # noqa: E402
from fetch_market_data import read_api_key_from_dotenv  # noqa: E402
from intraday_reopen_utils import chunk_date_ranges, coingecko_asset_id  # noqa: E402


COINGECKO_RANGE_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
ALPHAVANTAGE_FX_INTRADAY_URL = (
    "https://www.alphavantage.co/query"
    "?function=FX_INTRADAY&from_symbol=XAU&to_symbol=USD&interval=60min&outputsize=full&month={month}&apikey={api_key}"
)
YAHOO_TOKEN_INTRADAY_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?interval=60m&period1={period1}&period2={period2}&includePrePost=true&events=div%2Csplits"
)
YAHOO_FUTURES_INTRADAY_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
    "?interval=60m&period1={period1}&period2={period2}&includePrePost=true&events=div%2Csplits"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull separate intraday data for the weekend-reopen design.")
    parser.add_argument("--asset", default="paxg", choices=["paxg", "xaut"])
    parser.add_argument("--start", default="2019-09-26", help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-07-31", help="End date in YYYY-MM-DD.")
    parser.add_argument(
        "--token-provider",
        default="coingecko",
        choices=["coingecko", "yahoo"],
        help="Current intraday token source.",
    )
    parser.add_argument(
        "--gold-provider",
        default="yahoo_futures",
        choices=["alphavantage_fx", "yahoo_futures", "skip"],
        help="Current intraday gold source.",
    )
    return parser.parse_args()


def get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    response = requests.get(url, timeout=60, headers=headers or {"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.json()


def token_hourly_csv_path(asset: str) -> Path:
    return RAW_DIR / f"{normalize_asset(asset)}_coingecko_hourly.csv"


def token_yahoo_hourly_csv_path(asset: str) -> Path:
    return RAW_DIR / f"{normalize_asset(asset)}_yahoo_60m.csv"


def gold_hourly_csv_path() -> Path:
    return RAW_DIR / "xauusd_alphavantage_fx_60min.csv"


def gold_monthly_cache_path(month: str) -> Path:
    return RAW_DIR / f"xauusd_alphavantage_fx_60min_{month}.json"


def futures_hourly_csv_path() -> Path:
    return RAW_DIR / "gc_f_yahoo_60m.csv"


def fetch_token_hourly(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    coin_id = coingecko_asset_id(asset)
    rows: list[dict[str, float | str]] = []

    for left, right in chunk_date_ranges(start, end, days_per_chunk=90):
        params = {
            "vs_currency": "usd",
            "from": int(left.timestamp()),
            "to": int((right + pd.Timedelta(days=1)).timestamp()),
        }
        query = "&".join([f"{key}={value}" for key, value in params.items()])
        url = f"{COINGECKO_RANGE_URL.format(coin_id=coin_id)}?{query}"
        payload = get_json(url)

        for ts_ms, price in payload.get("prices", []):
            rows.append({"timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True), "price": price})

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out = token_hourly_csv_path(asset)
    df.to_csv(out, index=False)
    return out


def fetch_token_hourly_from_yahoo(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    rows: list[dict[str, float | str]] = []
    symbol = asset_ticker(asset)

    for left, right in chunk_date_ranges(start, end, days_per_chunk=180):
        url = YAHOO_TOKEN_INTRADAY_URL.format(
            symbol=symbol,
            period1=int(left.timestamp()),
            period2=int((right + pd.Timedelta(days=1)).timestamp()),
        )
        payload = get_json(url)
        result = payload["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        quote = result["indicators"]["quote"][0]
        for ts_s, open_px, high_px, low_px, close_px, volume_px in zip(
            timestamps,
            quote["open"],
            quote["high"],
            quote["low"],
            quote["close"],
            quote.get("volume", [None] * len(timestamps)),
        ):
            rows.append(
                {
                    "timestamp": pd.to_datetime(ts_s, unit="s", utc=True),
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                    "volume": volume_px,
                }
            )

    df = (
        pd.DataFrame(rows)
        .dropna(subset=["close"])
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    out = token_yahoo_hourly_csv_path(asset)
    df.to_csv(out, index=False)
    return out


def month_strings(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    months = pd.period_range(start=start.normalize(), end=end.normalize(), freq="M")
    return [str(month) for month in months]


def fetch_gold_fx_hourly(start: pd.Timestamp, end: pd.Timestamp) -> Path:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip() or read_api_key_from_dotenv()
    if not api_key:
        raise RuntimeError("No Alpha Vantage API key found for FX intraday pull.")

    rows: list[dict[str, float | str]] = []
    for month in month_strings(start, end):
        url = ALPHAVANTAGE_FX_INTRADAY_URL.format(month=month, api_key=api_key)
        payload = get_json(url)
        gold_monthly_cache_path(month).write_text(json.dumps(payload, indent=2))

        time_series_key = next((key for key in payload if "Time Series FX" in key), None)
        if not time_series_key:
            note = payload.get("Note") or payload.get("Information") or str(list(payload.keys()))
            raise RuntimeError(f"Alpha Vantage FX intraday did not return usable data for {month}: {note}")

        for ts_text, values in payload[time_series_key].items():
            rows.append(
                {
                    "timestamp": pd.to_datetime(ts_text, utc=True),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out = gold_hourly_csv_path()
    df.to_csv(out, index=False)
    return out


def fetch_gold_futures_hourly(start: pd.Timestamp, end: pd.Timestamp) -> Path:
    rows: list[dict[str, float | str]] = []
    for left, right in chunk_date_ranges(start, end, days_per_chunk=700):
        url = YAHOO_FUTURES_INTRADAY_URL.format(
            period1=int(left.timestamp()),
            period2=int((right + pd.Timedelta(days=1)).timestamp()),
        )
        payload = get_json(url)
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        for ts_s, open_px, high_px, low_px, close_px in zip(
            timestamps,
            quote["open"],
            quote["high"],
            quote["low"],
            quote["close"],
        ):
            rows.append(
                {
                    "timestamp": pd.to_datetime(ts_s, unit="s", utc=True),
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                }
            )

    df = (
        pd.DataFrame(rows)
        .dropna(subset=["open", "close"])
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    out = futures_hourly_csv_path()
    df.to_csv(out, index=False)
    return out


def main() -> None:
    args = parse_args()
    asset = normalize_asset(args.asset)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    if args.token_provider == "yahoo":
        token_path = fetch_token_hourly_from_yahoo(asset, start, end)
    else:
        token_path = fetch_token_hourly(asset, start, end)
    print(f"Saved token hourly data for {asset_name(asset)} -> {token_path}")

    if args.gold_provider == "alphavantage_fx":
        gold_path = fetch_gold_fx_hourly(start, end)
        print(f"Saved gold hourly data -> {gold_path}")
    elif args.gold_provider == "yahoo_futures":
        futures_path = fetch_gold_futures_hourly(start, end)
        print(f"Saved gold futures hourly data -> {futures_path}")


if __name__ == "__main__":
    main()
