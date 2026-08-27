from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from asset_config import (
    RAW_DIR,
    PROCESSED_DIR,
    asset_name,
    asset_ticker,
    normalize_asset,
    processed_basis_path,
    processed_summary_path,
    raw_price_path,
)

ROOT = Path(__file__).resolve().parents[1]
TOKEN_START = int(datetime(2019, 9, 26, tzinfo=timezone.utc).timestamp())
TOKEN_END = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp())
GOLD_URL = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d&period1=1514764800&period2=1800000000"
ALPHAVANTAGE_GOLD_URL = (
    "https://www.alphavantage.co/query"
    "?function=GOLD_SILVER_HISTORY&symbol=GOLD&interval=daily&apikey={api_key}"
)
ALPHAVANTAGE_CACHE_PATH = RAW_DIR / "gold_alphavantage_daily.json"


@dataclass
class FetchResult:
    name: str
    frame: pd.DataFrame


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch tokenized gold and gold benchmark data.")
    parser.add_argument(
        "--asset",
        default="paxg",
        choices=["paxg", "xaut"],
        help="Tokenized gold asset to analyze.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional Alpha Vantage API key. Overrides environment and .env values.",
    )
    parser.add_argument(
        "--allow-futures-fallback",
        action="store_true",
        help=(
            "If no Alpha Vantage key/cache is available, silently substitute Yahoo GC=F futures "
            "as the gold benchmark instead of failing. Runs produced this way will NOT reproduce "
            "the dissertation's spot-gold tables."
        ),
    )
    return parser.parse_args()


def read_api_key_from_dotenv() -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "ALPHAVANTAGE_API_KEY":
            return value.strip().strip("'").strip('"')
    return ""


def yahoo_token_url(asset: str) -> str:
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{asset_ticker(asset)}"
        f"?period1={TOKEN_START}&period2={TOKEN_END}&interval=1d&includeAdjustedClose=true"
    )


def parse_yahoo_token_payload(payload: dict[str, Any], asset: str) -> FetchResult:
    asset = normalize_asset(asset)
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "token_price": quote["close"],
            "token_total_volume": quote.get("volume", [np.nan] * len(timestamps)),
        }
    )
    frame["date"] = pd.to_datetime(frame["timestamp_s"], unit="s", utc=True).dt.date
    frame["token_market_cap"] = np.nan
    frame = frame.drop(columns=["timestamp_s"]).dropna(subset=["token_price"])
    frame = frame.drop_duplicates(subset=["date"])
    frame = frame[["date", "token_price", "token_total_volume", "token_market_cap"]]
    return FetchResult(name=asset, frame=frame)


def fetch_tokenized_gold_from_cached(asset: str) -> FetchResult:
    asset = normalize_asset(asset)
    raw_path = raw_price_path(asset)
    if not raw_path.exists():
        raise FileNotFoundError(f"No cached token file found at {raw_path}")
    payload = json.loads(raw_path.read_text())
    return parse_yahoo_token_payload(payload, asset)


def fetch_tokenized_gold(asset: str) -> FetchResult:
    asset = normalize_asset(asset)
    try:
        payload = get_json(yahoo_token_url(asset))
        raw_path = raw_price_path(asset)
        raw_path.write_text(json.dumps(payload, indent=2))
        print(f"Using live Yahoo token data for {asset_name(asset)}.")
        return parse_yahoo_token_payload(payload, asset)
    except Exception as exc:
        print(f"Live Yahoo token pull failed for {asset_name(asset)}: {exc}")
        print(f"Using cached Yahoo token data for {asset_name(asset)} instead.")
        return fetch_tokenized_gold_from_cached(asset)


def fetch_gold_benchmark_from_alphavantage(api_key: str) -> FetchResult:
    payload = get_json(ALPHAVANTAGE_GOLD_URL.format(api_key=api_key))
    if "data" not in payload:
        raise ValueError(
            "Alpha Vantage did not return a usable gold history payload. "
            f"Top-level keys: {list(payload.keys())}"
        )

    ALPHAVANTAGE_CACHE_PATH.write_text(json.dumps(payload, indent=2))
    return parse_alphavantage_payload(payload, source_name="gold_alphavantage")


def parse_alphavantage_payload(payload: dict[str, Any], source_name: str) -> FetchResult:
    if "data" not in payload:
        raise ValueError(
            "Alpha Vantage payload does not contain 'data'. "
            f"Top-level keys: {list(payload.keys())}"
        )

    frame = pd.DataFrame(payload["data"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.date

    value_column = None
    for candidate in ["value", "price", "close"]:
        if candidate in frame.columns:
            value_column = candidate
            break
    if value_column is None:
        raise ValueError(f"Could not find a gold price column in Alpha Vantage payload: {frame.columns.tolist()}")

    frame["gold_price"] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=["gold_price"]).sort_values("date")
    frame["gold_volume"] = np.nan
    frame = frame[["date", "gold_price", "gold_volume"]]
    return FetchResult(name=source_name, frame=frame)


def fetch_gold_benchmark_from_cached_alphavantage() -> FetchResult:
    if not ALPHAVANTAGE_CACHE_PATH.exists():
        raise FileNotFoundError(f"No cached Alpha Vantage file found at {ALPHAVANTAGE_CACHE_PATH}")
    payload = json.loads(ALPHAVANTAGE_CACHE_PATH.read_text())
    return parse_alphavantage_payload(payload, source_name="gold_alphavantage_cached")


def fetch_gold_benchmark_from_yahoo() -> FetchResult:
    payload = get_json(GOLD_URL)
    raw_path = RAW_DIR / "gold_yahoo_gc_f_1y.json"
    raw_path.write_text(json.dumps(payload, indent=2))

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    close_prices = result["indicators"]["quote"][0]["close"]
    volumes = result["indicators"]["quote"][0].get("volume", [])

    frame = pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "gold_price": close_prices,
            "gold_volume": volumes if len(volumes) == len(timestamps) else [np.nan] * len(timestamps),
        }
    )
    frame["date"] = pd.to_datetime(frame["timestamp_s"], unit="s", utc=True).dt.date
    frame = frame.drop(columns=["timestamp_s"]).dropna(subset=["gold_price"])
    frame = frame.drop_duplicates(subset=["date"])
    frame = frame[["date", "gold_price", "gold_volume"]]
    return FetchResult(name="gold_yahoo", frame=frame)


def fetch_gold_benchmark(api_key_override: str = "", allow_futures_fallback: bool = False) -> FetchResult:
    """Fetch the SPOT gold benchmark (Alpha Vantage), which is the conceptual
    benchmark used throughout the dissertation's weekday/weekend design.

    This does NOT silently substitute gold futures (GC=F) when no Alpha
    Vantage key or cache is available. Reproducing the reported spot-gold
    results requires an Alpha Vantage key (see .env.example). Pass
    allow_futures_fallback=True (or --allow-futures-fallback on the CLI) to
    opt into the old silent-fallback behaviour for quick experimentation —
    results produced this way use GC=F as the benchmark and will NOT match
    the spot-gold tables in the dissertation.
    """
    api_key = api_key_override.strip() or os.getenv("ALPHAVANTAGE_API_KEY", "").strip() or read_api_key_from_dotenv()
    if api_key:
        try:
            result = fetch_gold_benchmark_from_alphavantage(api_key)
            print("Using Alpha Vantage GOLD_SILVER_HISTORY for the gold benchmark.")
            return result
        except Exception as exc:
            print(f"Alpha Vantage gold pull failed: {exc}")
            if not allow_futures_fallback:
                raise RuntimeError(
                    "Alpha Vantage spot-gold pull failed and --allow-futures-fallback was not set. "
                    "Fix the API key/connection, or re-run with --allow-futures-fallback if you "
                    "explicitly want a GC=F-benchmarked (non-reproducing) run."
                ) from exc
            print("--allow-futures-fallback is set: falling back to Yahoo GC=F proxy.")

    elif ALPHAVANTAGE_CACHE_PATH.exists():
        try:
            result = fetch_gold_benchmark_from_cached_alphavantage()
            print("Using cached Alpha Vantage GOLD_SILVER_HISTORY for the gold benchmark.")
            return result
        except Exception as exc:
            print(f"Cached Alpha Vantage gold load failed: {exc}")
            if not allow_futures_fallback:
                raise RuntimeError(
                    "Cached Alpha Vantage spot-gold data failed to load and --allow-futures-fallback "
                    "was not set. Provide a working ALPHAVANTAGE_API_KEY (see .env.example), or "
                    "re-run with --allow-futures-fallback if you explicitly want a GC=F-benchmarked "
                    "(non-reproducing) run."
                ) from exc
            print("--allow-futures-fallback is set: falling back to Yahoo GC=F proxy.")

    elif not allow_futures_fallback:
        raise RuntimeError(
            "No ALPHAVANTAGE_API_KEY found (checked --api-key, the ALPHAVANTAGE_API_KEY environment "
            "variable, and .env) and no cached Alpha Vantage data exists at "
            f"{ALPHAVANTAGE_CACHE_PATH}. Reproducing the dissertation's spot-gold results requires "
            "an Alpha Vantage key: copy .env.example to .env and fill in ALPHAVANTAGE_API_KEY. "
            "If you explicitly want a GC=F-benchmarked run instead (this will NOT reproduce the "
            "spot-gold tables), re-run with --allow-futures-fallback."
        )

    print("Using Yahoo GC=F as the fallback gold benchmark (--allow-futures-fallback).")
    return fetch_gold_benchmark_from_yahoo()


def add_basis_features(frame: pd.DataFrame) -> pd.DataFrame:
    merged = frame.sort_values("date").reset_index(drop=True).copy()

    merged["token_log_price"] = np.log(merged["token_price"])
    merged["gold_log_price"] = np.log(merged["gold_price"])
    merged["basis"] = merged["token_log_price"] - merged["gold_log_price"]
    merged["abs_basis"] = merged["basis"].abs()
    merged["premium_discount_pct"] = (
        (merged["token_price"] - merged["gold_price"]) / merged["gold_price"]
    ) * 100
    merged["weekday"] = pd.to_datetime(merged["date"]).dt.day_name()
    merged["is_monday"] = (pd.to_datetime(merged["date"]).dt.weekday == 0).astype(int)
    merged["gold_return"] = merged["gold_log_price"].diff()
    merged["token_return"] = merged["token_log_price"].diff()
    merged["tracking_error"] = (merged["token_return"] - merged["gold_return"]).abs()
    merged["basis_change"] = merged["basis"].diff()

    return merged


def build_weekday_basis_dataset(token: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    merged = token.merge(gold, on="date", how="inner")
    return add_basis_features(merged)


def build_calendar_basis_dataset(token: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    token = token.copy()
    gold = gold.copy()
    token["date"] = pd.to_datetime(token["date"])
    gold["date"] = pd.to_datetime(gold["date"])

    calendar = pd.DataFrame(
        {"date": pd.date_range(start=token["date"].min(), end=token["date"].max(), freq="D")}
    )
    merged = (
        calendar.merge(gold, on="date", how="left")
        .merge(token, on="date", how="left")
        .sort_values("date")
    )
    merged["gold_forward_filled"] = merged["gold_price"].isna().astype(int)
    merged["gold_price"] = merged["gold_price"].ffill()
    merged["gold_volume"] = merged["gold_volume"].ffill()
    merged = merged.dropna(subset=["token_price", "gold_price"])
    return add_basis_features(merged)


def write_outputs(asset: str, weekday_dataset: pd.DataFrame, calendar_dataset: pd.DataFrame) -> None:
    weekday_path = processed_basis_path(asset, "weekday")
    calendar_path = processed_basis_path(asset, "calendar")
    summary_path = processed_summary_path(asset)

    weekday_dataset.to_csv(weekday_path, index=False)
    calendar_dataset.to_csv(calendar_path, index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "weekday_rows",
                "calendar_rows",
                "start_date",
                "end_date",
                "weekday_mean_basis",
                "calendar_mean_basis",
                "weekday_mean_abs_basis",
                "calendar_mean_abs_basis",
                "weekday_mean_premium_discount_pct",
                "calendar_mean_premium_discount_pct",
            ],
            "value": [
                len(weekday_dataset),
                len(calendar_dataset),
                weekday_dataset["date"].min(),
                weekday_dataset["date"].max(),
                weekday_dataset["basis"].mean(),
                calendar_dataset["basis"].mean(),
                weekday_dataset["abs_basis"].mean(),
                calendar_dataset["abs_basis"].mean(),
                weekday_dataset["premium_discount_pct"].mean(),
                calendar_dataset["premium_discount_pct"].mean(),
            ],
        }
    )
    summary.to_csv(summary_path, index=False)

    print("Saved:")
    print(f"  - {raw_price_path(asset)}")
    print(f"  - {weekday_path}")
    print(f"  - {calendar_path}")
    print(f"  - {summary_path}")


def main() -> None:
    args = parse_args()
    asset = normalize_asset(args.asset)
    ensure_dirs()
    token = fetch_tokenized_gold(asset).frame
    gold = fetch_gold_benchmark(api_key_override=args.api_key, allow_futures_fallback=args.allow_futures_fallback).frame
    weekday_dataset = build_weekday_basis_dataset(token, gold)
    calendar_dataset = build_calendar_basis_dataset(token, gold)
    write_outputs(asset, weekday_dataset, calendar_dataset)
    print()
    print(f"{asset_name(asset)} weekday sample tail:")
    print(weekday_dataset.tail(5).to_string(index=False))
    print()
    print(f"{asset_name(asset)} calendar sample tail:")
    print(calendar_dataset.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
