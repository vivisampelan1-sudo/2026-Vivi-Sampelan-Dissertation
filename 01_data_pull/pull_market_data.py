from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
PROJECT_ROOT = DISS3_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from fetch_market_data import (  # noqa: E402
    build_calendar_basis_dataset,
    build_weekday_basis_dataset,
    ensure_dirs,
    fetch_gold_benchmark,
    fetch_gold_benchmark_from_yahoo,
    fetch_tokenized_gold,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull daily market data for the dissertation pipeline.")
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


def main() -> None:
    args = parse_args()
    ensure_dirs()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

    # Primary weekday benchmark: spot gold (XAU). Fails clearly instead of
    # silently swapping in futures unless --allow-futures-fallback is passed.
    gold = fetch_gold_benchmark(api_key_override=api_key, allow_futures_fallback=args.allow_futures_fallback).frame

    # First-class weekend/tradeability benchmark: gold FUTURES (GC=F).
    # Pulled deliberately every run so the trading, OOS-filter and Monday-gap
    # scripts always read fresh futures data (data/raw/gold_yahoo_gc_f_1y.json),
    # rather than a stale cached file that was only refreshed on fallback.
    futures = fetch_gold_benchmark_from_yahoo()
    print(f"Pulled gold futures GC=F: {len(futures.frame)} rows "
          f"({futures.frame['date'].min()} to {futures.frame['date'].max()}) "
          f"-> data/raw/gold_yahoo_gc_f_1y.json")
    print()

    for asset in ["paxg", "xaut"]:
        token = fetch_tokenized_gold(asset).frame
        weekday_dataset = build_weekday_basis_dataset(token, gold)
        calendar_dataset = build_calendar_basis_dataset(token, gold)
        write_outputs(asset, weekday_dataset, calendar_dataset)
        print()


if __name__ == "__main__":
    main()
