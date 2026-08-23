from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "05_results"

ASSET_TO_TICKER = {
    "paxg": "PAXG-USD",
    "xaut": "XAUT-USD",
}

ASSET_TO_NAME = {
    "paxg": "PAXG",
    "xaut": "XAUT",
}


def normalize_asset(asset: str) -> str:
    key = asset.strip().lower()
    if key not in ASSET_TO_TICKER:
        raise ValueError(f"Unsupported asset '{asset}'. Choose one of: {', '.join(sorted(ASSET_TO_TICKER))}.")
    return key


def asset_name(asset: str) -> str:
    return ASSET_TO_NAME[normalize_asset(asset)]


def asset_ticker(asset: str) -> str:
    return ASSET_TO_TICKER[normalize_asset(asset)]


def raw_price_path(asset: str) -> Path:
    return RAW_DIR / f"{normalize_asset(asset)}_yahoo_daily_full.json"


def processed_basis_path(asset: str, sample: str) -> Path:
    return PROCESSED_DIR / f"{normalize_asset(asset)}_gold_basis_{sample}.csv"


def processed_summary_path(asset: str) -> Path:
    return PROCESSED_DIR / f"{normalize_asset(asset)}_gold_basis_summary.csv"


def result_path(asset: str, stem: str) -> Path:
    return RESULTS_DIR / f"{normalize_asset(asset)}_{stem}.csv"


def parse_asset_args(description: str, default_asset: str = "paxg") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--asset",
        default=default_asset,
        choices=sorted(ASSET_TO_TICKER),
        help="Tokenized gold asset to analyze.",
    )
    return parser.parse_args()
