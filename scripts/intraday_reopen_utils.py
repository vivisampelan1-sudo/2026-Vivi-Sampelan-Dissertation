from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


COINGECKO_ASSET_IDS = {
    "paxg": "pax-gold",
    "xaut": "tether-gold",
}

COMEX_TIMEZONE = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")


@dataclass
class WeekendReopenEvent:
    monday_date: pd.Timestamp
    friday_close_ts: pd.Timestamp
    token_pre_reopen_ts: pd.Timestamp
    benchmark_reopen_ts: pd.Timestamp
    benchmark_first_trade_ts: pd.Timestamp
    friday_close_price: float
    token_pre_reopen_price: float
    benchmark_reopen_price: float


def coingecko_asset_id(asset: str) -> str:
    key = asset.strip().lower()
    if key not in COINGECKO_ASSET_IDS:
        raise ValueError(f"Unsupported asset '{asset}'.")
    return COINGECKO_ASSET_IDS[key]


def chunk_date_ranges(start: pd.Timestamp, end: pd.Timestamp, days_per_chunk: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if end < start:
        raise ValueError("End date must be on or after start date.")

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    left = start.normalize()
    right_edge = end.normalize()
    delta = timedelta(days=days_per_chunk)

    while left <= right_edge:
        right = min(left + delta, right_edge)
        chunks.append((left, right))
        left = right + timedelta(days=1)

    return chunks


def comex_week_reopen_utc(monday_date: pd.Timestamp) -> pd.Timestamp:
    monday_ts = pd.Timestamp(monday_date).normalize()
    sunday_local = datetime(
        year=(monday_ts - pd.Timedelta(days=1)).year,
        month=(monday_ts - pd.Timedelta(days=1)).month,
        day=(monday_ts - pd.Timedelta(days=1)).day,
        hour=17,
        minute=0,
        second=0,
        tzinfo=COMEX_TIMEZONE,
    )
    return pd.Timestamp(sunday_local.astimezone(UTC)).tz_localize(None)


def ensure_utc_naive(series: pd.Series, unit: str = "ms") -> pd.Series:
    return pd.to_datetime(series, unit=unit, utc=True).dt.tz_convert(None)


def last_observation_before(frame: pd.DataFrame, ts_col: str, value_col: str, cutoff: pd.Timestamp) -> pd.Series | None:
    subset = frame.loc[frame[ts_col] <= cutoff, [ts_col, value_col]].dropna()
    if subset.empty:
        return None
    return subset.sort_values(ts_col).iloc[-1]


def first_observation_at_or_after(frame: pd.DataFrame, ts_col: str, value_col: str, cutoff: pd.Timestamp) -> pd.Series | None:
    subset = frame.loc[frame[ts_col] >= cutoff, [ts_col, value_col]].dropna()
    if subset.empty:
        return None
    return subset.sort_values(ts_col).iloc[0]
