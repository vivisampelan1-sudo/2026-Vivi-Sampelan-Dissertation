"""Pooled weekend x mid-week interaction test.

Hui's Point 5: the placebo test (10_run_placebo_decay.py) shows the mid-week
coefficient is smaller than the weekend coefficient, but does not formally test
whether the difference is statistically significant, since the two are separate
regressions on separate samples with separate standard errors.

This script pools the weekend-event sample (built the same way as
build_weekend_decay in 10_run_placebo_decay.py) and the mid-week placebo sample
(build_placebo), adds a weekend indicator D, and estimates:

    gold_return = a + b1*signal + b2*(signal x D) + b3*D + b4*basis + e

b1 is the mid-week (placebo) slope, b1 + b2 is the weekend slope, and b2 is the
direct test of whether the predictive relationship is significantly stronger
during the closure period. All regressions use HAC (Newey-West) standard errors
with the same default_hac_lags rule used throughout the project.

A robustness version that also interacts basis with the weekend dummy (so the
Friday/Wednesday-basis channel is allowed to differ by regime too) is included
to check the interaction coefficient is not an artefact of forcing a single
shared basis coefficient across the two regimes.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from asset_config import asset_name, normalize_asset  # noqa: E402
from econometrics_utils import default_hac_lags, fit_ols  # noqa: E402

RESULTS_DIR = DISS3_DIR / "05_results"
RAW_DIR = DISS3_DIR / "04_data" / "raw"
ASSETS = ["paxg", "xaut"]


def load_token(asset: str) -> pd.Series:
    r = json.load(open(RAW_DIR / f"{normalize_asset(asset)}_yahoo_daily_full.json"))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"ts": r["timestamp"], "close": q["close"]}).dropna(subset=["close"])
    d["date"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    return d.set_index("date")["close"].sort_index()


def load_gold() -> pd.Series:
    d = pd.DataFrame(json.load(open(RAW_DIR / "gold_alphavantage_daily.json"))["data"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["price"] = d["price"].astype(float)
    return d.dropna(subset=["price"]).set_index("date")["price"].sort_index()


def load_core_weekend_events(asset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{normalize_asset(asset)}_option2_events.csv", parse_dates=["monday_date"])
    return df.rename(columns={"monday_date": "date"}).sort_values("date").reset_index(drop=True)


def lg(series: pd.Series, d) -> float:
    return float(np.log(series.loc[d])) if d in series.index else np.nan


def build_weekend(asset: str) -> pd.DataFrame:
    core = load_core_weekend_events(asset)
    rows = []
    for _, r in core.iterrows():
        rows.append(
            {
                "date": r["date"],
                "signal": float(r["weekend_token_return"]),
                "basis": float(r["friday_basis"]),
                "gold_return": float(r["monday_gold_return"]),
                "weekend": 1,
            }
        )
    return pd.DataFrame(rows)


def build_placebo(tok: pd.Series, gold: pd.Series, asset: str) -> pd.DataFrame:
    core = load_core_weekend_events(asset)
    placebo_wednesdays = set((core["date"] + pd.Timedelta(days=2)).tolist())
    rows = []
    for wed in gold.index:
        if wed.weekday() != 2 or wed not in placebo_wednesdays:
            continue
        mon, thu = wed - pd.Timedelta(days=2), wed + pd.Timedelta(days=1)
        tm, tw = lg(tok, mon), lg(tok, wed)
        gw, gt = lg(gold, wed), lg(gold, thu)
        if any(np.isnan(v) for v in [tm, tw, gw, gt]):
            continue
        rows.append({"date": wed, "signal": tw - tm, "basis": tw - gw, "gold_return": gt - gw, "weekend": 0})
    return pd.DataFrame(rows)


def pooled_frame(asset: str) -> pd.DataFrame:
    tok, gold = load_token(asset), load_gold()
    wk = build_weekend(asset)
    pb = build_placebo(tok, gold, asset)
    pooled = pd.concat([wk, pb], ignore_index=True)
    pooled["signal_x_weekend"] = pooled["signal"] * pooled["weekend"]
    pooled["basis_x_weekend"] = pooled["basis"] * pooled["weekend"]
    return pooled.dropna(subset=["signal", "basis", "gold_return", "weekend", "signal_x_weekend", "basis_x_weekend"])


def run_asset(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = pooled_frame(asset)
    n = len(pooled)
    lags = default_hac_lags(n)
    n_weekend = int(pooled["weekend"].sum())
    n_midweek = int((pooled["weekend"] == 0).sum())

    X_main = pd.DataFrame({
        "const": 1.0, "signal": pooled["signal"], "signal_x_weekend": pooled["signal_x_weekend"],
        "weekend": pooled["weekend"], "basis": pooled["basis"],
    })
    res_main = fit_ols(pooled["gold_return"], X_main, hac_lags=lags)

    X_robust = pd.DataFrame({
        "const": 1.0, "signal": pooled["signal"], "signal_x_weekend": pooled["signal_x_weekend"],
        "weekend": pooled["weekend"], "basis": pooled["basis"], "basis_x_weekend": pooled["basis_x_weekend"],
    })
    res_robust = fit_ols(pooled["gold_return"], X_robust, hac_lags=lags)

    rows = []
    for spec_name, res, X in [("main", res_main, X_main), ("basis_also_interacted", res_robust, X_robust)]:
        for term in X.columns:
            rows.append({
                "asset": asset_name(asset), "spec": spec_name, "term": term,
                "coefficient": float(res.coefficients[term]),
                "std_error_hac": float(res.std_errors[term]),
                "t_stat_hac": float(res.t_stats[term]),
                "r_squared": float(res.r_squared), "n_obs": n, "hac_lags": lags,
                "n_weekend": n_weekend, "n_midweek": n_midweek,
            })
    detail = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        "asset": asset_name(asset),
        "mid_week_slope": float(res_main.coefficients["signal"]),
        "mid_week_slope_t": float(res_main.t_stats["signal"]),
        "weekend_slope": float(res_main.coefficients["signal"] + res_main.coefficients["signal_x_weekend"]),
        "interaction_coefficient": float(res_main.coefficients["signal_x_weekend"]),
        "interaction_t_stat_hac": float(res_main.t_stats["signal_x_weekend"]),
        "interaction_significant_5pct": bool(abs(res_main.t_stats["signal_x_weekend"]) > 1.96),
        "n_obs": n, "n_weekend": n_weekend, "n_midweek": n_midweek, "hac_lags": lags,
        "r_squared": float(res_main.r_squared),
        "robustness_interaction_coefficient_basis_also_interacted": float(res_robust.coefficients["signal_x_weekend"]),
        "robustness_interaction_t_stat_basis_also_interacted": float(res_robust.t_stats["signal_x_weekend"]),
    }])

    return summary, detail


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries, details = [], []
    for asset in ASSETS:
        s, d = run_asset(normalize_asset(asset))
        summaries.append(s)
        details.append(d)

    summary_df = pd.concat(summaries, ignore_index=True)
    detail_df = pd.concat(details, ignore_index=True)

    summary_path = RESULTS_DIR / "placebo_weekend_interaction_summary.csv"
    detail_path = RESULTS_DIR / "placebo_weekend_interaction_details.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
