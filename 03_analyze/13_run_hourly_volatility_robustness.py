"""Hourly realised-volatility robustness for H4.

Builds a supplementary weekend-volatility measure from hourly token returns
over the pre-reopen weekend window and tests whether that turbulence predicts
the size of the subsequent COMEX reopening gap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from asset_config import asset_name  # noqa: E402
from econometrics_utils import default_hac_lags, fit_ols  # noqa: E402


RAW_DIR = DISS3_DIR / "04_data" / "raw"
RESULTS_DIR = DISS3_DIR / "05_results"
ASSETS = ["paxg", "xaut"]


def load_hourly_token(asset: str) -> pd.DataFrame:
    path = RAW_DIR / f"{asset}_yahoo_60m.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_hourly_futures() -> pd.DataFrame:
    path = RAW_DIR / "gc_f_yahoo_60m.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_intraday_events(asset: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{asset}_intraday_reopen_events_yahoo_gc_f_yahoo.csv"
    df = pd.read_csv(
        path,
        parse_dates=[
            "monday_date",
            "friday_token_ts",
            "pre_reopen_token_ts",
            "friday_benchmark_ts",
            "reopen_benchmark_ts",
        ],
    )
    for col in ["friday_token_ts", "pre_reopen_token_ts", "friday_benchmark_ts", "reopen_benchmark_ts"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values("monday_date").reset_index(drop=True)


def build_hourly_h4_events(asset: str) -> pd.DataFrame:
    token = load_hourly_token(asset).set_index("timestamp")
    futures = load_hourly_futures().set_index("timestamp")
    events = load_intraday_events(asset)

    rows: list[dict[str, object]] = []
    for _, ev in events.iterrows():
        token_window = token.loc[ev["friday_token_ts"] : ev["pre_reopen_token_ts"]].copy()
        futures_window = futures.loc[[ev["friday_benchmark_ts"], ev["reopen_benchmark_ts"]]].copy()
        if len(token_window) < 3 or len(futures_window) != 2:
            continue

        log_close = np.log(token_window["close"].astype(float))
        hourly_returns = log_close.diff().dropna()
        hourly_rv = float(np.sqrt((hourly_returns**2).sum()))

        weekend_volume = float(token_window["volume"].fillna(0).iloc[1:].sum()) if "volume" in token_window else np.nan
        friday_basis = float(
            np.log(float(token_window.loc[ev["friday_token_ts"], "close"]))
            - np.log(float(futures.loc[ev["friday_benchmark_ts"], "close"]))
        )
        reopen_gap_abs = float(abs(ev["benchmark_reopen_gap_return"]))

        rows.append(
            {
                "monday_date": ev["monday_date"],
                "hourly_weekend_rv": hourly_rv,
                "log_weekend_volume": float(np.log(max(weekend_volume, 1.0))),
                "friday_basis": friday_basis,
                "abs_reopen_gap_return": reopen_gap_abs,
                "n_hourly_returns": int(len(hourly_returns)),
            }
        )

    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def run_regression(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    for predictor in predictors:
        X[predictor] = df[predictor]
    res = fit_ols(df["abs_reopen_gap_return"], X, hac_lags=default_hac_lags(len(df)))
    return pd.DataFrame(
        {
            "term": res.coefficients.index,
            "coefficient": res.coefficients.values,
            "std_error_hac": res.std_errors.values,
            "t_stat_hac": res.t_stats.values,
            "p_value_approx": [float(2 * norm.sf(abs(t))) for t in res.t_stats.values],
            "r_squared": res.r_squared,
            "n_obs": res.n_obs,
            "hac_lags": res.hac_lags,
        }
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    details = []
    summary_rows = []

    for asset in ASSETS:
        df = build_hourly_h4_events(asset)
        specs = {
            "hourly_rv_only": ["hourly_weekend_rv"],
            "hourly_rv_controlled": ["hourly_weekend_rv", "log_weekend_volume", "friday_basis"],
        }
        for model, predictors in specs.items():
            out = run_regression(df, predictors)
            out.insert(0, "model", model)
            out.insert(0, "asset", asset_name(asset))
            details.append(out)

        controlled = pd.concat(details, ignore_index=True)
        controlled = controlled[
            (controlled["asset"] == asset_name(asset))
            & (controlled["model"] == "hourly_rv_controlled")
            & (controlled["term"] == "hourly_weekend_rv")
        ].iloc[0]

        summary_rows.append(
            {
                "asset": asset_name(asset),
                "metric": "hourly_weekend_rv_beta",
                "value": float(controlled["coefficient"]),
            }
        )
        summary_rows.append(
            {
                "asset": asset_name(asset),
                "metric": "hourly_weekend_rv_t_stat",
                "value": float(controlled["t_stat_hac"]),
            }
        )
        summary_rows.append(
            {
                "asset": asset_name(asset),
                "metric": "hourly_weekend_rv_p_value",
                "value": float(controlled["p_value_approx"]),
            }
        )
        summary_rows.append(
            {
                "asset": asset_name(asset),
                "metric": "hourly_weekend_rv_r_squared",
                "value": float(controlled["r_squared"]),
            }
        )
        summary_rows.append(
            {
                "asset": asset_name(asset),
                "metric": "hourly_weekend_rv_n",
                "value": int(controlled["n_obs"]),
            }
        )

    details_df = pd.concat(details, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    details_path = RESULTS_DIR / "hourly_h4_volatility_robustness_details.csv"
    summary_path = RESULTS_DIR / "hourly_h4_volatility_robustness_summary.csv"
    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved: {details_path}")
    print(f"Saved: {summary_path}")
    print()
    print(details_df.to_string(index=False))
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
