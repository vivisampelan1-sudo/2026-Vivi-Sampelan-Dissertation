from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from econometrics_utils import default_hac_lags, fit_ols  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"
ASSETS = ["paxg", "xaut"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the intraday reopen regression summary.")
    parser.add_argument("--token-source", default="coingecko", choices=["coingecko", "yahoo"])
    parser.add_argument("--benchmark", default="gc_f_yahoo", choices=["gc_f_yahoo", "xauusd_fx"])
    return parser.parse_args()


def fit_asset(asset: str, token_source: str, benchmark: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{asset}_intraday_reopen_events_{token_source}_{benchmark}.csv"
    events = pd.read_csv(path, parse_dates=["monday_date"])
    if events.empty:
        return pd.DataFrame()

    X = pd.DataFrame(
        {
            "const": 1.0,
            "weekend_token_return_pre_reopen": events["weekend_token_return_pre_reopen"],
        }
    )
    res = fit_ols(
        events["benchmark_reopen_gap_return"],
        X,
        hac_lags=default_hac_lags(len(events)),
    )
    return pd.DataFrame(
        {
            "asset": [asset.upper()] * len(res.coefficients.index),
            "term": res.coefficients.index,
            "coefficient": res.coefficients.values,
            "std_error_hac": res.std_errors.values,
            "t_stat_hac": res.t_stats.values,
            "r_squared": res.r_squared,
            "n_obs": res.n_obs,
        }
    )


def main() -> None:
    args = parse_args()
    frames = [fit_asset(asset, args.token_source, args.benchmark) for asset in ASSETS]
    out = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    path = RESULTS_DIR / f"intraday_reopen_regression_summary_{args.token_source}_{args.benchmark}.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")
    if not out.empty:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
