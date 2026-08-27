from __future__ import annotations

import argparse
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

from econometrics_utils import default_hac_lags, fit_ols  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"
ASSETS = ["paxg", "xaut"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intraday sensitivity checks for the reopen-gap design.")
    parser.add_argument("--token-source", default="yahoo", choices=["coingecko", "yahoo"])
    parser.add_argument("--benchmark", default="gc_f_yahoo", choices=["gc_f_yahoo", "xauusd_fx"])
    return parser.parse_args()


def load_events(asset: str, token_source: str, benchmark: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{asset}_intraday_reopen_events_{token_source}_{benchmark}.csv"
    return pd.read_csv(
        path,
        parse_dates=["monday_date", "friday_benchmark_ts", "reopen_benchmark_ts"],
    ).sort_values("monday_date").reset_index(drop=True)


def fit_main(df: pd.DataFrame) -> dict[str, float]:
    X = pd.DataFrame(
        {
            "const": 1.0,
            "weekend_token_return_pre_reopen": df["weekend_token_return_pre_reopen"],
        }
    )
    res = fit_ols(
        df["benchmark_reopen_gap_return"],
        X,
        hac_lags=default_hac_lags(len(df)),
    )
    return {
        "beta": float(res.coefficients["weekend_token_return_pre_reopen"]),
        "std_error_hac": float(res.std_errors["weekend_token_return_pre_reopen"]),
        "t_stat_hac": float(res.t_stats["weekend_token_return_pre_reopen"]),
        "p_value_approx": float(2 * norm.sf(abs(res.t_stats["weekend_token_return_pre_reopen"]))),
        "r_squared": float(res.r_squared),
        "n_obs": int(res.n_obs),
    }


def huber_irls(df: pd.DataFrame, max_iter: int = 100, tol: float = 1e-9, k: float = 1.345) -> dict[str, float]:
    y = df["benchmark_reopen_gap_return"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df["weekend_token_return_pre_reopen"].to_numpy(dtype=float)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        resid = y - X @ beta
        scale = np.median(np.abs(resid - np.median(resid))) / 0.6745
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = max(np.std(resid), 1e-6)
        u = resid / scale
        weights = np.where(np.abs(u) <= k, 1.0, k / np.abs(u))
        W = np.diag(weights)
        beta_new = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    resid = y - X @ beta
    n_obs, n_params = X.shape
    sigma2 = float((weights * resid**2).sum() / max(n_obs - n_params, 1))
    xtwx_inv = np.linalg.inv(X.T @ W @ X)
    cov = sigma2 * xtwx_inv
    std_err = np.sqrt(np.diag(cov))
    t_stats = beta / std_err

    return {
        "beta": float(beta[1]),
        "std_error_hac": float(std_err[1]),
        "t_stat_approx": float(t_stats[1]),
        "p_value_approx": float(2 * norm.sf(abs(t_stats[1]))),
        "n_obs": int(n_obs),
    }


def scenario_rows(asset: str, df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_row(label: str, sample: pd.DataFrame) -> None:
        if len(sample) < 10:
            return
        fitted = fit_main(sample)
        rows.append(
            {
                "asset": asset.upper(),
                "scenario": label,
                **fitted,
                "directional_accuracy": float(sample["direction_match"].mean()),
            }
        )

    add_row("baseline", df)
    add_row("pre_2026", df[df["monday_date"] < pd.Timestamp("2026-01-01")].copy())

    # Conservative roll proxy for Yahoo's undocumented GC=F continuous-series stitching:
    # exclude first-Monday events in the main COMEX gold delivery months.
    delivery_months = {2, 4, 6, 8, 10, 12}
    likely_roll = (
        df["monday_date"].dt.month.isin(delivery_months)
        & (df["monday_date"].dt.day <= 7)
    )
    add_row("exclude_likely_roll_weekends", df.loc[~likely_roll].copy())

    ranked_gap = df.assign(abs_gap=df["benchmark_reopen_gap_return"].abs()).sort_values("abs_gap", ascending=False)
    for k in [1, 3, 5]:
        kept = ranked_gap.iloc[k:].drop(columns=["abs_gap"]).copy()
        add_row(f"exclude_top_{k}_abs_gap", kept)

    huber = huber_irls(df)
    rows.append(
            {
                "asset": asset.upper(),
                "scenario": "robust_huber_full_sample",
                "beta": huber["beta"],
                "std_error_hac": huber["std_error_hac"],
                "t_stat_hac": huber["t_stat_approx"],
                "p_value_approx": huber["p_value_approx"],
                "r_squared": np.nan,
                "n_obs": huber["n_obs"],
                "directional_accuracy": float(df["direction_match"].mean()),
        }
    )
    return rows


def leave_one_out_details(asset: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for idx in range(len(df)):
        sample = df.drop(index=idx).reset_index(drop=True)
        fitted = fit_main(sample)
        rows.append(
            {
                "asset": asset.upper(),
                "omitted_monday_date": df.loc[idx, "monday_date"],
                **fitted,
            }
        )

    detail = pd.DataFrame(rows).sort_values("omitted_monday_date").reset_index(drop=True)
    summary = {
        "asset": asset.upper(),
        "scenario": "leave_one_weekend_out_summary",
        "beta_min": float(detail["beta"].min()),
        "beta_median": float(detail["beta"].median()),
        "beta_max": float(detail["beta"].max()),
        "t_min": float(detail["t_stat_hac"].min()),
        "t_median": float(detail["t_stat_hac"].median()),
        "t_max": float(detail["t_stat_hac"].max()),
        "share_beta_positive": float((detail["beta"] > 0).mean()),
        "share_t_above_1_96": float((detail["t_stat_hac"] > 1.96).mean()),
        "n_obs": int(len(df)),
    }
    return detail, summary


def top_event_rows(asset: str, df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    ranked = df.assign(
        abs_gap=df["benchmark_reopen_gap_return"].abs(),
        abs_token=df["weekend_token_return_pre_reopen"].abs(),
    ).sort_values("abs_gap", ascending=False)
    out = ranked.head(top_k).copy()
    out.insert(0, "asset", asset.upper())
    return out[
        [
            "asset",
            "monday_date",
            "weekend_token_return_pre_reopen",
            "benchmark_reopen_gap_return",
            "abs_gap",
            "abs_token",
        ]
    ]


def main() -> None:
    args = parse_args()
    summary_rows: list[dict[str, object]] = []
    loo_summaries: list[dict[str, object]] = []
    loo_details: list[pd.DataFrame] = []
    top_events: list[pd.DataFrame] = []

    for asset in ASSETS:
        df = load_events(asset, args.token_source, args.benchmark)
        summary_rows.extend(scenario_rows(asset, df))
        detail, summary = leave_one_out_details(asset, df)
        loo_details.append(detail)
        loo_summaries.append(summary)
        top_events.append(top_event_rows(asset, df))

    summary_df = pd.DataFrame(summary_rows)
    loo_summary_df = pd.DataFrame(loo_summaries)
    loo_detail_df = pd.concat(loo_details, ignore_index=True)
    top_events_df = pd.concat(top_events, ignore_index=True)

    summary_path = RESULTS_DIR / f"intraday_sensitivity_summary_{args.token_source}_{args.benchmark}.csv"
    loo_summary_path = RESULTS_DIR / f"intraday_leave_one_out_summary_{args.token_source}_{args.benchmark}.csv"
    loo_detail_path = RESULTS_DIR / f"intraday_leave_one_out_details_{args.token_source}_{args.benchmark}.csv"
    top_events_path = RESULTS_DIR / f"intraday_top_events_{args.token_source}_{args.benchmark}.csv"

    summary_df.to_csv(summary_path, index=False)
    loo_summary_df.to_csv(loo_summary_path, index=False)
    loo_detail_df.to_csv(loo_detail_path, index=False)
    top_events_df.to_csv(top_events_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {loo_summary_path}")
    print(f"Saved: {loo_detail_path}")
    print(f"Saved: {top_events_path}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print(loo_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
