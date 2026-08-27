from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
PROJECT_ROOT = DISS3_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from asset_config import RAW_DIR, asset_name, normalize_asset  # noqa: E402
from econometrics_utils import default_hac_lags, fit_ols  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"
ASSETS = ["paxg", "xaut"]
GOLD_FUTURES_PATH = RAW_DIR / "gold_yahoo_gc_f_1y.json"


def result_path(stem: str, ext: str = "csv") -> Path:
    return RESULTS_DIR / f"monday_gap_robustness_{stem}.{ext}"


def load_yahoo_ohlc(path: Path, price_prefix: str) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "timestamp_s": timestamps,
            f"{price_prefix}_open": quote["open"],
            f"{price_prefix}_close": quote["close"],
        }
    )
    df["date"] = pd.to_datetime(df["timestamp_s"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    df = df.drop(columns=["timestamp_s"]).dropna(subset=[f"{price_prefix}_close"])
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_token(asset: str) -> pd.DataFrame:
    return load_yahoo_ohlc(RAW_DIR / f"{normalize_asset(asset)}_yahoo_daily_full.json", "token")


def load_gold_futures() -> pd.DataFrame:
    return load_yahoo_ohlc(GOLD_FUTURES_PATH, "gold")


def build_gap_events(asset: str) -> pd.DataFrame:
    token = load_token(asset)
    gold = load_gold_futures()
    token = token.sort_values("date").reset_index(drop=True)
    gold = gold.sort_values("date").reset_index(drop=True)
    gold["weekday_num"] = gold["date"].dt.weekday

    rows = []
    mondays = gold[gold["weekday_num"] == 0].copy()
    for _, mon in mondays.iterrows():
        monday = mon["date"]
        friday = monday - pd.Timedelta(days=3)
        sunday = monday - pd.Timedelta(days=1)

        friday_token_row = token[token["date"] == friday]
        sunday_token_row = token[token["date"] == sunday]
        friday_gold_row = gold[gold["date"] == friday]
        monday_gold_row = gold[gold["date"] == monday]
        if friday_token_row.empty or sunday_token_row.empty or friday_gold_row.empty or monday_gold_row.empty:
            continue

        friday_token_row = friday_token_row.iloc[0]
        sunday_token_row = sunday_token_row.iloc[0]
        friday_gold_row = friday_gold_row.iloc[0]
        monday_gold_row = monday_gold_row.iloc[0]

        if pd.isna(monday_gold_row["gold_open"]) or pd.isna(friday_gold_row["gold_close"]):
            continue

        weekend_token_return = float(np.log(sunday_token_row["token_close"]) - np.log(friday_token_row["token_close"]))
        monday_gap_return = float(np.log(monday_gold_row["gold_open"]) - np.log(friday_gold_row["gold_close"]))
        monday_close_return = float(np.log(monday_gold_row["gold_close"]) - np.log(friday_gold_row["gold_close"]))
        friday_basis = float(np.log(friday_token_row["token_close"]) - np.log(friday_gold_row["gold_close"]))

        rows.append(
            {
                "monday_date": monday,
                "weekend_token_return": weekend_token_return,
                "monday_gap_return": monday_gap_return,
                "monday_close_return": monday_close_return,
                "friday_basis": friday_basis,
                "has_nonzero_gap": int(abs(monday_gap_return) > 0),
            }
        )

    return pd.DataFrame(rows).sort_values("monday_date").reset_index(drop=True)


def fit_named(events: pd.DataFrame, target: str, predictors: list[str], model: str, asset: str) -> pd.DataFrame:
    X = pd.DataFrame({"const": 1.0}, index=events.index)
    for predictor in predictors:
        X[predictor] = events[predictor]
    res = fit_ols(events[target], X, hac_lags=default_hac_lags(len(events)))
    return pd.DataFrame(
        {
            "asset": asset_name(asset),
            "model": model,
            "target": target,
            "term": res.coefficients.index,
            "coefficient": res.coefficients.values,
            "std_error_hac": res.std_errors.values,
            "t_stat_hac": res.t_stats.values,
            "r_squared": res.r_squared,
            "n_obs": res.n_obs,
            "covariance_type": res.covariance_type,
            "hac_lags": res.hac_lags,
        }
    )


def run_asset(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = build_gap_events(asset)
    details = pd.concat(
        [
            fit_named(events, "monday_gap_return", ["weekend_token_return"], "Gap target weekend-only", asset),
            fit_named(events, "monday_gap_return", ["weekend_token_return", "friday_basis"], "Gap target with basis", asset),
            fit_named(events, "monday_close_return", ["weekend_token_return", "friday_basis"], "Close target with basis", asset),
        ],
        ignore_index=True,
    )

    def stat(model: str, term: str, field: str) -> float:
        return float(details.loc[(details["model"] == model) & (details["term"] == term), field].iloc[0])

    summary = pd.DataFrame(
        [
            {"asset": asset_name(asset), "metric": "gap_sample_mondays", "value": int(len(events))},
            {"asset": asset_name(asset), "metric": "gap_weekend_only_beta", "value": stat("Gap target weekend-only", "weekend_token_return", "coefficient")},
            {"asset": asset_name(asset), "metric": "gap_weekend_only_t_stat", "value": stat("Gap target weekend-only", "weekend_token_return", "t_stat_hac")},
            {"asset": asset_name(asset), "metric": "gap_weekend_only_r_squared", "value": stat("Gap target weekend-only", "weekend_token_return", "r_squared")},
            {"asset": asset_name(asset), "metric": "gap_with_basis_weekend_beta", "value": stat("Gap target with basis", "weekend_token_return", "coefficient")},
            {"asset": asset_name(asset), "metric": "gap_with_basis_weekend_t_stat", "value": stat("Gap target with basis", "weekend_token_return", "t_stat_hac")},
            {"asset": asset_name(asset), "metric": "gap_with_basis_r_squared", "value": stat("Gap target with basis", "weekend_token_return", "r_squared")},
            {"asset": asset_name(asset), "metric": "close_with_basis_weekend_beta", "value": stat("Close target with basis", "weekend_token_return", "coefficient")},
            {"asset": asset_name(asset), "metric": "close_with_basis_weekend_t_stat", "value": stat("Close target with basis", "weekend_token_return", "t_stat_hac")},
            {"asset": asset_name(asset), "metric": "close_with_basis_r_squared", "value": stat("Close target with basis", "weekend_token_return", "r_squared")},
        ]
    )
    return summary, details


def render_markdown(summary: pd.DataFrame) -> str:
    def pick(asset: str, metric: str) -> str:
        value = summary.loc[(summary["asset"] == asset) & (summary["metric"] == metric), "value"].iloc[0]
        return f"{float(value):.6f}"

    lines = [
        "# Monday Gap Robustness",
        "",
        "This check replaces the main target `Monday close - Friday close` with a cleaner reopening-gap target based on cached gold-futures proxy data: `Monday open - Friday close`.",
        "",
        "| Asset | Gap Target Weekend Beta | Gap Target Weekend t-stat | Gap Target R² | Gap+Basis Weekend Beta | Gap+Basis Weekend t-stat | Close+Basis Weekend Beta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for asset in ["PAXG", "XAUT"]:
        lines.append(
            f"| {asset} | {pick(asset, 'gap_weekend_only_beta')} | {pick(asset, 'gap_weekend_only_t_stat')} | {pick(asset, 'gap_weekend_only_r_squared')} | "
            f"{pick(asset, 'gap_with_basis_weekend_beta')} | {pick(asset, 'gap_with_basis_weekend_t_stat')} | {pick(asset, 'close_with_basis_weekend_beta')} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- If the weekend coefficient remains positive and meaningful for the `gap` target, then the main result is not driven only by Monday intraday trading.",
            "- If the gap-target coefficient is weaker than the close-target coefficient, that means part of the original effect comes from Monday trading after the open, which should be acknowledged as a limitation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    detail_frames = []
    for asset in ASSETS:
        asset = normalize_asset(asset)
        summary, details = run_asset(asset)
        summary_frames.append(summary)
        detail_frames.append(details)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    details_df = pd.concat(detail_frames, ignore_index=True)

    summary_path = result_path("summary")
    details_path = result_path("details")
    md_path = result_path("summary", ext="md")

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)
    md_path.write_text(render_markdown(summary_df))

    print(f"Saved: {summary_path}")
    print(f"Saved: {details_path}")
    print(f"Saved: {md_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
