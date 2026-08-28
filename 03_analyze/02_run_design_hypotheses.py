from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
PROJECT_ROOT = DISS3_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
PROCESS_SCRIPTS_DIR = DISS3_DIR / "02_process"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))
if str(PROCESS_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_SCRIPTS_DIR))

from asset_config import asset_name, normalize_asset  # noqa: E402
from econometrics_utils import adf_test, default_hac_lags, fit_ols  # noqa: E402

from option2_weekend_design import build_events, load_data as load_calendar_data  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"
ASSETS = ["paxg", "xaut"]


def result_path(stem: str, ext: str = "csv") -> Path:
    return RESULTS_DIR / f"design_hypotheses_{stem}.{ext}"


def h1_cointegration(asset: str, calendar_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = calendar_df[["date", "token_log_price", "gold_log_price"]].dropna().copy()
    X = pd.DataFrame({"const": 1.0, "gold_log_price": sample["gold_log_price"]})
    long_run = fit_ols(sample["token_log_price"], X, hac_lags=default_hac_lags(len(sample)))
    ect = sample["token_log_price"] - (
        long_run.coefficients["const"] + long_run.coefficients["gold_log_price"] * sample["gold_log_price"]
    )
    eg_adf = adf_test(
        ect,
        regression="n",
        test_name=f"{asset_name(asset)} Engle-Granger residual ADF",
        critical_values={"1%": -3.90, "5%": -3.34, "10%": -3.05},
        note="Residual stationarity test for tokenized gold versus spot gold.",
    )

    summary = pd.DataFrame(
        [
            {"asset": asset_name(asset), "hypothesis": "H1", "metric": "long_run_gold_beta", "value": float(long_run.coefficients["gold_log_price"])},
            {"asset": asset_name(asset), "hypothesis": "H1", "metric": "long_run_beta_t_stat_hac", "value": float(long_run.t_stats["gold_log_price"])},
            {"asset": asset_name(asset), "hypothesis": "H1", "metric": "engle_granger_residual_adf_stat", "value": float(eg_adf.test_stat)},
            {"asset": asset_name(asset), "hypothesis": "H1", "metric": "engle_granger_crit_5pct", "value": float(eg_adf.critical_values["5%"])},
            {
                "asset": asset_name(asset),
                "hypothesis": "H1",
                "metric": "cointegration_supported",
                "value": bool(eg_adf.test_stat < eg_adf.critical_values["5%"]),
            },
        ]
    )

    details = pd.DataFrame(
        [
            {
                "asset": asset_name(asset),
                "model": "H1 long-run regression",
                "term": term,
                "coefficient": float(long_run.coefficients[term]),
                "std_error_hac": float(long_run.std_errors[term]),
                "t_stat_hac": float(long_run.t_stats[term]),
                "r_squared": float(long_run.r_squared),
                "n_obs": int(long_run.n_obs),
            }
            for term in long_run.coefficients.index
        ]
    )
    return summary, details


def h3_weekend_prediction(asset: str, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_X = pd.DataFrame(
        {
            "const": 1.0,
            "weekend_token_return": events["weekend_token_return"],
            "log_weekend_volume": events["log_weekend_volume"],
            "friday_basis": events["friday_basis"],
        }
    )
    base_res = fit_ols(events["monday_gold_return"], base_X, hac_lags=default_hac_lags(len(events)))

    asym_X = pd.DataFrame(
        {
            "const": 1.0,
            "positive_weekend_return": events["positive_weekend_return"],
            "negative_weekend_return": events["negative_weekend_return"],
            "log_weekend_volume": events["log_weekend_volume"],
            "friday_basis": events["friday_basis"],
        }
    )
    asym_res = fit_ols(events["monday_gold_return"], asym_X, hac_lags=default_hac_lags(len(events)))

    basis_X = pd.DataFrame(
        {
            "const": 1.0,
            "basis_widening_weekend": events["basis_widening_weekend"],
            "log_weekend_volume": events["log_weekend_volume"],
            "friday_basis": events["friday_basis"],
        }
    )
    basis_res = fit_ols(events["basis_reversal_monday"], basis_X, hac_lags=default_hac_lags(len(events)))

    summary = pd.DataFrame(
        [
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "number_of_weekends", "value": int(len(events))},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "directional_accuracy", "value": float(events["direction_match"].mean())},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "weekend_return_beta", "value": float(base_res.coefficients["weekend_token_return"])},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "weekend_return_t_stat_hac", "value": float(base_res.t_stats["weekend_token_return"])},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "negative_weekend_return_beta", "value": float(asym_res.coefficients["negative_weekend_return"])},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "negative_weekend_return_t_stat_hac", "value": float(asym_res.t_stats["negative_weekend_return"])},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "positive_weekend_return_beta", "value": float(asym_res.coefficients["positive_weekend_return"])},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "positive_weekend_return_t_stat_hac", "value": float(asym_res.t_stats["positive_weekend_return"])},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "basis_reversal_beta", "value": float(basis_res.coefficients["basis_widening_weekend"])},
            {"asset": asset_name(asset), "hypothesis": "H3", "metric": "basis_reversal_t_stat_hac", "value": float(basis_res.t_stats["basis_widening_weekend"])},
            {
                "asset": asset_name(asset),
                "hypothesis": "H3",
                "metric": "h3_supported",
                "value": bool(
                    base_res.coefficients["weekend_token_return"] > 0
                    and base_res.t_stats["weekend_token_return"] > 1.96
                ),
            },
        ]
    )

    details = []
    for label, res in [
        ("H3 baseline Monday gold return", base_res),
        ("H3 asymmetry Monday gold return", asym_res),
        ("H3 Monday basis reversal", basis_res),
    ]:
        for term in res.coefficients.index:
            details.append(
                {
                    "asset": asset_name(asset),
                    "model": label,
                    "term": term,
                    "coefficient": float(res.coefficients[term]),
                    "std_error_hac": float(res.std_errors[term]),
                    "t_stat_hac": float(res.t_stats[term]),
                    "r_squared": float(res.r_squared),
                    "n_obs": int(res.n_obs),
                }
            )
    return summary, pd.DataFrame(details)


def render_markdown(summary: pd.DataFrame) -> str:
    def pick(asset: str, metric: str) -> str:
        value = summary.loc[(summary["asset"] == asset) & (summary["metric"] == metric), "value"].iloc[0]
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return f"{float(value):.6f}"

    lines = [
        "# Dissertation 3 Hypothesis Results",
        "",
        "## What This File Covers",
        "",
        "- `H1`: long-run validation through cointegration",
        "- `H3`: average weekend-to-Monday predictive effect (direction)",
        "- `H3` asymmetry: the positive vs negative weekend betas in the table below",
        "",
        "## Asset Summary",
        "",
        "| Asset | H1 Supported? | H3 Supported? | H3 Weekend Return Beta | H3 Weekend Return t-stat | Negative Weekend Beta | Positive Weekend Beta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for asset in ["PAXG", "XAUT"]:
        lines.append(
            f"| {asset} | {pick(asset, 'cointegration_supported')} | {pick(asset, 'h3_supported')} | "
            f"{pick(asset, 'weekend_return_beta')} | {pick(asset, 'weekend_return_t_stat_hac')} | "
            f"{pick(asset, 'negative_weekend_return_beta')} | {pick(asset, 'positive_weekend_return_beta')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `H1` is supported for `PAXG`: residual ADF rejects non-stationarity = {pick('PAXG', 'cointegration_supported')}.",
            f"- `H1` is supported for `XAUT`: residual ADF rejects non-stationarity = {pick('XAUT', 'cointegration_supported')}.",
            f"- `H3` is supported for `PAXG`: weekend beta = {pick('PAXG', 'weekend_return_beta')} with HAC t-stat = {pick('PAXG', 'weekend_return_t_stat_hac')}.",
            f"- `H3` is supported for `XAUT`: weekend beta = {pick('XAUT', 'weekend_return_beta')} with HAC t-stat = {pick('XAUT', 'weekend_return_t_stat_hac')}.",
            f"- For `PAXG`, negative weekend moves ({pick('PAXG', 'negative_weekend_return_beta')}) are stronger than positive weekend moves ({pick('PAXG', 'positive_weekend_return_beta')}).",
            f"- For `XAUT`, negative weekend moves ({pick('XAUT', 'negative_weekend_return_beta')}) are also stronger than positive weekend moves ({pick('XAUT', 'positive_weekend_return_beta')}).",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []

    for asset in ASSETS:
        asset = normalize_asset(asset)
        calendar_df = load_calendar_data(asset)
        events = build_events(calendar_df)

        h1_summary, h1_detail = h1_cointegration(asset, calendar_df)
        h3_summary, h3_detail = h3_weekend_prediction(asset, events)

        summary_frames.extend([h1_summary, h3_summary])
        detail_frames.extend([h1_detail, h3_detail])

    summary = pd.concat(summary_frames, ignore_index=True)
    detail = pd.concat(detail_frames, ignore_index=True)

    summary_path = result_path("summary")
    detail_path = result_path("details")
    md_path = result_path("summary", ext="md")

    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    md_path.write_text(render_markdown(summary))

    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {md_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
