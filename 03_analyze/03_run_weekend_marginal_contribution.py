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
from econometrics_utils import default_hac_lags, fit_ols  # noqa: E402

from option2_weekend_design import build_events, load_data as load_calendar_data  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"
ASSETS = ["paxg", "xaut"]


def result_path(stem: str, ext: str = "csv") -> Path:
    return RESULTS_DIR / f"weekend_marginal_contribution_{stem}.{ext}"


def fit_named_model(events: pd.DataFrame, predictors: list[str], name: str) -> tuple[object, float]:
    X = pd.DataFrame({"const": 1.0}, index=events.index)
    for predictor in predictors:
        X[predictor] = events[predictor]
    res = fit_ols(events["monday_gold_return"], X, hac_lags=default_hac_lags(len(events)))
    sse = float((res.residuals ** 2).sum())
    return res, sse


def run_asset(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = build_events(load_calendar_data(asset))

    specs = {
        "weekend_only": ["weekend_token_return"],
        "basis_only": ["friday_basis"],
        "volume_only": ["log_weekend_volume"],
        "weekend_plus_volume": ["weekend_token_return", "log_weekend_volume"],
        "basis_plus_volume": ["friday_basis", "log_weekend_volume"],
        "full_model": ["weekend_token_return", "friday_basis", "log_weekend_volume"],
    }

    fitted = {}
    detail_rows = []
    for name, predictors in specs.items():
        res, sse = fit_named_model(events, predictors, name)
        fitted[name] = {"res": res, "sse": sse}
        for term in res.coefficients.index:
            detail_rows.append(
                {
                    "asset": asset_name(asset),
                    "model": name,
                    "term": term,
                    "coefficient": float(res.coefficients[term]),
                    "std_error_hac": float(res.std_errors[term]),
                    "t_stat_hac": float(res.t_stats[term]),
                    "r_squared": float(res.r_squared),
                    "n_obs": int(res.n_obs),
                }
            )

    full_r2 = float(fitted["full_model"]["res"].r_squared)
    weekend_only_r2 = float(fitted["weekend_only"]["res"].r_squared)
    basis_only_r2 = float(fitted["basis_only"]["res"].r_squared)
    basis_plus_volume_r2 = float(fitted["basis_plus_volume"]["res"].r_squared)
    weekend_plus_volume_r2 = float(fitted["weekend_plus_volume"]["res"].r_squared)

    sse_full = fitted["full_model"]["sse"]
    sse_restricted_weekend = fitted["basis_plus_volume"]["sse"]
    sse_restricted_basis = fitted["weekend_plus_volume"]["sse"]

    partial_r2_weekend = (sse_restricted_weekend - sse_full) / sse_restricted_weekend if sse_restricted_weekend > 0 else float("nan")
    partial_r2_basis = (sse_restricted_basis - sse_full) / sse_restricted_basis if sse_restricted_basis > 0 else float("nan")

    summary = pd.DataFrame(
        [
            {"asset": asset_name(asset), "metric": "weekend_only_r_squared", "value": weekend_only_r2},
            {"asset": asset_name(asset), "metric": "basis_only_r_squared", "value": basis_only_r2},
            {"asset": asset_name(asset), "metric": "basis_plus_volume_r_squared", "value": basis_plus_volume_r2},
            {"asset": asset_name(asset), "metric": "weekend_plus_volume_r_squared", "value": weekend_plus_volume_r2},
            {"asset": asset_name(asset), "metric": "full_model_r_squared", "value": full_r2},
            {
                "asset": asset_name(asset),
                "metric": "incremental_r_squared_from_weekend_given_basis",
                "value": full_r2 - basis_plus_volume_r2,
            },
            {
                "asset": asset_name(asset),
                "metric": "incremental_r_squared_from_basis_given_weekend",
                "value": full_r2 - weekend_plus_volume_r2,
            },
            {
                "asset": asset_name(asset),
                "metric": "partial_r_squared_weekend_given_basis",
                "value": partial_r2_weekend,
            },
            {
                "asset": asset_name(asset),
                "metric": "partial_r_squared_basis_given_weekend",
                "value": partial_r2_basis,
            },
            {
                "asset": asset_name(asset),
                "metric": "full_model_weekend_t_stat",
                "value": float(fitted["full_model"]["res"].t_stats["weekend_token_return"]),
            },
            {
                "asset": asset_name(asset),
                "metric": "full_model_basis_t_stat",
                "value": float(fitted["full_model"]["res"].t_stats["friday_basis"]),
            },
        ]
    )

    return summary, pd.DataFrame(detail_rows)


def render_markdown(summary: pd.DataFrame) -> str:
    def pick(asset: str, metric: str) -> str:
        value = summary.loc[(summary["asset"] == asset) & (summary["metric"] == metric), "value"].iloc[0]
        return f"{float(value):.6f}"

    lines = [
        "# Weekend Marginal Contribution",
        "",
        "This file decomposes the high `R²` in the Monday gold return model into the part associated with the pre-existing `friday_basis` channel and the incremental part associated with the weekend return itself.",
        "",
        "| Asset | Weekend-Only R² | Basis-Only R² | Full-Model R² | Incremental R² from Weekend | Partial R² Weekend Given Basis | Weekend t-stat | Basis t-stat |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for asset in ["PAXG", "XAUT"]:
        lines.append(
            f"| {asset} | {pick(asset, 'weekend_only_r_squared')} | {pick(asset, 'basis_only_r_squared')} | "
            f"{pick(asset, 'full_model_r_squared')} | {pick(asset, 'incremental_r_squared_from_weekend_given_basis')} | "
            f"{pick(asset, 'partial_r_squared_weekend_given_basis')} | {pick(asset, 'full_model_weekend_t_stat')} | "
            f"{pick(asset, 'full_model_basis_t_stat')} |"
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `Basis-only R²` shows how much of Monday gold adjustment is explained by the pre-existing Friday token-gold gap.",
            "- `Incremental R² from Weekend` shows how much the model fit improves when weekend token returns are added on top of basis and volume controls.",
            "- `Partial R² Weekend Given Basis` isolates the weekend return's own marginal explanatory contribution.",
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
