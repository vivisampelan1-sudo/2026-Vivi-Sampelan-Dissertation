from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
PROJECT_ROOT = DISS3_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
PROCESS_SCRIPTS_DIR = DISS3_DIR / "03_process"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))
if str(PROCESS_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_SCRIPTS_DIR))

from asset_config import asset_name, normalize_asset  # noqa: E402
from econometrics_utils import default_hac_lags, fit_ols  # noqa: E402

from option2_weekend_design import build_events, load_data as load_calendar_data  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"
ASSETS = ["paxg", "xaut"]
INITIAL_TRAIN_WEEKS = 104


def result_path(stem: str, ext: str = "csv") -> Path:
    return RESULTS_DIR / f"oos_precision_{stem}.{ext}"


def fit_predict_spec(train: pd.DataFrame, test_row: pd.Series, predictors: list[str]) -> float:
    X_train = pd.DataFrame(
        {"const": 1.0},
        index=train.index,
    )
    for predictor in predictors:
        X_train[predictor] = train[predictor]
    res = fit_ols(train["monday_gold_return"], X_train, hac_lags=None)
    x_test = np.array([1.0] + [float(test_row[predictor]) for predictor in predictors])
    return float(x_test @ res.coefficients.to_numpy())


def majority_sign(train: pd.Series) -> int:
    mean_sign = float(train.mean())
    return 1 if mean_sign >= 0 else -1


def sign_hit(prediction: float, actual: float) -> int:
    return int(np.sign(prediction) == np.sign(actual))


def wilson_interval(hit_rate: float, n_obs: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n_obs == 0:
        return np.nan, np.nan
    denom = 1.0 + z**2 / n_obs
    center = (hit_rate + z**2 / (2.0 * n_obs)) / denom
    margin = z * math.sqrt((hit_rate * (1.0 - hit_rate) + z**2 / (4.0 * n_obs)) / n_obs) / denom
    return center - margin, center + margin


def normal_two_sided_pvalue(t_stat: float) -> float:
    return math.erfc(abs(float(t_stat)) / math.sqrt(2.0))


def normal_one_sided_upper_pvalue(t_stat: float) -> float:
    return 0.5 * math.erfc(float(t_stat) / math.sqrt(2.0))


def dm_test(loss_full: pd.Series, loss_benchmark: pd.Series) -> tuple[float, float, float]:
    diff = pd.Series(loss_benchmark).astype(float) - pd.Series(loss_full).astype(float)
    diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
    X = pd.DataFrame({"const": 1.0}, index=diff.index)
    res = fit_ols(diff, X, hac_lags=default_hac_lags(len(diff)))
    mean_diff = float(res.coefficients["const"])
    t_stat = float(res.t_stats["const"])
    p_value = normal_two_sided_pvalue(t_stat)
    return mean_diff, t_stat, p_value


def clark_west_test(
    actual: pd.Series,
    pred_full: pd.Series,
    pred_benchmark: pd.Series,
) -> tuple[float, float, float]:
    actual = pd.Series(actual).astype(float)
    pred_full = pd.Series(pred_full).astype(float)
    pred_benchmark = pd.Series(pred_benchmark).astype(float)

    e_full = actual - pred_full
    e_bench = actual - pred_benchmark
    adjusted_diff = (e_bench**2) - (e_full**2 - (pred_benchmark - pred_full) ** 2)
    adjusted_diff = adjusted_diff.replace([np.inf, -np.inf], np.nan).dropna()

    X = pd.DataFrame({"const": 1.0}, index=adjusted_diff.index)
    res = fit_ols(adjusted_diff, X, hac_lags=default_hac_lags(len(adjusted_diff)))
    mean_adjusted_diff = float(res.coefficients["const"])
    t_stat = float(res.t_stats["const"])
    p_value = normal_one_sided_upper_pvalue(t_stat)
    return mean_adjusted_diff, t_stat, p_value


def run_asset(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = build_events(load_calendar_data(asset)).sort_values("monday_date").reset_index(drop=True)
    rows = []
    model_specs = {
        "full_model": ["weekend_token_return", "log_weekend_volume", "friday_basis"],
        "basis_only": ["friday_basis"],
        "weekend_return_only": ["weekend_token_return"],
    }

    for idx in range(INITIAL_TRAIN_WEEKS, len(events)):
        train = events.iloc[:idx].copy()
        test = events.iloc[idx].copy()
        actual = float(test["monday_gold_return"])

        preds = {
            name: fit_predict_spec(train, test, predictors)
            for name, predictors in model_specs.items()
        }
        preds["historical_mean"] = float(train["monday_gold_return"].mean())
        preds["zero_benchmark"] = 0.0
        preds["majority_sign_rule"] = majority_sign(train["monday_gold_return"])
        preds["raw_weekend_sign_rule"] = 1 if float(test["weekend_token_return"]) >= 0 else -1

        rows.append(
            {
                "asset": asset_name(asset),
                "monday_date": test["monday_date"],
                "actual_return": actual,
                "full_model_pred": preds["full_model"],
                "basis_only_pred": preds["basis_only"],
                "weekend_return_only_pred": preds["weekend_return_only"],
                "historical_mean_pred": preds["historical_mean"],
                "zero_benchmark_pred": preds["zero_benchmark"],
                "majority_sign_rule_pred": preds["majority_sign_rule"],
                "raw_weekend_sign_rule_pred": preds["raw_weekend_sign_rule"],
                "full_model_error": actual - preds["full_model"],
                "basis_only_error": actual - preds["basis_only"],
                "weekend_return_only_error": actual - preds["weekend_return_only"],
                "historical_mean_error": actual - preds["historical_mean"],
                "zero_benchmark_error": actual - preds["zero_benchmark"],
                "full_model_hit": sign_hit(preds["full_model"], actual),
                "basis_only_hit": sign_hit(preds["basis_only"], actual),
                "weekend_return_only_hit": sign_hit(preds["weekend_return_only"], actual),
                "historical_mean_hit": sign_hit(preds["historical_mean"], actual),
                "majority_sign_rule_hit": sign_hit(preds["majority_sign_rule"], actual),
                "raw_weekend_sign_rule_hit": sign_hit(preds["raw_weekend_sign_rule"], actual),
            }
        )

    detail = pd.DataFrame(rows)

    rmse = {
        name: float(np.sqrt(np.mean(detail[f"{name}_error"] ** 2)))
        for name in ["full_model", "basis_only", "weekend_return_only", "historical_mean", "zero_benchmark"]
    }
    mae = {
        name: float(np.mean(np.abs(detail[f"{name}_error"])))
        for name in ["full_model", "basis_only", "weekend_return_only", "historical_mean", "zero_benchmark"]
    }
    accuracy = {
        name: float(detail[f"{name}_hit"].mean())
        for name in ["full_model", "basis_only", "weekend_return_only", "historical_mean", "majority_sign_rule", "raw_weekend_sign_rule"]
    }
    ci = {name: wilson_interval(rate, len(detail)) for name, rate in accuracy.items()}

    dm_rows = []
    for benchmark in ["basis_only", "weekend_return_only", "historical_mean", "zero_benchmark"]:
        mean_diff, t_stat, p_value = dm_test(
            detail["full_model_error"] ** 2,
            detail[f"{benchmark}_error"] ** 2,
        )
        dm_rows.extend(
            [
                {"asset": asset_name(asset), "metric": f"dm_mean_loss_diff_full_vs_{benchmark}", "value": mean_diff},
                {"asset": asset_name(asset), "metric": f"dm_t_stat_full_vs_{benchmark}", "value": t_stat},
                {"asset": asset_name(asset), "metric": f"dm_p_value_full_vs_{benchmark}", "value": p_value},
            ]
        )

    cw_rows = []
    for benchmark in ["basis_only", "weekend_return_only"]:
        mean_adjusted_diff, t_stat, p_value = clark_west_test(
            detail["actual_return"],
            detail["full_model_pred"],
            detail[f"{benchmark}_pred"],
        )
        cw_rows.extend(
            [
                {"asset": asset_name(asset), "metric": f"cw_adjusted_mean_diff_full_vs_{benchmark}", "value": mean_adjusted_diff},
                {"asset": asset_name(asset), "metric": f"cw_t_stat_full_vs_{benchmark}", "value": t_stat},
                {"asset": asset_name(asset), "metric": f"cw_p_value_full_vs_{benchmark}", "value": p_value},
            ]
        )

    summary = pd.DataFrame(
        [
            {"asset": asset_name(asset), "metric": "oos_observations", "value": int(len(detail))},
            {"asset": asset_name(asset), "metric": "initial_train_weeks", "value": int(INITIAL_TRAIN_WEEKS)},
            {"asset": asset_name(asset), "metric": "first_oos_forecast_date", "value": detail["monday_date"].iloc[0]},
            {"asset": asset_name(asset), "metric": "last_oos_forecast_date", "value": detail["monday_date"].iloc[-1]},
            {"asset": asset_name(asset), "metric": "full_model_directional_accuracy", "value": accuracy["full_model"]},
            {"asset": asset_name(asset), "metric": "basis_only_directional_accuracy", "value": accuracy["basis_only"]},
            {"asset": asset_name(asset), "metric": "weekend_return_only_directional_accuracy", "value": accuracy["weekend_return_only"]},
            {"asset": asset_name(asset), "metric": "historical_mean_directional_accuracy", "value": accuracy["historical_mean"]},
            {"asset": asset_name(asset), "metric": "majority_sign_rule_accuracy", "value": accuracy["majority_sign_rule"]},
            {"asset": asset_name(asset), "metric": "raw_weekend_sign_rule_accuracy", "value": accuracy["raw_weekend_sign_rule"]},
            {"asset": asset_name(asset), "metric": "full_model_directional_accuracy_ci_lower", "value": ci["full_model"][0]},
            {"asset": asset_name(asset), "metric": "full_model_directional_accuracy_ci_upper", "value": ci["full_model"][1]},
            {"asset": asset_name(asset), "metric": "basis_only_directional_accuracy_ci_lower", "value": ci["basis_only"][0]},
            {"asset": asset_name(asset), "metric": "basis_only_directional_accuracy_ci_upper", "value": ci["basis_only"][1]},
            {"asset": asset_name(asset), "metric": "weekend_return_only_directional_accuracy_ci_lower", "value": ci["weekend_return_only"][0]},
            {"asset": asset_name(asset), "metric": "weekend_return_only_directional_accuracy_ci_upper", "value": ci["weekend_return_only"][1]},
            {"asset": asset_name(asset), "metric": "historical_mean_directional_accuracy_ci_lower", "value": ci["historical_mean"][0]},
            {"asset": asset_name(asset), "metric": "historical_mean_directional_accuracy_ci_upper", "value": ci["historical_mean"][1]},
            {"asset": asset_name(asset), "metric": "majority_sign_rule_accuracy_ci_lower", "value": ci["majority_sign_rule"][0]},
            {"asset": asset_name(asset), "metric": "majority_sign_rule_accuracy_ci_upper", "value": ci["majority_sign_rule"][1]},
            {"asset": asset_name(asset), "metric": "raw_weekend_sign_rule_accuracy_ci_lower", "value": ci["raw_weekend_sign_rule"][0]},
            {"asset": asset_name(asset), "metric": "raw_weekend_sign_rule_accuracy_ci_upper", "value": ci["raw_weekend_sign_rule"][1]},
            {"asset": asset_name(asset), "metric": "full_model_rmse", "value": rmse["full_model"]},
            {"asset": asset_name(asset), "metric": "basis_only_rmse", "value": rmse["basis_only"]},
            {"asset": asset_name(asset), "metric": "weekend_return_only_rmse", "value": rmse["weekend_return_only"]},
            {"asset": asset_name(asset), "metric": "historical_mean_rmse", "value": rmse["historical_mean"]},
            {"asset": asset_name(asset), "metric": "zero_benchmark_rmse", "value": rmse["zero_benchmark"]},
            {"asset": asset_name(asset), "metric": "full_model_mae", "value": mae["full_model"]},
            {"asset": asset_name(asset), "metric": "basis_only_mae", "value": mae["basis_only"]},
            {"asset": asset_name(asset), "metric": "weekend_return_only_mae", "value": mae["weekend_return_only"]},
            {"asset": asset_name(asset), "metric": "historical_mean_mae", "value": mae["historical_mean"]},
            {"asset": asset_name(asset), "metric": "zero_benchmark_mae", "value": mae["zero_benchmark"]},
            {
                "asset": asset_name(asset),
                "metric": "rmse_improvement_vs_zero_pct",
                "value": float((rmse["zero_benchmark"] - rmse["full_model"]) / rmse["zero_benchmark"]) if rmse["zero_benchmark"] > 0 else np.nan,
            },
            {
                "asset": asset_name(asset),
                "metric": "pred_actual_correlation",
                "value": float(detail["full_model_pred"].corr(detail["actual_return"])),
            },
            *cw_rows,
            *dm_rows,
        ]
    )
    return summary, detail


def render_markdown(summary: pd.DataFrame) -> str:
    def pick(asset: str, metric: str) -> str:
        value = summary.loc[(summary["asset"] == asset) & (summary["metric"] == metric), "value"].iloc[0]
        if isinstance(value, str):
            return value
        return f"{float(value):.6f}"

    def pct(asset: str, metric: str) -> str:
        value = float(summary.loc[(summary["asset"] == asset) & (summary["metric"] == metric), "value"].iloc[0])
        return f"{100 * value:.1f}%"

    lines = [
        "# Out-of-Sample Forecast Precision",
        "",
        f"This file evaluates the weekend signal using an expanding-window forecast with an initial training window of `{INITIAL_TRAIN_WEEKS}` weekends.",
        "",
        "| Model | PAXG dir. acc. | PAXG RMSE | XAUT dir. acc. | XAUT RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Full model | {pct('PAXG', 'full_model_directional_accuracy')} | {pick('PAXG', 'full_model_rmse')} | {pct('XAUT', 'full_model_directional_accuracy')} | {pick('XAUT', 'full_model_rmse')} |",
        f"| Basis-only model | {pct('PAXG', 'basis_only_directional_accuracy')} | {pick('PAXG', 'basis_only_rmse')} | {pct('XAUT', 'basis_only_directional_accuracy')} | {pick('XAUT', 'basis_only_rmse')} |",
        f"| Weekend-return-only model | {pct('PAXG', 'weekend_return_only_directional_accuracy')} | {pick('PAXG', 'weekend_return_only_rmse')} | {pct('XAUT', 'weekend_return_only_directional_accuracy')} | {pick('XAUT', 'weekend_return_only_rmse')} |",
        f"| Historical-mean model | {pct('PAXG', 'historical_mean_directional_accuracy')} | {pick('PAXG', 'historical_mean_rmse')} | {pct('XAUT', 'historical_mean_directional_accuracy')} | {pick('XAUT', 'historical_mean_rmse')} |",
        f"| Majority-sign rule | {pct('PAXG', 'majority_sign_rule_accuracy')} | — | {pct('XAUT', 'majority_sign_rule_accuracy')} | — |",
        f"| Raw weekend-sign rule | {pct('PAXG', 'raw_weekend_sign_rule_accuracy')} | — | {pct('XAUT', 'raw_weekend_sign_rule_accuracy')} | — |",
        "",
        "| Full model directional-accuracy 95% CI | PAXG | XAUT |",
        "| --- | ---: | ---: |",
        f"| Wilson interval | {pct('PAXG', 'full_model_directional_accuracy_ci_lower')} to {pct('PAXG', 'full_model_directional_accuracy_ci_upper')} | {pct('XAUT', 'full_model_directional_accuracy_ci_lower')} to {pct('XAUT', 'full_model_directional_accuracy_ci_upper')} |",
        "",
        "| Clark-West comparison (nested models, squared-error loss, one-sided alternative) | PAXG p-value | XAUT p-value |",
        "| --- | ---: | ---: |",
        f"| Full vs basis-only | {pick('PAXG', 'cw_p_value_full_vs_basis_only')} | {pick('XAUT', 'cw_p_value_full_vs_basis_only')} |",
        f"| Full vs weekend-return-only | {pick('PAXG', 'cw_p_value_full_vs_weekend_return_only')} | {pick('XAUT', 'cw_p_value_full_vs_weekend_return_only')} |",
        "",
        "| Diebold-Mariano comparison (non-nested benchmarks, squared-error loss) | PAXG p-value | XAUT p-value |",
        "| --- | ---: | ---: |",
        f"| Full vs historical-mean | {pick('PAXG', 'dm_p_value_full_vs_historical_mean')} | {pick('XAUT', 'dm_p_value_full_vs_historical_mean')} |",
        f"| Full vs zero benchmark | {pick('PAXG', 'dm_p_value_full_vs_zero_benchmark')} | {pick('XAUT', 'dm_p_value_full_vs_zero_benchmark')} |",
        ]
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- The benchmark table compares the full model against the basis-only, weekend-return-only, historical-mean, majority-sign, and raw-sign alternatives.",
            "- Wilson intervals provide uncertainty bounds for the full-model directional-accuracy estimate.",
            "- Clark-West tests are used for the nested full-vs-basis-only and full-vs-weekend-return-only comparisons under squared-error loss with a one-sided alternative that the full model has lower MSPE.",
            "- Diebold-Mariano-style comparisons are retained only for the non-nested historical-mean and zero benchmarks under squared-error loss with HAC standard errors.",
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
        summary, detail = run_asset(asset)
        summary_frames.append(summary)
        detail_frames.append(detail)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    detail_df = pd.concat(detail_frames, ignore_index=True)

    summary_path = result_path("summary")
    detail_path = result_path("details")
    md_path = result_path("summary", ext="md")

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    md_path.write_text(render_markdown(summary_df), encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {md_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
