from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
PROJECT_ROOT = DISS3_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from asset_config import asset_name, normalize_asset, processed_basis_path  # noqa: E402
from econometrics_utils import default_hac_lags, fit_ols  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dissertation 3 option-2 weekend design.")
    parser.add_argument("--asset", default="paxg", choices=["paxg", "xaut"])
    parser.add_argument(
        "--window",
        type=int,
        default=52,
        help="Rolling window length in weekends. Default is 52.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also export a rolling-beta plot.",
    )
    return parser.parse_args()


def result_path(asset: str, stem: str, ext: str = "csv") -> Path:
    return RESULTS_DIR / f"{normalize_asset(asset)}_{stem}.{ext}"


def load_data(asset: str) -> pd.DataFrame:
    path = processed_basis_path(asset, "calendar")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["weekday_num"] = df["date"].dt.weekday
    return df


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mondays = df[df["weekday_num"] == 0].copy()

    for _, mon in mondays.iterrows():
        monday = mon["date"]
        friday = monday - pd.Timedelta(days=3)
        saturday = monday - pd.Timedelta(days=2)
        sunday = monday - pd.Timedelta(days=1)
        thursday = monday - pd.Timedelta(days=4)

        friday_row = df[df["date"] == friday]
        saturday_row = df[df["date"] == saturday]
        sunday_row = df[df["date"] == sunday]
        monday_row = df[df["date"] == monday]
        thursday_row = df[df["date"] == thursday]

        if friday_row.empty or saturday_row.empty or sunday_row.empty or monday_row.empty or thursday_row.empty:
            continue

        friday_row = friday_row.iloc[0]
        saturday_row = saturday_row.iloc[0]
        sunday_row = sunday_row.iloc[0]
        monday_row = monday_row.iloc[0]
        thursday_row = thursday_row.iloc[0]

        weekend_token_return = float(sunday_row["token_log_price"] - friday_row["token_log_price"])
        monday_gold_return = float(monday_row["gold_log_price"] - friday_row["gold_log_price"])
        monday_token_return = float(monday_row["token_log_price"] - sunday_row["token_log_price"])
        weekend_volume_sum = float(saturday_row["token_total_volume"] + sunday_row["token_total_volume"])
        friday_basis = float(friday_row["basis"])
        sunday_basis = float(sunday_row["basis"])
        monday_basis = float(monday_row["basis"])

        rows.append(
            {
                "monday_date": monday,
                "weekend_token_return": weekend_token_return,
                "positive_weekend_return": max(weekend_token_return, 0.0),
                "negative_weekend_return": min(weekend_token_return, 0.0),
                "abs_weekend_token_return": abs(weekend_token_return),
                "monday_gold_return": monday_gold_return,
                "monday_token_return": monday_token_return,
                "weekend_volume_sum": weekend_volume_sum,
                "log_weekend_volume": float(np.log(max(weekend_volume_sum, 1.0))),
                "weekend_volume_share": float(
                    weekend_volume_sum
                    / max(
                        thursday_row["token_total_volume"]
                        + friday_row["token_total_volume"]
                        + saturday_row["token_total_volume"]
                        + sunday_row["token_total_volume"],
                        1.0,
                    )
                ),
                "friday_basis": friday_basis,
                "sunday_basis": sunday_basis,
                "monday_basis": monday_basis,
                "basis_widening_weekend": sunday_basis - friday_basis,
                "basis_reversal_monday": monday_basis - sunday_basis,
                "direction_match": int(np.sign(weekend_token_return) == np.sign(monday_gold_return)),
            }
        )

    return pd.DataFrame(rows).sort_values("monday_date").reset_index(drop=True)


def run_regression(events: pd.DataFrame, target: str, predictors: list[str], model_label: str) -> pd.DataFrame:
    X = pd.DataFrame(index=events.index)
    X["const"] = 1.0
    for predictor in predictors:
        X[predictor] = events[predictor]

    res = fit_ols(events[target], X, hac_lags=default_hac_lags(len(events)))
    return pd.DataFrame(
        {
            "model": model_label,
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


def build_summary(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "number_of_weekends", "value": len(events)},
            {"metric": "directional_accuracy", "value": float(events["direction_match"].mean())},
            {
                "metric": "corr_weekend_return_monday_gold",
                "value": float(events["weekend_token_return"].corr(events["monday_gold_return"])),
            },
            {
                "metric": "corr_weekend_volume_monday_gold_abs",
                "value": float(events["log_weekend_volume"].corr(events["monday_gold_return"].abs())),
            },
            {"metric": "avg_weekend_volume_share", "value": float(events["weekend_volume_share"].mean())},
        ]
    )


def run_rolling_model(events: pd.DataFrame, window: int) -> pd.DataFrame:
    rows = []
    if len(events) < window:
        return pd.DataFrame(rows)

    for end in range(window, len(events) + 1):
        sample = events.iloc[end - window : end].copy()
        X = pd.DataFrame(
            {
                "const": 1.0,
                "weekend_token_return": sample["weekend_token_return"],
                "friday_basis": sample["friday_basis"],
            }
        )
        res = fit_ols(sample["monday_gold_return"], X, hac_lags=default_hac_lags(len(sample)))
        rows.append(
            {
                "window_end_date": sample["monday_date"].iloc[-1],
                "window_start_date": sample["monday_date"].iloc[0],
                "window_size": len(sample),
                "weekend_beta": float(res.coefficients["weekend_token_return"]),
                "weekend_beta_t_stat_hac": float(res.t_stats["weekend_token_return"]),
                "friday_basis_beta": float(res.coefficients["friday_basis"]),
                "r_squared": float(res.r_squared),
            }
        )
    return pd.DataFrame(rows)


def plot_rolling(rolling: pd.DataFrame, asset: str) -> Path | None:
    if rolling.empty:
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rolling["window_end_date"], rolling["weekend_beta"], color="#1f77b4", linewidth=1.8)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_title(f"{asset_name(asset)} rolling weekend-return beta for Monday gold return")
    ax.set_xlabel("Window end date")
    ax.set_ylabel("Rolling beta")
    fig.tight_layout()

    out = result_path(asset, "option2_rolling_weekend_beta", ext="png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    asset = normalize_asset(args.asset)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    events = build_events(load_data(asset))
    summary = build_summary(events)

    # Formal asymmetry test (H3b) via an interaction parametrisation:
    #   monday_gold = a + b1*weekend_return + b2*negative_weekend_return + controls
    # b1 is the slope for POSITIVE weekends; (b1 + b2) is the slope for NEGATIVE
    # weekends. Therefore b2 = (negative slope minus positive slope) IS the
    # asymmetry, and its HAC t-stat directly tests whether negative weekend moves
    # predict Monday gold more strongly than positive ones. This replaces the
    # informal "compare two separate coefficients" reading, which has no test of
    # the difference.
    asym_diff = run_regression(
        events,
        target="monday_gold_return",
        predictors=["weekend_token_return", "negative_weekend_return", "log_weekend_volume", "friday_basis"],
        model_label="Asymmetry difference test (neg minus pos slope)",
    )

    regressions = pd.concat(
        [
            run_regression(
                events,
                target="monday_gold_return",
                predictors=["weekend_token_return", "log_weekend_volume", "friday_basis"],
                model_label="Baseline Monday gold return",
            ),
            run_regression(
                events,
                target="monday_gold_return",
                predictors=["positive_weekend_return", "negative_weekend_return", "log_weekend_volume", "friday_basis"],
                model_label="Asymmetry Monday gold return",
            ),
            asym_diff,
            run_regression(
                events,
                target="basis_reversal_monday",
                predictors=["basis_widening_weekend", "log_weekend_volume", "friday_basis"],
                model_label="Monday basis reversal",
            ),
        ],
        ignore_index=True,
    )

    # Promote the asymmetry test to headline summary metrics.
    diff_row = asym_diff[asym_diff["term"] == "negative_weekend_return"].iloc[0]
    asym_beta = float(diff_row["coefficient"])
    asym_t = float(diff_row["t_stat_hac"])
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {"metric": "asymmetry_diff_beta_neg_minus_pos", "value": asym_beta},
                    {"metric": "asymmetry_diff_t_stat_hac", "value": asym_t},
                    {"metric": "asymmetry_significant_5pct", "value": float(abs(asym_t) > 1.96)},
                ]
            ),
        ],
        ignore_index=True,
    )

    rolling = run_rolling_model(events, window=args.window)
    plot_path = plot_rolling(rolling, asset) if args.plot else None

    summary_path = result_path(asset, "option2_summary")
    regressions_path = result_path(asset, "option2_regressions")
    rolling_path = result_path(asset, "option2_rolling_betas")
    events_path = result_path(asset, "option2_events")

    summary.to_csv(summary_path, index=False)
    regressions.to_csv(regressions_path, index=False)
    rolling.to_csv(rolling_path, index=False)
    events.to_csv(events_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {regressions_path}")
    print(f"Saved: {rolling_path}")
    print(f"Saved: {events_path}")
    if plot_path is not None:
        print(f"Saved: {plot_path}")
    print()
    print(f"{asset_name(asset)} Dissertation 3 option-2 summary")
    print(summary.to_string(index=False))
    print()
    print("Key regression coefficients")
    key_terms = ["weekend_token_return", "positive_weekend_return", "negative_weekend_return", "friday_basis"]
    print(regressions[regressions["term"].isin(key_terms)].to_string(index=False))
    print()
    verdict = "SIGNIFICANT" if abs(asym_t) > 1.96 else "not significant"
    print(
        f"Asymmetry test (H3b): negative-minus-positive slope = {asym_beta:+.4f}, "
        f"HAC t = {asym_t:+.2f} -> {verdict} at 5%. "
        f"(Positive => negative weekend moves predict Monday gold more strongly.)"
    )


if __name__ == "__main__":
    main()
