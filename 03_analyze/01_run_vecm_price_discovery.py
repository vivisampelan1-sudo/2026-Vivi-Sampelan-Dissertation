from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
PROJECT_ROOT = DISS3_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from asset_config import asset_name, normalize_asset, processed_basis_path  # noqa: E402
from econometrics_utils import adf_test, default_hac_lags, fit_ols  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"
RAW_DIR = DISS3_DIR / "04_data" / "raw"
ASSETS = ["paxg", "xaut"]
BENCHMARKS = ["spot", "futures"]
MAX_DIFF_LAGS = 5
RESID_DIAG_LAGS = 5
GG_BOOTSTRAP_REPS = 1000
GG_BOOTSTRAP_BLOCK = 20
GG_BOOTSTRAP_SEED = 20260823


def result_path(stem: str, ext: str = "csv") -> Path:
    return RESULTS_DIR / f"vecm_price_discovery_{stem}.{ext}"


def load_weekday_data_spot(asset: str) -> pd.DataFrame:
    path = processed_basis_path(asset, "weekday")
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def _yahoo_close(filename: str) -> pd.DataFrame:
    r = json.load(open(RAW_DIR / filename))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"ts": r["timestamp"], "close": q["close"]})
    d["date"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    return d.dropna(subset=["close"])[["date", "close"]]


def load_weekday_data_futures(asset: str) -> pd.DataFrame:
    """Token vs gold FUTURES (GC=F), weekday-synchronised. Futures carry a real
    exchange close, so this benchmark is not exposed to the spot series'
    undocumented/fuzzy closing time."""
    tok = _yahoo_close(f"{normalize_asset(asset)}_yahoo_daily_full.json")
    fut = _yahoo_close("gold_yahoo_gc_f_1y.json")
    m = tok.merge(fut, on="date", suffixes=("_t", "_g")).sort_values("date").reset_index(drop=True)
    return pd.DataFrame(
        {
            "date": m["date"],
            "token_log_price": np.log(m["close_t"]),
            "gold_log_price": np.log(m["close_g"]),
        }
    )


def load_weekday_data(asset: str, benchmark: str) -> pd.DataFrame:
    return load_weekday_data_spot(asset) if benchmark == "spot" else load_weekday_data_futures(asset)


def estimate_cointegration(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, object]:
    sample = df[["date", "token_log_price", "gold_log_price"]].dropna().copy()
    X = pd.DataFrame({"const": 1.0, "gold_log_price": sample["gold_log_price"]})
    long_run = fit_ols(sample["token_log_price"], X, hac_lags=default_hac_lags(len(sample)))
    ect = sample["token_log_price"] - (
        long_run.coefficients["const"] + long_run.coefficients["gold_log_price"] * sample["gold_log_price"]
    )
    return sample["date"], ect, long_run


def build_vecm_sample(df: pd.DataFrame, ect: pd.Series, q_lags: int) -> pd.DataFrame:
    sample = df[["date", "token_log_price", "gold_log_price"]].dropna().copy().reset_index(drop=True)
    sample["ect_lag"] = pd.Series(ect).shift(1).to_numpy()
    sample["d_token"] = sample["token_log_price"].diff()
    sample["d_gold"] = sample["gold_log_price"].diff()

    for lag in range(1, q_lags + 1):
        sample[f"d_token_lag_{lag}"] = sample["d_token"].shift(lag)
        sample[f"d_gold_lag_{lag}"] = sample["d_gold"].shift(lag)

    return sample.dropna().reset_index(drop=True)


def fit_vecm_equations(sample: pd.DataFrame, q_lags: int) -> tuple[object, object, list[str]]:
    regressors = ["ect_lag"]
    for lag in range(1, q_lags + 1):
        regressors.extend([f"d_token_lag_{lag}", f"d_gold_lag_{lag}"])

    X = sample[regressors].copy()
    token_res = fit_ols(sample["d_token"], X, hac_lags=default_hac_lags(len(sample)))
    gold_res = fit_ols(sample["d_gold"], X, hac_lags=default_hac_lags(len(sample)))
    return token_res, gold_res, regressors


def system_bic(token_res: object, gold_res: object, n_obs: int, n_params_per_eq: int) -> float:
    resid = np.column_stack([token_res.residuals, gold_res.residuals])
    sigma = (resid.T @ resid) / n_obs
    det_sigma = float(np.linalg.det(sigma))
    if det_sigma <= 0:
        det_sigma = 1e-12
    total_params = 2 * n_params_per_eq
    return n_obs * np.log(det_sigma) + total_params * np.log(n_obs)


def choose_vecm_lag(df: pd.DataFrame, ect: pd.Series, max_diff_lags: int = MAX_DIFF_LAGS):
    best = None
    for q_lags in range(0, max_diff_lags + 1):
        sample = build_vecm_sample(df, ect, q_lags)
        if len(sample) < 40:
            continue
        token_res, gold_res, regressors = fit_vecm_equations(sample, q_lags)
        bic = system_bic(token_res, gold_res, len(sample), len(regressors))
        candidate = (bic, q_lags, sample, token_res, gold_res)
        if best is None or bic < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("VECM lag selection failed.")
    _, q_lags, sample, token_res, gold_res = best
    return q_lags, sample, token_res, gold_res


def gonzalo_granger_shares(alpha_token: float, alpha_gold: float) -> tuple[float, float]:
    denom = alpha_gold - alpha_token
    if abs(denom) < 1e-12:
        return np.nan, np.nan
    return float(alpha_gold / denom), float(-alpha_token / denom)


def durbin_watson(resid: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=float)
    if resid.size < 2:
        return np.nan
    num = np.sum(np.diff(resid) ** 2)
    den = np.sum(resid**2)
    return float(num / den) if den > 0 else np.nan


def ljung_box_q(resid: np.ndarray, lags: int) -> tuple[float, float]:
    resid = np.asarray(resid, dtype=float)
    n_obs = resid.size
    if n_obs < 3:
        return np.nan, np.nan

    max_lag = min(lags, n_obs - 1)
    if max_lag < 1:
        return np.nan, np.nan

    centered = resid - resid.mean()
    denom = np.sum(centered**2)
    if denom <= 0:
        return np.nan, np.nan

    q_stat = 0.0
    for lag in range(1, max_lag + 1):
        rho = float(np.sum(centered[lag:] * centered[:-lag]) / denom)
        q_stat += (rho**2) / (n_obs - lag)
    q_stat *= n_obs * (n_obs + 2)
    p_value = float(chi2.sf(q_stat, df=max_lag))
    return float(q_stat), p_value


def circular_block_indices(n_obs: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n_obs <= 0:
        return np.array([], dtype=int)
    block_size = max(1, min(block_size, n_obs))
    starts = rng.integers(0, n_obs, size=int(np.ceil(n_obs / block_size)))
    idx = []
    for start in starts:
        idx.extend((start + np.arange(block_size)) % n_obs)
    return np.asarray(idx[:n_obs], dtype=int)


def bootstrap_gg_share_ci(
    sample: pd.DataFrame,
    q_lags: int,
    reps: int = GG_BOOTSTRAP_REPS,
    block_size: int = GG_BOOTSTRAP_BLOCK,
    seed: int = GG_BOOTSTRAP_SEED,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    boot_shares: list[float] = []

    for _ in range(reps):
        boot_idx = circular_block_indices(len(sample), block_size, rng)
        boot_sample = sample.iloc[boot_idx].reset_index(drop=True)
        try:
            token_res, gold_res, _ = fit_vecm_equations(boot_sample, q_lags)
            share, _ = gonzalo_granger_shares(
                float(token_res.coefficients["ect_lag"]),
                float(gold_res.coefficients["ect_lag"]),
            )
        except Exception:
            continue
        if np.isfinite(share):
            boot_shares.append(float(share))

    if not boot_shares:
        return np.nan, np.nan, 0

    arr = np.asarray(boot_shares, dtype=float)
    lower = float(np.quantile(arr, 0.025))
    upper = float(np.quantile(arr, 0.975))
    return lower, upper, int(arr.size)


def run_asset(asset: str, benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_weekday_data(asset, benchmark)
    dates, ect, long_run = estimate_cointegration(df)
    ect_adf = adf_test(
        ect,
        regression="n",
        test_name=f"{asset_name(asset)} ({benchmark}) Engle-Granger residual ADF",
        critical_values={"1%": -3.90, "5%": -3.34, "10%": -3.05},
        note="Residual stationarity test supporting the bivariate VECM specification.",
    )

    q_lags, vecm_sample, token_res, gold_res = choose_vecm_lag(df, ect)
    alpha_token = float(token_res.coefficients["ect_lag"])
    alpha_gold = float(gold_res.coefficients["ect_lag"])
    token_share, gold_share = gonzalo_granger_shares(alpha_token, alpha_gold)
    gg_ci_low, gg_ci_high, gg_boot_n = bootstrap_gg_share_ci(vecm_sample, q_lags)
    diag_lags = min(RESID_DIAG_LAGS, max(1, len(vecm_sample) // 10))
    token_lb_q, token_lb_p = ljung_box_q(token_res.residuals, diag_lags)
    gold_lb_q, gold_lb_p = ljung_box_q(gold_res.residuals, diag_lags)
    resid_cross_corr = float(np.corrcoef(token_res.residuals, gold_res.residuals)[0, 1])

    def row(metric, value, hypothesis="H2"):
        return {"asset": asset_name(asset), "benchmark": benchmark, "hypothesis": hypothesis, "metric": metric, "value": value}

    summary = pd.DataFrame(
        [
            row("weekday_rows", int(len(df.dropna(subset=["token_log_price", "gold_log_price"])))),
            row("selected_diff_lags", int(q_lags)),
            row("vecm_rows", int(len(vecm_sample))),
            row("long_run_gold_beta", float(long_run.coefficients["gold_log_price"]), hypothesis="H1"),
            row("engle_granger_residual_adf_stat", float(ect_adf.test_stat), hypothesis="H1"),
            row("engle_granger_crit_5pct", float(ect_adf.critical_values["5%"]), hypothesis="H1"),
            row("cointegration_supported", bool(ect_adf.test_stat < ect_adf.critical_values["5%"]), hypothesis="H1"),
            row("token_adjustment_alpha", alpha_token),
            row("token_adjustment_t_stat_hac", float(token_res.t_stats["ect_lag"])),
            row("gold_adjustment_alpha", alpha_gold),
            row("gold_adjustment_t_stat_hac", float(gold_res.t_stats["ect_lag"])),
            row("token_gonzalo_granger_share", token_share),
            row("gold_gonzalo_granger_share", gold_share),
            row("token_gg_share_ci_lower", gg_ci_low),
            row("token_gg_share_ci_upper", gg_ci_high),
            row("token_gg_bootstrap_reps", gg_boot_n),
            row("token_gg_share_ci_excludes_0_5", bool((gg_ci_low > 0.5) or (gg_ci_high < 0.5))),
            row("token_majority_price_discovery_share", bool(token_share > 0.5) if not np.isnan(token_share) else False),
            row("residual_diagnostic_lags", int(diag_lags)),
            row("token_resid_durbin_watson", durbin_watson(token_res.residuals)),
            row("gold_resid_durbin_watson", durbin_watson(gold_res.residuals)),
            row("token_resid_ljung_box_q", token_lb_q),
            row("token_resid_ljung_box_p", token_lb_p),
            row("gold_resid_ljung_box_q", gold_lb_q),
            row("gold_resid_ljung_box_p", gold_lb_p),
            row("vecm_residual_cross_correlation", resid_cross_corr),
        ]
    )

    detail_rows = []
    for model_name, res in [("Token equation", token_res), ("Gold equation", gold_res), ("Long-run regression", long_run)]:
        for term in res.coefficients.index:
            detail_rows.append(
                {
                    "asset": asset_name(asset),
                    "benchmark": benchmark,
                    "model": model_name,
                    "term": term,
                    "coefficient": float(res.coefficients[term]),
                    "std_error_hac": float(res.std_errors[term]),
                    "t_stat_hac": float(res.t_stats[term]),
                    "r_squared": float(res.r_squared),
                    "n_obs": int(res.n_obs),
                }
            )
    return summary, pd.DataFrame(detail_rows)


def render_markdown(summary: pd.DataFrame) -> str:
    def pick(asset, benchmark, metric):
        v = summary.loc[
            (summary["asset"] == asset) & (summary["benchmark"] == benchmark) & (summary["metric"] == metric),
            "value",
        ].iloc[0]
        if isinstance(v, bool):
            return "Yes" if v else "No"
        return f"{float(v):.3f}"

    lines = [
        "# VECM Price Discovery Summary (H2) — spot vs futures benchmark",
        "",
        "Weekday-synchronised bivariate VECM with Gonzalo-Granger price-discovery shares, estimated against **both** gold benchmarks. The token's share above 0.5 means the token leads; below 0.5 means the token follows.",
        "",
        "| Asset | Benchmark | Cointegrated? | Token alpha | Gold alpha | Token GG share | 95% bootstrap CI | Leads? |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for asset in ["PAXG", "XAUT"]:
        for b in ["spot", "futures"]:
            share = float(pick(asset, b, "token_gonzalo_granger_share"))
            ci_low = pick(asset, b, "token_gg_share_ci_lower")
            ci_high = pick(asset, b, "token_gg_share_ci_upper")
            leads = "token leads" if share > 0.5 else "token follows"
            lines.append(
                f"| {asset} | {b} | {pick(asset, b, 'cointegration_supported')} | "
                f"{pick(asset, b, 'token_adjustment_alpha')} | {pick(asset, b, 'gold_adjustment_alpha')} | "
                f"{pick(asset, b, 'token_gonzalo_granger_share')} | [{ci_low}, {ci_high}] | {leads} |"
            )

    lines.extend(
        [
            "",
            "## Residual diagnostics",
            "",
            "| Asset | Benchmark | DW token | DW gold | LB p-value token | LB p-value gold | Residual corr. |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for asset in ["PAXG", "XAUT"]:
        for b in ["spot", "futures"]:
            lines.append(
                f"| {asset} | {b} | {pick(asset, b, 'token_resid_durbin_watson')} | "
                f"{pick(asset, b, 'gold_resid_durbin_watson')} | {pick(asset, b, 'token_resid_ljung_box_p')} | "
                f"{pick(asset, b, 'gold_resid_ljung_box_p')} | {pick(asset, b, 'vecm_residual_cross_correlation')} |"
            )

    lines.extend(
        [
            "",
            "## Reading (honest conclusion)",
            "",
            "- **The weekday leadership result is not robust to the benchmark.** Against spot, the token appears to lead (GG share > 0.5); against gold futures, the token **follows** (GG share < 0.5).",
            "- This reversal is consistent with the non-synchronous-close bias: the token closes at midnight UTC while spot has no timestamp and futures close on a New York basis. The token's later daily snapshot can inflate its apparent leadership against the fuzzy spot series.",
            "- It is also economically sensible: COMEX gold futures are the deep, dominant gold price-discovery venue, so a small token following the futures on weekdays is the expected outcome.",
            "- Residual diagnostics are reported as Durbin-Watson statistics and Ljung-Box p-values. The Gonzalo-Granger shares are supplemented with circular block-bootstrap confidence intervals, but they should still be read cautiously because benchmark timing and continuous-futures stitching can affect the exact point estimate.",
            "- **Conclusion for H2:** we do not claim robust weekday price-discovery leadership. On weekdays the token behaves as a follower/tracker; its genuine independent informational role appears at the weekend (H3), where identification does not depend on closing-time alignment.",
            "- Spot is retained as the conceptual benchmark (the token represents physical gold); the futures result is the timing-robust diagnostic that reveals the leadership claim is fragile.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_frames, detail_frames = [], []
    for asset in ASSETS:
        for benchmark in BENCHMARKS:
            s, d = run_asset(normalize_asset(asset), benchmark)
            summary_frames.append(s)
            detail_frames.append(d)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    details_df = pd.concat(detail_frames, ignore_index=True)

    summary_df.to_csv(result_path("summary"), index=False)
    details_df.to_csv(result_path("details"), index=False)
    result_path("summary", ext="md").write_text(render_markdown(summary_df), encoding="utf-8")

    print(f"Saved: {result_path('summary')}")
    print(f"Saved: {result_path('details')}")
    print(f"Saved: {result_path('summary', ext='md')}")
    print()
    print(summary_df[summary_df["metric"].isin(["token_gonzalo_granger_share", "cointegration_supported"])].to_string(index=False))


if __name__ == "__main__":
    main()
