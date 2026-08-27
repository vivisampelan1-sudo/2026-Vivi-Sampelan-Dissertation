"""Does weekend LIQUIDITY strengthen the weekend price-discovery signal?

Motivation: tokenization's promise is 24/7 access. But an open market is not
necessarily an INFORMATIVE one. If the weekend signal is stronger when weekend
trading is heavier, that is evidence the signal reflects genuine price discovery
rather than thin-market noise.

Design care: raw token volume grows enormously over the sample, so a naive
high/low volume split would simply select recent years. Two DETRENDED liquidity
measures are used instead:
  1. weekend_volume_share = weekend volume / (Thu+Fri+Sat+Sun volume)
  2. relative_volume      = log(weekend vol) - trailing 8-weekend mean

All regressions use HAC (Newey-West) standard errors.
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
RAW_DIR = DISS3_DIR / "data" / "raw"
ASSETS = ["paxg", "xaut"]


def load_token(asset: str) -> pd.DataFrame:
    r = json.load(open(RAW_DIR / f"{normalize_asset(asset)}_yahoo_daily_full.json"))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"ts": r["timestamp"], "close": q["close"], "volume": q.get("volume")})
    d["date"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    return d.dropna(subset=["close"]).drop(columns=["ts"]).sort_values("date").reset_index(drop=True)


def load_gold_spot() -> pd.DataFrame:
    d = pd.DataFrame(json.load(open(RAW_DIR / "gold_alphavantage_daily.json"))["data"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["price"] = d["price"].astype(float)
    return d.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)


def build_events(asset: str) -> pd.DataFrame:
    tok = load_token(asset).set_index("date")
    gold = load_gold_spot()
    gprice = gold.set_index("date")["price"]

    rows = []
    for monday in gold["date"]:
        if monday.weekday() != 0:
            continue
        thu = monday - pd.Timedelta(days=4)
        fri, sat, sun = monday - pd.Timedelta(days=3), monday - pd.Timedelta(days=2), monday - pd.Timedelta(days=1)
        if not all(d in tok.index for d in (thu, fri, sat, sun)):
            continue
        if fri not in gprice.index or monday not in gprice.index:
            continue

        f, s, u, t = tok.loc[fri], tok.loc[sat], tok.loc[sun], tok.loc[thu]
        wknd_vol = float((s["volume"] or 0) + (u["volume"] or 0))
        surround = float((t["volume"] or 0) + (f["volume"] or 0)) + wknd_vol
        if wknd_vol <= 0 or surround <= 0:
            continue

        rows.append(
            {
                "monday_date": monday,
                "weekend_return": float(np.log(u["close"]) - np.log(f["close"])),
                "weekend_volume": wknd_vol,
                "log_weekend_volume": float(np.log(wknd_vol)),
                "weekend_volume_share": wknd_vol / surround,
                "friday_basis": float(np.log(f["close"]) - np.log(gprice.loc[fri])),
                "monday_gold_return": float(np.log(gprice.loc[monday]) - np.log(gprice.loc[fri])),
            }
        )

    ev = pd.DataFrame(rows).sort_values("monday_date").reset_index(drop=True)
    # detrended relative volume: log weekend volume minus its trailing 8-weekend mean
    ev["relative_volume"] = ev["log_weekend_volume"] - ev["log_weekend_volume"].rolling(8, min_periods=4).mean()
    return ev.dropna().reset_index(drop=True)


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def run_asset(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = build_events(asset)
    n = len(ev)
    detail_frames, summary_rows = [], []

    # is the liquidity measure detrended? (correlation with time index)
    t_index = pd.Series(np.arange(n), index=ev.index)
    for measure in ["weekend_volume_share", "relative_volume"]:
        summary_rows.append(
            {"asset": asset_name(asset), "metric": f"{measure}_corr_with_time", "value": float(ev[measure].corr(t_index))}
        )

    for measure in ["weekend_volume_share", "relative_volume"]:
        liq = z(ev[measure])
        base = pd.DataFrame({"const": 1.0, "weekend_return": ev["weekend_return"], "friday_basis": ev["friday_basis"]})

        # (A) continuous interaction
        Xc = base.copy()
        Xc["liq"] = liq
        Xc["weekend_return_x_liq"] = ev["weekend_return"] * liq
        rc = fit_ols(ev["monday_gold_return"], Xc, hac_lags=default_hac_lags(n))

        # (B) high/low split via dummy interaction
        high = (ev[measure] > ev[measure].median()).astype(float)
        Xd = base.copy()
        Xd["high_liq"] = high
        Xd["weekend_return_x_high_liq"] = ev["weekend_return"] * high
        rd = fit_ols(ev["monday_gold_return"], Xd, hac_lags=default_hac_lags(n))

        beta_low = float(rd.coefficients["weekend_return"])
        beta_high = beta_low + float(rd.coefficients["weekend_return_x_high_liq"])

        summary_rows.extend(
            [
                {"asset": asset_name(asset), "metric": f"{measure}__continuous_interaction_beta", "value": float(rc.coefficients["weekend_return_x_liq"])},
                {"asset": asset_name(asset), "metric": f"{measure}__continuous_interaction_t", "value": float(rc.t_stats["weekend_return_x_liq"])},
                {"asset": asset_name(asset), "metric": f"{measure}__beta_low_liquidity", "value": beta_low},
                {"asset": asset_name(asset), "metric": f"{measure}__beta_high_liquidity", "value": beta_high},
                {"asset": asset_name(asset), "metric": f"{measure}__high_minus_low_t", "value": float(rd.t_stats["weekend_return_x_high_liq"])},
            ]
        )

        for label, res in [(f"{measure} continuous interaction", rc), (f"{measure} high/low split", rd)]:
            for term in res.coefficients.index:
                detail_frames.append(
                    {
                        "asset": asset_name(asset), "model": label, "term": term,
                        "coefficient": float(res.coefficients[term]),
                        "std_error_hac": float(res.std_errors[term]),
                        "t_stat_hac": float(res.t_stats[term]),
                        "r_squared": float(res.r_squared), "n_obs": int(res.n_obs),
                    }
                )

    summary_rows.append({"asset": asset_name(asset), "metric": "n_weekends", "value": float(n)})
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_frames)


def main() -> None:
    warnings.filterwarnings("ignore")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    s_all, d_all = [], []
    for a in ASSETS:
        s, d = run_asset(normalize_asset(a))
        s_all.append(s); d_all.append(d)
    summary = pd.concat(s_all, ignore_index=True)
    details = pd.concat(d_all, ignore_index=True)
    summary.to_csv(RESULTS_DIR / "liquidity_conditioning_summary.csv", index=False)
    details.to_csv(RESULTS_DIR / "liquidity_conditioning_details.csv", index=False)
    print("Saved: liquidity_conditioning_summary.csv / _details.csv")
    print()
    print(summary.pivot(index="metric", columns="asset", values="value").to_string())


if __name__ == "__main__":
    main()
