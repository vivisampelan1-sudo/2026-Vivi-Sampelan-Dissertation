"""Extension beyond price returns: weekend VOLATILITY and MOMENTUM.

Addresses the request to test more than price levels/returns:
  (1) Volatility transmission - does a volatile token weekend predict a LARGER
      Monday gold move (magnitude), regardless of direction?
  (2) Momentum - does the token's pre-weekend trend add predictive power on top
      of the weekend return itself?
  (3) Volatility conditioning - is the weekend return signal STRONGER when the
      weekend was turbulent?

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
RAW_DIR = DISS3_DIR / "04_data" / "raw"
ASSETS = ["paxg", "xaut"]


def load_token(asset: str) -> pd.DataFrame:
    r = json.load(open(RAW_DIR / f"{normalize_asset(asset)}_yahoo_daily_full.json"))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame(
        {
            "ts": r["timestamp"],
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        }
    )
    d["date"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    return d.dropna(subset=["close"]).drop(columns=["ts"]).sort_values("date").reset_index(drop=True)


def load_gold_spot() -> pd.DataFrame:
    d = pd.DataFrame(json.load(open(RAW_DIR / "gold_alphavantage_daily.json"))["data"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["price"] = d["price"].astype(float)
    return d.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)


def build_events(asset: str) -> pd.DataFrame:
    tok = load_token(asset)
    gold = load_gold_spot()
    tclose = tok.set_index("date")["close"]
    gprice = gold.set_index("date")["price"]
    tok_idx = tok.set_index("date")

    rows = []
    for monday in gold["date"]:
        if monday.weekday() != 0:
            continue
        fri, sat, sun = monday - pd.Timedelta(days=3), monday - pd.Timedelta(days=2), monday - pd.Timedelta(days=1)
        if not all(d in tok_idx.index for d in (fri, sat, sun)):
            continue
        if fri not in gprice.index or monday not in gprice.index:
            continue

        f, s, u = tok_idx.loc[fri], tok_idx.loc[sat], tok_idx.loc[sun]
        r_sat = float(np.log(s["close"]) - np.log(f["close"]))
        r_sun = float(np.log(u["close"]) - np.log(s["close"]))

        # weekend realised volatility (magnitude of weekend turbulence)
        weekend_rv = float(np.sqrt(r_sat**2 + r_sun**2))
        # Parkinson-style range volatility across the two weekend bars
        ranges = []
        for bar in (s, u):
            if pd.notna(bar["high"]) and pd.notna(bar["low"]) and bar["low"] > 0:
                ranges.append(float(np.log(bar["high"] / bar["low"])))
        weekend_range = float(np.mean(ranges)) if ranges else np.nan

        # momentum: token trend BEFORE the weekend (uses only data up to Friday)
        def mom(days: int) -> float:
            past = tclose.asof(fri - pd.Timedelta(days=days))
            return float(np.log(f["close"]) - np.log(past)) if pd.notna(past) else np.nan

        vol_weekend = float((s["volume"] or 0) + (u["volume"] or 0))
        rows.append(
            {
                "monday_date": monday,
                "weekend_return": float(np.log(u["close"]) - np.log(f["close"])),
                "abs_weekend_return": abs(float(np.log(u["close"]) - np.log(f["close"]))),
                "weekend_rv": weekend_rv,
                "weekend_range": weekend_range,
                "momentum_5": mom(5),
                "momentum_20": mom(20),
                "log_weekend_volume": float(np.log(max(vol_weekend, 1.0))),
                "friday_basis": float(np.log(f["close"]) - np.log(gprice.loc[fri])),
                "monday_gold_return": float(np.log(gprice.loc[monday]) - np.log(gprice.loc[fri])),
            }
        )

    ev = pd.DataFrame(rows).dropna().reset_index(drop=True)
    ev["abs_monday_gold_return"] = ev["monday_gold_return"].abs()
    ev["high_vol_weekend"] = (ev["weekend_rv"] > ev["weekend_rv"].median()).astype(float)
    ev["weekend_return_x_highvol"] = ev["weekend_return"] * ev["high_vol_weekend"]
    return ev


def reg(ev: pd.DataFrame, target: str, predictors: list[str], label: str, asset: str) -> pd.DataFrame:
    X = pd.DataFrame(index=ev.index)
    X["const"] = 1.0
    for p in predictors:
        X[p] = ev[p]
    res = fit_ols(ev[target], X, hac_lags=default_hac_lags(len(ev)))
    return pd.DataFrame(
        {
            "asset": asset_name(asset),
            "model": label,
            "target": target,
            "term": res.coefficients.index,
            "coefficient": res.coefficients.values,
            "std_error_hac": res.std_errors.values,
            "t_stat_hac": res.t_stats.values,
            "r_squared": res.r_squared,
            "n_obs": res.n_obs,
        }
    )


def run_asset(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = build_events(asset)
    base = ["log_weekend_volume", "friday_basis"]
    out = pd.concat(
        [
            reg(ev, "monday_gold_return", ["weekend_return"] + base,
                "A. Baseline (return only)", asset),
            reg(ev, "monday_gold_return", ["weekend_return", "momentum_5", "momentum_20"] + base,
                "B. + Momentum", asset),
            reg(ev, "abs_monday_gold_return", ["weekend_rv"] + base,
                "C. Volatility -> Monday move size", asset),
            reg(ev, "abs_monday_gold_return", ["weekend_range"] + base,
                "D. Range vol -> Monday move size", asset),
            reg(ev, "monday_gold_return", ["weekend_return", "weekend_return_x_highvol"] + base,
                "E. Return x high-vol weekend", asset),
        ],
        ignore_index=True,
    )

    def g(model, term, col="t_stat_hac"):
        m = out[(out.model == model) & (out.term == term)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    summary = pd.DataFrame(
        [
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "n_weekends", "value": float(len(ev))},
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "baseline_weekend_beta", "value": g("A. Baseline (return only)", "weekend_return", "coefficient")},
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "baseline_weekend_t", "value": g("A. Baseline (return only)", "weekend_return")},
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "momentum_5_t", "value": g("B. + Momentum", "momentum_5")},
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "momentum_20_t", "value": g("B. + Momentum", "momentum_20")},
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "weekend_rv_beta", "value": g("C. Volatility -> Monday move size", "weekend_rv", "coefficient")},
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "weekend_rv_t", "value": g("C. Volatility -> Monday move size", "weekend_rv")},
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "weekend_range_t", "value": g("D. Range vol -> Monday move size", "weekend_range")},
            {"asset": asset_name(asset), "hypothesis": "H4", "metric": "highvol_interaction_t", "value": g("E. Return x high-vol weekend", "weekend_return_x_highvol")},
        ]
    )
    return summary, out


def render_md(summary: pd.DataFrame) -> str:
    def p(a, m):
        v = summary[(summary.asset == a) & (summary.metric == m)].value
        return f"{float(v.iloc[0]):.3f}" if len(v) else "n/a"

    lines = [
        "# Beyond price: weekend VOLATILITY and MOMENTUM",
        "",
        "Extends the weekend test past price returns. All HAC t-stats; |t| > 1.96 = significant.",
        "",
        "| Metric | PAXG | XAUT |",
        "| --- | ---: | ---: |",
        f"| Weekends (n) | {p('PAXG','n_weekends')} | {p('XAUT','n_weekends')} |",
        f"| Baseline weekend beta | {p('PAXG','baseline_weekend_beta')} | {p('XAUT','baseline_weekend_beta')} |",
        f"| Baseline weekend t | {p('PAXG','baseline_weekend_t')} | {p('XAUT','baseline_weekend_t')} |",
        f"| Momentum 5d (t) | {p('PAXG','momentum_5_t')} | {p('XAUT','momentum_5_t')} |",
        f"| Momentum 20d (t) | {p('PAXG','momentum_20_t')} | {p('XAUT','momentum_20_t')} |",
        f"| Weekend volatility -> Monday move size (beta) | {p('PAXG','weekend_rv_beta')} | {p('XAUT','weekend_rv_beta')} |",
        f"| Weekend volatility -> Monday move size (t) | {p('PAXG','weekend_rv_t')} | {p('XAUT','weekend_rv_t')} |",
        f"| Range volatility (t) | {p('PAXG','weekend_range_t')} | {p('XAUT','weekend_range_t')} |",
        f"| Return x high-vol interaction (t) | {p('PAXG','highvol_interaction_t')} | {p('XAUT','highvol_interaction_t')} |",
        "",
        "## How to read",
        "",
        "- **Momentum t**: does the token's pre-weekend trend add predictive power beyond the weekend move? Significant = yes.",
        "- **Weekend volatility -> Monday move size**: does a turbulent token weekend forecast a LARGER Monday gold move (either direction)? This is volatility/uncertainty transmission, not direction.",
        "- **Return x high-vol interaction**: is the weekend return signal stronger on turbulent weekends? Significant positive = yes.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    warnings.filterwarnings("ignore")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    s_all, d_all = [], []
    for a in ASSETS:
        s, d = run_asset(normalize_asset(a))
        s_all.append(s)
        d_all.append(d)
    summary = pd.concat(s_all, ignore_index=True)
    details = pd.concat(d_all, ignore_index=True)

    summary.to_csv(RESULTS_DIR / "volatility_momentum_summary.csv", index=False)
    details.to_csv(RESULTS_DIR / "volatility_momentum_details.csv", index=False)
    (RESULTS_DIR / "volatility_momentum_summary.md").write_text(render_md(summary), encoding="utf-8")

    print("Saved: volatility_momentum_summary.csv / _details.csv / _summary.md")
    print()
    print(summary.pivot(index="metric", columns="asset", values="value").to_string())


if __name__ == "__main__":
    main()
