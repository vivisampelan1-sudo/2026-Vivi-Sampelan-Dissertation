"""Placebo and decay tests: is the weekend effect really caused by the CLOSURE?

Challenge: if the token simply predicts gold in general (generic lead-lag), the
same relationship should appear mid-week, when gold is open. If it does not, the
weekend result is genuinely driven by the market closure.

(A) DECAY  - the weekend signal should predict MONDAY (the reopen) and then fade
             on Tuesday and Wednesday, because the information is absorbed at the
             reopen.
(B) PLACEBO - the identical structure applied to a MID-WEEK window where gold was
             OPEN throughout: a 2-day token move (Mon->Wed) predicting gold's next
             move (Wed->Thu). Gold already traded during the signal window, so a
             closure-driven effect should be absent here.

All regressions use HAC (Newey-West) standard errors, with the basis as control.
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


def load_token(asset: str) -> pd.Series:
    r = json.load(open(RAW_DIR / f"{normalize_asset(asset)}_yahoo_daily_full.json"))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"ts": r["timestamp"], "close": q["close"]}).dropna(subset=["close"])
    d["date"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    return d.set_index("date")["close"].sort_index()


def load_gold() -> pd.Series:
    d = pd.DataFrame(json.load(open(RAW_DIR / "gold_alphavantage_daily.json"))["data"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["price"] = d["price"].astype(float)
    return d.dropna(subset=["price"]).set_index("date")["price"].sort_index()


def load_core_weekend_events(asset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{normalize_asset(asset)}_option2_events.csv", parse_dates=["monday_date"])
    return df.rename(columns={"monday_date": "date"}).sort_values("date").reset_index(drop=True)


def lg(series: pd.Series, d) -> float:
    return float(np.log(series.loc[d])) if d in series.index else np.nan


def build_weekend_decay(asset: str, gold: pd.Series) -> pd.DataFrame:
    core = load_core_weekend_events(asset)
    rows = []
    for monday in core["date"]:
        tue, wed = monday + pd.Timedelta(days=1), monday + pd.Timedelta(days=2)
        gm, gt, gw = lg(gold, monday), lg(gold, tue), lg(gold, wed)
        if np.isnan(gm):
            continue
        row0 = core.loc[core["date"] == monday].iloc[0]
        rows.append(
            {
                "date": monday,
                "signal": float(row0["weekend_token_return"]),   # token move while gold CLOSED
                "basis": float(row0["friday_basis"]),
                "gold_monday": float(row0["monday_gold_return"]),  # exact core H3 Monday move
                "gold_tuesday": (gt - gm) if not np.isnan(gt) else np.nan,
                "gold_wednesday": (gw - gt) if not (np.isnan(gw) or np.isnan(gt)) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_placebo(tok: pd.Series, gold: pd.Series, asset: str) -> pd.DataFrame:
    """Mid-week analogue: 2-day token move (Mon->Wed) while gold is OPEN,
    predicting gold's next move (Wed->Thu)."""
    core = load_core_weekend_events(asset)
    placebo_wednesdays = set((core["date"] + pd.Timedelta(days=2)).tolist())
    rows = []
    for wed in gold.index:
        if wed.weekday() != 2 or wed not in placebo_wednesdays:
            continue
        mon, thu = wed - pd.Timedelta(days=2), wed + pd.Timedelta(days=1)
        tm, tw = lg(tok, mon), lg(tok, wed)
        gw, gt, gm = lg(gold, wed), lg(gold, thu), lg(gold, mon)
        if any(np.isnan(v) for v in [tm, tw, gw, gt, gm]):
            continue
        rows.append(
            {
                "date": wed,
                "signal": tw - tm,        # token move while gold OPEN
                "basis": tw - gw,
                "gold_next": gt - gw,     # gold's next move
            }
        )
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame, target: str, label: str, asset: str) -> dict:
    d = df[["signal", "basis", target]].dropna()
    if len(d) < 40:
        return {"asset": asset_name(asset), "test": label, "n": len(d), "beta": np.nan, "t": np.nan, "r2": np.nan}
    X = pd.DataFrame({"const": 1.0, "signal": d["signal"], "basis": d["basis"]})
    r = fit_ols(d[target], X, hac_lags=default_hac_lags(len(d)))
    return {
        "asset": asset_name(asset), "test": label, "n": int(len(d)),
        "beta": float(r.coefficients["signal"]), "t": float(r.t_stats["signal"]), "r2": float(r.r_squared),
    }


def run_asset(asset: str) -> pd.DataFrame:
    tok, gold = load_token(asset), load_gold()
    wk = build_weekend_decay(asset, gold)
    pb = build_placebo(tok, gold, asset)
    return pd.DataFrame(
        [
            fit(wk, "gold_monday", "A1. Weekend -> MONDAY (main)", asset),
            fit(wk, "gold_tuesday", "A2. Weekend -> Tuesday (decay)", asset),
            fit(wk, "gold_wednesday", "A3. Weekend -> Wednesday (decay)", asset),
            fit(pb, "gold_next", "B. PLACEBO mid-week (gold open)", asset),
        ]
    )


def render_md(res: pd.DataFrame) -> str:
    lines = [
        "# Placebo and decay tests - is the weekend effect caused by the closure?",
        "",
        "| Asset | Test | n | Beta | HAC t | R2 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in res.iterrows():
        lines.append(f"| {r['asset']} | {r['test']} | {r['n']} | {r['beta']:.3f} | {r['t']:.2f} | {r['r2']:.3f} |")
    lines += [
        "",
        "## How to read",
        "",
        "- **A1 (Monday)** is the main weekend result: the token moved while gold was shut, and gold catches up at the reopen.",
        "- **A2/A3 (Tue, Wed)**: if the information is absorbed at the reopen, these should be ~0. Persistence would suggest a trend effect rather than a closure effect.",
        "- **B (PLACEBO)**: the same structure applied mid-week, when gold was OPEN throughout. If this is ~0 while A1 is strong, the weekend effect is genuinely caused by the market closure - not generic token-gold lead-lag.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    warnings.filterwarnings("ignore")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.concat([run_asset(normalize_asset(a)) for a in ASSETS], ignore_index=True)
    res.to_csv(RESULTS_DIR / "placebo_decay_summary.csv", index=False)
    (RESULTS_DIR / "placebo_decay_summary.md").write_text(render_md(res), encoding="utf-8")
    print("Saved: placebo_decay_summary.csv / .md")
    print()
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
