"""Economic-significance test: can a trader actually use the weekend signal?

The realistic tradeable instrument is GOLD FUTURES (GC=F): liquid, low-cost, and
they reopen Sunday evening. We evaluate three layers per weekend:
  - Full     : Friday close -> Monday close (upper bound; NOT tradeable, includes the gap)
  - Gap      : Friday close -> Monday open  (the overnight reopening jump; NOT capturable)
  - Intraday : Monday open  -> Monday close (the TRADEABLE gold-futures return)

Signal = weekend token log-return (Friday close -> Sunday close). Position = sign(signal).
Filtered variants trade only when |weekend move| exceeds a threshold.

Two formal tests are added:
  - mean-return t-test : is the average strategy return (net of costs) significantly > 0?
  - hit-rate binomial : is directional accuracy significantly > 50%?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DISS3 = Path(__file__).resolve().parent.parent
RAW = DISS3 / "data" / "raw"
RESULTS = DISS3 / "05_results"
COST_BPS = 5.0  # round-trip transaction cost on liquid gold futures


def to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def ohlc(path: Path, pfx: str) -> pd.DataFrame:
    r = json.load(open(path))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"ts": r["timestamp"], f"{pfx}_open": q["open"], f"{pfx}_close": q["close"]})
    d["date"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    return d.drop(columns="ts").dropna(subset=[f"{pfx}_close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def build_events(asset: str, gold: pd.DataFrame) -> pd.DataFrame:
    tok = ohlc(RAW / f"{asset}_yahoo_daily_full.json", "token")
    g = gold.copy(); g["wd"] = g["date"].dt.weekday
    rows = []
    for _, m in g[g["wd"] == 0].iterrows():
        mon = m["date"]; fri = mon - pd.Timedelta(days=3); sun = mon - pd.Timedelta(days=1)
        ft, st, fg, mg = tok[tok.date == fri], tok[tok.date == sun], g[g.date == fri], g[g.date == mon]
        if ft.empty or st.empty or fg.empty or mg.empty:
            continue
        ft, st, fg, mg = ft.iloc[0], st.iloc[0], fg.iloc[0], mg.iloc[0]
        if pd.isna(mg.gold_open) or pd.isna(fg.gold_close) or pd.isna(mg.gold_close):
            continue
        rows.append(dict(
            sig=np.log(st.token_close) - np.log(ft.token_close),
            gap=np.log(mg.gold_open) - np.log(fg.gold_close),
            intraday=np.log(mg.gold_close) - np.log(mg.gold_open),
            full=np.log(mg.gold_close) - np.log(fg.gold_close),
        ))
    return pd.DataFrame(rows)


def backtest(ev: pd.DataFrame, target: str, thr: float = 0.0, tradeable: bool = True) -> dict:
    d = ev[ev.sig.abs() >= thr].copy()
    pos = np.sign(d.sig)
    cost = (COST_BPS / 1e4) if tradeable else 0.0
    ret = (pos * d[target] - cost * (pos != 0)).dropna()
    n = len(ret)
    # directional hit rate + binomial test vs 50%
    hits = int((np.sign(d.sig) == np.sign(d[target])).sum())
    hit_pct = 100 * hits / n if n else np.nan
    hit_p = stats.binomtest(hits, n, 0.5, alternative="greater").pvalue if n else np.nan
    # mean-return t-test (H1: mean > 0)
    if n > 1 and ret.std() > 0:
        t = ret.mean() / (ret.std(ddof=1) / np.sqrt(n))
        ret_p = stats.t.sf(t, df=n - 1)  # one-sided p
    else:
        t, ret_p = np.nan, np.nan
    sharpe = float(ret.mean() / ret.std() * np.sqrt(52)) if ret.std() > 0 else np.nan
    return dict(n=n, hit_pct=round(hit_pct, 1), hit_p=round(float(hit_p), 3),
                mean_bps=round(ret.mean() * 1e4, 1), ret_t=round(float(t), 2), ret_p=round(float(ret_p), 3),
                ann_sharpe=round(sharpe, 2), total_ret_pct=round(ret.sum() * 100, 1))


def main() -> None:
    gold = ohlc(RAW / "gold_yahoo_gc_f_1y.json", "gold")
    specs = [("Full (Fri->Mon close) [NOT tradeable]", "full", 0.0, False),
             ("Gap / overnight reopen [NOT capturable]", "gap", 0.0, False),
             ("Gold futures, all signals [TRADEABLE]", "intraday", 0.0, True),
             ("Gold futures, |signal|>0.5%", "intraday", 0.005, True),
             ("Gold futures, |signal|>1.0%", "intraday", 0.010, True)]
    out = []
    for asset in ["paxg", "xaut"]:
        ev = build_events(asset, gold)
        for label, tgt, thr, trad in specs:
            out.append({"asset": asset.upper(), "strategy": label, **backtest(ev, tgt, thr, trad)})
    df = pd.DataFrame(out)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "trading_strategy_summary.csv", index=False)
    (RESULTS / "trading_strategy_summary.md").write_text(
        "# Economic Significance: Gold-Futures Trading Test\n\n"
        "The realistic instrument is gold futures (GC=F). `hit_p` tests whether directional "
        "accuracy beats 50%; `ret_t`/`ret_p` test whether the mean return (net of 5bp cost) is "
        "significantly positive. A strategy is exploitable only if `ret_p < 0.05` on a *tradeable* row.\n\n"
        + to_markdown_table(df) + "\n",
        encoding="utf-8",
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
