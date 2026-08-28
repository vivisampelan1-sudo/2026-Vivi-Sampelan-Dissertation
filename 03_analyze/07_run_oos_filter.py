"""Out-of-sample test of the filtered gold-futures strategy.

Threshold-snooping guard: choose the |weekend-move| threshold on the FIRST half
(training), then apply it, unchanged, to the SECOND half (test) it never saw.
If the edge survives out-of-sample, it is real; if it collapses, it was data-snooped.
Tradeable return = gold-futures Monday open->close, net of 5bp cost.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DISS3 = Path(__file__).resolve().parent.parent
RAW = DISS3 / "04_data" / "raw"
RESULTS = DISS3 / "05_results"
COST = 5.0 / 1e4
GRID = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015]
MIN_TRAIN_TRADES = 10


def ohlc(path, pfx):
    r = json.load(open(path))["chart"]["result"][0]; q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"ts": r["timestamp"], f"{pfx}_open": q["open"], f"{pfx}_close": q["close"]})
    d["date"] = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    return d.drop(columns="ts").dropna(subset=[f"{pfx}_close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def build(asset, gold):
    tok = ohlc(RAW / f"{asset}_yahoo_daily_full.json", "token")
    g = gold.copy(); g["wd"] = g["date"].dt.weekday; rows = []
    for _, m in g[g["wd"] == 0].iterrows():
        mon = m["date"]; fri = mon - pd.Timedelta(days=3); sun = mon - pd.Timedelta(days=1)
        ft, st, fg, mg = tok[tok.date == fri], tok[tok.date == sun], g[g.date == fri], g[g.date == mon]
        if ft.empty or st.empty or fg.empty or mg.empty: continue
        ft, st, fg, mg = ft.iloc[0], st.iloc[0], fg.iloc[0], mg.iloc[0]
        if pd.isna(mg.gold_open) or pd.isna(mg.gold_close): continue
        rows.append(dict(date=mon, sig=np.log(st.token_close) - np.log(ft.token_close),
                         intraday=np.log(mg.gold_close) - np.log(mg.gold_open)))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def strat_returns(d, thr):
    s = d[d.sig.abs() >= thr]
    pos = np.sign(s.sig)
    return (pos * s.intraday - COST * (pos != 0)).dropna()


def sharpe(ret):
    return float(ret.mean() / ret.std() * np.sqrt(52)) if len(ret) > 1 and ret.std() > 0 else np.nan


def evaluate(ret):
    n = len(ret)
    if n < 2 or ret.std() == 0:
        return dict(n=n, mean_bps=np.nan, ret_t=np.nan, ret_p=np.nan, ann_sharpe=np.nan)
    t = ret.mean() / (ret.std(ddof=1) / np.sqrt(n))
    return dict(n=n, mean_bps=round(ret.mean() * 1e4, 1), ret_t=round(float(t), 2),
                ret_p=round(float(stats.t.sf(t, df=n - 1)), 3), ann_sharpe=round(sharpe(ret), 2))


def main():
    gold = ohlc(RAW / "gold_yahoo_gc_f_1y.json", "gold")
    out = []
    for asset in ["paxg", "xaut"]:
        ev = build(asset, gold)
        cut = len(ev) // 2
        train, test = ev.iloc[:cut], ev.iloc[cut:]
        # choose threshold on TRAIN by Sharpe, requiring enough trades
        best_thr, best_sh = 0.0, -np.inf
        for thr in GRID:
            r = strat_returns(train, thr)
            if len(r) >= MIN_TRAIN_TRADES and sharpe(r) > best_sh:
                best_sh, best_thr = sharpe(r), thr
        # apply chosen threshold to TEST (never seen)
        test_chosen = evaluate(strat_returns(test, best_thr))
        test_nofilter = evaluate(strat_returns(test, 0.0))
        out.append({"asset": asset.upper(), "chosen_thr_%": round(best_thr * 100, 2),
                    "train_sharpe_at_thr": round(best_sh, 2),
                    **{f"TEST_{k}": v for k, v in test_chosen.items()},
                    "TEST_nofilter_sharpe": test_nofilter["ann_sharpe"], "TEST_nofilter_ret_p": test_nofilter["ret_p"]})
    df = pd.DataFrame(out)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "oos_filter_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
