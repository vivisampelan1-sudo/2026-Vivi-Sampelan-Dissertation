"""Figure: weekday price-discovery share reverses with the benchmark (H2).

Reads the canonical VECM output and plots the token's Gonzalo-Granger share
against spot vs futures, with a 0.5 reference line (above = token leads,
below = token follows).
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

import pandas as pd

TMP_CACHE = Path(tempfile.gettempdir()) / "diss3_mplcache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISS3 = Path(__file__).resolve().parent.parent
df = pd.read_csv(DISS3 / "05_results" / "vecm_price_discovery_summary.csv")
share = df[df["metric"] == "token_gonzalo_granger_share"].copy()
share["value"] = share["value"].astype(float)

assets = ["PAXG", "XAUT"]
spot = [float(share[(share.asset == a) & (share.benchmark == "spot")].value.iloc[0]) for a in assets]
fut = [float(share[(share.asset == a) & (share.benchmark == "futures")].value.iloc[0]) for a in assets]

x = range(len(assets))
w = 0.36
fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar([i - w/2 for i in x], spot, w, label="vs Spot gold", color="#4C5FA6")
b2 = ax.bar([i + w/2 for i in x], fut, w, label="vs Gold futures", color="#C9822E")

ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1)
ax.text(1.45, 0.515, "0.5: token leads above / follows below", fontsize=8, color="#444444", ha="right")

for bars in (b1, b2):
    for bar in bars:
        ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)

ax.set_xticks(list(x)); ax.set_xticklabels(assets)
ax.set_ylabel("Token Gonzalo–Granger share")
ax.set_ylim(0, 1)
ax.set_title("Weekday price-discovery share reverses with the benchmark", fontsize=11)
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out = DISS3 / "05_results" / "fig_benchmark_reversal.png"
fig.savefig(out, dpi=200)
print(f"Saved: {out}")
