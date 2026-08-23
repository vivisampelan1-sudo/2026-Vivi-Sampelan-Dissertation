from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


TMP_CACHE = Path(tempfile.gettempdir()) / "diss3_mplcache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
DISS3_DIR = THIS_DIR.parent
SHARED_SCRIPTS_DIR = DISS3_DIR / "scripts"
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from econometrics_utils import fit_ols  # noqa: E402


RESULTS_DIR = DISS3_DIR / "05_results"


ASSET_CONFIG = {
    "PAXG": {
        "events_path": RESULTS_DIR / "paxg_option2_events.csv",
        "title": "PAXG: weekend token move vs Monday gold move",
    },
    "XAUT": {
        "events_path": RESULTS_DIR / "xaut_option2_events.csv",
        "title": "XAUT: weekend token move vs Monday gold move",
    },
}


def fit_weekend_only(events: pd.DataFrame) -> tuple[float, float, float]:
    X = pd.DataFrame(
        {
            "const": 1.0,
            "weekend_token_return": events["weekend_token_return"],
        }
    )
    res = fit_ols(events["monday_gold_return"], X)
    intercept = float(res.coefficients["const"])
    slope = float(res.coefficients["weekend_token_return"])
    r_squared = float(res.r_squared)
    return intercept, slope, r_squared


def axis_limits(series: pd.Series, pad_ratio: float = 0.05) -> tuple[float, float]:
    lo = float(series.min())
    hi = float(series.max())
    span = hi - lo
    pad = max(span * pad_ratio, 0.25)
    return lo - pad, hi + pad


def add_panel(ax: plt.Axes, asset: str, events: pd.DataFrame) -> None:
    intercept, slope, r_squared = fit_weekend_only(events)

    x = events["weekend_token_return"] * 100
    y = events["monday_gold_return"] * 100

    ax.scatter(x, y, s=28, color="#5C72B8", alpha=0.45, edgecolors="none")
    ax.axhline(0.0, color="#CFCFCF", linewidth=1.0)
    ax.axvline(0.0, color="#CFCFCF", linewidth=1.0)

    x_min, x_max = axis_limits(x)
    line_x = np.linspace(x_min, x_max, 200)
    line_y = (intercept + slope * (line_x / 100.0)) * 100
    ax.plot(line_x, line_y, color="#C97A1B", linewidth=2.5)

    ax.set_xlim(x_min, x_max)
    ax.set_title(ASSET_CONFIG[asset]["title"], fontsize=16, pad=14)
    ax.set_xlabel("Weekend token return (%)", fontsize=14)
    ax.set_ylabel("Monday gold return (%)", fontsize=14)

    label = f"slope (beta) = {slope:.2f}\nR² = {r_squared:.2f}"
    ax.text(
        0.05,
        0.90,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BDBDBD", "alpha": 0.95},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=13)


def main() -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(20.48, 10.24), dpi=100, constrained_layout=True)

    for ax, asset in zip(axes, ["PAXG", "XAUT"]):
        events = pd.read_csv(ASSET_CONFIG[asset]["events_path"], parse_dates=["monday_date"])
        add_panel(ax, asset, events)

    out_path = RESULTS_DIR / "fig_5_h3_prediction.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
