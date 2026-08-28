#!/usr/bin/env bash
# ------------------------------------------------------------------
# Dissertation 3 — run the analysis pipeline end to end.
# Usage: bash run_all.sh
# Results are written to 05_results/.
# ------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_script() {
  echo "   -> $*"
  "$PYTHON_BIN" "$@"
}

have_intraday_inputs() {
  [[ -f 04_data/raw/paxg_yahoo_60m.csv ]] &&
  [[ -f 04_data/raw/xaut_yahoo_60m.csv ]] &&
  [[ -f 04_data/raw/gc_f_yahoo_60m.csv ]]
}

echo "[1/5] Pull market data (needs internet + ALPHAVANTAGE_API_KEY in .env; see .env.example)..."
"$PYTHON_BIN" 01_data_pull/pull_market_data.py

echo "[2/5] Build weekend event tables used by the core design..."
"$PYTHON_BIN" scripts/option2_weekend_design.py --asset paxg
"$PYTHON_BIN" scripts/option2_weekend_design.py --asset xaut

echo "[3/5] Run core daily analysis scripts..."
run_script 03_analyze/01_run_vecm_price_discovery.py
run_script 03_analyze/02_run_design_hypotheses.py
run_script 03_analyze/03_run_weekend_marginal_contribution.py
run_script 03_analyze/04_run_oos_precision.py
run_script 03_analyze/05_run_monday_gap_robustness.py
run_script 03_analyze/06_run_trading_strategy.py
run_script 03_analyze/07_run_oos_filter.py
run_script 03_analyze/08_run_volatility_momentum.py
run_script 03_analyze/09_run_liquidity_conditioning.py
run_script 03_analyze/10_run_placebo_decay.py
run_script 03_analyze/13_run_placebo_weekend_interaction.py

echo "[4/5] Run intraday and hourly robustness scripts when inputs are available..."
if have_intraday_inputs; then
  "$PYTHON_BIN" 02_process/02_build_intraday_reopen_events.py --asset paxg --token-source yahoo --benchmark gc_f_yahoo
  "$PYTHON_BIN" 02_process/02_build_intraday_reopen_events.py --asset xaut --token-source yahoo --benchmark gc_f_yahoo
  run_script 03_analyze/11_run_intraday_reopen_regression.py --token-source yahoo --benchmark gc_f_yahoo
  run_script 03_analyze/12_run_intraday_sensitivity.py --token-source yahoo --benchmark gc_f_yahoo
  run_script 03_analyze/13_run_hourly_volatility_robustness.py
else
  echo "   -> Skipping intraday robustness block (missing 60-minute raw inputs in 04_data/raw/)."
fi

echo "[5/5] Regenerate figure files..."
run_script 03_analyze/14_plot_h3_prediction.py
run_script 03_analyze/plot_benchmark_reversal.py

echo
echo "Done. All outputs are in 05_results/"
