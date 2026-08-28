# Reproduction Notes

This repository is organized around the dissertation workflow rather than as a general-purpose package.

## What runs from scratch

The following steps can be reproduced directly from the repository:

1. Pull daily market data (requires an Alpha Vantage key — see below)
2. Build the aligned weekday and weekend-event datasets
3. Run the core H1–H4 analyses
4. Regenerate the H2 and H3 figures (`plot_benchmark_reversal.py`, `14_plot_h3_prediction.py`)

Command:

```bash
bash run_all.sh
```

Note: the dissertation's descriptive Figures 1–3 (price trends, basis over time, daily token volume) have no generating script in this repo and are not covered by step 4.

## Alpha Vantage key (spot gold benchmark)

The spot-gold benchmark used throughout the dissertation comes from Alpha Vantage. Copy `.env.example` to `.env` and set `ALPHAVANTAGE_API_KEY` before running step 1. Without a valid key (or a cached `04_data/raw/gold_alphavantage_daily.json`), `01_data_pull/pull_market_data.py` now fails with a clear error rather than silently substituting Yahoo GC=F futures as the benchmark — pass `--allow-futures-fallback` if you explicitly want that substitution (results won't match the dissertation's spot-gold tables).

## Intraday / hourly robustness block

The intraday and hourly reopening checks depend on three 60-minute raw files that are not committed to the repo (`.gitignore`d, like all of `04_data/raw/`):

- `04_data/raw/paxg_yahoo_60m.csv`
- `04_data/raw/xaut_yahoo_60m.csv`
- `04_data/raw/gc_f_yahoo_60m.csv`

Fetch them with the intraday pull script before running `run_all.sh`:

```bash
python3 01_data_pull/pull_intraday_reopen_data.py --asset paxg --token-provider yahoo --gold-provider yahoo_futures
python3 01_data_pull/pull_intraday_reopen_data.py --asset xaut --token-provider yahoo --gold-provider yahoo_futures
```

Once those files are present, `run_all.sh` will automatically run:

- intraday reopening-gap regressions
- hourly sensitivity checks
- hourly weekend-volatility robustness

If those files are missing, the runner skips that block and still completes the daily pipeline — but Tables 11, 12, and 16 (which report the hourly pre-reopen results) cannot be reproduced until the intraday pull step above is run.

## Main entry points

- `01_data_pull/pull_market_data.py`: daily raw data pull
- `scripts/option2_weekend_design.py`: cleaned daily datasets + core weekend-event construction
- `03_analyze/01_run_vecm_price_discovery.py`: weekday VECM and GG shares
- `03_analyze/02_run_design_hypotheses.py`: H1 and core H3 outputs
- `03_analyze/08_run_volatility_momentum.py`: H4 and momentum checks
- `03_analyze/10_run_placebo_decay.py`: placebo / decay checks
- `03_analyze/13_run_placebo_weekend_interaction.py`: pooled interaction test

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

The code uses standard scientific Python packages only:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `requests`

## Output locations

- raw inputs: `04_data/raw/`
- processed datasets: `04_data/processed/`
- generated tables and figures: `05_results/`

These generated folders are ignored by `.gitignore` so the repository stays focused on code.
