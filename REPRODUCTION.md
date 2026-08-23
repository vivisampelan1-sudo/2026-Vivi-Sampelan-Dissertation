# Reproduction Notes

This repository is organized around the dissertation workflow rather than as a general-purpose package.

## What runs from scratch

The following steps can be reproduced directly from the repository:

1. Pull daily market data
2. Build the aligned weekday and weekend-event datasets
3. Run the core H1–H4 analyses
4. Regenerate the main figures used in the dissertation

Command:

```bash
bash run_all.sh
```

## Intraday / hourly robustness block

The intraday and hourly reopening checks depend on 60-minute raw files:

- `data/raw/paxg_yahoo_60m.csv`
- `data/raw/xaut_yahoo_60m.csv`
- `data/raw/gc_f_yahoo_60m.csv`

If those files are present, `run_all.sh` will automatically run:

- intraday reopening-gap regressions
- hourly sensitivity checks
- hourly weekend-volatility robustness

If those files are missing, the runner skips that block and still completes the daily pipeline.

## Main entry points

- `02_data_pull/pull_market_data.py`: daily raw data pull
- `scripts/option2_weekend_design.py`: cleaned daily datasets + core weekend-event construction
- `04_analyze/01_run_vecm_price_discovery.py`: weekday VECM and GG shares
- `04_analyze/02_run_design_hypotheses.py`: H1 and core H3 outputs
- `04_analyze/08_run_volatility_momentum.py`: H4 and momentum checks
- `04_analyze/10_run_placebo_decay.py`: placebo / decay checks
- `04_analyze/13_run_placebo_weekend_interaction.py`: pooled interaction test

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

- raw inputs: `data/raw/`
- processed datasets: `data/processed/`
- generated tables and figures: `05_results/`

These generated folders are ignored by `.gitignore` so the repository stays focused on code.
