# Methodology

## 1. Research design & positioning

This is a **quantitative time-series study** testing whether tokenized gold performs **genuine price discovery** — that is, whether the liquidity tokenization delivers is *informationally meaningful* rather than merely present.

The design is econometric and observational, with a **quasi-experimental element**: the weekend closure of the gold market is a naturally occurring condition that isolates the token's independent informational role. All models are OLS-based, extended in two standard ways for financial data — cointegration methods for non-stationary prices, and HAC (Newey–West) standard errors for valid inference.

**Positioning.** Ashfaq et al. (2026) show tokenized gold *tracks* gold (descriptive, contemporaneous). Mafrur (2026) shows tokenized assets may be *represented on-chain without secondary-market liquidity*, measuring liquidity by turnover, active addresses and activity persistence. Neither asks whether the prices produced are **informative**. This study does.

The analysis is organised around one contrast:

> **When the gold market is OPEN, does the token lead? When it is CLOSED, does the token discover price?**

## 2. Hypothesis-to-method map

| Hypothesis | Method | Role |
|---|---|---|
| **H1** — long-run linkage | ADF + Engle–Granger cointegration | foundation |
| **H2** — weekday: which market leads? | bivariate VECM + Gonzalo–Granger shares, **both benchmarks** | contrast |
| **H3** — weekend off-hours price discovery | weekend→Monday predictive regression (HAC) + out-of-sample | **core** |
| *H3 identification* | **placebo** (mid-week window) + **decay** (Tue/Wed) | supports closure interpretation |
| **H4** — weekend uncertainty transmission | weekend volatility → magnitude of Monday move | extension |
| *Efficiency (discussion)* | speed of impounding at the reopen | interpretation |
| *Robustness (nulls)* | momentum, asymmetry, liquidity conditioning | thoroughness |

## 3. Data

**Assets & benchmarks.**
- `PAXG`, `XAUT` — tokenized gold (Yahoo Finance, daily, trade 7 days/week).
- **Spot gold (XAU)** — the *conceptual* benchmark (the token represents physical gold); used for linkage and the core weekend regression.
- **Gold futures (GC=F)** — genuinely closed at weekends, with a real exchange close and open price; used for the reopening-gap and timing-robustness analyses.

**Sample.** PAXG from Sep 2019; XAUT from Feb 2020; both to Jul 2026. Gold futures from Jan 2018 (before both tokens), so futures-based tests run over each token's full sample. Approximately 2,470 / 2,340 daily observations and around 353 / 333 weekend events. *(Confirm the exact daily and weekend counts against the single final run used for the Results chapter, so Methodology and Results agree.)*

**Frequency.** Daily. Sufficient for linkage, adjustment and weekend-to-Monday prediction; not an intraday microstructure study (stated as a limitation).

**Two calendar structures.**
- *Weekday-synchronised* (both markets open) — for cointegration and the VECM.
- *Calendar / 24-7* (weekends retained) — for the weekend event tests only. The VECM is never estimated on forward-filled weekend gold, which would manufacture apparent token leadership.

## 4. Data quality

**Liquidity.** PAXG is liquid throughout (median ≈ \$11m/day; ≈ \$169m over the final year). XAUT is thin early (median ≈ \$2.8m) but highly liquid recently (≈ \$195m). Neither exhibits zero-volume or stale (zero-return) days. XAUT's early thinness is acknowledged, and its weekend results are re-estimated over the more liquid recent sub-period.

**Timing alignment.** The token closes at ~00:00 UTC; gold futures on a New York basis (~2–3 hours earlier); the spot series carries **no intraday timestamp at all**. Exact alignment is therefore impossible with daily data. This bias is most relevant to the weekday VECM (H2) and is addressed directly there; the weekend design avoids it, because the identifying variation is a genuine two-day closure rather than a timestamp offset.

## 5. Variable construction

- `log price = ln(price)`; returns are first differences.
- **Basis (deviation):** `basis_t = ln(token_t) − ln(gold_t)` — the disequilibrium term, central to both the price-discovery test and as a control that separates mechanical convergence from information.
- **Weekend event variables:** `weekend_token_return` (Fri close → Sun close), `monday_gold_return` (Fri → Mon), `monday_gap` (Fri close → Mon open, futures), `weekend_realised_volatility`, `weekend_range_volatility`, `log_weekend_volume`, `weekend_volume_share`, `friday_basis`.

**Definition of the "Monday adjustment."** The core weekend prediction target, `monday_gold_return`, is the log change in the gold benchmark from **Friday's close to Monday's close**. The reopening-gap variant uses the gold-futures **Monday open** (Friday close → Monday open), which captures the discrete jump at the reopen. Timing note: token prices close at approximately **00:00 UTC**, spot gold carries no intraday timestamp, and gold futures settle on a **New York** basis, so the "close" prices are not perfectly synchronous across series (addressed in Section 4, Timing alignment).

## 6. H1 — Long-run linkage

ADF unit-root tests confirm prices are I(1) and returns I(0). Engle–Granger then estimates `ln(token) = a + b·ln(gold) + u` and applies an ADF test to the residual `u`. A stationary residual with `b ≈ 1` establishes cointegration.

*Role:* necessary foundation — without a genuine long-run link, later results are uninterpretable. Also re-establishes Ashfaq's tracking result on independent data.

## 7. H2 — Weekday price discovery: which market leads?

A bivariate VECM is estimated on weekday-synchronised data:

`Δtoken_t = λ_token·ECT_{t−1} + Σ …`  and  `Δgold_t = λ_gold·ECT_{t−1} + Σ …`

The adjustment coefficients λ measure how strongly each market moves to restore equilibrium; **the market that adjusts less is the leader**. This is summarised by the Gonzalo–Granger component share, `GG_token = λ_gold / (λ_gold − λ_token)`, where a value above 0.5 indicates token leadership. Lag length is selected by system BIC. Gonzalo–Granger rather than Hasbrouck because the data are daily.

**Critically, the VECM is estimated against both benchmarks** (spot and futures) precisely because the timing of daily closes differs between them. Reporting only one would conceal the sensitivity.

**Sample robustness.** The estimated share is sensitive to the gold-futures sample period, so the full 2018–2026 futures series is used (the longest available, and therefore the most robust). The substantive conclusion — that the token does **not** lead against futures — is stable across sample choices; only the exact share moves, which itself indicates that weekday leadership is not a robust finding.

*Role:* establishes the **contrast** — whether the token has an independent informational role while the reference market is open.

## 8. H3 — Weekend off-hours price discovery (core)

Using the weekend-event panel, Monday's gold return is regressed on the weekend token return:

`monday_gold_return = a + b·weekend_token_return + c·log(weekend_volume) + d·friday_basis + ε`

estimated with Newey–West HAC standard errors. The coefficient `b` captures the off-hours information channel. **Including the Friday basis is essential**: part of any Monday move is the mechanical closing of a pre-existing gap rather than new information, and controlling for it isolates the informational component. The signal's own explanatory power beyond the basis is quantified by a partial R².

**Out-of-sample validation.** An expanding-window procedure estimates the model on past data only and evaluates directional accuracy and forecast error on subsequent, unseen weekends.

### 8.1 Identification — does the market closure account for the effect?

Two tests support a closure-based interpretation rather than assuming it.

- **Placebo.** The identical specification is applied to a **mid-week window in which gold traded throughout** (a two-day token move, Monday→Wednesday, predicting gold's subsequent move). Since gold was open, it has already impounded that information, so a closure-driven effect should be largely absent here. Comparing the weekend and mid-week coefficients quantifies how much the closure amplifies the token's predictive role.
- **Decay.** The weekend signal is tested against Tuesday's and Wednesday's gold returns. If weekend information is impounded at the reopen, predictive power should be concentrated on Monday and vanish immediately after — ruling out a momentum or trend interpretation.

## 9. H4 — Weekend uncertainty transmission

Beyond direction, the study tests whether the token conveys information about the **magnitude** of the coming adjustment. Weekend realised volatility (from the Saturday and Sunday returns) and a Parkinson-style range measure are regressed on the **absolute** Monday gold return, with the same controls and HAC errors.

*Role:* a second dimension of informativeness. It is also the least mechanical result — nothing about 1:1 gold backing implies that weekend turbulence should forecast the size of Monday's move.

## 10. Efficiency (reported in discussion)

Rather than testing a trading strategy, the study asks **how quickly weekend information is impounded**. Decomposing Monday's move into the reopening gap (Friday close → Monday open) and the subsequent session (Monday open → close) shows where the predictable component sits. This is a statement about informational efficiency, not investment advice.

## 11. Robustness checks

- **Momentum:** pre-weekend token trend (5- and 20-day) added to the weekend regression.
- **Asymmetry:** an interaction specification testing whether negative weekend moves predict more strongly than positive ones — a single coefficient with its own t-statistic, rather than an informal comparison of two separately estimated slopes.
- **Liquidity conditioning:** whether the weekend signal strengthens with weekend liquidity, using two **detrended** measures (weekend volume share; volume relative to a trailing mean) because raw token volume grows by orders of magnitude across the sample.
- **XAUT sub-period:** re-estimation over the more liquid recent sample.
- **Benchmark robustness:** the core weekend regression re-estimated against gold futures.

## 12. Inference and software

All reported t-statistics use HAC (Newey–West) standard errors, so inference is robust to the heteroskedasticity and autocorrelation characteristic of financial returns. Statistical significance is assessed at conventional levels. All data collection, processing and estimation are implemented in Python, and the full pipeline is scripted and reproducible from raw data.
