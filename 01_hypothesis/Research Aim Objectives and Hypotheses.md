# Research Aim, Objectives, Questions, and Hypotheses

## Working Title (to confirm with supervisor)
**Tokenized but Informative? Off-Hours Price Discovery in Tokenized Gold (PAXG and XAUT)**

*(Alternative, closer to the earlier framing: "Price Discovery and Prediction Between Tokenized Gold and Gold: Weekday Dynamics and Weekend Off-Hours Effects")*

---

## Motivation

Tokenization is promoted on the promise of **liquidity and continuous (24/7) access** to assets whose traditional markets close. But **an open market is not necessarily an informative one**: weekend token trading is thin and retail-dominated, and could be pure noise — tradeable, yet informationally empty.

Mafrur (2026) makes this distinction explicit, showing that *on-chain representation and secondary-market liquidity are distinct outcomes*, and measuring liquidity through turnover, active addresses, and activity persistence. But activity alone does not establish that the resulting **prices carry information**.

This dissertation asks the next question: **is the liquidity that tokenization delivers informationally meaningful?**

## Where price discovery in gold happens — and the gap tokenization opens

A well-established literature finds that price discovery in gold concentrates in the **futures** market rather than spot, a pattern that also holds in cryptocurrency markets (Khan, 2022; Sundararajan et al., 2023; Kapar & Olmo, 2019). Futures — deep, liquid, and dominated by informed institutional participants — tend to move first, with spot following. This provides the natural benchmark for the present study: rather than asking whether tokenized gold leads *spot* (the weaker venue), the sharper question is how it sits against **futures**, the venue that actually leads gold price formation.

Crucially, tokenized gold differs from both spot and futures in one respect that reframes the entire question. Spot and futures trade on the **same schedule** and close together; tokenized gold trades **continuously**. The comparison against futures is therefore not only about *who leads when both are open*, but about *what happens when the futures market is closed* — a question that cannot even arise between two venues that shut at the same time. This is the gap the existing price-discovery literature could not address, and the one this study exploits: **futures lead gold price discovery while they are open, but they are closed every weekend, and it is precisely then that tokenized gold may take over the role of price formation.**

## Positioning — what this study does differently

| Study | Question | Measure |
|---|---|---|
| Ashfaq et al. (2026) | Does tokenized gold **track** gold? | tracking accuracy, deviation, correlation |
| Mafrur (2026) | Is there **liquidity**? | turnover, active addresses, active months |
| **This study** | Is that liquidity **informative**? | **price discovery and prediction** |

## Research Aim

To examine whether tokenized gold performs **genuine price discovery**, distinguishing periods when the reference gold market is **open** (weekdays) from periods when it is **closed** (weekends), and thereby to test whether tokenization delivers informationally meaningful off-hours liquidity.

## Research Objectives

1. **(Linkage)** Establish that PAXG and XAUT are faithful representations of gold, sharing a stable long-run equilibrium.
2. **(Weekday)** Determine which market leads price discovery while the gold market is open.
3. **(Weekend)** Test whether weekend token movements predict gold's adjustment at the reopen, and establish that this is caused by the closure.
4. **(Magnitude)** Test whether the token transmits information about the *size* of the adjustment, not only its direction.
5. **(Efficiency)** Assess how quickly weekend information is impounded once the market reopens.

## Research Questions

1. Are PAXG and XAUT cointegrated with gold, with a unit long-run relationship?
2. Which market leads weekday price discovery, and is that result robust to the benchmark?
3. Do weekend token movements predict Monday's gold adjustment, and is the effect caused by the market closure?
4. Do weekend token movements also predict the magnitude of that adjustment?
5. How rapidly is the weekend information impounded at the reopen?

---

## Hypotheses

### H1 — Long-run linkage (foundation)
> PAXG and XAUT are each cointegrated with gold, with a long-run beta close to one.

*Test:* Engle–Granger (OLS long-run regression + ADF on the residual).
*Result:* beta ≈ 0.998; residual strongly stationary (ADF ≈ −16.4 vs −3.34 at 5%). **Supported.**
*Role:* Necessary baseline — without a genuine link, later results are uninterpretable. Also re-establishes Ashfaq's tracking result on independent data.

### H2 — Weekday price discovery: does the token lead the venue that already leads gold?
> While the futures market is open, does the token contribute the majority of price discovery relative to gold futures?

*Prior (from the literature):* because futures lead gold price discovery (Khan, 2022; Sundararajan et al., 2023; Kapar & Olmo, 2019), a small tokenized spot-claim is not expected to lead the deep futures market. The reasoned hypothesis is therefore that **the token follows futures on weekdays.**

*Test:* Bivariate VECM with Gonzalo–Granger component shares, estimated against **both** benchmarks.
*Result:* Against **futures** (the informative benchmark), token GG share **0.28 (PAXG) / 0.41 (XAUT)** — the token **follows**, as predicted. Against spot the share is higher (0.62 / 0.80), i.e. the token *appears* to lead the weaker venue, but this reverses against futures.
*Interpretation:* The reversal is consistent with the non-synchronous-close bias (token closes midnight UTC; spot carries no timestamp; futures close on a New York basis). The futures result is the credible one — it is both economically sensible (COMEX dominance) and in line with the literature; the spot-leadership is best read as a timing artifact. A trivariate specification produced an implausible token-leads ordering, reinforcing that daily price-discovery shares are distorted by non-synchronous closes and should be read with caution.
*Role:* This is the **contrast that makes H3 meaningful.** While futures are open, the token does not lead price formation — it follows. But futures close every weekend, and H3 tests what happens then.

### H3 — Weekend off-hours price discovery (CORE)
> Weekend tokenized-gold movements predict gold's adjustment when the market reopens.

*Test:* Predictive regression of Monday's gold return on the weekend token return, with HAC standard errors, controlling for the Friday basis and weekend volume; plus out-of-sample validation.
*Result:* beta = **0.60 (t = 4.33) PAXG**, **0.85 (t = 7.28) XAUT** against spot; also significant against futures (t ≈ 3.6–9.2). Holds **out-of-sample**. Partial R² of the weekend signal beyond the basis ≈ 0.22. **Supported.**

**Identification — the effect is caused by the closure:**
- **Placebo:** the identical specification applied to a mid-week window (gold open) yields only 0.09 (t = 2.18) / 0.11 (t = 2.88) — the weekend effect is **6–7× larger**. The small mid-week residual is consistent with the few-hours closing-time offset.
- **Decay:** the weekend signal predicts Monday but is **exactly zero by Tuesday** (t = −0.02, both tokens), ruling out a momentum or trend interpretation.

### H4 — Weekend uncertainty transmission (magnitude)
> Weekend token volatility predicts the *magnitude* of gold's adjustment at the reopen.

*Test:* Weekend realised volatility and range volatility regressed on the absolute Monday gold return (HAC).
*Result:* **XAUT significant** (t = 3.09 realised, 3.23 range); **PAXG marginal** (t = 1.92, just below the 5% threshold). **Partially supported.**
*Role:* A second dimension of informativeness — the token conveys information about *uncertainty*, not only direction. This is the least mechanical of the results and therefore strong evidence against the "this is definitional" critique.

---

## Efficiency finding (reported in discussion, not a hypothesis)

How fast is the weekend information impounded? **Immediately at the reopen.** The predictable movement is concentrated in the reopening gap; almost nothing remains during the subsequent trading session (directional accuracy in the post-open session ≈ 51%, not significant). This is a statement about **informational efficiency**, not a trading strategy: the information is real, and the market prices it in at once.

## Robustness checks with no effect (reported briefly)

- **Momentum:** pre-weekend token trend adds no robust predictive power (only PAXG 5-day is significant, contradicted by the 20-day horizon and by both XAUT horizons).
- **Asymmetry:** negative weekend moves do not predict significantly more strongly than positive ones (difference t = 0.72 / 1.31), following Mourey et al. (2025). A well-tested null.
- **Liquidity conditioning:** the strength of the weekend signal does not robustly depend on weekend liquidity (measures disagree; one significant negative, one null).

## Confidence (state honestly in the write-up)

- **Strong:** H1 (linkage); H3 (weekend prediction — robust across benchmarks, out-of-sample, with placebo and decay identification).
- **Robust but negative:** H2 (no weekday leadership — benchmark-sensitive; reported as the contrast, not a failure).
- **Partial:** H4 (volatility transmission — clear for XAUT, marginal for PAXG).
- **Well-tested nulls:** momentum, asymmetry, liquidity conditioning.

## Benchmark and data note

- **Spot gold (XAU)** is the conceptual benchmark (the token represents physical gold) and is used for linkage and the core weekend regression.
- **Gold futures (GC=F)** are genuinely closed at weekends and carry a real exchange close, so they are used for the reopening-gap and timing-robustness analyses.
- The spot series carries no intraday timestamp and thin weekend quotes; gold is therefore treated as effectively closed at weekends. Exact intraday alignment is not possible with daily data — stated as a limitation, and the reason the weekend design (a genuine two-day closure) is preferred to weekday comparisons.

---

## The argument in one line

**The token is a faithful gold instrument (H1). While gold is open it merely follows (H2). While gold is closed it independently discovers price — in both direction (H3) and magnitude (H4) — and that information is impounded the instant the market reopens.**
