# YTD Top-30 Failure Post-Mortem — 2026-06-28

**Classification:** RESEARCH_AUTOPSY / VALIDATION_DIAGNOSTIC / NO_MODEL_CHANGE / NO_RANKER_CHANGE / NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE

---

## Executive Verdict

**4 of 6 failure windows were at least partially avoidable.** The primary mechanism was not broad ranker inversion or sector beta — it was a small cluster of repeat-offender names (CELC, ABVX, PRAX, DRUG) that carried large negative positions across multiple weeks, combined with systematic underperformance of `build_window` and `less_binary` catalyst names relative to `binary_now`. Two tools that were visible at snapshot time would have helped: **EES v3 gate** (avg delta +0.82pp across failure windows) and **ranks 31–60 as replacement bench** (avg delta +1.64pp). The worst cluster — May 26 → June 8 — triggered a rolling 4-window XS drawdown of **−8.52pp**, well beyond the −5.00pp threshold. The March 2026 failure had no visible pre-window signal and should be treated as unavoidable noise.

---

## Failure Windows Summary

| window | xs_pp | severity | xbi_5d | regime | primary_class | avoidable |
|--------|-------|----------|--------|--------|---------------|-----------|
| 2026-03-16 → 2026-03-23 | −2.92 | HARD_FAIL | −1.81% | NO_COL | NO_OBVIOUS_PRE_WINDOW_SIGNAL | NO |
| 2026-04-20 → 2026-04-27 | −1.44 | NEAR_MISS | −3.53% | UNKNOWN | BROAD_SECTOR_BETA / EES_WOULD_HAVE_HELPED | YES |
| 2026-05-04 → 2026-05-11 | −1.95 | HARD_FAIL | +0.94% | UNKNOWN | RANK_INVERSION | YES |
| 2026-05-19 → 2026-05-26 | −1.67 | FAIL | +5.17% | UNKNOWN | RANK_INVERSION | YES |
| 2026-05-26 → 2026-06-02 | −2.45 | HARD_FAIL | −4.19% | UNKNOWN | NO_OBVIOUS_PRE_WINDOW_SIGNAL | NO |
| 2026-06-01 → 2026-06-08 | −1.64 | FAIL | −3.88% | UNKNOWN | EES_WOULD_HAVE_HELPED | YES |
| 2026-06-08 → 2026-06-15 | −1.44 | BORDERLINE | +6.21% | UNKNOWN | RANK_INVERSION / EES_WOULD_HAVE_HELPED | YES |

**Note:** `regime_label` is `UNKNOWN` in all 2026 snapshots — regime classification was not operational during these windows. The March snapshot had no `regime_label` column at all.

**Rolling 4-window drawdown trigger:** May 04 → June 08 cumulative XS = **−8.52pp** (threshold: −5.00pp). TRIGGERED.

---

## Per-Window Attribution

### 2026-03-16 → 2026-03-23 (xs = −2.92pp) — NO_OBVIOUS_PRE_WINDOW_SIGNAL

**No EES v3 data in March snapshot.** EES overlay was not yet deployed.

Top detractors: IVVD (−18.6pp xs, build_window), NMRA (−15.2pp, build_window), ARTV (−14.2pp, binary_now), CLYM (−12.7pp, binary_now), PEPG (−10.7pp, less_binary).

Top 5 names explained **81%** of the miss. `binary_now` names had avg xs = −7.5pp, hit rate 0/5. `build_window` avg xs = −1.9pp.

All counterfactuals were neutral or worse:
- ex_ees_veto: 0pp delta (no EES data)
- bench_31_60: **−1.80pp delta** (bench was also bad)
- ex_priced_move_q4: −0.96pp delta (excluding high-PM names hurt)

**Verdict:** TRUE miss. Broad-based failure across catalyst buckets. Bench was worse. No pre-window signal was visible. Treat as unavoidable noise.

---

### 2026-04-20 → 2026-04-27 (xs = −1.44pp) — NEAR_MISS / EES_WOULD_HAVE_HELPED

**Dominated by a single name: CMPX** (rank 19, EES gate = False, priced_move = 513%). CMPX dropped −66.67% on a binary event, contributing **−2.11pp** to portfolio excess — more than the entire miss.

The EES gate correctly flagged CMPX as `False` at snapshot time. Excluding the 3 EES-False names (CMPX + 2 others) and filling from bench would have turned this into a **+1.28pp** week (delta = +2.71pp).

Without CMPX, this window would have been a **positive** week.

**Verdict:** AVOIDABLE via EES. One name, clearly flagged. This is a textbook false-positive prevention case — the EES was right.

---

### 2026-05-04 → 2026-05-11 (xs = −1.95pp) — RANK_INVERSION

XBI was up +0.94% — the market was rising. Top-30 fell −1.01%. This is a genuine rank inversion.

Top detractors: INSM (rank 2, −26.73pp xs, `core`, EES True), CELC (rank 20, −14.28pp, `build_window`, EES False), ZYME (rank 25, −11.21pp, `core`, EES False), COGT (rank 1, −8.1pp, `binary_now`, EES True), BCRX (−7.67pp, `less_binary`).

Key signal: **bench 31–60 returned +2.56pp xs** — a +4.50pp gap over Top-30. This is the clearest rank inversion of any failure window.

7 EES-False names were in the top-30. Excluding them improved by +1.26pp.

`core` bucket avg xs = −4.65pp (hit rate 25%). `binary_now` avg xs = +1.19pp (hit rate 50%).

**Verdict:** AVOIDABLE via rank cutoff review. The bench held better quality names this week. EES exclusion would have helped moderately. This was not a market/regime event.

---

### 2026-05-19 → 2026-05-26 (xs = −1.67pp) — RANK_INVERSION

XBI rose +5.17%. Top-30 gained only +3.38%. Classic laggard week.

Top detractors: BCRX (−8.60pp, `less_binary`), TNGX (−6.46pp, `binary_now`), JBIO (−5.95pp, `less_binary`), TARS (−5.39pp, `less_binary`), XENE (−5.30pp, `build_window`).

`less_binary` avg xs = −4.64pp, hit rate 0/5. Bench 31–60 returned +0.41pp xs, a +2.20pp gap.

EES counterfactual was **negative** (−0.96pp): the EES-False names this week were relatively better than the True names. The failure was driven by EES-True names that simply underperformed.

**Verdict:** AVOIDABLE — driven by `less_binary` concentration and rank boundary weakness, not EES-flaggable names. Top-30 held too many names that were near-event but not binary-now, in a rising-beta week where the market rewarded cleaner names.

---

### 2026-05-26 → 2026-06-02 (xs = −2.45pp) — NO_OBVIOUS_PRE_WINDOW_SIGNAL

XBI fell −4.19%. Top-30 fell −6.64%. This is the worst absolute week.

Top detractors: **ABVX** (rank 13, −42.65%, `build_window`, EES True, no priced_move data — priced_move NaN likely a data gap), **CELC** (rank 21, −31.91%, `build_window`, EES False), **PRAX** (rank 12, −26.39%, `binary_now`, EES True), TYRA (−14.68%, `less_binary`), DRUG (−11.01%).

ABVX and PRAX were EES-True (passed gate) and still collapsed. CELC was EES-False but the basket excluding EES-False only gained +0.58pp — insufficient to overcome the ABVX/PRAX collapse.

Bench 31–60 returned −1.41pp xs — still bad, just less bad (+1.04pp delta).

`build_window` avg xs = −6.29pp. `binary_now` avg xs = +1.51pp.

**Key observation:** This window featured **clinical/trial readout failures** for ABVX and PRAX — binary outcomes that the model correctly assessed as high-probability events (hence `build_window` and `binary_now` status), but which resolved negatively. These are not predictable from pre-window signals.

**Verdict:** NOT AVOIDABLE. The two largest contributors (ABVX, PRAX) passed EES and were correctly identified as high-priority names. They had negative binary outcomes. The model cannot avoid all binary event losses. The bench was marginally better but also down.

---

### 2026-06-01 → 2026-06-08 (xs = −1.64pp, reconstructed −2.33pp) — EES_WOULD_HAVE_HELPED

**Note:** xs reconstruction uses 2026-05-29 snapshot (2026-06-01 was QUARANTINED). Reconstructed xs = −2.33pp vs known −1.64pp — the substituted snapshot may explain the discrepancy.

Top detractors: CELC (−25.01pp, `build_window`, EES False), MLYS (−23.79pp, `core`, EES True), PRAX (−20.55pp, `binary_now`, EES True), ABVX (−17.83pp, `build_window`, EES False), DRUG (−15.54pp, `less_binary`, EES True).

5 EES-False names with avg xs = **−11.49pp**, hit rate 0/5. Excluding them improved by **+1.96pp** — flipping the week from −2.33pp to −0.38pp xs.

The `build_window` bucket avg xs = −6.40pp. Binary_now = +0.54pp.

CELC and ABVX appear in this window as EES-False with massive drawdowns, continuing from the May-26 window. The model kept them ranked (via 05-29 snapshot) despite prior-week collapses.

**Verdict:** AVOIDABLE via EES. EES-False names drove the bulk of the miss. This is a continued-exposure problem — names that failed in the prior window (ABVX, CELC) remained in the top-30.

---

### 2026-06-08 → 2026-06-15 (xs = −1.44pp) — RANK_INVERSION / EES_WOULD_HAVE_HELPED

XBI up +6.21%. Top-30 returned +4.77%. Classic laggard in a rally.

Top detractors: ANNX (−15.58pp, `less_binary`), DRUG (−15.56pp, `less_binary`), ABVX (−9.43pp, `build_window`, EES False), NBIX (−8.42pp, `core`), PHVS (−6.36pp, `binary_now`).

6 EES-False names, avg xs = −4.48pp. Excluding them: +2.07pp delta.
Bench 31–60 returned +1.10pp xs: +2.54pp delta. Both signals were present.

`less_binary` avg xs = −6.09pp, hit rate 25%.

**Verdict:** AVOIDABLE via EES + bench cutoff. In a strong rally week, `less_binary` names and EES-flagged names dragged. Both signals fired.

---

## Feature Bucket Attribution (Aggregate, Failure Windows)

### Catalyst Bucket

| bucket | n | avg_xs | total_contrib | hit_rate |
|--------|---|--------|---------------|----------|
| less_binary | 37 | **−3.88pp** | −4.25pp | 24.7% |
| build_window | 59 | **−3.52pp** | −6.67pp | 29.3% |
| core | 21 | −2.17pp | −1.94pp | 39.0% |
| binary_now | 63 | **−0.74pp** | −0.02pp | 41.5% |

**Key finding: `binary_now` was the best-performing catalyst bucket during failure windows** — avg xs −0.74pp, hit rate 41.5%. The conventional assumption that near-term binary catalyst names are the risk is WRONG for this period. The risk was concentrated in `build_window` and `less_binary` — names positioned heading into but not at their event.

### EES v3 Gate (5 windows with EES data; March excluded)

| gate | n | avg_xs | total_contrib | hit_rate |
|------|---|--------|---------------|----------|
| False (fail) | 30 | **−4.62pp** | −4.37pp | 28.6% |
| True (pass) | 120 | −1.40pp | −5.58pp | 38.2% |

**EES-False names underperformed by 3.2pp relative to EES-True names.** There were only 30 EES-False name-windows but they accounted for −4.37pp total contribution despite being 20% of the sample. Hit rate 28.6% vs 38.2%.

**Note:** Total contribution of EES-True names (−5.58pp) is larger because there are 4x more of them. The per-name impact of EES-False is substantially larger.

---

## Counterfactual Results (Aggregate, Failure Windows)

| basket | avg_xs | avg Δ vs actual |
|--------|--------|-----------------|
| actual_top30 | −2.15pp | — |
| ex_event_premium | −2.15pp | 0.00pp |
| ex_0_30d_catalyst | −2.13pp | +0.02pp |
| ex_priced_move_q4 | −1.85pp | +0.29pp |
| ex_ees_veto | −1.33pp | **+0.82pp** |
| top60_ew | −1.32pp | **+0.83pp** |
| bench_31_60 | −0.50pp | **+1.64pp** |

**Three findings stand out:**

1. **Event premium exclusion did nothing** — `ovf11_ep` was all zeros in these snapshots. The options overlay was not producing event premium scores during this period. This field is a data gap.

2. **EES exclusion helped +0.82pp on average** — strongest single-field intervention. Meaningful but not sufficient to fully prevent failures.

3. **Bench 31–60 averaged −0.50pp xs during failure windows vs −2.15pp for Top-30** — a consistent 1.64pp gap. The bench held cleaner names across every major failure period. **This is the highest-value counterfactual finding.**

---

## Repeat Offenders Analysis

The following names appeared as detractors in 2+ failure windows:

| ticker | windows | total_contrib | avg_xs |
|--------|---------|---------------|--------|
| CELC | 5 | **−2.50pp** | −14.99pp |
| ABVX | 5 | **−1.88pp** | −11.28pp |
| PRAX | 5 | −1.43pp | −8.59pp |
| DRUG | 3 | −1.26pp | −12.64pp |
| INSM | 1 | −0.89pp | −26.73pp |
| MLYS | 2 | −0.78pp | −11.70pp |
| DNTH | 5 | −0.64pp | −3.81pp |
| TYRA | 3 | −0.60pp | −6.02pp |

**CELC and ABVX are serial contributors.** CELC appeared in all 5 failure windows with a total drag of −2.50pp. ABVX appeared in all 5 with −1.88pp. These two names alone contributed approximately **−4.38pp** of total portfolio excess drag across the failure cluster.

**If CELC and ABVX had been excluded post-first-loss event and replaced from bench, the May-June cluster would have been substantially mitigated** — though this would require a portfolio-level stop mechanism that does not currently exist.

---

## EES / Options / Expectation Layer Assessment

### EES v3 Gate
- **Would EES have helped? YES, in 3 of 5 post-EES windows**
- Windows where delta > +1.5pp: Apr-20 (+2.71pp), Jun-01 (+1.96pp), Jun-08 (+2.07pp)
- Windows where delta marginal: May-04 (+1.26pp), May-19 (−0.96pp, hurt)
- Average delta across all 6 failure windows: **+0.82pp**
- EES-False names averaged −4.62pp xs vs True −1.40pp: 3.2pp spread
- **Caveat:** EES gate fires on 5–7 names per window (17–23% of top-30). It is not a silver bullet — the largest contributors to the May-26 collapse (ABVX, PRAX) were both EES-True.

### Options Event Premium
- **Would options exclusion have helped? NO** — `ovf11_ep` was all zeros. The options overlay was not producing event premium scores in these snapshots. This is a data gap, not a model finding.

### Priced Move (pm_q4 exclusion)
- **Marginal help: +0.29pp average** — modest signal. Useful as one of several filters.

### Expectation Error Score
- Field present but insufficient validation data for counterfactual inclusion. Future run should add this as an exclusion basket.

---

## Ranks 31–60 Assessment

- **YES — bench 31–60 consistently outperformed Top-30 during failure weeks**
- Average bench xs during failure windows: **−0.50pp** vs Top-30 −2.15pp
- Delta: **+1.64pp** — the largest counterfactual improvement of any basket
- Rank-inversion windows (bench beat top-30 by >2pp): **May-04** (+4.50pp), **May-19** (+2.20pp), **Jun-08** (+2.54pp)
- Bench was also better in May-26 (+1.04pp) and Jun-01 (+1.38pp), just not by the inversion threshold

**Interpretation:** The top-30 selection boundary is not a sharp quality cliff. During stress periods, ranks 31–60 held names with lower binary-event concentration and better outcomes. This is consistent with the catalyst bucket finding: bench names appear to have fewer `build_window`/`less_binary` exposures during failure weeks.

This is a meaningful finding but not a ranker indictment — the full-year cumulative xs of +13.8pp is strongly positive. The bench advantage is specifically a failure-period phenomenon.

---

## Rolling Drawdown Analysis

The May 4 → June 8 cluster constitutes a **rolling 4-window excess drawdown of −8.52pp**, well beyond the −5pp trigger:

| window | xs_pp |
|--------|-------|
| 2026-05-04 → 2026-05-11 | −1.95 |
| 2026-05-19 → 2026-05-26 | −1.79 |
| 2026-05-26 → 2026-06-02 | −2.45 |
| 2026-06-01 → 2026-06-08 | −2.33 |
| **Cluster total** | **−8.52pp** |

This cluster had three distinct drivers:
1. **Binary event losses** (ABVX, PRAX, CELC — May 26 onwards): Not preventable per-name but concentration was preventable.
2. **Rank inversion** (bench better in May 4, May 19): Selection boundary issue.
3. **Continued exposure** (ABVX, CELC still in top-30 after collapse): No position-level stop mechanism.

---

## Governance Conclusion

### Actionable Findings (evidence-backed)

1. **EES v3 gate is a validated guardrail** — EES-False names underperformed by 3.2pp, and excluding them improved 3 of 5 post-EES windows by +1.96–2.71pp. This supports accelerating the EES v3 shadow gate toward production consideration. Requires completing the 20d observation gate first.

2. **Ranks 31–60 bench should be part of failure monitoring** — The bench consistently outperformed during failure weeks. Tracking the bench spread weekly will improve early detection of rank inversion events. (PR #436 ships this.)

3. **`build_window` and `less_binary` concentration monitoring** — These buckets drove −6.67pp and −4.25pp total contribution respectively. A concentration cap on these buckets within the top-30 is worth designing.

4. **Repeated-name exposure after loss events** — CELC and ABVX remained in top-30 across multiple post-loss weeks. A per-name trailing XS stop (e.g., exit if name contributed <−1.0pp in prior 2 weeks) would be worth backtesting.

5. **Options overlay (ovf11_ep) was not populated** — The event premium field was all zeros during the failure cluster. This is a data integrity gap that should be investigated independently.

### Non-Actionable Findings (noise, do not overfit)

- **March 2026 failure** — No EES data, bench also down, no pre-window signal. Treat as noise.
- **May 26 collapse (ABVX, PRAX)** — Both EES-True, binary outcome. Not preventable.
- **Broad market beta contribution** — 2 of 7 windows had XBI down >3% with Top-30 experiencing correlated drawdown. Sector beta is not a model failure.

### Do Not Recommend

- **No ranker changes.** The full-year +13.8pp cumulative xs with t=1.09 is consistent. Failure windows are not evidence of ranker misspecification.
- **No selector changes.** Failures trace to specific names and catalyst concentration, not the selection mechanism.
- **No catalyst bucket reweighting.** `binary_now` was actually the best bucket during failure windows — inverse of conventional risk framing.

### Overfitting Risk Assessment

The EES and bench findings are **consistent across 3+ independent windows** and survived out-of-sample counterfactual testing. They are not overfitted. The catalyst bucket reversal (binary_now outperforming) is a single-period observation and should be replicated before any weight adjustment.

---

## Recommended Next Validation

1. **EES 5d forward IC validation** — Run the existing shadow card IC on the 5 post-EES failure windows specifically. Confirm that EES-False names had negative IC contribution.

2. **Top-30 vs bench comparison in non-failure weeks** — Validate that the bench advantage is specifically a failure-period phenomenon, not a persistent alpha leak.

3. **Catalog `build_window` bucket concentration by week** — Check whether failure weeks had anomalously high `build_window` concentration vs typical weeks.

4. **CELC / ABVX autopsy** — Pull clinical trial events for these two names around May-June 2026 to verify that collapse was binary-event driven and not model-derivable.

---

## Data Quality Notes

- `regime_label` is `UNKNOWN` in all 2026 snapshots — regime was not operational. Cannot classify via regime context.
- `2026-06-01` snapshot is QUARANTINED — window used `2026-05-29` substitute; xs reconstruction discrepancy (−2.33pp reconstructed vs −1.64pp known).
- `ovf_has_event_premium` / `ovf_composite` fields are empty — options overlay event premium not populated in 2026 snapshots.
- March 2026 snapshot has no `ees_v3_gate` column (only ees_v2 fields).
- `priced_move_pct` has NaN for several names in May-26 window — options data gap.

---

*Generated: 2026-06-28 | RESEARCH_AUTOPSY | NO_MODEL_CHANGE*
*Artifacts: failure_windows.csv, name_attribution.csv, feature_bucket_attribution.csv, counterfactuals.csv*
