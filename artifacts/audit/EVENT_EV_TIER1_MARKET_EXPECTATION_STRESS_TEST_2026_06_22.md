# Event EV Tier 1 Market-Expectation Stress Test — 2026-06-22

**Verdict:** `PASS_EVENT_EV_TIER1_STRESS_TEST_SHADOW_ONLY_NO_MODEL_CHANGE`
**Caveat:** `TIER1_STACK_PARTIALLY_INSTRUMENTED_PRICED_MOVE_ONLY`

**Date:** 2026-06-22
**Scope:** Shadow-only diagnostic stress test of Tier 1 market-expectation inputs in the Event EV model
**Production model freeze:** ACTIVE — no ranker, selector, sizing, final_score, or gate changes
**Prerequisite tasks:** PASS (Tasks 1–3 locked at commit f701f283)

---

## Executive Summary

The Tier 1 market-expectation stress test is a **partial pass** on available data. Of five proposed
Tier 1 inputs, one is fully available and safe (`priced_move_pct` / straddle-implied move), one is
available as a partial proxy (`actual_implied_move_pctile` at 64% event-ticker coverage — a move-size
percentile, not an IV-level percentile), and three are absent from the pipeline
(`iv_percentile_or_rank` as a true IV level vs history, `options_volume_oi_quality`, and
`skew_asymmetry` above usable thresholds).

`short_interest_pct` does not materially overinfluence the baseline (max shift 0.016 from removal).
The single-name `estimate()` path is unsafe for `priced_move_pct` due to raw-float normalization;
the batch path (`estimate_batch()`) is safe and was used for 100% of evaluated events. No production
scoring files were modified by this test.

The verdict is `PASS` because the stress test was correctly executed within shadow boundaries and the
available partial composite behaves as expected. The `TIER1_STACK_PARTIALLY_INSTRUMENTED_PRICED_MOVE_ONLY`
caveat is mandatory: a true fully-instrumented Tier 1 composite requires the three missing inputs.

---

## 1. Input Availability

### 1.1 Tier 1 Input Status

| Input | Category | Coverage (291 universe) | Coverage (171 event tickers) | Status |
|-------|----------|------------------------|------------------------------|--------|
| `priced_move_pct` / straddle-implied move | Tier 1 ANCHOR | 253/291 (87%) | 161/171 (94%) | ✅ Available — batch-safe |
| `actual_implied_move_pctile` | Tier 1 partial proxy | 166/291 (57%) | 110/171 (64%) | ⚠️ Partial — move-size percentile only (not IV-level vs 1yr history) |
| `iv_percentile_or_rank` (true IV rank) | Tier 1 MISSING | 0/291 (0%) | 0/171 (0%) | ❌ Absent — requires rolling ATM IV vs 1-year history |
| `iv_term_structure_kink` (true kink) | Tier 1 MISSING | 0/291 (0%) | 0/171 (0%) | ❌ Absent — `opt_term_slope` is a slope proxy, not a true kink metric |
| `options_volume_oi_quality` | Tier 1 MISSING | 0/291 (0%) | 0/171 (0%) | ❌ Absent — `opt_liquidity_ok` is a binary flag only (not OI/volume composite) |
| `skew_asymmetry` (`opt_put_call_skew`) | Tier 1 sparse | 79/291 (27%) | 53/171 (31%) | ❌ Too sparse for cross-sectional ranking |

**Note on `actual_implied_move_pctile`:** This field (57% coverage) represents where the current
straddle-implied move ranks historically for that ticker. It is *not* a true IV level percentile
(i.e., it does not answer "is ATM IV high vs its 1-year history"). It is used as the best available
partial proxy for input 2, clearly labeled.

**Note on `opt_term_slope`:** Available at 96% coverage (280/291) and used in the stress test as a
term structure proxy. It is the slope of the IV surface across expiries, not a kink deviation from
a smooth term structure. It is an adequate proxy for backwardation/contango direction, but does not
isolate catalyst-driven term structure dislocations.

### 1.2 Available Non-Tier-1 Baseline Inputs

| Input | Coverage (event tickers) | Current Weight | Role |
|-------|--------------------------|----------------|------|
| `coinvest_score_z` | 171/171 (100%) | 0.30 | Institutional co-investment |
| `inst_delta_z` | 171/171 (100%) | 0.20 | Institutional accumulation delta |
| `de_alpha_60d` | 171/171 (100%) | 0.15 | Pre-event drift |
| `de_rsi_14d` | 171/171 (100%) | 0.10 | Momentum |
| `insider_net_buy_z` | 0/171 (0%) | 0.10 | **INERT** — Form 4 lane closed 2026-04-05 |
| `short_interest_pct` | 169/171 (99%) | 0.05 (inverted) | Crowding/risk modifier only |

---

## 2. Events Evaluated

| Metric | Value |
|--------|------:|
| Total event graph nodes | 4,268 |
| Evaluated (0–180d cohort) | 428 |
| Unique tickers in leaderboard | 197 |
| Tickers matched to rankings.csv | 171 |
| Tickers not in screener universe | 26 |
| Unique matched tickers evaluated (one best event per ticker) | 171 |
| Tickers with `priced_move_pct` | 161/171 (94%) |
| Tickers with `actual_implied_move_pctile` | 110/171 (64%) |
| Tickers with `opt_term_slope` | 166/171 (97%) |
| Tickers with `opt_put_call_skew` | 53/171 (31%) |
| Tickers skipped (not in universe) | 26 — no market features in rankings.csv |

**Skipped reason:** 26 event tickers originate from SEC 8K ledger, PDUFA, and M3 supplement but
are not in the 291-ticker screener universe and have no corresponding market features.

**Shadow path used:** Cross-sectional percentile normalization on all 291-ticker universe, then
event-ticker extraction. No single-name `estimate()` call was made.

---

## 3. Stress Test Results

### 3.1 Baseline Model Weights

| Feature | Weight | Source Classification |
|---------|--------|-----------------------|
| `coinvest_score_z` | 0.30 | Institutional (dominant) |
| `inst_delta_z` | 0.20 | Institutional accumulation |
| `de_alpha_60d` | 0.15 | Pre-event drift |
| `de_rsi_14d` | 0.10 | Momentum |
| `insider_net_buy_z` | 0.10 | **INERT** (Form 4 lane closed) |
| `priced_move_pct` | 0.05 | Direct market expectation |
| `event_premium` (front_iv − back_iv) | 0.05 | Term structure proxy |
| `short_interest_inv` | 0.05 | Crowding/risk modifier |

**Institutional signals dominate at 50% effective weight. Direct market-expectation inputs
contribute 10% (0.05 + 0.05). With insider lane inert, effective denominator is ~90%.**

### 3.2 Tier 1-Heavy Shadow Composite (partial — 1/5 complete + 2/5 proxy)

Proposed weights applied with available inputs only:

| Feature | Proposed Weight | Available | Disposition |
|---------|----------------|-----------|-------------|
| `priced_move_pct` | 0.35 | ✅ | Used — cross-sectional percentile rank |
| `iv_percentile_or_rank` | 0.20 | ❌ | `actual_implied_move_pctile` used as partial proxy (labeled) |
| `iv_term_structure_kink` | 0.20 | ❌ | `opt_term_slope` percentile used as slope proxy (labeled) |
| `options_volume_oi_quality` | 0.10 | ❌ | Dropped — `opt_liquidity_ok` is binary, not a composite |
| `skew_asymmetry` | 0.15 | ❌ | `opt_put_call_skew` used where available (31% coverage); dropped for rest |

**Effective Tier 1-Heavy weights (renormalized over available inputs, per ticker):**

Varying per ticker depending on which proxy fields are available. Approximation:

| Feature | Approximate Weight |
|---------|-------------------|
| `priced_move_pct` | 0.35 |
| `actual_implied_move_pctile` (or `opt_atm_iv` pctile) | 0.20 |
| `opt_term_slope` pctile | 0.20 |
| `opt_put_call_skew` pctile (where available) | 0.15 |
| `coinvest_score_z` | 0.15 |
| `inst_delta_z` | 0.10 |

**Results (computed from live 2026-06-22 data, shadow only):**
- Events with per-ticker best-event selected: **171**
- Mean |shift| in shadow implied_p_hit vs baseline: **0.0606**
- Max single-ticker shift: **+0.176 (OCGN)**
- Min single-ticker shift: **−0.197 (APGE)**

### 3.3 Top 15 Expectation-Gap Changes (Baseline → Tier 1-Heavy)

Computed from live rankings.csv and shadow event EV scores. Shadow diagnostic only.

| Ticker | p_hit | Baseline implied | Tier1H implied | Shift | PM | IVPctile | Skew |
|--------|-------|-----------------|----------------|-------|----|----------|------|
| APGE | 0.346 | 0.596 | 0.399 | −0.197 | ✅ | ✅ | — |
| TRVI | 0.350 | 0.637 | 0.452 | −0.184 | ✅ | ✅ | — |
| OCGN | 0.556 | 0.394 | 0.569 | +0.176 | ✅ | ✅ | ✅ |
| BDTX | 0.397 | 0.399 | 0.570 | +0.171 | ✅ | — | — |
| KYTX | 0.355 | 0.419 | 0.586 | +0.167 | ✅ | ✅ | — |
| IVVD | 0.556 | 0.471 | 0.626 | +0.155 | ✅ | ✅ | — |
| VYGR | 0.538 | 0.445 | 0.600 | +0.155 | ✅ | — | — |
| ABOS | 0.359 | 0.418 | 0.569 | +0.150 | ✅ | — | ✅ |
| CMPS | 0.584 | 0.630 | 0.485 | −0.145 | ✅ | ✅ | ✅ |
| ADCT | 0.599 | 0.486 | 0.631 | +0.145 | ✅ | ✅ | ✅ |
| COGT | 0.614 | 0.616 | 0.474 | −0.142 | ✅ | ✅ | — |
| RVMD | 0.599 | 0.661 | 0.520 | −0.141 | ✅ | — | ✅ |
| ALT | 0.559 | 0.434 | 0.565 | +0.131 | ✅ | — | ✅ |
| GOSS | 0.606 | 0.429 | 0.553 | +0.124 | ✅ | ✅ | ✅ |
| BMEA | 0.405 | 0.416 | 0.538 | +0.122 | ✅ | — | ✅ |

**Key:** PM = priced_move_pct available; IVPctile = actual_implied_move_pctile available; Skew = opt_put_call_skew available.

**Interpretation (shadow only):** Shifts are material (mean |shift| 0.061) and directionally
plausible. Tickers with high `priced_move_pct` rank shift upward (OCGN, KYTX, IVVD), reflecting
elevated options-market uncertainty. Tickers where institutional signals drove high baseline implied
probability but market-expectation inputs are lower shift downward (APGE, TRVI, COGT, RVMD). This
directional behavior is expected under Tier 1-Heavy weighting.

**No action, alpha promotion, or selection implication.**

### 3.4 Ablation 3 — Excluding `short_interest_pct`

| Ticker | Baseline implied | No-SI implied | Shift | SI% |
|--------|-----------------|---------------|-------|-----|
| ONC | 0.414 | 0.398 | −0.016 | 1.53% |
| VRTX | 0.414 | 0.398 | −0.016 | 2.00% |
| TNGX | 0.617 | 0.633 | +0.016 | 23.46% |
| GILD | 0.427 | 0.411 | −0.015 | 2.07% |
| CLLS | 0.440 | 0.425 | −0.015 | 0.64% |

**Mean |shift| from removing short_interest_pct: 0.0066**
**Max |shift|: 0.0162**

`short_interest_pct` is functioning as a low-weight crowding/risk modifier at its current 0.05
weight. **No overinfluence finding.** Short interest does not drive outcomes in the baseline.

### 3.5 Ablation 4 — `priced_move_pct` as Dominant Tier 1 Anchor

Weights: `priced_move_pct`=0.35, `coinvest_score_z`=0.25, `inst_delta_z`=0.20,
`de_alpha_60d`=0.10, `event_premium`=0.10

- Mean |shift| vs baseline: **0.0381**
- Max shift: **+0.106**
- Directional behavior consistent with Tier 1-Heavy
- No anomalies — safe to use as anchor in batch mode
- Confirms priced_move_pct is a stable input when cross-sectionally normalized

---

## 4. Normalization Safety — `priced_move_pct`

**FINDING: Single-name estimate() path is UNSAFE for priced_move_pct.**

| Mode | Normalization | Status |
|------|--------------|--------|
| `estimate_batch()` (production path) | Cross-sectional percentile rank → [0, 1] | ✅ SAFE |
| `estimate()` (single-name path) | Raw percentage value used directly | ❌ UNSAFE |

`priced_move_pct` values in rankings.csv range from **1.12% (APGE) to 2078.94% (TBPH)** with a
mean of 101.27%. In `_normalize_features()`, this value is stored as a raw float without clamping
to [0, 1]. If `priced_move_pct` weight were increased in a single-name path, TBPH would contribute
`2078.94 × w` to the belief score — completely dominating.

**The current production scoring path uses `estimate_batch()` only**, so this normalization flaw
does not affect production outputs. However, the single-name `estimate()` method must not be used
with elevated `priced_move_pct` weight without fixing the normalization in `_normalize_features()`.

**Recommended fix (future, non-blocking):** In `_normalize_features()`, clamp `priced_move_pct`
to a reasonable range (e.g., `_clamp(v / 100.0, 0, 1)` treating 100% move as the cap).

This stress test used **zero single-name estimate() calls**. Normalization safety is confirmed for
the batch path.

---

## 5. Missing Tier 1 Input Report

| Input | Current Field | Gap | Path to Instrument |
|-------|--------------|-----|-------------------|
| `iv_percentile_or_rank` | None (`actual_implied_move_pctile` is move-size pctile, not IV-level pctile) | HIGH | Compute rolling IV percentile from `opt_atm_iv` vs 1-year history; requires daily IV series storage |
| `iv_term_structure_kink` | `opt_term_slope` (slope proxy) | MEDIUM | Define kink as deviation from smooth power-law term structure; requires multi-expiry chain (front/back insufficient) |
| `options_volume_oi_quality` | `opt_liquidity_ok` (binary flag), `opt_liquidity_state` (thin/liquid/absent) | HIGH | Add options OI and volume to daily options refresh; current binary liquidity gate is insufficient for composite scoring |
| `skew_asymmetry` | `opt_put_call_skew` (27% coverage), `opt_rr_25d` (19% coverage) | HIGH | Expand options chain coverage; current sparse liquidity means skew is only available for the most liquid 20-30% of the universe |

**Summary:** Only `priced_move_pct` (straddle-implied move) is fully instrumented as a Tier 1
input. `actual_implied_move_pctile` provides a partial proxy for the IV percentile concept but
measures move size vs history rather than IV level vs history — a meaningfully different signal.
`opt_term_slope` provides a usable but imprecise proxy for term structure shape.

Until `iv_percentile_or_rank`, `options_volume_oi_quality`, and broad `skew_asymmetry` are
instrumented, the proposed Tier 1 composite cannot be computed at full weight. The partial composite
(`priced_move_pct` as dominant anchor, with `opt_term_slope` as structure proxy) is the correct
shadow baseline for the current data state.

---

## 6. Safety Check — Production Isolation

**No production/scoring files were touched by this stress test.**

All computations were performed in-memory from:
- `artifacts/audit/event_ev_shadow_2026_06_22/2026-06-22_event_ev_scores.json` (read-only)
- `data/snapshots/2026-06-22/rankings.csv` (read-only)

| Safety check | Result |
|-------------|--------|
| `ranker_v2_score` modified | NO |
| `final_score` modified | NO |
| `selector_score` modified | NO |
| `sizing` modified | NO |
| Eligibility fields modified | NO |
| Action/gate outputs modified | NO |
| Snapshot directories modified | NO |
| Production data modified | NO |
| Single-name `estimate()` called | NO — 0/171 tickers |
| RSI/alpha_60d/inst_delta substituted for missing options inputs | NO — labeled as baseline, not Tier 1 |
| `short_interest_pct` treated as Tier 1 | NO — confirmed crowding modifier only |
| Alpha promotion language | NONE |
| Trading/action language | NONE |

**Pre-existing git working-tree modifications (not from this test):**
- `artifacts/audit/SCIENTIFIC_CARTOGRAPHY_OPERATIONAL_REVIEW_2026_06_22.md` — linter reformat
- `artifacts/audit/cross_signal_forward_shadow/buckets.jsonl` — cron write (pre-existing)
- `artifacts/audit/inst_delta_forward_shadow/checkpoints.jsonl` — cron write (pre-existing)
- `data/expression_decision_log.jsonl` — daily production cron write (pre-existing)

---

## 7. Conclusions

### Verdict: `PASS_EVENT_EV_TIER1_STRESS_TEST_SHADOW_ONLY_NO_MODEL_CHANGE`
### Caveat: `TIER1_STACK_PARTIALLY_INSTRUMENTED_PRICED_MOVE_ONLY`

### Pre-report answers

| Question | Answer |
|---------|--------|
| Available Tier 1 inputs | `priced_move_pct` (94% event-ticker coverage) |
| Available Tier 1 partial proxies | `actual_implied_move_pctile` (64%), `opt_term_slope` (97%) |
| Missing Tier 1 inputs | `iv_percentile_or_rank` (true), `options_volume_oi_quality`, `skew_asymmetry` (sparse) |
| Events evaluated | 171 matched tickers (best event per ticker); 428 total in 0–180d cohort |
| Events skipped | 26 tickers not in screener universe; 3,840 nodes outside evaluation window |
| `priced_move_pct` normalization safe | YES — batch path only; single-name path unsafe (documented) |
| `short_interest_pct` materially changes top gaps | NO — max removal shift 0.016, mean 0.007 |
| Production/scoring files touched | NO — confirmed |

### What this test proved (positive)

1. `priced_move_pct` is available, safe, and correctly normalized in batch mode
2. `opt_term_slope` (97% coverage) provides a usable IV term structure proxy
3. `actual_implied_move_pctile` (64% coverage) provides a partial move-size percentile proxy
4. `short_interest_pct` is NOT overinfluent at its current 0.05 weight (max shift 0.016)
5. Upweighting direct market-expectation inputs to 0.35+ produces material (mean |shift| 0.061)
   but directionally plausible changes in shadow implied probability
6. The current production path is safe for `priced_move_pct` — no normalization bug in production
7. No bad-outcome conditions were triggered (see safety check table)

### What remains unproven (Tier 1 stack incomplete)

1. Whether a true IV rank (IV level vs 1-year history) would complement or dominate `priced_move_pct`
2. Whether OI/volume quality would materially filter or reweight expectations vs the institutional model
3. Whether broad skew coverage (currently 31% of event tickers) would confirm or reverse directional
   shifts observed in the partial composite
4. Whether the partial composite improves forward IC vs the institutional-dominant baseline

### Non-blocking maintenance items

1. Fix `_normalize_features()` `priced_move_pct` normalization for single-name path
   (clamp to [0, 1] using a reasonable max, e.g. 100% = 1.0)
2. Instrument `iv_percentile_or_rank` from rolling `opt_atm_iv` history (requires daily IV series)
3. Expand options chain OI/volume data fetch for `options_volume_oi_quality`
4. Expand options chain coverage to improve `skew_asymmetry` above 31% event-ticker coverage

### Freeze status

Production model freeze remains ACTIVE. This test produced no changes to any model weights,
scoring logic, snapshots, or production outputs. No alpha promotion. No action language.

---

*Diagnostic only — does not feed selector/ranker/construction.*
*Inputs: artifacts/audit/event_ev_shadow_2026_06_22/2026-06-22_event_ev_scores.json (read-only),*
*data/snapshots/2026-06-22/rankings.csv (read-only). No files written to production paths.*
