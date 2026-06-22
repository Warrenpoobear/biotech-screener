# Event EV Tier 1 Market-Expectation Stress Test — 2026-06-22

**Verdict:** `FAIL_TIER1_INPUTS_INCOMPLETE`

**Date:** 2026-06-22
**Scope:** Shadow-only diagnostic stress test of market-expectation inputs in the Event EV expectation model
**Production model freeze:** ACTIVE — no ranker, selector, sizing, final_score, or gate changes

---

## Executive Summary

The Tier 1 market-expectation stress test is a **partial pass**. Two of five proposed Tier 1 inputs
(`priced_move_pct` and `event_premium_magnitude`) are available and safe in batch mode with strong
coverage. Three inputs — `iv_percentile_or_rank`, `options_volume_oi_quality`, and `skew_asymmetry`
— are not present in the data pipeline. The verdict is `FAIL_TIER1_INPUTS_INCOMPLETE` because the
full Tier 1 composite cannot be computed until those inputs are sourced.

`short_interest_pct` does not materially overinfluence the baseline. Single-name
`priced_move_pct` normalization is unsafe (raw percentage, not [0,1]); the batch path is safe.
No production scoring files were modified.

---

## 1. Input Availability

### 1.1 Available Tier 1 Inputs (partial composite only)

| Input | Coverage (screener universe) | Coverage (171 matched event tickers) | Status |
|-------|------------------------------|--------------------------------------|--------|
| `priced_move_pct` | 253/291 (87%) | 161/171 (94%) | ✅ Available |
| `event_premium_magnitude` (front_iv − back_iv) | 280/291 (96%) | 166/171 (97%) | ✅ Available |

### 1.2 Missing Tier 1 Inputs

| Input | Coverage | Status |
|-------|---------|--------|
| `iv_percentile_or_rank` | 0/291 (0%) | ❌ Not in rankings.csv — requires historical IV series |
| `iv_term_structure_kink` | 0/291 (0%) | ❌ Not computed — `opt_term_slope` is a proxy only |
| `options_volume_oi_quality` | 0/291 (0%) | ❌ Not in rankings.csv — requires options chain OI data |
| `skew_asymmetry` (`opt_put_call_skew`) | 79/291 (27%) / 53/171 (31%) | ⚠️ Insufficient — too sparse for cross-sectional ranking |

### 1.3 Available Non-Tier-1 Inputs (baseline model)

| Input | Coverage (matched events) | Current Weight |
|-------|--------------------------|----------------|
| `coinvest_score_z` | 171/171 (100%) | 0.30 |
| `inst_delta_z` | 171/171 (100%) | 0.20 |
| `alpha_60d` | 171/171 (100%) | 0.15 |
| `rsi_14d` | 171/171 (100%) | 0.10 |
| `short_interest_pct` | 169/171 (99%) | 0.05 (inverted) |

---

## 2. Events Evaluated

| Metric | Value |
|--------|------:|
| Total events in leaderboard | 428 |
| Unique tickers in leaderboard | 197 |
| Tickers matched to rankings.csv | 171 |
| Tickers not in screener universe | 26 |
| Tickers with `priced_move_pct` available | 161 |
| Tickers with `event_premium_magnitude` available | 166 |
| Tickers skipped (no market features) | 26 |

**Skipped reason:** 26 event tickers (from SEC 8K ledger, PDUFA, and M3 supplement) are not in the
screener universe and have no market features in rankings.csv.

---

## 3. Stress Test Results

### 3.1 Baseline Current Model Weights

| Feature | Weight | Source |
|---------|--------|--------|
| `coinvest_score_z` | 0.30 | Institutional (dominant) |
| `inst_delta_z` | 0.20 | Institutional accumulation |
| `alpha_60d` | 0.15 | Pre-event drift |
| `rsi_14d` | 0.10 | Momentum |
| `insider_net_buy_z` | 0.10 | **INERT** post 2026-04-05 Form 4 lane closure |
| `priced_move_pct` | 0.05 | Direct market expectation (pass-through diagnostic) |
| `event_premium_magnitude` | 0.05 | Term structure inversion |
| `short_interest_inv` | 0.05 | Crowding/risk modifier |

**Institutional signals dominate at 50% effective weight (0.30 + 0.20). Direct market-expectation
inputs (options) contribute 10% (0.05 + 0.05).**

### 3.2 Tier 1-Heavy Shadow Composite (partial — 2/5 inputs)

Proposed composite applied with available inputs only:

| Feature | Proposed Weight | Available |
|---------|----------------|-----------|
| `priced_move_pct` | 0.35 | ✅ |
| `iv_percentile_or_rank` | 0.20 | ❌ Missing |
| `iv_term_structure_kink` | 0.20 | ❌ Missing |
| `event_premium_magnitude` | 0.10 | ✅ (substituted for iv_term_structure_kink) |
| `options_volume_oi_quality` | 0.10 | ❌ Missing |
| `skew_asymmetry` | 0.15 | ❌ Too sparse |

**Effective Tier 1-Heavy weights run (partial, 2/5 inputs):**

| Feature | Weight |
|---------|--------|
| `priced_move_pct` | 0.35 |
| `event_premium_magnitude` | 0.20 |
| `coinvest_score_z` | 0.20 |
| `inst_delta_z` | 0.10 |
| `alpha_60d` | 0.10 |
| `short_interest_pct` | 0.05 |

**Results:**
- Mean |shift| in `implied_p_hit` vs baseline: **0.0814** (material)
- Max single-ticker shift: +0.224 (BDTX), -0.203 (MIRM)

### 3.3 Top 15 Expectation-Gap Changes (Baseline → Tier 1-Heavy)

| Ticker | p_hit | Baseline implied | Tier1H implied | Shift | PM | TS | Skew |
|--------|-------|-----------------|----------------|-------|----|----|------|
| BDTX | 0.397 | 0.335 | 0.559 | +0.224 | ✅ | ✅ | — |
| AARD | 0.556 | 0.372 | 0.594 | +0.222 | ✅ | ✅ | — |
| UPB | 0.346 | 0.403 | 0.622 | +0.219 | ✅ | ✅ | — |
| MIRM | 0.584 | 0.639 | 0.436 | -0.203 | ✅ | ✅ | — |
| CTNM | 0.350 | 0.459 | 0.658 | +0.199 | ✅ | ✅ | — |
| ACIU | 0.363 | 0.348 | 0.547 | +0.199 | ✅ | ✅ | — |
| TARA | 0.602 | 0.405 | 0.595 | +0.190 | ✅ | ✅ | — |
| APGE | 0.346 | 0.640 | 0.453 | -0.187 | ✅ | ✅ | — |
| DNTH | 0.562 | 0.635 | 0.450 | -0.185 | ✅ | ✅ | — |
| CLLS | 0.321 | 0.331 | 0.515 | +0.184 | ✅ | ✅ | — |
| ALT | 0.559 | 0.354 | 0.538 | +0.184 | ✅ | ✅ | ✅ |
| TBPH | 0.593 | 0.417 | 0.596 | +0.179 | ✅ | ✅ | — |
| TVTX | 0.603 | 0.548 | 0.371 | -0.177 | ✅ | ✅ | — |
| GERN | 0.603 | 0.451 | 0.625 | +0.174 | ✅ | ✅ | ✅ |
| AVIR | 0.556 | 0.402 | 0.576 | +0.174 | ✅ | ✅ | — |

**Interpretation:** Shifts are material (mean 0.08) but directionally plausible — high-priced-move
tickers shift upward in Tier 1-Heavy (market pricing more uncertainty), low-priced-move tickers
(MIRM, DNTH, APGE, TVTX) shift downward. This is the expected directional behavior.

### 3.4 Ablation 3 — Excluding `short_interest_pct`

| Ticker | Baseline implied | No-Short implied | Shift | SI% |
|--------|-----------------|-----------------|-------|-----|
| ALNY | 0.324 | 0.372 | +0.048 | 5.3% |
| TECX | 0.572 | 0.529 | -0.043 | 26.0% |
| BCYC | 0.419 | 0.457 | +0.038 | 7.4% |
| BIIB | 0.466 | 0.502 | +0.036 | 3.6% |
| CLLS | 0.331 | 0.364 | +0.033 | 0.6% |

**Max shift from removing short_interest_pct: 0.048** — not material. `short_interest_pct` is
functioning as a crowding/risk modifier at its current 0.05 weight. **No overinfluence finding.**

### 3.5 Ablation 4 — `priced_move_pct` as Dominant Tier 1 Anchor

Weights: `priced_move_pct`=0.35, `coinvest_score_z`=0.25, `inst_delta_z`=0.20, `alpha_60d`=0.10,
`event_premium_magnitude`=0.10

- Mean |shift| vs baseline: **0.0622** (moderate, smaller than full Tier 1-Heavy)
- Directional behavior consistent with Tier 1-Heavy
- No anomalies — safe to use as anchor in batch mode

---

## 4. Normalization Safety — `priced_move_pct`

**FINDING: Single-name estimate() path is UNSAFE for priced_move_pct.**

| Mode | Normalization | Status |
|------|--------------|--------|
| `estimate_batch()` (production path) | Cross-sectional percentile rank → [0, 1] | ✅ SAFE |
| `estimate()` (single-name path) | Raw percentage value (1.12% to 2078.94%) used directly | ❌ UNSAFE |

`priced_move_pct` values in rankings.csv range from **1.12% (APGE) to 2078.94% (TBPH)**. In
`_normalize_features()`, this value is stored as a raw float without clamping to [0, 1]. If
`priced_move_pct` weight were increased in a single-name path, TBPH would contribute `2078.94 × w`
to the belief score — completely dominating.

**The current production scoring path uses `estimate_batch()` only**, so this normalization flaw
does not affect production outputs. However, the single-name `estimate()` method should not be
used with elevated `priced_move_pct` weight without fixing the normalization in
`_normalize_features()`.

**Recommended fix (future, non-blocking):** In `_normalize_features()`, clamp `priced_move_pct`
to a reasonable range (e.g., `_clamp(v / 100.0, 0, 1)` treating 100% move as the cap).

---

## 5. Missing Tier 1 Input Report

| Input | Reason Missing | Path to Instrument |
|-------|---------------|-------------------|
| `iv_percentile_or_rank` | Not precomputed; requires historical IV series | Compute rolling IV percentile from `opt_atm_iv` vs 1-year history |
| `iv_term_structure_kink` | Not defined; `opt_term_slope` is a proxy | Define kink as deviation from smooth term structure; requires multi-expiry chain |
| `options_volume_oi_quality` | Not in rankings.csv; requires options chain fetch | Add OI and volume to daily options refresh from Robinhood/market data |
| `skew_asymmetry` (`opt_put_call_skew`) | 31% coverage on event tickers — too sparse | Expand options chain coverage; current sparse liquidity limits this signal |

**Until these four inputs are instrumented, the proposed Tier 1 composite cannot be computed
at full weight. The partial composite (priced_move_pct + event_premium_magnitude) is the correct
shadow baseline for now.**

---

## 6. Safety Check — Production Isolation

**No production/scoring files were touched by this stress test.**

All computations were performed in-memory from:
- `artifacts/audit/event_ev_shadow_2026_06_22/2026-06-22_event_ev_scores.json` (read-only)
- `data/snapshots/2026-06-22/rankings.csv` (read-only)

Git status check:
- `rankings.csv`: not modified
- `ranker_v2_score`, `final_score`, `selector_score`: not present in any modified file
- Snapshot directories: not modified
- Production data: not modified

Pre-existing modifications on the working tree (not from this test):
- `artifacts/audit/SCIENTIFIC_CARTOGRAPHY_OPERATIONAL_REVIEW_2026_06_22.md` — linter reformat
- `artifacts/audit/cross_signal_forward_shadow/buckets.jsonl` — cron write (pre-existing)
- `artifacts/audit/inst_delta_forward_shadow/checkpoints.jsonl` — cron write (pre-existing)
- `data/expression_decision_log.jsonl` — daily production cron write (pre-existing)

---

## 7. Conclusions and Follow-Up

### Verdict: `FAIL_TIER1_INPUTS_INCOMPLETE`

The Tier 1 stress test is a **partial pass** on available inputs. The two available inputs
(`priced_move_pct` and `event_premium_magnitude`) behave as expected and are safe in batch mode.
Three of five proposed inputs are not yet in the pipeline.

### What this test proved (positive)

1. `priced_move_pct` and `event_premium_magnitude` are available, safe, and correctly
   cross-sectionally normalized in the batch path
2. `short_interest_pct` is NOT overinfluent at its current 0.05 weight (max shift 0.048)
3. Upweighting direct market-expectation inputs to 0.35+0.20 produces material but directionally
   plausible shifts in `implied_p_hit` (mean 0.08)
4. Current production path (`estimate_batch`) is safe for `priced_move_pct` — no normalization
   bug in production

### What remains unproven

1. Whether the full Tier 1 composite (with IV rank, OI/volume, skew) would dominate or
   complement the institutional signals
2. Whether Tier 1-Heavy weighting improves IC vs the baseline institutional-dominant model
3. Whether the expectation-gap changes (BDTX, AARD, MIRM, APGE) reflect genuine mispricing
   corrections or noise amplification from sparse inputs

### Non-blocking maintenance items

1. Fix `_normalize_features()` priced_move_pct normalization for single-name path
   (clamp to [0, 1] using a reasonable max, e.g. 100% = 1.0)
2. Instrument `iv_percentile_or_rank` from rolling `opt_atm_iv` history
3. Expand options chain OI/volume data fetch for `options_volume_oi_quality`

### Freeze status

Production model freeze remains ACTIVE. This test produced no changes to any model weights,
scoring logic, snapshots, or production outputs. No alpha promotion. No action language.

---

*Diagnostic only — does not feed selector/ranker/construction.*
