# Spec 116 — Expectation Error Score (EES) Quality Model

**Status:** IMPLEMENTED (overlay-only) / MONITORING  
**Created:** 2026-11-15  
**Governance Tier:** 4 (Research Judgment) for any promotion; Tier 2 (Medium) for current overlay-only status  
**Module:** `event_ev/expectation_error_model.py`  
**Data Contract:** `event_ev/data_contracts.py` (`ExpectationErrorScore`)  
**Parent Spec:** Spec 057 (Bayesian Biotech Event EV Engine, Layer 4)  
**Companion Specs:** Spec 060 (Daily Event EV Scoring), Spec 105 (Expectation Layer Coverage Verification)

---

## 1. Motivation

The DEM production pipeline ranks names by institutional signal (coinvest_score_z). It
does not model whether the options market is pricing catalysts correctly. Two systematic
errors are observable in biotech event pricing:

1. **Trap error:** Names that appear cheap relative to base-rate distributions often
   underperform. The crowd knows something the naive comparison misses. Screening for
   "obvious value" in event-driven biotech is adversely selected.

2. **Structure error:** Expensive implied moves on catalysts with uncertain timing create
   negative-carry positions. The thesis may be correct, but the cost of being early scales
   with timing uncertainty. Good ideas in bad structures drag portfolios.

Neither error is captured by the current production stack (selector, ranker, or decision
engine). EES detects both as a post-scoring diagnostic overlay, without touching the
decision path.

### Intellectual Framework

EES descends from a "six-mistake" framework for structured expectation errors in market
pricing. The original six mistakes were:

1. Base-rate neglect (market ignores historical move distributions)
2. Mispriced conditionals (scenario-weighted EV diverges from implied price)
3. Execution friction (thin books and slippage in small caps)
4. Platform divergence (option surface disconnected from stock behavior)
5. Favorite bias (crowding distorts pricing via short-interest dynamics)
6. Timing decay (expensive setups with uncertain catalysts bleed value)

Empirical validation showed only mistakes 1, 2, and 6 carry signal. Mistakes 3-5 are
either noise (divergence, crowding) or contaminated (slippage). The surviving model is
simpler than the framework that produced it.

### The Inversion Insight

The critical finding was that the surviving signals work in the opposite direction from
their naive interpretation:

- **Base-rate gap and conditional misprice were designed to find underpriced names.** In
  practice, names that appear underpriced by these measures underperform. The signal value
  is in *avoiding* apparent bargains, not in buying them.

- **Timing decay risk was designed to measure execution drag.** In practice, it is the
  strongest standalone quality signal — names with certain timing and moderate implied
  moves outperform.

Both overlays are therefore negatively coded: higher scores = safer positions, not better
alpha opportunities. EES is a drag-removal system, not an alpha-generation system.

---

## 2. Architecture

### 2.1 Position in the Event EV Stack

```
Spec 057 Event EV Engine
├── Layer 1: Catalyst Graph (catalyst_graph.py)
├── Layer 2: Timing Hazard (timing_model.py)
├── Layer 3: Outcome Model (outcome_model.py)
├── Layer 4: Expectation Model ← EES lives here
│   └── expectation_error_model.py
├── Layer 5: Payoff Engine (payoff_engine.py)
└── Layer 6: Portfolio Layer (not implemented)
```

EES is Layer 4's primary output. It consumes features already present in `rankings.csv`
(no new data dependencies) and writes diagnostic columns back into the same CSV plus
sidecar JSON artifacts.

### 2.2 Data Flow

```
rankings.csv (priced_move_pct, short_interest_pct, market_cap_mm,
              close_price, implied_event_move, catalyst_family,
              lead_program_phase, clinical_days_precision)
        │
        ▼
ExpectationErrorModel.score_batch()
        │
        ├── Cross-sectional SI percentiles (P50, P90) computed per batch
        ├── Per-ticker: 6 sub-scores → v1 composite (deprecated) + v2 overlays
        │
        ▼
ExpectationErrorModel.compute_gates()
        │
        ├── Percentile-based quality and trap gates
        ├── Degeneracy detection (>80% identical values → bypass gate)
        │
        ▼
enrich_csv_rows()
        │
        ├── 15 columns injected into csv_rows (in-place)
        ├── Returns List[ExpectationErrorScore] for sidecar JSON
        │
        ▼
build_gate_diagnostics()
        │
        └── ees_gate_diagnostics.json (distribution stats, filtered names, correlation)
```

### 2.3 Isolation Policy

EES is **overlay-only**. It is explicitly prohibited from entering the decision path:

- `expectation_error_model.py` must NEVER be imported by `selector_engine.py`,
  `ranker_engine.py`, or `decision_engine.py`
- EES columns in `rankings.csv` are for operator review and downstream research only
- Gate outputs (`ees_quality_gate`, `ees_trap_gate`, `ees_eligible`) are informational —
  they do not filter the production Top-30

Promotion to the decision path would be a **Tier 4 governance decision** requiring:
a Checklist v2 battery, forward shadow evidence, a promotion spec, and operator sign-off.

---

## 3. Sub-Score Specifications

All sub-scores are normalized to approximately [-1, +1] via `tanh(v/2)` soft-scaling
(except timing_decay_risk, which is hard-clamped to [0, 1]).

### 3.1 Base-Rate Gap Score (B)

**Purpose:** Detect base-rate neglect — how far the market's implied move deviates from
the historical realized distribution for this event class.

**Formula:**
```
B = tanh((priced_move_pct - hist_median) / (hist_IQR + ε) / 2)
```

**Lookup table** (keyed by `{catalyst_family}|{phase_bucket}`):

| Family|Phase | Median (p50) | IQR (p75-p25) | Source |
|---|---|---|---|
| CLINICAL\|phase3 | 35.0% | 37.0% | payoff_engine blended HIT/MISS |
| CLINICAL\|phase2 | 35.0% | 40.0% | " |
| CLINICAL\|early | 25.0% | 38.0% | " |
| REGULATORY\|phase3 | 19.0% | 27.0% | " |
| REGULATORY\|phase2 | 23.5% | 30.0% | " |
| REGULATORY\|early | 20.0% | 30.0% | " |
| SAFETY\|any | 20.0% | 35.0% | " |

**Phase bucketing:** `lead_program_phase >= 3` → phase3, `>= 2` → phase2, else → early.

**Interpretation:** Positive B = market implies a larger move than base rates suggest.
Negative B = market implies a smaller move than base rates. As a trap component, high B
(expensive vs base rate) is penalized.

**Missing data:** Returns 0.0 if `priced_move_pct` is null or if no lookup table entry
matches.

### 3.2 Conditional Misprice Score (C)

**Purpose:** Detect mispriced conditionals — whether the scenario-weighted expected move
diverges from the implied move.

**Formula:**
```
C = tanh((conditional_EV - priced_move_pct) / (priced_move_pct + ε) / 2)
```

**Conditional move table** (scenario-weighted expected absolute move):

| Family|Phase | Formula | Expected Move |
|---|---|---|
| CLINICAL\|phase3 | 0.58×20 + 0.35×50 + 0.07×2 | 29.2% |
| CLINICAL\|phase2 | 0.31×30 + 0.58×40 + 0.11×2 | 32.7% |
| CLINICAL\|early | 0.63×25 + 0.30×25 + 0.07×2 | 23.4% |
| REGULATORY\|phase3 | 0.58×8 + 0.35×30 + 0.07×0 | 15.1% |
| REGULATORY\|phase2 | 0.31×12 + 0.58×35 + 0.11×0 | 24.0% |
| REGULATORY\|early | 0.63×15 + 0.30×25 + 0.07×0 | 17.0% |

Scenario weights derive from outcome model Wong et al. priors (p_hit, p_miss, p_mixed)
crossed with payoff engine per-scenario move magnitudes.

**Interpretation:** Positive C = scenario EV exceeds the market-implied move (appears
underpriced). Negative C = scenario EV is below implied (appears overpriced). As a trap
component, high C (apparently underpriced) is penalized — the "obvious bargains" signal
is adversely selected.

**Missing data:** Returns 0.0 if `priced_move_pct` is null, near-zero, or if no
conditional table entry matches.

### 3.3 Slippage Penalty Score (S) — DEPRECATED

**Status:** DEPRECATED as of PIT audit 2026-04-12. Always returns 0.0.

**Reason for deprecation:** The `market_cap_mm` field in PIT archives contained
current (2026-04) prices, not historical values. The slippage penalty's IC went from
+0.106 to -0.088 under PIT-safe data, revealing that the signal was entirely driven by
look-ahead bias (micro-caps outperform, and today's market cap correlates with past
returns).

**Interface retained** for backwards compatibility. The function signature persists but
the implementation is a stub.

### 3.4 Divergence Score (D) — INACTIVE

**Purpose:** Detect option surface vs stock-behavior divergence.

**Formula:**
```
D = tanh((priced_move_pct - implied_event_move) / (implied_event_move + ε) / 2)
```

Uses `implied_event_move` from the `pos_divergence` module as a realized-vol proxy.

**Status:** Retained in the model but empirically inactive — IC approximately 0 across
all horizons. Not used in either v2 overlay. May be revisited if the `implied_event_move`
data source improves.

### 3.5 Crowding Bias Score (F) — INACTIVE

**Purpose:** Detect favorite bias via short-interest positioning.

**Formula:**
```
F = tanh((short_interest_pct - SI_P50) / (SI_P90 - SI_P50 + ε) / 2)
```

SI_P50 and SI_P90 are computed cross-sectionally per batch (not from a static table).

**Status:** Retained but empirically inactive — IC approximately 0. Not used in either
v2 overlay. Cross-sectional anchoring is methodologically sound; the signal simply does
not predict forward returns in this universe.

### 3.6 Timing Decay Risk Score (T)

**Purpose:** Penalize expensive setups with uncertain catalyst timing. This is the sole
surviving quality signal.

**Formula:**
```
T = clamp(priced_move_pct × timing_factor / 20.0, 0.0, 1.0)
```

**Timing factors** (from `clinical_days_precision`):

| Precision | Factor | Interpretation |
|---|---|---|
| DAY | 0.0 | Exact date known — zero timing risk |
| WEEK | 0.25 | Low uncertainty |
| MONTH | 0.60 | Moderate uncertainty |
| QUARTER | 1.0 | High uncertainty |
| HALF_YEAR | 1.0 | High uncertainty |
| YEAR | 1.0 | High uncertainty |
| UNKNOWN | 1.0 | Assumed worst case |

**Interpretation:** High T means the position is expensive (large implied move) and the
catalyst timing is uncertain. The quality overlay inverts this: `quality = -(1.0 × T)`,
so names with high timing risk get low quality scores.

**Range:** [0.0, 1.0] (hard-clamped, not soft-scaled).

**Missing data:** Returns 0.0 if `priced_move_pct` is null.

---

## 4. Composite Scores

### 4.1 v1 Composite — DEPRECATED

```
ees_v1 = 0.30×B + 0.20×C + 0.10×D + 0.15×F - 0.00×S - 0.10×T
```

Retained in the `expectation_error_score` CSV column and `ExpectationErrorScore.expectation_error_score`
field for backwards compatibility. Slippage weight was zeroed post-PIT-audit. The v1
composite was mis-specified: its alpha-side sub-scores (B, C, D, F) are anti-predictive
when used as positive signals. **Do not cite v1 for any purpose.**

### 4.2 v2 Overlays (PIT-Safe, 2026-04-12)

The v2 model decomposes into two independent, orthogonal overlays:

#### Quality Overlay

```python
quality_overlay_score = -(1.0 × timing_decay_risk_score)
```

- **Detects:** Poor structure (expensive implied move + uncertain timing)
- **IC:** +0.078 at 63d, t-stat = 18.1 (PIT-safe)
- **Interpretation:** Higher = safer. Names with certain timing and moderate implied
  moves score highest.
- **Range:** [-1.0, 0.0]

#### Trap Overlay

```python
trap_overlay_score = -(0.50 × base_rate_gap_score + 0.50 × conditional_misprice_score)
```

- **Detects:** Fake edge (names that appear cheap by base-rate and conditional analysis
  but underperform)
- **IC:** +0.077 at 63d, t-stat = 18.5 (PIT-safe)
- **Interpretation:** Higher = safer. Names where the market's pricing is consistent with
  (or more conservative than) base rates score highest.
- **Range:** approximately [-1.0, +1.0]

#### v2 Composite

```python
ees_v2_score = 0.50 × quality_overlay_score + 0.50 × trap_overlay_score
```

Both overlays have approximately equal IC and are both PIT-safe, justifying equal
weighting. The overlays are orthogonal (correlation = 0.179 at launch).

### 4.3 Confidence Score

```python
confidence = 1.0
if priced_move_pct is None:     confidence *= 0.75
if short_interest_pct is None:  confidence *= 0.80
if market_cap_mm is None:       confidence *= 0.80
confidence *= (1.0 - 0.20 × timing_factor)
confidence = clamp(confidence, 0.0, 1.0)
```

Confidence measures input completeness and timing certainty. It does not enter the v2
overlays but is available for downstream filtering.

---

## 5. Gate System

### 5.1 Gate Mechanics

Gates are percentile-based filters computed cross-sectionally per snapshot:

```python
q_threshold = sorted_quality_scores[int(n × quality_cut_pct / 100)]
t_threshold = sorted_trap_scores[int(n × trap_cut_pct / 100)]

ees_quality_gate = quality_overlay_score > q_threshold
ees_trap_gate    = trap_overlay_score > t_threshold
ees_eligible     = ees_quality_gate AND ees_trap_gate
```

### 5.2 Gate Modes

| Mode | Quality Cutoff | Trap Cutoff | Use Case |
|------|---|---|---|
| **normal** | 0% (OFF) | 20% | Default — trap gate only |
| **conservative** | 15% | 20% | Both gates active |

The trap gate is the primary filter. The quality gate is off by default because timing
discipline is less universally applicable — some strategies intentionally hold through
timing uncertainty.

### 5.3 Degeneracy Protection

When a gate distribution cannot differentiate names, the gate is automatically bypassed:

1. **All identical:** If `min == max` for a gate's scores, all names pass
2. **Mode-dominated:** If >80% of values are identical (single mode dominates), all pass

This prevents false filtering when input data is sparse or missing (e.g., if no names
have `priced_move_pct`, all trap scores are 0.0, and the gate bypasses).

### 5.4 Threshold Sweep Evidence

A grid search across 81 combinations (quality cuts 0-40% × trap cuts 0-40%) on 267 dates
identified the following optimal operating points (20d horizon, Top-30):

| Operating Point | Sharpe | Mean Return | Hit Rate | Frequency |
|---|---|---|---|---|
| Baseline (no gates) | 0.211 | +2.10% | 59% | 100% |
| Default (T20) | 0.221 | +2.34% | ~59% | ~90% |
| Conservative (Q15/T20) | 0.414 | +4.04% | 68% | 63% |
| Optimal (Q15/T20, sweep) | 0.508 | +4.93% | 68% | 63% |

The conservative operating point achieves a 2.4× Sharpe improvement over baseline at the
cost of sitting out ~37% of dates.

---

## 6. Outputs

### 6.1 CSV Columns (15 columns injected into rankings.csv)

| Column | Type | Range | Description |
|---|---|---|---|
| `base_rate_gap_score` | float | [-1, +1] | Implied vs historical base rate |
| `conditional_misprice_score` | float | [-1, +1] | Scenario EV vs implied |
| `slippage_penalty_score` | float | 0.0 | DEPRECATED — always zero |
| `divergence_score` | float | [-1, +1] | Option surface vs stock divergence |
| `crowding_bias_score` | float | [-1, +1] | Short-interest positioning |
| `timing_decay_risk_score` | float | [0, 1] | Timing uncertainty × cost |
| `expectation_error_score` | float | ~[-1, +1] | v1 composite (DEPRECATED) |
| `expectation_confidence` | float | [0, 1] | Input completeness + timing certainty |
| `expectation_notes` | str | — | Human-readable diagnostic flags |
| `quality_overlay_score` | float | [-1, 0] | v2 quality: timing discipline |
| `trap_overlay_score` | float | ~[-1, +1] | v2 trap: avoid fake edge |
| `ees_v2_score` | float | ~[-1, +0.5] | v2 composite (quality + trap) |
| `ees_quality_gate` | bool | — | Passes quality percentile gate |
| `ees_trap_gate` | bool | — | Passes trap percentile gate |
| `ees_eligible` | bool | — | Both gates pass |

### 6.2 Sidecar Artifacts (per snapshot)

- **`ees_gate_diagnostics.json`** — Universe size before/after each gate, quality-trap
  correlation, distribution stats (min/p10/p25/median/p75/p90/max) for both overlays,
  filtered-out ticker lists
- **`ees_gate_performance.json`** — Realized returns by gate bucket (eligible vs
  quality-fail vs trap-fail), mean return and hit rate per bucket

### 6.3 Production Integration

- **Wired in:** `run_screen.py` Step 5k.21 (pre-write enrichment via `enrich_csv_rows`)
- **Shadow monitor:** `scripts/research/ees_v2_phase3_shadow_monitor.py` (daily)
- **Gate diagnostics:** Post-write via `build_gate_diagnostics()` and
  `build_gate_performance()` in `run_screen.py`

---

## 7. Input Features

All inputs are PIT-safe fields already present in `rankings.csv`. EES introduces no new
data dependencies.

| Feature | Source | Used In | Coverage Req (Spec 105) |
|---|---|---|---|
| `priced_move_pct` | Options pricing | B, C, D, T | >= 0.80 |
| `short_interest_pct` | Lending data | F (cross-sectional) | >= 0.90 |
| `market_cap_mm` | Market data | S (DEPRECATED) | >= 0.95 |
| `close_price` | Market data | S (DEPRECATED) | >= 0.99 |
| `implied_event_move` | pos_divergence module | D | Not gated |
| `catalyst_family` | Catalyst graph | B, C (lookup key) | — |
| `lead_program_phase` | Clinical data | B, C (phase bucket) | — |
| `clinical_days_precision` | Timing data | T (timing factor) | — |

---

## 8. Research Harnesses

Five research scripts validate EES from different angles:

### 8.1 Backtest IC (`scripts/research/backtest_ees.py`)

Walk-forward Spearman IC and decile spread for 8 signals (6 sub-scores + composite +
confidence) against 5d/20d/63d forward returns across 421 snapshots. Outputs:
`summary.json`, `by_date_signal.csv`, `summary.md`.

### 8.2 Ablation Study (`scripts/research/ablation_ees.py`)

Five-arm ablation decomposing v2 into quality-only, trap-only, original alpha (positive
direction), combined, and v1 composite. This study is what revealed the inversion: alpha-
side sub-scores are anti-predictive, and the correct model separates quality (timing) from
trap (fake cheapness).

### 8.3 Filter Test (`scripts/research/ees_filter_test.py`)

Six-arm comparison of portfolio construction methods: baseline (no filter), v2 linear
blend, quality filter, quality+trap filter at 20% and 30%, and quality filter + trap
rank. Demonstrated that hard filtering outperforms linear blending.

### 8.4 Threshold Sweep (`scripts/research/ees_threshold_sweep.py`)

81-cell grid search (quality 0-40% × trap 0-40%) producing Sharpe, mean return, and
frequency heatmaps. Identified Q15/T20 as the optimal operating point on the
Sharpe-vs-frequency efficient frontier.

### 8.5 Interaction Test (`scripts/research/ees_interaction_test.py`)

Tests whether trap is additive to quality or redundant: conditional IC of trap within
top-quality quintile, conditional IC of quality within non-bottom-trap quintile, 2D
quintile grid of mean forward returns, and horizon splits per overlay. Confirmed
orthogonality and additive value.

---

## 9. Validation History

### 9.1 PIT Audit (2026-04-12) — The Defining Event

The PIT audit was the model's most important validation. It destroyed the slippage
penalty, killed the v1 composite, and reshaped the architecture:

| Component | Pre-Audit IC | Post-Audit IC | Verdict |
|---|---|---|---|
| Quality (w/ slippage) | +0.107, t=18.5 | DEAD | Look-ahead contaminated |
| Slippage alone | +0.106 | -0.088 | **Reversed** under PIT data |
| Timing decay alone | +0.078, t=18.1 | +0.078, t=18.1 | PIT-SAFE |
| Trap overlay | +0.077, t=18.5 | +0.077, t=18.5 | PIT-SAFE |

**Root cause:** `market_cap_mm` in the PIT archives contained current (2026-04) prices,
not historical values. The slippage penalty tracked future market-cap growth, creating a
perfect look-ahead signal.

**Outcome:** Slippage deprecated (always returns 0.0). Quality overlay rebuilt as timing-
only. v2 composite recalibrated. All pre-audit evidence for slippage retracted.

This is the same contamination vector later found system-wide in the 2026-11-13 market-
cap look-ahead deprecation (see `docs/governance/2026-11-13-market-cap-lookahead-deprecation.md`).
The EES audit was an early warning that was not generalized quickly enough.

### 9.2 Shadow Monitor (2026-06-23 — present)

EES shadow monitor chain is complete and runs daily:
```
scripts/research/ees_v2_phase3_shadow_monitor.py --as-of-date YYYY-MM-DD
```

**Gates unmet as of 2026-11-15:** Needs 20 completed 5d exits + 20 completed 20d exits
before any reading is authoritative. Observation-only, non-blocking.

---

## 10. Evidence Assessment (As of 2026-11-15)

### 10.1 What Exists

| Evidence Type | Status | Notes |
|---|---|---|
| PIT-safe IC (timing) | +0.078, t=18.1 | 421 snapshots, post-2022-03 |
| PIT-safe IC (trap) | +0.077, t=18.5 | 421 snapshots, post-2022-03 |
| Ablation (5 arms) | Complete | Quality/trap separation validated |
| Threshold sweep | Complete | Q15/T20 optimal operating point |
| Interaction test | Complete | Overlays confirmed orthogonal |
| Unit tests | 125 passing | Comprehensive coverage |
| Daily shadow monitor | Running | Gates unmet |

### 10.2 What Is Missing

| Evidence Type | Status | Required For |
|---|---|---|
| Checklist v2 battery | NOT RUN | Promotion to decision path |
| Bootstrap CI on gate-filtered returns | NOT RUN | Confidence interval on Sharpe claim |
| LOSO (Leave-One-Subperiod-Out) | NOT RUN | Temporal stability |
| FDR correction | NOT RUN | Multiple-testing adjustment (6 sub-scores) |
| FM regression | NOT RUN | Controlling for existing B6 features |
| Forward shadow with completed exits | ACCUMULATING | 20×5d + 20×20d gate |
| Empirical base-rate validation | NOT DONE | Tables are literature-sourced, not calibrated to this universe |

### 10.3 Rating

**Overall: 3/5 under Checklist v2 framework — PROMISING, NOT PRODUCTION-GRADE.**

| Dimension | Score | |
|---|---|---|
| PIT safety | 4/5 | Caught contamination; clean components survive |
| Statistical robustness | 2/5 | IC-only; no bootstrap/LOSO/FDR/FM |
| Economic magnitude | 3/5 | Conservative mode strong but low frequency |
| Architecture quality | 3/5 | Clean after pruning; 4/6 sub-scores dead |
| Operational readiness | 4/5 | Well-tested, well-wired, proper isolation |
| Forward evidence | 0/5 | Shadow running but gates unmet |

### 10.4 Trust Bucket

**MONITORING / CONDITIONAL USE.** Safe as a diagnostic overlay. Not ready for
selector/ranker/construction integration without Checklist v2 and forward evidence.

---

## 11. Generalizable Concept: Asymmetric Return Screening

The EES framework encodes a broader insight applicable to any public equity screen seeking
asymmetric return opportunities. The core principle: **filtering out structurally
disadvantaged positions is more valuable than filtering in apparently advantaged ones.**

### 11.1 The Trap Inversion

Traditional screens look for undervalued names (low P/E, high FCF yield, wide discounts
to base rates). EES found that in event-driven biotech, names that score highest on naive
valuation metrics — those with the largest apparent mispricings — underperform. The
pattern generalizes:

- **Value traps in broader equity:** Low P/E stocks in secular decline, "cheap" cyclicals
  at peak earnings, beaten-down names with deteriorating fundamentals
- **Event traps in M&A:** Wide spreads on deals with known regulatory risk, where the
  spread exists precisely because informed capital has priced the downside
- **Momentum traps:** Late-stage momentum names where the crowd is fully positioned and
  any catalyst disappointment triggers crowded exits

The screen should not find the cheapest names. It should *exclude* names where cheapness
is consensus and the consensus is usually right to avoid them.

### 11.2 The Structure Filter

Timing decay risk generalizes to any strategy where the cost of being early matters:

- **Merger arb:** Uncertain regulatory timeline + high spread cost = negative carry
- **Earnings revisions:** Revision catalyst quarters away = implicit theta
- **Turnarounds:** No concrete timeline + management uncertainty = open-ended holding cost

The general formula: `structure_penalty = cost_of_carry × timing_uncertainty`. High
product = bad structure, regardless of thesis quality.

### 11.3 The Framework Applied

For any equity universe seeking asymmetric returns:

1. **Classify** each name into a situation type (sector × catalyst × regime)
2. **Compute base rates** for that type (historical return distribution)
3. **Extract market-implied expectations** (options, relative valuation, consensus)
4. **Score the gap** — then invert: large apparent mispricings are warnings, not
   opportunities
5. **Score the structure** — cost of carry × timing uncertainty
6. **Gate, don't rank** — exclude the bottom quintile on each dimension, then let the
   primary alpha model rank survivors

The intellectual discipline: accept that the alpha model cannot distinguish real
opportunities from traps, but a dedicated trap detector can tell you which "opportunities"
to skip.

---

## 12. Promotion Path

EES promotion to the decision path requires all of:

1. **Checklist v2 battery** on gate-filtered portfolio (FM, bootstrap CI, FDR, LOSO)
2. **Forward shadow** with 20 completed 5d + 20 completed 20d exits showing positive
   gate separation
3. **Empirical base-rate validation** — confirm literature-sourced tables against this
   universe's realized moves
4. **Promotion spec** (separate from this document) with operator approval
5. **Tier 4 governance review** (architecture change, signal admission)
6. **Clock reset** under the forward validation mandate (SM-20260629-001)

Until these gates are met, EES remains overlay-only.

---

## 13. Test Coverage

**File:** `tests/test_expectation_error_model.py` — 125 tests

| Test Class | Coverage |
|---|---|
| TestHelpers | clamp, safe_float, phase_bucket |
| TestBaseRateGap | Above/below/at base rate, missing data |
| TestConditionalMisprice | Underpriced/overpriced scenarios |
| TestSlippagePenalty | Deprecated (always 0) |
| TestDivergence | Options rich/cheap |
| TestCrowdingBias | High/low SI with cross-sectional percentiles |
| TestTimingDecayRisk | Exact/uncertain timing |
| TestCompositeScore | Positive/negative EES scenarios |
| TestV2Overlays | Quality, trap, combined v2 |
| TestConfidence | Missing data penalties |
| TestBatchScoring | Cross-sectional crowding, order preservation |
| TestNotes | Diagnostic flag generation |
| TestEnrichCsvRows | Column injection roundtrip |
| TestSerialisation | to_dict() roundtrip |
| TestGateComputation | Percentile-based filtering |
| TestGateDiagnostics | Universe sizing, correlation |
| TestGatePerformance | Realized returns by gate bucket |
| TestRegimeDetector | Normal/conservative mode triggers |
| TestRegimeToggle | Gate mode resolution |

---

## 14. Known Limitations

1. **Base-rate tables are hardcoded from literature**, not empirically calibrated to this
   universe's realized moves. If the biotech event landscape shifts (e.g., FDA RTCT
   initiative changes trial dynamics), the tables will drift.

2. **Model validity starts post-2022-03** (when options data exists). The 421-snapshot
   backtest covers ~48 months, shorter than the B6 selector's validation window.

3. **4 of 6 sub-scores are dead.** The surviving model is effectively a 2-factor system
   (timing + base-rate/conditional). The 6-mistake framework is intellectually useful but
   over-engineered for the empirical reality.

4. **IC t-stats are inflated.** t=18+ across 421 snapshots reflects many non-independent
   observations. The effective degrees of freedom are much lower.

5. **Conservative mode frequency is low.** At the optimal operating point (Q15/T20), the
   screen trades only 63% of dates — sitting out a third of opportunities.

6. **No forward shadow with completed exits.** Shadow monitor is running but the 20×5d
   + 20×20d gates have not been met. All evidence is in-sample.

7. **The 2026-11-13 market-cap look-ahead disclosure** affects the broader credibility
   environment. EES's PIT-safe components do not use market cap, but the discovery that
   contamination was systemic raises the bar for all claims.

---

## Appendix A: Data Contract

```python
@dataclass(frozen=True)
class ExpectationErrorScore:
    ticker: str
    as_of_date: str

    # Six sub-scores
    base_rate_gap_score: float
    conditional_misprice_score: float
    slippage_penalty_score: float      # DEPRECATED: always 0.0
    divergence_score: float
    crowding_bias_score: float
    timing_decay_risk_score: float

    # v1 composite (DEPRECATED)
    expectation_error_score: float
    expectation_confidence: float      # [0, 1]
    expectation_notes: str

    # v2 overlays (PIT-safe)
    quality_overlay_score: float
    trap_overlay_score: float
    ees_v2_score: float

    # Audit trail
    features_used: Dict[str, Any]
    model_version: str = "ees_v2.0"
```

## Appendix B: Regime Toggle

The `suggest_gate_mode()` function (if implemented) would switch between normal and
conservative based on:

**Input-side triggers (structural):**
- correlation(quality, trap) > 0.40 → CONSERVATIVE
- eligible % < 50% → CONSERVATIVE
- trap fail rate > 35% → CONSERVATIVE

**Output-side triggers (outcome-based):**
- trap_fail mean return > -3% (gate not working) → CONSERVATIVE
- eligible-vs-excluded gap < 1% (gates not separating) → CONSERVATIVE

## Appendix C: File Manifest

| File | Purpose |
|---|---|
| `event_ev/expectation_error_model.py` | Model implementation (669 lines) |
| `event_ev/data_contracts.py` (lines 437-490) | ExpectationErrorScore dataclass |
| `tests/test_expectation_error_model.py` | 125 unit tests |
| `run_screen.py` (Step 5k.21) | Production integration |
| `scripts/research/backtest_ees.py` | Walk-forward IC harness |
| `scripts/research/ablation_ees.py` | 5-arm ablation study |
| `scripts/research/ees_filter_test.py` | 6-arm filter comparison |
| `scripts/research/ees_threshold_sweep.py` | 81-cell grid search |
| `scripts/research/ees_interaction_test.py` | Orthogonality analysis |
| `scripts/research/ees_v2_phase3_shadow_monitor.py` | Daily shadow monitor |
| `specs/changes/spec_057_event_ev_engine.md` | Parent spec (Layer 4) |
| `specs/changes/spec_060_daily_event_ev_scoring.md` | Daily production wiring |
| `specs/changes/spec_105_expectation_layer_coverage_verification.md` | Coverage gates |
