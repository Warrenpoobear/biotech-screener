# Decision Engine v1 Framework

> Replaces scalar "Module 5 composite rank" with an interpretable decision
> artifact: **eligibility gates + dev-tiering + overlays + sizing guidance**.
> Implemented in `decision_engine.py`, integrated via `run_screen.py`.

---

## 1. Why It Exists

The Module 5 composite blender produces a single scalar rank per ticker.
Empirically, this rank is **anti-predictive after XBI beta removal** (negative
residual IC at 60d, t=-1.3). Meanwhile, `clinical_optionality_pct_dev` — an
inverted clinical score within the dev cohort — is a **validated dev-stage
signal** (t=+2.12 at 60d).

The decision engine replaces "look at rank #1" with a structured output:

- **Eligibility** — hard exclusions that protect capital
- **Dev-tiering** (A/B/C/D) — aligns with the validated optionality signal
- **Overlays** — sponsorship, catalyst proximity, momentum, risk flags
- **Sizing guidance** (XS/S/M/L) — rule-based with auditable reasons
- **Backtestable evaluation** — group return tables, raw + XBI-residualized

Module 5 composite rank is **preserved** alongside the decision columns.
Nothing upstream is deleted.

---

## 2. Where It Lives in the Pipeline

```
Modules 1-5 (unchanged)
    │
    ▼
save_validation_snapshot()          ← run_screen.py
    │
    ├─ Build csv_rows from ranked_securities
    ├─ Compute clinical_optionality_pct_dev (inverted percentile within dev cohort)
    ├─ *** Decision Engine: compute_decision_fields() per ticker ***
    │       Uses rec_by_ticker lookup (NOT index-based)
    └─ Write rankings.csv with all columns
```

- **File**: `decision_engine.py` (~270 lines, pure functions, no side effects)
- **Integration**: called from `run_screen.py` → `save_validation_snapshot()`
- **No upstream changes**: Modules 1-5 are untouched

---

## 3. Inputs (per ticker)

| Input | Source | Type |
|-------|--------|------|
| `rec` | `ranked_securities[i]` via `rec_by_ticker[ticker]` | dict |
| `archetype` | `results["company_archetypes"][ticker]` | str |
| `clinical_optionality_pct_dev` | Already computed in csv_row | float or None |

---

## 4. Outputs (new columns in rankings.csv)

### Meta — Versioning

| Column | Values | Notes |
|--------|--------|-------|
| `decision_engine_version` | `"v1.0.0"` | Bump on any logic change |
| `decision_engine_ruleset_id` | `"36c17dca"` (sha256[:8] of all params) | Auto-recomputes on any threshold change |

### Layer 0 — Eligibility

| Column | Values | Notes |
|--------|--------|-------|
| `eligible` | `"1"` / `"0"` | Hard gates only |
| `ineligible_reasons` | pipe-separated | e.g. `fundamental_red_flag\|deep_drawdown` |

### Layer 2 — Overlays

| Column | Source | Values |
|--------|--------|--------|
| `sponsor_tier1_count` | `rec["coinvest"]["tier1_count"]` | int or blank |
| `sponsor_overlap_count` | `rec["smart_money_signal"]["overlap_count"]` | int or blank |
| `sponsor_net_buying` | `holders_increasing` vs `holders_decreasing` (list lengths) | `buying` / `selling` / `neutral` / blank |
| `catalyst_days` | `rec["catalyst_decay"]["days_to_catalyst"]` | int or blank |
| `catalyst_in_window` | `rec["catalyst_decay"]["in_optimal_window"]` | `"1"` / `"0"` / blank |
| `catalyst_mode` | derived from `catalyst_days` + `catalyst_in_window` | `specific_days` / `blended_window` / `no_upcoming` / `missing` |
| `runway_bucket` | severity + `fundamental_red_flag_reasons` | `adequate` / `short` / `critical` / blank |
| `mom_state` | `score_breakdown.enhancements.momentum.alpha_60d` (fallback: `rec["momentum_signal"]`) | `tailwind` / `headwind` / `neutral` |
| `risk_flags` | `rec["defensive_features"]` + `confidence_overall` | pipe-separated |

### Layer 3 — Sizing

| Column | Values | Notes |
|--------|--------|-------|
| `size_band` | `XS` / `S` / `M` / `L` | Rule-based with clamp |
| `size_reasons` | pipe-separated | e.g. `tier_a_dev\|sponsor_confirmed\|momentum_tailwind` |

### Layer 4 — Dev Tiering

| Column | Values | Notes |
|--------|--------|-------|
| `tier_dev` | `A` / `B` / `C` / `D` / blank | drug_developer only; blank for non-dev |
| `tier_reason` | descriptive string | e.g. `high_opt+catalyst_near`, `ineligible` |

---

## 5. Core Semantics

### 5.1 Clinical Optionality Is the Dev-Stage Return Tilt

`clinical_optionality_pct_dev` is an **inverted** percentile: higher = more
optionality = empirically better forward performance for drug developers.

Do NOT assume "higher clinical quality = better returns" for dev-stage. The
empirical finding is the opposite (t=+2.27 at 60d when flipped). The composite
blender embeds the wrong direction.

| Run | IC_60d (t) | Resid_60d (t) | QS_60d |
|---|---|---|---|
| clinical (dev, higher-is-better) | -0.032 (-2.08) | -0.025 (-1.71) | -6.44% |
| clinical **FLIPPED** (dev) | **+0.036** (+2.27) | **+0.028** (+1.91) | **+6.97%** |
| clinical FLIPPED (all) | +0.009 (+0.40) | +0.000 (+0.02) | +2.30% |
| clinical (commercial) | -0.026 (-0.86) | -0.014 (-0.47) | -1.66% |

### 5.2 Hard Gates Are Intentionally Narrow

Hard gates: `fundamental_red_flag`, `SEV3`, drawdown < -40%, ADV/liquidity flags.

**Confidence is NOT a hard gate.** It is a risk flag only (`low_confidence` in
`risk_flags` when confidence_overall < 0.30). Rationale: sparse dev-stage data
coverage means a confidence gate would exclude the exact optionality names that
drive the validated signal.

### 5.3 Catalyst Proximity Has Two Modes

This is the most non-obvious behavior in the engine.

**Mode 1 — Specific days:**
`days_to_catalyst > 0`. The pipeline found an upcoming event with a real date.
"Near" = days <= 90.

**Mode 2 — Blended window:**
`days_to_catalyst == 0` AND `in_optimal_window == True`. The pipeline could not
pin a specific number of days, but its proximity scoring is active
(`blend_mode = "full_blend"`, `catalyst_proximity_blended = True`). This is NOT
"no catalyst" — it's a degraded but real signal indicating catalyst proximity
is active.

**No catalyst data:**
`days_to_catalyst` is missing/None AND `in_optimal_window` is missing/False.
Only this case means the pipeline has no catalyst information.

The tier logic routes each mode differently:

| Catalyst state | Tier A eligible? | `tier_reason` suffix |
|---|---|---|
| Specific days <= 90 | Yes | `catalyst_near` |
| Blended window (days=0, window=True) | Yes | `catalyst_window` |
| Specific days > 90 | No (caps at B) | `catalyst_far` |
| No data | No (caps at B) | `no_catalyst_data` |

On the 2026-02-07 production run: 119 dev tickers have specific_days mode,
64 have no_upcoming (pipeline ran but no actionable catalyst). Note: blended
window mode (days=0 + window=True) is possible but was not observed in this
snapshot.

### 5.4 Blank vs Zero

| Value in CSV | Meaning |
|---|---|
| blank (empty string) | Data not available — do not treat as zero |
| `0` | Data is present and the measured value is zero |

This applies to `sponsor_tier1_count`, `sponsor_overlap_count`,
`sponsor_net_buying`, `catalyst_days`, `catalyst_in_window`. Getting this wrong
permanently confuses "no coverage" with "no sponsors" in backtests.

**Concrete examples:**

```csv
# Ticker where catalyst fields are absent (no catalyst module output / no coverage):
ACME,sponsor_tier1_count=3,catalyst_days=,catalyst_in_window=,catalyst_mode=missing

# Ticker where pipeline ran catalyst analysis and found nothing actionable:
BCDE,sponsor_tier1_count=0,catalyst_days=,catalyst_in_window=0,catalyst_mode=no_upcoming

# Ticker in blended catalyst window (proximity active, no pinned date):
IRON,sponsor_tier1_count=2,catalyst_days=0,catalyst_in_window=1,catalyst_mode=blended_window
```

Rule of thumb: if the pipeline's sub-module executed and returned a result,
write the result (even if it is `0`). If the sub-module had no data to run on,
write blank.

### 5.5 Catalyst Mode (Derived Column)

`catalyst_mode` is a human-readable label derived from `catalyst_days` +
`catalyst_in_window`:

| `catalyst_mode` | Condition | Meaning |
|---|---|---|
| `specific_days` | `days > 0` | Real date found; `catalyst_days` = distance |
| `blended_window` | `days == 0` AND `window == True` | Proximity scoring active, no pinned date |
| `no_upcoming` | `window == False/0` AND `days` is blank/None/0 | Pipeline ran, no actionable catalyst |
| `missing` | Both fields absent | No catalyst data at all |

This column exists so downstream consumers never need to re-derive the
three-way catalyst semantics from the raw fields.

---

## 6. Tier Rules (dev only)

Only applies when `archetype == "drug_developer"`.

| Tier | Condition | `tier_reason` examples |
|------|-----------|----------------------|
| **D** | Ineligible (any hard gate failed) | `ineligible` |
| **A** | Eligible + optionality >= 0.60 + catalyst near/window | `high_opt+catalyst_near`, `high_opt+catalyst_window` |
| **B** | Eligible + high opt without near catalyst, OR moderate opt (>= 0.30) with near/window catalyst | `high_opt+catalyst_far`, `high_opt+no_catalyst_data`, `mod_opt+catalyst_near`, `mod_opt+catalyst_window` |
| **C** | Eligible but lower optionality or no catalyst confirmation | `low_opt`, `mod_opt+no_catalyst_data`, `low_opt+no_catalyst_data`, `no_optionality_data` |

Non-dev archetypes: `tier_dev` and `tier_reason` are blank.

**Tier distribution on 2026-02-07 (183 dev tickers):**

| Tier | Count | % |
|------|-------|---|
| A | 18 | 9.8% |
| B | 58 | 31.7% |
| C | 79 | 43.2% |
| D | 28 | 15.3% |

**Tier reason breakdown:**

| Reason | Count |
|--------|-------|
| `low_opt` | 46 |
| `ineligible` | 28 |
| `high_opt+catalyst_far` | 25 |
| `high_opt+no_catalyst_data` | 19 |
| `low_opt+no_catalyst_data` | 18 |
| `high_opt+catalyst_near` | 18 |
| `mod_opt+no_catalyst_data` | 15 |
| `mod_opt+catalyst_near` | 14 |

---

## 7. Sizing Rules

Start at **M** (index 2 in [XS, S, M, L]), then apply adjustments:

| Condition | Adjustment | Reason tag |
|-----------|-----------|------------|
| Tier A dev + optionality >= 0.60 | +1 | `tier_a_dev` |
| `sponsor_tier1_count` >= 2 | +1 | `sponsor_confirmed` |
| `mom_state` == tailwind | +1 | `momentum_tailwind` |
| `mom_state` == headwind | -1 | `momentum_headwind` |
| `runway_bucket` in (short, critical) | -1 | `runway_short` / `runway_critical` |
| `risk_flags` contains high_vol or high_beta | -1 | `high_risk` |

Clamp to [0, 3] -> map to [XS, S, M, L].
Ineligible tickers -> **XS** unconditionally.

**Sizing is intentionally orthogonal to tier.** A Tier C name can land in L if
sponsorship + momentum are strong, because sizing reflects risk/confirmation
signals independent of optionality. If this creates optics issues ("why is C
sized Large?"), add a tier cap (e.g., C max M, D forced XS) in a future
version — but only after group-by evaluation confirms the interaction.

**Calibration knob**: The `sponsor_confirmed` threshold (tier1_count >= 2) may
be too permissive given broad 13F coverage. On 2026-02-07, 103 of 124 L-band
tickers hit this condition. Adjust threshold based on `--group-by size_band`
evaluation results.

---

## 8. Backtesting & Validation Protocol

### IC Is Not the Right Metric for Tiers

Encoding A/B/C/D as 4/3/2/1 and computing Spearman IC treats tier as ordinal.
The real validation is **per-group return separation**.

### Group-by Evaluation Mode

```bash
python run_rank_ic_backtest.py \
    --group-by tier_dev \
    --subset dev \
    --output output/tier_eval.json
```

Output per group per horizon:

| Metric | Description |
|--------|-------------|
| Mean % | Arithmetic mean forward return |
| Median % | Median forward return |
| Winsor % | 5% winsorized mean (robustness check) |
| Hit % | Fraction with positive return |
| Resid % | XBI-residualized mean return |
| Res Hit % | Fraction with positive residual return |

**Hypothesis**: A > B > C on dev subset, especially on 60d residualized returns.

### Available Group-by Columns

`tier_dev`, `size_band`, `mom_state`, `eligible`, `tier_reason`,
`catalyst_mode`, `severity`, `archetype`

### Signal Evaluation (for IC computation)

New signal choices added to `--signal`:

| Signal | Encoding | Higher is better |
|--------|----------|-----------------|
| `eligible` | 1.0 / 0.0 | Yes |
| `tier_dev` | A=4, B=3, C=2, D=1 | Yes |
| `size_band` | L=4, M=3, S=2, XS=1 | Yes |

---

## 9. Implementation Invariants

### 9.1 Ticker-based Join (Never Index-based)

```python
rec_by_ticker = {rec["ticker"]: rec for rec in ranked}
for row in csv_rows:
    rec = rec_by_ticker.get(row["ticker"])
    ...
```

Eliminates a class of silent mis-labeling bugs from future sorting/filtering
changes.

### 9.2 Graceful Degradation

- Missing catalyst data: tier caps at B, reason includes `no_catalyst_data`
- Missing optionality: tier defaults to C, reason = `no_optionality_data`
- Missing sponsorship: fields are blank (not zero)
- Old archives without decision columns: `--group-by` shows `_missing` group
  without crashing

### 9.3 Data Source Mapping (Critical)

These are **different from** `score_breakdown.enhancements`:

| Signal | Read from | NOT from |
|--------|-----------|----------|
| Sponsorship | `rec["smart_money_signal"]` (tier1_holders, holders_inc/dec) + `rec["coinvest"]` (tier1_count) | `score_breakdown.enhancements.smart_money` (only has summary overlap_count + score) |
| Catalyst | `rec["catalyst_decay"]` (TOP-LEVEL key) | `score_breakdown.enhancements.catalyst_decay` (only has factor + in_optimal_window) |
| Momentum | `score_breakdown.enhancements.momentum.alpha_60d` -> fallback `rec["momentum_signal"]` | (momentum is actually correct in enhancements) |
| Defensive | `rec["defensive_features"]` | (no alternative path) |
| Red flags | `rec["fundamental_red_flag"]` + `rec["fundamental_red_flag_reasons"]` | (no alternative path) |

### 9.4 Change Control

Any change to thresholds, gate logic, or tier rules **must** bump `VERSION`
and will auto-update `RULESET_ID` (sha256 of all tunable parameters).
Backfilled archives retain the ruleset ID they were generated with, so
comparing `decision_engine_ruleset_id` across snapshots reveals exactly which
rule regime produced each row. Never silently change a threshold without a
version bump — if two archives share a ruleset ID, their decision fields must
be reproducible from the same code.

### 9.5 Golden Record Sanity Check

After any code change, run `tests/test_decision_engine_golden_records.py`
which uses synthetic rec fixtures covering all three catalyst modes. The table
below is anchored to the frozen 2026-02-07 archive artifact for reference, but
the **authoritative** golden records are the unit tests (they cover
`blended_window` even when production data doesn't).

| Ticker | archetype | optionality | catalyst_days | catalyst_in_window | catalyst_mode | tier_dev | tier_reason | size_band |
|--------|-----------|-------------|---------------|-------------------|---------------|----------|-------------|-----------|
| **IRON** | drug_developer | 0.654 | 87 | 0 | specific_days | A | high_opt+catalyst_near | M |
| **PCVX** | drug_developer | 0.758 | (blank) | 0 | no_upcoming | B | high_opt+no_catalyst_data | L |
| **TBPH** | drug_developer | 0.038 | (blank) | 0 | no_upcoming | C | low_opt+no_catalyst_data | L |

These three cover: catalyst near with high optionality (A path), no
actionable catalyst degrading high optionality to B, and low optionality
landing in C regardless of catalyst.

---

## 10. Next Steps

### Phase 1 — Observe Only (current)

Decision columns ship in every `rankings.csv` but are **not yet used** to
filter, sort, or size positions. The existing Module 5 composite rank remains
the primary output. Phase 1 goals:

1. **Backfill** decision columns into historical archives (33 archives,
   2024-01-31 to 2026-02-07). This enables tier evaluation on historical data.

2. Run tier group evaluation:
   ```bash
   python run_rank_ic_backtest.py --group-by tier_dev --subset dev
   ```
   Confirm A > B > C separation on both raw and residual 60d returns.

3. Run sizing evaluation:
   ```bash
   python run_rank_ic_backtest.py --group-by size_band
   ```

4. Accumulate 2-3 live snapshots with decision columns to verify stability.

**Exit criterion for Phase 1:** Tier A mean residual 60d return > Tier C mean
residual 60d return across backfilled archives, with reasonable monotonicity.

### Phase 2 — Actionable

Once Phase 1 confirms tier separation:

- Replace "sort by composite rank" with "filter by tier + sort within tier"
- Use `size_band` as position-sizing input
- Gate new positions on `eligible == 1`

### Calibration

Two knobs to tune using group-by tables:

| Knob | Current | Tuning signal |
|------|---------|---------------|
| `sponsor_confirmed` threshold | tier1_count >= 2 | If L-band doesn't outperform M-band, raise threshold |
| Optionality cutoffs | A >= 0.60, B >= 0.30 | Adjust if A/B separation is weak or A is too small/large |

### Future Layers

- **Conviction overlay**: thesis-gate + smart-money reinforcement interaction
- **Commercial-stage tiering**: separate tier system for commercial archetypes
  (clinical is noise for them)
- **Dynamic thresholds**: regime-conditional optionality cutoffs
- **Rebalance analysis**: monthly snapshots imply monthly rebalance; verify
  alignment with signal horizon (60d)
