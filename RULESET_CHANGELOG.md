# Decision Engine Ruleset Changelog

All material changes to decision engine logic, parameters, or governance.

Format: `[engine_version] ruleset_id — date — summary`

---

## [v1.3.0] 18d44abd — 2026-02-11 — Add cost-aware sizing (L3-only, opt-in)

**Parameters added** (vs 5a9faad9):
- `enable_cost_haircut`: False (opt-in flag, no behavior change when disabled)
- `cost_haircut_buckets`: ((50, 1.0), (100, 0.85), (150, 0.70))
- `cost_haircut_floor_mult`: 0.55

**Behavior changes**:
- **None when disabled** (default). All existing outputs are identical.
- When `enable_cost_haircut=True` and `est_cost_bps` is provided:
  - Trading cost is mapped to a multiplier via bucket thresholds (1.0/0.85/0.70/0.55)
  - Multiplier scales raw weight before normalization in `compute_target_weights()`
  - Heavy haircut (mult <= 0.70) also triggers a one-step band downgrade
  - New output fields: `cost_mult`, `cost_bucket`, `cost_haircut_applied`

**Rationale**: High-cost / illiquid names should carry less weight to reflect
implementation friction. Opt-in flag ensures zero impact until calibrated and
deliberately enabled. Affects L3 sizing only — no eligibility or tier changes.

**Governance**: 20 new unit tests (`test_cost_aware_sizing.py`), pinned regression
green (bands/weights/tickers unchanged), contract/replay tests green.

**Files**: `v1.2.1_candidate.json` (active, schema-expanded)

---

## [v1.3.0] c88bd4cc — 2026-02-10 — Regime-aware drawdown gate (XBI-relative AND logic)

**Parameters changed** (vs 181346fe):
- `drawdown_rel_xbi_gate`: -0.15 → -0.20
- Added `drawdown_gate_require_both`: True (AND logic — both absolute AND relative must breach)

**Behavior changes**:
- Drawdown gate now uses AND logic: a ticker is only failed for deep_drawdown if BOTH
  absolute drawdown < -0.40 AND relative drawdown (ticker - XBI) < -0.20.
- When relative drawdown data is unavailable, falls back to absolute-only (safe degradation).
- `drawdown_gate_require_both=False` restores original absolute-only behavior.
- New risk flag `deep_drawdown_rel_xbi` fires when relative drawdown < threshold.
- New snapshot/panel columns: `de_drawdown_xbi`, `de_drawdown_rel_xbi`, `drawdown_abs`,
  `drawdown_xbi`, `drawdown_rel_xbi`.

**Rationale**: In tape-down regimes (e.g. late-2025), absolute -0.40 gate collapses dev
eligibility because the entire sector is down. The AND gate distinguishes ticker-specific
weakness from market-wide drawdowns. Strictly looser than absolute-only — no currently-eligible
ticker loses eligibility.

**Files**: `v1.2.0_candidate.json` (active)

---

## [v1.2.0] d3cdf5c8 — 2025-10-31 — Phase-2 production default

**Parameters changed** (vs v1.0.0):
- `tier_a_optionality_floor`: 0.60 → 0.55
- Added `drawdown_gate_mode`, `drawdown_size_penalty`, `drawdown_hard_floor` (hard mode defaults)

**Rationale**: Calibration sweep (`run_decision_ruleset_sweep.py`, 54 combos, 22 archive
snapshots) showed a_floor=0.55 dominates: +12.70% spread vs top-K at 60d, +13.13%
strategy residual, 53.1% hit rate. Policy matrix confirmed A+B filter with K=20 as
optimal operating point.

**Governance**: Pinned regression test (`tests/test_phase2_portfolio_regression.py`,
2025-10-31 snapshot). Health gate calibrated against 2025 steady-state (WARN=20%, FAIL=0%).

**Files**: `v2_phase2_default.json` (active), `v1.2_candidate.json` (candidate copy)

---

## [v1.0.0] d4f1f8a8 — 2024-11-01 — Original defaults

Initial decision engine release. Hard-coded thresholds:
- `tier_a_optionality_floor`: 0.60
- `drawdown_gate`: -0.40 (hard mode only)
- `catalyst_near_days`: 90
- `sponsor_confirm_threshold`: 2

**Status**: Retired. Superseded by v1.2.0 after calibration showed a_floor=0.60
produces negative tier-A vs tier-C spread.

**Files**: `v1.json`

---

## [v1.3.0] f6c99132 — 2026-02-10 — Catalyst strength bands

**Parameters changed** (vs eb833c56):
- Added `catalyst_mid_days`: 180 (new parameter — upper bound of MID strength band)

**Behavior changes**:
- Catalyst strength computed as NEAR (≤ near_days) / MID (near_days < d ≤ mid_days) / FAR (> mid_days) / MISSING
- A-tier now requires actionable catalyst: strength ∈ {near, mid}. FAR treated same as MISSING for tier gating.
- FAR catalysts get +1 sizing band step ("catalyst_far_lift")
- Sort key includes catalyst_strength between catalyst_mode and catalyst_days
- `catalyst_strength` added to DECISION_COLUMNS, SNAPSHOT_COLUMNS, PANEL_COLUMNS

**Rationale**: 365d trial window experiment proved wider coverage dilutes tier separation
(+0.97pp→+0.30pp) because far-out catalysts flood A/B tiers. Strength bands preserve
wider data capture without polluting tier signal.

**catalyst_mid_days validation** (2D sweep: 7 a_floor × 7 mid_days, 2025-only, 1808 rows):
Ridge monotonically favors tighter boundaries (150→+2.78pp, 180→+2.53pp, 210→+2.21pp,
270→+1.03pp). Optimizer picks mid=150, but A-count=1.7 (below health gate warn=2).
Decision: keep mid=180 (A-count=2.3, +2.53pp separation, 5/5 neighbor stability).
Catalyst strength signal: near median +5.56% / mid +4.50% >> far -0.15% / missing -0.63%.

**Files**: `v1.2.0_candidate.json` (updated with `catalyst_mid_days: 180`)

---

## [v1.2.0] eb833c56 — 2026-02-10 — 2D calibration (a_floor + catalyst timing)

**Parameters changed** (vs d3cdf5c8):
- `tier_a_optionality_floor`: 0.55 → 0.60
- `catalyst_near_days`: 90 → 120

**Rationale**: 2D calibration sweep (`calibrate_ruleset_from_panel.py`, 40 combos = 8 a_floor
× 5 catalyst_near_days, 2025-only panel, 1808 rows, 10 snapshots) found current baseline
a_floor=0.55 / catalyst_near=90 has *negative* AB-CD separation (-0.57pp). Best candidate
a_floor=0.60 / catalyst_near=120 achieves +0.97pp separation, +1.41pp AB return improvement,
lower turnover (13.4% vs 15.7%), and higher top-25 overlap (86.6% vs 84.3%). Neighbor stability
STRONG (5/5 adjacent grid cells pass constraints). Ridge summary confirms catalyst_near=120
dominates across all a_floor values.

**Governance**: Contract tests (257 passed), replay regression (3 archive dates), release
summary (`artifacts/release_summary_2d_2025.md`). Calibration artifacts:
`artifacts/calibration_report_2d_2025.md`, `artifacts/walkforward_report_2025.md`.

**Files**: `v1.2.0_candidate.json` (candidate)

---

## [v1.3.0] 5a9faad9 — 2026-02-11 — Loosen relative drawdown gate (AND)

**Parameters changed** (vs c88bd4cc):
- `drawdown_rel_xbi_gate`: -0.20 -> -0.25

**Behavior changes**:
- Under AND logic (require_both=True), both absolute (-0.40) AND relative must breach to fail.
  Loosening the relative gate from -0.20 to -0.25 means fewer tickers hit the combined gate.
- 38 previously ineligible names become rescued (abs breaches, rel passes at -0.25).
- All 11 currently rescued names remain rescued (their rel margins are already positive).

**Rationale**: Rescued-vs-clean walkforward analysis (10 snapshots, 2025-01-31 to 2025-10-31,
ruleset c88bd4cc, 1808 panel rows) showed:
- Current rescued (N=11): mean +13.95%, hit 63.6%, max-DD -20.62% (comparable to clean)
- Newly rescued at -0.25 (N=38): mean +33.48%, hit 63.2%, max-DD -21.90% (strong performers)
- Combined rescued at -0.25 (N=49): mean +29.10%, better max-DD than clean (-21.6% vs -24.1%)
- DD rel near-gate pressure: 34.5% of eligible dev tickers within 5pp — binding constraint
- Calibration sweep: separation +4.54pp at -0.25 vs +2.27pp at -0.20; STRONG neighbor stability

**Governance**: Walkforward panel (`artifacts/walkforward_panel_rescued.csv`, 1808 rows),
rescued outcome analysis, calibration artifacts (`artifacts/calibration_report__2025_ddrel_AND.*`).
Pinned regression test updated. Contract/replay tests green.

**Files**: `v1.2.1_candidate.json` (candidate)
