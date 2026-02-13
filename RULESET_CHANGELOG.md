# Decision Engine Ruleset Changelog

All material changes to decision engine logic, parameters, or governance.

Format: `[engine_version] ruleset_id — date — summary`

---

## [v1.3.0] bf6815e2 + f3454ef7 — 2026-02-13 — Catalyst source/type ordering policy

**Parameters added** (schema expansion, all rulesets):
- `enable_catalyst_priority`: **False** (opt-in flag, no behavior change when disabled)
- `catalyst_priority_map`: Priority ladder for (event_type, source) pairs
- `catalyst_priority_default`: **9** (no-catalyst fallback)
- `catalyst_priority_unknown`: **99** (broken-data sink)

**New ruleset**: `v1.3.0_candidate.json` (ID: f3454ef7)
- Same parameters as v1.2.2 + `enable_catalyst_priority=true`
- Priority ladder: FDA=1, CTGOV/readout=2, corporate/ongoing=3, none=9, unknown=99

**Sort key change**: `compute_actionable_sort_key()` now returns 11-element tuple
(was 10). New element at position 3: `cat_priority`. When disabled, value=0 (neutral,
no effect). When enabled, lower priority values sort first within same tier.

**Call site wiring**: catalyst_event_type + catalyst_source now passed at all call sites:
- `run_screen.py`: added `_nearest_catalyst_event_type()`, `catalyst_event_type` in SNAPSHOT_COLUMNS
- `run_decision_strategy_backtest.py`, `run_phase2_health_calibration.py`

**Behavior changes**: None for v1.2.2 (enable_catalyst_priority=false). v1.3.0_candidate
requires backtest acceptance before promotion.

**Tests**: 32 in `test_decision_actionable_ordering.py` (+16 new: resolve_catalyst_priority
unit tests + sort key integration with priority enabled/disabled)

---

## [v1.3.0] bf6815e2 — 2026-02-13 — Version bump to v1.2.2_candidate (no logic change)

**Parameters changed**: None. Identical content-hash to v1.2.1_candidate.

**Rationale**: Pairs with drift guardrail calibration (commit 496b27b) which tightened
four WARN thresholds from the 2025 panel baseline:
- `warn_rs_morningstar_share_low`: 70% -> 85% (median=94.4%, IQR=4.5)
- `warn_rs_unknown_share_high`: 10% -> 8.5% (median=1.7%, IQR=3.4)
- `warn_cat_eligible_share_low`: 80% -> 95% (median=100%, IQR=0)
- `warn_cs_ctgov_share_low`: 50% -> 37% (2025 median=39.7%, IQR=1.2)

**Verification**: Panel diff between v1.2.1 and v1.2.2 is zero bytes.

**Files**: `v1.2.2_candidate.json` (active), `v1.2.1_candidate.json` (retired, preserved for audit)

---

## [v1.3.0] bf6815e2 — 2026-02-12 — Add momentum state weight tilt schema (defaults neutral)

**Parameters added** (vs b92f9338):
- `enable_mom_state_tilt`: **False** (opt-in flag, no behavior change when disabled)
- `mom_state_tilt_mults`: **tailwind=1.0, neutral=1.0, headwind=1.0** (neutral defaults)

**Behavior changes**: None. Tilt is off by default, and default mults are all 1.0.
All portfolio weights, tiers, eligibility, and membership are bit-for-bit identical
to b92f9338. The only change is the ruleset ID (new fields in canonical JSON).

**Rationale**: Infrastructure for momentum-state weight modifiers — L3-only,
membership-preserving. 2024 inversion attribution found `mom_state=neutral` as the
#1 driver of AB-D spread inversion. Sweep will test neutral penalty values
(0.80-0.95) on walkforward panel before enabling.

**Files**: all 6 production rulesets re-saved with new fields, manifest IDs updated.

---

## [v1.3.0] b92f9338 — 2026-02-12 — Add dd_rel_margin rescue schema (defaults off)

**Parameters added** (vs 68b2c45e):
- `enable_dd_rel_margin_rescue`: **False** (opt-in flag, no behavior change when disabled)
- `dd_rel_margin_rescue_threshold`: **-0.05** (relative drawdown margin threshold for rescue)

**Behavior changes**: None. Rescue is off by default. All portfolio weights, tiers,
eligibility, and membership are bit-for-bit identical to 68b2c45e. The only change
is the ruleset ID (new fields in canonical JSON).

**Rationale**: Infrastructure for conservative drawdown rescue — when both abs + rel
gates breach under AND mode, override if relative margin is close to the gate
(sector-driven drawdown, not idiosyncratic blow-up). Walkforward validation showed
-0.05 threshold is too aggressive for production: worsens 2025 AB-CD separation by
-2.05pp and increases turnover by +14.8pp. Flag remains OFF pending tighter threshold
calibration. Telemetry (`dd_rel_margin_rescued`) and drift monitoring
(`dd_rel_margin_rescue_share_pct`, WARN > 5%) are in place for when the flag is
eventually enabled.

**Governance**: 9 unit tests (`TestDdRelMarginRescue`), 78 drift report tests,
6538 total tests green. Walkforward validation memo:
`artifacts/dd_rel_margin_rescue_validation_2026-02-12.md`. All manifest IDs updated.
No replay required (identical behavior when disabled).

---

## [v1.3.0] 68b2c45e — 2026-02-12 — Add catalyst tilt schema (defaults off)

**Parameters added** (vs 131800e4):
- `enable_catalyst_tilt`: **False** (new field, disabled by default)
- `catalyst_tilt_mults`: **NEAR=1.10, MID=1.05, FAR=0.95, MISSING=0.90** (new field)

**Behavior changes**: None. Catalyst tilt is off by default. All portfolio
weights, bands, eligibility, and membership are bit-for-bit identical to
131800e4. The only change is the ruleset ID (new fields in canonical JSON).

**Rationale**: Infrastructure for opt-in weight tilt by catalyst proximity
band. Evidence from backtest: NEAR median +5.56% / MID +4.50% >> FAR -0.15%
/ MISSING -0.63% at 60d. Requires explicit `--enable-catalyst-tilt true` to
activate. Cost params also made explicit in JSON (previously implicit defaults)
and bumpable via CLI (`--cost-haircut-buckets`, `--catalyst-tilt-mults`).

**Governance**: 379 tests pass. Regression test re-pinned. All manifest IDs
updated. No replay required (identical behavior).

---

## [v1.3.0] 131800e4 — 2026-02-11 — Enable cost-aware sizing (cap=1000)

**Parameters changed** (vs 34bb662d):
- `enable_cost_haircut`: False → **True**
- `cost_impact_cap_bps`: 200.0 → **1000.0**

**Behavior changes**:
- Cost-aware sizing is now **active in the production ruleset**. Portfolio weights
  are adjusted based on estimated round-trip trading costs (ADV-derived).
- Biotech-calibrated bucket thresholds (400/1000/2000 bps) produce meaningful
  differentiation: ~26% of A+B positions get no haircut, ~32% get 0.85x, ~32%
  get 0.70x + band step-down, ~11% get 0.55x floor + band step-down.
- Membership unchanged (same 20 names), eligibility unchanged, turnover unchanged.
- Weight redistribution: liquid names (AKRO, MRUS, CELC at $80-150M ADV) gain
  weight; illiquid micro-caps (NAUT, IKT at $0.6-0.8M ADV) lose weight.
- One band step-down observed: CNTX L→M (cost_mult=0.70, ADV $3.6M).

**Rationale**: Cap calibration sweep (`artifacts/cost_cap_sweep_2026-02-11.md`,
8 values from 500 to uncapped) found clear elbow at cap=1000: binding drops to
2.7% (from 71.6% at default 200), all 4 bucket tiers populated, performance
plateaus from 1000 through uncapped (+1.61pp 60d residual vs disabled baseline).
ADV coverage audit confirmed 100% of A+B positions costed, bucket distribution
healthy (26/32/32/11%), zero missing tickers.

**Governance**: ADV coverage audit (`artifacts/cost_coverage_audit_2026-02-11.md`).
Cap sweep 8 backtests, 10 snapshots each. Phase2 portfolio regression re-pinned
(2025-10-31 snapshot, 20 positions). Contract fingerprint unchanged. Full suite green.

**Files**: `v1.2.1_candidate.json` (active, +2 fields)

---

## [v1.3.0] 34bb662d — 2026-02-11 — Recalibrate cost haircut buckets for biotech

**Parameters changed** (vs 9bc38c2d):
- `cost_haircut_buckets`: ((50, 1.0), (100, 0.85), (150, 0.70)) → ((400, 1.0), (1000, 0.85), (2000, 0.70))

**Behavior changes**:
- **None when disabled** (default). All existing outputs are identical.
- When `enable_cost_haircut=True`, buckets now produce meaningful differentiation
  across the biotech cost distribution (P30/P70/P90 percentile breaks on round-trip bps).
- Old thresholds (50/100/150 bps) placed 100% of biotech positions at the floor
  multiplier (0.55x), making the haircut a uniform no-op after weight normalization.

**Rationale**: Wired backtest with `cost_impact_cap_bps=2000` showed all 183 portfolio
positions have round-trip costs ≥178 bps (median 514, max 4036). Old large-cap equity
thresholds provided zero differentiation. New breaks at 400/1000/2000 bps create
~30%/40%/20%/10% bucket distribution matching the actual biotech cost curve.

**Governance**: 5 tests updated in `test_cost_aware_sizing.py` (bucket labels + cost
values), all pinned IDs cascaded. Contract fingerprint unchanged (cost haircut disabled
by default). 6465 tests green.

**Files**: `decision_engine.py` (default parameter), all ID-pinned files cascaded

---

## [v1.3.0] 9bc38c2d — 2026-02-11 — Add cost_impact_cap_bps + cost telemetry

**Parameters added** (vs 18d44abd):
- `cost_impact_cap_bps`: 200.0 (governs CostSchedule.impact_cap_bps when cost haircut is enabled)

**Behavior changes**:
- **None** — default 200.0 matches `DEFAULT_SCHEDULE.impact_cap_bps` exactly.
- When callers enable cost haircut, they now read `ruleset.cost_impact_cap_bps` to construct
  the `CostSchedule`, allowing cap tuning without code changes.
- Strategy backtest emits `cost_telemetry` in details JSON when panel export is active:
  `cost_coverage_pct`, `cap_binding_pct`, `n_costed`, `n_positions`.

**Rationale**: Backtest revealed cap degeneracy — `impact_cap_bps=200` compresses all biotech
round-trip costs to ~410 bps with zero differentiation across buckets. Externalizing the cap
into the ruleset enables future calibration (e.g. 2000 bps) without code changes. Telemetry
provides programmatic detection of cap-binding and low coverage.

**Governance**: 3 new tests in `test_cost_model.py` (cap dispersion, telemetry basics,
default cap invariant). All pinned IDs cascaded. Contract fingerprint unchanged.

**Files**: `v1.2.1_candidate.json` (active, schema-expanded)

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

## [v1.2.0] 010b4332 — 2025-10-15 — Candidate v1.2 (a_floor=0.55, pre-drawdown)

Pre-promotion copy of the Phase-2 production config. Identical effective
configuration to `v2_phase2_default.json` (promoted as d3cdf5c8). Predates
the drawdown gate and cost-aware sizing fields (uses engine defaults).

**Status**: Legacy candidate. Superseded by promoted v2_phase2_default.json.

**Files**: `v1.2_candidate.json`

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
