# Decision Engine Ruleset Changelog

All material changes to decision engine logic, parameters, or governance.

Format: `[engine_version] ruleset_id — date — summary`

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
