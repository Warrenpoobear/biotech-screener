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
