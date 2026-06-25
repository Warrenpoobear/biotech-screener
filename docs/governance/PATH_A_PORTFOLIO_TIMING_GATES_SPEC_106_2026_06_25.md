# Spec 106 — Path A Portfolio Timing Gates (Design)

**Status:** DESIGN LOCK (draft) — implementation blocked until scoped freeze lift  
**Governance:** Tier 2 design / Tier 3 implementation (touches portfolio construction + health gates)  
**Date:** 2026-06-25  
**Supersedes:** Path C temporary override (2026-05-28 → 2026-06-03)  
**Authority:** Operator sign-off required before implementation PR

---

## 1. Executive summary

Path A is the **durable construction-policy fix** for the policy/signal mismatch exposed during Path C.

| Layer | Role | Path A change? |
| --- | --- | --- |
| Selector (`coinvest_score_z`) | Ranks institutional conviction | **No** |
| Ranker (`final_score`, pairwise_minimal) | Ordinal sort within cohort | **No** |
| **Portfolio construction** | EW Top-30 bucket allocation | **Yes** |
| Phase 2 health gates | Readiness / exposure monitoring | **Align thresholds** |

**Core idea:** Let the selector and ranker express institutional consensus without forcing the portfolio to inherit accidental near-term catalyst concentration. Enforce explicit **timing gates** at construction time:

| Gate | Constraint | Rationale |
| --- | --- | --- |
| **T0 cap** | ≤ **30%** weight where `catalyst_days ≤ 7` | Path C observed 40.83% in 0–7d vs 10% legacy policy target — concentration was real but unbounded |
| **T3+ floor** | ≥ **40%** weight where `catalyst_days ≥ 90` | Path C regime observed ~26.7% in 91–180d vs 55% legacy `binary_91_180` target — under-allocated far window |

Path A does **not** add alpha signals, change weights, or reopen dead research lanes. It is a **portfolio timing overlay** on the existing EW Top-30 construction mode.

---

## 2. Problem statement

### 2.1 What broke readiness (Path C context)

During the 2026-05-28 governance review:

- **Observed:** `catalyst_le_7d_weight_pct` = **40.83%** (7 names ≤7d)
- **Legacy policy:** `binary_0_30` bucket target = **10%** (`production_data/portfolio_policy.json`)
- **Health gate FAIL:** `fail_catalyst_le_7d_weight_pct` = **35%** (`Phase2HealthThresholds`)
- **Diagnosis:** Near-term concentration reflected **real institutional consensus** (COGT, RVMD, SYRE, PRAX), not selector bias or data failure

Path C approved a **time-bounded override** (through 2026-06-03) monitored by forward-eval IC floor (0.0200). Path A replaces the override with **structural gates** that accept intentional near-term exposure up to a cap while guaranteeing minimum far-horizon allocation.

### 2.2 Why not fix the selector?

Research evidence (Checklist v2, Spec 050) shows:

- B6 coinvest-only selector + pairwise ranker + EW Top-30 is the validated bundle
- `inst_delta_z` works within top-30 in ranker (NW-t +3.32) but was zeroed in selector for negative selector IC
- Clinical, options, insider standalone lanes are closed

The mismatch is **construction policy vs signal output**, not signal quality per se. Changing selector weights to "spread out" catalyst timing would destroy the institutional thesis without evidence.

### 2.3 Non-goals

- Change `coinvest_score_z`, `final_score`, ranker, or selector weights
- Introduce rank-weighting or confidence sizing (pairwise ECE = 0.129 — ordinal-only remains correct)
- Replace EW Top-30 construction mode
- Use quarantined autonomous PIT research (PR #379) as promotion evidence
- Modify production scoring paths during architecture freeze

---

## 3. Design principles

1. **Signal/construction separation** — Selection and ranking are upstream; timing gates apply only at portfolio assembly.
2. **Determinism** — Same `rankings.csv` + policy + `as_of_date` → byte-identical positions artifact.
3. **PIT safety** — All `catalyst_days` inputs come from snapshot fields with `data_available_timestamp ≤ as_of_date - 1 day`.
4. **Fail-visible** — Infeasible gate combinations emit `PATH_A_INFEASIBLE` in construction manifest, not silent relaxation.
5. **Config-driven** — Gate thresholds live in `portfolio_policy.json`; no hard-coded allocation ratios in Python.
6. **Governance-first** — Promotion requires forward-shadow ablation, not historical backfill alone.

---

## 4. Timing zone model

### 4.1 Canonical zones (Path A)

Path A uses **raw `catalyst_days`** from portfolio rows (same field as `compute_exposure_metrics` in `run_phase2_snapshot_delta.py`), not only decision-engine bucket labels.

| Zone ID | Condition | Path A gate | Maps to action bucket |
| --- | --- | --- | --- |
| **T0** (imminent) | `1 ≤ catalyst_days ≤ 7` | **MAX 30%** weight | Subset of `binary_0_30` |
| **T1** (near) | `8 ≤ catalyst_days ≤ 30` | No hard gate (inherits bucket policy) | `binary_0_30` |
| **T2** (build) | `31 ≤ catalyst_days ≤ 89` | No hard gate | `binary_31_90` |
| **T3** (far) | `90 ≤ catalyst_days ≤ 180` | Counts toward **T3+ floor** | `binary_91_180` |
| **T4** (none/distant) | `catalyst_days > 180` OR `catalyst_mode ∈ {no_upcoming, missing}` | Counts toward **T3+ floor** | `less_binary` / `core` |

**Notes:**

- `catalyst_days = 0` with `blended_window` → T4 (no specific dated catalyst); does not count toward T0.
- Hysteresis: reuse `bucket_hysteresis_days` (7) from portfolio policy when classifying borderline names for **construction** stability; exposure metrics for health gates use raw days (no hysteresis) for conservative monitoring.

### 4.2 Relationship to existing `bucket_targets`

Current `portfolio_policy.v6` targets (promoted 2026-04-01):

```json
"bucket_targets": {
  "binary_91_180": 0.55,
  "binary_31_90": 0.25,
  "binary_0_30": 0.10,
  "less_binary": 0.10
}
```

Path A gates operate on **orthogonal dimensions**:

| Mechanism | Granularity | Purpose |
| --- | --- | --- |
| `bucket_targets` | Action buckets (`binary_*`) | Historical sleeve construction |
| **Path A T0 cap** | `catalyst_days ≤ 7` | Hard near-term event-risk ceiling |
| **Path A T3+ floor** | `catalyst_days ≥ 90` + T4 | Hard far-horizon minimum |

**Design decision:** Path A gates are enforced **after** bucket-aware Top-K selection in `live_shadow_portfolio.py`. If gates conflict with `bucket_targets`, **gates win** and construction manifest records which constraint bound.

**Post-implementation policy sync:** Once Path A ships, revise `bucket_targets.binary_0_30` documentation to clarify it is a soft target subordinate to T0 cap; consider calibrating `binary_91_180` soft target from 55% → 45% to reduce tension with T3+ floor (operator decision at promotion).

---

## 5. Construction algorithm

### 5.1 Inputs

| Input | Source |
| --- | --- |
| Ranked universe | `rankings.csv` (selector + ranker output, unchanged) |
| Portfolio policy | `production_data/portfolio_policy.json` + `path_a_timing_gates` block |
| As-of date | `as_of_date` (ISO 8601) |
| Prior positions | Optional (for hysteresis / rebalance buffer) |

### 5.2 Algorithm (deterministic greedy)

```
1. BUILD_CANDIDATE_POOL
   - Start from rank-ordered list (final_score desc, existing tie-breakers)
   - Tag each row with path_a_zone (T0..T4)

2. INITIAL_SELECT
   - Run existing EW Top-30 bucket selection (live_shadow_portfolio.py)
   - Produce initial 30 names + weights (EW within selected set)

3. MEASURE_GATES
   - t0_weight = sum(weight where zone == T0)
   - t3plus_weight = sum(weight where zone in {T3, T4})

4. ENFORCE_T0_CAP (if t0_weight > t0_max)
   - Sort T0 names by ascending rank (drop worst first)
   - Remove names until t0_weight <= t0_max
   - Backfill from highest-ranked non-T0 names not yet in portfolio
   - Repeat until |portfolio| == 30 or pool exhausted

5. ENFORCE_T3PLUS_FLOOR (if t3plus_weight < t3plus_min)
   - Sort non-T3/T4 names by ascending rank
   - Swap out lowest-ranked non-T3/T4 for highest-ranked T3/T4 not in portfolio
   - Repeat until t3plus_weight >= t3plus_min or infeasible

6. REWEIGHT
   - Re-equal-weight across final 30 (preserve EW Top-30 mode)
   - Re-apply global_name_cap (3.0%) and risk_layer checks

7. EMIT_MANIFEST
   - Record gate measurements, swaps, infeasibility flags
```

### 5.3 Infeasibility handling

| Condition | Action | Readiness impact |
| --- | --- | --- |
| Cannot fill 30 names while satisfying both gates | `PATH_A_INFEASIBLE` = true; relax **floor first** (T3+), never cap (T0) | WARN + operator review |
| Cannot satisfy T0 cap with 30 names | Drop to min portfolio size (existing alpha_health rules) | FAIL — do not promote |
| Pool has <30 eligible names | Existing construction fallback | unchanged |

**Rationale:** T0 cap protects event-risk; T3+ floor is a diversification preference. Under stress, cap integrity dominates.

### 5.4 Pseudocode contract

```python
@dataclass(frozen=True)
class PathATimingGates:
    enabled: bool
    t0_max_days: int = 7
    t0_max_weight_pct: Decimal = Decimal("30.0")
    t3_min_days: int = 90
    t3_plus_min_weight_pct: Decimal = Decimal("40.0")
    infeasibility_relax_order: tuple[str, ...] = ("t3_plus_floor",)

def classify_path_a_zone(catalyst_days: int | None, catalyst_mode: str) -> str: ...
def enforce_path_a_gates(positions: list[Position], ranked: list[Row], gates: PathATimingGates) -> PathAResult: ...
```

All arithmetic: `Decimal`, `ROUND_HALF_UP`. No `float` in gate enforcement.

---

## 6. Configuration schema

Add to `portfolio_policy.json` (new section; schema bump to `portfolio_policy.v7`):

```json
{
  "schema": "portfolio_policy.v7",
  "path_a_timing_gates": {
    "enabled": false,
    "schema": "path_a_timing_gates.v1",
    "t0_imminent": {
      "max_days": 7,
      "max_weight_pct": 30.0
    },
    "t3_plus": {
      "min_days": 90,
      "min_weight_pct": 40.0,
      "include_zones": ["T3", "T4"]
    },
    "enforcement": "post_selection_overlay",
    "infeasibility_relax_order": ["t3_plus_floor"],
    "hysteresis_days": 7
  }
}
```

**Rollout:** `enabled: false` until promotion; shadow runs with `enabled: true` via `--policy` override.

---

## 7. Integration map

| Component | Change type | File |
| --- | --- | --- |
| Construction overlay | **Implement** | `tools/live_shadow_portfolio.py` |
| Zone classifier | **Implement** | `tools/path_a_timing_gates.py` (new) |
| Policy schema | **Extend** | `production_data/portfolio_policy.json` |
| Exposure metrics | **Verify** (existing) | `run_phase2_snapshot_delta.py::compute_exposure_metrics` |
| Health thresholds | **Align** | `production_data/phase2_health_thresholds/v1.json` |
| Readiness scorecard | **Extend** | `tools/weekly_readiness_scorecard.py` |
| Shadow monitor | **Extend** | `tools/build_shadow_monitor.py` (gate compliance time series) |
| Daily production | **Wire** | `tools/run_daily_production.py` (manifest hash) |

### 7.1 Health gate alignment (proposed)

| Metric | Current default | Path A aligned |
| --- | --- | --- |
| `warn_catalyst_le_7d_weight_pct` | 20% | 25% |
| `fail_catalyst_le_7d_weight_pct` | 35% | **30%** (matches T0 cap) |
| `warn_catalyst_le_7d_count` | 4 | 5 |
| `fail_catalyst_le_7d_count` | 7 | 6 |

New metric (add to exposure block):

```json
"catalyst_ge_90d_weight_pct": 42.5,
"catalyst_ge_90d_floor_breach": false
```

### 7.2 Outputs / artifacts

| Artifact | Path | Schema |
| --- | --- | --- |
| Positions (existing) | `data/snapshots/{date}/shadow_positions.json` | unchanged + `path_a_compliance` block |
| Gate manifest | `artifacts/portfolio_construction/{date}_path_a_manifest.json` | `path_a_manifest.v1` |
| Compliance summary | `artifacts/portfolio_construction/{date}_path_a_compliance.md` | human-readable |

**Manifest required fields:**

```json
{
  "schema": "path_a_manifest.v1",
  "as_of_date": "2026-06-24",
  "generated_at": "2026-06-24T00:00:00Z",
  "gates": {
    "t0_weight_pct": 28.5,
    "t0_cap_pct": 30.0,
    "t3_plus_weight_pct": 41.2,
    "t3_plus_floor_pct": 40.0
  },
  "compliant": true,
  "swaps": [],
  "infeasibility": null,
  "_governance": {
    "run_id": "sha256:...",
    "policy_hash": "sha256:...",
    "pit_cutoff": "2026-06-23"
  }
}
```

---

## 8. Promotion path (Rule 12)

### 8.1 Prerequisites (sequential)

1. **Freeze lift recorded** — `docs/governance/FREEZE_LIFT_FORWARD_EVIDENCE_PACKAGE_2026_06_25.md` operator table filled
2. **Forward evidence reviewed** — advisory verdict `POSITIVE` or `OBSERVE` with explicit Path A authorization
3. **Path C closed** — `artifacts/governance/path_c_window_close_{date}.json` on file
4. **This spec design-locked** — operator initials in §12 below

### 8.2 Evidence battery (pre-implementation)

| Test | Command / tool | Pass criterion |
| --- | --- | --- |
| Replay Path C snapshot | `live_shadow_portfolio.py --as-of-date 2026-05-28 --policy path_a_shadow.json` | T0 ≤ 30%, T3+ ≥ 40%, readiness gate no longer FAIL on `catalyst_7d_weight_high` |
| Forward shadow ablation | Compare 30d shadow PnL: baseline EW vs Path A | Sharpe delta ≥ 0 (non-inferior); no CRITICAL drawdown regression |
| Determinism | Run twice same inputs | Byte-identical manifest + positions hash |
| Infeasibility fixture | Synthetic rankings with 20 T0 names | `PATH_A_INFEASIBLE` emitted; T0 cap never breached |

### 8.3 Implementation tiers

| Phase | Scope | Governance |
| --- | --- | --- |
| **A0** (shadow-only) | `path_a_timing_gates.py` + tests; policy `enabled: false` in prod | Tier 2 — safe during freeze if shadow-only |
| **A1** (shadow enabled) | `run_path_a_shadow.sh` wired in `run_operator_host_setup.sh` step 3 | Tier 2 — observability |
| **A2** (production) | Flip `enabled: true` in production policy; align health thresholds | **Tier 3 — requires freeze lift** |
| **A3** (ruleset) | Bump ruleset manifest entry; update `MODEL_DOCUMENTATION.md` | Tier 3 |

### 8.4 Rollback

- Set `path_a_timing_gates.enabled: false`
- Revert to `portfolio_policy.v6` targets
- Prior snapshot positions remain immutable (new `run_id` on rerun)

---

## 9. Test plan

| ID | Type | Description |
| --- | --- | --- |
| PA-01 | Unit | `classify_path_a_zone` boundary cases (0, 7, 8, 89, 90, 181, missing) |
| PA-02 | Unit | T0 cap enforcement drops correct names deterministically |
| PA-03 | Unit | T3+ floor backfill prefers highest-ranked eligible |
| PA-04 | Unit | Infeasibility relaxes floor before cap |
| PA-05 | Integration | Path C 2026-05-28 fixture → compliant portfolio |
| PA-06 | Integration | Health gate `catalyst_7d_weight_high` no longer FAIL at 28% T0 |
| PA-07 | Leakage | `catalyst_days` source_date ≤ `as_of_date - 1` for all swapped names |
| PA-08 | Determinism | Double-run hash match on manifest + positions |
| PA-09 | Regression | EW Top-30 count always 30 when pool allows |
| PA-10 | Governance | Manifest includes `_governance` block with `parameters_hash` |

---

## 10. Monitoring (post-promotion)

| Signal | Frequency | Threshold |
| --- | --- | --- |
| `t0_weight_pct` | Daily (post-construction) | ≤ 30% |
| `t3_plus_weight_pct` | Daily | ≥ 40% |
| Gate swap count | Daily | > 5 swaps → WARN (regime shift) |
| `PATH_A_INFEASIBLE` | Daily | Any → operator digest |
| Forward eval IC | Daily (existing) | Floor 0.0200 unchanged |
| Shadow ablation delta | Weekly | Sharpe vs baseline |

Wire into: `tools/build_shadow_monitor.py`, `tools/weekly_readiness_scorecard.py`, `agents/ops_supervisor/supervisor.py` (read-only escalation).

---

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Gates fight ranker order too aggressively | Limit swaps to minimum needed; log every swap with rank delta |
| Thin T3+ pool in bull regime | Infeasibility path + WARN; do not silently violate T0 cap |
| `catalyst_days` stale / missing | Fall back to `catalyst_mode`; count as T4; flag in manifest |
| Confusion with `bucket_targets` | Document precedence; emit both bucket and zone weights in manifest |
| Freeze scope creep | A0/A1 shadow-only PRs explicitly exclude `enabled: true` in production policy |

---

## 12. Design lock (operator fill-in)

| Field | Value |
| --- | --- |
| Spec ID | **106** |
| Design approved? | YES / NO / DEFER |
| T0 cap (30%) | CONFIRM / REVISE: ___% |
| T3+ floor (40%) | CONFIRM / REVISE: ___% |
| Implementation authorized? | A0 only / A0+A1 / Full A2 |
| Operator | |
| Date | |

---

## 13. Related documents

| Doc | Relationship |
| --- | --- |
| `docs/governance/FREEZE_LIFT_FORWARD_EVIDENCE_PACKAGE_2026_06_25.md` | Prerequisite gate before A2 |
| `docs/hermes_skills/path-c-governance-monitoring.md` | Path C override being replaced |
| `docs/hermes_skills/path-c-operational-runbook.md` | Window close → Path A handoff |
| `production_data/portfolio_policy.json` | Current EW Top-30 v6 targets |
| `run_phase2_snapshot_delta.py` | Exposure metrics + health gates |
| `tools/live_shadow_portfolio.py` | Primary implementation target |
| `docs/governance/RULE_12_PROMOTION_CHECKLIST.md` | Promotion battery |
| `.claude/rules/research-backtest.md` | Dead lanes (do not reopen) |

---

**End of Spec 106 draft.**
