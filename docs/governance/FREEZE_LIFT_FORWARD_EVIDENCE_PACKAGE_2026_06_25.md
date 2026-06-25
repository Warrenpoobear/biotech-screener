# Freeze Lift + Forward Evidence Package (2026-06-25)

**Status:** OPERATOR DECISION REQUIRED  
**Governance:** Tier 0 — evidence assembly only; does not lift the scoped production model freeze  
**Authority:** Operator sign-off required before any ranker/selector/sizing/`final_score` change

---

## Purpose

This memo defines the sequential gate after host onboarding (`run_operator_host_setup.sh`):

1. **Explicit freeze lift** (operator decision, documented here)
2. **Forward evidence refresh** (`coinvest_score_z`, `final_score`, forward-eval IC, coinvest shadow)
3. **Strategic fork** — re-affirm or re-examine the coinvest-only selector thesis

The tooling produces artifacts; **only the operator lifts the freeze.**

---

## Prerequisites

| Step | Requirement |
| --- | --- |
| Host setup | `bash tools/run_operator_host_setup.sh` on WSL |
| Snapshots | `data/snapshots/{date}/rankings.csv` through as-of date |
| PIT archive (Spec 100 battery) | `data/snapshots_pit_v2/` for full Checklist v2 rerun |
| Stalled loops | Close F-2026-005 / F-2026-006 before `SELFIMPROVE_GATES_MET=1` |

---

## Run the evidence package

```bash
# Preview (no writes)
bash tools/run_forward_evidence_package.sh --dry-run

# Persist artifacts (operator acknowledgment required)
export FREEZE_LIFT_ACK=1
bash tools/run_forward_evidence_package.sh --write
```

**Outputs:**

| Artifact | Path |
| --- | --- |
| Evidence JSON | `artifacts/forward_evidence/{date}_package.json` |
| Evidence markdown | `artifacts/forward_evidence/{date}_package.md` |
| Path C close | `artifacts/governance/path_c_window_close_{date}.json` |

---

## Path C retrospective close (overdue)

The catalyst-timing override (2026-05-28 through **2026-06-03**) was never formally closed in the repo.

```bash
python3 tools/path_c_window_close_decision.py --window-end 2026-06-03 --write
```

| Decision | Meaning |
| --- | --- |
| `PATH_C_VALID` | Forward-eval IC ≥ 0.0200 at window end — override stood |
| `PATH_C_REVOKE` | IC below floor — revert to HOLD, accelerate Path A |
| `IC_UNOBSERVABLE` | Operator chooses extend vs revert (document rationale) |

---

## Freeze lift checklist (operator sign-off)

- [ ] Host battery artifacts reviewed (`output/checklist_v2_rerun/`, Spec 105)
- [ ] Forward evidence package generated with `FREEZE_LIFT_ACK=1`
- [ ] Path C retrospective decision recorded
- [ ] Advisory verdict reviewed: `POSITIVE` | `OBSERVE` | `NEGATIVE` | `INSUFFICIENT_DATA`
- [ ] Prior baseline superseded: coinvest pooled IC −0.031 (2026-05-13, OBSERVE)
- [ ] Decision documented with date and operator initials below
- [ ] If lift approved: Path A portfolio timing gate design authorized — see **Spec 106** (`docs/governance/PATH_A_PORTFOLIO_TIMING_GATES_SPEC_106_2026_06_25.md`)

### Strategic fork

| Advisory verdict | Action |
| --- | --- |
| **POSITIVE** | Re-affirm coinvest-only selector from clean forward data |
| **OBSERVE** | Continue forward shadow accumulation; no selector changes |
| **NEGATIVE** | Structural selector re-examination — not plumbing tweaks |
| **INSUFFICIENT_DATA** | Fix host data gaps; do not lift freeze |

---

## Freeze lift record (operator fill-in)

| Field | Value |
| --- | --- |
| Decision date | |
| Lift approved? | YES / NO / DEFER |
| Advisory verdict at decision | |
| Operator | |
| Notes | |

---

## Governance constraints (unchanged)

- Scoped freeze remains in effect until operator records lift above
- Autonomous PIT/backtest research (PR #379) is **quarantined** — not accepted evidence
- No ranker/selector/sizing/`final_score` production changes without Spec + lift
- Path A timing gates are the durable fix for catalyst concentration (post-freeze build)
- **Design spec:** `docs/governance/PATH_A_PORTFOLIO_TIMING_GATES_SPEC_106_2026_06_25.md` (Spec 106)

---

## Related

- `docs/governance/INC-2026-06-20-AUTOPUSH/` — scoped freeze origin
- `.claude/rules/operational-state.md` — sequential gate reference
- `docs/hermes_skills/path-c-governance-monitoring.md` — Path C monitoring framework
- `docs/hermes_skills/path-c-operational-runbook.md` — daily monitoring runbook
