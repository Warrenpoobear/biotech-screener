# Rule 12 — Promotion Checklist (Operator)

**Authority:** Tier 0 governance / self-learning plumbing only.  
**Canonical skill:** `skills/self-improving/SKILL.md` (Rule 12) — do not fork thresholds (F-2026-001).  
**Last updated:** 2026-06-24

---

## When to use this doc

Use this checklist when promoting a captured lesson (LRN or failure-pattern) into HOT memory, a skill patch draft, or a harvest_log merge. **Capture ≠ promote.** Promotion requires clearing the gates below.

---

## Candidate feeds (do not re-count from chat)

| Source | Filter |
| --- | --- |
| `.learnings/LEARNINGS.md` | `Pattern-Key` present; recurrence per Rule 3 |
| Failure-patterns feed | `recurrence_count >= 3`, `promotion_status: PENDING` |
| Town Correction Ledger | `recurrence_count >= 3` (already-counted rows) |

Run `python3 tools/audit_learnings.py` — promotion section must match these feeds.

---

## Promotion gates

| Gate | Threshold | Action |
| --- | --- | --- |
| Recurrence | Pattern-Key **≥ 3** (7-day window for behavioral; all-time for failure modes) | Promote to HOT `memory.md` or `domains/` |
| Skill-path + recurrence | `Skill-Path` set AND recurrence **≥ 2** | Draft patch only (no auto-merge) |
| Operator verdict | **≥ 3** helpful verdicts on same skill (telemetry) | Eligible for skill merge |
| Observation | **7+** days true-PIT production telemetry | Eligible for routing/behavior changes |

---

## Lane gate (refuse wrong lane)

Every LRN must declare `Promotion-lane:` (`skill` | `spec` | `none`).

- **`spec`** — signal/scoring/research findings → governance Spec / `projects/biotech_screener.md`. `pattern_to_skillpatch.py` **MUST refuse** these entries.
- **`skill`** — ops/plumbing/docs patches only (Tier 0 default).
- **`none`** — log only; no promotion path.

---

## Propose-only path (Rule 11 FENCE)

Automated promotion is **staged**. Both env gates are required for their respective tools:

| Env var | Tool | Effect |
| --- | --- | --- |
| `SELFIMPROVE_IMMEDIATE_VERDICT=1` | `run_agent_direct.py` | Enables `record_feedback()` |
| `SELFIMPROVE_GATES_MET=1` | `pattern_to_skillpatch.py` | Writes drafts to `artifacts/skill_patch_drafts/` |

`pattern_to_skillpatch.py` also refuses when stalled-loop rows are still **OPEN** in `.learnings/memory.md` (even when `SELFIMPROVE_GATES_MET=1`).

**Weekly operator workflow (Friday):**

```bash
bash tools/cron_weekly_skills_review.sh          # digest + audit (always safe)
# After stalled-loop gates close:
export SELFIMPROVE_GATES_MET=1
python3 tools/pattern_to_skillpatch.py --min-recurrence 3 --out artifacts/skill_patch_drafts
# Review drafts → edit skills/<dir>/SKILL.md → sync → harvest_log
python3 tools/sync_hermes_skills.py
python3 tools/audit_hermes_skills.py
```

Install reference: `bash tools/install_agent_fleet_crontab.sh`

### Host onboarding (fleet migration complete — phases 2–15)

```bash
bash tools/run_operator_host_setup.sh
```

Unified setup runs fleet onboarding, then the research battery when PIT snapshots exist on the host.

Fleet-only or research-only:

```bash
bash tools/run_operator_host_setup.sh --skip-research
bash tools/run_operator_host_setup.sh --research-only
bash tools/run_fleet_host_onboarding.sh
bash tools/run_research_host_battery.sh
```

```bash
bash tools/install_agent_fleet_crontab.sh   # paste block into crontab -e
bash tools/run_fleet_operator_checklist.sh  # audit → fleet_ops → crontab verify
```

Watchdog (`cron_watchdog.sh`) auto-runs Herald health + `--recover` on FAIL (F-2026-005).  
Close stalled-loop rows in `.learnings/memory.md` only after host confirms recovery.

---

## Efficacy back-check (2 weeks post-merge)

A patch is not done at merge — it is done when recurrence stays at zero.

Append to `docs/hermes_skills/harvest_log.md`:

```markdown
### Patch verification (YYYY-MM-DD)
- **skill:** <skill-name>
- **metric:** <what was watched>
- **result:** <e.g. 0 recurrence of cron_missed import errors since 2026-06-24>
- **stalled-loop:** <F-2026-XXX or N/A>
```

If the pattern recurs: bump `Recurrence-Count`, set `promotion_status` → `PENDING`, escalate.

---

## Stalled-loop blockers (efficacy gate)

Efficacy tracking on an outage fix **cannot start** until host recovery is confirmed.

| ID | System | Operator close criterion |
| --- | --- | --- |
| F-2026-005 | Herald digest pipeline | Host produces deduped + classified JSONL ≥1 trading day; 14d zero recurrence post-fix |
| F-2026-006 | GitHub CI | Actions budget restored; `tests` workflow green on `main` |

Until closed: keep `SELFIMPROVE_GATES_MET` unset; do not mark related patches RESOLVED in failure-patterns feed.

**Herald recovery (F-2026-005):** `bash tools/herald_recovery.sh` or `python3 tools/herald_health_check.py --recover`

---

## Related

- `skills/self-improving/REFERENCE.md` — LRN template, efficacy template
- `artifacts/governance/selfimprove_audit_2026-06-24.md` — Rule 11 FENCE memo
- `.learnings/memory.md` — stalled-loop verdict table
- `governance/AGENT_ROUTING_POLICY.md` — tier definitions
