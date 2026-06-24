# Agent Knowledge Stack (biotech-screener)

Map of **recursive self-improvement** stores. Skills tell agents *how to behave*; this tree holds *what was learned*.

## Load order (Hermes / Cursor)

1. **HOT** — `.learnings/memory.md` (always, ≤100 lines)
2. **WARM** — `.learnings/projects/biotech_screener.md` (this repo, ≤200 lines)
3. **WARM** — `.learnings/domains/*.md` (cross-cutting, ≤200 lines each)
4. **INDEX** — `.learnings/LEARNINGS.md` (structured LRN entries; search by Pattern-Key)
5. **LOG** — `.learnings/corrections.md`, `ERRORS.md` (raw, newest first)
6. **Ops runtime** — `artifacts/ops/knowledge_layer/` (build on host; WSL authoritative for cron)

Executable guidance lives in `skills/` → sync → `docs/hermes_skills/`. Loop spec: `skills/self-improving/SKILL.md`.

## Recursive loop

```
Observe → Log → Distill → Promote → Skill-patch → Sync → Verify
```

| Tier | Path | Role |
| --- | --- | --- |
| LOG | `corrections.md`, `ERRORS.md` | Immediate corrections / failures |
| INDEX | `LEARNINGS.md` | `[LRN-YYYYMMDD-XXX]` + Pattern-Key + Recurrence-Count |
| HOT | `memory.md` | 3× confirmed, cross-session critical |
| WARM | `projects/biotech_screener.md` | Repo-specific durable facts |
| WARM | `domains/{name}.md` | Domain slices (e.g. `agent_ops.md`) |
| COLD | `archive/` | Demoted / superseded (create when archiving) |
| Ops | `artifacts/ops/*_ledger/` | Held specs, contradictions, first-fire (Spec 089) |
| Skills | `skills/<dir>/SKILL.md` | Agent-executable runbooks |
| History | `docs/hermes_skills/harvest_log.md` | Git-auditable skill promotion log |

## Maintenance commands

```bash
# Read-only audit: tiers, promotion candidates, stale hints
python3 tools/audit_learnings.py

# Weekly operator digest (promotion candidates + telemetry + drafts)
python3 tools/weekly_skills_digest.py

# Skill-patch drafts (gated — operator review required)
SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py

# Hermes ops ledgers (operator WSL for authoritative cron)
python3 tools/build_hermes_knowledge_layer.py

# After skill edits
python3 tools/sync_hermes_skills.py
python3 tools/audit_hermes_skills.py
```

## Promotion rules

- **Log immediately**: explicit user correction or preference
- **LEARNINGS entry**: non-trivial lesson with `Pattern-Key`, `Area`, `Promotion-lane`
- **Promotion-lane `skill`**: ops/tooling → may become `skills/` patch after review
- **Promotion-lane `spec`**: research/portfolio/signal → governance Spec only, never auto skill-patch
- **Promotion-lane `none`**: historical log, no promotion expected
- **HOT (`memory.md`)**: same Pattern-Key ≥3× in 7 days or critical ops/governance
- **Skill patch**: operator workflow, tooling, conventions — never scoring weights without Spec
- **Verify**: patch verification block in `harvest_log.md` 2 weeks post-merge

## Memory optimization (HOT tier)

`memory.md` is the **session bootstrap** — optimize for recursion, not encyclopedic coverage.

| Principle | Practice |
| --- | --- |
| Bootstrap first | `## Bootstrap (read first)` table: recursion, CodeGraph, host, governance |
| One home per fact | Detail in `domains/` or `projects/`; HOT holds pointers + Pattern-Keys |
| Tag promotions | Bold `**pattern_key**:` in HOT; set LRN `Status: promoted` when copied |
| Stay under budget | ≤100 lines HOT, ≤200 WARM; audit warns at >70 HOT lines |
| Demote don't delete | Move stale bullets to `.learnings/archive/` with operator OK |

```bash
python3 tools/audit_learnings.py   # bootstrap lines, compaction hints, LRN/HOT mismatches
```

## Governance

Learnings and knowledge files are **Tier 0**. They must not encode ranker/selector/sizing/`final_score` changes. SOUL.md and runtime cron beat stale markdown.

## Related

| Resource | Path |
| --- | --- |
| Self-improving skill | `skills/self-improving/SKILL.md` |
| Templates | `skills/self-improving/REFERENCE.md` |
| Hermes MCP ledgers | `mcp_server/hermes_server.py` → `knowledge_read` |
| Spec 089 | `specs/changes/spec_089_hermes_knowledge_layer.md` |
