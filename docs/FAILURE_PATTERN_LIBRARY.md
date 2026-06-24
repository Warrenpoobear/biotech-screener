# Failure Pattern Library

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18
**Last updated:** 2026-05-25
**Scope:** Reference documentation only

---

## Purpose

A queryable, categorized catalog of confirmed operational failure modes for the biotech screener pipeline and supporting infrastructure.

This is a **reference catalog**, not an active monitoring system. It does not trigger alerts, modify agent behavior, or feed into any automation. Its sole function is to serve as a memory index of known failure modes so that operators and agents can recognize recurring problems and apply documented resolutions before re-investigating from scratch.

## Governance

- Entries require **confirmed root cause** before inclusion. No speculative entries.
- Unresolved entries may only receive `recurrence_count` and diagnostic updates, not speculative fixes.
- This document does not modify any: models, rankers, selectors, sizing logic, KG, agent registry, skills, MCP configs, runtime state, or cron jobs.
- Prevention rules documented here are **reference only** — they are not automatically promoted into skills or agent prompts.

---

## Taxonomy

| Code | Category | Description |
|------|----------|-------------|
| DS | Data staleness | Input data exceeded freshness threshold |
| CM | Cache miss / corruption | Cache file missing, truncated, or stale |
| PT | Pipeline timeout | Job exceeded time limit |
| LE | Logic error | Code produced wrong output from correct input |
| DG | Doc-sync gap | Documents contradicted each other or production state |
| IF | Infrastructure failure | Service unreachable, API down, credential expired |
| GL | Governance lapse | Process skipped, gate bypassed, approval missing |
| NC | Naming collision | Same concept referenced by different names across docs |
| MA | Misattribution | Metric or claim attributed to wrong source or scope |

---

## Entry Schema

Each catalog entry uses the following keys:

```yaml
failure_id: F-YYYY-NNN
category: <code from taxonomy>
first_seen: YYYY-MM-DD
recurrence_count: <integer>
severity: LOW | MEDIUM | HIGH | CRITICAL
summary: <one-line description>
root_cause: <what actually went wrong>
affected_systems: [<list of skills/agents/pipelines>]
resolution: <what fixed it, or UNRESOLVED | PARTIALLY RESOLVED>
prevention_rule: <what should change to prevent recurrence>
related_specs: [<spec numbers if applicable>]
related_findings: [<C1, W3, G5 etc. from doc review>]
```

---

## Usage Rules

1. **Search before investigating.** Before investigating a new error, search this catalog by category and keywords first.
2. **Apply known fixes first.** If a match exists and resolution is documented, apply the known fix before re-investigating.
3. **Update recurrence only.** If a match exists but resolution is `UNRESOLVED` or `PARTIALLY RESOLVED`, update `recurrence_count` and diagnostic information only. Do not add speculative resolutions.
4. **Confirmed root cause required.** New entries require confirmed root cause. Do not add speculative entries.
5. **Promotion threshold.** Entries with `recurrence_count >= 3` may be *proposed* for prevention-rule promotion into the relevant skill document, but promotion is never automatic — it requires explicit operator approval.
6. **No automation.** This catalog does not feed into any automated monitoring, alerting, or remediation system.

---

## Catalog

### F-2026-001

```yaml
failure_id: F-2026-001
category: NC
first_seen: 2026-05-17
recurrence_count: 4
severity: HIGH
summary: Selector terminology mismatch — "sponsorship_score_z" / "momentum_delta_z" used in external docs; production uses "coinvest_score_z" / "inst_delta_z" since v1.14.0
root_cause: Signal renaming in code did not propagate to all documentation layers
affected_systems: [selector-ranker, institutional-signal, .docx exports]
resolution: CON-1 cross-reference note added to selector-ranker skill. GitHub/docx not updated.
prevention_rule: Any signal rename must include a doc-propagation checklist — skill -> GitHub model docs -> .docx exports -> agent prompts
related_specs: []
related_findings: [C1, W6]
```

### F-2026-002

```yaml
failure_id: F-2026-002
category: MA
first_seen: 2026-05-13
recurrence_count: 1
severity: CRITICAL
summary: IC tooling scope conflation — run_rank_ic_backtest.py measured composite_score IC, not production final_score IC. Composite_rank correlates only 0.25 with actionable_rank.
root_cause: Tool was built to measure one signal; was silently assumed to measure the production signal
affected_systems: [ic-evaluation, selector-ranker, all historical IC claims]
resolution: Spec 100 committed (2faa88e6, 2026-05-17). Prior claims invalidated. Checklist v2 battery rerun deferred post-freeze.
prevention_rule: Any IC measurement tool must explicitly declare which score field it measures in its output header. No IC claim is valid without matching the declared field to the production sort key.
related_specs: [Spec 095, Spec 100]
related_findings: [G2]
```

### F-2026-003

```yaml
failure_id: F-2026-003
category: DG
first_seen: 2026-05-16
recurrence_count: 2
severity: HIGH
summary: inst_delta_z scope misattribution — skill stated "excluded from ranker since Spec 051" when it was excluded from the SELECTOR, not the ranker. inst_delta_z remains active in ranker (NW-t = +3.32).
root_cause: Imprecise language in skill update — "ranker" used when "selector" was meant
affected_systems: [selector-ranker, institutional-signal]
resolution: Code Review H3 fix noted; skill text corrected
prevention_rule: When describing signal status changes, always specify the exact engine layer (selector vs. ranker vs. composite vs. decision engine)
related_specs: [Spec 051]
related_findings: [C2]
```

### F-2026-004

```yaml
failure_id: F-2026-004
category: PT
first_seen: 2026-04-01
recurrence_count: 3
severity: MEDIUM
summary: AACT pipeline timeout — 4500s timeout killed daily pipeline mid-AACT ingestion, particularly on Monday runs (weekend AACT batch)
root_cause: Timeout set too aggressively for worst-case AACT ingestion duration
affected_systems: [screener-ops daily pipeline, catalyst-resolution]
resolution: Timeout increased from 4500s to 6000s
prevention_rule: Monday pipeline runs should have extended timeout or dedicated monitoring. Any timeout change should be validated against 4-week Monday duration distribution.
related_specs: []
related_findings: []
```

### F-2026-005

```yaml
failure_id: F-2026-005
category: IF
first_seen: 2026-04-14
recurrence_count: 6
severity: CRITICAL
summary: Herald Digest extended outage — zero output for 6+ consecutive weeks. No deduped or classified JSONL files generated.
root_cause: UNRESOLVED
affected_systems: [Herald pipeline, press release monitoring, downstream news-driven signals]
resolution: OPEN (2026-06-24). Code fixes merged; host recovery unconfirmed. Last repo classified JSONL 2026-02-26. Efficacy tracking blocked until operator confirms Herald cron output on WSL. Target 2026-07-01.
prevention_rule: Herald should have a max-dark-days SLA (proposed 3 days) with automatic escalation. See operational-health-baselines skill.
related_specs: []
related_findings: [G6]
```

### F-2026-006

```yaml
failure_id: F-2026-006
category: IF
first_seen: 2026-05-08
recurrence_count: 1
severity: HIGH
summary: CI pipeline extended red state — CI red ~17 days as of May 25. PR #285 open/unmerged. phase2-daily-production cron dark.
root_cause: Budget exhaustion (GitHub Actions). CI Diagnostic Report and CI Fix Checklist produced May 14-16; remediation not confirmed complete.
affected_systems: [All merge gates, production deployment confidence, Herald Digest restore]
resolution: OPEN (2026-06-24). GitHub Actions failing in ~3–4s on main (budget exhaustion pattern). Host must restore budget and confirm green CI. Blocks Herald restore verification. Target 2026-07-01.
prevention_rule: CI red > 5 days should trigger merge block and operator escalation. See operational-health-baselines skill.
related_specs: []
related_findings: [C4]
```

### F-2026-007

```yaml
failure_id: F-2026-007
category: DG
first_seen: 2026-05-16
recurrence_count: 2
severity: MEDIUM
summary: Clinical score denominator confusion — GitHub referenced denominator 117; clinical-scoring skill showed 120. Both internally consistent at different layers.
root_cause: Two fixes applied at different layers without cross-referencing each other
affected_systems: [clinical-scoring, GitHub model_documentation_root.md]
resolution: RESOLVED (C7, Run 4). Both references documented as internally consistent.
prevention_rule: When a fix touches a score denominator or weight at one layer, check all other layers that reference the same total.
related_specs: []
related_findings: [C3, C7]
```

### F-2026-008

```yaml
failure_id: F-2026-008
category: NC
first_seen: 2026-05-17
recurrence_count: 5
severity: MEDIUM
summary: Agent fleet count inconsistency — count appears as 17, 26, 27, 28, and 30 across different documents
root_cause: Agent count is a moving target as agents are added/suppressed/retired; documents capture point-in-time counts without dating them
affected_systems: [All governance documents, compliance memos, exec overview]
resolution: agent_governance.md (GitHub, 2026-05-17) designated as most authoritative (30 agents: 27 active + 1 shadow + 1 suppressed + 1 retired). Other docs remain stale.
prevention_rule: Agent count should always be sourced from agent_governance.md with a dated citation. All other documents should say "see agent_governance.md for current count" rather than hardcoding a number.
related_specs: []
related_findings: [C6, W3]
```

### F-2026-009

```yaml
failure_id: F-2026-009
category: DG
first_seen: 2026-05-25
recurrence_count: 1
severity: MEDIUM
summary: coinvest_score_z deployed vs. trained weight discrepancy — skill implied ranker weight = +0.0613 (trained basis); deployed weight = +0.02 (capped Family C live pilot).
root_cause: Skill referenced the artifact's trained coefficient without noting the live-pilot cap applied on top
affected_systems: [selector-ranker skill]
resolution: selector-ranker skill updated 2026-05-25 to document both values and distinguish deployed (+0.02) from trained (+0.0613).
prevention_rule: Whenever a ranker weight has a live-pilot cap or override applied, both the trained value and the deployed value must be explicitly documented in the skill.
related_specs: []
related_findings: [N1]
```

---

## Revision History

| Date | Change |
|------|--------|
| 2026-05-18 | Initial catalog created (F-2026-001 through F-2026-008) |
| 2026-06-24 | F-2026-005/F-2026-006 stalled-loop verdicts filled OPEN (cloud evidence); targets 2026-07-01 pending host confirm. Checklist v2 vs final_score blocked in cloud — see `docs/research/CHECKLIST_V2_FINAL_SCORE_BLOCKER_2026_06_24.md`. |
| 2026-05-25 | F-2026-009 added. F-2026-005 recurrence updated to 6+ weeks. F-2026-006 CI red updated to ~17 days. Normalized to strict YAML schema. |
