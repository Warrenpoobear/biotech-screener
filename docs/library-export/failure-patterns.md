# Failure Pattern Library

**Status:** ACTIVE (activated 2026-07-15)
**Created:** 2026-05-18
**Priority:** 2 of 7

## Purpose

Provide a queryable, categorized catalog of past failure modes so Hermes agents can recognize recurring problems, apply known fixes, and avoid re-investigating solved issues.

## Preconditions

- This is a reference catalog, not an active monitoring system.
- Entries are added manually after post-mortem or audit, not automatically.
- Each entry must have a confirmed root cause before being added (no speculative entries).

---

## Failure Category Taxonomy

| Category | Code | Description |
| --- | --- | --- |
| Data staleness | DS | Input data exceeded freshness threshold |
| Cache miss / corruption | CM | Cache file missing, truncated, or stale |
| Pipeline timeout | PT | Job exceeded time limit |
| Logic error | LE | Code produced wrong output from correct input |
| Doc-sync gap | DG | Documents contradicted each other or production state |
| Infrastructure failure | IF | Service unreachable, API down, credential expired |
| Governance lapse | GL | Process skipped, gate bypassed, approval missing |
| Naming collision | NC | Same concept referenced by different names across docs |
| Misattribution | MA | Metric or claim attributed to wrong source or scope |

---

## Failure Entry Schema

```
failure_id: F-YYYY-NNN
category: [code from taxonomy]
first_seen: YYYY-MM-DD
recurrence_count: N
severity: LOW | MEDIUM | HIGH | CRITICAL
summary: One-line description
root_cause: What actually went wrong
affected_systems: [list of skills/agents/pipelines]
resolution: What fixed it (or "UNRESOLVED")
prevention_rule: What should change to prevent recurrence
promotion_status: PENDING | PROMOTED(skill, date) | DECLINED(reason)
related_specs: [spec numbers if applicable]
related_findings: [C1, W3, G5 from doc review if applicable]
```

---

## Catalog

### F-2026-001 | NC | Selector Terminology Mismatch

- **First seen:** 2026-05-17 (doc review Run 1)
- **Recurrence:** 4+ documents affected
- **Severity:** HIGH
- **Summary:** "sponsorship_score_z" and "momentum_delta_z" used in external docs; production uses "coinvest_score_z" and "inst_delta_z" since v1.14.0 rename
- **Root cause:** Signal renaming in code did not propagate to all documentation layers
- **Affected systems:** selector-ranker, institutional-signal, all .docx exports
- **Resolution:** CON-1 cross-reference note added to selector-ranker skill. GitHub/docx not updated.
- **Prevention rule:** Any signal rename must include a doc-propagation checklist: skill -> GitHub model docs -> .docx exports -> agent prompts
- **Related findings:** C1 (doc review), W6 (RULESET_CHANGELOG lacks naming note)


- **Promotion status:** PROMOTED(coding-standards, 2026-06-06) — doc-propagation checklist for signal/schema renames encoded as a standing rule in `coding-standards` (Git Workflow). Note: actual back-propagation of the v1.14.0 rename to GitHub model docs and .docx exports remains a separate open remediation item.

### F-2026-002 | MA | IC Tooling Scope Conflation

- **First seen:** 2026-05-13 (Spec 095)
- **Recurrence:** Pervasive in all prior IC claims
- **Severity:** CRITICAL
- **Summary:** run_rank_ic_backtest.py measured composite_score IC, not production final_score IC. Composite_rank correlates only 0.25 with actionable_rank.
- **Root cause:** Tool was built to measure one signal; was silently assumed to measure the production signal
- **Affected systems:** ic-evaluation, selector-ranker, all historical IC claims
- **Resolution:** Spec 100 committed (2faa88e6, 2026-05-17). Prior claims invalidated. Checklist v2 battery rerun UNBLOCKED — architecture freeze LIFTED 2026-05-26; this is highest-priority post-freeze action.
- **Prevention rule:** Any IC measurement tool must explicitly declare which score field it measures in its output header. No IC claim is valid without matching the declared field to the production sort key.
- **Related findings:** Spec 095, Spec 100, G2


- **Promotion status:** PROMOTED(ic-evaluation, 2026-06-06) — IC-tool field-declaration rule (declare the measured score field + universe in tool output; no IC claim valid unless the declared field matches the production sort key) encoded as a standing rule in `ic-evaluation`. Note: the Checklist v2 battery rerun against final_score remains a separate open production action.

### F-2026-003 | DG | inst_delta_z Scope Misattribution

- **First seen:** 2026-05-16 (Code Review H3)
- **Recurrence:** 2 (selector-ranker skill + institutional-signal skill contradicted)
- **Severity:** HIGH
- **Summary:** selector-ranker stated inst_delta_z "excluded from ranker since Spec 051" when it was excluded from the SELECTOR, not the ranker. inst_delta_z remains active in the ranker (NW-t = +3.32).
- **Root cause:** Imprecise language in skill update -- "ranker" used when "selector" was meant
- **Affected systems:** selector-ranker, institutional-signal
- **Resolution:** Code Review H3 fix noted; skill text corrected
- **Prevention rule:** When describing signal status changes, always specify the exact engine layer (selector vs. ranker vs. composite vs. decision engine)
- **Related findings:** C2 (doc review)


- **Promotion status:** PROMOTED(selector-ranker / institutional-signal, 2026-05-16) — skill text corrected to specify engine layer.

### F-2026-004 | PT | AACT Pipeline Timeout

- **First seen:** Pre-May 2026 (recurring)
- **Recurrence:** Multiple (especially Mondays)
- **Severity:** MEDIUM
- **Summary:** 4500s timeout was killing the daily pipeline mid-AACT ingestion, particularly on Monday runs (weekend AACT batch)
- **Root cause:** Timeout set too aggressively for worst-case AACT ingestion duration
- **Affected systems:** screener-ops daily pipeline, catalyst-resolution
- **Resolution:** Timeout increased from 4500s to 6000s
- **Prevention rule:** Monday pipeline runs should have extended timeout or dedicated monitoring. Any timeout change should be validated against 4-week Monday duration distribution.


- **Promotion status:** PROMOTED(screener-ops, 2026-06-06) — Monday extended-timeout/monitoring rule and the trailing-4-week Monday-distribution validation requirement encoded as a standing rule in `screener-ops` (Pipeline Timeout).

### F-2026-005 | IF | Herald Digest Extended Outage

- **First seen:** ~2026-04-14 (estimated; ~10 weeks dark as of Jun 25)
- **Recurrence:** Ongoing (~10+ consecutive weeks as of Jun 25)
- **Severity:** CRITICAL
- **Summary:** Herald Digest has produced zero output for 7+ consecutive weeks. No deduped or classified JSONL files generated.
- **Root cause:** UNRESOLVED as of 2026-06-25 — CI budget exhaustion was primary blocker; code fixes merged but terminal verification not yet confirmed.
- **Affected systems:** Herald pipeline, press release monitoring, downstream news-driven signals
- **Resolution:** PARTIALLY RESOLVED — code fixes merged. Recovery target was June 1; status as of Jun 25 unconfirmed (verify terminal output).
- **Prevention rule:** Herald should have a max-dark-days SLA (proposed: 3 days) with automatic escalation. See operational-health-baselines skill.
- **Related findings:** G6, weekly signal counts memory


- **Promotion status:** PENDING — Herald max-dark-days SLA proposed in `operational-health-baselines` (DRAFT, not active); outage UNRESOLVED.

### F-2026-006 | IF | CI Pipeline Extended Red State

- **First seen:** ~2026-05-08
- **Recurrence:** Ongoing (~48 days as of Jun 25)
- **Severity:** HIGH
- **Summary:** CI has been red since approximately May 8. PR #285 remains open/unmerged. phase2-daily-production cron is dark.
- **Root cause:** Budget exhaustion (GitHub Actions). CI Diagnostic Report and CI Fix Checklist produced May 14-16; remediation not confirmed complete.
- **Affected systems:** All merge gates, production deployment confidence, Herald Digest restore
- **Resolution:** UNRESOLVED as of Jun 25. PR #285 open. Recovery target was June 1 (unconfirmed). Architecture freeze LIFTED 2026-05-26 — CI recovery is top post-freeze priority.
- **Prevention rule:** CI red > 5 days should trigger merge block and operator escalation. See operational-health-baselines skill.
- **Related findings:** C4 (highest operational risk, doc review)


- **Promotion status:** PENDING — CI>5d merge-block SLA proposed in `operational-health-baselines` (DRAFT, not active); outage UNRESOLVED.

### F-2026-007 | DG | Clinical Score Denominator Confusion

- **First seen:** 2026-05-16 (Code Review H1)
- **Recurrence:** 2 (clinical-scoring skill vs GitHub model docs)
- **Severity:** MEDIUM (RESOLVED)
- **Summary:** Clinical score denominator settled at 117 with execution base 12, max 22 (commit 3ad7b904, 2026-05-18). Earlier confusion arose when GitHub showed 117 and skill showed 120 during an intermediate state where execution base was being adjusted (12 to 15). Final resolution: denominator 117 is authoritative (clinical-scoring skill, L2). The intermediate 120 figure (execution base 15) was superseded.
- **Root cause:** Two fixes applied at different layers without cross-referencing each other; intermediate state (denom 120) briefly existed in the skill before final settlement at 117.
- **Affected systems:** clinical-scoring, GitHub model_documentation_root.md, document-lineage (authority table corrected Jun 22)
- **Resolution:** RESOLVED (C7, Run 4; document-lineage authority table corrected Jun 22). Denominator 117 is canonical.
- **Prevention rule:** When a fix touches a score denominator or weight at one layer, check all other layers that reference the same total.
- **Related findings:** C3/C7 (doc review)


- **Promotion status:** PROMOTED(clinical-scoring, 2026-05-16) — denominators documented internally consistent across layers; RESOLVED.

### F-2026-008 | NC | Agent Fleet Count Inconsistency

- **First seen:** 2026-05-17 (doc review Run 1)
- **Recurrence:** 5 different counts across documents (17, 26, 27, 28, 30)
- **Severity:** MEDIUM
- **Summary:** Agent count appears as 17, 26, 27, 28, and 30 across different documents
- **Root cause:** Agent count is a moving target as agents are added/suppressed/retired, and documents capture point-in-time counts without dating them
- **Affected systems:** All governance documents, compliance memos, exec overview
- **Resolution:** agent_governance.md (GitHub, 2026-05-17) designated as most authoritative (30 agents: 27 active + 1 shadow + 1 suppressed + 1 retired). Other docs remain stale.
- **Prevention rule:** Agent count should always be sourced from agent_governance.md with a dated citation. All other documents should say "see agent_governance.md for current count" rather than hardcoding a number.
- **Related findings:** C6/W3 (doc review)


- **Promotion status:** PROMOTED(hermes-runtime, 2026-06-06) — 'cite agent_governance.md with a dated reference, never hardcode the agent count' encoded as a standing rule in `hermes-runtime` (Agent Fleet Configuration). Note: cleanup of downstream docs that still hardcode a count remains a separate open remediation item.

### F-2026-009 | DG | coinvest_score_z Deployed vs. Trained Weight Discrepancy

- **First seen:** 2026-05-25 (Drift Run 8, N1)
- **Recurrence:** 1
- **Severity:** MEDIUM
- **Summary:** selector-ranker skill implied `coinvest_score_z` ranker weight = +0.0613 (trained basis). Deployed weight = +0.02 (capped Family C live pilot, per model_documentation.md v1.7.2). Direct conflict.
- **Root cause:** Skill referenced the artifact's trained coefficient without noting the live-pilot cap applied on top.
- **Affected systems:** selector-ranker skill (ranker feature roles section)
- **Resolution:** selector-ranker skill updated 2026-05-25 to document both values and distinguish deployed (+0.02) from trained (+0.0613).
- **Prevention rule:** Whenever a ranker weight has a live-pilot cap or override applied, both the trained value and the deployed value must be explicitly documented in the skill.
- **Related findings:** N1 (Drift Run 8)


- **Promotion status:** PROMOTED(selector-ranker, 2026-05-25) — deployed +0.02 vs trained +0.0613 both documented.



### F-2026-010 | NC | "DEM" Mislabeled as Baseline vs Current Ranker

- **First seen:** 2026-06-27 (DEM current-ranker YTD diagnostic)
- **Recurrence:** 1
- **Severity:** MEDIUM
- **Summary:** "DEM" was framed as a separate baseline algorithm competing with the A4 selector ("DEM vs A4"), when DEM is actually the stored production top-30 output of the current pipeline (`actionable_rank` = `final_score` rank = A4 selector + clinical_50 ranker).
- **Root cause:** A stored-production-output artifact and fresh-recompute variants (forward-filled inst_delta_z) were compared under labels implying two independent algorithms.
- **Affected systems:** selector-ranker, ic-evaluation, backtest comparison artifacts
- **Resolution:** Architecture verdict DEM_IS_CURRENT_RANKER confirmed across snapshots (2026-06-27). Decision recorded as D-2026-008 (decision-audit-trail).
- **Prevention rule:** When comparing selector/ranker arms, label each arm as either stored production output or fresh recomputation, and state the exact engine layer. Never present the production top-30 as a "baseline" competitor to its own selector overlay.
- **Promotion status:** PENDING
- **Related findings:** D-2026-008



### F-2026-011 | LE | test-trust-audit T2 Detector False Positives (pytest.fail + skip double-flag)

- **First seen:** 2026-07-14 (test-trust-audit first run, biotech-screener)
- **Recurrence:** 1
- **Severity:** MEDIUM
- **Summary:** The T2 ("no effective assertion") detector counted `pytest.fail()` / `self.fail()` tests as hollow and double-flagged skip/xfail-decorated placeholders as both T2 and T7, so the raw T2 = 92 overstated the real hollow-test backlog.
- **Root cause:** T2 recognized assertions only via `ast.Assert`, `assert*`-prefixed calls, and `pytest.raises`; it did not treat `pytest.fail`/`self.fail` as effective, and did not defer skip/xfail-decorated functions to T7 only.
- **Affected systems:** test-trust-audit skill, `tools/test_trust_audit.py`
- **Resolution:** Fixed in PR #496 (branch `chore/test-trust-audit-hygiene-2026-07-14`; UNMERGED pending CI-red root cause). Corrected auditor measured T2 92 → 73 (~7 `pytest.fail`/`self.fail` false positives no longer flagged; skip/xfail-decorated stubs reclassified to T7, which rose 3 → 14). The 19-finding drop was measurement correction, not test fixes.
- **Prevention rule:** A hollow-test detector must recognize every effective failure signal the framework provides — `assert`, `assert*` calls, `pytest.raises`, AND `pytest.fail`/`self.fail` — and must never double-classify one finding across mutually exclusive detectors (a skip-decorated stub is a T7 skip, not a T2 no-assert). Treat any raw detector count as a prior to be triaged, not a problem count.
- **Promotion status:** PROMOTED(test-trust-audit, 2026-07-15) — T4-triage guidance and the detector-precision caveat encoded as standing guidance in the `test-trust-audit` skill.
- **Related findings:** D-2026-009, D-2026-010; PR #496; "test-trust-audit — First Run" doc



---

## Usage Rules

1. Before investigating a new error, search this catalog by category and keywords first.
2. If a match exists and resolution is documented, apply the known fix before re-investigating.
3. If a match exists but resolution is UNRESOLVED, add recurrence count and any new diagnostic information.
4. New entries require confirmed root cause. Do not add speculative entries.
5. Entries with recurrence_count >= 3 should be considered for prevention rule promotion into the relevant skill document. Threshold is the canonical >= 3 occurrences defined in the `self-improving` skill (failure modes count all-time, not the 7-day behavioral window). When a prevention rule is promoted into a skill, set that entry's `promotion_status` to PROMOTED(skill, date).
