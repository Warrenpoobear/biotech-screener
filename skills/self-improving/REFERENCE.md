# Self-Improving — Reference (biotech-screener)

Long-form templates and checklists. Entry point: `skills/self-improving/SKILL.md`.

---

## LRN entry template (`.learnings/LEARNINGS.md`)

```markdown
## [LRN-YYYYMMDD-XXX] short_snake_title

**Logged**: ISO-8601 timestamp
**Priority**: low | medium | high | critical
**Status**: pending | promoted | resolved | superseded
**Area**: tooling | ops | research | portfolio | ci | hermes_ops | cloud_environment | data_pipeline
**Promotion-lane**: skill | spec | none  # skill=patch skills/; spec=governance Spec only; none=log only

### Summary
One sentence.

### Details
What happened, root cause, blast radius.

### Suggested Action
Concrete next step (skill patch, test, operator check).

### Metadata
- Source: session_id or tool name
- Related Files: paths
- Tags: comma-separated
- Pattern-Key: snake_case (for recurrence counting)
- Recurrence-Count: N
- Promotion-lane: skill | spec | none
- Skill-Path: optional target skill dir (e.g. screener_ops, codegraph)
```

---

## Harvest log template (`docs/hermes_skills/harvest_log.md`)

```markdown
## YYYY-MM-DD (short title)

### Skill sources refreshed
- **skill-name** (`skills/<dir>/SKILL.md`): what changed
- Ran `sync_hermes_skills.py` — updated `mirror.md`

### Learnings promoted
- LRN-... → HOT `memory.md` | WARM `projects/biotech_screener.md`

### Governance
- Skills/docs only. No ranker, selector, sizing, or scoring changes.
```

---

## Patch verification template (2 weeks post-merge)

```markdown
### Patch verification (YYYY-MM-DD)
- **skill:** openclaw-cron-scheduler-debug Class J
- **metric:** cron_missed Town emails citing import errors / skill telemetry success rate
- **result:** 0 recurrence since fix / ≥80% success over N execs
- **action:** close LRN or bump Recurrence-Count if pattern recurred
```

---

## Weekly operator workflow

```bash
python3 tools/audit_learnings.py
python3 tools/weekly_skills_digest.py
SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py
# Review artifacts/skill_patch_drafts/ and artifacts/skills_learning/weekly_digest_*.md
```

---

## Session-end audit (5 minutes)

```bash
python3 tools/audit_learnings.py    # tier limits, promotion candidates, stale hints
```

1. Any user correction? → `corrections.md` + consider `LEARNINGS.md`
2. Recurrence-Count ≥ 3 for same Pattern-Key? → promote tier (`memory.md` / `projects/` / `domains/`)
3. Operator workflow or tooling lesson? → candidate for `skills/*/SKILL.md`
4. If skill edited: `sync_hermes_skills.py` + `audit_hermes_skills.py`
5. If skill committed: append `harvest_log.md` section

Full stack: `.learnings/README.md`

---

## Skill-patch review checklist

Before merging a learning-driven skill PR:

- [ ] Change is **docs/skills only** OR separate code PR with tests
- [ ] No ranker/selector/sizing/final_score/decision_engine/KG production touch
- [ ] No cron or gateway config in repo (operator host only)
- [ ] Mirror drift audit clean
- [ ] LEARNINGS entry or harvest_log cites the pattern
- [ ] Stale mirror sections removed (no duplicate gates)

---

## Anti-patterns (do not automate)

- Promoting research signal findings into production weights without Spec + CRT+IC+PIT
- Writing hedge/cron truth into skills without WSL verification
- Expanding SKILL.md with full runbooks (use REFERENCE.md or Hermes-native docs)
- Inferring user preferences from silence
- Deleting `.learnings/` entries without operator confirmation

---

## Related skills

| Skill | Recursive role |
| --- | --- |
| `screener_ops` | Ops gates, knowledge layer, WSL authority |
| `codegraph` | Structural preflight, bounded proof |
| `openclaw-agent-optimize` | Context/skill bloat → compaction candidates |
| `memory_steward` | Tier demotion, archive (read-only default) |
| `selector_ranker` | Tier 3 preflight — learnings do not override BLOCKED |
