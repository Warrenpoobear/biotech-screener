---
name: self-improving
triggers:
  - learning capture
  - correction logging
  - memory promotion
  - self-reflection
  - preference storage
  - skill improvement
  - memory tiering
description: >
  Structured learning capture with promotion and demotion rules for persistent
  improvement. Logs corrections, preferences, and self-reflections. Tiered
  storage (HOT/WARM/COLD) for Town and Hermes environments. Never infers
  preferences from silence; confirms patterns after 3 occurrences.
---

# Self-Improving Skill

Structured learning capture with promotion, demotion, and **repo skill recursion** for persistent improvement.

## When to Use

- User corrects you or points out mistakes
- You complete significant work and want to evaluate the outcome
- You notice a repeatable pattern in your own output
- A session discovered operator workflow, tooling, or convention knowledge that should survive the next agent

## Recursive improvement loop (biotech-screener)

Knowledge compounds in **four stores**, then feeds back into **skills**:

```
Observe → Log → Distill → Promote → Skill-patch → Sync → Verify → (next session)
```

| Step | Action | Store |
| --- | --- | --- |
| **Observe** | Correction, CI surprise, ops gate miss, codegraph gap | Session context |
| **Log** | Raw event | `.learnings/corrections.md` or `.learnings/LEARNINGS.md` `[LRN-...]` |
| **Distill** | 3x same Pattern-Key in 7 days | `.learnings/memory.md` bootstrap block (HOT) or `projects/` / `domains/` (WARM) |
| **Promote** | Cross-session, operator-relevant workflow | `skills/<dir>/SKILL.md` |
| **Sync** | Hermes mirror + registry | `python3 tools/sync_hermes_skills.py` |
| **Verify** | No drift | `python3 tools/audit_hermes_skills.py` |
| **Record** | Human-readable history | `docs/hermes_skills/harvest_log.md` + git commit |

**Rule:** learnings improve **how agents work**; they do not change production scoring without governance Spec.

Long templates: `skills/self-improving/REFERENCE.md`.

### What may become a skill patch

| Eligible | Ineligible without Spec |
| --- | --- |
| CodeGraph MCP vs CLI naming | Ranker/selector/sizing weights |
| WSL vs Cloud authority | `decision_engine`, `final_score` behavior |
| CI budget vs code failure | Snapshot schema / promotion semantics |
| Sync/harvest workflow | Cron lines or gateway `config.yaml` |
| Test contract / flake patterns | Production KG wiring |

Default skill targets: `screener_ops`, `codegraph`, `openclaw-agent-optimize`, `self-improving` (meta). Tier 3: add preflight notes to `selector_ranker` only — never relax BLOCKED gates.

### Session-end trigger (significant work)

1. One-line reflection: met expectations? repeatable?
2. If correction → append `corrections.md`
3. If Pattern-Key recurrence ≥ 3 → new `LEARNINGS.md` entry or bump Recurrence-Count
4. If promoted to HOT/WARM → update `memory.md` or `projects/biotech_screener.md` (stay within line limits)
5. If skill-worthy → patch `skills/`, sync, audit, `harvest_log.md`, commit

---

## Memory Architecture

### Town Environment (Primary)

Use Town's native memory system with tiered importance:

| Tier | Storage | Purpose |
|------|---------|---------|
| HOT (critical) | `add_memory()` global | Always-active patterns affecting all routines |
| WARM (routine-specific) | `add_memory(routine_slug=...)` | Per-routine learned behaviors |
| COLD (archived) | Delete from active; note in user profile if historically significant | Decayed patterns no longer needed |

### Hermes / Cursor Environment (this repo)

File-based storage in `.learnings/`:

| Tier | File | Limit | Purpose |
|------|------|-------|---------|
| HOT | `memory.md` | <=100 lines | **Bootstrap block first** — recursion, governance, Pattern-Keys; detail in domains/ |
| WARM | `projects/biotech_screener.md` | <=200 lines | Per-project learnings |
| WARM | `domains/{name}.md` | <=200 lines | Domain-specific patterns |
| COLD | `archive/` | Unlimited | Decayed patterns |
| LOG | `corrections.md` | Last 50 | Raw correction log |
| INDEX | `LEARNINGS.md` | Growing | Structured LRN entries with Pattern-Key + Recurrence-Count |

Repo skills (loaded by Cursor/Hermes) are the **executable** layer above `.learnings/`.

---

## Learning Signals

### Log immediately
- **Corrections**: "No, that's not right...", "Actually, it should be...", "You're wrong about...", "I prefer X, not Y", "Stop doing X"
- **Preferences**: "I like when you...", "Always do X for me", "Never do Y"

### Track and promote after 3x
- **Patterns**: Recurring mistakes, recurring successes, recurring workarounds (use Pattern-Key in LEARNINGS.md)

### Ignore
- One-time instructions
- Context-specific guidance
- Hypotheticals

---

## Self-Reflection Framework

After completing significant work:
1. Did it meet expectations?
2. What could be better?
3. Is this a pattern (Pattern-Key)?

Log format:
```
CONTEXT: [task type]
REFLECTION: [what I noticed]
LESSON: [what to do differently]
PATTERN-KEY: [snake_case]  # optional, for recurrence
SKILL-CANDIDATE: [screener_ops|codegraph|none]
```

---

## Core Rules

### Rule 1 - Learning
Log explicit corrections and self-identified improvements. Never infer from silence. Confirm patterns after 3 identical lessons.

### Rule 2 - Tiered Storage
- **HOT**: Critical patterns, active preferences (Town: global memories; Hermes: memory.md)
- **WARM**: Per-routine or per-project (Town: routine-scoped memories; Hermes: projects/*.md)
- **COLD**: Archived, decayed (Town: deleted with note; Hermes: archive/)

### Rule 3 - Promotion / Demotion
- **Pattern-Key ≥3** (canonical bar — shared with Town; do not fork): behavioral patterns within a **rolling 7-day window**; **failure modes** count **all-time**. See Rule 12.
- Unused 30 days → demote to WARM
- Unused 90 days → archive or delete (ask first)

### Rule 4 - Namespace Isolation
- Town: Use `routine_slug` parameter for routine-specific memories; omit for global
- Hermes: Projects in `projects/{name}.md`, global in `memory.md`, domains in `domains/{name}.md`

### Rule 5 - Conflict Resolution
Most specific wins: routine-specific > domain > global. Most recent wins at same level. **SOUL.md / runtime cron beat stale skill text** — refresh skills when execution truth diverges.

### Rule 6 - Compaction
Merge similar corrections. Archive unused patterns to `.learnings/archive/`. Never delete without asking. HOT `memory.md`: bootstrap table + tagged Pattern-Keys; demote prose to `domains/` or `projects/`. Run `audit_learnings.py` for compaction hints. Prefer **short SKILL.md + REFERENCE.md** over bloated skills (see `openclaw-agent-optimize`).

### Rule 7 - Transparency
When applying a learned pattern, mention it briefly. Offer periodic digests. Full export on demand.

### Rule 8 - Security
Never store credentials, health data, or third-party confidential information in memories.

### Rule 9 - Graceful Degradation
Town: Global memories load first. Routine-specific memories load per-session.
Hermes: Load `memory.md` first. Load namespaces on demand.

### Rule 10 - Skill recursion governance
Architecture freeze: skill updates are **Tier 0 docs/plumbing** unless they encode scoring or selector behavior. When in doubt, log to LEARNINGS as `pending` and stop.

### Rule 11 - Selfimprove containment (FENCE)

Automated skill promotion is **staged**, not live by default (2026-06-24):

| Gate | Env var | Effect |
| --- | --- | --- |
| Immediate verdict | `SELFIMPROVE_IMMEDIATE_VERDICT=1` | Enables `record_feedback()` in `run_agent_direct.py` |
| Skill-patch drafts | `SELFIMPROVE_GATES_MET=1` | Enables `pattern_to_skillpatch.py` (writes drafts only; never edits skills directly) |

Without these gates, tools exit cleanly. Operator reviews drafts before any skill merge. Memo: `artifacts/governance/selfimprove_audit_2026-06-24.md`.

### Rule 12 — Promotion checklist (mechanical bar before a lesson becomes a skill patch)

A lesson is *captured* the moment it's logged; it is *promoted* only when it clears this checklist. The bar is shared with Town — do **not** define a parallel threshold here. The canonical definition lives in this skill's `>= 3` rule (Rule 3): failure modes count all-time; behavioral patterns count within a rolling 7-day window. Forking the threshold is itself a definition-drift failure mode (F-2026-001 class).

**Candidate feed (where promotion candidates come from):**
- **Hermes side:** `.learnings/LEARNINGS.md` entries with a `Pattern-Key`, and `failure-patterns` entries with `recurrence_count >= 3` and `promotion_status: PENDING`.
- **Town side:** the Town Correction Ledger (`content://collections/self-improvement/correction-ledger`), `recurrence_count >= 3` rows. These are already-counted and deterministic — use them as the feed rather than re-deciding promotability from ad-hoc chat corrections.

**Promotion gates (a candidate must clear the matching row before its Action):**

| Gate | Threshold | Action |
| --- | --- | --- |
| Recurrence | Pattern-Key `>= 3` (7-day window for behavioral; all-time for failure modes) | Promote to `memory.md` HOT or `domains/` |
| Skill-path + recurrence | LRN has a `Skill-Path` AND recurrence `>= 2` | Draft a patch (do NOT auto-merge) |
| Operator verdict | `>= 3` "helpful" verdicts on the same skill (requires telemetry) | Eligible for skill merge |
| Observation period | `7+` days of true-PIT production telemetry | Eligible for routing/behavior changes |

**Lane gate (refuse the wrong lane):**
- Every promotable entry carries `Area:` (`hermes_ops` | `data_pipeline` | `research` | `portfolio`) and `Promotion-lane:` (`skill` | `spec` | `none`).
- `pattern_to_skillpatch.py` MUST refuse any entry with `Promotion-lane: spec`. Signal/scoring/research findings (size confound, portfolio drag, ranker weights) go to `projects/biotech_screener.md` and a governance Spec — never silently into a production skill. This preserves Rule 10.

**Promote = propose, never auto-apply (Rule 11 FENCE still binds):**
1. `SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py --min-recurrence 3 --out artifacts/skill_patch_drafts`
2. Operator reviews drafts in `artifacts/skill_patch_drafts/`, hand-edits `skills/<dir>/SKILL.md`.
3. `python3 tools/sync_hermes_skills.py` → `python3 tools/audit_hermes_skills.py` (expect 32/32 registered — 40 Town skills minus 8 framework skills with no Hermes mirror; not drift).
4. Append a `harvest_log.md` entry recording the merge.

**Efficacy back-check (closes the loop — a patch isn't "done" when merged, it's done when it sticks):**
- Two weeks post-merge, add a `harvest_log.md` verification block: skill, the metric watched, and recurrence result (e.g. "0 recurrence of cron_missed import errors since 2026-06-24").
- If the pattern recurs: bump `Recurrence-Count`, set `promotion_status` back to PENDING, and escalate — the patch was incomplete.
- Efficacy tracking on an *outage* fix cannot begin until that outage's recovery is confirmed (you cannot measure "0 recurrence since fix" on an unconfirmed fix). Stalled-loop entries (e.g. F-2026-005 Herald, F-2026-006 CI) block their own efficacy check until RESOLVED.

Set `Promotion-lane` on every new LRN: `skill` | `spec` | `none`. See `REFERENCE.md` template.

---

## Repo commands (skill + knowledge recursion)

```bash
# Knowledge hygiene (read-only)
python3 tools/audit_learnings.py

# Skill-patch drafts (operator review required — Rule 12)
SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py --min-recurrence 3 --out artifacts/skill_patch_drafts

# After editing skills/<dir>/SKILL.md
python3 tools/sync_hermes_skills.py
python3 tools/audit_hermes_skills.py

# Ops ledgers (operator WSL authoritative for cron)
python3 tools/build_hermes_knowledge_layer.py
```

Knowledge stack map: `.learnings/README.md` · Runbook: `docs/hermes_agents/operator_host_skills.md` · History: `docs/hermes_skills/harvest_log.md`

---

## Town → Hermes learning bridge

Town is the primary memory surface; Hermes/Cursor uses `.learnings/` files. **Dual-write** significant corrections:

1. **Town:** `add_memory()` or Town Correction Ledger (`content://collections/self-improvement/correction-ledger`)
2. **Hermes:** append `[LRN-...]` to `LEARNINGS.md` with `Pattern-Key`, `Area`, `Promotion-lane`
3. Run `audit_learnings.py` — promotion candidates must match Rule 12 feeds (do not re-count from chat)
4. If skill-worthy and `Promotion-lane: skill` → Rule 12 propose path → sync → `harvest_log.md` → commit

Use `memory-steward` to flag stale Town memories that contradict repo skills (`town_ls skills://`).

---

## Town-Specific Actions

### Logging a correction
```
add_memory(content="When drafting emails for Darren, never use exclamation points in professional contexts - only in genuinely appreciative notes.")
```

### Logging a routine-specific learning
```
add_memory(routine_slug="town-morning-briefing", content="Include biotech catalyst calendar items in the morning briefing even when no price movement has occurred.")
```

### Reviewing current learnings
```
get_memories()  # global
get_memories(routine_slug="town-morning-briefing")  # routine-specific
```

### Archiving a stale memory
```
delete_memory(memory_id="...")  # after confirming it's no longer relevant
```

---

## Scope

**ONLY**: Learns from corrections and self-reflection; stores preferences via Town memories or Hermes files; reads memory state; proposes and implements **docs/skills** recursion when eligible.

**NEVER**: Accesses calendar/email/contacts for learning purposes; makes network requests for learning; infers preferences from silence; deletes memories without asking; changes production scoring from learnings alone.
