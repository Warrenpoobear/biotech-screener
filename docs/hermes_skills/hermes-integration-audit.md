---
name: hermes-integration-audit
description: "Read-only end-to-end health check of the Hermes integration surface for the biotech screener: Knowledge Layer, Town Bridge, Gateway, routing policy, skills/agents drift, Telegram handler, alerts. Encodes concrete bug classes found in prior audits so they re-surface on rerun."
when_to_use: "User asks 'check Hermes', 'audit the Hermes integration', 'is Hermes healthy', or before activating a new Hermes phase (e.g., Spec 090 Phase B). Also run after any merge that touches build_hermes_knowledge_layer.py, common/operator_delivery.py, common/alerts.py, tools/telegram_command_handler.py, docs/hermes_agents/, docs/hermes_skills/, or .claude/settings.json."
---

# Hermes Integration Audit

Read-only sweep of the Hermes/OpenClaw integration surface. Produces a punch list,
does NOT modify anything by default.

Covers the biotech screener at `/mnt/c/Projects/biotech_screener/biotech-screener`
(and any fresh clone — most checks are repo-state, not runtime).

## Hard rules

- Read-only by default. Never delete, rename, or edit until the operator approves.
- Distinguish repo-state checks (always valid on a fresh clone) from runtime checks
  (need the live host — `artifacts/ops/*`, `.env`, `~/.hermes/`, cron, processes).
- Report absences on a fresh clone as "expected absent", not as bugs.
- Spot-check every claim. Subagent reports describe intent, not outcome.

---

## Audit areas (six-pass sweep)

### 1. Knowledge Layer (Spec 089)

**Files:** `tools/build_hermes_knowledge_layer.py`, `specs/changes/spec_089_hermes_knowledge_layer.md`,
`artifacts/ops/{knowledge_layer,first_fire_ledger,contradiction_ledger,held_spec_ledger}/`.

Confirm: four output ledgers wired, contradiction scans live, FIRST_FIRE_SEED + HELD_ITEMS_SEED
present. On the live host, check artifact mtimes — stale ledgers indicate the
builder stopped firing.

### 2. Town-Hermes Bridge (Spec 090)

**Files:** `common/operator_delivery.py`, `common/alert_email.py`, `tools/smoke_operator_delivery.py`,
`specs/changes/spec_090_town_hermes_bridge.md`, `tests/test_operator_delivery.py`.

Confirm Phase A coded, `OPERATOR_DELIVERY_DRY_RUN` default is `"1"`, SMTP module
importable. For Phase B status: grep for callers of `send_operator_event(channel="town", ...)` outside
tests/smoke — if none, the bridge has no live producer (Phase B blocked regardless
of env config).

### 3. Hermes Gateway

**Files:** `docs/HERMES_GATEWAY_SETUP.md`. Port 8642 loopback, systemd user service,
auth via `~/.hermes/config.yaml` (NOT env var alone — systemd doesn't see operator env).

### 4. Routing policy

**Files:** `docs/ops/hermes_openclaw_routing_policy.md`, `governance/AGENT_ROUTING_POLICY.md`.

Lane A (deterministic, no LLM) / Lane B (cheap monitoring, LLM escalation via
`run_agent_direct.py`) / Lane C (manual). Tier 0-4 governance overlay. No cron may
depend on a gateway token.

### 5. Skills + agent roster

**Files:** `docs/hermes_agents/agent_roster.md`, `docs/hermes_skills/`, `.hermes/skills/`,
`docs/hermes_skills/harvest_log.md`, `agents/AGENT_REGISTRY.json`.

Cross-check skills referenced by recurring cron jobs against files in
`docs/hermes_skills/`. Anything only present in `~/.hermes/skills/` is unrecoverable
if that host directory is lost.

### 6. Telegram + outbound alerts

**Files:** `tools/telegram_command_handler.py`, `common/alerts.py`, `tests/test_telegram_command_handler.py`,
`docs/TELEGRAM_COMMAND_HANDLER.md`. Long-polling daemon, PID guard, chat-ID auth,
dedupe via `common/alert_dedupe.py`.

---

## Bug class taxonomy

These classes have been confirmed in production audits. Look for each on every run.

### Class A — Lex-sort of `iterdir()` poisoned by non-date sibling

**Signature:** A function meant to pick "the latest dated subdir" picks a non-date
sibling instead (e.g., `data/snapshots/resolutions/`). Affects telegram `/status`,
telegram `/snap`, any digest builder that does the same.

**Confirmed instance:** `tools/telegram_command_handler.py` `:251` and `:335`
(commit `dde2ae9`, 2026-05-18). `'r'` (0x72) lex-sorts after `'2'` (0x32) so
`resolutions/` always won `sorted()[-1]`.

**Diagnostic recipe:**

```bash
grep -rn 'sorted.*iterdir.*is_dir' --include='*.py' tools/ common/ | head
# For each hit, check the parent dir for non-date subdirs
ls -d data/snapshots/*/ | grep -v '^data/snapshots/[0-9]'
```

**Fix pattern:**

```python
import re
SNAPSHOT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
def _list_snapshot_dates(snap_dir):
    if not snap_dir.exists():
        return []
    return sorted(
        d.name for d in snap_dir.iterdir()
        if d.is_dir() and SNAPSHOT_DATE_RE.match(d.name)
    )
```

### Class B — Dual-field (seed `status` + runtime `eval`) violates spec

**Signature:** A producer copies a seed dict, then writes a runtime evaluation
to a NEW field instead of overwriting the seed field. On-disk artifact carries
both keys; consumers degrade to `.get("new_field", item["old_field"])`.

**Confirmed instance:** `check_first_fire_status()` in `tools/build_hermes_knowledge_layer.py`
emitted both `status` (stale seed) and `eval` (live). Spec 089 documents a single
`status` field. Three internal readers had to use the fallback (`:541`, `:682`,
`:831`). Fixed in commit `37d9997`, 2026-05-18.

**Diagnostic recipe:**

```bash
# Find producers that copy a seed dict then add new keys
grep -nE 'result\s*=\s*dict\(' --include='*.py' tools/ common/ | head
# Cross-reference to the spec's documented schema
```

**Fix pattern:** overwrite the spec-documented field name in place. If a "seed
vs live" trace is needed, copy to a clearly historical name like `status_at_seed`.

### Class C — Broken hook reference after `.deprecated` rename

**Signature:** `.claude/settings.json` (or crontab, or workflow YAML) references
a script that has been renamed `*.deprecated`. Hook fails silently — Claude Code
does not surface missing-command errors at hook execution time.

**Confirmed instance:** `UserPromptSubmit` hook in `.claude/settings.json:9`
pointed to `./skills/self-improvement/scripts/activator.sh`, but the file was
`activator.sh.deprecated`. Fixed in commit `ec08489`, 2026-05-18.

**Diagnostic recipe:**

```bash
# For each .deprecated file, search the repo for refs to the un-suffixed name
find . -name '*.deprecated' -not -path './.git/*' | while read f; do
  base=$(basename "$f" .deprecated)
  echo "--- refs to $base ---"
  grep -rn "$base" .claude/ .github/ crontab* 2>/dev/null | grep -v "\.deprecated"
done
```

**Fix pattern:** when deprecating a hook script, ALWAYS update or remove the
referencing config in the same commit. Or delete entirely if the deprecation is
permanent.

### Class D — Phase B prerequisite missing from agent roster

**Signature:** A spec defines a Phase B trigger job (e.g., `hermes-held-spec-ledger`
for Spec 090) that does not exist in `docs/hermes_agents/agent_roster.md`. The
spec stays in Phase A indefinitely because no one tracks the prereq.

**Confirmed instance:** `hermes-held-spec-ledger` job absent from roster as of
2026-05-18 (placeholder added in commit `86cedda`).

**Diagnostic recipe:**

```bash
# Grep specs for Phase B job names
grep -nE 'Phase B.*hermes-' specs/changes/*.md
# Cross-check against roster
grep -E '^###' docs/hermes_agents/agent_roster.md
```

**Fix pattern:** add a "Planned (not yet created)" section to the roster as soon
as a Phase B job is named in a spec, even before it's wired. Visibility prevents
silent omission.

### Class E — Production skill files unprotected by git

**Signature:** Agents in `docs/hermes_agents/agent_roster.md` reference skills
that exist ONLY in `~/.hermes/skills/` on the host, with no mirror under
`docs/hermes_skills/`. If the host directory is lost, the skill is unrecoverable.

**Confirmed instance:** `biotech-screener-audit`, `openclaw-fleet-triage`,
`aa-model-tracker`, `signal-shared-regime-check` lack repo mirrors as of 2026-05-18.

**Diagnostic recipe:**

```bash
# Skill names referenced by recurring jobs
grep -hE '^\| ' docs/hermes_agents/agent_roster.md | awk -F'|' '{print $3}' | sort -u
# Cross-check against repo-tracked skills
ls docs/hermes_skills/*.md | xargs -n1 basename | sed 's/\.md$//'
```

**Fix pattern:** include the production-job skills in the daily harvester's sync
list, not just the debug skills.

### Class F — Duplicate or nested-self skill files

**Signature:** A skill harvester or a manual sync creates nested copies, e.g.,
`docs/hermes_skills/references/references/ir_url_population.md` (same file,
nested folder).

**Confirmed instance:** removed in commit `86cedda`, 2026-05-18.

**Diagnostic recipe:**

```bash
find docs/hermes_skills/ -name '*.md' -printf '%s %p\n' | sort | uniq -d -w 12
# Same-size files at different paths are duplicate candidates
diff -q docs/hermes_skills/references/{,references/}ir_url_population.md 2>/dev/null
```

---

## Subagent: hermes-operator

The `.claude/agents/hermes-operator.md` subagent is read-only and well-suited to
this audit. Prompt it with the six audit areas above and the bug class taxonomy.
Spot-check at least two of its findings before reporting them upstream — the
agent describes its intent, not necessarily what it did.

## Reporting

End every audit with a punch list of actionable issues, grouped by:

- **Done in repo** — fixes that landed in this audit's branch
- **Operator decision required** — broken-but-ambiguous state (e.g., the
  `.deprecated` hook: delete-and-remove-hook vs. restore-the-script)
- **Host-side only** — items the auditor cannot touch from a remote clone
  (`.env`, `~/.hermes/`, cron, running daemons, Town routine setup)
- **Out of scope** — large categories the audit should not unilaterally clean
  (e.g., 166 top-level `.py` orphan candidates — requires dedicated import-graph
  analysis)

## Known-stable surfaces (do not re-audit deeply)

- `common/alerts.py` Telegram outbound — dedupe + rate-limit verified
- `common/alert_email.py` SMTP — minimal, soft-fails, well-tested
- Routing policy lanes A/B/C — aligned with Tier 0-4 governance, no conflicts
- Gateway loopback binding — by design, do not propose exposing publicly

## Session reference

This skill was created from the 2026-05-18 → 2026-05-19 audit session.
Commits: `dde2ae9`, `86cedda`, `37d9997`, `ec08489` on
`claude/check-hermes-telegram-dHe8a`.
