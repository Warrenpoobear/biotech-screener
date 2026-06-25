# Operator Host Hermes Skills Layout

Last updated: 2026-06-24 · Full skills + learnings sync workflow documented

This runbook separates **repo authority** from **operator runtime copies** so Hermes and Cursor do not drift silently.

## Four layers (repo — do not blur)

| Layer | Path | Edit here? |
|-------|------|------------|
| Editable source | `skills/<dir>/SKILL.md` | **Yes** (normal skills) |
| Long references | `skills/<dir>/REFERENCE.md` | **Yes** (dossier, excel, word) |
| Hermes mirror | `docs/hermes_skills/*.md` | **Only** Hermes-native docs (see below) |
| Fleet index | `docs/hermes_skills/_meta.json` | Via `tools/sync_hermes_skills.py --register-meta` |

**Hermes-native** (no `skills/` source): edit directly under `docs/hermes_skills/` — e.g. `town-operator-bridge.md`, `hermeslink-state-capture.md`, `openclaw-*-debug.md`, Path C runbooks.

**Hermes-authoritative:** `memory-steward` — mirror is not overwritten by sync; treat `docs/hermes_skills/memory-steward.md` as source of truth.

Each `_meta.json` entry includes `source_authority` (see audit) describing which path owns the body.

## Repo sync (always run after editing `skills/` or `.learnings/`)

```bash
python3 tools/audit_learnings.py
python3 tools/sync_hermes_skills.py --register-meta
python3 tools/audit_hermes_skills.py
git status -sb
```

Expected clean audit:

- **32** Hermes `.md` files (excluding `harvest_log.md`)
- **32** registered in `_meta.json`
- No unregistered files; no mirror drift warnings for cursor-synced skills

Commit mirror changes when sync updates files:

```bash
git add docs/hermes_skills skills docs/hermes_agents tools
git commit -m "docs(hermes): sync skills mirror and operator layout runbook"
```

## Operator runtime (WSL — outside git)

Hermes Gateway / CLI may read skill bodies from paths **not** tracked in this repo:

| Location | Typical use |
|----------|-------------|
| `~/.hermes/skills/` | Hermes skill catalog roots |
| `~/.hermes/skills/devops/` | DevOps / OpenClaw debug skills (cron, pipeline, scope audit) |

Repo backup example (optional): `.hermes/skills/devops/memory-steward.SKILL.md`

### Operational rule

1. Edit `skills/<dir>/SKILL.md` (or Hermes-native doc) in the repo.
2. Run `sync_hermes_skills.py` + `audit_hermes_skills.py`.
3. Commit repo mirror + `_meta.json`.
4. **Only then**, if Hermes on the host still reads `~/.hermes/skills/`, copy updated bodies to the operator runtime.

Do **not** assume `docs/hermes_skills/` and `~/.hermes/skills/` are identical without verification.

### Hermes Agent version update check

Hermes Agent v0.16.0 / v2026.6.5 is available upstream. This Cloud checkout
cannot verify or update the operator-host install, so run the non-mutating
preflight on WSL before changing runtime skills:

```bash
hermes version || hermes --version
hermes update --check
```

Before `hermes update`, back up `~/.hermes/config.yaml` and
`~/.hermes/skills/`; v0.16.0 has a known first-config-write risk that can
expand defaults and drop hand-curated provider blocks. Full checklist:
`docs/HERMES_GATEWAY_SETUP.md`.

### Copy to operator host (when needed)

From repo root on WSL (adjust paths if your clone differs):

```bash
# Example: refresh one devops skill after repo sync
cp docs/hermes_skills/openclaw-cron-scheduler-debug.md \
   ~/.hermes/skills/devops/openclaw-cron-scheduler-debug/SKILL.md

# memory-steward (authoritative in repo mirror)
cp docs/hermes_skills/memory-steward.md \
   ~/.hermes/skills/devops/memory-steward/SKILL.md
```

Prefer copying from `docs/hermes_skills/*.md` (post-sync), not from `skills/` directly, so Hermes-only sections (e.g. Path C in screener-ops) are preserved.

### Verification on operator host

```bash
hermes -s screener-ops "status"
hermes -s codegraph "status"
```

If Cursor shows updated guidance but Hermes CLI does not, check for **stale `~/.hermes/skills/`** copies before debugging sync logic.

## Rules

- Do **not** hand-edit mirrored files that `source_authority` marks as `skills/.../SKILL.md` or `skills/.../REFERENCE.md` — edit `skills/` and sync.
- Do **not** treat `docs/hermes_skills/` and `~/.hermes/skills/` as automatically identical.
- If behavior differs between Cursor and Hermes CLI, compare `source_authority` in `_meta.json` and runtime paths above.
- Agent **behavior** is not in this tree — use `agents/<name>/SOUL.md` and `AGENT_REGISTRY.json` (see `docs/hermes_agents/agent_roster.md`).

## Cursor skills knowledge (index)

| Need | Location |
|------|----------|
| Edit Cursor skill source | `skills/<dir>/SKILL.md` or `REFERENCE.md` |
| Hermes mirror (after sync) | `docs/hermes_skills/*.md` |
| Sync + register `_meta.json` | `python3 tools/sync_hermes_skills.py --register-meta` |
| Drift audit | `python3 tools/audit_hermes_skills.py` |
| Screener ops + fleet model routing | `skills/screener_ops/SKILL.md` → `screener-ops.md` (incl. repo scale table) |
| Codegraph in Cursor | `skills/codegraph/SKILL.md` → `codegraph.md` |
| Hermes MCP (IDE, read-only) | `mcp_server/hermes_server.py` · bootstrap: `.cursor/rules/hermes-context.mdc` |
| Model surfaces (gateway vs direct) | [`hermes_tools_map.md`](hermes_tools_map.md) §5 · [`HERMES_GATEWAY_SETUP.md`](../HERMES_GATEWAY_SETUP.md) |
| WSL acceptance gate (cron + B1b + gateway) | `skills/screener_ops/SKILL.md` → Host authority + acceptance gate sections |
| Fleet host onboarding (phases 2–15) | `bash tools/run_fleet_host_onboarding.sh` · `docs/AGENT_FLEET_ARCHITECTURE_INDEX.md` |
| Unified host setup (fleet + research) | `bash tools/run_operator_host_setup.sh` |
| Research host battery (Spec 100/105) | `bash tools/run_research_host_battery.sh` · `docs/research/CHECKLIST_V2_FINAL_SCORE_BLOCKER_2026_06_24.md` |
| Sync history | [`../hermes_skills/harvest_log.md`](../hermes_skills/harvest_log.md) |
| Recursive self-improvement loop | `skills/self-improving/SKILL.md` → `self-improving.md` · `REFERENCE.md` → `self-improving-reference.md` |
| Knowledge stack + audit | `.learnings/README.md` · `python3 tools/audit_learnings.py` |
| Skills learning telemetry | `tools/skills_execution_logger.py` · `tools/hermes_skills_learning_loop_v2.py` → `artifacts/skills_learning/` |
| Hermes taxonomy | [`hermes_tools_map.md`](hermes_tools_map.md) · [`agent_roster.md`](agent_roster.md) |

**Rule:** edit `skills/` first, then sync and commit mirrors. After significant sessions, run the self-improving loop (log → promote → skill-patch → sync → harvest_log). Do not hand-edit mirrored files unless `source_authority` is `HERMES_NATIVE` or `memory-steward` (authoritative).

## Related

- **Canonical tool taxonomy:** [`hermes_tools_map.md`](hermes_tools_map.md) (MCP vs repo tools vs Lane A jobs vs CLI)
- Sync implementation: `tools/sync_hermes_skills.py`
- Audit: `tools/audit_hermes_skills.py`
- Harvest / sync history: `docs/hermes_skills/harvest_log.md`
- Agent fleet (separate from skills): `docs/hermes_agents/agent_roster.md`
