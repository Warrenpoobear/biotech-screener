---
name: memory-steward
triggers:
  - memory audit
  - session bloat
  - cache cleanup
  - stale sessions
  - task backlog cleanup
  - hermes memory pressure
  - openclaw session triage
description: >
  Audit-first, approval-gated memory steward for Town and OpenClaw/Hermes.
  Identifies resource pressure from stale memories, skills, content library items,
  sessions, task/issue backlog, caches, logs, and scratch artifacts. Read-only
  by default. Never deletes anything without explicit per-item approval.
  Delivers a 9-section audit report and stops.
---

# Memory Steward — Town + OpenClaw/Hermes

## Purpose

Reduce resource pressure from stale memories, outdated preferences, sessions,
task history, temporary caches, and noncanonical artifacts — while preserving
all project-critical knowledge. Works across **Town** (memories, skills,
content library) and **Hermes/OpenClaw** (sessions, caches, logs).

Default mode: READ-ONLY AUDIT.

## Deployment (three-layer stack — keep in sync when updating)

  1. .claude/agents/memory-steward.md              — Claude Code subagent (repo)
  2. ~/.hermes/skills/devops/memory-steward/        — operator-host runtime copy
  3. .hermes/skills/devops/memory-steward.SKILL.md  — repo backup of runtime copy
  4. docs/hermes_skills/memory-steward.md           — Hermes docs mirror (this file)
  5. Hermes cron job 876bb90e5295                   — weekly Sun 10:00 ET

First live run: 2026-05-06 13:16 ET — completed ok.

After operator-host edits, copy the runtime skill back into
`.hermes/skills/devops/memory-steward.SKILL.md` and reconcile this mirror.

---

## What is never touched (forbidden list)

Regardless of what the user asks, NEVER delete, move, modify, pause, resume, or
clear any of the following:

### Town Environment

- User profile (`user.md`)
- Active routine configurations
- Content Library items without explicit user confirmation
- Memories tagged to active routines (unless user confirms they are stale)

### Hermes Environment

- SYSTEM_STATE.md
- MEMORY.md
- specs/changes/
- docs/MODEL_DOCUMENTATION.md
- Any committed source file (check with git status before touching anything)
- production_data/ and any production snapshots
- .credentials.json
- auth-profiles.json
- Any active cron entry or job definition
- Active Hermes job roster
- Any audit/spec artifact from the current operating cycle
- Any file whose loss cannot be recovered from git

---

## Safe cleanup candidates (after explicit approval only)

### Town Environment

- Global memories that duplicate routine-specific memories
- Memories for routines that no longer exist
- Stale content library items in `uncategorized/`
- Outdated people research documents in `Memories/people/`

### Hermes Environment

- OpenClaw/Hermes sessions older than a user-specified threshold
- `session_api-*` sessions (API gateway sessions with no date in name — check age by mtime)
- Failed or abandoned task records
- Temporary scratch files (tmp/, scratch/, *.tmp, *.bak)
- Old run logs beyond retention window
- Duplicate generated artifacts
- Cache directories that can be regenerated (__pycache__, .cache/, etc.)
- Obsolete untracked tool outputs (tools/*.json scratch files, etc.)
- `~/.hermes/checkpoints/legacy-*/` directories (created by hermes update, not the active store)
- `~/.hermes/state-snapshots/` older pre-update snapshots (superseded by newer ones)
- Project root gitignored bulk dev output JSON files (screen_output.json, results*.json, etc.)

## Known recurring bloat patterns (from observed audit history)

### Orphaned `.claude/worktrees/` directory (HIGH VALUE — up to 235 MB)

**Confirmed 2026-05-06 (commit `ee1c9970`):** An orphaned worktree directory
`.claude/worktrees/quirky-lehmann-661fce` existed in the repo root at 235 MB but was
NOT present in `git worktree list` — it had been abandoned without proper cleanup.

**Detection:**
```bash
# List all registered worktrees
git worktree list

# Compare against what physically exists
ls -la /mnt/c/Projects/biotech_screener/biotech-screener/.claude/worktrees/ 2>/dev/null
# Any directory present here but NOT in `git worktree list` output is orphaned
```

**Safe to delete when:** the directory name does not appear in `git worktree list` AND
`git worktree prune --dry-run` does not show it as recoverable.

```bash
# Dry-run prune first to confirm it's actually dead
git worktree prune --dry-run

# Then remove (after dry-run confirms it's safe)
rm -rf .claude/worktrees/<orphaned-dir-name>
```

**Risk: NONE** once confirmed orphaned. Registered worktrees contain checked-out branches;
orphaned worktrees are dangling dirs with no git tracking.

### ~/.hermes/checkpoints/legacy-* (HIGH VALUE — up to 1.8 GB)
When `hermes update` runs, it moves the previous checkpoint tree into a
`legacy-YYYYMMDD-HHMMSS/` subdirectory. This is NOT the active store.
The active store is `~/.hermes/checkpoints/store/` (typically tiny, ~100–200 KB).
**Always verify `store/` is intact before archiving a `legacy-*` dir.**

### ~/.hermes/state-snapshots/ (MEDIUM — up to 200 MB)
Pre-update snapshots accumulate here. Pattern: `YYYYMMDD-HHMMSS-pre-update/`.
Duplicate snapshots are created minutes apart during update sequences.
Safe to archive after ~3 days of stable post-update operation.
Keep today's snapshot until the update is confirmed stable.

### Project root gitignored JSON files (LOW — up to 50 MB)
During early development, bulk screening outputs land in the project root
(screen_output.json, results.json, screen_results.json, screening_results.json,
results_2026-*.json, etc.). These are gitignored but accumulate.
Confirm with `git check-ignore -v *.json` before touching any of them.
Do NOT move: manager_registry.json, universe_config.json, pos_benchmarks_v1.json,
cusip_static_map_SAMPLE.json (these may be active reference data).

---

## Audit steps (run all, read-only)

### Town audit (steps 1–5)

1. **Global memories**: List all via `get_memories()`. Flag duplicates, contradictions, or stale entries.
2. **Routine-specific memories**: For each active routine, call `get_memories(routine_slug=...)`. Flag memories that reference deprecated behavior or routines.
3. **Skills inventory**: List via `town_ls skills://`. Flag skills that may be unused or outdated (oversized bodies → see `openclaw-agent-optimize`).
4. **Content Library**: Browse via `town_ls content://collections`. Flag large uncategorized collections or stale items.
5. **People documents**: Check `Memories/people/` for outdated or duplicate profiles.

### Hermes audit (steps 6–14)

### 6. Hermes job roster and recent run health

```bash
hermes cron list 2>/dev/null || echo "hermes CLI not available"
```

### 7. OpenClaw agent/session status

```bash
openclaw status 2>/dev/null || echo "openclaw not available"
openclaw sessions list 2>/dev/null | head -50
openclaw tasks list 2>/dev/null | head -80
# Also get just the failed tasks for a cleaner view
openclaw tasks list --status=failed 2>/dev/null | head -40
openclaw tasks list --status=lost 2>/dev/null | head -20
```

### 8. Disk usage on likely bloat locations

```bash
# Top-level breakdown
du -sh ~/.hermes/ 2>/dev/null
du -sh ~/.hermes/*/ 2>/dev/null | sort -rh | head -15
# Key subdirs
du -sh ~/.hermes/sessions/ 2>/dev/null
du -sh ~/.hermes/logs/ 2>/dev/null
du -sh ~/.hermes/cache/ 2>/dev/null
du -sh ~/.hermes/history/ 2>/dev/null
# IMPORTANT: probe checkpoints and state-snapshots — known major bloat sources
du -sh ~/.hermes/checkpoints/*/ 2>/dev/null | sort -rh | head -10
du -sh ~/.hermes/state-snapshots/*/ 2>/dev/null | sort -rh | head -10
# Hermes-agent subdirectory breakdown (venv/node_modules are large but needed)
du -sh ~/.hermes/hermes-agent/*/ 2>/dev/null | sort -rh | head -10
# OpenClaw
du -sh ~/.openclaw/ 2>/dev/null
du -sh ~/.openclaw/*/ 2>/dev/null | sort -rh | head -10
```

### 9. Large log files

```bash
find ~/.hermes/ -name "*.log" -size +1M 2>/dev/null | xargs ls -lh 2>/dev/null | sort -k5 -rh | head -20
find ~/.openclaw/ -name "*.log" -size +1M 2>/dev/null | xargs ls -lh 2>/dev/null | sort -k5 -rh | head -20
find /mnt/c/Projects/biotech_screener/biotech-screener -name "*.log" -size +1M 2>/dev/null | xargs ls -lh 2>/dev/null | sort -k5 -rh | head -20
```

### 10. Old log files (>7 days)

```bash
find ~/.hermes/ -name "*.log" -mtime +7 2>/dev/null | head -30
find ~/.openclaw/ -name "*.log" -mtime +7 2>/dev/null | head -30
```

### 11. Stale sessions by age

```bash
ls -lt ~/.hermes/sessions/ 2>/dev/null | head -30
ls -lt ~/.openclaw/sessions/ 2>/dev/null | head -30
# Check oldest sessions
ls -ltr ~/.hermes/sessions/ 2>/dev/null | head -20
ls -ltr ~/.openclaw/sessions/ 2>/dev/null | head -20
```

### 12. Temporary and scratch files in project

```bash
# Use maxdepth to avoid timeouts on deep node_modules trees
find /mnt/c/Projects/biotech_screener/biotech-screener -maxdepth 4 -name "*.tmp" 2>/dev/null
find /mnt/c/Projects/biotech_screener/biotech-screener -maxdepth 4 -name "*.bak" 2>/dev/null
find /mnt/c/Projects/biotech_screener/biotech-screener -maxdepth 3 -name "__pycache__" -type d 2>/dev/null | head -20
find /mnt/c/Projects/biotech_screener/biotech-screener/tools -maxdepth 1 -name "*.json" 2>/dev/null | xargs ls -lh 2>/dev/null
find /mnt/c/Projects/biotech_screener/biotech-screener -maxdepth 3 \( -name "scratch*" -o -name "tmp*" -o -name "test_output*" \) 2>/dev/null | head -20
# Project root gitignored JSON files (known recurring bulk output location)
ls -lt /mnt/c/Projects/biotech_screener/biotech-screener/*.json 2>/dev/null | head -30
du -ch /mnt/c/Projects/biotech_screener/biotech-screener/*.json 2>/dev/null | tail -1

# Orphaned .claude/worktrees/ — confirmed source of 235 MB on 2026-05-06
git -C /mnt/c/Projects/biotech_screener/biotech-screener worktree list
ls -la /mnt/c/Projects/biotech_screener/biotech-screener/.claude/worktrees/ 2>/dev/null
# Flag any directory present here but NOT in `git worktree list` as orphaned
```

### 13. Cache directories

```bash
find /mnt/c/Projects/biotech_screener/biotech-screener -name ".cache" -type d 2>/dev/null
find /mnt/c/Projects/biotech_screener/biotech-screener -name "*.pyc" 2>/dev/null | wc -l
du -sh /mnt/c/Projects/biotech_screener/biotech-screener/artifacts/ 2>/dev/null
ls -lt /mnt/c/Projects/biotech_screener/biotech-screener/artifacts/ 2>/dev/null | head -20
```

### 14. Hermes job history files

```bash
find ~/.hermes/ -name "*.jsonl" -o -name "*.json" 2>/dev/null | xargs ls -lh 2>/dev/null | sort -k5 -rh | head -20
```

---

## Required audit deliverable (9 sections)

After running all audit commands, produce this report in full:

---
MEMORY STEWARD AUDIT REPORT
Generated: <timestamp>
---

### 1. Current Memory/Session/Task Status
- Town: global + per-routine memory counts, skills catalog size, content library state
- Hermes: job count, last run dates, any failed jobs
- OpenClaw: agent count, active sessions, memory-core status
- Token pressure indicators where visible

### 2. Largest Resource Consumers
- Directories by disk size (top 10)
- Largest individual files

### 3. Stale Sessions by Age
- Session ID | Age | Last activity | Token count (if available)
- Flag any session older than 7 days

### 4. Stale Tasks/Issues
- Failed tasks with no resolution
- Abandoned tasks (no activity > 3 days)
- Issue count vs. active count

### 5. Safe Cleanup Candidates
Table: Path/ID | Type | Size/Count | Age | Why Safe

### 6. Unsafe Cleanup Candidates
Table: Path/ID | Type | Why Unsafe (what would be lost)

### 7. Exact Proposed Commands
One command per target. No broad wildcards.
Example:
  rm -rf ~/.hermes/sessions/abc123   # session from 2026-04-01, 0 tokens, abandoned

### 8. Estimated Risk Per Candidate
NONE / LOW / MEDIUM / HIGH with brief rationale

### 9. Rollback / Backup Plan
- What to back up before each action
- How to recover if something goes wrong
- For Town memories: note full content before `delete_memory` so it can be re-added
- Prefer: mv to archive location over rm

---
REFERENCE FILES
  references/environment-disk-baseline.md — observed disk sizes, session counts,
    project root JSON inventory, and fleet state as of 2026-05-06. Use as
    baseline when interpreting future audits (what's normal vs. new bloat).
---

---
REPO BACKUP NOTE
  This skill is backed up in the repo at:
  .hermes/skills/devops/memory-steward.SKILL.md
  If you update this skill, re-copy to keep them in sync:
    cp ~/.hermes/skills/devops/memory-steward/SKILL.md \
       /mnt/c/Projects/biotech_screener/biotech-screener/.hermes/skills/devops/memory-steward.SKILL.md
  A matching Claude Code subagent lives at:
    .claude/agents/memory-steward.md
  The Hermes skill is the source of truth. The subagent and cron job (876bb90e5295,
  Sundays 10:00 ET) both reference it.
---

---
DECISION OPTIONS
  AUDIT_ONLY              — no action taken, report only (default)
  CLEAN_STALE_MEMORIES    — remove confirmed-stale Town memories only
  CLEAN_LEGACY_CHECKPOINT — archive only the legacy checkpoint dir(s) identified
                            above (highest value, lowest risk — verify store/ first)
  CLEAN_SAFE_SNAPSHOTS    — archive only superseded pre-update state-snapshots
  CLEAN_ROOT_JSON         — archive only the gitignored dev output JSON files
                            from project root (confirmed gitignored first)
  CLEAN_SAFE_CACHES       — delete only regenerable cache dirs listed above
  CLEAN_STALE_SESSIONS    — archive/remove only the specific stale sessions listed
  CLEAN_STALE_TASKS       — clear only specific failed/abandoned tasks listed
  FULL_APPROVAL_REQUIRED  — approve each lettered item (A, B, C…) individually
---

STOP HERE. Do not execute any cleanup until user explicitly says proceed.

## Pitfalls

- **`find` without `-maxdepth` on WSL2 project dirs hangs** — node_modules trees
  are huge. Always add `-maxdepth 3` or `-maxdepth 4` to any `find` in the
  project root. The `find *.bak` without maxdepth will time out.

- **`session_api-*` files have no date in the filename** — these are API gateway
  sessions (Open WebUI, etc.). Use `ls -lt` mtime to gauge age, not the name.
  They tend to be small (a few KB each) but accumulate to ~74+ files. Medium
  risk if a gateway client is actively referencing them.

- **`hermes` CLI may not be in PATH** — the gateway runs as a systemd user
  service but the `hermes` CLI command may still not resolve in a cron shell.
  Fall back to `openclaw status` which IS available and covers the same ground.

- **"Failed" tasks usually mean delivery failure, not agent failure** — when
  `delivery=not_applicable` and the summary has real content, the cron agent
  completed but the delivery channel failed. Don't treat these as broken agents.
  Distinguish from truly empty/named-only failures (dispatch failures).

- **`~/.hermes/checkpoints/legacy-*` is NOT the active checkpoint store** — the
  active store is always `~/.hermes/checkpoints/store/`. Verify `store/` exists
  and is non-empty before archiving any `legacy-*` directory.

- **`~/.hermes/hermes-agent/venv/` is 826 MB but DO NOT DELETE** — it's the
  running Python environment for the gateway service. Removing it breaks all
  agents. It's regenerable but requires `pip install` and service restart.

- **Project root JSON files that look safe may be active reference data** —
  Always run `git check-ignore -v *.json` before touching any JSON in the
  project root. Files like `manager_registry.json`, `universe_config.json`,
  `pos_benchmarks_v1.json` are NOT bulk output and may be in active use.

---

## Execution rules (when approved)

- For Town memories: note the full content before calling `delete_memory`
- For Hermes files: back up before deleting (`cp -r` or tar archive)
- Prefer `mv` to `~/archive/` over `rm` for Hermes artifacts
- Never `rm -rf` without first listing exact targets
- Print the exact target list before executing
- Execute one category at a time, confirm after each
- If anything is ambiguous, skip it and flag for manual review
