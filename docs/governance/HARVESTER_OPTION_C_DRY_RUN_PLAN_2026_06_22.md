# Harvester Option C — Dry-Run Plan (2026-06-22)

**Status:** `DRY_RUN_COMPLETE — ALL_AC_PASS`
**Branch:** `harvester-dry-run-followup-2026-06-22`
**Supersedes:** plan committed on `harvester-manualization-2026-06-22` (never reached main via squash merge)

---

## Dry-run result (2026-06-22)

**PR #373 merged at:** `37111ad4` (squash merge, governance doc + pending dir)

**Run method:** Path B — direct `hermes chat -q` with extracted prompt. Path A (scheduler trigger) was attempted first and **must not be used** — see the Path A safety finding below.

| AC | Check | Result |
|---|---|---|
| AC-1 | `weekly-skill-harvester` still `enabled: False / state: paused` | ✅ PASS |
| AC-2 | `docs/hermes_skills/pending/HARVEST_2026-06-22.md` created | ✅ PASS |
| AC-3 | HEAD unchanged (`37111ad4`) | ✅ PASS |
| AC-4 | No files outside `docs/hermes_skills/` changed | ✅ PASS (`.codegraph/daemon.pid` is a codegraph side-effect, not repo data) |
| AC-5 | Pending file contains `PENDING_OPERATOR_REVIEW`, proposed commit message, operator instructions | ✅ PASS |
| AC-6 | Zero executed git tool calls | ✅ PASS |

**Pending file written:** `docs/hermes_skills/pending/HARVEST_2026-06-22.md`

Captured two real patterns from today's governance work:
- `openclaw-agent-scope-audit.md` — Class I (OpenClaw fenced/legacy dormant, commit `4d1a4fd8`)
- `openclaw-cron-scheduler-debug.md` — Class J (pre-push guard for main, commit `ded2d3b0`)

**Conclusion:** Option C manualization is verified. The harvester writes a proposal file and stops. No autonomous git operations occur.

---

## ⚠️ Path A safety finding — UNSAFE, DO NOT USE

> **Standing rule:** Do not use `hermes cron run` + `hermes cron tick` to test paused jobs. Use direct `hermes chat -q` (Path B) only.

**What happened:** Before the successful dry-run, `hermes cron run a15dbdcb6f41` + `hermes cron tick` was attempted. The tick re-enabled the paused job as a side effect:
- `enabled` flipped: `False` → `True`
- `state` changed: `paused` → `scheduled`
- `next_run_at` was set to `2026-06-22T20:00:00-04:00`

The job had NOT yet fired (last_run_at was unchanged at 2026-06-17). The flip was caught before 20:00 and the job was re-paused via `hermes cron pause weekly-skill-harvester`.

**Root cause:** `hermes cron tick` processes queued "force-run" requests by transitioning a job out of `paused` state, not by running it in place. The job becomes live and scheduled rather than running once and returning to paused.

**Path A is now marked OFF-LIMITS for paused-job dry-runs.** Do not use it unless Hermes scheduler behavior is explicitly fixed and re-reviewed.

---

## Approved invocation path (Path B)

```bash
# 1. Pre-flight: confirm job is paused
python3 -c "
import json
from pathlib import Path
d = json.loads((Path.home() / '.hermes/cron/jobs.json').read_text())
j = next(x for x in d['jobs'] if x['id'] == 'a15dbdcb6f41')
assert j['enabled'] == False and j['state'] == 'paused', 'ABORT: job not paused'
print('PRE-FLIGHT OK:', j['state'])
"

# 2. Save prompt to temp file
python3 -c "
import json
from pathlib import Path
d = json.loads((Path.home() / '.hermes/cron/jobs.json').read_text())
j = next(x for x in d['jobs'] if x['id'] == 'a15dbdcb6f41')
Path('/tmp/harvester_prompt.txt').write_text(j['prompt'])
print('saved to /tmp/harvester_prompt.txt')
"

# 3. Run as one-shot session — no cron, no scheduler, no jobs.json edit
cd /mnt/c/Projects/biotech_screener/biotech-screener
hermes chat \
  -t "terminal,file,session_search,skills" \
  -s "openclaw-fleet-triage,openclaw-cron-scheduler-debug,openclaw-agent-scope-audit,openclaw-session-routing-debug,openclaw-data-pipeline-debug,aa-model-tracker" \
  --accept-hooks \
  -q "$(cat /tmp/harvester_prompt.txt)"
```

**Why Path B is safe:**
- `hermes chat -q` has no awareness of the cron scheduler — it cannot flip `enabled` or `state`
- The exact same prompt runs as when the cron fires
- `jobs.json` is never touched
- The run exits cleanly after one session

---

## Context

`weekly-skill-harvester` (job `a15dbdcb6f41`) was manualized via Option C on 2026-06-22.
Step 8 was rewritten: autonomous `git add → commit → push` replaced with a pending-commit
proposal file at `docs/hermes_skills/pending/HARVEST_<date>.md`. Steps 1–7 unchanged.

---

## Pre-flight checks (run before any future invocation)

```bash
# 1. Job still paused
python3 -c "
import json
from pathlib import Path
d = json.loads((Path.home() / '.hermes/cron/jobs.json').read_text())
j = next(x for x in d['jobs'] if x['id'] == 'a15dbdcb6f41')
print('enabled:', j['enabled'], '  state:', j['state'])
assert j['enabled'] == False and j['state'] == 'paused', 'ABORT: job not paused'
print('PRE-FLIGHT OK — job is paused')
"

# 2. Baseline git state
git -C /mnt/c/Projects/biotech_screener/biotech-screener status --short

# 3. Baseline pending dir
ls /mnt/c/Projects/biotech_screener/biotech-screener/docs/hermes_skills/pending/

# 4. Record current HEAD
git -C /mnt/c/Projects/biotech_screener/biotech-screener log --oneline -1

# 5. Confirm Step 8 prohibition
python3 -c "
import json
from pathlib import Path
d = json.loads((Path.home() / '.hermes/cron/jobs.json').read_text())
j = next(x for x in d['jobs'] if x['id'] == 'a15dbdcb6f41')
assert 'do NOT run any git commands' in j['prompt'].split('STEP 8')[1], 'ABORT'
print('STEP 8 PROHIBITION OK')
"
```

---

## Acceptance criteria

| # | Check | Pass condition |
|---|---|---|
| AC-1 | Job still paused | `enabled: False  state: paused` |
| AC-2 | Pending file created | `HARVEST_<date>.md` in `docs/hermes_skills/pending/` |
| AC-3 | No new commits | HEAD matches pre-flight |
| AC-4 | No unexpected file changes | only `docs/hermes_skills/` mods + pending file (untracked) |
| AC-5 | Pending file is proposal-only | contains `PENDING_OPERATOR_REVIEW`, proposed commit msg, operator instructions |
| AC-6 | No executed git tool calls | zero matches for `git (add\|commit\|push\|pull)` in shell tool invocations |

---

## No-go criteria

| Trigger | Action |
|---|---|
| Any executed `git add`, `git commit`, `git push` | Kill. Check `git log`. Open incident note. |
| New commit in `git log` | Record hash. Do NOT push. Investigate. |
| `enabled` flips to `True` | Re-pause immediately: `hermes cron pause weekly-skill-harvester` |
| Any file outside `docs/hermes_skills/` changes | Revert: `git checkout -- <file>`. Investigate prompt. |

---

## Post-run operator review

```bash
# 1. Read the pending proposal
cat docs/hermes_skills/pending/HARVEST_<date>.md

# 2. Diff the skill changes
git diff docs/hermes_skills/

# 3. If approved, commit manually
git add docs/hermes_skills/
git commit -m "docs(skills): weekly skill harvest <date> — <summary>"
git push
rm docs/hermes_skills/pending/HARVEST_<date>.md

# 4. If not approved, discard (no harm done)
rm docs/hermes_skills/pending/HARVEST_<date>.md
```

---

## Decision gate: re-enabling for production

- [x] Dry-run completed successfully (all 6 AC pass) — 2026-06-22
- [x] Pending proposal reviewed and captures real findings — 2026-06-22
- [x] PR #373 merged to main — `37111ad4`
- [ ] Operator explicitly authorizes re-enable

Re-enable command (authorized only):
```bash
hermes cron resume weekly-skill-harvester
hermes cron list | grep harvester   # verify state: scheduled
```

---

## Rollback

```bash
cp ~/.hermes/cron/jobs.json.bak.20260622_harvester_manualize ~/.hermes/cron/jobs.json
# Restores original auto-push behavior. Do not rollback without operator review.
```
