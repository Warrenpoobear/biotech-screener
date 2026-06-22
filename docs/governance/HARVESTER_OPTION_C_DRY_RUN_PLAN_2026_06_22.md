# Harvester Option C — Dry-Run Plan (2026-06-22)

**Status:** `PLAN_ONLY — DO_NOT_RUN_UNTIL_REVIEWED`
**Branch:** `harvester-manualization-2026-06-22`
**Prerequisite:** PR #373 must be reviewed (not necessarily merged) before executing.

---

## Context

`weekly-skill-harvester` (job `a15dbdcb6f41`) was manualized via Option C on 2026-06-22.
Step 8 was rewritten: autonomous `git add → commit → push` replaced with a pending-commit
proposal file at `docs/hermes_skills/pending/HARVEST_<date>.md`. Steps 1–7 unchanged.

This plan governs a **manual, one-time dry-run** to verify the rewritten step behaves as
designed before the job is ever re-enabled on the cron scheduler.

---

## Pre-flight checks (run before invoking)

```bash
# 1. Job still paused
python3 -c "
import json; d = json.load(open('/root/.hermes/cron/jobs.json' if False else
    __import__('pathlib').Path.home() / '.hermes/cron/jobs.json'))
j = next(x for x in d['jobs'] if x['id'] == 'a15dbdcb6f41')
print('enabled:', j['enabled'], '  state:', j['state'])
assert j['enabled'] == False and j['state'] == 'paused', 'ABORT: job not paused'
print('PRE-FLIGHT OK — job is paused')
"

# 2. Baseline git state — zero uncommitted changes (ignoring known pre-existing diffs)
git -C /mnt/c/Projects/biotech_screener/biotech-screener status --short

# 3. Baseline pending dir — only .gitkeep (no prior runs)
ls /mnt/c/Projects/biotech_screener/biotech-screener/docs/hermes_skills/pending/

# 4. Baseline git log tip — record current HEAD to detect any new commits
git -C /mnt/c/Projects/biotech_screener/biotech-screener log --oneline -1

# 5. Confirm Step 8 prohibition is in the prompt
python3 -c "
import json
from pathlib import Path
d = json.loads((Path.home() / '.hermes/cron/jobs.json').read_text())
j = next(x for x in d['jobs'] if x['id'] == 'a15dbdcb6f41')
prompt = j['prompt']
assert 'do NOT run any git commands' in prompt, 'ABORT: Step 8 prohibition missing'
assert 'git add' not in prompt.split('STEP 8')[1].split('CONSTRAINTS')[0].split('do NOT')[1], \
    'ABORT: git add found after prohibition'
print('STEP 8 PROHIBITION OK')
"
```

All five checks must pass before invoking. If any assertion fails, stop.

---

## Invocation

`hermes cron run` queues a job for the next scheduler tick. Since the job is paused
(`enabled: false`), the scheduler may skip it — verify behavior before running:

```bash
# Probe: does hermes cron run respect paused state?
hermes cron run a15dbdcb6f41 2>&1
# Expected output variants:
#   A) "Scheduled job a15dbcb6f41 to run on next tick" → proceed to tick
#   B) "Job is paused / disabled" → use direct-prompt fallback below
```

**Path A — scheduler trigger (if probe succeeds):**
```bash
hermes cron tick 2>&1
# This runs due jobs once and exits. Watch for git command mentions in output.
```

**Path B — direct-prompt fallback (if job is skipped because paused):**
```bash
# Extract prompt from jobs.json and run it as a one-shot hermes session.
# Do NOT enable the job first. Do NOT edit jobs.json.
#
# This approach is preferred over temporarily enabling the job because it:
#   - keeps enabled=false / state=paused unchanged
#   - runs the exact same prompt without scheduler involvement
#   - exits cleanly after one invocation
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads((Path.home() / '.hermes/cron/jobs.json').read_text())
j = next(x for x in d['jobs'] if x['id'] == 'a15dbdcb6f41')
print("Run this command:")
print()
skills = ' '.join(f'--skill {s}' for s in j.get('skills', []))
print(f"hermes {skills} --workdir '{j['workdir']}' \\")
print(f"  '{j['prompt'][:80]}...'")
print()
print("(Full prompt is in ~/.hermes/cron/jobs.json, id=a15dbdcb6f41)")
PY
```

---

## Acceptance criteria

After the run completes, verify all of the following:

| # | Check | Command | Pass condition |
|---|---|---|---|
| AC-1 | Job still paused | `python3 -c "import json,pathlib; d=json.loads((pathlib.Path.home()/'.hermes/cron/jobs.json').read_text()); j=next(x for x in d['jobs'] if x['id']=='a15dbdcb6f41'); print(j['enabled'],j['state'])"` | `False paused` |
| AC-2 | Pending file created | `ls docs/hermes_skills/pending/` | `HARVEST_<date>.md` present |
| AC-3 | No new commits | `git log --oneline -1` | same HEAD as pre-flight |
| AC-4 | No unexpected file changes | `git status --short` | only pre-existing diffs + `docs/hermes_skills/pending/HARVEST_<date>.md` (untracked) |
| AC-5 | Pending file is proposal-only | `cat docs/hermes_skills/pending/HARVEST_*.md` | contains `PENDING_OPERATOR_REVIEW`, proposed commit msg, operator instructions |
| AC-6 | No git commands in output | review run output | no `git add`, `git commit`, `git push`, `git pull` in stdout |

All 6 must pass. Any failure is a defect in the manualization.

---

## No-go criteria (abort and investigate)

Stop immediately if any of the following occur **during or after** the run:

| Trigger | Action |
|---|---|
| Output contains `git add`, `git commit`, `git push`, `git pull`, `git checkout` | Kill the process. Check if any commit was made (`git log`). |
| New commit appears in `git log` | Record the commit hash. Do NOT push. Open an incident note. |
| Any file outside `docs/hermes_skills/` changes (production data, configs, scripts) | `git checkout -- <file>` to revert. Investigate the prompt. |
| Job `enabled` flips to `true` or `state` changes from `paused` | Re-pause immediately: `hermes cron pause weekly-skill-harvester` |
| Run writes to `~/.hermes/skills/` AND also runs git (double-write) | Acceptable to write `~/.hermes/skills/` — that is Steps 4–5 behavior. Git is the no-go. |
| `docs/hermes_skills/pending/` contains a file with git commands inside it | The proposal template itself must not contain executable git commands — it should contain copy-paste operator instructions only. Review the file. |

---

## Post-run operator review

If all acceptance criteria pass:

1. Read the pending file: `cat docs/hermes_skills/pending/HARVEST_<date>.md`
2. Inspect what skills were patched: `diff -u ~/.hermes/skills/devops/<name>/SKILL.md docs/hermes_skills/<name>.md`
3. If approved, apply manually:
   ```bash
   git diff docs/hermes_skills/
   git add docs/hermes_skills/
   git commit -m "docs(skills): weekly skill harvest <date> — <summary>"
   git push
   rm docs/hermes_skills/pending/HARVEST_<date>.md
   ```
4. If not approved, discard: `rm docs/hermes_skills/pending/HARVEST_<date>.md` — no harm done.

---

## Decision gate: re-enabling for production

Do NOT re-enable the cron job (`hermes cron resume weekly-skill-harvester`) until:

- [ ] Dry-run completed successfully (all 6 AC pass)
- [ ] Pending proposal reviewed and makes sense (skill patches are accurate, no hallucination)
- [ ] PR #373 merged to main
- [ ] Operator explicitly authorizes re-enable

Re-enable command (authorized only):
```bash
hermes cron resume weekly-skill-harvester
# Verify:
hermes cron list | grep harvester
```

---

## Rollback (if dry-run reveals a defect in the manualization)

```bash
# Restore original auto-push behavior (ONLY if decided to revert to pre-manualization)
cp ~/.hermes/cron/jobs.json.bak.20260622_harvester_manualize ~/.hermes/cron/jobs.json
# NOTE: this restores the autonomous git push — do not revert without operator review.
```
