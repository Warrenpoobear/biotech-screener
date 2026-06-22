# Weekly-Skill-Harvester Manualization — 2026-06-22

**Status:** `APPLIED — OPTION_C_PENDING_OPERATOR_REVIEW`  
**Branch:** `harvester-manualization-2026-06-22`  
**Job ID:** `a15dbdcb6f41`  
**Current state:** `paused` (`enabled: false`) — unchanged  

---

## Background

`weekly-skill-harvester` is a Hermes cron job (Mondays 20:00 ET) that reviews git commits + sessions, patches skills in `~/.hermes/skills/`, syncs them to `docs/hermes_skills/`, and — critically — runs an autonomous `git add → commit → push` to `main`.

That last step is the same class of risk as **INC-2026-06-20-AUTOPUSH**: an agent with write access to `main` operating without operator review. The job was paused as part of the containment package on 2026-06-22T13:14. Last successful run: 2026-06-17T17:19.

---

## Option C: Manualization

**What changed:** Step 8 of the job prompt was replaced. The agent no longer runs any git commands.

| Before | After |
|---|---|
| `git checkout main && git pull --ff-only` | ❌ removed |
| `git add docs/hermes_skills/` | ❌ removed |
| `git commit -m "docs(skills): ..."` | ❌ removed |
| `git push` | ❌ removed |
| (nothing) | ✅ writes `docs/hermes_skills/pending/HARVEST_<date>.md` |

**Steps 1–7 are unchanged:** git activity harvest, session harvest, gap analysis, skill patching (in `~/.hermes/skills/`), new skill creation, sync to `docs/hermes_skills/`, harvest log update.

**What the agent now produces instead of a commit:**

```
docs/hermes_skills/pending/HARVEST_YYYY-MM-DD.md
```

This file lists:
- Files changed in `docs/hermes_skills/`
- The proposed commit message
- Copy-paste operator instructions to review and commit manually

**Operator review flow (manual, after harvest runs):**

```bash
# 1. Review what the agent changed
git diff docs/hermes_skills/
cat docs/hermes_skills/pending/HARVEST_<date>.md

# 2. If approved, commit and push
git add docs/hermes_skills/
git commit -m "docs(skills): weekly skill harvest <date> — <summary>"
git push

# 3. Clean up the pending proposal
rm docs/hermes_skills/pending/HARVEST_<date>.md
```

---

## Why Option C (not A or B)

| Option | Description | Rejected reason |
|---|---|---|
| A | Re-enable with pre-push hook only | Hook bypass possible; `ALLOW_AGENT_PUSH=1` still works |
| B | Delete the job entirely | Skill harvesting has value; just the push is dangerous |
| **C** | Remove git operations; write proposal file | **Chosen** — preserves harvest value, eliminates push risk |

---

## Backup

`~/.hermes/cron/jobs.json.bak.20260622_harvester_manualize`

---

## Pending/ directory

`docs/hermes_skills/pending/` is created with a `.gitkeep`. Harvest proposal files written there are not auto-committed. The operator commits them after review.

---

## Re-enabling

Job remains `paused`. To re-enable when ready:

```bash
hermes cron toggle a15dbdcb6f41
# or via Hermes UI: Jobs → weekly-skill-harvester → Enable
```

The manualized prompt is safe to re-enable: it cannot push to any branch autonomously.

---

## Rollback

To restore the original prompt (auto-push behavior):

```bash
cp ~/.hermes/cron/jobs.json.bak.20260622_harvester_manualize ~/.hermes/cron/jobs.json
```

**Do not rollback without operator review** — the original prompt has an unconstrained `git push` to `main`.

---

## Change log

| Date | Actor | Action |
|---|---|---|
| 2026-06-22 13:14 | containment | Job paused as part of INC-2026-06-20-AUTOPUSH response |
| 2026-06-22 | Claude Code | Step 8 replaced with pending-commit report; job remains paused |
