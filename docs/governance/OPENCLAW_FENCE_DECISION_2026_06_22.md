# OpenClaw — Fence Decision

**Date:** 2026-06-22  
**Branch:** `openclaw-fence-retire-2026-06-22`  
**Verdict:** `FENCED — LEGACY_READ_ONLY_DORMANT`  
**Retirement decision:** deferred (process remains running; hard-retire on next reboot or explicit operator kill)

---

## 1. State at time of audit

| Property | Value |
|---|---|
| Gateway process | PID 7171, Node.js, `127.0.0.1:19001` (loopback only), running since 2026-06-20 |
| Version | `2026.6.8` (lastTouchedAt 2026-06-17T13:13:30Z) |
| Agents configured | 31 |
| Agent last activity | **2026-06-17** — all WAL files share the same mtime; 5+ days dormant |
| Internal cron | **Migrated/disabled** — `cron/jobs.json.migrated` exists; no active `jobs.json`; 9 jobs not running |
| OS crontab | No `@reboot` or scheduled entry for OpenClaw gateway — natural fence on reboot |
| Gateway binding | `127.0.0.1:19001` loopback only — not LAN-exposed |
| Plugins installed | `codex`, `discord`, `slack` (npm registry) |

---

## 2. Risk findings

### 2.1 exec-approvals `**` wildcard (CRITICAL — now removed)

The global `*` scope contained a `**` wildcard granting all 31 agents unrestricted
command execution. Combined with `/usr/bin/bash`, `/bin/bash`, `/bin/sh` also in the
global scope, any agent could run `git push`, `gh pr create`, or any other arbitrary
command without further approval.

Five per-agent scopes also had `**`: `postmortem`, `bioshort_watch`, `grok_biotech_watch`,
`crt_resolution_watcher`, `review_queue_steward`.

**This was the same class of risk as INC-2026-06-20-AUTOPUSH** (shell execution reaching
git write operations without enforcement at the executor layer).

**Action taken:** See §4.1.

### 2.2 SOUL.md missing from 3 agents

`biotech_news_digest`, `company_news_ingest`, and `policy_shadow_watch` had no SOUL.md
and therefore no explicit write/git restriction in their behavioral instructions. 24 of
35 agent directories had SOUL.md with "no git push" wording.

**Action taken:** See §4.2.

### 2.3 No scheduler resurrection path

OpenClaw's internal cron is in `jobs.json.migrated` — the scheduler does not have an
active job file. The OS crontab has no entry that would restart the OpenClaw gateway
(`@reboot` points to `cron_daily_production.sh`, not OpenClaw). **The gateway will not
auto-restart on reboot** — this is the natural retirement path.

**No action needed.** Documented here as a confirmed-closed vector.

### 2.4 Git/gh not in explicit exec-approvals

No `git` or `gh` binary appeared in the pre-fence allowlist by name. They were reachable
only through `**` + bash. With both removed, there is no pre-approved path to git or gh
for any agent.

---

## 3. Fence scope decision

**FENCE, not immediate retire.** Rationale:

- The gateway is idle and loopback-only. Killing it now has no operational benefit.
- Hard-retire (kill PID, archive `~/.openclaw/`) should be a deliberate operator action,
  not an automated script. The natural path is: next reboot removes the process.
- The config changes in §4 close the actual risk (exec wildcard + missing SOUL.md)
  without requiring process termination.
- 31 agents represent historical diagnostic work; the workspaces remain readable.

**Ownership:** Hermes is now the primary orchestrator. OpenClaw agents are legacy
read-only. No new jobs, no new agents, no cron resurrection in OpenClaw.

---

## 4. Changes made

### 4.1 exec-approvals fenced (`~/.openclaw/exec-approvals.json`)

Backup: `~/.openclaw/exec-approvals.json.bak.20260622_fence`

**Removed from global `*` scope:**

| Pattern | Reason |
|---|---|
| `**` | Catch-all wildcard — unrestricted execution |
| `/usr/bin/bash` | Arbitrary shell execution |
| `/bin/bash` | Arbitrary shell execution |
| `/bin/sh` | Arbitrary shell execution |
| `/mnt/c/Projects/biotech_screener/biotech-screener/**` | Repo-scoped wildcard — can execute any script in the repo |
| `/usr/bin/cp` | Write-capable (file copy) |
| `/usr/bin/mv` | Write-capable (file move) |
| `/usr/bin/mkdir` | Write-capable (directory creation) |

**Removed from per-agent scopes:**

| Agent | Change |
|---|---|
| `postmortem` | `**` → `[]` (empty) |
| `bioshort_watch` | `**` → `/usr/bin/test` only (sole observed command) |
| `grok_biotech_watch` | `**` → `[]` |
| `crt_resolution_watcher` | `**` → `[]` |
| `review_queue_steward` | `**` → `[]` |

**Global allowlist after fence (read-only):**
`python3`, `ls`, `cat`, `head`, `tail`, `grep`, `find`, `wc`, `date`, `stat`, `diff`, `sort`, `jq`

Added `_governance` field to exec-approvals.json documenting the fence.

### 4.2 SOUL.md added to 3 agents

`biotech_news_digest`, `company_news_ingest`, `policy_shadow_watch` — each received a
minimal SOUL.md with explicit "no git operations" and "write to memory/output dirs only"
constraints, matching the pattern of the 24 agents that already had SOUL.md.

---

## 5. Verified: no new scheduler vectors

| Check | Result |
|---|---|
| OpenClaw internal cron active jobs | None — `jobs.json.migrated` only |
| OS crontab `@reboot` for OpenClaw | Not present |
| systemd unit for OpenClaw | Not present (`systemctl --user status openclaw` returns nothing) |
| Gateway auto-restart on reboot | **No** — natural retirement on next reboot |

---

## 6. Hard-retire checklist (operator-triggered, not automated)

When the operator is ready to fully retire OpenClaw:

```bash
# 1. Kill the gateway process
kill 7171

# 2. Confirm port is free
ss -tlnp | grep 19001

# 3. Archive config (keep for reference, remove live runtime)
mv ~/.openclaw ~/.openclaw.archived.2026-06-22

# 4. Remove any remaining OpenClaw references from ~/.hermes/config.yaml if present
# (currently none — openclaw is not in mcp_servers)

# 5. Update this doc: change verdict to RETIRED
```

**Do not delete** `~/.openclaw/` before archiving — agent memory and logs may be useful
for retrospective audit.

---

## 7. Ownership transfer

| Workload | From | To |
|---|---|---|
| Daily production pipeline | OpenClaw (ops-daily) | OS crontab (`tools/cron_daily_production.sh`) — already active |
| Fleet monitoring | OpenClaw (fleet_steward) | Hermes (ops_supervisor + ic_health_monitor) |
| New agent orchestration | OpenClaw | Hermes only |
| Sentinel / QA | OpenClaw (sentinel-daily, qa-daily) | Hermes (pending separate branch) |

---

*Fence complete. Hard-retire is a separate operator-triggered action. Next branch: harvester-manualization.*
