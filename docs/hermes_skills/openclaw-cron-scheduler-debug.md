---
name: openclaw-cron-scheduler-debug
description: "Diagnose cron and scheduler failures for the OpenClaw biotech screener fleet. Encodes real failure patterns from production: crontab REPLACE silent death, WSL2 sleep cliff, Hermes scheduler stall, watchdog idempotency loops, and announce/webchat channel delivery errors."
when_to_use: "User says 'cron didn't fire', 'production missed', 'agents not running', 'watchdog looping', 'consecutive errors', or fleet receipt shows broad STALE across many agents with no obvious code cause. Also load when OpenClaw jobs show delivery errors with mode=announce."
---

# OpenClaw Cron / Scheduler Debugger

Covers the biotech screener at `/mnt/c/Projects/biotech_screener/biotech-screener`.
Production is Mon-Fri only. Sleep cliff is ~17:30 ET on weeknights.

## Hard rules

- Diagnose-only by default. Never edit crontab without operator approval + preview diff.
- Always diff before apply. Never `sed -i` without a preview step.
- One remediation at a time.

---

## Failure taxonomy

### Class A — Crontab REPLACE silent death

**Signature:** Root cron daemon alive (root hourly fires in syslog), but zero user
CMD entries for all arrenchulz jobs from a specific timestamp forward. No syslog
error. Only clue is a `REPLACE` event near the gap.

**Cause:** A crontab edit wrote a file the daemon could not parse. On bad parse,
the daemon reloads and silently drops ALL user jobs with no stderr, no email, no alert.

**Confirmed instance:** 2026-05-02T12:43+12:46 ET — two REPLACE ops during a triage
session. All jobs from 12:46 forward were silent: data_refresh, data_extras,
production (16:30), agents (17:00-18:55), heartbeat_checks, data_auditor all missed.

**Diagnostic recipe:**

```bash
# 1. Find the REPLACE events
grep "arrenchulz.*REPLACE\|arrenchulz.*RELOAD" /var/log/syslog | tail -20

# 2. Confirm absence of user CMD entries after the REPLACE
grep "2026-MM-DD" /var/log/syslog | grep "CRON\[" | grep "CMD" | grep -v "root\|hourly\|e2scrub" | tail -20
# Empty despite root cron firing = Class A confirmed

# 3. Confirm current crontab parses
crontab -l > /tmp/cron_test.txt && echo "PARSE OK" || echo "PARSE FAIL"
```

**Resolution:**
1. Verify current crontab is parseable.
2. Determine what was missed — check `data/snapshots/<date>` for weekday gaps.
3. Manual production backfill for a missed weekday:
   `bash tools/cron_daily_production.sh <YYYY-MM-DD>`
   (script guards against weekends — no need to check manually)

**Pitfall — verify the date is a WEEKDAY before assuming production missed:**

```bash
python3 -c "
import datetime, sys
d = datetime.date.fromisoformat('YYYY-MM-DD')
print('WEEKDAY — production expected' if d.weekday() < 5 else 'WEEKEND — no production, do NOT backfill')
"
```

`data/snapshots/YYYY-MM-DD` absence on a weekend date is NORMAL.
`cron_daily_production.sh <date>` will silently exit 0 with "SKIP: is a weekend" —
it does not error, so you will not get feedback that the backfill was skipped.
Always pre-check the day-of-week before running a manual backfill.

**Confirmed 2026-05-04:** Attempted backfill of "missed" 2026-05-02 production.
Script exited cleanly: "SKIP: 2026-05-02 is a weekend (day 6)".
The apparent miss was the data_auditor weekend false-positive (Class E), not a real gap.

---

### Class B — WSL2 host-sleep cliff

**Signature:** Multiple STALE findings in fleet receipt, all with last-write ≤ 17:30
ET on a weeknight. Everything after ~17:30 ET is sleep-roulette.

**Confirmed schedule:** Production (16:30) and heartbeat_checks (17:30) are the last
reliable slots. Phase-2 agents (18:00-18:55), data_auditor (18:00), bellringer
results (18:30), evening catchup (22:00) all at risk.

**Diagnostic recipe:**

```bash
# 1. Cron daemon uptime
service cron status | grep "Active:"
# If "active since" predates missing run, daemon was down

# 2. Last production entry
tail -5 logs/cron.log

# 3. Which logs stopped at ~17:30?
for log in logs/*.log; do
  mt=$(stat -c '%Y' "$log")
  printf '%s  %s\n' "$(date -d @$mt '+%F %H:%M')" "$(basename $log)"
done | sort
```

**Mitigations already in place:**
- Tier-1: critical jobs scheduled ≤ 17:30 ET (patched 2026-05-02)
- Tier-2: `tools/cron_evening_catchup.sh` at 22:00 + @reboot
- Tier-3: weekend calibration catchup (Sat/Sun 09:00 ET)

---

### Class C — Hermes scheduler stall

**Signature:** Linux cron alive and user CMD entries firing, but Hermes cron jobs
show `last_run_at: null` and `next_run_at` in the past. These are SEPARATE systems.

**Confirmed instances:**
- 2026-05-03 00:33 ET — Hermes scheduler had 3 of 4 jobs with `last_run_at: null`
  overdue by 27h+. auth-sync never fired → 9 agent profiles expired simultaneously.
- 2026-05-05 13:18 ET — auth-sync cron (4cfe9fb5d466, every 6h) last ran
  2026-05-03T22:02, stalled 39h across ~6 missed cycles. All 31 agents EXPIRED+DRIFT.
  Root cause same: Hermes scheduler silently paused, no alert, no log entry.

**Pattern:** This stall recurs. The Hermes scheduler does not self-heal reliably
after WSL2 sleep. Treat any auth failure spike as a Class C suspect first.

**Diagnostic:** Use `mcp_cronjob action='list'`. Red flags: `last_run_at: null` with
`next_run_at` in the past, or auth-sync job last ran >8h ago on a day with no sleep cliff.

**Resolution order:**
1. Run `~/.local/bin/openclaw-auth-sync` manually — immediate relief for all agents.
2. Kick the stalled Hermes job: `mcp_cronjob action='run' job_id='4cfe9fb5d466'`.
3. Re-trigger `tools/agent_heartbeat_checks.py` if fleet receipts are stale.
4. Confirm auth-sync next_run_at reset to ~now+6h.

**Quick health check (run first when 2+ agents show auth errors):**
```python
import json, glob, os
from datetime import datetime, timezone
now = datetime.now(timezone.utc).timestamp()
creds = json.load(open(os.path.expanduser('~/.claude/.credentials.json')))
src_fp = creds['claudeAiOauth']['accessToken'][-8:]
profiles = sorted(glob.glob(os.path.expanduser('~/.openclaw/agents/*/agent/auth-profiles.json')))
issues = []
for p in profiles:
    ag = p.split('/agents/')[1].split('/')[0]
    d = json.load(open(p))
    for name, prof in d.get('profiles', {}).items():
        if 'claude-cli' in name and isinstance(prof, dict):
            exp = prof.get('expires', 0)/1000
            fp = prof.get('access','')[-8:]
            if exp < now or fp != src_fp:
                issues.append(f"{ag}: {'EXPIRED' if exp < now else 'ok'} {'DRIFT' if fp != src_fp else 'synced'}")
print(f"{len(issues)} issues" if issues else "all synced")
for i in issues: print(i)
```
If issues > 0: run openclaw-auth-sync immediately, then kick cron 4cfe9fb5d466.

---

### Class D — Watchdog idempotency loop

**Signature:** `logs/watchdog.log` shows the same phase-2 agents "recovered" every
30 minutes, with "Phase-2 agent recovery complete" each cycle, yet next cycle still
sees them as MISSED.

**Cause:** The "already recovered for <date>" marker is not being written, written
to the wrong path, or checking the wrong date key.

**Confirmed instance:** 2026-05-04 14:00 + 14:30 ET — same 5 agents recovered twice.

**Diagnostic:**

```bash
# What does the watchdog use as its "already ran" check?
grep -n "PROD_RAN\|already.ran\|skip\|rankings\|snapshot" tools/cron_watchdog.sh | head -20

# Check the production ran marker
ls -la data/snapshots/$(date +%Y-%m-%d)/rankings.csv 2>/dev/null || echo "MISSING"
ls -la logs/.daily_production.lock 2>/dev/null || echo "NO LOCK"
```

**Note:** On weekends this loop is benign (Mon-Fri jobs, harmless recovery calls).

---

### Class E — data_auditor weekend false-positive

**Signature:** `integrity_report_<YYYY-MM-DD>.json` on Saturday or Sunday shows
`verdict: FAIL` with `archive_verification: FAIL` ("Archive missing for <Sat> AND <Fri>")
and cascade ERRORs on universe_ipo, pit_financials, financial_consistency, price_data_gaps.

**Cause:** `run_audit.py --daily-only` doesn't guard against weekend execution.
Expects same-day rankings.csv that will never exist for Sat/Sun.

**Known behavior. Disregard entirely unless the FAIL date is a weekday.**

```python
import json, datetime
report = json.load(open('artifacts/data_auditor/integrity_report_YYYY-MM-DD.json'))
date = datetime.date.fromisoformat(report['as_of_date'])
if date.weekday() >= 5:
    print("WEEKEND FALSE-POSITIVE — disregard")
```

---

### Class F — LLM agent invoked for a task requiring real tool execution (architectural mismatch)

**Signature:** Agent fires on schedule and logs "success" but produces no artifacts.
Invocation log shows the agent describing what it *would* do but finding no data.
Specifically: agent checks for env vars or files via bash commands that return nothing,
then aborts cleanly. run_agent_direct.py always returns status=success even when the
agent did no real work.

**Root cause:** `run_agent_direct.py` is a plain Anthropic SDK text call — no tools,
no bash execution. Agents that require real shell commands (API calls, file reads,
xAI search) were designed for the OpenClaw gateway which provides real bash tooling.
When invoked via run_agent_direct.py, bash commands in the agent's response produce
no output, env vars appear absent, and the agent aborts.

**Confirmed instance (2026-05-04):** `grok_biotech_watch` invoked 3x daily via
run_agent_direct.py with --message "SCAN". Agent logged "XAI_API_KEY not found"
and "HEARTBEAT: FAIL" despite XAI_API_KEY being present in .env and loaded into
os.environ by run_agent_direct.py main(). Root cause: the bash env check the agent
wrote was never executed — it was text in the response, not a real shell call.
35 days of silent failures before diagnosed.

**Diagnostic recipe:**

```bash
# 1. Check invocation log for the tell-tale pattern
cat logs/agents_direct/<agent>_<date>.json | python3 -c "
import json,sys; d=json.load(sys.stdin)
r = d.get('response','')
for kw in ['not found','FAIL','cannot','no filesystem','abort']:
    if kw.lower() in r.lower():
        # Find lines with keywords
        for line in r.split(chr(10)):
            if kw.lower() in line.lower(): print(line[:120])
"

# 2. Confirm the agent has a builder script alternative
ls tools/build_<agent>*.py 2>/dev/null

# 3. Check if the agent's TOOLS.md documents a builder entrypoint
grep -i "builder\|build_\|python tools/" agents/<agent>/TOOLS.md | head -5
```

**Resolution (confirmed working):**
1. Check TOOLS.md for the existing builder script (`tools/build_<agent>.py`).
2. Swap the crontab entry from `run_agent_direct.py --agent X --message SCAN`
   to `build_X.py --as-of-date $(date +%Y-%m-%d) --send-email`.
3. Smoke test with `--digest-only` flag before enabling email sends.
4. Builder scripts run in the same shell as cron and inherit the full env
   including all keys loaded from `.env` via `source .env`.

**Pattern:** Any agent whose SOUL.md lists "xAI Grok API", "HTTP API call", or
"external service" under Boundaries is likely to fail silently under
run_agent_direct.py. Check TOOLS.md for a builder alternative before debugging
further.

---

### Class H — Heartbeat checker stall (monitor-of-monitors blind spot)

**Signature:** Fleet receipt is >24h stale (often 5+ days). The receipt file at
`agents/fleet_steward/memory/<date>_receipt.md` is the detection mechanism for ALL
agent stalls — when it goes dark, you lose visibility into every other agent's state.
Production snapshots may still be created (via Hermes scheduler or manual trigger),
making this easy to miss: "production is running, so everything must be fine."

**Confirmed instance (2026-06-17):** Fleet receipt last generated 2026-06-12 (5 days
stale). `tools/agent_heartbeat_checks.py` had not produced a receipt since June 12.
Linux cron daemon was running, production snapshots existed for 06-15/06-16/06-17,
but the heartbeat checker cron job was not firing. Root cause: likely Hermes scheduler
stall (Class C variant) — the heartbeat checker runs via Hermes scheduler, not Linux
crontab. When the Hermes scheduler stalls, production may continue (if triggered
manually or via a different path) while the monitoring layer goes completely dark.

**Key insight:** This is a META failure — the tool that DETECTS stalls is itself stalled.
The normal diagnostic chain (check receipt → find STALE agents → diagnose each) breaks
at step 1 because the receipt doesn't exist. You must fall back to direct artifact
inspection for every agent.

**Diagnostic recipe:**

```bash
# 1. Check receipt age
ls -lt agents/fleet_steward/memory/*_receipt.md | head -1
# If >24h old, heartbeat checker is stalled

# 2. Check if heartbeat checker has a Hermes cron job
# Use mcp_cronjob action='list' and look for heartbeat-related jobs
# If the job exists but last_run_at is null or stale → Hermes scheduler stall (Class C)

# 3. Check if it's in Linux crontab
crontab -l | grep heartbeat
# If absent, it's Hermes-scheduler-dependent

# 4. Verify production is still running despite monitoring gap
ls -lt data/snapshots/ | head -5
# If recent snapshots exist, production is functional — only monitoring is dark

# 5. Manually trigger heartbeat checker to restore visibility
python3 tools/agent_heartbeat_checks.py
# This produces a fresh receipt and reveals current fleet state
```

**Resolution:**
1. Manually run `tools/agent_heartbeat_checks.py` to produce a fresh receipt.
2. Check Hermes scheduler for the heartbeat job — if stalled, kick it:
   `mcp_cronjob action='run' job_id='<heartbeat_job_id>'`.
3. If the Hermes job doesn't exist (was removed), recreate it.
4. After fresh receipt lands, re-triage using normal fleet-triage workflow.

**Triage adaptation when receipt is stale:** When the receipt is >24h old, the triage
MUST use direct artifact inspection for every agent rather than relying on receipt
findings. The receipt's STALE/FAIL/WARN flags are all stale. Check `ls -lt artifacts/<agent>/`
and `ls -lt agents/<agent>/memory/` directly for each agent of interest.

---

### Class I — Hermes cron job silent failure (NameError / RuntimeError unnoticed for weeks)

**Signature:** A Hermes cron job shows `last_run_at` with an error status, but no alert
was generated. The job continues to appear in `mcp_cronjob action='list'` with the same
error, never recovering. Weeks pass with the job silently failing on every scheduled run.

**Confirmed instance (2026-06-17):** Two Hermes cron jobs found with stale errors:
- `a15dbdcb6f41` (weekly-skill-harvester): `NameError: name '_pool_may_recover_from_rate_limit' is not defined` — last run 2026-05-18 (30 days stale)
- `a955f533907b` (morning-briefing): `RuntimeError: WAKE ROBIN MORNING BRIEFING` — last run 2026-05-24 (24 days stale)

Both jobs had been failing silently for weeks. No operator alert was generated because
Hermes cron job failures are logged but not actively surfaced (no email, no openclaw
delivery, no escalation). The jobs remained in the scheduler with stale error states.

**Diagnostic recipe:**

```bash
# List all Hermes cron jobs and check for stale error states
# Use mcp_cronjob action='list'
# Look for:
#   - last_run_at > 7 days ago
#   - error field non-empty
#   - next_run_at in the past (job gave up retrying)
```

**Resolution:**
1. Identify the failing job(s) from the cron list.
2. Check `~/.hermes/cron/output/<job_id>/` for the error output.
3. Fix the underlying error (NameError = code bug in the job's prompt/skill;
   RuntimeError = the job's own error handling raised).
4. Re-trigger: `mcp_cronjob action='run' job_id='<job_id>'`.
5. Consider adding a monitoring check: any Hermes cron job with last_run_at > 3 days
   and non-empty error should be surfaced in daily triage.

**Prevention:** The fleet steward heartbeat checker should include a Hermes cron job
health check — list all jobs and flag any with stale errors. Currently the heartbeat
checker only inspects OpenClaw agents and repo artifacts, not Hermes cron jobs.

---

### Class J — Cron sys.path isolation (ModuleNotFoundError for repo imports)

**Signature:** Hermes/Linux cron entry script logs `ModuleNotFoundError: No module named 'tools'`
(or `common`) on every scheduled run. Interactive shell works because virtualenv activation
or `PYTHONPATH` masks the gap.

**Confirmed instance (2026-06-24):** `agents_direct` cron fired 42× before fix (735ac3f7).
Town may surface this as `cron_missed` before agents.log is checked.

**Fix pattern:**

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # adjust depth per script
sys.path.insert(0, str(PROJECT_ROOT))

from tools.skills_logger_v2 import ...  # now resolves
```

**Cross-ref:** `openclaw-data-pipeline-debug` Class P · `town-operator-bridge` triage table ·
`domains/agent_ops.md` · LRN-20260624-001.

---

## Quick-reference triage

```
Broad STALE (≥3 agents, last write ~17:30 ET)?
  → Class B. Run the log-mtime sweep first.

Broad STALE but syslog has user CMDs all day?
  → Class A. Find the REPLACE event in syslog.

Fleet receipt itself is >24h old?
  → fleet_steward's own cron slot missed (17:30 ET).
  → Also check Hermes scheduler (Class C).
  → If receipt is 5+ days stale AND production still running → Class H (heartbeat checker stall).

Hermes cron job failing silently for weeks (NameError, RuntimeError)?
  → Class I. Check mcp_cronjob list for stale error states. No alert is generated.

agents.log shows ModuleNotFoundError: No module named 'tools' on cron runs?
  → Class J. Cron sys.path isolation. Insert PROJECT_ROOT before repo imports.
  → Town cron_missed may be first signal — see town-operator-bridge triage table.

data_auditor FAIL on a weekend date?
  → Class E. Known false-positive. Disregard.

Watchdog recovering same agents every 30 min?
  → Class D. Check the idempotency marker.

Hermes jobs show last_run_at null despite Linux cron alive?
  → Class C. Separate scheduler.

2+ agents show auth errors / EXPIRED+DRIFT cluster?
  → Class C first. Check auth-sync cron last_run_at before anything else.
  → If stalled >8h: run openclaw-auth-sync manually, kick cron 4cfe9fb5d466.

  OpenClaw jobs show 20+ consecutive delivery errors, mode=announce?
  → Class G. Add --best-effort-deliver to each affected job.
```

---

## Support files

- `references/biotech-screener-git-branch-hygiene.md` — git hygiene context
- `references/openclaw-announce-webchat-jobs-2026-05-05.md` — confirmed job IDs and state for the Class G webchat delivery incident; includes a one-liner to verify bestEffort:true is still set on all announce jobs

### Class G — announce/webchat delivery errors (channel not resolvable in isolated sessions)

**Signature:** OpenClaw cron jobs configured with `delivery.mode: "announce"` and
`channel: "webchat"` accumulate 20+ consecutive errors with message:
"Channel is required (no configured channels detected). Set delivery.channel
explicitly or use a main session with a previous channel."

Jobs with `delivery.mode: "none"` are unaffected and run cleanly.

**Root cause:** OpenClaw cannot resolve the "webchat" channel ID in isolated cron
sessions — it needs an active dashboard WebSocket session open at completion time.
These jobs were never delivering; they were failing immediately after execution.

**Confirmed instance (2026-05-05):** 7 jobs (ops-daily, sentinel-daily,
daily-production-brief, ops-digest-summary, dashboard-validation-ping,
calibration-weekly, weekly-policy-review) all at 20-21 consecutive errors.

**Resolution — choose based on whether you need webchat output:**

Option A — `--no-deliver` (preferred for internal ops-only jobs):
Strips announce delivery entirely. Jobs run cleanly, no channel lookup attempted.
Right choice when nobody is reading webchat output from these crons.

```bash
openclaw cron edit <job-id> --no-deliver
```

Verify: `openclaw cron list` should show `not requested (not requested)` in the delivery column.

Option B — `--best-effort-deliver` (preferred if you want opportunistic delivery):
Jobs continue attempting webchat delivery but do not FAIL when it can't resolve.
Run counts as success. Consecutive error counter resets on next execution.
Fires if dashboard is open; silently skips if not.

```bash
openclaw cron edit <job-id> --best-effort-deliver
```

**Confirmed working (2026-05-07):** All 7 affected jobs patched with `--no-deliver`
in a loop. Delivery column flipped from `announce -> webchat (Channel is required...)`
to `not requested (not requested)`. Jobs will run cleanly on next fire.

**Verification step:** after patching, run `openclaw cron list | grep '<job-name>'` and confirm
the delivery column reads `not requested (not requested)`. Historical runs still show `error`
until the next scheduled execution; do not misread that as a failed remediation.

**Do NOT use `--channel none`** — OpenClaw rejects it with "Unsupported channel: none".
**Do NOT use `--channel last`** — flaky in isolated cron; tends to reproduce the error
with a different message.

---

## Related skills

- `biotech-screener-catchup-hardening` — load this when moving from diagnosis to remediation. Covers writing resilient catch-up scripts, idempotency patterns, weekend backstops, and the SMTP-creds hallucination pattern. Complements this skill: this skill diagnoses, that skill hardens.
- `openclaw-fleet-triage` — load first for daily triage. This skill is the drilldown when triage suspects a cron/scheduler root cause.
- `openclaw-agent-scope-audit` — load when Class A (crontab REPLACE) or Class D (watchdog loop) points to a mis-configured agent prompt string rather than a pure cron issue.

## Pitfalls

- **Check weekday before concluding production missed.** Observed 2026-05-04: manual
  backfill attempted for 2026-05-02 (Saturday). cron_daily_production.sh correctly refused
  with "SKIP: 2026-05-02 is a weekend (day 6)". Any date returning 6 or 7 from
  `date -d YYYY-MM-DD +%u` is not a production date and has no missing snapshot by design.
  ```bash
  python3 -c "import datetime; d=datetime.date.fromisoformat('YYYY-MM-DD'); print('WEEKDAY' if d.weekday()<5 else 'WEEKEND — no production expected')"
  ```

- **data/snapshots/ absence on a weekend date is NORMAL.** Do not attempt backfill.
  The catch-up script also guards against this — if it says "no missed runs found"
  for a date range containing only weekends, that is correct behavior.

```bash
# 1. Backup (timestamped)
crontab -l > "$HOME/crontab.bak.$(date +%Y%m%d_%H%M%S)"

# 2. PREVIEW — process substitution, no side effects
diff <(crontab -l) <(crontab -l | sed -E '...')
# STOP — show operator the diff. Wait for "apply" confirmation.

# 3. APPLY — identical pipeline from step 2 (only after explicit approval)

# 4. VERIFY
crontab -l | grep '<changed entry>'
```
