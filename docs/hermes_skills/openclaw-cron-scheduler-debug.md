---
name: openclaw-cron-scheduler-debug
description: "Diagnose cron and scheduler failures for the OpenClaw biotech screener fleet. Encodes real failure patterns: crontab REPLACE silent death, WSL2 sleep cliff, Hermes scheduler stall, watchdog idempotency loops, announce/webchat delivery errors, Hermes job token bloat from pre-loaded skills (Class F), sleep-cliff multi-firing without idempotency guard (Class G), cron job script-writing retry loop (Class H)."
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

## Class J — Pre-push guard for main (defense-in-depth, 2026-06-22)

**Signature:** Non-interactive `git push` to `main` is blocked by a local git hook.
Agents/cron/CI that attempt to push to main will fail with "Non-interactive push to
main blocked" unless `ALLOW_AGENT_PUSH=1` is set in the environment.

**Confirmed instance (2026-06-22, commit `ded2d3b0`):**
- Hook: `tools/githooks/pre-push` (installed via `tools/githooks/install-hooks.sh`)
- Trigger: stderr is not a TTY (agent/cron/CI) AND target branch is `main`
- Escape hatch: `ALLOW_AGENT_PUSH=1 git push` bypasses the check
- Chains to preserved git-lfs pre-push hook (`pre-push.lfs-orig`)
- Context: INC-2026-06-20-AUTOPUSH defense-in-depth (work-plan step 3). Free GitHub
  plan = no server-side branch protection, so this is the local backstop.

**Diagnostic recipe:**

```bash
# 1. Check if the hook is installed
ls -la .git/hooks/pre-push
# If symlink to ../../tools/githooks/pre-push → installed

# 2. Check if it would block (dry test)
echo "" | ALLOW_AGENT_PUSH=0 bash tools/githooks/pre-push origin https://github.com/test/test 2>&1
# Should print "Non-interactive push to main blocked"

# 3. If an agent push failed unexpectedly, check for the escape hatch
grep ALLOW_AGENT_PUSH tools/githooks/pre-push
```

**Triage implication:** If fleet agents or cron jobs need to push to main (e.g., the
skill harvester before manualization), they must set `ALLOW_AGENT_PUSH=1` in the
environment. After the harvester manualization (commit `37111ad4`), no Hermes cron
job pushes to main — the hook is a pure backstop.

**Do NOT remove this hook** without explicit operator approval. It is the last line
of defense against accidental agent pushes to main on a free-plan repo (no server-side
branch protection).

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

git push to main fails from agent/cron context?
  → Class J. Pre-push guard blocking non-interactive pushes.
  → Set ALLOW_AGENT_PUSH=1 or use interactive terminal.

A merged PR's fix keeps recurring in daily production output?
  → Class M. Check checkout drift before re-debugging: `git rev-list --count HEAD..origin/main`.

OpenClaw agents show 5+ days dormant, no cron active?
  → OpenClaw is FENCED (LEGACY_READ_ONLY_DORMANT) as of 2026-06-22.
  → See Class I in openclaw-agent-scope-audit. Hermes is primary orchestrator.
```

---

## Class K — LangGraph review node artifact_dir None guard (third-runtime pattern)

**Signature:** LangGraph-based review nodes crash with AttributeError or TypeError when
`artifact_dir` is None. The LangGraph runtime (third alongside Hermes and OpenClaw) has
different null-handling expectations than the other two runtimes.

**Root cause (confirmed 2026-06-22, commit `afced5d4`):** The LangGraph review node
assumed `artifact_dir` was always a valid path string. When the node was invoked in
contexts where no artifact directory was configured (e.g., diagnostic-only runs, certain
cron triggers), `artifact_dir` was None, causing the node to crash when attempting path
operations.

**Runtime boundary context:** Commit `96ffea36` (2026-06-22) established the
"Hermes/OpenClaw/LangGraph runtime boundary map" — LangGraph is now a third runtime
alongside Hermes (cron-managed agents) and OpenClaw (gateway-managed agents). Each
runtime has different null-handling, tool-execution, and artifact-path conventions:

- **Hermes**: cron-managed, tool-execution via Hermes agent loop, artifacts at
  `artifacts/<agent>/` or agent-specific paths. Null args typically default to None
  and agents handle gracefully.
- **OpenClaw**: gateway-managed, real bash tool execution, artifacts at declared paths
  in AGENTS.md. Null args cause tool-execution failures that agents can diagnose.
- **LangGraph**: graph-based workflow, node-level execution, artifacts at node-declared
  paths. Null args cause node crashes (AttributeError/TypeError) unless explicitly
  guarded. See `docs/governance/runtime_boundary_map.md` for the full matrix.

**Diagnostic chain:**

```bash
# 1. Check if the crash is in a LangGraph node (not Hermes/OpenClaw agent)
grep -r "langgraph\|StateGraph\|node.*artifact_dir" tools/ agents/ --include="*.py" | head -10

# 2. Check the node's artifact_dir parameter handling
grep -B2 -A5 "artifact_dir" tools/<langgraph_tool>.py | head -20
# If the node does path operations (e.g., artifact_dir / "file.json") without
# checking for None first, this is the Class K pattern.

# 3. Check the invocation context
# LangGraph nodes can be invoked from multiple contexts (cron, manual, diagnostic).
# Some contexts may not configure artifact_dir. Check the call site.
```

**Resolution pattern:**
- Add explicit None guard at the node entry point: `if artifact_dir is None: return {"status": "skip", "reason": "no artifact_dir configured"}`
- Or provide a default: `artifact_dir = artifact_dir or Path("artifacts/default_review")`
- The fix should match the node's contract — if the node is designed to work without
  artifacts in some contexts, return a skip status; if it always needs artifacts, raise
  a clear error.

**Detection heuristic:** If a LangGraph node crashes with AttributeError/TypeError on
path operations and the traceback points to `artifact_dir` being None, this is Class K.
Check the runtime boundary map to confirm the node is LangGraph (not Hermes/OpenClaw).

**Note:** LangGraph is a newer runtime in this ecosystem (admitted 2026-06-22 via
Package B governance review). Skills and diagnostic patterns are still being built out.
When encountering LangGraph-specific failures, check `docs/governance/runtime_boundary_map.md`
for the runtime's conventions before applying Hermes/OpenClaw diagnostic patterns.

---

## Class L — Financial calculation unit-mismatch (periodicity confusion)

**Signature:** Financial metrics (burn rate, runway, cash consumption) are silently wrong
by a factor of 3-4×. The calculation runs without error, produces plausible-looking numbers,
but the underlying periodicity assumption is wrong (quarterly divisor applied to annual data,
or annual divisor applied to quarterly data).

**Root cause (confirmed 2026-06-23, commit `c6e1700c`):** Module 2 (financial health) had
two fallback burn-rate calculation paths in NetIncome and R&D expense handling. Both paths
hard-coded `/3` (quarterly assumption) when the SEC filing data was actually annual (Dec
fiscal year-end 10-K). This overstated monthly burn 4× and understated runway 4× for tickers
reaching these last-resort fallback paths.

**Confirmed instance:** Tickers with SEC annual filings (fiscal year-end December) reaching
the NetIncome or R&D fallback paths. The fix ported `_ytd_months_from_date()` from v1 to
use the `NetIncome_date` / `R&D_date` fields (already passed through in financial_data) to
infer the correct period. Default remains 3 when date is missing (no behavior change for
existing data without date fields).

**Diagnostic chain:**

```bash
# 1. Check if financial metrics look plausible
python3 - << 'EOF'
import json, glob
files = sorted(glob.glob('data/snapshots/*/rankings.csv'), reverse=True)
if not files: exit()
import csv
with open(files[0]) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 5: break
        burn = row.get('monthly_burn_rate_mm')
        runway = row.get('runway_months')
        cash = row.get('cash_and_securities_mm')
        if burn and runway:
            print(f"{row.get('ticker')}: burn={burn}, runway={runway}, cash={cash}")
            # Sanity check: runway ≈ cash / burn
            try:
                expected_runway = float(cash) / float(burn) if float(burn) > 0 else None
                if expected_runway and abs(expected_runway - float(runway)) > 0.5:
                    print(f"  ⚠️  Runway mismatch: expected {expected_runway:.1f}, got {runway}")
            except:
                pass
EOF

# 2. Check the calculation code for hardcoded divisors
grep -n "/3\|/12\|_ytd_months\|period.*month" tools/financial_health.py | head -20
# Look for: hardcoded /3 or /12 without checking the actual filing period

# 3. Check if date fields are used to infer periodicity
grep -n "NetIncome_date\|R&D_date\|fiscal_year_end\|period_end" tools/financial_health.py | head -20
# If date fields exist but aren't used to determine the divisor, this is the Class L pattern

# 4. Verify the fix (post-c6e1700c)
grep -A10 "_ytd_months_from_date" tools/financial_health.py | head -15
# Should see: uses the date field to infer months, not hardcoded /3
```

**Pattern to catch:** Any financial calculation that divides by a hardcoded number (3, 4, 12)
to convert between periodicities (annual↔quarterly↔monthly) WITHOUT checking the actual
filing period or date field is suspect. SEC filings have mixed periodicity:
- 10-K (annual) → divide by 12 for monthly
- 10-Q (quarterly) → divide by 3 for monthly
- If the code doesn't distinguish, it will silently produce wrong results

**Test coverage added (commit c6e1700c):** 2 new golden cases verify the annual path
(Dec date → /12). 136/136 module_2 tests pass; 44/44 golden tests pass.

**Impact on recent screens:** Any screen run between the Module 2 activation and c6e1700c
had burn rates overstated 4× and runways understated 4× for tickers reaching the fallback
paths. Use `ls -lt data/snapshots/*/rankings.csv` to find affected dates; the fix produces
correct burn/runway for annual filers.

**General rule:** When financial metrics look implausible (runway < 6 months for a company
with $100M+ cash, or burn rate 4× higher than peers), check the periodicity assumption in
the calculation code. The bug is silent — no exception, no warning, just wrong numbers.

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

### Class F — Hermes job token bloat from pre-loaded skills

**Signature:** A single cron session consumes millions of tokens. `hermes insights --days 7`
shows one job's session dramatically above all others (e.g. 6.1M tokens in one run).
The job's `jobs.json` entry has a non-null `"skill"` field or a non-empty `"skills"` array.

**Root cause:** The `"skill"` field in `~/.hermes/cron/jobs.json` injects a full SKILL.md
as the system prompt for the session. The `"skills"` array pre-loads additional skills into
context before the first user turn. Large skills (openclaw-fleet-triage = ~100K chars / ~25K
tokens) are loaded even when irrelevant to the job's actual task.

**Confirmed instance (2026-06-25):** `weekly-skill-harvester` (a15dbdcb6f41) consumed 6.1M
tokens in one session. Root causes:
- `"skill": "openclaw-fleet-triage"` — loaded the full 100K-char fleet triage skill as primary
  instruction, causing the job to run a complete fleet triage before harvesting skills
- `"skills": ["openclaw-fleet-triage", "openclaw-cron-scheduler-debug", ...]` — 6 skills
  pre-loaded totaling ~286K chars / ~72K tokens upfront
- `weekly-signal-regime-sweep` also loaded `openclaw-fleet-triage` in its `skills` array
  despite being a signal regime check with no fleet triage need

**Token hog audit recipe:**

```python
import json
with open('/home/arrenchulz/.hermes/cron/jobs.json') as f:
    data = json.load(f)

for job in data['jobs']:
    skill = job.get('skill')
    skills = job.get('skills', [])
    if skill or skills:
        print(f"{job['id']} {job['name']}")
        print(f"  skill={skill!r}, skills={skills}")
```

Then check skill sizes:
```bash
find ~/.hermes/skills -name "SKILL.md" | xargs wc -c | sort -rn | head -10
```

**Fix:**
1. Set `"skill": null` — removes the primary skill system-prompt injection
2. Set `"skills": []` — removes all pre-loaded context skills
3. Rewrite prompt to load skills lazily: add `skill_view(name='<skill-name>')` call in the
   step that actually needs it, rather than front-loading everything

**Patch via Python (safe, preserves all other fields):**
```python
import json
with open('/home/arrenchulz/.hermes/cron/jobs.json') as f:
    data = json.load(f)
for job in data['jobs']:
    if job['id'] == '<job-id>':
        job['skill'] = None
        job['skills'] = []
        # also update prompt to use lazy skill_view() calls
with open('/home/arrenchulz/.hermes/cron/jobs.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

**Pitfall:** `jobs.json` top-level is `{"jobs": [...]}`, not a bare list. Iterating `data`
directly (without `data['jobs']`) raises `AttributeError: 'str' object has no attribute 'get'`.

**Known large skills (as of 2026-06-25):**
- `openclaw-fleet-triage`: ~100K chars / ~25K tokens — PINNED, never auto-patch
- `biotech-screener-audit`: ~59K chars / ~15K tokens
- `aa-model-tracker`: ~42K chars / ~11K tokens
- `biotech-screener-output-qa`: ~29K chars / ~7K tokens

---

### Class G — Sleep-cliff job multi-firing (no idempotency guard)

**Signature:** A weekly or once-daily Hermes cron job shows 3–5 sessions on the same day in
`hermes insights`. No `[SILENT]` in any output. Each session ran the full workload.

**Root cause:** WSL2 sleep cliff causes the cron watchdog to re-trigger missed jobs on wake.
If the job has no idempotency guard it runs in full each time, multiplying token cost.

**Confirmed instances (2026-06-24):**
- `weekly-signal-regime-sweep` (7e79501afb6e): 3 sessions Jun 24, each running full sweep
- `event-outcome-binder-watch` (f7635b487132): 3 sessions Jun 24, each running full report
- `weekly-skill-harvester` (a15dbdcb6f41): ran multiple times due to sleep cliff catchup

**Fix — STEP 0 idempotency guard (prepend to job prompt):**

For jobs that write a dated output file:
```
STEP 0 — IDEMPOTENCY CHECK
Run: date +%Y-%m-%d
Check for recent output:
  ls -t ~/.hermes/cron/output/<job-id>/ 2>/dev/null | head -1
Extract date from filename (format: YYYY-MM-DD_HH-MM-SS.md).
If that date is today, respond with exactly [SILENT] and stop.
```

For jobs that write a repo artifact:
```
STEP 0 — IDEMPOTENCY CHECK
Run: date +%Y-%m-%d
  ls -t /mnt/c/Projects/biotech_screener/biotech-screener/<artifact-path>*.md 2>/dev/null | head -1
If a file exists with today's date (or within N days for weekly jobs), respond [SILENT] and stop.
```

**Rule for N (window size):**
- Daily jobs: N = same-day check (today only)
- Weekly jobs: N = 5–7 days (prevents re-run across the full schedule window)

**`[SILENT]` behavior:** Hermes cron treats `[SILENT]` as a suppressed-delivery no-op run.
The session completes, `last_run_at` updates, but no output is delivered to the user.
Combine with the guard: if the guard fires, respond with ONLY `[SILENT]` — no other text.

---

### Class H — Cron job script-writing retry loop

**Signature:** A normally lightweight cron job (< 300K tokens/run) spikes to 1M+ tokens.
Job has terminal access. Output log shows repeated write_file + SyntaxError + fix cycles.

**Confirmed instance (2026-06-24):** `pdufa-proximity-alert` (e84535b22a2a) hit 1.19M tokens
in one session. It wrote a Python filtering script, encountered a SyntaxError, tried to fix it,
repeated the cycle. Normal runs are ~200K tokens.

**Root cause:** Agent has `terminal` toolset and no constraint against writing scripts.
When faced with data filtering, it writes a Python script instead of using read_file + in-memory
manipulation or available MCP tools.

**Fix — add constraint to job prompt:**
```
CONSTRAINT: Do NOT write any Python scripts or temporary files at any point.
Read files directly with read_file and process data in-memory.
Do not use write_file to create helper scripts. If data filtering is needed, do it inline.
```

**Also add idempotency guard** (Class G) so a spike run doesn't trigger additional catch-up runs.

---

## Class M — Production checkout drift behind origin/main (merged fixes silently inert)

**Signature:** A PR merges a fix to `main`, but the production cron's local checkout
does not auto-pull. The fix does not take effect until the checkout is fast-forwarded.
No error is thrown — cron runs green using stale code, and the bug the PR fixed keeps
reproducing in daily runs as if the fix never landed.

**Confirmed instance (2026-07-08, commit `b79d1940`, issue #484):** The `forward_eval`
IC-scoping fix (PRs #474/#475) and other merged fixes did not take effect in production
runs because `tools/cron_daily_production.sh` checks out a fixed local clone that is
never fast-forwarded automatically. Diagnosed after repeated recurrence of already-"fixed"
bugs in daily monitoring output.

**Fix applied:** Added a non-blocking drift guard to `cron_daily_production.sh` — fetches
`origin/main` (30s timeout, tolerates offline) at run start, computes commits-behind via
`git rev-list --count HEAD..origin/main`, and logs INFO (1-5 behind) or WARN (>5 behind,
`DRIFT_WARN_THRESHOLD=5`) with the local HEAD short-hash. Logs only — never aborts the run.

**Diagnostic recipe:**

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
git fetch --quiet origin main
git rev-list --count HEAD..origin/main   # commits behind — nonzero means drift
git rev-parse --short HEAD               # what the checkout is actually running
grep -A3 "Drift guard" logs/cron.log | tail -10   # confirm guard is firing
```

**Pattern to watch for:** If a bug that was supposedly fixed in a merged PR keeps
recurring in production output, check checkout drift BEFORE re-debugging the fix itself —
the fix may be correct but simply not live. This is a distinct failure mode from Class A
(crontab REPLACE) and Class C (Hermes scheduler stall): the scheduler is firing correctly,
the job is running correctly, but on stale code.

**Resolution:** `git pull --ff-only` (or equivalent fast-forward) on the production
checkout. The drift guard added in `b79d1940` surfaces this at every run start going
forward so it can't recur silently again.

---

## Class O — Watchdog false-positive recovery before scheduled production time (pre-market snapshot pollution)

**Signature:** Watchdog fires on WSL2 boot (~09:00 ET), sees no snapshot for today, and
treats that as a MISSED run — triggering an immediate manual recovery that produces a
**pre-market** snapshot hours before the real 16:30 ET scheduled window. The real
post-close run then finds "already done" and skips regeneration, so the day's snapshot
is built on stale/incomplete same-day prices. Symptoms surface as several apparently
unrelated gate warnings simultaneously: `pnl_attribution` coverage 0.0% (price_d1 null
for all positions, n_priced=0), `market_data` a day stale, and `drift_monitoring`
next_earnings_date coverage dropping ~20pp.

**Confirmed instance (2026-08-09, commit `e8a56c0a`, PR #566):**
`fix(watchdog): a run that is not due yet is not a missed run`. Production start times
that week were 08:37, 09:05, 09:12, 09:54, 09:05 — none the scheduled 16:30. Thursday
passed only by accident (that day's run happened to land at 14:52, after prices existed).

**Diagnostic chain:**

```bash
# 1. Check actual production start times vs scheduled 16:30 ET
grep "Running daily_production" logs/watchdog.log | tail -10

# 2. Check if the guard exists
grep -n "PROD_DUE_HHMM\|not due" tools/cron_watchdog.sh

# 3. Cross-check the three co-occurring symptoms if guard is absent
#    - pnl_attribution coverage 0.0%
#    - market_data age_days >= 1
#    - drift_monitoring next_earnings_date coverage drop
# All three together on a weekday = likely pre-market snapshot pollution, not
# three independent bugs.
```

**Resolution (confirmed working):** Guard the watchdog's production-recovery branch on
scheduled time. Before `PROD_DUE_HHMM` (default 1630) an absent snapshot is "not due",
not "missed" — no recovery fires. At/after that time, behavior is unchanged (recovery
still fires for genuinely missed runs). Override via `PROD_DUE_HHMM` env var for testing
or a changed schedule. Deliberately narrow — does not touch the watchdog's other
recovery paths (phase-2 artifacts, heartbeat receipt, ops_supervisor, herald, bellringer,
trapops).

**Pattern to catch:** Any watchdog/recovery mechanism that infers "missed" purely from
absence-of-artifact without checking whether the artifact's scheduled generation time has
actually passed will false-positive on any early trigger (boot, WSL2 wake, manual kick).
The false recovery's output can itself satisfy the real job's own idempotency check,
silently suppressing the correct run for the rest of the day.

---

## Class N — Mandate evidence lost to git-state interference during production cron (Spec 115)

**Signature:** Forward-validation "mandate-eligible" evidence windows go missing or get
silently overwritten for specific trading days, with no script error — the production
run itself succeeds (exit 0) but the evidence ledger used to certify mandate-eligible
forward-validation windows loses entries. Root cause is NOT the model/data; it's the
cron path's git interaction stepping on the evidence file mid-run.

**Confirmed instance (Spec 115, commit `dd13e097`, PR #540, merged 2026-07-29):**
`fix(fv): protect mandate evidence from git-state interference (Spec 115 P1+P2a)`.
Documented root cause (per `docs/audit/FV-GAP-2026-07-27.md`, referenced in Spec 115):
2 of 10 trading days in a 07-15→07-28 window lost mandate-eligible forward-validation
windows to git mechanics — i.e., a concurrent git operation (checkout, stash, or reset
in the same working tree used by the cron path) interfered with the evidence-capture
step. Phase 1 (early-bind/diagnose git-state before capture) and Phase 2a (visibility —
WARN on detected interference) landed in this commit. Phase 2b (commit-after-capture)
and Phase 3 (advisory lock) remain open, tracked separately (see held-spec ledger).

**Diagnostic chain:**

```bash
# 1. Check for the git-state guard introduced by Spec 115
grep -n "git-state\|_prior_dir\|mandate.*evidence\|git_lock\|advisory.lock" \
  tools/run_daily_production.py tools/*forward_eval*.py 2>/dev/null | head -20

# 2. Check whether a given historical date's evidence window is present
grep -n "<YYYY-MM-DD>" artifacts/readiness/forward_eval_ic_baseline.json 2>/dev/null

# 3. Read the spec for the full P1/P2a/P2b/P3 phase breakdown
cat specs/changes/spec_115_production_run_git_lock_*.md 2>/dev/null | head -60
```

**Pattern to catch:** If any cron-managed pipeline step both (a) reads/writes a
persistent evidence ledger AND (b) runs in a working tree that other agents/processes
may `git checkout`/`stash`/`commit` in concurrently, the evidence step needs either an
early git-state check (bail/WARN if dirty or mid-operation) or to be relocated outside
git entirely. This is distinct from Class M (checkout drift behind origin/main) — Class M
is about running stale code; Class N is about a live git operation corrupting evidence
capture *during* the run. Do not conflate the two when triaging forward-validation gaps.

**Governance note:** Do not treat Phase 2a's WARN-only visibility as remediation — it
detects the problem, it does not yet prevent it (Phase 2b/3 pending explicit operator
decision on 2a-relocate-ledger vs 2b-commit-after-capture approach).

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

- **CLI flag typos cause silent argparse crashes.** Confirmed 2026-06-22 (commit `4e6815ed`):
  snapshot generator had `--as-o` instead of `--as-of` in the CLI flag definition. argparse
  silently accepted the truncated flag but it didn't match the expected parameter name,
  causing the tool to crash when invoked with `--as-of <date>`. Diagnostic: if a tool
  suddenly stops producing output with no log entries and no error, check the argparse
  flag definitions for truncation or typo. `python3 tools/<tool>.py --help` will show
  the actual registered flags — compare against the crontab invocation.

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
