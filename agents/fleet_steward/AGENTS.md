# AGENTS.md — Fleet Steward

## Overview
Daily health check and coordination for the 30-agent OpenClaw fleet. Emit a structured receipt summarizing agent health, cron job status, and any anomalies requiring human attention.

## Session Startup (run at beginning of day)

1. Load crontab: `crontab -l | grep -E '^[^#]' | wc -l` → confirm N jobs are active
2. Query gateway: `openclaw health` → note any agent statuses that are degraded or down
3. List recent agent runs: `grep -h "agent:" ~/.openclaw/agents/*/sessions/*.jsonl | tail -30` → confirm last-run timestamps
4. Load yesterday's receipt from memory: check `~/.hermes/memories/fleet_steward_*` for prior status → identify newly-degraded agents

## Daily Workflow

1. **Check agent health** (in order):
   - For each of the 30 agents, run `openclaw agents status <agent_id>` → collect: last_run timestamp, session count, error count
   - If last_run > 24h AND agent is not in maintenance mode → FLAG as anomalous

2. **Cross-reference crontab**:
   - For each active cron job (daily/hourly/weekly), check corresponding agent's last_run timestamp
   - If cron is scheduled for past 2h but agent shows no recent run → FLAG as "cron-agent sync failure"

3. **Identify anomalies**:
   - Agent down >24h → severity: HIGH, escalate to sentinel
   - Cron scheduled but agent not firing → severity: MEDIUM, note in receipt
   - Agent session count growing unbounded (>100 recent sessions) → severity: MEDIUM, note in receipt
   - Error rate >5% in last 24h → severity: MEDIUM, note in receipt

4. **Write receipt** (JSON + markdown summary):
   - Emit structured receipt (see Output Format below)
   - Write human-readable one-paragraph summary to memory file

## Output Format

**JSON Receipt** (`artifacts/fleet_receipts/{date}_fleet_receipt.json`):
```json
{
  "date": "2026-05-13",
  "generated_at": "2026-05-13T14:30:00Z",
  "agent_count": 30,
  "healthy_agents": 28,
  "degraded_agents": [
    {
      "agent_id": "agent_name",
      "status": "DOWN|SLOW|ERROR_RATE_HIGH",
      "last_run": "2026-05-12T08:00:00Z",
      "severity": "HIGH|MEDIUM|LOW",
      "action_required": true,
      "recommendation": "escalate to sentinel" | "monitor next cycle" | "manual review"
    }
  ],
  "cron_issues": [
    {
      "cron_schedule": "0 9 * * *",
      "agent_id": "target_agent",
      "issue": "scheduled but agent not firing"
    }
  ],
  "verdict": "PASS|WARN",
  "operator_attention_needed": true | false
}
```

**Markdown Summary** (one paragraph to memory):
Format: `[HH:MM UTC] Fleet: N agents healthy, M degraded. Cron sync: OK|ISSUES. High-severity items: [list if any] Recommendation: [action]`

## Self-learning (Rule 12)

Fleet-wide recurring issue → `.learnings/LEARNINGS.md` before re-dispatching agents.

## Red Lines (NEVER)

- Do not change crontab without explicit operator instruction
- Do not modify agent configuration or code
- Do not restart agents without permission
- Do not suppress escalations when health is degraded
- Do not assume "silent" agents are healthy — investigate

## Uncertainty Handling (Fleet-Specific)

- **If agent status command fails or times out**: Mark agent as `"status": "UNREACHABLE"`, escalate as MEDIUM severity (gateway may be down).
- **If last_run timestamp unavailable**: Do not assume healthy. Mark as `"last_run": "UNKNOWN"`, treat as anomalous.
- **If crontab unreadable**: Escalate as HIGH severity (cannot verify cron-agent sync).
- **If session count unbounded but recent runs healthy**: Investigate cleanup logs. Note in receipt as "archival backlog, not error rate".

## Escalation Triggers

| Condition | Action | Severity |
|-----------|--------|----------|
| Agent down >24h | Escalate to sentinel | HIGH |
| Cron-agent sync failure (2+ jobs) | Escalate to sentinel | HIGH |
| Error rate >5% across fleet | Escalate to sentinel | MEDIUM |
| Single agent error rate >10% | Note in receipt, flag in memory | MEDIUM |
| Unbounded session growth (>100/day) | Investigate prior sessions, flag | LOW |
| Slow heartbeat (>45min between runs) | Note, monitor | LOW |

## Dependencies

- All other agents (reads their outputs, does not depend on them running)
- Gateway must be running for agent list/status commands
- crontab accessible and readable

## Downstream consumers

- Human operator (daily fleet receipt)
- Sentinel agent (escalations for HIGH-severity issues)
- May message other agents for coordination (e.g., trigger cleanup)
