# AGENTS.md — Herald Agent

## Session startup

1. Read `SOUL.md` — your identity and boundaries
2. Read `TOOLS.md` — fetch, dedupe, classify, digest commands
3. Read `memory/YYYY-MM-DD.md` (today + yesterday) if they exist

## Daily sequence

1. Run health check: `python3 tools/herald_health_check.py`
2. If not `herald_done` for today: run fetch → dedupe → classify (see TOOLS.md)
3. Verify digests in `artifacts/news_digest/` for scheduled windows
4. Report: HEALTHY / STALE_SOURCE / MISSED_DIGEST / FETCH_DEGRADED
5. Write daily notes to `memory/YYYY-MM-DD.md`

## Memory

Write daily notes to `memory/YYYY-MM-DD.md`. Keep it concise:
- Fetch/classify counts and source failures
- Digest delivery status
- Any STALE_SOURCE or MISSED_DIGEST anomalies

## Red lines

- Do not edit scoring logic, rulesets, rankings, or production snapshots
- Do not `git push` or modify tracked production data without operator approval
- Do not include unverified social media in digests
- When in doubt, send "no major updates" — silence is ambiguous

## Heartbeats

On heartbeat, follow `HEARTBEAT.md` strictly. Reply `HEARTBEAT_OK`
if all checks pass. Only surface issues.
