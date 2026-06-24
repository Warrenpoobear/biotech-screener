# AGENTS.md — Ops Agent

## Session startup

1. Read `SOUL.md` — your identity and boundaries
2. Read `TOOLS.md` — commands and daily working set
3. Read `memory/YYYY-MM-DD.md` (today + yesterday) if they exist

## Daily sequence

1. Check if today's production ran: `ls data/snapshots/$(date +%Y-%m-%d)/`
2. If not, and it's a weekday: run the production pipeline (see TOOLS.md)
3. Read the ops digest: `artifacts/ops_digest/YYYY-MM-DD_digest.md`
4. Compare today's action items against the prior digest
5. Report: NEW items, RESOLVED items, UNCHANGED items
6. Flag any A-tier names with <=7 days to catalyst that are new since prior
7. One-line shadow performance: cumulative, excess vs XBI, Sharpe, periods

## Memory

Write daily notes to `memory/YYYY-MM-DD.md`. Keep it concise:
- Action items surfaced and their resolution status
- Any anomalies observed (missing data, gate failures)
- Digest attention level (CLEAR / REVIEW / ACTION_REQUIRED)

## Self-learning (Rule 12)

When a recurring ops pattern appears (≥2× in 7d):
1. Search `docs/FAILURE_PATTERN_LIBRARY.md` for an existing pattern ID
2. Append `.learnings/LEARNINGS.md` with `Promotion-lane: skill`
3. Run `python3 tools/herald_health_check.py` when Herald-related

## Red lines

- Do not edit `.py` files, scoring logic, rulesets, or manifest
- Do not `git push`, `git commit`, or modify tracked files
- Do not delete snapshots, caches, or position history
- Do not promote shadow candidates or change the active ruleset
- Do not run commands with `--ruleset` override
- When in doubt, report the issue and wait for human decision

## Heartbeats

On heartbeat, follow `HEARTBEAT.md` strictly. Reply `HEARTBEAT_OK`
if all checks pass. Only surface issues.
