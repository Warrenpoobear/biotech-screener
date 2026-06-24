# AGENTS.md — Universe Maintenance Agent

## Session startup

1. Read `SOUL.md` — identity and boundaries
2. Read `TOOLS.md` — data sources and output paths
3. Load current universe.json

## Weekly sequence

1. Check for delistings, acquisitions, ticker changes
2. Check price coverage — flag tickers with stale or missing prices
3. Check market data coverage — flag tickers with missing fundamentals
4. Flag any tickers that should be reviewed for addition or removal
5. Write report JSON + MD

## Memory protocol

Write session summaries to `agents/universe_maintenance/memory/`.
Track: universe size, tickers flagged, coverage gaps.

## Self-learning (Rule 12)

Recurring delist/coverage gap → `.learnings/LEARNINGS.md`.

## Red lines

- Do not modify universe.json or any production data
- Do not add or remove tickers without human approval
- Do not modify scoring, rulesets, or rankings
