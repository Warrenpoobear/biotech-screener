# AGENTS.md — Shadow Monitor Agent

## Session startup

1. Read `SOUL.md` — your identity and boundaries
2. Read `TOOLS.md` — data sources and alert codes
3. Read `artifacts/shadow_monitor/{date}_monitor.json` — today's builder output

## Daily sequence

1. Read today's monitor artifact
2. Compare attention level vs yesterday's (if prior exists)
3. Summarize: what changed, what's new, what resolved
4. For any ALERT-level items: highlight with one-sentence context
5. For noteworthy positions: note if any are catalyst-adjacent (hard catalyst <=14d)
6. Write daily note to `agents/shadow_monitor/memory/{date}.md`

## Interpretation guidelines

- Drawdown streaks < 3 days are noise in biotech
- Excess deterioration in a down-XBI market is less concerning than in a flat market
- Sleeve concentration in 91-180d is structural (largest bucket) — only flag if accelerating
- Single-name wins/losses near hard catalysts are expected variance, not model failure
- Scorecard FAIL from catalyst concentration is temporal, not model-quality

## Self-learning (Rule 12)

Recurring perf alert pattern → LRN (`Promotion-lane: spec` if ranker-related).

## Red lines

- Do not edit `.py` files, scoring logic, rulesets, or manifest
- Do not `git push`, `git commit`, or modify tracked files
- Do not recommend trades, exits, or position changes
- Do not write outside `agents/shadow_monitor/memory/`
