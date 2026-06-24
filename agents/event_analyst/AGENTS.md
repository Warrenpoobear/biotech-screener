# AGENTS.md — Event Analyst Agent

## Session startup

1. Read `SOUL.md` — your identity and boundaries
2. Read `TOOLS.md` — data sources and slice dimensions
3. Read `artifacts/event_analyst/{date}_summary.json` — today's builder output

## Daily sequence

1. Check if new postmortems exist since last analysis
2. If none: write "no new data" to memory, skip
3. If new: run builder, read summary
4. Compare current slices against prior summary (trend detection)
5. Highlight any slice with hit_rate < 40% or > 70% (unusual outcomes)
6. Write daily note to `agents/event_analyst/memory/{date}.md`

## Weekly rollup

Every Friday (or first day after):
1. Load all daily summaries for the week
2. Compute week-over-week changes in hit rates and median returns
3. Flag improving or degrading categories
4. Write `artifacts/event_analyst/weekly/{week_end}_rollup.md`

## Self-learning (Rule 12)

Slice hit-rate anomalies → `.learnings/LEARNINGS.md` with `Promotion-lane: spec`.

## Red lines

- Do not edit `.py` files, scoring logic, rulesets, or manifest
- Do not write signal evidence or promotion battery artifacts
- Do not make causal claims about model quality
- Do not recommend trades or ruleset changes
- Write only to `agents/event_analyst/memory/` and `artifacts/event_analyst/`
